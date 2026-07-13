"""Tests for backtester.py — grouping helpers, P&L math, and entry direction."""
from datetime import UTC, date, datetime, timedelta
from unittest.mock import MagicMock

import pytest

# historical.py imports kalshi_python_sync.api.historical_api softly (the
# module is absent in some SDK builds), so backtester.py is always importable
# and its pure-logic functions are unit-testable offline.
from kalshi_betting import backtester
from kalshi_betting.backtester import (
    _extract_pairs,
    _find_entry,
    _group_by_exact_title,
    _group_by_normalized_title,
    _pair_key,
    _settlement_receipt,
    run_backtest,
)


def _md(ticker, event_ticker, title="", subtitle="", event_title=""):
    """Build a minimal market dict matching what _market_to_dict produces."""
    return {
        "ticker": ticker,
        "event_ticker": event_ticker,
        "event_title": event_title,
        "title": title,
        "subtitle": subtitle,
    }


class TestPairKey:
    def test_combines_event_and_title(self):
        m = _md("T1", "E1", title="Trump", event_title="2024 Election Winner")
        key = _pair_key(m)
        assert "2024 Election Winner" in key
        assert "Trump" in key
        assert "|" in key

    def test_falls_back_when_event_title_missing(self):
        m = _md("T1", "E1", title="Will BTC exceed $80k", event_title="")
        assert _pair_key(m) == "Will BTC exceed $80k"


class TestExactTitleGrouping:
    def test_cross_event_same_option_label_separated(self):
        # Same option title "Trump", different event_title — must NOT group together
        mA = _md("A1", "ELECT", title="Trump", event_title="2024 Election Winner")
        mB = _md("B1", "TIME",  title="Trump", event_title="2024 Time Person of the Year")
        groups = _group_by_exact_title([mA, mB])
        # Each lands in its own group, but only groups with >= 2 members survive,
        # so the result should be empty.
        assert groups == {}

    def test_same_event_title_groups_together(self):
        mA = _md("A1", "EVT-A", title="Republicans win majority",
                 event_title="2026 Senate Control")
        mB = _md("B1", "EVT-B", title="Republicans win majority",
                 event_title="2026 Senate Control")
        groups = _group_by_exact_title([mA, mB])
        assert len(groups) == 1
        key = next(iter(groups))
        assert key[0] == "2026 Senate Control"
        assert key[1] == "Republicans win majority"


class TestNormalizedTitleGrouping:
    def test_same_event_different_deadlines_groups(self):
        # Same event_title and same dateless market title — these should group as
        # time-series candidates (different deadlines stripped from title).
        mA = _md("A1", "EVT-MAR", title="BTC over $80k by March 2026",
                 event_title="BTC price tracker")
        mB = _md("B1", "EVT-JUN", title="BTC over $80k by June 2026",
                 event_title="BTC price tracker")
        groups = _group_by_normalized_title([mA, mB])
        assert len(groups) == 1
        # Both members should be in the group
        assert len(next(iter(groups.values()))) == 2

    def test_cross_event_same_label_separated(self):
        # Same market title but different event titles — must NOT group
        mA = _md("A1", "ELECT", title="Trump", event_title="2024 Election Winner")
        mB = _md("B1", "TIME",  title="Trump", event_title="2024 Time Person of the Year")
        groups = _group_by_normalized_title([mA, mB])
        # Each lands alone, filtered out by len>=2
        assert groups == {}


class TestExtractPairsCanonHandling:
    def test_three_tuple_key_uses_title_not_event(self):
        # Build a same-title group with a 3-tuple key and verify canon is the
        # market title (key[1]), not the event title (key[0]).
        mA = _md("A1", "EVT-A", title="Republicans win majority",
                 event_title="2026 Senate Control")
        mB = _md("B1", "EVT-B", title="Republicans win majority",
                 event_title="2026 Senate Control")
        groups = _group_by_exact_title([mA, mB])
        pairs = _extract_pairs(groups, "same_title")
        assert len(pairs) == 1
        _, _, canon, _ = pairs[0]
        assert canon == "Republicans win majority"

    def test_group_key_includes_event_title_for_dedup(self):
        # Two unrelated events sharing an option label ("Trump") must produce
        # DIFFERENT group_keys even though they share the same display canon —
        # otherwise run_backtest's one-pair-per-group dedup collapses two
        # legitimate, independent pairs into one.
        mA1 = _md("E1-A", "EVT1A", title="Trump", event_title="Election Winner")
        mA2 = _md("E1-B", "EVT1B", title="Trump", event_title="Election Winner")
        mB1 = _md("E2-A", "EVT2A", title="Trump", event_title="Person of the Year")
        mB2 = _md("E2-B", "EVT2B", title="Trump", event_title="Person of the Year")
        groups = _group_by_exact_title([mA1, mA2, mB1, mB2])
        pairs = _extract_pairs(groups, "same_title")
        assert len(pairs) == 2
        canons = {canon for _, _, canon, _ in pairs}
        assert canons == {"Trump"}
        group_keys = {group_key for _, _, _, group_key in pairs}
        assert len(group_keys) == 2, "distinct events must yield distinct group_keys"


class TestSettlementReceipt:
    def test_payoff_table(self):
        # n NO contracts on A + n YES contracts on B; each winning contract
        # pays exactly $1 — the receipt is independent of entry prices.
        n = 10
        assert _settlement_receipt(n, "yes", "yes") == 10   # only B pays
        assert _settlement_receipt(n, "no", "yes") == 20    # both pay
        assert _settlement_receipt(n, "no", "no") == 10     # only A pays
        assert _settlement_receipt(n, "yes", "no") == 0     # loss scenario


def _candle(ts: int, yes_ask: float, no_ask: float) -> dict:
    return {"ts": ts, "yes_ask_close": yes_ask, "no_ask_close": no_ask}


# Monday 2026-01-05 09:00 UTC — first Monday on/after 2026-01-01
_MONDAY_TS = int(datetime(2026, 1, 5, 9, 0, tzinfo=UTC).timestamp())


class TestFindEntryDirection:
    def _markets(self):
        # 13-day deadline gap — inside the short (15%) tier, so these tests
        # exercise direction rules, not the deadline-gap threshold tiers
        # (covered separately in TestFindEntryTieredThreshold).
        mA = {"ticker": "EARLY", "event_ticker": "E1",
              "close_time": "2026-02-01T00:00:00+00:00"}
        mB = {"ticker": "LATE", "event_ticker": "E2",
              "close_time": "2026-02-14T00:00:00+00:00"}
        return mA, mB

    def test_time_series_rejects_pricier_later_contract(self):
        # Earlier cheap (0.30), later pricey (0.60): normal term structure —
        # the live scanner never trades this direction (requires the EARLIER
        # contract to be the expensive one), so the backtest must not either.
        mA, mB = self._markets()
        candles_early = [_candle(_MONDAY_TS, 0.30, 0.70)]
        candles_late  = [_candle(_MONDAY_TS, 0.60, 0.40)]
        entry = _find_entry(candles_early, candles_late, mA, mB,
                            "time_series", date(2026, 1, 1))
        assert entry is None

    def test_time_series_accepts_pricier_earlier_contract(self):
        # Earlier pricey (pA=0.60, nA=0.40), later cheap (pB=0.30). At the
        # 13-day deadline gap the short tier applies: price gap 0.30 >= 0.15,
        # nA+pB = 0.70 <= 0.85, spread clears fees.
        mA, mB = self._markets()
        candles_early = [_candle(_MONDAY_TS, 0.60, 0.40)]
        candles_late  = [_candle(_MONDAY_TS, 0.30, 0.70)]
        entry = _find_entry(candles_early, candles_late, mA, mB,
                            "time_series", date(2026, 1, 1))
        assert entry is not None
        # Market A must be the earlier-closing contract (the NO leg)
        assert entry["mA"]["ticker"] == "EARLY"
        assert entry["pA"] == pytest.approx(0.60)
        assert entry["pB"] == pytest.approx(0.30)

    def test_same_title_canonicalizes_by_price(self):
        # For same-title pairs, direction is price-only: A = expensive side.
        mA, mB = self._markets()
        candles_a = [_candle(_MONDAY_TS, 0.55, 0.47)]
        candles_b = [_candle(_MONDAY_TS, 0.70, 0.32)]
        entry = _find_entry(candles_a, candles_b, mA, mB,
                            "same_title", date(2026, 1, 1))
        assert entry is not None
        assert entry["mA"]["ticker"] == "LATE"  # the pricier side becomes A


class TestFindEntryTieredThreshold:
    """_find_entry mirrors the scanner's deadline-gap-tiered price threshold:
    15% for deadline gaps <= 15 days, 30% for 16-30 days, nothing beyond 30."""

    def _markets(self, gap_days: int):
        close_a = datetime(2026, 2, 1, tzinfo=UTC)
        close_b = close_a + timedelta(days=gap_days)
        mA = {"ticker": "EARLY", "event_ticker": "E1",
              "close_time": close_a.isoformat()}
        mB = {"ticker": "LATE", "event_ticker": "E2",
              "close_time": close_b.isoformat()}
        return mA, mB

    def _entry(self, gap_days: int, pA: float, pB: float):
        mA, mB = self._markets(gap_days)
        candles_early = [_candle(_MONDAY_TS, pA, round(1.0 - pA, 4))]
        candles_late  = [_candle(_MONDAY_TS, pB, round(1.0 - pB, 4))]
        return _find_entry(candles_early, candles_late, mA, mB,
                           "time_series", date(2026, 1, 1))

    def test_short_gap_18pct_price_gap_accepted(self):
        # 10-day deadline gap → 15% tier; pA=0.48/pB=0.30 (18% gap, nA+pB=0.82
        # <= 0.85) qualifies
        entry = self._entry(10, pA=0.48, pB=0.30)
        assert entry is not None
        assert entry["mA"]["ticker"] == "EARLY"

    def test_long_gap_18pct_price_gap_rejected(self):
        # The SAME prices at a 20-day deadline gap fall under the 30% tier
        # and must be rejected — this is exactly what the old flat 15%
        # threshold would have (wrongly) accepted
        assert self._entry(20, pA=0.48, pB=0.30) is None

    def test_long_gap_38pct_price_gap_accepted(self):
        # 20-day gap → 30% tier; pA=0.68/pB=0.30 (38% gap, nA+pB=0.62 <= 0.70)
        # clears both the tiered threshold and the tiered price-sum ceiling
        entry = self._entry(20, pA=0.68, pB=0.30)
        assert entry is not None

    def test_over_max_gap_rejected_regardless_of_price(self):
        # 35-day deadline gap exceeds MAX_DEADLINE_GAP_DAYS — even a 40% price
        # gap never enters
        assert self._entry(35, pA=0.70, pB=0.30) is None

    def test_same_title_ignores_deadline_gap(self):
        # same_title keeps the flat 5% threshold: a 6% divergence on markets
        # closing 20 days apart is still an entry (nA+pB = 0.94 <= 0.95)
        mA, mB = self._markets(20)
        candles_a = [_candle(_MONDAY_TS, 0.36, 0.64)]
        candles_b = [_candle(_MONDAY_TS, 0.30, 0.70)]
        entry = _find_entry(candles_a, candles_b, mA, mB,
                            "same_title", date(2026, 1, 1))
        assert entry is not None


class TestRunBacktestPnL:
    def test_profit_not_double_counted_and_cash_sized(self, monkeypatch):
        """End-to-end regression for the P&L double-subtraction and sizing bugs.

        One same-title pair, both markets resolve YES (co-resolution): the
        realized profit must equal the guaranteed floor (receipt n, cost+fees
        out), the equity curve must end at initial_balance + profit, and the
        trade must be sized against the running balance.
        """
        markets = [
            {"ticker": "SA", "event_ticker": "EA", "event_title": "EV",
             "title": "Q", "subtitle": "", "result": "yes",
             "close_time": "2026-02-01T00:00:00+00:00",
             "settlement_ts": "2026-02-01T12:00:00+00:00"},
            {"ticker": "SB", "event_ticker": "EB", "event_title": "EV",
             "title": "Q", "subtitle": "", "result": "yes",
             "close_time": "2026-02-01T00:00:00+00:00",
             "settlement_ts": "2026-02-01T12:00:00+00:00"},
        ]
        candles = {
            # SA is the expensive side: pA=0.70, nA=0.32
            "SA": [_candle(_MONDAY_TS, 0.70, 0.32)],
            # SB is the cheap side: pB=0.55
            "SB": [_candle(_MONDAY_TS, 0.55, 0.47)],
        }
        monkeypatch.setattr(backtester, "fetch_all_settled_markets",
                            lambda *a, **k: markets)
        monkeypatch.setattr(backtester, "fetch_daily_candlesticks",
                            lambda _c, ticker, *a, **k: candles[ticker])

        trades, equity = run_backtest(
            hist_client=MagicMock(), live_client=MagicMock(),
            start_date=date(2026, 1, 1), initial_balance=1000.0,
        )

        assert len(trades) == 1
        t = trades[0]
        # Sized against the balance at entry, not hardcoded
        assert t.balance_at_entry == pytest.approx(1000.0)
        assert t.total_cost + t.fees <= 1000.0
        # Both YES → only the YES leg pays: receipt is exactly n dollars
        assert t.actual_payoff == pytest.approx(float(t.n))
        # Profit deducts cost and fees exactly once
        assert t.profit == pytest.approx(t.actual_payoff - t.total_cost - t.fees)
        # Both-YES is the guaranteed-floor scenario: profit == expected_payoff
        assert t.profit == pytest.approx(t.expected_payoff)
        assert t.profit > 0
        assert t.slippage == pytest.approx(0.0, abs=1e-9)
        # Equity curve: ends at initial balance + realized profit (no double count)
        final_value = float(equity["portfolio_value"].iloc[-1])
        assert final_value == pytest.approx(1000.0 + t.profit)
