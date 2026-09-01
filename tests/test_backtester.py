"""Tests for backtester.py — grouping helpers, P&L math, and entry direction."""
import time
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# historical.py imports kalshi_python_sync.api.historical_api softly (the
# module is absent in some SDK builds), so backtester.py is always importable
# and its pure-logic functions are unit-testable offline.
from kalshi_betting import backtester
from kalshi_betting.backtester import (
    _can_ever_enter,
    _extract_pairs,
    _fetch_candles_parallel,
    _find_entry,
    _group_by_exact_title,
    _group_by_normalized_title,
    _pair_key,
    _parse_iso_date,
    _parse_iso_datetime,
    _settlement_receipt,
    run_backtest,
)
from kalshi_betting.config import MAX_DEADLINE_GAP_DAYS, fee_leg_exact
from kalshi_betting.scanner import CandidatePair
from kalshi_betting.strategy import compute_trade


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


class TestFindEntryHorizon:
    """_find_entry's optional max_horizon_days caps how far the later-closing
    leg's close date may be from the checkpoint being evaluated — relative to
    the SIMULATED Monday, not real-world now."""

    def _markets(self):
        # Same 13-day-gap fixture as TestFindEntryDirection — close_a
        # 2026-02-01, close_b 2026-02-14.
        mA = {"ticker": "EARLY", "event_ticker": "E1",
              "close_time": "2026-02-01T00:00:00+00:00"}
        mB = {"ticker": "LATE", "event_ticker": "E2",
              "close_time": "2026-02-14T00:00:00+00:00"}
        return mA, mB

    def test_none_horizon_matches_unfiltered_behavior(self):
        mA, mB = self._markets()
        candles_early = [_candle(_MONDAY_TS, 0.60, 0.40)]
        candles_late  = [_candle(_MONDAY_TS, 0.30, 0.70)]
        entry = _find_entry(candles_early, candles_late, mA, mB,
                            "time_series", date(2026, 1, 1), max_horizon_days=None)
        assert entry is not None
        assert entry["entry_date"] == date(2026, 1, 5)

    def test_within_horizon_at_first_checkpoint_enters_immediately(self):
        # At the first Monday (Jan 5), close_b (Feb 14) is 40 days out — well
        # within a generous 60-day horizon, so entry is unaffected.
        mA, mB = self._markets()
        candles_early = [_candle(_MONDAY_TS, 0.60, 0.40)]
        candles_late  = [_candle(_MONDAY_TS, 0.30, 0.70)]
        entry = _find_entry(candles_early, candles_late, mA, mB,
                            "time_series", date(2026, 1, 1), max_horizon_days=60)
        assert entry is not None
        assert entry["entry_date"] == date(2026, 1, 5)

    def test_beyond_horizon_at_early_checkpoints_defers_to_a_later_monday(self):
        # close_b (Feb 14) is 40 days from Jan 5, 33 from Jan 12, 26 from
        # Jan 19 — all beyond a 20-day horizon — but only 19 days from Jan 26,
        # so entry must land on Jan 26 rather than the earliest price-qualifying
        # Monday. A single candle at _MONDAY_TS is found by _candle_at_or_before
        # for every later checkpoint too, so price qualifies throughout.
        mA, mB = self._markets()
        candles_early = [_candle(_MONDAY_TS, 0.60, 0.40)]
        candles_late  = [_candle(_MONDAY_TS, 0.30, 0.70)]
        entry = _find_entry(candles_early, candles_late, mA, mB,
                            "time_series", date(2026, 1, 1), max_horizon_days=20)
        assert entry is not None
        assert entry["entry_date"] == date(2026, 1, 26)

    def test_no_checkpoint_in_scan_window_satisfies_horizon_returns_none(self):
        # The scan window ends at close_a - 1 day (Jan 31); the closest that
        # gets to close_b (Feb 14) within the window is Jan 26 at 19 days out.
        # A 3-day horizon excludes every checkpoint in range, even though the
        # price gap would otherwise qualify at every one of them.
        mA, mB = self._markets()
        candles_early = [_candle(_MONDAY_TS, 0.60, 0.40)]
        candles_late  = [_candle(_MONDAY_TS, 0.30, 0.70)]
        entry = _find_entry(candles_early, candles_late, mA, mB,
                            "time_series", date(2026, 1, 1), max_horizon_days=3)
        assert entry is None


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
        monkeypatch.setattr(backtester, "fetch_candlesticks",
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
        # The fee-inclusive outlay must fit the Kelly budget it was sized
        # against — fees ride on top of the contract cost, so this only holds
        # because Pass 2 shrinks n (see TestKellyShrinkParity)
        assert t.total_cost + t.fees <= 1000.0 * t.kelly_fraction + 1e-9
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

    def test_max_horizon_days_threads_through_and_excludes_trade(self, monkeypatch):
        # Same fixture as test_profit_not_double_counted_and_cash_sized: both
        # legs close 2026-02-01, only candle is at _MONDAY_TS (2026-01-05),
        # 27 days before close. A 3-day horizon excludes every checkpoint in
        # the scan window (Jan 5 - Jan 31), so run_backtest must report zero
        # trades even though the same fixture produces exactly one trade with
        # max_horizon_days=None.
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
            "SA": [_candle(_MONDAY_TS, 0.70, 0.32)],
            "SB": [_candle(_MONDAY_TS, 0.55, 0.47)],
        }
        monkeypatch.setattr(backtester, "fetch_all_settled_markets",
                            lambda *a, **k: markets)
        monkeypatch.setattr(backtester, "fetch_candlesticks",
                            lambda _c, ticker, *a, **k: candles[ticker])

        trades, _ = run_backtest(
            hist_client=MagicMock(), live_client=MagicMock(),
            start_date=date(2026, 1, 1), initial_balance=1000.0,
            max_horizon_days=3,
        )

        assert trades == []


# Same-title fixture shared by the sizing-parity and malformed-timestamp tests:
# SA is the expensive side (pA=0.70, nA=0.32), SB the cheap one (pB=0.55), both
# resolving YES on 2026-02-01. These are the numbers TestRunBacktestPnL uses, so
# the parity assertions below are pinned to the same trade that test checks.
_PARITY_PA, _PARITY_NA, _PARITY_PB = 0.70, 0.32, 0.55


def _same_title_markets(close_time_b: str = "2026-02-01T00:00:00+00:00") -> list[dict]:
    """Two markets in one exact-title group; close_time_b is overridable so a
    malformed timestamp can be injected on exactly one leg."""
    return [
        {"ticker": "SA", "event_ticker": "EA", "event_title": "EV",
         "title": "Q", "subtitle": "", "result": "yes",
         "close_time": "2026-02-01T00:00:00+00:00",
         "settlement_ts": "2026-02-01T12:00:00+00:00"},
        {"ticker": "SB", "event_ticker": "EB", "event_title": "EV",
         "title": "Q", "subtitle": "", "result": "yes",
         "close_time": close_time_b,
         "settlement_ts": "2026-02-01T12:00:00+00:00"},
    ]


class TestKellyShrinkParity:
    """Pass 2 must size a trade exactly like live strategy.compute_trade (BS-06).

    The Kelly budget covers the CONTRACTS only — the exact ceiling-rounded taker
    fees are charged on top — so the raw n = int(budget / (nA + pB)) systematically
    overshoots the cap. compute_trade shrinks n until the fee-inclusive cost fits
    (CLAUDE.md forbids removing that loop); the backtest skipped the shrink
    entirely and only rejected trades that didn't fit the whole CASH balance, so
    every simulated trade was sized slightly above the Kelly fraction it reported.
    """

    @staticmethod
    def _live_n(balance_cents: int) -> int:
        """The contract count live compute_trade picks for the same inputs."""
        def _market(ticker):
            # compute_trade only reads .close_time (for days_to_close, which
            # doesn't affect n) and .ticker
            return SimpleNamespace(
                ticker=ticker, close_time=datetime.now(UTC) + timedelta(days=27),
            )

        pair = CandidatePair(
            market_a=_market("SA"), market_b=_market("SB"),
            pA=_PARITY_PA, pB=_PARITY_PB, nA=_PARITY_NA,
            tradeable=True, canonical_title="Q", pair_type="same_title",
            # 0 = not depth-capped, which is the only state the backtest can
            # model (a backtest has candle closes, no orderbook)
            max_contracts=0,
        )
        spec = compute_trade(pair, balance_cents)
        assert spec is not None
        return spec.x

    def test_backtest_n_matches_compute_trade_n(self, monkeypatch):
        candles = {
            "SA": [_candle(_MONDAY_TS, _PARITY_PA, _PARITY_NA)],
            "SB": [_candle(_MONDAY_TS, _PARITY_PB, 0.47)],
        }
        monkeypatch.setattr(backtester, "fetch_all_settled_markets",
                            lambda *a, **k: _same_title_markets())
        monkeypatch.setattr(backtester, "fetch_candlesticks",
                            lambda _c, ticker, *a, **k: candles[ticker])

        trades, _ = run_backtest(
            hist_client=MagicMock(), live_client=MagicMock(),
            start_date=date(2026, 1, 1), initial_balance=1000.0,
        )
        assert len(trades) == 1
        t = trades[0]

        # The load-bearing assertion: identical (nA, pB, kelly, balance) inputs
        # must produce an identical contract count on both code paths.
        assert t.n == self._live_n(100_000)

        # Guard against a vacuous pass: the shrink loop must actually have bitten
        # here, i.e. the un-shrunk n did NOT fit the Kelly budget once fees were
        # added. (Pre-fix the backtest recorded exactly that raw n.)
        budget = 1000.0 * t.kelly_fraction
        raw_n = int(budget / (_PARITY_NA + _PARITY_PB))
        assert raw_n > t.n
        raw_cost = raw_n * (_PARITY_NA + _PARITY_PB)
        raw_fees = fee_leg_exact(raw_n, _PARITY_NA) + fee_leg_exact(raw_n, _PARITY_PB)
        assert raw_cost + raw_fees > budget
        # ...and the chosen n does fit, which is the invariant being restored
        assert t.total_cost + t.fees <= budget + 1e-9

    def test_shrunk_size_still_respects_the_budget_fraction(self, monkeypatch):
        # Second balance point, so the parity isn't an artifact of one number.
        candles = {
            "SA": [_candle(_MONDAY_TS, _PARITY_PA, _PARITY_NA)],
            "SB": [_candle(_MONDAY_TS, _PARITY_PB, 0.47)],
        }
        monkeypatch.setattr(backtester, "fetch_all_settled_markets",
                            lambda *a, **k: _same_title_markets())
        monkeypatch.setattr(backtester, "fetch_candlesticks",
                            lambda _c, ticker, *a, **k: candles[ticker])

        trades, _ = run_backtest(
            hist_client=MagicMock(), live_client=MagicMock(),
            start_date=date(2026, 1, 1), initial_balance=250.0,
        )
        assert len(trades) == 1
        t = trades[0]
        assert t.n == self._live_n(25_000)
        assert t.total_cost + t.fees <= 250.0 * t.kelly_fraction + 1e-9


class TestRunBacktestMalformedTimestamps:
    """A single market with an unparseable close_time must not kill a run.

    The time-series extraction path already drops such members while windowing
    by close date (TestExtractPairsWindowedEquivalence), but the same-title path
    is naive and carries them straight into the candlestick fetch, where the
    pre-pool window computation used a bare datetime.fromisoformat — on the main
    thread, before any worker starts, so it aborted the whole backtest (BS-07).
    """

    def test_malformed_close_time_is_skipped_not_raised(self, monkeypatch, caplog):
        markets = _same_title_markets(close_time_b="not-a-timestamp")
        # Only SA has a fetchable window; a KeyError here would mean SB reached
        # the fetch despite having no parseable close_time.
        candles = {"SA": [_candle(_MONDAY_TS, _PARITY_PA, _PARITY_NA)]}
        monkeypatch.setattr(backtester, "fetch_all_settled_markets",
                            lambda *a, **k: markets)
        monkeypatch.setattr(backtester, "fetch_candlesticks",
                            lambda _c, ticker, *a, **k: candles[ticker])

        with caplog.at_level("WARNING"):
            trades, equity = run_backtest(
                hist_client=MagicMock(), live_client=MagicMock(),
                start_date=date(2026, 1, 1), initial_balance=1000.0,
            )

        # No trade: _find_entry can't derive a scan window for the bad leg
        assert trades == []
        assert float(equity["portfolio_value"].iloc[-1]) == pytest.approx(1000.0)
        # The dropped ticker is named, so an empty series can't be mistaken for
        # "this market genuinely had no prices"
        warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert any("SB" in msg and "close_time" in msg for msg in warnings)


class TestActiveTickerRelease:
    """A ticker is blocked only while its position is OPEN (BS-24).

    Live, scanner.get_held_tickers() reads positions with count_filter="position",
    so a ticker leaves the blocked set the moment its market settles. The
    backtest used to add tickers to active_tickers and never remove them, so one
    early trade blocked that ticker for the entire remaining simulation.

    Fixture: TX/TY share an exact title (same-title pair, entering at the first
    Monday); TX/TZ share a normalized title with an 18-day deadline gap
    (time-series pair, which can only enter at the second Monday because TZ has
    no earlier candle). Both candidates therefore contain TX, at different entry
    dates. TX settles before its close_time — an early determination, which is
    what makes re-entry on a shared ticker reachable at all.
    """

    _M2 = int(datetime(2026, 1, 12, 9, 0, tzinfo=UTC).timestamp())

    @staticmethod
    def _markets(tx_settlement: str) -> list[dict]:
        return [
            {"ticker": "TX", "event_ticker": "EX", "event_title": "EV",
             "title": "Team wins by March 2026", "subtitle": "", "result": "yes",
             "open_time": "2026-01-01T00:00:00+00:00",
             "close_time": "2026-02-02T00:00:00+00:00",
             "settlement_ts": tx_settlement},
            {"ticker": "TY", "event_ticker": "EY", "event_title": "EV",
             "title": "Team wins by March 2026", "subtitle": "", "result": "yes",
             "open_time": "2026-01-01T00:00:00+00:00",
             "close_time": "2026-01-20T00:00:00+00:00",
             "settlement_ts": "2026-01-08T12:00:00+00:00"},
            {"ticker": "TZ", "event_ticker": "EZ", "event_title": "EV",
             "title": "Team wins by April 2026", "subtitle": "", "result": "yes",
             "open_time": "2026-01-01T00:00:00+00:00",
             "close_time": "2026-02-20T00:00:00+00:00",
             "settlement_ts": "2026-02-20T12:00:00+00:00"},
        ]

    def _run(self, monkeypatch, tx_settlement):
        candles = {
            # TX: expensive YES with a cheap NO — clears the same-title gap
            # against TY and the 30% long-gap tier against TZ
            "TX": [_candle(_MONDAY_TS, 0.90, 0.05)],
            "TY": [_candle(_MONDAY_TS, 0.70, 0.30)],
            # TZ's first candle is the SECOND Monday, so the TX/TZ pair cannot
            # enter until then
            "TZ": [_candle(self._M2, 0.10, 0.90)],
        }
        monkeypatch.setattr(backtester, "fetch_all_settled_markets",
                            lambda *a, **k: self._markets(tx_settlement))
        monkeypatch.setattr(backtester, "fetch_candlesticks",
                            lambda _c, ticker, *a, **k: candles[ticker])
        trades, _ = run_backtest(
            hist_client=MagicMock(), live_client=MagicMock(),
            start_date=date(2026, 1, 1), initial_balance=1000.0,
        )
        return trades

    def test_ticker_is_reusable_once_its_trade_has_settled(self, monkeypatch):
        # TX settles 2026-01-08, four days before the second pair's entry
        trades = self._run(monkeypatch, "2026-01-08T12:00:00+00:00")
        assert len(trades) == 2
        first, second = trades
        assert (first.pair_type, first.ticker_a, first.ticker_b) == ("same_title", "TX", "TY")
        assert first.entry_date == date(2026, 1, 5)
        assert first.exit_date == date(2026, 1, 8)
        # Same ticker, entered again after the first position closed
        assert (second.pair_type, second.ticker_a, second.ticker_b) == ("time_series", "TX", "TZ")
        assert second.entry_date == date(2026, 1, 12)
        assert second.entry_date > first.exit_date

    def test_ticker_stays_blocked_while_its_trade_is_open(self, monkeypatch):
        # Only change: TX settles 2026-02-02, i.e. AFTER the second pair's
        # entry date — the positions would overlap, so the second is skipped
        trades = self._run(monkeypatch, "2026-02-02T12:00:00+00:00")
        assert len(trades) == 1
        assert trades[0].pair_type == "same_title"
        assert trades[0].exit_date == date(2026, 2, 2)


# Anchor Monday reused across the eligibility-prefilter tests below — matches
# _MONDAY_TS's 2026-01-05 anchor so date arithmetic stays consistent with the
# rest of this file. All offsets are computed with timedelta so weekday
# arithmetic can't be hand-miscounted.
_KNOWN_MONDAY = date(2026, 1, 5)


def _mkt(open_d: date | None, close_d: date | None) -> dict:
    """Minimal market dict for _can_ever_enter — only open_time/close_time matter."""
    m = {}
    if open_d is not None:
        m["open_time"] = datetime(open_d.year, open_d.month, open_d.day,
                                   tzinfo=UTC).isoformat()
    if close_d is not None:
        m["close_time"] = datetime(close_d.year, close_d.month, close_d.day,
                                    tzinfo=UTC).isoformat()
    return m


class TestParseIsoDate:
    def test_parses_valid_iso_string(self):
        assert _parse_iso_date("2026-01-05T09:00:00+00:00") == date(2026, 1, 5)

    def test_none_returns_none(self):
        assert _parse_iso_date(None) is None

    def test_empty_string_returns_none(self):
        assert _parse_iso_date("") is None

    def test_unparseable_string_returns_none(self):
        assert _parse_iso_date("not-a-date") is None


class TestParseIsoDatetime:
    """Sibling of _parse_iso_date that keeps the time-of-day the candlestick
    fetch window needs — truncating to midnight would move the window."""

    def test_parses_valid_iso_string_with_time(self):
        assert _parse_iso_datetime("2026-01-05T09:30:00+00:00") == datetime(
            2026, 1, 5, 9, 30, tzinfo=UTC
        )

    def test_keeps_time_of_day_that_the_date_helper_drops(self):
        value = "2026-01-05T23:45:00+00:00"
        assert _parse_iso_datetime(value).hour == 23
        assert _parse_iso_date(value) == date(2026, 1, 5)

    def test_none_returns_none(self):
        assert _parse_iso_datetime(None) is None

    def test_empty_string_returns_none(self):
        assert _parse_iso_datetime("") is None

    def test_unparseable_string_returns_none(self):
        assert _parse_iso_datetime("not-a-timestamp") is None


class TestCanEverEnter:
    """_can_ever_enter mirrors _find_entry's exact scan-window construction:
    lower = max(open_time, start_date), upper = close_time - 1 day, and the
    market is eligible iff a Monday falls in [lower, upper]."""

    def test_two_hour_market_never_spans_a_monday(self):
        # open == close date (a 2-hour market truncates to the same calendar
        # day) — upper = close_date - 1 day is always before lower, so this
        # short-circuits to False regardless of which weekday it falls on.
        d = _KNOWN_MONDAY + timedelta(days=3)
        m = _mkt(d, d)
        assert _can_ever_enter(m, start_date=_KNOWN_MONDAY) is False

    def test_market_spanning_a_monday_comfortably_is_eligible(self):
        # Opens the Thursday before, closes three weeks after the Monday —
        # multiple Mondays fall well inside [open, close - 1 day].
        open_d  = _KNOWN_MONDAY - timedelta(days=4)
        close_d = _KNOWN_MONDAY + timedelta(days=15)
        m = _mkt(open_d, close_d)
        assert _can_ever_enter(m, start_date=open_d) is True

    def test_tuesday_open_closing_following_wednesday_spans_with_margin(self):
        # Opens the Tuesday after _KNOWN_MONDAY, closes 8 days later (the
        # following Wednesday). upper = close - 1 day lands exactly on the
        # next Monday, so it just barely spans it.
        open_d  = _KNOWN_MONDAY + timedelta(days=1)   # Tuesday
        close_d = open_d + timedelta(days=8)           # following Wednesday
        m = _mkt(open_d, close_d)
        assert _can_ever_enter(m, start_date=_KNOWN_MONDAY) is True

    def test_closes_same_week_as_the_spanned_monday_is_ineligible(self):
        # Opens the Sunday before a Monday, but CLOSES on that same Monday —
        # upper = close_date - 1 day is the Sunday, strictly before the
        # Monday the window nominally "spans". No candle-eligible checkpoint
        # exists (Monday > close - 1 day), so this must be ineligible.
        open_d  = _KNOWN_MONDAY - timedelta(days=1)  # Sunday
        close_d = _KNOWN_MONDAY                       # closes ON the Monday
        m = _mkt(open_d, close_d)
        assert _can_ever_enter(m, start_date=open_d) is False

    def test_missing_open_time_keeps_the_market(self):
        m = _mkt(None, _KNOWN_MONDAY + timedelta(days=10))
        assert _can_ever_enter(m, start_date=_KNOWN_MONDAY) is True

    def test_missing_close_time_keeps_the_market(self):
        m = _mkt(_KNOWN_MONDAY - timedelta(days=10), None)
        assert _can_ever_enter(m, start_date=_KNOWN_MONDAY) is True

    def test_both_missing_keeps_the_market(self):
        assert _can_ever_enter({}, start_date=_KNOWN_MONDAY) is True

    def test_monday_before_start_date_is_not_counted(self):
        # The market's own window contains a Monday, but that Monday falls
        # strictly before start_date — _find_entry never scans before
        # start_date, so this must be ineligible.
        prev_monday = _KNOWN_MONDAY - timedelta(days=7)
        open_d  = prev_monday - timedelta(days=3)
        close_d = prev_monday + timedelta(days=2)
        m = _mkt(open_d, close_d)
        assert _can_ever_enter(m, start_date=_KNOWN_MONDAY) is False

    def test_same_window_eligible_once_start_date_moves_earlier(self):
        # Identical market window to the previous test, but start_date no
        # longer clips out the Monday the window actually contains.
        prev_monday = _KNOWN_MONDAY - timedelta(days=7)
        open_d  = prev_monday - timedelta(days=3)
        close_d = prev_monday + timedelta(days=2)
        m = _mkt(open_d, close_d)
        earlier_start = prev_monday - timedelta(days=10)
        assert _can_ever_enter(m, start_date=earlier_start) is True


def _ts_member(ticker: str, event_ticker: str, close_d: date | None) -> dict:
    """Minimal time-series group member for _extract_pairs windowing tests."""
    m = {"ticker": ticker, "event_ticker": event_ticker}
    if close_d is not None:
        m["close_time"] = datetime(close_d.year, close_d.month, close_d.day,
                                    tzinfo=UTC).isoformat()
    return m


def _naive_time_series_pairs(members: list[dict], margin_days: int) -> set[frozenset]:
    """Independent oracle: naive O(n^2) double loop over the same group,
    filtering by the same margin-inclusive close-time gap and event_ticker
    rule _extract_pairs applies, but without any sorting/windowing. Written
    standalone (no backtester internals besides plain dict/date arithmetic)
    so it can serve as ground truth for the windowed implementation.
    """
    result: set[frozenset] = set()
    n = len(members)
    for i in range(n):
        a = members[i]
        close_a_raw = a.get("close_time")
        if not close_a_raw:
            continue
        da = datetime.fromisoformat(close_a_raw).date()
        for j in range(i + 1, n):
            b = members[j]
            close_b_raw = b.get("close_time")
            if not close_b_raw:
                continue
            db = datetime.fromisoformat(close_b_raw).date()
            if abs((db - da).days) > margin_days:
                continue
            if a["event_ticker"] == b["event_ticker"]:
                continue
            result.add(frozenset([a["ticker"], b["ticker"]]))
    return result


class TestExtractPairsWindowedEquivalence:
    """_extract_pairs' close-time-windowed sweep for time-series groups must
    select exactly the same candidate pairs as a naive double loop using the
    same MAX_DEADLINE_GAP_DAYS + 1 margin (the margin is deliberate slack —
    _find_entry applies the exact .days > MAX_DEADLINE_GAP_DAYS cutoff itself
    afterwards, so a few boundary candidates one day past the true limit are
    harmless to include here and are proven out separately in
    TestFindEntryTieredThreshold.test_over_max_gap_rejected_regardless_of_price).
    """

    def _build_synthetic_group(self) -> list[dict]:
        base = date(2026, 1, 1)
        members = []

        # Cluster A: five markets closely spaced, well inside the window.
        for i, offset in enumerate([0, 5, 10, 15, 20]):
            members.append(_ts_member(f"A{i}", f"EA{i}", base + timedelta(days=offset)))

        # Cluster B: another tight cluster, but its earliest member is 80
        # days after cluster A's latest — far outside the window, so no
        # cross-cluster pairs should ever appear.
        for i, offset in enumerate([100, 105, 110, 115, 120]):
            members.append(_ts_member(f"B{i}", f"EB{i}", base + timedelta(days=offset)))

        # Boundary probes around the exact margin edge (MAX_DEADLINE_GAP_DAYS
        # + 1 = 31 days): 30 (in), 31 (in, margin), 32 (out).
        ref = base + timedelta(days=200)
        members.append(_ts_member("REF", "EREF", ref))
        members.append(_ts_member("PLUS30", "EPLUS30", ref + timedelta(days=30)))
        members.append(_ts_member("PLUS31", "EPLUS31", ref + timedelta(days=31)))
        members.append(_ts_member("PLUS32", "EPLUS32", ref + timedelta(days=32)))

        # Same-event_ticker pair close in time — must be skipped regardless
        # of how close their close_time gap is.
        members.append(_ts_member("SAMEEVT-1", "SHARED-EVT", base + timedelta(days=300)))
        members.append(_ts_member("SAMEEVT-2", "SHARED-EVT", base + timedelta(days=301)))

        # Missing close_time — must be dropped from the sweep entirely rather
        # than crashing or pairing with anything.
        m = _ts_member("NOCLOSE", "ENOCLOSE", base + timedelta(days=300))
        del m["close_time"]
        members.append(m)

        return members

    def test_matches_naive_oracle_exactly(self):
        members = self._build_synthetic_group()
        groups = {"synthetic": members}

        windowed = _extract_pairs(groups, "time_series")
        windowed_set = {frozenset([a["ticker"], b["ticker"]]) for a, b, _, _ in windowed}

        margin_days = MAX_DEADLINE_GAP_DAYS + 1
        naive_set = _naive_time_series_pairs(members, margin_days)

        assert windowed_set == naive_set
        # Sanity: the set isn't trivially empty or trivially "everything"
        assert 0 < len(windowed_set) < (len(members) * (len(members) - 1)) // 2

    def test_boundary_probes_land_exactly_where_expected(self):
        members = self._build_synthetic_group()
        groups = {"synthetic": members}
        windowed = _extract_pairs(groups, "time_series")
        pair_tickers = {frozenset([a["ticker"], b["ticker"]]) for a, b, _, _ in windowed}

        assert frozenset(["REF", "PLUS30"]) in pair_tickers
        assert frozenset(["REF", "PLUS31"]) in pair_tickers   # inside the +1 margin
        assert frozenset(["REF", "PLUS32"]) not in pair_tickers  # outside the margin

    def test_cross_cluster_pairs_never_appear(self):
        members = self._build_synthetic_group()
        windowed = _extract_pairs({"synthetic": members}, "time_series")
        for a, b, _, _ in windowed:
            assert not (a["ticker"].startswith("A") and b["ticker"].startswith("B"))
            assert not (a["ticker"].startswith("B") and b["ticker"].startswith("A"))

    def test_same_event_ticker_pair_is_skipped(self):
        members = self._build_synthetic_group()
        windowed = _extract_pairs({"synthetic": members}, "time_series")
        pair_tickers = {frozenset([a["ticker"], b["ticker"]]) for a, b, _, _ in windowed}
        assert frozenset(["SAMEEVT-1", "SAMEEVT-2"]) not in pair_tickers

    def test_missing_close_time_member_produces_no_pairs(self):
        members = self._build_synthetic_group()
        windowed = _extract_pairs({"synthetic": members}, "time_series")
        for a, b, _, _ in windowed:
            assert a["ticker"] != "NOCLOSE"
            assert b["ticker"] != "NOCLOSE"


class TestRunBacktestEndToEndWithPrefilter:
    """The eligibility prefilter must not change results — only skip work.
    A hourly-ladder flood sharing the same exact-title group as a genuinely
    tradeable pair must be dropped before grouping/extraction/candlestick
    fetching, leaving the recorded trade identical to the pre-filter
    baseline in TestRunBacktestPnL.test_profit_not_double_counted_and_cash_sized.
    """

    def test_hourly_noise_is_filtered_without_changing_the_trade(self, monkeypatch):
        valid_markets = [
            {"ticker": "SA", "event_ticker": "EA", "event_title": "EV",
             "title": "Q", "subtitle": "", "result": "yes",
             "close_time": "2026-02-01T00:00:00+00:00",
             "settlement_ts": "2026-02-01T12:00:00+00:00"},
            {"ticker": "SB", "event_ticker": "EB", "event_title": "EV",
             "title": "Q", "subtitle": "", "result": "yes",
             "close_time": "2026-02-01T00:00:00+00:00",
             "settlement_ts": "2026-02-01T12:00:00+00:00"},
        ]
        # 200 intraday markets in the SAME exact-title group ("EV" | "Q") —
        # each has an explicit open_time/close_time 2 hours apart, so
        # _can_ever_enter drops every one of them before grouping.
        noise_markets = [
            {"ticker": f"NOISE-{i}", "event_ticker": f"NEV-{i}", "event_title": "EV",
             "title": "Q", "subtitle": "", "result": "yes",
             "open_time": "2026-01-10T00:00:00+00:00",
             "close_time": "2026-01-10T02:00:00+00:00",
             "settlement_ts": "2026-01-10T02:00:00+00:00"}
            for i in range(200)
        ]
        markets = valid_markets + noise_markets

        candles = {
            "SA": [_candle(_MONDAY_TS, 0.70, 0.32)],
            "SB": [_candle(_MONDAY_TS, 0.55, 0.47)],
        }
        monkeypatch.setattr(backtester, "fetch_all_settled_markets",
                            lambda *a, **k: markets)

        def _fetch_candles(_c, ticker, *a, **k):
            # A KeyError here means a noise market slipped past the
            # eligibility prefilter and got queried — that would be the
            # filter silently failing, not just a slow path.
            return candles[ticker]

        monkeypatch.setattr(backtester, "fetch_candlesticks", _fetch_candles)

        trades, equity = run_backtest(
            hist_client=MagicMock(), live_client=MagicMock(),
            start_date=date(2026, 1, 1), initial_balance=1000.0,
        )

        assert len(trades) == 1
        t = trades[0]
        assert t.ticker_a == "SA"
        assert t.ticker_b == "SB"
        assert t.balance_at_entry == pytest.approx(1000.0)
        assert t.profit == pytest.approx(t.expected_payoff)
        final_value = float(equity["portfolio_value"].iloc[-1])
        assert final_value == pytest.approx(1000.0 + t.profit)


class TestExtractPairsPerformanceSmoke:
    """Performance regression guard for the windowed time-series sweep.

    Scoped to _extract_pairs directly rather than the full run_backtest
    pipeline: grouping (_group_by_normalized_title / _group_by_exact_title)
    is already a single O(n) pass over the market list, so it was never the
    bottleneck — _extract_pairs' pairwise enumeration is the O(n^2) risk this
    whole change exists to fix, so it's the meaningful unit to time here.
    """

    def test_50k_member_group_completes_in_seconds(self):
        n = 50_000
        base = date(2020, 1, 1)
        # close_time spaced 1 calendar day apart: with a ~31-day window, each
        # member has on the order of ~30 eligible neighbors, so the windowed
        # sweep does roughly n * 30 comparisons (~1.5M) instead of the naive
        # n^2/2 (~1.25 billion) the un-windowed loop would require.
        members = [
            _ts_member(f"T{i}", f"E{i}", base + timedelta(days=i))
            for i in range(n)
        ]
        groups = {"synthetic-ladder": members}

        t0 = time.perf_counter()
        pairs = _extract_pairs(groups, "time_series")
        elapsed = time.perf_counter() - t0

        assert elapsed < 30, f"windowed _extract_pairs took {elapsed:.1f}s for {n} members"
        assert len(pairs) > 0

        # Spot-check a sample of results stay within the margin-inclusive gap
        # (proves the speed isn't coming from silently dropping the loop body)
        margin_days = MAX_DEADLINE_GAP_DAYS + 1
        for mA, mB, _, _ in pairs[:2000]:
            da = datetime.fromisoformat(mA["close_time"]).date()
            db = datetime.fromisoformat(mB["close_time"]).date()
            assert abs((db - da).days) <= margin_days


class TestFetchCandlesParallel:
    """The per-ticker candlestick fetch is the dominant cost of a backtest
    (~4.3 tickers/sec sequentially, live-measured 2026-08-03). It must overlap
    across tickers, stay result-identical to the sequential version, and never
    swallow a worker error.
    """

    @staticmethod
    def _needed(n):
        # Ticker -> market dict, shaped like historical._market_to_dict output.
        return {
            f"T{i:02d}": {"ticker": f"T{i:02d}",
                          "close_time": "2026-02-01T00:00:00+00:00"}
            for i in range(n)
        }

    def test_runs_in_parallel_and_matches_sequential(self, monkeypatch):
        import threading

        monkeypatch.setattr(backtester, "CANDLESTICK_FETCH_MAX_WORKERS", 4)
        needed = self._needed(8)
        expected = {t: [_candle(_MONDAY_TS, 0.70, 0.32)] for t in needed}

        concurrent = 0
        peak = 0
        lock = threading.Lock()
        barrier_wait = threading.Event()

        def slow_fetch(_c, ticker, *_a, **_k):
            nonlocal concurrent, peak
            with lock:
                concurrent += 1
                peak = max(peak, concurrent)
            # Hold the "connection" until enough workers pile up (or we give
            # up), so peak concurrency is observable without a fixed sleep.
            barrier_wait.wait(timeout=2.0)
            with lock:
                if peak >= 4:
                    barrier_wait.set()
                concurrent -= 1
            return expected[ticker]

        monkeypatch.setattr(backtester, "fetch_candlesticks", slow_fetch)
        result = _fetch_candles_parallel(MagicMock(), needed,
                                         date(2026, 1, 1), True)

        assert peak > 1, f"candlestick fetches ran sequentially (peak {peak})"
        # Completion order is arbitrary, but the mapping must be byte-identical
        # to what the old sequential loop produced.
        assert result == expected

    def test_window_bounds_match_sequential_formula(self, monkeypatch):
        # open_ts is hoisted out of the loop now (it only depends on
        # start_date); close_ts is still per-ticker close + one day.
        seen = {}

        def _record(_c, ticker, open_ts, close_ts, use_cache):
            seen[ticker] = (open_ts, close_ts, use_cache)
            return []

        monkeypatch.setattr(backtester, "fetch_candlesticks", _record)
        needed = {
            "EARLY": {"ticker": "EARLY", "close_time": "2026-02-01T00:00:00+00:00"},
            "LATE":  {"ticker": "LATE",  "close_time": "2026-03-01T00:00:00+00:00"},
        }
        _fetch_candles_parallel(MagicMock(), needed, date(2026, 1, 1), False)

        expected_open = int(datetime(2026, 1, 1, tzinfo=UTC).timestamp())
        for ticker, close_time in (("EARLY", "2026-02-01T00:00:00+00:00"),
                                   ("LATE", "2026-03-01T00:00:00+00:00")):
            open_ts, close_ts, use_cache = seen[ticker]
            assert open_ts == expected_open
            assert close_ts == int(datetime.fromisoformat(close_time).timestamp()) + 86400
            assert use_cache is False

    def test_missing_close_time_skips_the_api(self, monkeypatch):
        # No close window to request, so the old loop short-circuited to [] —
        # the parallel version must not hand it to a worker either.
        def _never(*_a, **_k):
            raise AssertionError("fetch_candlesticks called for a market with no close_time")

        monkeypatch.setattr(backtester, "fetch_candlesticks", _never)
        needed = {"NOCLOSE": {"ticker": "NOCLOSE", "close_time": None}}
        assert _fetch_candles_parallel(MagicMock(), needed,
                                       date(2026, 1, 1), True) == {"NOCLOSE": []}

    def test_worker_exception_propagates(self, monkeypatch):
        # fetch_candlesticks already fail-softs network errors to [] on its
        # own, so anything that still raises is a real defect (e.g. a ticker
        # that should have been prefiltered out). The pool must not turn it
        # into "this ticker has no prices".
        def _boom(_c, ticker, *_a, **_k):
            if ticker == "T03":
                raise KeyError(ticker)
            return []

        monkeypatch.setattr(backtester, "fetch_candlesticks", _boom)
        with pytest.raises(KeyError):
            _fetch_candles_parallel(MagicMock(), self._needed(8),
                                    date(2026, 1, 1), True)

    def test_run_backtest_surfaces_worker_exception(self, monkeypatch):
        # Same guarantee end-to-end: the three existing run_backtest fixtures
        # rely on an unknown ticker raising KeyError out of the whole run as a
        # prefilter regression guard, so the pool must stay transparent there.
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
        monkeypatch.setattr(backtester, "fetch_all_settled_markets",
                            lambda *a, **k: markets)
        monkeypatch.setattr(backtester, "fetch_candlesticks",
                            lambda _c, ticker, *a, **k: {}[ticker])

        with pytest.raises(KeyError):
            run_backtest(hist_client=MagicMock(), live_client=MagicMock(),
                         start_date=date(2026, 1, 1), initial_balance=1000.0)


class TestRunBacktestFeasibilityPreCheck:
    """BS-11: no Monday 09:00 UTC checkpoint in the window means no trade can
    ever be entered, so run_backtest must skip the fetch entirely rather than
    discover that only after paying for it.

    `backtester.date` (not the stdlib `datetime.date`) is patched with a
    thin subclass whose `.today()` is frozen, since backtester.py imports
    `date` by name (`from datetime import ... date ...`) and calls
    `date.today()` through that module-level binding.
    """

    class _FrozenDate(date):
        _fixed: date

        @classmethod
        def today(cls):
            return cls._fixed

    def _freeze(self, monkeypatch, today: date):
        frozen = type("FrozenDate", (self._FrozenDate,), {"_fixed": today})
        monkeypatch.setattr(backtester, "date", frozen)

    def _fetch_should_not_be_called(self, monkeypatch):
        def _boom(*_a, **_k):
            raise AssertionError("fetch_all_settled_markets was called despite an infeasible window")

        monkeypatch.setattr(backtester, "fetch_all_settled_markets", _boom)

    def test_start_date_equal_to_today_is_infeasible(self, monkeypatch, caplog):
        # start_date == today on a NON-Monday: the window is the single day
        # [Friday, Friday], which contains no Monday checkpoint at all. (The
        # guard's window now ends at today rather than yesterday, since a
        # market that settled early can carry a future close_time and make
        # today a legitimate checkpoint — so "today" being a Monday would be a
        # feasible window; see test_today_is_monday_is_feasible.)
        today = date(2026, 8, 28)  # a Friday
        self._freeze(monkeypatch, today)
        self._fetch_should_not_be_called(monkeypatch)

        with caplog.at_level("WARNING"):
            trades, equity = run_backtest(
                hist_client=MagicMock(), live_client=MagicMock(),
                start_date=today, initial_balance=1000.0,
            )

        assert trades == []
        assert list(equity.columns) == ["date", "portfolio_value", "daily_return"]
        assert any("no monday" in r.getMessage().lower()
                   for r in caplog.records if r.levelname == "WARNING")

    def test_no_monday_falls_in_a_short_midweek_window(self, monkeypatch, caplog):
        # Tuesday start, "today" the same-week Friday: the range [Tue, Fri]
        # contains no Monday at all — the next Monday doesn't arrive until
        # after "today".
        start = date(2026, 8, 25)  # Tuesday
        today = date(2026, 8, 28)  # Friday, same week
        self._freeze(monkeypatch, today)
        self._fetch_should_not_be_called(monkeypatch)

        with caplog.at_level("WARNING"):
            trades, equity = run_backtest(
                hist_client=MagicMock(), live_client=MagicMock(),
                start_date=start, initial_balance=500.0,
            )

        assert trades == []
        assert equity["portfolio_value"].iloc[0] == pytest.approx(500.0)
        warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert any("no monday" in w.lower() for w in warnings)

    def test_feasible_window_still_fetches(self, monkeypatch):
        # Sanity check on the guard itself: a window that DOES contain a
        # Monday must still reach the fetch call (the existing PnL tests
        # already cover the full pipeline from there).
        today = date(2026, 8, 28)  # Friday
        self._freeze(monkeypatch, today)
        called = []
        monkeypatch.setattr(backtester, "fetch_all_settled_markets",
                            lambda *a, **k: called.append(True) or [])

        run_backtest(hist_client=MagicMock(), live_client=MagicMock(),
                     start_date=date(2026, 8, 10), initial_balance=1000.0)

        assert called == [True]

    def test_today_is_monday_is_feasible(self, monkeypatch):
        # Today IS the only Monday in the window. The guard used to end the
        # window at today - 1 day and wrongly skipped this run — but a market
        # that settled early can carry a close_time in the future, which makes
        # today a legitimate checkpoint. The guard is only for the
        # structurally-impossible case, so it must not fire here.
        today = date(2026, 8, 31)  # Monday
        self._freeze(monkeypatch, today)
        called = []
        monkeypatch.setattr(backtester, "fetch_all_settled_markets",
                            lambda *a, **k: called.append(True) or [])

        run_backtest(hist_client=MagicMock(), live_client=MagicMock(),
                     start_date=date(2026, 8, 26), initial_balance=1000.0)

        assert called == [True]
