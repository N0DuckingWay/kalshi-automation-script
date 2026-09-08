"""Tests for backtester.py — grouping helpers, P&L math, and entry direction."""
import time
from dataclasses import astuple
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

# backtester.py imports no SDK module directly (historical.py reaches every
# /historical route through its own _signed_raw_get, since the pinned SDK has
# no historical_api module at all), so backtester.py is always importable
# and its pure-logic functions are unit-testable offline.
from kalshi_betting import backtester
from kalshi_betting.backtester import (
    _can_ever_enter,
    _extract_pairs,
    _fetch_candles_parallel,
    _find_entry,
    _group_by_exact_title,
    _group_by_normalized_title,
    _interval_calibration,
    _log_interval_calibration,
    _pair_key,
    _parse_iso_date,
    _parse_iso_datetime,
    _settlement_receipt,
    run_backtest,
)
from kalshi_betting.config import (
    BUDGET_FRACTION,
    MAX_DEADLINE_GAP_DAYS,
    TIME_SERIES_INTERVAL_PROB_DISCOUNT,
    fee_leg_exact,
    fee_per_pair_approx,
    min_price_diff_for_gap,
    time_series_profit_prob,
)
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

    def test_distinct_subtitles_separate_groups(self):
        # The subtitle is the only intra-title discriminator: two DIFFERENT
        # outcomes sharing one question title must never land in one group,
        # or the backtester pairs them as the same contract. Cache records now
        # populate subtitle from the API's yes_sub_title (2026-08 drift).
        mA = _md("A1", "EVT-A", title="Who will the next Pope be?",
                 subtitle="Pierbattista Pizzaballa", event_title="Papal Conclave")
        mB = _md("B1", "EVT-B", title="Who will the next Pope be?",
                 subtitle="Peter Turkson", event_title="Papal Conclave")
        # Each lands alone, then the len>=2 filter drops both.
        assert _group_by_exact_title([mA, mB]) == {}

    def test_identical_subtitles_group_together(self):
        mA = _md("A1", "EVT-A", title="Who will the next Pope be?",
                 subtitle="Pierbattista Pizzaballa", event_title="Papal Conclave")
        mB = _md("B1", "EVT-B", title="Who will the next Pope be?",
                 subtitle="Pierbattista Pizzaballa", event_title="Papal Conclave")
        groups = _group_by_exact_title([mA, mB])
        assert len(groups) == 1
        key = next(iter(groups))
        assert key == ("Papal Conclave", "Who will the next Pope be?",
                       "Pierbattista Pizzaballa")
        assert len(groups[key]) == 2


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
        pairs = _extract_pairs(groups)
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
        pairs = _extract_pairs(groups)
        assert len(pairs) == 2
        canons = {canon for _, _, canon, _ in pairs}
        assert canons == {"Trump"}
        group_keys = {group_key for _, _, _, group_key in pairs}
        assert len(group_keys) == 2, "distinct events must yield distinct group_keys"


class TestOldCacheToleranceMissingTickAndSubtitleFields:
    """Cache records written before the tick-structure groundwork (2026-08)
    have no price_level_structure/price_ranges keys at all — and pre-2026-08
    records may also lack subtitle entirely, not just carry it as "". None of
    the grouping/extraction helpers should ever KeyError on that; they only
    ever read via .get()."""

    @staticmethod
    def _old_style_dict(ticker, event_ticker, title, close_time, event_title=""):
        # Deliberately omits price_level_structure, price_ranges, AND subtitle
        # — the exact shape of a pre-2026-08 cache record.
        return {
            "ticker": ticker,
            "event_ticker": event_ticker,
            "event_title": event_title,
            "title": title,
            "close_time": close_time,
        }

    def test_pair_key_handles_missing_fields(self):
        m = self._old_style_dict("T1", "E1", "Will BTC exceed $80k",
                                  "2026-01-01T00:00:00Z")
        assert _pair_key(m) == "Will BTC exceed $80k"

    def test_group_by_exact_title_handles_missing_fields(self):
        mA = self._old_style_dict("A1", "EVT-A", "Republicans win majority",
                                   "2026-01-01T00:00:00Z", event_title="2026 Senate Control")
        mB = self._old_style_dict("B1", "EVT-B", "Republicans win majority",
                                   "2026-01-08T00:00:00Z", event_title="2026 Senate Control")
        groups = _group_by_exact_title([mA, mB])
        assert len(groups) == 1

    def test_extract_pairs_same_title_handles_missing_fields(self):
        mA = self._old_style_dict("A1", "EVT-A", "Republicans win majority",
                                   "2026-01-01T00:00:00Z", event_title="2026 Senate Control")
        mB = self._old_style_dict("B1", "EVT-B", "Republicans win majority",
                                   "2026-01-08T00:00:00Z", event_title="2026 Senate Control")
        groups = _group_by_exact_title([mA, mB])
        pairs = _extract_pairs(groups)
        assert len(pairs) == 1

    def test_extract_pairs_time_series_handles_missing_fields(self):
        # Deadline gap kept within MAX_DEADLINE_GAP_DAYS + 1 margin so the
        # windowed sweep actually materializes a pair (not a windowing edge case).
        # "Month day, year" is stripped by normalize_title regardless of day,
        # so both titles normalize to the same key.
        mA = self._old_style_dict("A1", "EVT-A", "Will BTC exceed $80k by March 1, 2026",
                                   "2026-03-01T00:00:00Z")
        mB = self._old_style_dict("B1", "EVT-B", "Will BTC exceed $80k by March 20, 2026",
                                   "2026-03-20T00:00:00Z")
        groups = _group_by_normalized_title([mA, mB])
        pairs = _extract_pairs(groups)
        assert len(pairs) == 1


class TestSettlementReceipt:
    """_settlement_receipt pays n per leg whose market resolved to the side
    that leg bought (scanner.leg_sides), so the table depends on pair type."""

    def test_same_title_payoff_table(self):
        # n NO contracts on A + n YES contracts on B; each winning contract
        # pays exactly $1 — the receipt is independent of entry prices.
        n = 10
        assert _settlement_receipt(n, "yes", "yes", "same_title") == 10   # only B pays
        assert _settlement_receipt(n, "no", "yes", "same_title") == 20    # both pay
        assert _settlement_receipt(n, "no", "no", "same_title") == 10     # only A pays
        assert _settlement_receipt(n, "yes", "no", "same_title") == 0     # loss scenario

    def test_time_series_three_cell_table(self):
        # n YES on the earlier contract A + n NO on the later contract B.
        # Exactly three cells exist for a cumulative-deadline pair.
        n = 10
        assert _settlement_receipt(n, "yes", "yes", "time_series") == 10  # event by A: YES on A pays
        assert _settlement_receipt(n, "no", "no", "time_series") == 10    # never by B: NO on B pays
        assert _settlement_receipt(n, "no", "yes", "time_series") == 0    # in between: loss cell

    def test_time_series_premise_violation_raises(self):
        # Earlier YES with later NO cannot happen for a cumulative-deadline
        # pair — it is a premise violation, never a payout cell.
        with pytest.raises(ValueError, match="premise violated"):
            _settlement_receipt(10, "yes", "no", "time_series")

    def test_unknown_pair_type_uses_same_title_sides(self):
        # Mirrors scanner.leg_sides: anything but "time_series" is same-title,
        # so an unknown label can never be paid as the directional bet.
        assert _settlement_receipt(10, "no", "yes", "bogus") == 20


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

    def test_time_series_rejects_pricier_earlier_contract(self):
        # Earlier pricey (0.60), later cheap (0.30): the live scanner never
        # trades this direction (the anomaly it disputes is the LATER contract
        # priced higher), so the backtest must not either.
        mA, mB = self._markets()
        candles_early = [_candle(_MONDAY_TS, 0.60, 0.40)]
        candles_late  = [_candle(_MONDAY_TS, 0.30, 0.70)]
        entry = _find_entry(candles_early, candles_late, mA, mB,
                            "time_series", date(2026, 1, 1))
        assert entry is None

    def test_time_series_accepts_pricier_later_contract(self):
        # Earlier cheap (pA=0.30), later pricey (pB=0.60, nB=0.40). At the
        # 13-day deadline gap the short tier applies: price gap 0.30 >= 0.15,
        # leg prices pA+nB = 0.70 <= 0.85, spread clears fees.
        mA, mB = self._markets()
        candles_early = [_candle(_MONDAY_TS, 0.30, 0.70)]
        candles_late  = [_candle(_MONDAY_TS, 0.60, 0.40)]
        entry = _find_entry(candles_early, candles_late, mA, mB,
                            "time_series", date(2026, 1, 1))
        assert entry is not None
        # Market A must be the earlier-closing contract (the YES leg)
        assert entry["mA"]["ticker"] == "EARLY"
        assert entry["pA"] == pytest.approx(0.30)
        assert entry["pB"] == pytest.approx(0.60)
        assert entry["nA"] == pytest.approx(0.70)
        assert entry["nB"] == pytest.approx(0.40)

    def test_time_series_swaps_to_keep_a_as_the_earlier_contract(self):
        # Same fixture with the markets passed in the opposite order: A must
        # still come back as EARLY, with every quote following its market.
        mA, mB = self._markets()
        candles_early = [_candle(_MONDAY_TS, 0.30, 0.70)]
        candles_late  = [_candle(_MONDAY_TS, 0.60, 0.40)]
        entry = _find_entry(candles_late, candles_early, mB, mA,
                            "time_series", date(2026, 1, 1))
        assert entry is not None
        assert entry["mA"]["ticker"] == "EARLY"
        assert entry["mB"]["ticker"] == "LATE"
        assert (entry["pA"], entry["nB"]) == pytest.approx((0.30, 0.40))

    def test_time_series_wide_later_book_is_rejected_by_the_sum_ceiling(self):
        # Same YES asks (gap 0.30 clears the tier) but the later NO ask is
        # 0.60: the traded legs pA+nB = 0.90 exceed the short-tier ceiling of
        # 0.85 — the ceiling is applied to the LEG prices, not to (nA, pB).
        mA, mB = self._markets()
        candles_early = [_candle(_MONDAY_TS, 0.30, 0.70)]
        candles_late  = [_candle(_MONDAY_TS, 0.60, 0.60)]
        entry = _find_entry(candles_early, candles_late, mA, mB,
                            "time_series", date(2026, 1, 1))
        assert entry is None

    def test_same_title_canonicalizes_by_price(self):
        # For same-title pairs, direction is price-only: A = expensive side.
        mA, mB = self._markets()
        candles_a = [_candle(_MONDAY_TS, 0.55, 0.47)]
        candles_b = [_candle(_MONDAY_TS, 0.70, 0.32)]
        entry = _find_entry(candles_a, candles_b, mA, mB,
                            "same_title", date(2026, 1, 1))
        assert entry is not None
        assert entry["mA"]["ticker"] == "LATE"  # the pricier side becomes A
        # The canonical B's NO ask is carried too (reporting only for same_title)
        assert entry["nB"] == pytest.approx(0.47)


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
        # Tight books on both legs (NO ask = 1 - YES ask), so the traded leg
        # prices pA + nB sum to exactly 1 - (pB - pA): the price-sum ceiling
        # binds precisely when the gap is under the tier.
        mA, mB = self._markets(gap_days)
        candles_early = [_candle(_MONDAY_TS, pA, round(1.0 - pA, 4))]
        candles_late  = [_candle(_MONDAY_TS, pB, round(1.0 - pB, 4))]
        return _find_entry(candles_early, candles_late, mA, mB,
                           "time_series", date(2026, 1, 1))

    def test_short_gap_18pct_price_gap_accepted(self):
        # 10-day deadline gap → 15% tier; pA=0.30/pB=0.48 (18% gap, leg prices
        # pA+nB = 0.30+0.52 = 0.82 <= 0.85) qualifies
        entry = self._entry(10, pA=0.30, pB=0.48)
        assert entry is not None
        assert entry["mA"]["ticker"] == "EARLY"
        assert entry["nB"] == pytest.approx(0.52)

    def test_long_gap_18pct_price_gap_rejected(self):
        # The SAME prices at a 20-day deadline gap fall under the 30% tier
        # and must be rejected — this is exactly what a flat 15% threshold
        # would (wrongly) accept
        assert self._entry(20, pA=0.30, pB=0.48) is None

    def test_long_gap_38pct_price_gap_accepted(self):
        # 20-day gap → 30% tier; pA=0.30/pB=0.68 (38% gap, leg prices pA+nB =
        # 0.30+0.32 = 0.62 <= 0.70) clears both the tiered threshold and the
        # tiered price-sum ceiling
        entry = self._entry(20, pA=0.30, pB=0.68)
        assert entry is not None

    def test_over_max_gap_rejected_regardless_of_price(self):
        # 35-day deadline gap exceeds MAX_DEADLINE_GAP_DAYS — even a 40% price
        # gap never enters
        assert self._entry(35, pA=0.30, pB=0.70) is None

    def test_same_title_ignores_deadline_gap(self):
        # same_title keeps the flat 5% threshold: a 6% divergence on markets
        # closing 20 days apart is still an entry (nA+pB = 0.94 <= 0.95)
        mA, mB = self._markets(20)
        candles_a = [_candle(_MONDAY_TS, 0.36, 0.64)]
        candles_b = [_candle(_MONDAY_TS, 0.30, 0.70)]
        entry = _find_entry(candles_a, candles_b, mA, mB,
                            "same_title", date(2026, 1, 1))
        assert entry is not None

    def test_gap_tier_uses_datetime_arithmetic_like_live_scanner(self):
        # The deadline gap must be measured the way scanner.find_time_series_pairs
        # measures it: timedelta.days on the tz-aware close_time DATETIMES, which
        # floors. Feb 1 23:00Z -> Feb 17 01:00Z is 15 days 2 hours, so the live
        # gap is 15 -> the SHORT (15%) tier. Calendar-date subtraction would
        # count 16 boundaries and wrongly apply the long (30%) tier.
        mA = {"ticker": "EARLY", "event_ticker": "E1",
              "close_time": "2026-02-01T23:00:00+00:00"}
        mB = {"ticker": "LATE", "event_ticker": "E2",
              "close_time": "2026-02-17T01:00:00+00:00"}
        # 25% price gap (later 0.60 over earlier 0.35): clears the short tier
        # (>= 0.15) but not the long one (>= 0.30); leg prices pA + nB = 0.75
        # <= 0.85, the short tier's price-sum ceiling.
        candles_early = [_candle(_MONDAY_TS, 0.35, 0.65)]
        candles_late  = [_candle(_MONDAY_TS, 0.60, 0.40)]
        entry = _find_entry(candles_early, candles_late, mA, mB,
                            "time_series", date(2026, 1, 1))
        assert entry is not None
        assert entry["mA"]["ticker"] == "EARLY"

    def test_max_gap_cutoff_uses_datetime_arithmetic_like_live_scanner(self):
        # Same disagreement at the MAX_DEADLINE_GAP_DAYS (30) cutoff. Feb 1
        # 23:00Z -> Mar 4 01:00Z is 30 days 2 hours (Feb 2026 has 28 days), so
        # the live gap is exactly 30 — allowed, long tier. Calendar-date
        # subtraction counts 31 and rejects the pair outright.
        mA = {"ticker": "EARLY", "event_ticker": "E1",
              "close_time": "2026-02-01T23:00:00+00:00"}
        mB = {"ticker": "LATE", "event_ticker": "E2",
              "close_time": "2026-03-04T01:00:00+00:00"}
        # 40% price gap (later 0.70 over earlier 0.30) clears the long tier
        # (>= 0.30); leg prices pA + nB = 0.60 <= 0.70.
        candles_early = [_candle(_MONDAY_TS, 0.30, 0.70)]
        candles_late  = [_candle(_MONDAY_TS, 0.70, 0.30)]
        entry = _find_entry(candles_early, candles_late, mA, mB,
                            "time_series", date(2026, 1, 1))
        assert entry is not None
        assert entry["mA"]["ticker"] == "EARLY"


class TestFindEntryHorizon:
    """_find_entry's optional max_horizon_days caps how far the later-closing
    leg's close date may be from the checkpoint being evaluated — relative to
    the SIMULATED Monday, not real-world now."""

    def _markets(self):
        # Same 13-day-gap fixture as TestFindEntryDirection — close_a
        # 2026-02-01, close_b 2026-02-14; every case prices the later
        # contract higher (0.60 over 0.30) so the price gap always qualifies.
        mA = {"ticker": "EARLY", "event_ticker": "E1",
              "close_time": "2026-02-01T00:00:00+00:00"}
        mB = {"ticker": "LATE", "event_ticker": "E2",
              "close_time": "2026-02-14T00:00:00+00:00"}
        return mA, mB

    def test_none_horizon_matches_unfiltered_behavior(self):
        mA, mB = self._markets()
        candles_early = [_candle(_MONDAY_TS, 0.30, 0.70)]
        candles_late  = [_candle(_MONDAY_TS, 0.60, 0.40)]
        entry = _find_entry(candles_early, candles_late, mA, mB,
                            "time_series", date(2026, 1, 1), max_horizon_days=None)
        assert entry is not None
        assert entry["entry_date"] == date(2026, 1, 5)

    def test_within_horizon_at_first_checkpoint_enters_immediately(self):
        # At the first Monday (Jan 5), close_b (Feb 14) is 40 days out — well
        # within a generous 60-day horizon, so entry is unaffected.
        mA, mB = self._markets()
        candles_early = [_candle(_MONDAY_TS, 0.30, 0.70)]
        candles_late  = [_candle(_MONDAY_TS, 0.60, 0.40)]
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
        candles_early = [_candle(_MONDAY_TS, 0.30, 0.70)]
        candles_late  = [_candle(_MONDAY_TS, 0.60, 0.40)]
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
        candles_early = [_candle(_MONDAY_TS, 0.30, 0.70)]
        candles_late  = [_candle(_MONDAY_TS, 0.60, 0.40)]
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

    Prices: TX yes 0.40 / no 0.60, TY yes 0.30 / no 0.70, TZ yes 0.75 / no
    0.25. Same-title TX/TY: TX is the pricier side, gap 0.10 >= 0.05, legs
    nA+pB = 0.60+0.30 = 0.90 <= 0.95. Time-series TX/TZ: TX is the earlier
    contract and TZ (later) is priced 0.35 higher, clearing the 30% long-gap
    tier; legs pA+nB = 0.40+0.25 = 0.65 <= 0.70, and under the interval
    discount (p = 1 - 0.75*0.35) the Kelly fraction is ~0.204 — positive, so
    the pair really is entered. TX/TY as a time-series pair (13-day gap, TY
    earlier at 0.30 vs TX 0.40) misses the 15% tier, and TY/TZ is 31 days
    apart — beyond MAX_DEADLINE_GAP_DAYS — so TX/TZ is the only time-series
    candidate.
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
            # TX: pricier than TY (same-title gap) yet cheaper than the
            # later-closing TZ (time-series gap) — see the class docstring
            "TX": [_candle(_MONDAY_TS, 0.40, 0.60)],
            "TY": [_candle(_MONDAY_TS, 0.30, 0.70)],
            # TZ's first candle is the SECOND Monday, so the TX/TZ pair cannot
            # enter until then
            "TZ": [_candle(self._M2, 0.75, 0.25)],
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

        windowed = _extract_pairs(groups)
        windowed_set = {frozenset([a["ticker"], b["ticker"]]) for a, b, _, _ in windowed}

        margin_days = MAX_DEADLINE_GAP_DAYS + 1
        naive_set = _naive_time_series_pairs(members, margin_days)

        assert windowed_set == naive_set
        # Sanity: the set isn't trivially empty or trivially "everything"
        assert 0 < len(windowed_set) < (len(members) * (len(members) - 1)) // 2

    def test_boundary_probes_land_exactly_where_expected(self):
        members = self._build_synthetic_group()
        groups = {"synthetic": members}
        windowed = _extract_pairs(groups)
        pair_tickers = {frozenset([a["ticker"], b["ticker"]]) for a, b, _, _ in windowed}

        assert frozenset(["REF", "PLUS30"]) in pair_tickers
        assert frozenset(["REF", "PLUS31"]) in pair_tickers   # inside the +1 margin
        assert frozenset(["REF", "PLUS32"]) not in pair_tickers  # outside the margin

    def test_cross_cluster_pairs_never_appear(self):
        members = self._build_synthetic_group()
        windowed = _extract_pairs({"synthetic": members})
        for a, b, _, _ in windowed:
            assert not (a["ticker"].startswith("A") and b["ticker"].startswith("B"))
            assert not (a["ticker"].startswith("B") and b["ticker"].startswith("A"))

    def test_same_event_ticker_pair_is_skipped(self):
        members = self._build_synthetic_group()
        windowed = _extract_pairs({"synthetic": members})
        pair_tickers = {frozenset([a["ticker"], b["ticker"]]) for a, b, _, _ in windowed}
        assert frozenset(["SAMEEVT-1", "SAMEEVT-2"]) not in pair_tickers

    def test_missing_close_time_member_produces_no_pairs(self):
        members = self._build_synthetic_group()
        windowed = _extract_pairs({"synthetic": members})
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
        pairs = _extract_pairs(groups)
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

    def test_candlestick_progress_denominator_counts_only_fetched_tickers(
        self, monkeypatch, caplog,
    ):
        # C6: 5 of the 56 needed tickers have no close_time and are resolved
        # to [] without ever entering `work`, so only 51 tickers actually pass
        # through a worker. The progress line's denominator must reflect that
        # (len(work)), not len(needed_tickers) (56) — the old bug meant the
        # counter could never reach its own denominator whenever any ticker
        # was skipped.
        needed = self._needed(51)
        for i in range(5):
            needed[f"NOCLOSE{i:02d}"] = {"ticker": f"NOCLOSE{i:02d}", "close_time": None}

        monkeypatch.setattr(backtester, "fetch_candlesticks",
                            lambda *_a, **_k: [])

        with caplog.at_level("INFO"):
            _fetch_candles_parallel(MagicMock(), needed, date(2026, 1, 1), False)

        messages = [r.getMessage() for r in caplog.records if r.levelname == "INFO"]
        assert any(m.endswith("50 / 51") for m in messages)
        assert not any("50 / 56" in m for m in messages)

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


class TestDropCrossTypeDuplicates:
    """Pass 1 must not carry the same ticker pair as both a same-title and a
    time-series candidate — the live pipeline never does (main._dedup_pairs)."""

    def test_drop_cross_type_duplicates_prefers_same_title(self, caplog):
        same = {"pair_type": "same_title",
                "mA": {"ticker": "X"}, "mB": {"ticker": "Y"}}
        # Same two tickers as `same`, but discovered in the OPPOSITE leg order:
        # the frozenset key must still recognise it as the same pair.
        dupe = {"pair_type": "time_series",
                "mA": {"ticker": "Y"}, "mB": {"ticker": "X"}}
        other = {"pair_type": "time_series",
                 "mA": {"ticker": "P"}, "mB": {"ticker": "Q"}}

        with caplog.at_level("INFO"):
            result = backtester._drop_cross_type_duplicates([same, dupe, other])

        # Input order is preserved and only the cross-type duplicate is dropped
        assert result == [same, other]
        messages = [r.getMessage() for r in caplog.records if r.levelname == "INFO"]
        assert any("Dropped 1 time-series candidate(s) already covered" in m
                   for m in messages), messages

    def test_no_duplicates_logs_nothing_and_returns_input_order(self, caplog):
        same = {"pair_type": "same_title",
                "mA": {"ticker": "X"}, "mB": {"ticker": "Y"}}
        other = {"pair_type": "time_series",
                 "mA": {"ticker": "P"}, "mB": {"ticker": "Q"}}

        with caplog.at_level("INFO"):
            result = backtester._drop_cross_type_duplicates([other, same])

        assert result == [other, same]
        assert not any("already covered by a same-title candidate" in r.getMessage()
                       for r in caplog.records)


class TestRunBacktestCrossTypeDedup:
    """End-to-end proof that run_backtest applies the cross-type dedup (C4).

    Fixture shape: two markets sharing an identical (event_title, title,
    subtitle) — so _group_by_exact_title pairs them as same_title — whose
    titles therefore also normalize to one key, so _group_by_normalized_title
    pairs the SAME two tickers as time_series. Distinct close dates 7 days
    apart keep the time-series copy inside the short (<= 15 day) tier, so it
    genuinely qualifies at the 15% threshold rather than being filtered out
    by _find_entry.
    """

    _MARKETS = [
        {"ticker": "DA", "event_ticker": "EA", "event_title": "EV",
         "title": "Q", "subtitle": "", "result": "yes",
         "close_time": "2026-02-01T00:00:00+00:00",
         "settlement_ts": "2026-02-01T12:00:00+00:00"},
        {"ticker": "DB", "event_ticker": "EB", "event_title": "EV",
         "title": "Q", "subtitle": "", "result": "yes",
         "close_time": "2026-02-08T00:00:00+00:00",
         "settlement_ts": "2026-02-08T12:00:00+00:00"},
    ]
    # DA (earlier, closes Feb 1) yes 0.30 / no 0.70; DB (later, Feb 8) yes
    # 0.60 / no 0.40 — the flow-through fixture. Same-title copy: DB is the
    # pricier side, gap 0.30 >= 0.05, legs nA+pB = 0.40+0.30 = 0.70 <= 0.95.
    # Time-series copy: the later contract is priced 0.30 higher, clearing the
    # 15% short-gap tier; legs pA+nB = 0.30+0.40 = 0.70 <= 0.85, and under the
    # interval discount the Kelly fraction is ~0.188 — positive, so BOTH
    # copies form in Pass 1 and the dedup under test is not vacuous.
    _CANDLES = {
        "DA": [_candle(_MONDAY_TS, 0.30, 0.70)],
        "DB": [_candle(_MONDAY_TS, 0.60, 0.40)],
    }

    def test_fixture_lands_in_both_groupings(self):
        # The whole test rests on this pair being discovered twice, so assert it
        # directly rather than trusting the grouping helpers to stay aligned.
        assert len(_group_by_exact_title(self._MARKETS)) == 1
        assert len(_group_by_normalized_title(self._MARKETS)) == 1
        assert len(_extract_pairs(_group_by_exact_title(self._MARKETS))) == 1
        assert len(_extract_pairs(_group_by_normalized_title(self._MARKETS))) == 1

    def test_time_series_duplicate_is_dropped_before_pass_two(self, monkeypatch):
        monkeypatch.setattr(backtester, "fetch_all_settled_markets",
                            lambda *a, **k: self._MARKETS)
        monkeypatch.setattr(backtester, "fetch_candlesticks",
                            lambda _c, ticker, *a, **k: self._CANDLES[ticker])

        real = backtester._drop_cross_type_duplicates
        seen: dict = {}

        def _spy(candidates):
            seen["in"] = list(candidates)
            out = real(candidates)
            seen["out"] = list(out)
            return out

        monkeypatch.setattr(backtester, "_drop_cross_type_duplicates", _spy)

        trades, _ = run_backtest(
            hist_client=MagicMock(), live_client=MagicMock(),
            start_date=date(2026, 1, 1), initial_balance=1000.0,
        )

        key = frozenset({"DA", "DB"})

        def _typed(cands, pair_type):
            return [c for c in cands
                    if c["pair_type"] == pair_type
                    and frozenset({c["mA"]["ticker"], c["mB"]["ticker"]}) == key]

        # Pass 1 really does produce BOTH copies of this ticker pair — without
        # that, the dedup under test would be vacuous.
        assert len(_typed(seen["in"], "same_title")) == 1
        assert len(_typed(seen["in"], "time_series")) == 1
        # ...and only the same-title copy survives into Pass 2.
        assert len(_typed(seen["out"], "same_title")) == 1
        assert _typed(seen["out"], "time_series") == []

        assert len(trades) == 1
        assert trades[0].pair_type == "same_title"
        # The same-title copy canonicalizes A as the pricier side (DB)
        assert (trades[0].ticker_a, trades[0].ticker_b) == ("DB", "DA")


class TestCheckpointOpeningBalanceSizing:
    """Pass 2 must size every candidate of one entry date against that
    checkpoint's OPENING balance (live: one verify_auth read per run feeding
    every compute_trade call), then admit greedily against the running cash
    (live: strategy.select_portfolio's decrementing budget)."""

    # Two same-title pairs whose only qualifying Monday is 2026-01-05. Distinct
    # event_titles ("EV1"/"EV2") keep them in separate (event_title, title,
    # subtitle) groups, so Pass 1's best-per-group filter keeps both.
    @staticmethod
    def _markets():
        out = []
        for n, ev in ((1, "EV1"), (2, "EV2")):
            out.append({"ticker": f"S{n}A", "event_ticker": f"E{n}A", "event_title": ev,
                        "title": f"Q{n}", "subtitle": "", "result": "yes",
                        "close_time": "2026-02-01T00:00:00+00:00",
                        "settlement_ts": "2026-02-01T12:00:00+00:00"})
            out.append({"ticker": f"S{n}B", "event_ticker": f"E{n}B", "event_title": ev,
                        "title": f"Q{n}", "subtitle": "", "result": "yes",
                        "close_time": "2026-02-01T00:00:00+00:00",
                        "settlement_ts": "2026-02-01T12:00:00+00:00"})
        return out

    # Identical prices for both pairs: expensive leg yes 0.60 / no 0.40, cheap
    # leg yes 0.35 — a 0.25 gap, so both clear SAME_TITLE_MIN_PRICE_DIFF and
    # both land on the same capped Kelly fraction.
    @staticmethod
    def _candles():
        return {
            "S1A": [_candle(_MONDAY_TS, 0.60, 0.40)],
            "S1B": [_candle(_MONDAY_TS, 0.35, 0.65)],
            "S2A": [_candle(_MONDAY_TS, 0.60, 0.40)],
            "S2B": [_candle(_MONDAY_TS, 0.35, 0.65)],
        }

    def _patch(self, monkeypatch):
        markets, candles = self._markets(), self._candles()
        monkeypatch.setattr(backtester, "fetch_all_settled_markets",
                            lambda *a, **k: markets)
        monkeypatch.setattr(backtester, "fetch_candlesticks",
                            lambda _c, ticker, *a, **k: candles[ticker])

    def test_same_day_trades_are_sized_off_the_checkpoint_opening_balance(self, monkeypatch):
        # Both pairs enter on the same Monday. Live would size BOTH against the
        # one balance read at the top of the run, so both must get the same n
        # and record the same balance_at_entry — the second must not be shrunk
        # by the first's cost.
        self._patch(monkeypatch)

        trades, _ = run_backtest(
            hist_client=MagicMock(), live_client=MagicMock(),
            start_date=date(2026, 1, 1), initial_balance=1000.0,
        )

        assert len(trades) == 2
        assert {t.entry_date for t in trades} == {date(2026, 1, 5)}
        assert trades[0].n == trades[1].n
        for t in trades:
            assert t.balance_at_entry == pytest.approx(1000.0)
            # Each trade's fee-inclusive outlay fits the Kelly budget taken off
            # the CHECKPOINT balance, not off whatever cash was left.
            assert t.total_cost + t.fees <= 1000.0 * t.kelly_fraction + 1e-9

    def test_greedy_fit_skips_rather_than_shrinks(self, monkeypatch):
        # With the real BUDGET_FRACTION (0.20) two same-day trades always fit
        # (0.2 + 0.2 < 1), so the greedy-skip branch is unreachable. Raise the
        # cap to 0.60 for this test only — backtester imports the constant by
        # value (`from .config import BUDGET_FRACTION`), so patching the module
        # attribute is what Pass 1's `min(BUDGET_FRACTION, kelly_f)` reads.
        monkeypatch.setattr(backtester, "BUDGET_FRACTION", 0.60)
        self._patch(monkeypatch)

        trades, _ = run_backtest(
            hist_client=MagicMock(), live_client=MagicMock(),
            start_date=date(2026, 1, 1), initial_balance=1000.0,
        )

        # The first trade takes ~60% of the balance; the second, sized off the
        # same checkpoint balance, no longer fits the ~40% left. select_portfolio
        # SKIPS such a spec — it never shrinks it to fit.
        assert len(trades) == 1
        t = trades[0]
        assert t.kelly_fraction == pytest.approx(0.60)
        assert t.balance_at_entry == pytest.approx(1000.0)
        assert t.total_cost + t.fees <= 1000.0 * 0.60 + 1e-9
        # Not shrunk: it is still the full-size trade the checkpoint budget buys.
        assert t.total_cost + t.fees > 1000.0 * 0.50


class TestRunBacktestTimeSeriesFlow:
    """End-to-end flow of a time-series pair through run_backtest.

    Fixture (the plan's hand-picked illustrative numbers, not market data):
    EA closes 2026-02-01 and EB 2026-02-14 (13-day gap, short tier, threshold
    0.15), same event_title "EV", distinct event tickers, titles that
    normalize to one key but are NOT exact-title equal (so no same-title copy
    forms to dedup the pair away), open_time 2026-01-01, settlement = close +
    12h. Candles at the first Monday: EA yes 0.30 / no 0.70, EB yes 0.60 / no
    0.40. Balance $10,000.

    Legs are YES on EA at 0.30 and NO on EB at 0.40: gap 0.30 >= 0.15, cost
    0.70 <= 0.85, fee_approx 0.0315 < 0.30 => entry. Pass 1: net 0.2685,
    b 0.3836, p = 1 - 0.75*0.30 = 0.775, f* = 0.1884 (below the 0.20 cap, so
    Kelly sizes it). Pass 2: budget 1884.08 => raw n 2691, shrunk to 2575 by
    the fee loop (cost 1802.50, exact fees 81.12, cash out 1883.62), win
    profit 691.38. Settlement: event by EA => +691.38; never by EB =>
    +691.38; in between => -1883.62; EA yes / EB no is a premise violation
    and is excluded with a counted WARNING.
    """

    _PA, _NA = 0.30, 0.70   # EA (earlier) YES / NO ask
    _PB, _NB = 0.60, 0.40   # EB (later) YES / NO ask

    @staticmethod
    def _markets(result_a: str, result_b: str) -> list[dict]:
        return [
            {"ticker": "EA", "event_ticker": "EVA", "event_title": "EV",
             "title": "Team wins by February 1, 2026", "subtitle": "",
             "result": result_a,
             "open_time": "2026-01-01T00:00:00+00:00",
             "close_time": "2026-02-01T00:00:00+00:00",
             "settlement_ts": "2026-02-01T12:00:00+00:00"},
            {"ticker": "EB", "event_ticker": "EVB", "event_title": "EV",
             "title": "Team wins by February 14, 2026", "subtitle": "",
             "result": result_b,
             "open_time": "2026-01-01T00:00:00+00:00",
             "close_time": "2026-02-14T00:00:00+00:00",
             "settlement_ts": "2026-02-14T12:00:00+00:00"},
        ]

    def _run(self, monkeypatch, result_a, result_b, eb_yes=None, eb_no=None):
        eb_yes = self._PB if eb_yes is None else eb_yes
        eb_no = self._NB if eb_no is None else eb_no
        candles = {
            "EA": [_candle(_MONDAY_TS, self._PA, self._NA)],
            "EB": [_candle(_MONDAY_TS, eb_yes, eb_no)],
        }
        markets = self._markets(result_a, result_b)
        monkeypatch.setattr(backtester, "fetch_all_settled_markets",
                            lambda *a, **k: markets)
        monkeypatch.setattr(backtester, "fetch_candlesticks",
                            lambda _c, ticker, *a, **k: candles[ticker])
        return run_backtest(
            hist_client=MagicMock(), live_client=MagicMock(),
            start_date=date(2026, 1, 1), initial_balance=10_000.0,
        )

    def test_fixture_is_a_time_series_group_only(self):
        # The pair must be discovered by the normalized-title grouping and NOT
        # by the exact-title one — otherwise the cross-type dedup would drop
        # the very candidate this class exercises.
        markets = self._markets("yes", "yes")
        assert len(_group_by_normalized_title(markets)) == 1
        assert _group_by_exact_title(markets) == {}

    @staticmethod
    def _expected_kelly(pA, nB, pB):
        # p - (1 - p)/b computed from the config helpers, so this pins the
        # model THROUGH run_backtest rather than a hardcoded number
        net = (1.0 - pA - nB) - fee_per_pair_approx(pA, nB)
        b = net / (pA + nB)
        p = time_series_profit_prob(pA, pB)
        return p - (1.0 - p) / b

    def _assert_entry_and_sizing(self, t):
        assert t.pair_type == "time_series"
        assert (t.ticker_a, t.ticker_b) == ("EA", "EB")
        assert t.entry_date == date(2026, 1, 5)
        assert t.exit_date == date(2026, 2, 14)
        assert t.entry_pA == pytest.approx(self._PA)
        assert t.entry_pB == pytest.approx(self._PB)
        assert t.entry_nA == pytest.approx(self._NA)
        assert t.entry_nB == pytest.approx(self._NB)
        expected_f = self._expected_kelly(self._PA, self._NB, self._PB)
        assert expected_f == pytest.approx(0.1884, abs=5e-4)
        assert expected_f < BUDGET_FRACTION  # Kelly, not the cap, sized this pair
        assert t.kelly_fraction == pytest.approx(expected_f)
        assert t.balance_at_entry == pytest.approx(10_000.0)
        assert t.n == 2575
        # Sized on the LEG prices (pA + nB), never on (nA + pB)
        assert t.total_cost == pytest.approx(2575 * (self._PA + self._NB))
        assert t.total_cost == pytest.approx(1802.50)
        assert t.fees == pytest.approx(
            fee_leg_exact(2575, self._PA) + fee_leg_exact(2575, self._NB))
        assert t.fees == pytest.approx(81.12)
        assert t.total_cost + t.fees <= 10_000.0 * t.kelly_fraction + 1e-9
        assert t.expected_payoff == pytest.approx(691.38)

    def test_event_by_earlier_deadline_wins(self, monkeypatch):
        # EA yes, EB yes: YES on EA pays n, NO on EB worthless
        trades, equity = self._run(monkeypatch, "yes", "yes")
        assert len(trades) == 1
        t = trades[0]
        self._assert_entry_and_sizing(t)
        assert t.actual_payoff == pytest.approx(2575.0)
        assert t.profit == pytest.approx(691.38)
        assert t.slippage == pytest.approx(0.0, abs=1e-9)
        assert float(equity["portfolio_value"].iloc[-1]) == pytest.approx(10_691.38)

    def test_event_never_by_later_deadline_wins(self, monkeypatch):
        # EA no, EB no: NO on EB pays n, YES on EA worthless
        trades, equity = self._run(monkeypatch, "no", "no")
        assert len(trades) == 1
        t = trades[0]
        self._assert_entry_and_sizing(t)
        assert t.actual_payoff == pytest.approx(2575.0)
        assert t.profit == pytest.approx(691.38)
        assert t.slippage == pytest.approx(0.0, abs=1e-9)
        assert float(equity["portfolio_value"].iloc[-1]) == pytest.approx(10_691.38)

    def test_event_in_between_loses_the_full_stake(self, monkeypatch):
        # EA no, EB yes: both legs worthless — the loss cell
        trades, equity = self._run(monkeypatch, "no", "yes")
        assert len(trades) == 1
        t = trades[0]
        self._assert_entry_and_sizing(t)
        assert t.actual_payoff == pytest.approx(0.0)
        assert t.profit == pytest.approx(-1883.62)
        assert t.profit == pytest.approx(-(t.total_cost + t.fees))
        assert t.slippage == pytest.approx(-1883.62 - 691.38)
        assert float(equity["portfolio_value"].iloc[-1]) == pytest.approx(10_000.0 - 1883.62)

    def test_premise_violation_is_excluded_and_warned(self, monkeypatch, caplog):
        # EA yes, EB no cannot happen for a cumulative-deadline pair: the
        # candidate is excluded (never traded, never paid), counted once, and
        # the equity curve stays flat.
        with caplog.at_level("WARNING"):
            trades, equity = self._run(monkeypatch, "yes", "no")
        assert trades == []
        warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        premise = [w for w in warnings if "cumulative-deadline premise" in w]
        assert len(premise) == 1
        assert premise[0].startswith("Excluded 1 time-series candidate(s)")
        assert "earlier YES, later NO" in premise[0]
        assert "snapshot markets" in premise[0]
        # Flat equity curve: nothing left and nothing came back
        assert equity["portfolio_value"].min() == pytest.approx(10_000.0)
        assert equity["portfolio_value"].max() == pytest.approx(10_000.0)

    def test_no_premise_violation_logs_nothing(self, monkeypatch, caplog):
        # Summary-warning idiom: silent at zero
        with caplog.at_level("WARNING"):
            self._run(monkeypatch, "yes", "yes")
        assert not any("cumulative-deadline premise" in r.getMessage()
                       for r in caplog.records)

    def test_wide_gap_is_capped_at_budget_fraction(self, monkeypatch):
        # Later candle 0.70 / 0.30: gap 0.40, legs 0.60, p = 0.70 — the
        # uncapped Kelly fraction is ~0.214, so BUDGET_FRACTION binds.
        trades, _ = self._run(monkeypatch, "yes", "yes", eb_yes=0.70, eb_no=0.30)
        assert len(trades) == 1
        t = trades[0]
        assert t.pair_type == "time_series"
        assert t.entry_nB == pytest.approx(0.30)
        uncapped = self._expected_kelly(self._PA, 0.30, 0.70)
        assert uncapped == pytest.approx(0.214, abs=1e-3)
        assert uncapped > BUDGET_FRACTION
        assert t.kelly_fraction == pytest.approx(BUDGET_FRACTION)
        assert t.total_cost + t.fees <= 10_000.0 * BUDGET_FRACTION + 1e-9

    def test_wide_later_book_drives_kelly_negative_and_skips(self, monkeypatch):
        # Same YES asks but the later NO ask is 0.50: legs pA+nB = 0.80 still
        # clear the 0.85 ceiling, yet the edge no longer covers the modelled
        # loss probability — Kelly < 0, so nothing is entered.
        assert self._expected_kelly(self._PA, 0.50, self._PB) < 0
        trades, equity = self._run(monkeypatch, "yes", "yes", eb_yes=0.60, eb_no=0.50)
        assert trades == []
        assert float(equity["portfolio_value"].iloc[-1]) == pytest.approx(10_000.0)

    def test_default_k_equals_explicit_config_k(self, monkeypatch):
        """The k-boundary extraction is result-identical.

        run_backtest (which passes k=None) and _simulate_at_discount called
        with k set explicitly to the config constant must produce the same
        trades and the same equity curve — proving the None sentinel resolves
        to TIME_SERIES_INTERVAL_PROB_DISCOUNT at call time, and that lifting
        the k-independent prologue (_prepare_entries) out of Pass 1 changed no
        outcome.
        """
        trades, equity = self._run(monkeypatch, "yes", "yes")
        assert len(trades) == 1  # the fixture really did enter a trade

        # _run's monkeypatches are still in force, so the prologue replays the
        # very same fixture markets and candles run_backtest just consumed.
        raw_entries = backtester._prepare_entries(
            MagicMock(), MagicMock(), date(2026, 1, 1), True, None
        )
        point = backtester._simulate_at_discount(
            raw_entries, date(2026, 1, 1), 10_000.0,
            k=TIME_SERIES_INTERVAL_PROB_DISCOUNT,
        )

        assert point.k == TIME_SERIES_INTERVAL_PROB_DISCOUNT
        assert [astuple(t) for t in point.trades] == [astuple(t) for t in trades]
        pd.testing.assert_frame_equal(point.equity_df, equity)

    def test_prepare_entries_returns_none_when_no_monday_exists(self, monkeypatch):
        # The feasibility short-circuit is now a None sentinel on the prologue
        # (distinguishing "no simulation is possible" from "nothing entered"),
        # which run_backtest turns back into the empty-result shape.
        monkeypatch.setattr(backtester, "fetch_all_settled_markets",
                            lambda *a, **k: pytest.fail("fetch must be skipped"))
        today = date.today()
        assert backtester._prepare_entries(
            MagicMock(), MagicMock(), today + timedelta(days=1), True, None
        ) is None

    def _prepared(self, monkeypatch, result_a, result_b, eb_yes=None, eb_no=None):
        # _run installs the fixture's fetch monkeypatches and leaves them in
        # force, so the prologue below replays exactly the same markets and
        # candles run_backtest just consumed (the idiom
        # test_default_k_equals_explicit_config_k already uses).
        self._run(monkeypatch, result_a, result_b, eb_yes=eb_yes, eb_no=eb_no)
        return backtester._prepare_entries(
            MagicMock(), MagicMock(), date(2026, 1, 1), True, None
        )

    def test_calibration_is_k_independent(self, monkeypatch):
        # The in-between cell: this pair is exactly what the discount models.
        raw = self._prepared(monkeypatch, "no", "yes")

        # Simulating at two very different discounts must not disturb the
        # measurement — the population comes from the k-independent prologue,
        # never from a simulation's surviving candidates.
        before = _interval_calibration(raw)
        backtester._simulate_at_discount(raw, date(2026, 1, 1), 10_000.0, k=0.40)
        backtester._simulate_at_discount(raw, date(2026, 1, 1), 10_000.0, k=1.00)
        after = _interval_calibration(raw)
        assert before == after

    def test_calibration_counts_a_pair_the_kelly_gate_rejects(self, monkeypatch):
        # k = 1.00 takes the market at face value, so Kelly is <= 0 and the
        # simulation enters nothing — yet the pair still settled in-between and
        # must stay in the denominator. Filtering the population by Kelly is
        # what would make the estimate circular.
        raw = self._prepared(monkeypatch, "no", "yes")
        at_default = backtester._simulate_at_discount(raw, date(2026, 1, 1), 10_000.0)
        at_face_value = backtester._simulate_at_discount(
            raw, date(2026, 1, 1), 10_000.0, k=1.00)
        assert len(at_default.trades) == 1
        assert at_face_value.trades == []

        calib = _interval_calibration(raw)
        assert calib.pooled.n == 1
        assert calib.pooled.realised_rate == pytest.approx(1.0)
        assert calib.pooled.mean_implied == pytest.approx(self._PB - self._PA)

    def test_calibration_buckets_the_fixture_under_its_own_tier(self, monkeypatch):
        # 13-day gap => the 8-15d band, at the short tier _find_entry filtered
        # it under. The gap rides out of _find_entry, so the report can never
        # bucket a pair under a gap it was not actually filtered by.
        raw = self._prepared(monkeypatch, "no", "no")
        calib = _interval_calibration(raw)
        assert [b.label for b in calib.buckets] == ["8-15d"]
        assert calib.buckets[0].tier == min_price_diff_for_gap(13)
        assert calib.buckets[0].n == 1
        assert calib.buckets[0].realised_rate == pytest.approx(0.0)

    def test_premise_count_is_larger_than_the_k_dependent_warning(
        self, monkeypatch, caplog,
    ):
        # The two premise-violation counts are different quantities. The wide
        # later book (nB 0.50) drives Kelly negative, so _simulate_at_discount
        # never reaches its premise check and its WARNING stays silent — while
        # the k-independent calibration still excludes and counts the pair.
        with caplog.at_level("WARNING"):
            raw = self._prepared(monkeypatch, "yes", "no", eb_no=0.50)
        assert not any("cumulative-deadline premise" in r.getMessage()
                       for r in caplog.records)

        calib = _interval_calibration(raw)
        assert calib.excluded_premise_violations == 1
        # Excluded from the denominator entirely — neither an in-between event
        # nor a valid non-event.
        assert calib.pooled.n == 0
        assert calib.pooled.empirical_k is None
        assert calib.buckets == []


def _cal_entry(gap_days, pA, pB, result_a, result_b, pair_type="time_series"):
    """Build one _prepare_entries record shaped as _interval_calibration reads it."""
    return {
        "pair_type": pair_type,
        "canon": "canon",
        "group_key": "group",
        "entry": {
            "entry_date": date(2026, 1, 5),
            "pA": pA, "pB": pB, "nA": 1.0 - pA, "nB": 1.0 - pB,
            "mA": {"ticker": "A", "result": result_a},
            "mB": {"ticker": "B", "result": result_b},
            "gap_days": gap_days,
        },
    }


class TestIntervalCalibration:
    """_interval_calibration: the k-independent empirical-discount measurement."""

    def test_returns_none_without_time_series_candidates(self):
        # Summary-line idiom: nothing to say, so the caller stays silent
        # rather than logging an all-zero table.
        assert _interval_calibration([]) is None
        assert _interval_calibration([
            _cal_entry(None, 0.60, 0.50, "yes", "no", pair_type="same_title"),
        ]) is None

    def test_non_binary_outcomes_are_ignored_entirely(self):
        # An unsettled or voided leg can be classified neither as in-between
        # nor as a valid non-event, so it is neither numerator nor denominator
        # nor a premise violation.
        assert _interval_calibration([
            _cal_entry(3, 0.10, 0.70, "", "yes"),
            _cal_entry(3, 0.10, 0.70, "no", "void"),
        ]) is None

    def test_known_arithmetic_over_one_band(self):
        # Four candidates, all with gaps inside the 0-7d band. Implied gaps
        # 0.60 / 0.40 / 0.50 / 0.50 => mean 0.50; exactly one settled
        # in-between (earlier NO, later YES) => realised rate 0.25; so the
        # empirical discount is 0.25 / 0.50 = 0.50 — the market priced twice
        # the in-between mass that materialized.
        calib = _interval_calibration([
            _cal_entry(3, 0.10, 0.70, "no", "yes"),   # in-between  (0.60)
            _cal_entry(5, 0.20, 0.60, "yes", "yes"),  # by A        (0.40)
            _cal_entry(0, 0.25, 0.75, "no", "no"),    # never by B  (0.50)
            _cal_entry(7, 0.30, 0.80, "no", "no"),    # never by B  (0.50)
        ])
        assert calib.excluded_premise_violations == 0
        assert [b.label for b in calib.buckets] == ["0-7d"]

        band = calib.buckets[0]
        assert band.n == 4
        assert band.realised_rate == pytest.approx(0.25)
        assert band.mean_implied == pytest.approx(0.50)
        assert band.empirical_k == pytest.approx(0.50)
        assert band.tier == min_price_diff_for_gap(7)

        # One band only, so pooled repeats it — with no single tier of its own
        pooled = calib.pooled
        assert pooled.label == "POOLED"
        assert (pooled.n, pooled.realised_rate) == (4, pytest.approx(0.25))
        assert pooled.empirical_k == pytest.approx(0.50)
        assert pooled.tier == 0.0

    def test_bands_are_split_at_the_config_tier_boundary(self):
        # 0-7d and 8-15d share the short tier; 16-30d takes the long one. The
        # tiers come from config.min_price_diff_for_gap, never a literal.
        calib = _interval_calibration([
            _cal_entry(7, 0.10, 0.30, "no", "no"),
            _cal_entry(8, 0.10, 0.30, "no", "no"),
            _cal_entry(15, 0.10, 0.30, "no", "no"),
            _cal_entry(16, 0.10, 0.50, "no", "no"),
            _cal_entry(MAX_DEADLINE_GAP_DAYS, 0.10, 0.50, "no", "no"),
        ])
        assert [(b.label, b.n, b.tier) for b in calib.buckets] == [
            ("0-7d", 1, min_price_diff_for_gap(7)),
            ("8-15d", 2, min_price_diff_for_gap(15)),
            ("16-30d", 2, min_price_diff_for_gap(MAX_DEADLINE_GAP_DAYS)),
        ]
        assert calib.buckets[0].tier == calib.buckets[1].tier   # same tier
        assert calib.buckets[2].tier > calib.buckets[1].tier    # long-gap tier
        assert calib.pooled.n == 5

    def test_empty_bands_are_omitted(self):
        calib = _interval_calibration([_cal_entry(20, 0.10, 0.50, "no", "yes")])
        assert [b.label for b in calib.buckets] == ["16-30d"]
        assert calib.pooled.n == 1
        assert calib.pooled.realised_rate == pytest.approx(1.0)

    def test_premise_violations_leave_the_denominator(self):
        # Earlier YES with later NO is not a cumulative-deadline pair at all:
        # counted separately, and absent from both numerator and denominator.
        calib = _interval_calibration([
            _cal_entry(3, 0.10, 0.70, "no", "yes"),
            _cal_entry(3, 0.10, 0.70, "yes", "no"),
            _cal_entry(3, 0.10, 0.70, "yes", "no"),
        ])
        assert calib.excluded_premise_violations == 2
        assert calib.pooled.n == 1
        assert calib.pooled.realised_rate == pytest.approx(1.0)

    def test_all_premise_violations_still_reports_the_count(self):
        # Nothing measurable, but the exclusion count is the whole diagnostic:
        # every time-series pair the grouping found was non-cumulative.
        calib = _interval_calibration([_cal_entry(3, 0.10, 0.70, "yes", "no")])
        assert calib.excluded_premise_violations == 1
        assert calib.buckets == []
        assert (calib.pooled.n, calib.pooled.realised_rate) == (0, 0.0)
        assert calib.pooled.mean_implied == 0.0
        assert calib.pooled.empirical_k is None

    def test_non_positive_implied_mass_yields_no_ratio(self):
        # Undefined, not zero: reporting 0.0 would read as "the market
        # overstated everything" rather than "not measurable". Unreachable
        # from a real entry (the tier requires pB - pA >= 0.15), but reporting
        # code must not divide by zero.
        calib = _interval_calibration([_cal_entry(3, 0.50, 0.50, "no", "no")])
        assert calib.pooled.n == 1
        assert calib.pooled.mean_implied == pytest.approx(0.0)
        assert calib.pooled.empirical_k is None

    def test_gap_days_none_still_counts_in_pooled(self):
        # Defensive: a time-series entry always carries a gap, but if one ever
        # arrived without it the observation must not vanish from the pooled
        # measurement just because it fits no band.
        calib = _interval_calibration([_cal_entry(None, 0.10, 0.70, "no", "yes")])
        assert calib.buckets == []
        assert calib.pooled.n == 1
        assert calib.pooled.empirical_k == pytest.approx(1.0 / 0.60)


class TestLogIntervalCalibration:
    """The report's presentation: silent when there is nothing to say."""

    @staticmethod
    def _messages(caplog):
        return [r.getMessage() for r in caplog.records]

    def test_none_logs_nothing(self, caplog):
        with caplog.at_level("INFO"):
            _log_interval_calibration(None)
        assert caplog.records == []

    def test_report_lines(self, caplog):
        calib = _interval_calibration([
            _cal_entry(3, 0.10, 0.70, "no", "yes"),
            _cal_entry(5, 0.20, 0.60, "yes", "yes"),
            _cal_entry(0, 0.25, 0.75, "no", "no"),
            _cal_entry(7, 0.30, 0.80, "no", "no"),
        ])
        with caplog.at_level("INFO"):
            _log_interval_calibration(calib)
        msgs = self._messages(caplog)

        assert msgs[0].startswith("Interval-discount calibration")
        assert "k_hat" in msgs[1] and "realised" in msgs[1] and "implied" in msgs[1]
        # One row per band, then the pooled row
        assert msgs[2].split() == ["0-7d", "0.15", "4", "0.2500", "0.5000", "0.500"]
        assert msgs[3].split() == ["POOLED", "-", "4", "0.2500", "0.5000", "0.500"]
        assert f"{TIME_SERIES_INTERVAL_PROB_DISCOUNT:.3f}" in msgs[4]
        assert "pooled empirical k_hat = 0.500" in msgs[4]
        # The standing rule: this is advice, not an edit
        assert any("config.py is never written" in m for m in msgs)
        # Summary-line idiom: silent at zero exclusions
        assert not any("premise violation" in m for m in msgs)

    def test_premise_exclusions_are_reported_when_non_zero(self, caplog):
        calib = _interval_calibration([
            _cal_entry(3, 0.10, 0.70, "no", "yes"),
            _cal_entry(3, 0.10, 0.70, "yes", "no"),
        ])
        with caplog.at_level("INFO"):
            _log_interval_calibration(calib)
        excluded = [m for m in self._messages(caplog) if "premise violation" in m]
        assert len(excluded) == 1
        assert "Excluded 1 premise violation(s)" in excluded[0]

    def test_unmeasurable_pooled_row_renders_a_dash(self, caplog):
        calib = _interval_calibration([_cal_entry(3, 0.10, 0.70, "yes", "no")])
        with caplog.at_level("INFO"):
            _log_interval_calibration(calib)
        msgs = self._messages(caplog)
        assert msgs[2].split() == ["POOLED", "-", "0", "0.0000", "0.0000", "-"]
        assert "pooled empirical k_hat = -" in msgs[3]
