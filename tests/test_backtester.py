"""Tests for backtester.py — grouping helpers, P&L math, and entry direction."""
import gc
import inspect
import logging
import random
import re
import statistics
import time
import weakref
from array import array
from collections import defaultdict
from dataclasses import astuple
from datetime import UTC, date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

# backtester.py imports no SDK module directly (historical.py reaches every
# /historical route through its own _signed_raw_get, since the pinned SDK has
# no historical_api module at all), so backtester.py is always importable
# and its pure-logic functions are unit-testable offline.
from kalshi_betting import backtester, historical, scanner
from kalshi_betting.backtester import (
    _can_ever_enter,
    _extract_pairs,
    _fetch_candles_parallel,
    _find_entry,
    _group_by_exact_title,
    _group_by_normalized_title,
    _interval_calibration,
    _log_interval_calibration,
    _log_rss,
    _pair_key,
    _parse_iso_date,
    _parse_iso_datetime,
    _settlement_receipt,
    run_backtest,
    run_backtest_sweep,
)
from kalshi_betting.config import (
    BACKTEST_DEFAULT_SPREAD_BAND,
    BUDGET_FRACTION,
    INTERVAL_DISCOUNT_SWEEP,
    MAX_DEADLINE_GAP_DAYS,
    MVE_SERIES_FAMILY_PREFIX,
    SPREAD_BAND_SWEEP_CEILINGS,
    SPREAD_BAND_SWEEP_FLOORS,
    TIME_SERIES_INTERVAL_PROB_DISCOUNT,
    fee_leg_exact,
    fee_per_pair_approx,
    min_price_diff_for_gap,
    time_series_profit_prob,
)
from kalshi_betting.scanner import CandidatePair
from kalshi_betting.strategy import compute_trade


class _WeakrefDict(dict):
    """A market dict a test can take a weak reference to.

    Plain dicts do not support weak references, so a residency test has no way
    to observe when the code under test has let go of one. This subclass adds
    the slot and changes nothing else — the backtester only ever reads these
    through .get()/[] like any other cached market record.
    """

    __slots__ = ("__weakref__",)


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

    def test_distinct_outcome_labels_separate_groups(self):
        # DR-01: the subtitle is part of the time-series key too, not just the
        # same-title one. Two strikes of a daily family share a title and
        # differ only here.
        mA = _md("A1", "EVT-SEP14", title="Solana price on Sep 14, 2026?",
                 subtitle="$180 or above", event_title="Solana price on Sep 14, 2026?")
        mB = _md("B1", "EVT-SEP18", title="Solana price on Sep 18, 2026?",
                 subtitle="$190 or above", event_title="Solana price on Sep 18, 2026?")
        assert _group_by_normalized_title([mA, mB]) == {}

    def test_identical_outcome_labels_at_two_deadlines_group(self):
        mA = _md("A1", "EVT-SEP14", title="Solana price on Sep 14, 2026?",
                 subtitle="$180 or above", event_title="Solana price on Sep 14, 2026?")
        mB = _md("B1", "EVT-SEP18", title="Solana price on Sep 18, 2026?",
                 subtitle="$180 or above", event_title="Solana price on Sep 18, 2026?")
        groups = _group_by_normalized_title([mA, mB])
        assert len(groups) == 1
        assert len(next(iter(groups.values()))) == 2

    def test_old_cache_none_subtitle_still_groups_by_title_alone(self):
        # Day slices written before the 2026-08 subtitle fix carry
        # subtitle=None; time_series_group_key reads a non-str as absent, so
        # such records keep the pre-DR-01 title-only grouping rather than
        # raising. Backtest fidelity only — no live-money path reads them.
        mA = _md("A1", "EVT-MAR", title="BTC over $80k by March 2026",
                 event_title="BTC price tracker")
        mB = _md("B1", "EVT-JUN", title="BTC over $80k by June 2026",
                 event_title="BTC price tracker")
        mA["subtitle"] = None
        mB["subtitle"] = None
        groups = _group_by_normalized_title([mA, mB])
        assert len(groups) == 1
        assert len(next(iter(groups.values()))) == 2


class TestTimeSeriesOutcomeDiscriminator:
    """DR-01 mirror: the backtester keys time-series groups through the same
    scanner.time_series_group_key the live scanner uses.

    Before this, both sides keyed on the date-stripped title alone, so the
    backtester reproduced the live defect exactly and could never have
    detected it: a whole daily strike family was one group, and _extract_pairs
    emitted every early-strike x late-strike combination as a candidate.
    """

    _STRIKES = ("$180 or above", "$190 or above", "$200 or above", "$210 or above")
    _EVENTS = (("KXSOLD-26SEP14", "14", "2026-09-14"), ("KXSOLD-26SEP18", "18", "2026-09-18"))

    def _family(self, *, strike_in_subtitle: bool = True,
                snapshot_wording: bool = False) -> list[dict]:
        """Two deadline events of one daily family, four strikes each.

        The titles are CUMULATIVE ("price by <date>"), mirroring the live
        fixture: DR-01 is about the outcome label in the grouping key and must
        keep being tested on a family the deadline rule admits.
        `snapshot_wording=True` returns the original "price ON <date>" shape —
        a real KXSOLD family — which that rule now refuses outright.
        """
        preposition = "on" if snapshot_wording else "by"
        markets = []
        for i, strike in enumerate(self._STRIKES):
            for event_ticker, day, close_day in self._EVENTS:
                title = f"Solana price {preposition} Sep {day}, 2026?"
                markets.append({
                    "ticker": f"{event_ticker}-T{i}",
                    "event_ticker": event_ticker,
                    "event_title": title,
                    "title": title,
                    "subtitle": strike if strike_in_subtitle else "",
                    "close_time": f"{close_day}T21:00:00Z",
                })
        return markets

    def test_a_snapshot_family_yields_no_candidate(self):
        # Mirror of the live TestOutcomeDiscriminator::
        # test_a_snapshot_family_forms_no_pairs_at_all. The eight markets still
        # GROUP — the key is untouched — but no candidate survives, because
        # SOL >= $180 on Sep 14 does not imply SOL >= $180 on Sep 18.
        groups = _group_by_normalized_title(self._family(snapshot_wording=True))
        assert len(groups) == len(self._STRIKES)
        assert _extract_pairs(groups) == []

    def test_each_strike_is_its_own_group(self):
        groups = _group_by_normalized_title(self._family())
        assert len(groups) == len(self._STRIKES)
        for members in groups.values():
            assert len({m["subtitle"] for m in members}) == 1
            assert len(members) == 2

    def test_extract_pairs_emits_no_cross_strike_candidate(self):
        pairs = _extract_pairs(_group_by_normalized_title(self._family()))
        assert len(pairs) == len(self._STRIKES)
        for mA, mB, _canon, _group_key in pairs:
            assert mA["subtitle"] == mB["subtitle"]

    def test_without_the_discriminator_every_combination_is_a_candidate(self):
        # The defect, reproduced: one group of eight, and _extract_pairs (which
        # has no best-pair rule — that is run_backtest's job) materializes all
        # 4 x 4 early/late combinations, 12 of which are cross-strike.
        groups = _group_by_normalized_title(self._family(strike_in_subtitle=False))
        assert len(groups) == 1
        pairs = _extract_pairs(groups)
        assert len(pairs) == 16
        cross = [(a, b) for a, b, _, _ in pairs if a["ticker"][-2:] != b["ticker"][-2:]]
        assert len(cross) == 12


class TestExtractPairsCanonHandling:
    def test_three_tuple_key_uses_title_not_event(self):
        # Build a same-title group with a 3-tuple key and verify canon is the
        # market title (key[1]), not the event title (key[0]).
        #
        # RE-PINNED (DR-02/DR-54): the event tickers used to be EVT-A/EVT-B,
        # which share the series prefix "EVT" — _extract_pairs now reads that
        # as two instances of one recurring fixture and forms no pair, so the
        # assertion below would have failed for a reason that has nothing to do
        # with canon selection. Two DIFFERENT series restore the shape the
        # same-title strategy was built for.
        #
        # RE-PINNED (DR-74): the same-title close gate refuses a pair whose
        # close_time is missing (fail closed), so both records now carry one
        # shared close — added HERE, not to the shared _md, whose other
        # callers group or key records and never reach the gate.
        mA = _md("A1", "EVA-1", title="Republicans win majority",
                 event_title="2026 Senate Control")
        mB = _md("B1", "EVB-1", title="Republicans win majority",
                 event_title="2026 Senate Control")
        for m in (mA, mB):
            m["close_time"] = "2026-11-04T05:00:00Z"
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
        #
        # RE-PINNED (DR-74): each group's two records share a close_time, which
        # the same-title close gate needs (a missing one fails closed). Added
        # locally rather than to the shared _md.
        mA1 = _md("E1-A", "EVT1A", title="Trump", event_title="Election Winner")
        mA2 = _md("E1-B", "EVT1B", title="Trump", event_title="Election Winner")
        mB1 = _md("E2-A", "EVT2A", title="Trump", event_title="Person of the Year")
        mB2 = _md("E2-B", "EVT2B", title="Trump", event_title="Person of the Year")
        for m in (mA1, mA2):
            m["close_time"] = "2026-11-04T05:00:00Z"
        for m in (mB1, mB2):
            m["close_time"] = "2026-12-10T15:00:00Z"
        groups = _group_by_exact_title([mA1, mA2, mB1, mB2])
        pairs = _extract_pairs(groups)
        assert len(pairs) == 2
        canons = {canon for _, _, canon, _ in pairs}
        assert canons == {"Trump"}
        group_keys = {group_key for _, _, _, group_key in pairs}
        assert len(group_keys) == 2, "distinct events must yield distinct group_keys"


class TestOneEventSeriesIsTwoFixturesBacktest:
    """DR-02 / DR-54 mirror: _extract_pairs refuses two events of ONE series.

    The rule is the live scanner's, applied to cached records through
    backtester._same_series_dicts / _identical_wording_dicts, which resolve the
    series identity through the same scanner.event_series the live path uses —
    including its collapse of every combo (KXMVE*) prefix onto one family. Both
    branches of _extract_pairs carry it: the 3-tuple (same-title) branch on the
    series alone, because the group key already guarantees identical wording,
    and the string (time-series) branch on the conjunct, because there the
    wording is only date-stripped-equal.
    """

    @staticmethod
    def _rec(ticker, event_ticker, title, close_time, *, subtitle="Yes", event_title=""):
        return {
            "ticker": ticker,
            "event_ticker": event_ticker,
            "event_title": event_title,
            "title": title,
            "subtitle": subtitle,
            "close_time": close_time,
        }

    @classmethod
    def _npb(cls):
        # The real sandbox pair: one fixture listed on two game days.
        title = "Fukuoka Hawks vs Orix Buffaloes: First Inning Run?"
        event_title = "Fukuoka Hawks vs Orix Buffaloes: First Inning Run"
        return [
            cls._rec("KXNPBRFI-26SEP160500FUKORI-Y", "KXNPBRFI-26SEP160500FUKORI",
                     title, "2026-09-18T09:00:00Z", event_title=event_title),
            cls._rec("KXNPBRFI-26SEP150500FUKORI-Y", "KXNPBRFI-26SEP150500FUKORI",
                     title, "2026-09-17T09:00:00Z", event_title=event_title),
        ]

    def test_same_title_branch_rejects_two_game_days_of_one_fixture(self):
        assert _extract_pairs(_group_by_exact_title(self._npb())) == []

    def test_time_series_branch_rejects_the_same_two_records(self):
        # Grouping still puts them together (their wording is identical, so it
        # is trivially date-stripped-equal) — the conjunct is what drops them.
        groups = _group_by_normalized_title(self._npb())
        assert len(groups) == 1
        assert _extract_pairs(groups) == []

    def test_two_combo_events_are_rejected(self):
        # A combo ticket's wording names its legs but never its date, so one
        # wording recurs across fixture instances and two tickets with
        # identical leg wording are two DIFFERENT tickets. Since DR-55 these
        # two do not resolve to a literal prefix at all: both event tickers
        # start with config.MVE_SERIES_FAMILY_PREFIX, so scanner.event_series
        # answers "KXMVE" for each and _same_series_dicts sees one family.
        # (Before DR-55 the same verdict came from the literal
        # "KXMVECROSSCATEGORY" prefix — which is why the cross-prefix case
        # below needed its own test.)
        recs = [
            self._rec("KXMVECROSSCATEGORY-SHARD1-S6471E4699E9-Y",
                      "KXMVECROSSCATEGORY-SHARD1-S6471E4699E9",
                      "Parlay", "2026-09-15T20:00:00Z",
                      subtitle="All legs hit", event_title="Cross-category combo"),
            self._rec("KXMVECROSSCATEGORY-SHARD1-S93FFD638F77-Y",
                      "KXMVECROSSCATEGORY-SHARD1-S93FFD638F77",
                      "Parlay", "2026-09-15T20:00:00Z",
                      subtitle="All legs hit", event_title="Cross-category combo"),
        ]
        assert _extract_pairs(_group_by_exact_title(recs)) == []
        assert _extract_pairs(_group_by_normalized_title(recs)) == []

    def test_two_combo_events_of_two_kxmve_series_are_rejected(self):
        # DR-55 mirror: the same wording listed under two DIFFERENT KXMVE*
        # series. Kalshi lists combos under several prefixes, so the literal
        # prefix read these as two series and both _extract_pairs branches
        # formed the pair. scanner.event_series collapses the whole KXMVE
        # family onto one series, and _same_series_dicts inherits that because
        # it resolves through the same helper the live path uses.
        recs = [
            self._rec("KXMVECROSSCATEGORY-SHARD1-S6471E4699E9-Y",
                      "KXMVECROSSCATEGORY-SHARD1-S6471E4699E9",
                      "Parlay", "2026-09-15T20:00:00Z",
                      subtitle="All legs hit", event_title="Cross-category combo"),
            self._rec("KXMVESPORTSMULTIGAMEEXTENDED-SHARD1-S93FFD638F77-Y",
                      "KXMVESPORTSMULTIGAMEEXTENDED-SHARD1-S93FFD638F77",
                      "Parlay", "2026-09-17T20:00:00Z",
                      subtitle="All legs hit", event_title="Cross-category combo"),
        ]
        # Guard that the fixture is the shape the rule must catch: two literal
        # prefixes, identical wording, one collapsed series.
        assert (recs[0]["event_ticker"].split("-")[0]
                != recs[1]["event_ticker"].split("-")[0])
        assert backtester._identical_wording_dicts(recs[0], recs[1]) is True
        assert backtester._same_series_dicts(recs[0], recs[1]) is True
        assert _extract_pairs(_group_by_exact_title(recs)) == []
        assert _extract_pairs(_group_by_normalized_title(recs)) == []

    def test_a_non_mve_series_pair_is_untouched_by_the_family_collapse(self):
        # GUARD on the collapse's blast radius: only KXMVE* collapses, so two
        # ordinary series still read as different and still pair.
        recs = [
            self._rec("KXFEDDEC-26-T25", "KXFEDDEC-26",
                      "Fed cuts rates in December?", "2026-12-10T19:00:00Z",
                      event_title="Fed December decision"),
            self._rec("KXMVPAWARD-26-T25", "KXMVPAWARD-26",
                      "Fed cuts rates in December?", "2026-12-10T19:00:00Z",
                      event_title="Fed December decision"),
        ]
        assert backtester._same_series_dicts(recs[0], recs[1]) is False
        assert len(_extract_pairs(_group_by_exact_title(recs))) == 1

    def test_two_different_series_asking_one_question_still_pair(self):
        recs = [
            self._rec("KXFEDDEC-26-T25", "KXFEDDEC-26",
                      "Fed cuts rates in December?", "2026-12-10T19:00:00Z",
                      event_title="Fed December decision"),
            self._rec("FEDCUTDEC-26-T25", "FEDCUTDEC-26",
                      "Fed cuts rates in December?", "2026-12-10T19:00:00Z",
                      event_title="Fed December decision"),
        ]
        pairs = _extract_pairs(_group_by_exact_title(recs))
        assert len(pairs) == 1
        assert {pairs[0][0]["ticker"], pairs[0][1]["ticker"]} == {
            "KXFEDDEC-26-T25", "FEDCUTDEC-26-T25"
        }

    def test_a_dated_pair_of_one_series_is_untouched(self):
        # The deadline lives IN the wording, so _identical_wording_dicts is
        # False and the time-series conjunct never fires. ELIGIBILITY only —
        # a "price on <date>" family is a snapshot family, not a cumulative
        # one, and the premise-violation counter is what judges that (see the
        # live mirror, TestOneEventSeriesIsTwoFixtures::
        # test_a_dated_pair_of_one_series_is_untouched).
        recs = self._sold_family("by")
        pairs = _extract_pairs(_group_by_normalized_title(recs))
        assert len(pairs) == 1

    def _sold_family(self, preposition: str) -> list[dict]:
        """One KXSOLD strike listed by two deadline events of ONE series."""
        return [
            self._rec("KXSOLD-26SEP14-T180", "KXSOLD-26SEP14",
                      f"Solana price {preposition} Sep 14, 2026?", "2026-09-14T21:00:00Z",
                      subtitle="$180 or above",
                      event_title=f"Solana price {preposition} Sep 14, 2026?"),
            self._rec("KXSOLD-26SEP18-T180", "KXSOLD-26SEP18",
                      f"Solana price {preposition} Sep 18, 2026?", "2026-09-18T21:00:00Z",
                      subtitle="$180 or above",
                      event_title=f"Solana price {preposition} Sep 18, 2026?"),
        ]

    def test_the_snapshot_spelling_of_that_same_family_is_now_refused(self):
        # Mirror of the live TestOneEventSeriesIsTwoFixtures::
        # test_the_snapshot_spelling_of_that_same_family_is_now_refused. The
        # one-series rule still does not fire (the legs are worded
        # differently), so this is the cumulative-deadline rule's verdict
        # alone, on the very fixture that used to document the gap.
        recs = self._sold_family("on")
        assert backtester._identical_wording_dicts(recs[0], recs[1]) is False
        assert _extract_pairs(_group_by_normalized_title(recs)) == []

    def test_an_unreadable_event_ticker_fails_closed(self):
        # A record whose fixture identity cannot be read must NOT be replayed
        # on the 95% co-resolution prior — the same direction the live helper
        # fails in.
        recs = [
            self._rec("A1", "", "Fed cuts rates in December?", "2026-12-10T19:00:00Z",
                      event_title="Fed December decision"),
            self._rec("B1", "FEDCUTDEC-26", "Fed cuts rates in December?",
                      "2026-12-10T19:00:00Z", event_title="Fed December decision"),
        ]
        assert _extract_pairs(_group_by_exact_title(recs)) == []
        assert backtester._same_series_dicts(recs[0], recs[1]) is True

    def test_a_missing_event_ticker_key_also_fails_closed(self):
        # Old cache records predate nothing here, but .get() must not raise and
        # an absent key must read as unknown, not as a distinct series.
        assert backtester._same_series_dicts({}, {"event_ticker": "KXSOLD-26SEP14"}) is True

    def test_identical_wording_mirror_reads_missing_keys_as_empty(self):
        assert backtester._identical_wording_dicts({}, {}) is True
        assert backtester._identical_wording_dicts(
            {"title": "Q", "subtitle": "Yes", "event_title": "E"},
            {"title": "Q", "subtitle": "Yes", "event_title": "E"},
        ) is True
        assert backtester._identical_wording_dicts(
            {"title": "Q by March"}, {"title": "Q by June"},
        ) is False


class TestDeadlineGuardFinders:
    """Backtester mirror of test_scanner.py::TestDeadlineGuardFinders — the
    same fixtures, through the dict-based grouping/extraction path, so a
    fail-open guard that is pinned live but unpinned in the backtester cannot
    silently diverge: cumulative_deadline_pair is one shared helper, but each
    path reaches it through its own field extraction.

    Every test that asserts [] also asserts, inside the test, that its two
    legs land in one _group_by_normalized_title group (so a normalize_title
    drift cannot make the test pass vacuously by splitting the legs apart
    before the guard under test is ever reached) and carries an in-test
    positive control.
    """

    @staticmethod
    def _rec(ticker, event_ticker, title, close_time, *, event_title=""):
        rec = _md(ticker, event_ticker, title=title, event_title=event_title)
        rec["close_time"] = close_time
        return rec

    @staticmethod
    def _one_group(markets):
        groups = _group_by_normalized_title(markets)
        assert len(groups) == 1
        [members] = groups.values()
        assert len(members) == len(markets)
        return groups

    def test_level_at_instant_legs_are_refused(self):
        # control — kills M03full.
        t1 = "Will BTC be above $100k at the close on Sep 30, 2026, before Oct 1, 2026?"
        t2 = "Will BTC be above $100k at the close on Oct 9, 2026, before Oct 10, 2026?"
        mA = self._rec("PA-1", "EVA-1", t1, "2026-09-30T00:00:00Z")
        mB = self._rec("PB-1", "EVB-1", t2, "2026-10-09T00:00:00Z")
        groups = self._one_group([mA, mB])
        assert backtester._deadline_profile_dict(mA) == (
            scanner.DEADLINE_SNAPSHOT, ("before oct 1, 2026",),
        )
        assert backtester._deadline_profile_dict(mB) == (
            scanner.DEADLINE_SNAPSHOT, ("before oct 10, 2026",),
        )
        assert _extract_pairs(groups) == []

        # Control: same fixture with "at the close on <date>," removed.
        cA = self._rec(
            "PA-1", "EVA-1", "Will BTC be above $100k before Oct 1, 2026?",
            "2026-09-30T00:00:00Z",
        )
        cB = self._rec(
            "PB-1", "EVB-1", "Will BTC be above $100k before Oct 10, 2026?",
            "2026-10-09T00:00:00Z",
        )
        assert len(_extract_pairs(_group_by_normalized_title([cA, cB]))) == 1

    def test_dated_leg_never_pairs_with_spanless_cumulative_leg(self):
        # control — kills M02 at the finder level.
        evt = "Will X happen at any time?"
        mA = self._rec(
            "PA-1", "EVA-1", "Will X happen by March 1?", "2026-03-01T00:00:00Z",
            event_title=evt,
        )
        mB = self._rec(
            "PB-1", "EVB-1", "Will X happen Mar 9?", "2026-03-09T00:00:00Z",
            event_title=evt,
        )
        groups = self._one_group([mA, mB])
        assert backtester._deadline_profile_dict(mA) == (
            scanner.DEADLINE_CUMULATIVE, ("by march 1",),
        )
        assert backtester._deadline_profile_dict(mB) == (scanner.DEADLINE_CUMULATIVE, ())
        assert _extract_pairs(groups) == []

        # Control: B names its own deadline.
        mB2 = self._rec(
            "PB-1", "EVB-1", "Will X happen by March 9?", "2026-03-09T00:00:00Z",
            event_title=evt,
        )
        assert len(_extract_pairs(_group_by_normalized_title([mA, mB2]))) == 1

    def test_dated_leg_never_pairs_with_unknown_leg(self):
        # control — kills M02+M03a together (see test_scanner.py's mirror for
        # why M03a alone cannot be isolated by this fixture).
        mA = self._rec("PA-1", "EVA-1", "Will X happen by March 1?", "2026-03-01T00:00:00Z")
        mB = self._rec("PB-1", "EVB-1", "Will X happen Mar 9?", "2026-03-09T00:00:00Z")
        groups = self._one_group([mA, mB])
        assert backtester._deadline_profile_dict(mA) == (
            scanner.DEADLINE_CUMULATIVE, ("by march 1",),
        )
        assert backtester._deadline_profile_dict(mB) == (scanner.DEADLINE_UNKNOWN, ())
        assert _extract_pairs(groups) == []

        # Control: B named its own deadline instead.
        mB2 = self._rec("PB-1", "EVB-1", "Will X happen by March 9?", "2026-03-09T00:00:00Z")
        assert len(_extract_pairs(_group_by_normalized_title([mA, mB2]))) == 1


class TestDateTokenBoundaries:
    """Backtester mirror of test_scanner.py::TestDateTokenBoundaries'
    finder-level row (DR-68). The token rows themselves are pinned once, on
    the shared scanner.deadline_profile; this proves the dict-based path
    reaches the tightened weekday token too.
    """

    def test_weekday_dated_legs_seven_days_apart_pair(self):
        # regression — before DR-68 both legs' spans truncated to "by friday", so the
        # pair read as one deadline stated twice and was refused.
        mA = TestDeadlineGuardFinders._rec(
            "PA-1", "EVA-1", "Will X happen by Friday, Sep 19, 2026?",
            "2026-09-19T00:00:00Z",
        )
        mB = TestDeadlineGuardFinders._rec(
            "PB-1", "EVB-1", "Will X happen by Friday, Sep 26, 2026?",
            "2026-09-26T00:00:00Z",
        )
        groups = TestDeadlineGuardFinders._one_group([mA, mB])
        assert backtester._deadline_profile_dict(mA) == (
            scanner.DEADLINE_CUMULATIVE, ("by friday, sep 19, 2026",),
        )
        assert backtester._deadline_profile_dict(mB) == (
            scanner.DEADLINE_CUMULATIVE, ("by friday, sep 26, 2026",),
        )
        [(a, b, _canon, key)] = _extract_pairs(groups)
        assert isinstance(key, str)  # a string key is the time-series branch
        assert {a["ticker"], b["ticker"]} == {"PA-1", "PB-1"}


class TestPhrasingSkipCounts:
    """Backtester mirror of test_scanner.py::TestPhrasingSkipCounts (DR-72):
    the single folded "not a cumulative-deadline pair" line is split into
    three honest, separately-reported reasons here too, over
    backtester._deadline_profile_dict rather than the live scanner's
    attribute-based profile. Each reason gets its own two-member group on a
    distinct series pair (EVA-x / EVB-x) so the DR-02 one-series conjunct —
    which runs before the deadline check on this branch too — never fires
    ahead of the check under test.
    """

    @staticmethod
    def _stub_profiles(overrides: dict):
        def fake(m):
            return overrides[m["ticker"]]
        return fake

    @staticmethod
    def _refusal_lines(caplog):
        return [
            r.getMessage() for r in caplog.records
            if r.getMessage().startswith("Time-series candidates refused because")
        ]

    def test_each_reason_is_reported_once(self, monkeypatch, caplog):
        # regression — fails on revert to the single folded counter, which
        # reported one line instead of three, so none of the three
        # exact-prefix assertions below would ever have matched.
        #
        # Each reason gets a DIFFERENT candidate count (1, 2, 3) — not just a
        # different fixture — so a mutant that swaps which counter a reason
        # increments cannot pass by coincidence: with every reason at count
        # 1, such a swap still emits three "...: 1" lines and this test could
        # not tell (mirrors test_scanner.py's TestPhrasingSkipCounts fix).
        rec = TestDeadlineGuardFinders._rec
        snap_a = rec("SNAP-A", "EVA-1", "Will Group Snap happen?", "2026-06-01T00:00:00Z")
        snap_b = rec("SNAP-B", "EVB-1", "Will Group Snap happen?", "2026-06-05T00:00:00Z")
        nod_a = rec("NOD-A", "EVA-2", "Will Group Nodate happen?", "2026-06-01T00:00:00Z")
        nod_b = rec("NOD-B", "EVB-2", "Will Group Nodate happen?", "2026-06-05T00:00:00Z")
        nod2_a = rec("NOD2-A", "EVA-4", "Will Group Nodate2 happen?", "2026-06-01T00:00:00Z")
        nod2_b = rec("NOD2-B", "EVB-4", "Will Group Nodate2 happen?", "2026-06-05T00:00:00Z")
        same_a = rec("SAME-A", "EVA-3", "Will Group Same happen?", "2026-06-01T00:00:00Z")
        same_b = rec("SAME-B", "EVB-3", "Will Group Same happen?", "2026-06-05T00:00:00Z")
        same2_a = rec("SAME2-A", "EVA-5", "Will Group Same2 happen?", "2026-06-01T00:00:00Z")
        same2_b = rec("SAME2-B", "EVB-5", "Will Group Same2 happen?", "2026-06-05T00:00:00Z")
        same3_a = rec("SAME3-A", "EVA-6", "Will Group Same3 happen?", "2026-06-01T00:00:00Z")
        same3_b = rec("SAME3-B", "EVB-6", "Will Group Same3 happen?", "2026-06-05T00:00:00Z")
        members = [
            snap_a, snap_b,
            nod_a, nod_b, nod2_a, nod2_b,
            same_a, same_b, same2_a, same2_b, same3_a, same3_b,
        ]

        overrides = {
            "SNAP-A": (scanner.DEADLINE_SNAPSHOT, ("on june 1, 2026",)),
            "SNAP-B": (scanner.DEADLINE_CUMULATIVE, ("by june 10, 2026",)),
            "NOD-A": (scanner.DEADLINE_CUMULATIVE, ("by june 1, 2026",)),
            "NOD-B": (scanner.DEADLINE_UNKNOWN, ()),
            "NOD2-A": (scanner.DEADLINE_CUMULATIVE, ("by june 1, 2026",)),
            "NOD2-B": (scanner.DEADLINE_UNKNOWN, ()),
            "SAME-A": (scanner.DEADLINE_CUMULATIVE, ("by june 1, 2026",)),
            "SAME-B": (scanner.DEADLINE_CUMULATIVE, ("by june 1, 2026",)),
            "SAME2-A": (scanner.DEADLINE_CUMULATIVE, ("by june 1, 2026",)),
            "SAME2-B": (scanner.DEADLINE_CUMULATIVE, ("by june 1, 2026",)),
            "SAME3-A": (scanner.DEADLINE_CUMULATIVE, ("by june 1, 2026",)),
            "SAME3-B": (scanner.DEADLINE_CUMULATIVE, ("by june 1, 2026",)),
        }
        monkeypatch.setattr(
            backtester, "_deadline_profile_dict", self._stub_profiles(overrides)
        )

        groups = _group_by_normalized_title(members)
        assert len(groups) == 6  # six distinct titles -> six groups
        with caplog.at_level(logging.INFO):
            pairs = _extract_pairs(groups)
        assert pairs == []

        lines = self._refusal_lines(caplog)
        assert len(lines) == 3
        assert any(
            line.startswith(
                "Time-series candidates refused because a leg's deciding "
                "field is snapshot wording"
            ) and line.endswith(": 1")
            for line in lines
        )
        assert any(
            line.startswith(
                "Time-series candidates refused because a leg's deciding "
                "field carries no recognised deadline wording or no "
                "comparable date"
            ) and line.endswith(": 2")
            for line in lines
        )
        assert any(
            line.startswith(
                "Time-series candidates refused because the two deciding "
                "fields state the same deadline, or truncate to one"
            ) and line.endswith(": 3")
            for line in lines
        )
        # This function has no gap-CAP line of its own: its close-date window
        # is a performance bound, not a rule. (Since M10 the pairs beyond
        # that window are counted as never visited — worded apart from the
        # live finder's gap-cap line, which counts pairs already worded as two
        # cumulative deadlines.) Recorded here so the two stay distinct.
        assert not any("gap cap" in m for m in (r.getMessage() for r in caplog.records))

    def test_silent_at_zero(self, caplog):
        # control — kills a mutant that logs a refusal line unconditionally
        # (dropping the `if snapshot_skips:` / etc. guards). A group whose
        # one candidate pair is genuinely eligible must produce none of the
        # three lines.
        rec = TestDeadlineGuardFinders._rec
        a = rec("OK-A", "EVA-1", "Will Group OK happen by June 1, 2026?", "2026-06-01T00:00:00Z")
        b = rec("OK-B", "EVB-1", "Will Group OK happen by June 10, 2026?", "2026-06-10T00:00:00Z")
        groups = TestDeadlineGuardFinders._one_group([a, b])
        with caplog.at_level(logging.INFO):
            pairs = _extract_pairs(groups)
        assert len(pairs) == 1
        assert self._refusal_lines(caplog) == []


# M10: the lines _extract_pairs gained, by prefix — four refusals, plus the two
# things the time-series sweep sets aside before visiting anything. The DR-72
# and DR-73 lines are read by the same prefix table so a test can check that
# EVERY pair of a group's members is accounted for.
_TS_UNDATED = "Time-series group members without a readable close_time"
_TS_BEYOND = "Time-series candidate pairs the sweep never visits"
_TS_SAME_EVENT = ("Time-series candidates skipped because both markets carry "
                  "the same event ticker")
_TS_SERIES = "Time-series candidates skipped as two instances of one event series"
_ST_SAME_EVENT = ("Same-title candidates skipped because both markets carry the "
                  "same event ticker")
_ST_SERIES = "Same-title candidates skipped as two instances of one event series"
# DR-74: the same-title close gate's two lines — the close-gap refusal
# (verbatim with find_same_title_pairs') and the backtest-only unreadable-close
# one. Distinct prefixes ("candidates" vs "candidate pairs"), so neither can be
# read as the other.
_ST_CLOSE_GAP = "Same-title candidates refused because the two markets close more than"
_ST_UNDATED = "Same-title candidate pairs refused because a market's close_time"
_TS_WORDING = "Time-series candidates refused because"
_LADDER_REFUSED = "Same-event ladder candidates refused because"
_LADDER_GAP_CAP = "Same-event ladder candidates worded as two different cumulative"
_LADDER_FORMED = "Same-event ladder candidates among the time-series candidates"
# _prepare_candidates' always-logged grouping line (M10), for the empty-grouping
# case _extract_pairs cannot see.
_GROUPS = "Groups of two or more markets:"


def _logged_counts(caplog) -> dict[str, int]:
    """Sum the trailing ': N' of every _extract_pairs count line, by prefix.

    Every count is of candidate PAIRS except _TS_UNDATED's, which counts
    group MEMBERS — read it on its own, never in a sum with the others.
    """
    out: dict[str, int] = defaultdict(int)
    prefixes = (_TS_UNDATED, _TS_BEYOND, _TS_SAME_EVENT, _TS_SERIES, _ST_SAME_EVENT,
                _ST_SERIES, _ST_CLOSE_GAP, _ST_UNDATED, _TS_WORDING, _LADDER_REFUSED,
                _LADDER_GAP_CAP, _LADDER_FORMED)
    for message in caplog.messages:
        for prefix in prefixes:
            if message.startswith(prefix):
                out[prefix] += int(message.rsplit(": ", 1)[1])
    return out


class TestOneSeriesAndSameEventCounts:
    """M10: every pair of a group's members is returned or counted.

    The one-series rule (DR-02, DR-54, both branches) and the same-event skip
    were bare `continue`s here, so the 2026-09-24 7-day run logged "Potential
    pairs: 0 time-series, 0 same-title" over 184,178 groupable markets with
    no cause for the same-title zero. Each now has a silent-at-zero INFO line,
    and so do the two things the time-series sweep sets aside before visiting
    anything (members with no readable close_time, pairs beyond its
    close-date window); an EMPTY grouping, which reaches _extract_pairs as {},
    is reported by _prepare_candidates' always-logged group-count line. DR-74
    added two more same-title lines, counted after the series test: a pair
    whose two markets close more than SAME_TITLE_MAX_CLOSE_GAP_SECONDS apart,
    and (backtest-only) a pair whose close_time cannot be read or compared.
    Counting must move no control flow: the pair lists are pinned by every
    other test in this module, and the partition test below proves that the
    counts and the returned pairs together account for each pair exactly
    once.
    """

    @staticmethod
    def _rec(ticker, event_ticker, wording, close):
        # The shape of a real combo record in a day slice: title == subtitle
        # (the leg wording), event_title blank (it is patched in at assembly
        # for well under 1% of records).
        return {"ticker": ticker, "event_ticker": event_ticker, "event_title": "",
                "title": wording, "subtitle": wording, "close_time": close}

    @classmethod
    def _combo_heavy(cls):
        """Eight records in three groups (seven candidates), every candidate
        refused by one of M10's two rules.

        W1: two tickets on ONE combo event (1 same-event candidate) and one
        under a different KXMVE prefix (2 one-series candidates, DR-55).
        W2: two tickets under two KXMVE prefixes (1 one-series candidate).
        MLB: three game days of one non-combo series (3 one-series candidates).
        Same-event 1, one-series 6: different counts, so swapping the two
        counters cannot pass.
        """
        w1 = "yes Over 5.5 runs scored,no Over 2.5 runs scored"
        w2 = "yes Lakers win,yes Over 3.5 runs scored"
        mlb = "Over 5.5 runs scored"
        return [
            cls._rec("KXMVECROSSCATEGORY-S1-A", "KXMVECROSSCATEGORY-S1", w1,
                     "2026-09-18T23:59:00Z"),
            cls._rec("KXMVECROSSCATEGORY-S1-B", "KXMVECROSSCATEGORY-S1", w1,
                     "2026-09-18T23:59:00Z"),
            cls._rec("KXMVECROSSCATEGORY0-S2-A", "KXMVECROSSCATEGORY0-S2", w1,
                     "2026-09-19T23:59:00Z"),
            cls._rec("KXMVESPORTSMULTIGAMEEXTENDED-S3-A",
                     "KXMVESPORTSMULTIGAMEEXTENDED-S3", w2, "2026-09-20T23:59:00Z"),
            cls._rec("KXMVECROSSCATEGORY-S4-A", "KXMVECROSSCATEGORY-S4", w2,
                     "2026-09-21T23:59:00Z"),
            cls._rec("KXMLBTOTAL-26SEP181840STLPIT-5", "KXMLBTOTAL-26SEP181840STLPIT",
                     mlb, "2026-09-18T23:57:58Z"),
            cls._rec("KXMLBTOTAL-26SEP191840STLPIT-5", "KXMLBTOTAL-26SEP191840STLPIT",
                     mlb, "2026-09-19T23:57:58Z"),
            cls._rec("KXMLBTOTAL-26SEP201840STLPIT-5", "KXMLBTOTAL-26SEP201840STLPIT",
                     mlb, "2026-09-20T23:57:58Z"),
        ]

    @staticmethod
    def _group_sizes(groups):
        return sorted(len(members) for members in groups.values())

    @pytest.mark.parametrize("grouping,same_event,series", [
        (_group_by_exact_title, _ST_SAME_EVENT, _ST_SERIES),
        (_group_by_normalized_title, _TS_SAME_EVENT, _TS_SERIES),
    ])
    def test_a_combo_heavy_zero_is_fully_explained(
        self, caplog, grouping, same_event, series,
    ):
        # regression — fails on the pre-M10 code, which returned [] here and
        # logged none of these lines, so the zero had no cause in the log.
        groups = grouping(self._combo_heavy())
        # Not vacuous: three groups of 3, 2 and 3, so 3 + 1 + 3 = 7 candidates.
        assert self._group_sizes(groups) == [2, 3, 3]
        with caplog.at_level(logging.INFO):
            assert _extract_pairs(groups, same_event_ladders=False) == []
        counts = _logged_counts(caplog)
        assert counts[same_event] == 1
        assert counts[series] == 6
        # Every one of the seven candidates is on exactly one of the two lines.
        assert sum(counts.values()) == 7

    def test_with_ladders_on_the_sweep_leaves_same_event_candidates_to_the_sub_pass(
        self, caplog,
    ):
        # control — kills counting the sweep's same-event skip unconditionally.
        # With the switch on, a same-event candidate is the ladder sub-pass's,
        # which counts it (here: identical wording); counting it on the
        # sweep's line as well would report it twice.
        groups = _group_by_normalized_title(self._combo_heavy())
        with caplog.at_level(logging.INFO):
            assert _extract_pairs(groups, same_event_ladders=True) == []
        counts = _logged_counts(caplog)
        assert _TS_SAME_EVENT not in counts
        assert counts[_TS_SERIES] == 6
        assert counts[_LADDER_REFUSED] == 1
        assert counts[_LADDER_FORMED] == 0

    def test_the_same_title_lines_are_the_live_finders_verbatim(self, caplog):
        # The backtest mirrors find_same_title_pairs' two lines word for word,
        # so one grep reads either path. Same fixture through both: the live
        # finder gets priced namespaces, the backtester the dicts.
        recs = self._combo_heavy()
        live = [SimpleNamespace(ticker=r["ticker"], event_ticker=r["event_ticker"],
                                title=r["title"], subtitle=r["subtitle"],
                                _event_title=r["event_title"], yes_ask_dollars="0.50",
                                no_ask_dollars="0.50",
                                close_time=_parse_iso_datetime(r["close_time"]))
                for r in recs]
        with caplog.at_level(logging.INFO):
            scanner.find_same_title_pairs(live)
        live_lines = [m for m in caplog.messages
                      if m.startswith((_ST_SAME_EVENT, _ST_SERIES, _ST_CLOSE_GAP))]
        caplog.clear()
        with caplog.at_level(logging.INFO):
            _extract_pairs(_group_by_exact_title(recs))
        backtest_lines = [m for m in caplog.messages
                          if m.startswith((_ST_SAME_EVENT, _ST_SERIES, _ST_CLOSE_GAP))]
        assert len(live_lines) == 2
        assert backtest_lines == live_lines

    def test_the_time_series_series_line_shares_the_live_prefix(self, caplog):
        # Only the parenthesis differs from find_time_series_pairs' line, as
        # it does for the DR-72 lines: the sweep is windowed, the live loop
        # is not.
        with caplog.at_level(logging.INFO):
            _extract_pairs(_group_by_normalized_title(self._combo_heavy()))
        [line] = [m for m in caplog.messages if m.startswith(_TS_SERIES)]
        assert line == (
            "Time-series candidates skipped as two instances of one event series "
            "(identical wording, different fixture; within the deadline-gap "
            "window, before price filters): 6"
        )

    @pytest.mark.parametrize("grouping", [_group_by_exact_title, _group_by_normalized_title])
    def test_silent_at_zero(self, caplog, grouping):
        # control — kills a mutant that drops any of the six `if count:`
        # guards. Two different series asking one cumulative question at two
        # deadlines: nothing is refused by either rule on either branch — and
        # the two records share a close, so the same-title close gate
        # (DR-74) refuses nothing either.
        recs = [
            self._rec("KXAA-1-T", "KXAA-1", "Will X happen by March 1, 2026?",
                      "2026-03-01T00:00:00Z"),
            self._rec("KXBB-1-T", "KXBB-1", "Will X happen by March 1, 2026?",
                      "2026-03-01T00:00:00Z"),
        ]
        with caplog.at_level(logging.INFO):
            _extract_pairs(grouping(recs), same_event_ladders=False)
        counts = _logged_counts(caplog)
        for prefix in (_TS_UNDATED, _TS_BEYOND, _TS_SAME_EVENT, _TS_SERIES,
                       _ST_SAME_EVENT, _ST_SERIES, _ST_CLOSE_GAP, _ST_UNDATED):
            assert prefix not in counts

    @pytest.mark.parametrize("ladders", [False, True])
    def test_a_zero_the_sweep_never_visits_is_explained(self, caplog, ladders):
        # regression — fails on the first cut of M10, which counted only the
        # candidates the sweep VISITS: a group whose dated members close 61
        # days apart, beside one with no close_time, returned [] and logged
        # nothing (the P4 critics' reproducer). Worded as two different
        # cumulative deadlines on two series, so no rule would refuse the
        # pair — the window and the missing close are the whole cause.
        recs = [
            self._rec("KXAA-1-T", "KXAA-1", "Will X happen by March 1, 2026?",
                      "2026-03-01T00:00:00Z"),
            self._rec("KXBB-1-T", "KXBB-1", "Will X happen by May 1, 2026?",
                      "2026-05-01T00:00:00Z"),
            self._rec("KXCC-1-T", "KXCC-1", "Will X happen by April 1, 2026?", None),
        ]
        groups = _group_by_normalized_title(recs)
        assert self._group_sizes(groups) == [3]
        with caplog.at_level(logging.INFO):
            assert _extract_pairs(groups, same_event_ladders=ladders) == []
        counts = _logged_counts(caplog)
        assert counts[_TS_UNDATED] == 1
        assert counts[_TS_BEYOND] == 1
        # Nothing was visited, so nothing was refused.
        for prefix in (_TS_SAME_EVENT, _TS_SERIES, _TS_WORDING, _LADDER_REFUSED,
                       _LADDER_GAP_CAP):
            assert prefix not in counts
        [line] = [m for m in caplog.messages if m.startswith(_TS_BEYOND)]
        # Not the live finder's gap-cap line: no rule was evaluated here.
        assert "more than 31 days apart" in line and "gap cap" not in line

    def test_an_empty_grouping_is_reported_by_prepare_candidates(
        self, monkeypatch, caplog,
    ):
        # regression — fails on the first cut of M10. A grouping with no
        # group of two or more reaches _extract_pairs as {}, where every line
        # is silent, so "0 same-title" had no cause in the log. Here the two
        # markets share only the time-series key (their titles differ), so
        # the same-title grouping is empty.
        corpus = [
            _ss1_record("RA", "RAINA-1", "Rain falls by March 1, 2026",
                        event_title="RAIN", close="2026-03-01"),
            _ss1_record("RB", "RAINB-1", "Rain falls by March 20, 2026",
                        event_title="RAIN", close="2026-03-20"),
        ]
        TestGroupableSubset._patch(monkeypatch, corpus)
        with caplog.at_level(logging.INFO):
            TestGroupableSubset._prepare(ladders=False)
        messages = caplog.messages
        [groups_at] = [i for i, m in enumerate(messages) if m.startswith(_GROUPS)]
        assert messages[groups_at] == (
            "Groups of two or more markets: 1 time-series (2 markets), "
            "0 same-title (0 markets) — pairs form only inside a group"
        )
        census_at = next(i for i, m in enumerate(messages)
                         if m.startswith("Deadline phrasing over"))
        pairs_at = next(i for i, m in enumerate(messages)
                        if m.startswith("Potential pairs:"))
        # After the census, before any per-candidate line and the pair count.
        assert census_at < groups_at < pairs_at
        assert messages[pairs_at] == "Potential pairs: 1 time-series, 0 same-title"

    @staticmethod
    def _fuzz(seed: int) -> list[dict]:
        """Every rule's shape at random: one-series fixtures, KXMVE prefixes,
        same-event duplicates, empty event tickers, ladders, and cumulative,
        snapshot and deadline-less wording, over a 60-day close spread so the
        sweep's window cuts some candidates.

        Every other re-listed duplicate also copies its source's close_time
        (DR-74), so some same-title candidates close at one instant and survive
        the close gate while the rest are refused by it. That copy is keyed on
        the record's index, not on a random draw, so the rng stream — and with
        it every other field of every record — is the one the fuzz always
        drew."""
        rng = random.Random(seed)
        stems = ["Will X happen", "Will Y win", "Starship launches", "Q"]
        preps = ["by", "before", "on", "in", ""]
        months = ["March", "April", "May"]
        series = ["KXAA", "KXBB", "KXMVECROSSCATEGORY", "KXMVECROSSCATEGORY0", ""]
        recs = []
        for i in range(240):
            prep = rng.choice(preps)
            stem = rng.choice(stems)
            title = (f"{stem} {prep} {rng.choice(months)} {rng.randint(1, 28)}, 2026?"
                     if prep else f"{stem}?")
            ser = rng.choice(series)
            rec = {
                "ticker": f"F{seed}-{i:04d}",
                "event_ticker": f"{ser}-{rng.randint(0, 5)}" if ser else "",
                "event_title": rng.choice(["", "Event A"]),
                "title": title,
                "subtitle": rng.choice(["", "Yes", title]),
                "close_time": (datetime(2026, 3, 1, tzinfo=UTC)
                               + timedelta(days=rng.randint(0, 60))).isoformat(),
            }
            if recs and rng.random() < 0.25:
                # Re-list an earlier market's exact wording — on its own event
                # half the time (a same-event duplicate), otherwise on this
                # record's random event — so every rule has candidates to
                # refuse whatever the seed.
                src = rng.choice(recs)
                rec.update(title=src["title"], subtitle=src["subtitle"],
                           event_title=src["event_title"])
                if rng.random() < 0.5:
                    rec["event_ticker"] = src["event_ticker"]
                if i % 2 == 0:
                    rec["close_time"] = src["close_time"]
            recs.append(rec)
        return recs

    @pytest.mark.parametrize("ladders", [False, True])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_every_candidate_is_returned_or_counted_exactly_once(
        self, caplog, seed, ladders,
    ):
        # The property the finding asks for: the counts EXPLAIN the result.
        # The candidates are enumerated here independently of _extract_pairs
        # (the sweep's window is the only rule restated), then compared with
        # what it returned plus what it counted.
        recs = self._fuzz(seed)
        # Some members with no readable close_time, which the sweep sets
        # aside before visiting anything (one malformed, the rest absent). The
        # same-title branch does NOT set them aside — its groups are not
        # close-filtered — so they are what feeds its backtest-only
        # unreadable-close count (DR-74; 1/3/5 such candidates for seeds
        # 0/1/2).
        for k, rec in enumerate(recs):
            if k % 17 == 5:
                rec["close_time"] = "not a date" if k == 5 else None
        margin = timedelta(days=MAX_DEADLINE_GAP_DAYS + 1)
        ts_groups = _group_by_normalized_title(recs)
        st_groups = _group_by_exact_title(recs)

        cross = same_event_in_window = ladder_population = 0
        undated = beyond = dated_pairs = 0
        for members in ts_groups.values():
            dated = [m for m in members if _parse_iso_date(m["close_time"]) is not None]
            undated += len(members) - len(dated)
            dated_pairs += len(dated) * (len(dated) - 1) // 2
            for x, a in enumerate(dated):
                for b in dated[x + 1:]:
                    gap = abs(_parse_iso_date(a["close_time"]) - _parse_iso_date(b["close_time"]))
                    if gap > margin:
                        beyond += 1
                        continue
                    if a["event_ticker"] == b["event_ticker"]:
                        same_event_in_window += 1
                    else:
                        cross += 1
            buckets = defaultdict(int)
            for m in dated:
                buckets[m["event_ticker"]] += 1
            ladder_population += sum(n * (n - 1) // 2 for n in buckets.values())
        st_candidates = sum(len(v) * (len(v) - 1) // 2 for v in st_groups.values())

        with caplog.at_level(logging.INFO):
            ts_pairs = _extract_pairs(ts_groups, same_event_ladders=ladders)
        ts = _logged_counts(caplog)
        caplog.clear()
        with caplog.at_level(logging.INFO):
            st_pairs = _extract_pairs(st_groups, same_event_ladders=ladders)
        st = _logged_counts(caplog)

        ladder_pairs = ts[_LADDER_FORMED]
        # Not vacuous: each rule fires, and something survives — including
        # both same-title close-gate refusals (DR-74): the malformed/absent
        # close_times set above supply the unreadable ones, and the fuzz's
        # uncopied closes the ones more than an hour apart.
        assert ts[_TS_SERIES] and st[_ST_SERIES] and st[_ST_SAME_EVENT]
        assert st[_ST_UNDATED] and st[_ST_CLOSE_GAP]
        assert len(ts_pairs) and len(st_pairs)
        # What the sweep set aside before visiting anything, and the whole
        # partition of the dated members' pairs: beyond the window, or
        # visited (same-event or cross-event).
        assert ts[_TS_UNDATED] == undated > 0
        assert ts[_TS_BEYOND] == beyond > 0
        assert dated_pairs == beyond + same_event_in_window + cross
        assert cross == (len(ts_pairs) - ladder_pairs) + ts[_TS_SERIES] + ts[_TS_WORDING]
        if ladders:
            assert _TS_SAME_EVENT not in ts
            assert ladder_population == (
                ts[_LADDER_REFUSED] + ts[_LADDER_GAP_CAP] + ladder_pairs
            )
        else:
            assert ts[_TS_SAME_EVENT] == same_event_in_window > 0
            assert ladder_pairs == 0
        assert st_candidates == (
            len(st_pairs) + st[_ST_SAME_EVENT] + st[_ST_SERIES]
            + st[_ST_UNDATED] + st[_ST_CLOSE_GAP]
        )


class TestSameTitleCloseGapBacktest:
    """DR-74 mirror: _extract_pairs' same-title branch refuses identical
    wording on two DIFFERENT series unless both markets close within
    SAME_TITLE_MAX_CLOSE_GAP_SECONDS of each other.

    The verdict is scanner.closes_apart, the ONE definition the live finder
    uses; the backtest only parses the cached close strings first
    (_comparable_closes_dicts) and counts a pair it cannot date on a line of
    its own, because the live finder never reaches such a market. The backtest
    reads REALIZED closes. Mirror of test_scanner.py::TestSameTitleCloseGap.
    """

    _WIU_TITLE = "Western Illinois at Eastern Illinois Winner?"

    @classmethod
    def _wiu(cls, *, womens_close="2026-01-13T23:30:00Z"):
        # The real pair's REALIZED closes sat 2 h 41 m apart (the clock times
        # here are the fixture's own; the gap is the measured one).
        def rec(ticker, event_ticker, close):
            return {"ticker": ticker, "event_ticker": event_ticker,
                    "event_title": "Western Illinois at Eastern Illinois",
                    "title": cls._WIU_TITLE, "subtitle": "Western Illinois",
                    "close_time": close}
        return [
            rec("KXNCAAMBGAME-26JAN13WIUEIU-WIU", "KXNCAAMBGAME-26JAN13WIUEIU",
                "2026-01-14T02:11:00Z"),
            rec("KXNCAAWBGAME-26JAN13WIUEIU-WIU", "KXNCAAWBGAME-26JAN13WIUEIU",
                womens_close),
        ]

    def test_mens_and_womens_game_are_refused_and_counted_once(self, caplog):
        recs = self._wiu()
        gap = abs(_parse_iso_datetime(recs[0]["close_time"])
                  - _parse_iso_datetime(recs[1]["close_time"]))
        assert gap == timedelta(hours=2, minutes=41)
        assert backtester._same_series_dicts(*recs) is False
        groups = _group_by_exact_title(recs)
        assert len(groups) == 1
        with caplog.at_level(logging.INFO):
            assert _extract_pairs(groups) == []
        counts = _logged_counts(caplog)
        assert counts[_ST_CLOSE_GAP] == 1
        assert _ST_UNDATED not in counts
        assert _ST_SERIES not in counts

    def test_the_same_pair_at_one_close_instant_is_a_candidate(self):
        # Positive control: only the women's close moves.
        recs = self._wiu(womens_close="2026-01-14T02:11:00Z")
        assert len(_extract_pairs(_group_by_exact_title(recs))) == 1

    @pytest.mark.parametrize("gap_seconds,pairs", [
        (0, 1), (3_600, 1), (3_601, 0),
    ])
    def test_the_bound_is_one_hour_inclusive(self, gap_seconds, pairs):
        close = datetime(2026, 1, 14, 2, 11, tzinfo=UTC) - timedelta(seconds=gap_seconds)
        recs = self._wiu(womens_close=close.isoformat())
        assert len(_extract_pairs(_group_by_exact_title(recs))) == pairs

    def test_the_gate_is_the_scanner_definition(self, monkeypatch, caplog):
        # The backtest has no bound of its own: narrowing scanner's binding
        # (the one closes_apart reads) narrows this gate too — and the bound
        # its refusal line PRINTS, which must be the one the gate applied, not
        # a second copy of the constant still reading 60 minutes.
        monkeypatch.setattr(scanner, "SAME_TITLE_MAX_CLOSE_GAP_SECONDS", 60)
        base = datetime(2026, 1, 14, 2, 11, tzinfo=UTC)
        at_60 = self._wiu(womens_close=(base - timedelta(seconds=60)).isoformat())
        at_61 = self._wiu(womens_close=(base - timedelta(seconds=61)).isoformat())
        assert len(_extract_pairs(_group_by_exact_title(at_60))) == 1
        with caplog.at_level(logging.INFO):
            assert _extract_pairs(_group_by_exact_title(at_61)) == []
        assert [m for m in caplog.messages if m.startswith(_ST_CLOSE_GAP)] == [
            "Same-title candidates refused because the two markets close more "
            "than 1 minute apart (two different games or instants, not one "
            "question listed twice): 1"
        ]

    @pytest.mark.parametrize("womens_close", [
        None,                     # absent
        "",                       # blank
        "not-a-timestamp",        # malformed
        "2026-01-14T02:11:00",    # naive, beside an aware partner
    ])
    def test_an_unreadable_close_is_refused_on_its_own_line(self, caplog, womens_close):
        # Fail CLOSED, and on the backtest-only line: the live finder never
        # reaches such a market (_filter_active_markets drops a missing close
        # before grouping), so counting it on the close-gap line would make
        # that line stop being the live one's twin.
        recs = self._wiu(womens_close=womens_close)
        with caplog.at_level(logging.INFO):
            assert _extract_pairs(_group_by_exact_title(recs)) == []
        counts = _logged_counts(caplog)
        assert counts[_ST_UNDATED] == 1
        assert _ST_CLOSE_GAP not in counts
        assert caplog.messages.count(
            "Same-title candidate pairs refused because a market's close_time "
            "cannot be read (fail closed; the live finder drops such markets "
            "before grouping): 1"
        ) == 1

    def test_the_lines_are_silent_at_zero(self, caplog):
        recs = self._wiu(womens_close="2026-01-14T02:11:00Z")
        with caplog.at_level(logging.INFO):
            assert len(_extract_pairs(_group_by_exact_title(recs))) == 1
        counts = _logged_counts(caplog)
        assert _ST_CLOSE_GAP not in counts and _ST_UNDATED not in counts

    def test_the_close_gap_line_is_the_live_finders_verbatim(self, caplog):
        # A SEPARATE fixture that reaches the gate: two non-combo series,
        # identical wording, closes more than an hour apart (the combo-heavy
        # fixture is refused by the series rule first and never gets here).
        # One grep reads either path, so the two lines must match word for
        # word — the count included.
        recs = self._wiu() + [
            {"ticker": "KXNCAAWBGAME-26FEB10WIUEIU-WIU",
             "event_ticker": "KXNCAAWBGAME-26FEB10WIUEIU",
             "event_title": "Western Illinois at Eastern Illinois",
             "title": self._WIU_TITLE, "subtitle": "Western Illinois",
             "close_time": "2026-02-10T23:30:00Z"},
        ]
        live = [SimpleNamespace(ticker=r["ticker"], event_ticker=r["event_ticker"],
                                title=r["title"], subtitle=r["subtitle"],
                                _event_title=r["event_title"],
                                yes_ask_dollars=str(0.40 - 0.1 * i),
                                no_ask_dollars=str(0.60 + 0.1 * i),
                                close_time=_parse_iso_datetime(r["close_time"]))
                for i, r in enumerate(recs)]
        with caplog.at_level(logging.INFO):
            assert scanner.find_same_title_pairs(live) == []
        live_lines = [m for m in caplog.messages
                      if m.startswith((_ST_SAME_EVENT, _ST_SERIES, _ST_CLOSE_GAP))]
        caplog.clear()
        with caplog.at_level(logging.INFO):
            assert _extract_pairs(_group_by_exact_title(recs)) == []
        backtest_lines = [m for m in caplog.messages
                          if m.startswith((_ST_SAME_EVENT, _ST_SERIES, _ST_CLOSE_GAP))]
        # Men's vs each women's listing: two close-gap refusals; the two
        # women's listings of one series: one series refusal.
        assert live_lines == [
            "Same-title candidates skipped as two instances of one event series "
            "(identical wording, different fixture): 1",
            "Same-title candidates refused because the two markets close more "
            "than 60 minutes apart (two different games or instants, not one "
            "question listed twice): 2",
        ]
        assert backtest_lines == live_lines

    @pytest.mark.parametrize("a,b,readable", [
        ({}, {"close_time": "2026-01-14T02:11:00Z"}, False),
        ({"close_time": None}, {"close_time": "2026-01-14T02:11:00Z"}, False),
        ({"close_time": "not-a-timestamp"}, {"close_time": "2026-01-14T02:11:00Z"}, False),
        ({"close_time": "2026-01-14T02:11:00"}, {"close_time": "2026-01-14T02:11:00Z"}, False),
        ({"close_time": "2026-01-14T02:11:00Z"}, {"close_time": "2026-01-14T02:11:00"}, False),
        ({"close_time": "2026-01-14T02:11:00"}, {"close_time": "2026-01-14T05:00:00"}, True),
        ({"close_time": "2026-01-14T02:11:00Z"}, {"close_time": "2026-01-15T02:11:00+00:00"}, True),
    ])
    def test_comparable_closes_dicts(self, a, b, readable):
        out = backtester._comparable_closes_dicts(a, b)
        if not readable:
            assert out is None
            return
        ca, cb = out
        assert ca == _parse_iso_datetime(a["close_time"])
        assert cb == _parse_iso_datetime(b["close_time"])


class TestDeadlineProfileParity:
    """The live scanner (attribute-based) and the backtester (dict-based)
    field extraction must agree on every input, since only the field
    EXTRACTION differs between the two paths — the classification itself is
    one function, scanner.deadline_profile, called by both (pinned by AST in
    test_strategy.py). An argument swap or a dropped subtitle in either
    extraction helper would otherwise split the two paths silently, with
    nothing to catch it: the AST pin only checks that deadline_profile is
    called, never what it is called WITH.
    """

    # control — kills M23a/b/c/d/f/g (see test_missing_event_title_key_reads_
    # as_absent below for the M23d case: a genuinely ABSENT "event_title" key,
    # as distinct from a present-but-blank one).
    @pytest.mark.parametrize("title, subtitle, event_title", [
        # Conflicting markers across fields — subtitle decides.
        ("Bitcoin price on Sep 15, 2026?", "$80,000 by June 30", ""),
        # Conflicting markers — title decides over event_title, both
        # directions.
        ("Will X happen before Jan 1, 2027?", "", "Will X happen in 2026?"),
        ("Top 10 in October?", "", "Will X happen by Dec 31, 2026?"),
        # subtitle=None — a legitimate cached shape (historical._market_to_dict
        # stores `subtitle or yes_sub_title`, which is None when both are
        # absent) — must read as absent on both paths.
        ("Will X happen?", None, ""),
        # blank event_title.
        ("Will X happen by March 1?", "", ""),
    ])
    def test_dict_and_live_extraction_agree(self, title, subtitle, event_title):
        live_market = SimpleNamespace(
            title=title, subtitle=subtitle or "", _event_title=event_title,
        )
        record = {"title": title, "subtitle": subtitle, "event_title": event_title}
        assert (
            backtester._deadline_profile_dict(record)
            == scanner._market_deadline_profile(live_market)
        )

    def test_missing_event_title_key_reads_as_absent(self):
        # control — old cache records predate "event_title" entirely (see
        # _deadline_profile_dict's own docstring), which the parametrized
        # cases above never exercise: every record there carries the key,
        # with "" as its blank value, and `m.get("event_title") or ""` cannot
        # tell a genuinely missing key from a present blank one. This case
        # covers the record side; the live side's matching default is a
        # market object with no _event_title attribute at all, exercising
        # _market_deadline_profile's own getattr(..., "") default.
        record = {"title": "Will X happen by March 1?", "subtitle": ""}
        live_market = SimpleNamespace(title="Will X happen by March 1?", subtitle="")
        assert "event_title" not in record
        assert not hasattr(live_market, "_event_title")
        assert (
            backtester._deadline_profile_dict(record)
            == scanner._market_deadline_profile(live_market)
            == (scanner.DEADLINE_CUMULATIVE, ("by march 1",))
        )

    @pytest.mark.parametrize("title, live_event_title, spans", [
        ("Will X cut by June 1, 2026?", "Will X cut by June 20, 2026?", ("by june 1, 2026",)),
        ("Will X cut by June 20, 2026?", "Will X cut by June 1, 2026?", ("by june 20, 2026",)),
    ])
    def test_blank_cached_event_title_agrees_with_live(self, title, live_event_title, spans):
        # regression — DR-69. Live ingest attaches an event title to nearly
        # every market; most cached records carry "" there. When the TITLE
        # decides, the event title must not change the profile, or the two
        # paths judge the same market differently. Before DR-69 the live
        # profile folded the event title's (different) date into the spans
        # and the blank-event-title record did not, so these two disagreed.
        live_market = SimpleNamespace(
            title=title, subtitle="", _event_title=live_event_title,
        )
        record = {"title": title, "subtitle": "", "event_title": ""}
        assert (
            scanner._market_deadline_profile(live_market)
            == backtester._deadline_profile_dict(record)
            == (scanner.DEADLINE_CUMULATIVE, spans)
        )


class TestSpansFromDecidingField:
    """Backtester mirror of test_scanner.py::TestSpansFromDecidingField (DR-69):
    the same fixtures through the dict-based grouping and extraction path.
    Spans come from the field that decided the verdict only, which can refuse
    a pair the old all-field union admitted AND admit one it refused.

    The fixtures carry their event title in the record, so this exercises the
    rule on a cache that has event titles; the blank-event-title case is
    TestDeadlineProfileParity::test_blank_cached_event_title_agrees_with_live.
    """

    @staticmethod
    def _rec(ticker, event_ticker, title, close_time, *, subtitle="", event_title=""):
        rec = _md(ticker, event_ticker, title=title, subtitle=subtitle,
                  event_title=event_title)
        rec["close_time"] = close_time
        return rec

    def test_spanless_subtitle_cannot_borrow_event_title_spans(self):
        # regression — before DR-69 _extract_pairs emitted this pair: the
        # spanless "At any time" subtitle decided "cumulative" and borrowed
        # the event titles' distinct dates.
        mA = self._rec(
            "PA-1", "EVA-1", "Will SOL be above $180 on Sep 14, 2026?",
            "2026-09-14T00:00:00Z", subtitle="At any time",
            event_title="SOL above $180 by Sep 14, 2026?",
        )
        mB = self._rec(
            "PB-1", "EVB-1", "Will SOL be above $180 on Sep 18, 2026?",
            "2026-09-18T00:00:00Z", subtitle="At any time",
            event_title="SOL above $180 by Sep 18, 2026?",
        )
        groups = TestDeadlineGuardFinders._one_group([mA, mB])
        assert backtester._deadline_profile_dict(mA) == (scanner.DEADLINE_CUMULATIVE, ())
        assert backtester._deadline_profile_dict(mB) == (scanner.DEADLINE_CUMULATIVE, ())
        assert _extract_pairs(groups) == []

        # Control: the deciding subtitle names the two deadlines itself.
        cA = self._rec(
            "PA-1", "EVA-1", "Will SOL be above $180 on Sep 14, 2026?",
            "2026-09-14T00:00:00Z", subtitle="By Sep 14, 2026",
            event_title="SOL above $180 by Sep 14, 2026?",
        )
        cB = self._rec(
            "PB-1", "EVB-1", "Will SOL be above $180 on Sep 18, 2026?",
            "2026-09-18T00:00:00Z", subtitle="By Sep 18, 2026",
            event_title="SOL above $180 by Sep 18, 2026?",
        )
        assert len(_extract_pairs(TestDeadlineGuardFinders._one_group([cA, cB]))) == 1

    def test_deciding_field_spans_can_admit_a_pair(self):
        # regression — before DR-69 this pair was refused: each event title
        # names the OTHER leg's date, so the all-field span unions matched.
        mA = self._rec(
            "PA-1", "EVA-1", "Will X cut by June 1, 2026?", "2026-06-01T00:00:00Z",
            event_title="Will X cut by June 20, 2026?",
        )
        mB = self._rec(
            "PB-1", "EVB-1", "Will X cut by June 20, 2026?", "2026-06-20T00:00:00Z",
            event_title="Will X cut by June 1, 2026?",
        )
        groups = TestDeadlineGuardFinders._one_group([mA, mB])
        assert backtester._deadline_profile_dict(mA) == (
            scanner.DEADLINE_CUMULATIVE, ("by june 1, 2026",),
        )
        assert backtester._deadline_profile_dict(mB) == (
            scanner.DEADLINE_CUMULATIVE, ("by june 20, 2026",),
        )
        [(a, b, _canon, key)] = _extract_pairs(groups)
        assert isinstance(key, str)  # a string key is the time-series branch
        assert {a["ticker"], b["ticker"]} == {"PA-1", "PB-1"}

    def test_mve_event_title_route_still_pairs(self):
        # control — kills a mutant that stops _deciding_field falling through
        # to the event title. Dateless option label, empty title, deadline in
        # the parent event title (the deciding field).
        mA = self._rec(
            "PA-1", "EVA-1", "", "2026-03-01T00:00:00Z", subtitle="Trump",
            event_title="Presidential Election Winner by March 1, 2026",
        )
        mB = self._rec(
            "PB-1", "EVB-1", "", "2026-03-20T00:00:00Z", subtitle="Trump",
            event_title="Presidential Election Winner by March 20, 2026",
        )
        groups = TestDeadlineGuardFinders._one_group([mA, mB])
        assert backtester._deadline_profile_dict(mA) == (
            scanner.DEADLINE_CUMULATIVE, ("by march 1, 2026",),
        )
        assert len(_extract_pairs(groups)) == 1


class TestClassifyOncePerMarket:
    """Backtester mirror of test_scanner.py::TestClassifyOncePerMarket.
    Patches backtester.deadline_profile directly: deadline_profile is
    imported BY NAME into backtester.py's module namespace, so patching
    scanner.deadline_profile would never reach _deadline_profile_dict's call
    at all.
    """

    def test_extract_pairs_classifies_each_member_once(self, monkeypatch):
        # control — kills M11b (900 calls instead of 30 — one per candidate
        # pair rather than one per group member).
        base = datetime(2026, 1, 1, tzinfo=UTC)
        members = []
        for i in range(29):
            d = base + timedelta(days=i)
            rec = _md(f"T{i}", f"EVT{i}", title=f"Will X happen by {d:%B %d, %Y}?")
            rec["close_time"] = d.isoformat()
            members.append(rec)
        # A duplicate deadline (same stated span as T5) forces a REAL
        # phrasing refusal inside the group, exercising classify-once on the
        # refusal path too.
        d5 = base + timedelta(days=5)
        dup = _md("T5DUP", "EVT5DUP", title=f"Will X happen by {d5:%B %d, %Y}?")
        dup["close_time"] = (d5 + timedelta(hours=1)).isoformat()
        members.append(dup)

        groups = _group_by_normalized_title(members)
        assert len(groups) == 1
        [dated_members] = groups.values()

        calls = {"n": 0}
        original = backtester.deadline_profile

        def counting(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(backtester, "deadline_profile", counting)
        pairs = _extract_pairs(groups)
        assert len(pairs) >= 1
        assert calls["n"] == len(dated_members)


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
        # RE-PINNED (DR-02/DR-54): EVT-A/EVT-B share the series prefix "EVT",
        # which _extract_pairs now refuses as two instances of one recurring
        # fixture. The tickers name two DIFFERENT series so this test keeps
        # pinning what it is for — that an old cache record missing the tick
        # and subtitle keys still pairs.
        #
        # RE-PINNED (DR-74): the two records closed a week apart, which the
        # same-title close gate now refuses as two different fixtures; they
        # share one close so the pair again turns on the missing fields alone.
        mA = self._old_style_dict("A1", "EVA-1", "Republicans win majority",
                                   "2026-01-01T00:00:00Z", event_title="2026 Senate Control")
        mB = self._old_style_dict("B1", "EVB-1", "Republicans win majority",
                                   "2026-01-01T00:00:00Z", event_title="2026 Senate Control")
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

    @staticmethod
    def _same_date_markets():
        # Both legs close on the SAME UTC date, twelve hours apart — a genuine
        # zero-day-gap pair (short 15% tier), and the population TS-06
        # mis-ordered: _parse_iso_date collapses both to 2026-02-09, so
        # neither `close_b < close_a` nor its mirror is true and the swap
        # never fired.
        early = {"ticker": "EARLY", "event_ticker": "E1",
                 "close_time": "2026-02-09T09:00:00+00:00"}
        late  = {"ticker": "LATE", "event_ticker": "E2",
                 "close_time": "2026-02-09T21:00:00+00:00"}
        return early, late

    def test_same_date_legs_are_ordered_by_close_time_not_close_date(self):
        # Passed LATE-first (what the group list produces when it happens to
        # hold the later contract first): A must still come back as the 09:00
        # contract and the YES leg must be bought on it.
        early, late = self._same_date_markets()
        candles_early = [_candle(_MONDAY_TS, 0.30, 0.70)]
        candles_late  = [_candle(_MONDAY_TS, 0.60, 0.40)]
        entry = _find_entry(candles_late, candles_early, late, early,
                            "time_series", date(2026, 1, 1))
        assert entry is not None
        assert entry["mA"]["ticker"] == "EARLY"
        assert entry["mB"]["ticker"] == "LATE"
        # The YES leg is bought on A at pA; the NO leg on B at nB
        assert (entry["pA"], entry["nB"]) == pytest.approx((0.30, 0.40))
        # Twelve hours apart floors to a zero-day deadline gap (short tier)
        assert entry["gap_days"] == 0

    def test_same_date_entry_is_independent_of_the_argument_order(self):
        early, late = self._same_date_markets()
        candles_early = [_candle(_MONDAY_TS, 0.30, 0.70)]
        candles_late  = [_candle(_MONDAY_TS, 0.60, 0.40)]
        early_first = _find_entry(candles_early, candles_late, early, late,
                                  "time_series", date(2026, 1, 1))
        late_first  = _find_entry(candles_late, candles_early, late, early,
                                  "time_series", date(2026, 1, 1))
        assert early_first is not None
        assert early_first == late_first

    def test_same_date_pricier_earlier_contract_rejected_in_both_orders(self):
        # The earlier contract is the dear one (0.60 vs 0.30) — never a
        # candidate in either direction. Before TS-06 the LATE-first order
        # skipped the swap, so pB − pA read as +0.30 and the pair ENTERED with
        # its legs inverted; its genuine in-between settlement then booked as
        # the impossible A=YES/B=NO premise violation.
        early, late = self._same_date_markets()
        candles_early = [_candle(_MONDAY_TS, 0.60, 0.40)]
        candles_late  = [_candle(_MONDAY_TS, 0.30, 0.70)]
        assert _find_entry(candles_early, candles_late, early, late,
                           "time_series", date(2026, 1, 1)) is None
        assert _find_entry(candles_late, candles_early, late, early,
                           "time_series", date(2026, 1, 1)) is None

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

    The time-series extraction path drops such members while windowing by
    close date (TestExtractPairsWindowedEquivalence). The same-title path is
    naive and used to carry them straight into the candlestick fetch, where
    the pre-pool window computation used a bare datetime.fromisoformat — on
    the main thread, before any worker starts, so it aborted the whole
    backtest (BS-07); _fetch_candles_parallel then learned to resolve such a
    ticker to an empty series with a WARNING.

    Since DR-74 the same-title branch refuses the pair itself, before any
    candle is requested: the close gate cannot show that a market it cannot
    date closes with its partner, so it fails closed and counts the pair on
    its own backtest-only line. The pipeline therefore no longer reaches
    BS-07's branch at all, and the second test below calls
    _fetch_candles_parallel directly so that branch stays covered.
    """

    def test_malformed_close_time_is_skipped_not_raised(self, monkeypatch, caplog):
        markets = _same_title_markets(close_time_b="not-a-timestamp")
        fetched = []
        monkeypatch.setattr(backtester, "fetch_all_settled_markets",
                            lambda *a, **k: markets)
        monkeypatch.setattr(backtester, "fetch_candlesticks",
                            lambda _c, ticker, *a, **k: fetched.append(ticker) or [])

        with caplog.at_level(logging.INFO):
            trades, equity = run_backtest(
                hist_client=MagicMock(), live_client=MagicMock(),
                start_date=date(2026, 1, 1), initial_balance=1000.0,
            )

        # No trade, no exception — and no candle request at all: the pair was
        # refused before the fetch, so neither leg is a needed ticker.
        assert trades == []
        assert float(equity["portfolio_value"].iloc[-1]) == pytest.approx(1000.0)
        assert fetched == []
        # Refused on the backtest-only unreadable-close line (DR-74), not on
        # the close-gap line — the gap between them was never measurable.
        assert caplog.messages.count(
            "Same-title candidate pairs refused because a market's close_time "
            "cannot be read (fail closed; the live finder drops such markets "
            "before grouping): 1"
        ) == 1
        assert not any(m.startswith(_ST_CLOSE_GAP) for m in caplog.messages)
        assert "Potential pairs: 0 time-series, 0 same-title" in caplog.messages

    def test_the_candle_fetch_still_resolves_an_unparseable_close_to_nothing(
        self, monkeypatch, caplog,
    ):
        # BS-07's branch, called directly now that the pipeline cannot reach
        # it: a malformed close_time resolves to an empty series with a
        # WARNING naming the ticker, requests nothing, and never raises.
        def must_not_fetch(*_a, **_k):
            raise AssertionError("fetch_candlesticks called for an undatable market")

        monkeypatch.setattr(backtester, "fetch_candlesticks", must_not_fetch)
        with caplog.at_level(logging.WARNING):
            out = _fetch_candles_parallel(
                MagicMock(), {"SB": {"ticker": "SB", "close_time": "not-a-timestamp"}},
                date(2026, 1, 1), False,
            )
        assert out == {"SB": []}
        warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        # The dropped ticker is named, so an empty series can't be mistaken
        # for "this market genuinely had no prices".
        assert any("SB" in msg and "close_time" in msg for msg in warnings)


class TestActiveTickerRelease:
    """A ticker is blocked only while its position is OPEN (BS-24).

    Live, scanner.get_held_tickers() reads positions with count_filter="position",
    so a ticker leaves the blocked set the moment its market settles. The
    backtest used to add tickers to active_tickers and never remove them, so one
    early trade blocked that ticker for the entire remaining simulation.

    Fixture: TX/TY share an exact title and one close instant, 2026-02-02 (a
    same-title pair, entering at the first Monday — the one close is what the
    same-title close gate, DR-74, requires); TX/TZ share a normalized title
    with an 18-day deadline gap (time-series pair, which can only enter at the
    second Monday because TZ has no earlier candle). Both candidates therefore
    contain TX, at different entry dates. TX settles before its close_time —
    an early determination, which is what makes re-entry on a shared ticker
    reachable at all (TY settles early too, on 2026-01-08).

    Prices: TX yes 0.40 / no 0.60 throughout; TY yes 0.30 / no 0.70 at the
    first Monday and 0.60 / 0.40 from the second; TZ yes 0.75 / no 0.25 from
    the second Monday. Same-title TX/TY at the first Monday: TX is the pricier
    side, gap 0.10 >= 0.05, legs nA+pB = 0.60+0.30 = 0.90 <= 0.95.
    Time-series TX/TZ at the second Monday: TX is the earlier contract and TZ
    (later) is priced 0.35 higher, clearing the 30% long-gap tier; legs
    pA+nB = 0.40+0.25 = 0.65 <= 0.70, and under the interval discount
    (p = 1 - 0.75*0.35) the Kelly fraction is ~0.180 — positive, so the pair
    really is entered. TX/TY is never a time-series candidate: their titles
    are identical, so they state one deadline ("by March 2026") twice and the
    cumulative-deadline rule refuses them (DR-67). TY/TZ IS a candidate — TY
    now closes with TX, 18 days before TZ, and the two are worded as two
    different deadlines — which is why TY carries a second-Monday candle: at
    0.60, TZ's 0.75 is only 0.15 above it, under the 30% long-gap tier, and at
    the first Monday TZ has no candle at all. So TX/TZ is the only
    time-series entry.

    RE-DESIGNED (DR-74): TY used to close on 2026-01-20, 13 days before TX,
    which the same-title close gate now refuses as two fixtures; aligning TY's
    close alone was not enough, because it brought TY/TZ inside the 30-day
    deadline-gap cap (it had been 31 days apart), where TY's first-Monday 0.30
    would have entered it at the second Monday. Its second-Monday candle keeps
    it out. No assertion changed.
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
             # TX's close instant: the same-title close gate (DR-74).
             "close_time": "2026-02-02T00:00:00+00:00",
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
            # TY: 0.30 at the first Monday (the same-title entry), then 0.60 —
            # keeps TY/TZ (18 days apart, like TX/TZ) under the 30% tier at the
            # second Monday; see the class docstring
            "TY": [_candle(_MONDAY_TS, 0.30, 0.70), _candle(self._M2, 0.60, 0.40)],
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
    upper = close_time's date - 1 day, and the market is eligible iff some
    Monday on/after start_date and on or before upper has its 09:00 UTC
    checkpoint strictly after open_time. Every market here opens at MIDNIGHT
    UTC, where that is the same as the pre-P5 date test (a Monday on/after
    max(open date, start_date)); the time-of-day cases are
    TestCanEverEnterAtTheCheckpointInstant's."""

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


# ─── P5 (M8): the prefilter tests the checkpoint INSTANT, not the date ────────

def _date_granular_can_ever_enter(m: dict, start_date: date) -> bool:
    """The pre-P5 backtester._can_ever_enter, verbatim: open_time read as a DATE.

    The reference the checkpoint-precise predicate must never widen, and the
    predicate the result-neutrality tests below run the old pipeline under.
    """
    open_d = _parse_iso_date(m.get("open_time"))
    close_d = _parse_iso_date(m.get("close_time"))
    if open_d is None or close_d is None:
        return True
    lower = max(open_d, start_date)
    upper = close_d - timedelta(days=1)
    if lower > upper:
        return False
    d = lower
    while d.weekday() != 0:
        d += timedelta(days=1)
    return d <= upper


_P5_HOUR = 3_600


def _first_candle_ts(open_dt: datetime) -> int:
    """The earliest candle a market can carry under the measured rule: none
    ends at or before the start of the hour it opened in, so the first ends at
    the next top of the hour (historical.fetch_candlesticks stores the period
    END as a candle's ts)."""
    ts = int(open_dt.timestamp())
    return ts - ts % _P5_HOUR + _P5_HOUR


def _scan_reachable(m: dict, start_date: date) -> bool:
    """An independent statement of what _find_entry could ever reach with this
    market as a leg: some checkpoint _monday_timestamps would scan for SOME
    partner (the union of every partner's window is [start_date, close date
    - 1 day]) has a candle of this market at or before it, i.e. is at or after
    the market's first possible candle. A naive open_time has no instant, so
    the date test is its whole answer."""
    open_dt = _parse_iso_datetime(m.get("open_time"))
    close_d = _parse_iso_date(m.get("close_time"))
    if open_dt is None or close_d is None:
        return True
    if open_dt.utcoffset() is None:
        return _date_granular_can_ever_enter(m, start_date)
    first = _first_candle_ts(open_dt)
    return any(first <= c for c in backtester._monday_timestamps(
        start_date, close_d - timedelta(days=1)))


def _mkt_at(open_time: str | None, close_time: str | None) -> dict:
    """A market dict carrying only the two timestamps _can_ever_enter reads."""
    return {"open_time": open_time, "close_time": close_time}


class TestCanEverEnterAtTheCheckpointInstant:
    """M8 of the 2026-09-24 review (P5): _find_entry needs a candle at or
    before Monday 09:00 UTC for both legs and a market has none before its
    opening hour, so a market that opened AT or AFTER its only checkpoint can
    never be entered. The pre-P5 predicate compared open_time by DATE and kept
    it. 2026-09-21 is the 7-day review window's only Monday."""

    _START = date(2026, 9, 17)
    _CLOSE = "2026-09-24T12:15:33Z"   # no later Monday is admissible

    def test_the_review_example_is_dropped_and_the_date_test_kept_it(self):
        # KXLOLMAP-26SEP240800MVKACBC-1-MVKA, the first record of the
        # 2026-09-17 assembled cache: opened Monday 22:16 UTC, closes Thursday.
        m = _mkt_at("2026-09-21T22:16:00Z", self._CLOSE)
        assert _can_ever_enter(m, self._START) is False
        assert _date_granular_can_ever_enter(m, self._START) is True

    @pytest.mark.parametrize(("open_time", "kept"), [
        ("2026-09-21T08:00:00Z", True),
        # its first candle ends at 09:00, which _candle_at_or_before accepts
        ("2026-09-21T08:30:00Z", True),
        ("2026-09-21T08:59:59Z", True),
        ("2026-09-21T08:59:59.999999Z", True),
        # opened AT the checkpoint: its first candle ends at 10:00
        ("2026-09-21T09:00:00Z", False),
        ("2026-09-21T09:00:01Z", False),
        ("2026-09-21T23:59:59Z", False),
    ])
    def test_the_boundary_is_strictly_before_the_checkpoint(self, open_time, kept):
        assert _can_ever_enter(_mkt_at(open_time, self._CLOSE), self._START) is kept

    def test_a_late_monday_opening_waits_for_the_next_monday(self):
        # The next Monday (09-28) is admissible only when the close date is
        # 09-29 or later (upper = close date - 1 day).
        opened = "2026-09-21T10:00:00Z"
        assert _can_ever_enter(_mkt_at(opened, "2026-09-29T00:00:00Z"), self._START) is True
        assert _can_ever_enter(_mkt_at(opened, "2026-09-28T23:00:00Z"), self._START) is False

    @pytest.mark.parametrize(("open_time", "kept"), [
        ("2026-09-21T04:59:59-04:00", True),    # 08:59:59Z
        ("2026-09-21T05:00:00-04:00", False),   # 09:00:00Z
        ("2026-09-21T22:59:59+14:00", True),    # 08:59:59Z
        # a Tuesday local date, 10:30Z on the Monday: dropped either way
        ("2026-09-22T00:30:00+14:00", False),
        # a SUNDAY local date, 11:30Z on the Monday: the date test read Sunday
        # and kept it; the instant is after the checkpoint
        ("2026-09-20T23:30:00-12:00", False),
    ])
    def test_the_opening_is_compared_as_a_utc_instant(self, open_time, kept):
        assert _can_ever_enter(_mkt_at(open_time, self._CLOSE), self._START) is kept

    def test_a_naive_opening_keeps_the_date_test(self):
        # No offset: the instant is unknown, so the looser date test stands.
        m = _mkt_at("2026-09-21T22:16:00", self._CLOSE)
        assert _can_ever_enter(m, self._START) is True
        assert _can_ever_enter(m, self._START) == _date_granular_can_ever_enter(m, self._START)

    @pytest.mark.parametrize(("open_time", "close_time", "kept"), [
        # The UTC instant falls outside datetime's range (astimezone(UTC)
        # raises OverflowError): the pre-P5 date test decides, exactly as it
        # did.
        ("0001-01-01T00:00:00+14:00", "2026-01-20T00:00:00Z", True),
        ("9999-12-31T23:00:00-05:00", "9999-12-31T23:00:00Z", False),
        # Monday 9999-12-27 opened after its checkpoint: the next Monday lies
        # past date.max, so adding the week would raise ...
        ("9999-12-27T10:00:00Z", "9999-12-31T00:00:00Z", False),
        # ... and before it, that Monday is still reachable.
        ("9999-12-27T08:00:00Z", "9999-12-31T00:00:00Z", True),
        # The advance to Monday would pass date.max, and a close on date.min
        # would put the upper bound before it: the pre-P5 predicate raised on
        # both.
        ("9999-12-28T00:00:00Z", "9999-12-31T00:00:00Z", False),
        ("2026-01-01T00:00:00Z", "0001-01-01T00:00:00Z", False),
    ])
    def test_the_ends_of_the_date_range_never_raise(self, open_time, close_time, kept):
        # The predicate runs inside the fetch's assembly workers, where an
        # exception ends the run; any parseable timestamp must give a verdict.
        assert _can_ever_enter(_mkt_at(open_time, close_time), date(2026, 1, 1)) is kept

    def test_the_ends_of_the_date_range_stay_a_subset_of_the_date_test(self):
        # Seeded sweep of openings, closes and start dates within a few weeks
        # of either end of datetime's range, in offsets from -12:00 to +14:00:
        # never an exception, and never a market the date test dropped.
        rng = random.Random(1_2026_0924)
        zones = [timezone(timedelta(minutes=15 * q)) for q in range(-48, 57)]
        lo, hi = datetime(1, 1, 1), datetime(9999, 12, 31, 23, 59)
        cases = 0
        for _ in range(20_000):
            end = rng.choice((lo, hi))
            sign = 1 if end is lo else -1

            def near(end=end, sign=sign):
                return end + sign * timedelta(minutes=rng.randrange(0, 60 * 24 * 40))

            open_s = near().replace(tzinfo=rng.choice(zones)).isoformat()
            close_s = near().replace(tzinfo=rng.choice(zones)).isoformat()
            start = near().date()
            m = _mkt_at(open_s, close_s)
            new = _can_ever_enter(m, start)
            try:
                old = _date_granular_can_ever_enter(m, start)
            except OverflowError:
                continue   # the pre-P5 predicate had no verdict to compare
            cases += 1
            assert not (new and not old), (open_s, close_s, start)
        assert cases > 10_000

    def test_a_start_date_on_the_opening_monday(self):
        monday = date(2026, 9, 21)
        assert _can_ever_enter(_mkt_at("2026-09-21T07:00:00Z", self._CLOSE), monday) is True
        assert _can_ever_enter(_mkt_at("2026-09-21T12:00:00Z", self._CLOSE), monday) is False

    def test_an_opening_before_start_date_is_unaffected(self):
        m = _mkt_at("2026-09-10T15:00:00Z", self._CLOSE)
        assert _can_ever_enter(m, self._START) is True

    def test_the_checkpoint_is_the_scans_shared_definition(self, monkeypatch):
        # The prefilter reads the checkpoint through the one helper
        # _monday_timestamps builds the scan from, so moving it moves both.
        monday = date(2026, 9, 21)
        assert backtester._monday_timestamps(monday, monday) == [
            int(backtester._checkpoint_datetime(monday).timestamp())]
        assert backtester._checkpoint_datetime(monday) == datetime(2026, 9, 21, 9, tzinfo=UTC)
        m = _mkt_at("2026-09-21T09:30:00Z", self._CLOSE)
        assert _can_ever_enter(m, self._START) is False
        monkeypatch.setattr(backtester, "_checkpoint_datetime",
                            lambda d: datetime(d.year, d.month, d.day, 10, tzinfo=UTC))
        assert _can_ever_enter(m, self._START) is True


def _p5_grid() -> list[tuple[dict, date]]:
    """Every (market, start_date) the grid tests sweep: openings every 30
    minutes over three weeks from Monday 2026-09-14 plus one second either
    side of each Monday's checkpoint, rendered in four UTC offsets in turn
    (every seventh also naive), against eight close dates and seven start
    dates (one of each weekday)."""
    zones = [UTC, timezone(timedelta(hours=-5)), timezone(timedelta(hours=14)),
             timezone(timedelta(hours=5, minutes=30))]
    base = datetime(2026, 9, 14, tzinfo=UTC)
    instants = [base + timedelta(minutes=30 * i) for i in range(21 * 48)]
    for week in range(3):
        cp = base + timedelta(weeks=week, hours=9)
        instants += [cp - timedelta(seconds=1), cp, cp + timedelta(seconds=1)]
    opens: list[datetime] = []
    for i, t in enumerate(instants):
        opens.append(t.astimezone(zones[i % len(zones)]))
        if i % 7 == 0:
            opens.append(t.replace(tzinfo=None))   # naive, same wall clock
    starts = [date(2026, 9, 10) + timedelta(days=k) for k in range(7)]
    grid = []
    for o in opens:
        od = o.date()
        for k in (0, 1, 2, 6, 7, 8, 9, 14):
            close = datetime(od.year, od.month, od.day, 12, tzinfo=UTC) + timedelta(days=k)
            m = _mkt_at(o.isoformat(), close.isoformat())
            grid.extend((m, s) for s in starts)
    return grid


class TestCanEverEnterMatchesTheScan:
    """The predicate is EXACTLY what the scan could ever reach — no looser
    (the M8 slack) and no tighter (a dropped market that could enter would
    move a result) — and it never admits a market the pre-P5 date test
    dropped, so a corpus assembled under it is a subset of an old one."""

    def test_the_predicate_is_exactly_what_the_scan_can_reach(self):
        grid = _p5_grid()
        assert len(grid) > 50_000
        mismatches = [(m, s) for m, s in grid if _can_ever_enter(m, s) != _scan_reachable(m, s)]
        assert mismatches == []

    def test_it_never_admits_what_the_date_test_dropped(self):
        grid = _p5_grid()
        new = [_can_ever_enter(m, s) for m, s in grid]
        old = [_date_granular_can_ever_enter(m, s) for m, s in grid]
        assert not any(n and not o for n, o in zip(new, old, strict=True))
        # ... and it is a real tightening on this grid, not a relabelling
        assert sum(o and not n for n, o in zip(new, old, strict=True)) > 1_000
        assert sum(n for n in new) > 1_000


_P5_START = date(2026, 1, 1)                       # a Thursday
_P5_MONDAYS = [date(2026, 1, 5), date(2026, 1, 12), date(2026, 1, 19)]
# Seconds from a Monday's 09:00 UTC checkpoint: the boundary on both sides,
# and openings far enough either way to exercise the next Monday.
_P5_OFFSETS = [-3 * 86_400, -5 * _P5_HOUR, -_P5_HOUR, -1_800, -1, 0, 1, 1_800,
               _P5_HOUR, 5 * _P5_HOUR, 14 * _P5_HOUR + 3_540]


def _p5_candles(open_dt: datetime, close_dt: datetime,
                yes_ask: float, no_ask: float) -> list[dict]:
    """A candle series obeying the measured rule: hourly for a day and a half
    from the first possible candle, then every six hours, plus every Monday
    checkpoint the market is open over — never a candle ending at or before
    the start of its opening hour."""
    first = _first_candle_ts(open_dt)
    end = int(close_dt.timestamp())
    stamps = set(range(first, min(end, first + 36 * _P5_HOUR) + 1, _P5_HOUR))
    stamps.update(range(first, end + 1, 6 * _P5_HOUR))
    stamps.update(c for c in backtester._monday_timestamps(_P5_START, close_dt.date())
                  if first <= c <= end)
    return [_candle(ts, yes_ask, no_ask) for ts in sorted(stamps)]


def _p5_corpus(seed: int) -> tuple[list[dict], dict[str, list[dict]]]:
    """Same-title, cross-event time-series and same-event ladder families
    whose legs open around a Monday checkpoint (both sides of the boundary,
    and far enough either way to reach the next Monday), plus singletons,
    with candles obeying the measured rule at prices every such pair clears
    the gates at. The ladder rungs both close at one instant, so only their
    STATED deadlines order and tier them (and only with ladders on). The two
    legs of a same-title family close at one instant too — the later of their
    two close draws — which the same-title close gate (DR-74) requires; the
    draws themselves are unchanged, so every other record of the corpus is
    the one it always was."""
    rng = random.Random(seed)
    markets: list[dict] = []
    candles: dict[str, list[dict]] = {}

    def opening() -> datetime:
        monday = rng.choice(_P5_MONDAYS)
        off = rng.choice(_P5_OFFSETS)
        if off not in (-1, 0, 1):
            off += rng.randrange(0, 1_800)
        return backtester._checkpoint_datetime(monday) + timedelta(seconds=off)

    def add(ticker, event_ticker, event_title, title, open_dt, close_dt, yes, no):
        close = close_dt.isoformat()
        markets.append({"ticker": ticker, "event_ticker": event_ticker,
                        "event_title": event_title, "title": title, "subtitle": "",
                        "result": "yes", "open_time": open_dt.isoformat(),
                        "close_time": close, "settlement_ts": close})
        candles[ticker] = _p5_candles(open_dt, close_dt, yes, no)

    for i in range(40):   # same-title: one question on two series
        legs = []
        for leg, yes, no in (("A", 0.70, 0.32), ("B", 0.55, 0.47)):
            o = opening()
            close = (datetime(o.year, o.month, o.day, 12, tzinfo=UTC)
                     + timedelta(days=rng.choice((1, 2, 3, 6, 8, 9, 13))))
            legs.append((leg, yes, no, o, close))
        # RE-PINNED (DR-74): one close for both legs, the later of the two
        # draws, so the pair passes the same-title close gate. The draws are
        # made in the same order as before, and each leg's candles are built
        # from the shared close.
        shared_close = max(close for *_, close in legs)
        for leg, yes, no, o, _close in legs:
            add(f"S{i}{leg}", f"SER{leg}{i}-1", f"ST{i}", "Q", o, shared_close, yes, no)
    for j in range(40):   # time-series: two cumulative deadlines, 14 days apart
        for leg, deadline, close, yes, no in (
            ("A", "January 20", datetime(2026, 1, 20, tzinfo=UTC), 0.30, 0.72),
            ("B", "February 3", datetime(2026, 2, 3, tzinfo=UTC), 0.60, 0.42),
        ):
            add(f"R{j}{leg}", f"RAIN{leg}{j}-1", f"RAIN{j}",
                f"Rain falls by {deadline}, 2026", opening(), close, yes, no)
    for j in range(30):   # same-event ladders: two rungs of ONE event (DR-73)
        # Both rungs close at one instant, early enough on some families that
        # only the first Monday is admissible.
        close = rng.choice((datetime(2026, 1, 7, tzinfo=UTC), datetime(2026, 1, 14, tzinfo=UTC),
                            datetime(2026, 2, 3, tzinfo=UTC)))
        for leg, deadline, yes, no in (("A", "January 20", 0.30, 0.72),
                                       ("B", "February 3", 0.60, 0.42)):
            add(f"L{j}{leg}", f"LAD{j}-1", f"LADDER{j}",
                f"Will it launch by {deadline}, 2026?", opening(), close, yes, no)
    for k in range(30):   # singletons: counted by the census, never grouped
        o = opening()
        add(f"N{k}", f"NOISE{k}-1", f"N{k}", f"Unique question {k}", o,
            o + timedelta(days=rng.choice((1, 4, 9))), 0.5, 0.5)
    return markets, candles


class TestCheckpointPrefilterIsResultNeutral:
    """The tightened prefilter drops only markets _find_entry can never enter,
    so it removes pairs that could never enter and nothing else: the pairs it
    keeps are exactly the old ones minus those with a dropped leg (nothing is
    added, and each group's own pairs keep their order), every one enters
    exactly as before, every pair it drops entered nowhere, and a run's
    trades and equity curve are identical. What it CAN move is the order of
    whole groups: a group whose first eligible member is dropped first
    appears later in the eligible stream, so its pairs move later in the list
    (test_a_group_whose_first_member_is_dropped_moves_later; measured on the
    real 2026-05-01 corpus: 826 of 2,054 pairs kept, 9 order inversions, the
    same 3 trades). Run through the real _prepare_candidates /
    _entries_for_band / run_backtest with the fetch and candle seams mocked;
    the OLD pipeline is the same code with the pre-P5 predicate patched in."""

    @staticmethod
    def _patch(monkeypatch, seed, old):
        markets, candles = _p5_corpus(seed)
        requested: list[str] = []

        def fetch_candles(_c, ticker, *a, **k):
            requested.append(ticker)
            return candles[ticker]

        monkeypatch.setattr(backtester, "fetch_all_settled_markets", lambda *a, **k: markets)
        monkeypatch.setattr(backtester, "fetch_candlesticks", fetch_candles)
        if old:
            monkeypatch.setattr(backtester, "_can_ever_enter", _date_granular_can_ever_enter)
        return markets, requested

    def _candidates(self, monkeypatch, seed, old, ladders=None):
        with monkeypatch.context() as mp:
            markets, requested = self._patch(mp, seed, old)
            c = backtester._prepare_candidates(
                MagicMock(), MagicMock(), _P5_START, True, None,
                same_event_ladders=ladders)
        return markets, requested, c

    @staticmethod
    def _key(item):
        (mA, mB, _canon, _group_key), pair_type = item
        return pair_type, mA["ticker"], mB["ticker"]

    @staticmethod
    def _every_entry(c) -> dict:
        """_find_entry for EVERY candidate pair, None included."""
        out = {}
        for item in c.all_pairs:
            (mA, mB, _canon, _group_key), pair_type = item
            out[TestCheckpointPrefilterIsResultNeutral._key(item)] = _find_entry(
                c.candles_by_ticker[mA["ticker"]], c.candles_by_ticker[mB["ticker"]],
                mA, mB, pair_type, c.start_date,
                max_horizon_days=c.max_horizon_days,
                same_event_ladders=c.same_event_ladders)
        return out

    @pytest.mark.parametrize("ladders", [False, True])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_it_drops_only_pairs_that_never_enter(self, monkeypatch, seed, ladders):
        markets, req_old, old = self._candidates(monkeypatch, seed, True, ladders)
        _, req_new, new = self._candidates(monkeypatch, seed, False, ladders)
        if ladders:
            # the ladder sub-pass contributes pairs of its own, and loses some
            ladder_old = [k for k in map(self._key, old.all_pairs) if k[1].startswith("L")]
            ladder_new = [k for k in map(self._key, new.all_pairs) if k[1].startswith("L")]
            assert ladder_new and len(ladder_new) < len(ladder_old)
        kept = {m["ticker"] for m in markets if _can_ever_enter(m, _P5_START)}

        old_keys = [self._key(i) for i in old.all_pairs]
        new_keys = [self._key(i) for i in new.all_pairs]
        # The new candidate list is the old one minus every pair with a
        # dropped leg, nothing added. On THIS corpus it is also in the old
        # order, because every family has exactly two members, so a surviving
        # group always keeps its first member. A larger group can lose its
        # first member and move later in the list; see
        # test_a_group_whose_first_member_is_dropped_moves_later.
        assert new_keys == [k for k in old_keys if k[1] in kept and k[2] in kept]
        new_set = set(new_keys)
        dropped = [k for k in old_keys if k not in new_set]
        assert dropped   # the tightening actually bites on this corpus

        old_entries = self._every_entry(old)
        new_entries = self._every_entry(new)
        # Every pair the tightening removed could never have entered ...
        assert all(old_entries[k] is None for k in dropped)
        # ... and every pair it kept enters exactly as before.
        assert new_entries == {k: old_entries[k] for k in new_keys}
        assert TestPrepareEntriesGolden._rows(backtester._entries_for_band(new)) == \
            TestPrepareEntriesGolden._rows(backtester._entries_for_band(old))
        assert sum(e is not None for e in new_entries.values()) >= 10

        # What the change does move: the eligible census shrinks by exactly
        # the dropped markets, and fewer tickers are fetched.
        n_dropped = sum(1 for m in markets
                        if _date_granular_can_ever_enter(m, _P5_START) and m["ticker"] not in kept)
        assert n_dropped > 0
        assert new.label_coverage.total == old.label_coverage.total - n_dropped
        assert set(req_new) < set(req_old)

    def test_the_boundary_is_exercised_on_both_sides(self, monkeypatch):
        # Non-vacuity on seed 0: a leg that opened in the hour before a
        # checkpoint enters AT that checkpoint, and every market that opened
        # AT a checkpoint and closes before the next Monday is admissible is
        # dropped — the two sides of the boundary, both present in the corpus.
        markets, _req, new = self._candidates(monkeypatch, 0, old=False)
        by_ticker = {m["ticker"]: m for m in markets}
        entered_at_the_edge = False
        for key, e in self._every_entry(new).items():
            if e is None:
                continue
            cp = backtester._checkpoint_datetime(e["entry_date"])
            for t in key[1:]:
                opened = datetime.fromisoformat(by_ticker[t]["open_time"])
                entered_at_the_edge |= cp - timedelta(hours=1) <= opened < cp
        assert entered_at_the_edge
        on_the_checkpoint = [
            m for m in markets
            if datetime.fromisoformat(m["open_time"]) in
            {backtester._checkpoint_datetime(d) for d in _P5_MONDAYS}]
        assert on_the_checkpoint
        assert not any(_can_ever_enter(m, _P5_START) for m in on_the_checkpoint
                       if datetime.fromisoformat(m["close_time"]).date()
                       - datetime.fromisoformat(m["open_time"]).date() < timedelta(days=8))

    def test_a_group_whose_first_member_is_dropped_moves_later(self, monkeypatch):
        # A same-title group on THREE series whose first member (in corpus
        # order) opened an hour AFTER the only admissible checkpoint: the old
        # date test keeps it, the checkpoint test drops it, and the other two
        # members still pair. A two-member group sits between them in the
        # corpus. Groups are keyed in first-appearance order, so the first
        # group now first appears AFTER the second. The pair SET is the old
        # one minus the dropped leg's pairs, each group's pairs keep their own
        # order, and the two groups swap places in the candidate list. Nothing
        # enters differently. The two groups' pairs are priced identically,
        # so they tie exactly in _simulate_at_discount's sort, and the LISTING
        # order of their simultaneous trades may follow the list. Cash does
        # not bind here, so every trade and the equity curve are unchanged.
        cp = backtester._checkpoint_datetime(_P5_MONDAYS[0])
        before, after = cp - timedelta(days=2), cp + timedelta(hours=1)
        close = cp + timedelta(days=4)          # before the next Monday
        markets: list[dict] = []
        candles: dict[str, list[dict]] = {}
        for ticker, series, group, opened, yes, no in (
            ("X1", "SXA", "GX", after, 0.70, 0.32),
            ("Y1", "SYA", "GY", before, 0.70, 0.32),
            ("Y2", "SYB", "GY", before, 0.55, 0.47),
            ("X2", "SXB", "GX", before, 0.70, 0.32),
            ("X3", "SXC", "GX", before, 0.55, 0.47),
        ):
            markets.append({"ticker": ticker, "event_ticker": f"{series}-1",
                            "event_title": group, "title": "Q", "subtitle": "",
                            "result": "yes", "open_time": opened.isoformat(),
                            "close_time": close.isoformat(),
                            "settlement_ts": close.isoformat()})
            candles[ticker] = _p5_candles(opened, close, yes, no)
        assert _date_granular_can_ever_enter(markets[0], _P5_START)
        assert not _can_ever_enter(markets[0], _P5_START)

        def run(old: bool):
            with monkeypatch.context() as mp:
                mp.setattr(backtester, "fetch_all_settled_markets",
                           lambda *a, **k: markets)
                mp.setattr(backtester, "fetch_candlesticks",
                           lambda _c, ticker, *a, **k: candles[ticker])
                if old:
                    mp.setattr(backtester, "_can_ever_enter",
                               _date_granular_can_ever_enter)
                c = backtester._prepare_candidates(
                    MagicMock(), MagicMock(), _P5_START, True, None)
                trades, eq = run_backtest(
                    hist_client=MagicMock(), live_client=MagicMock(),
                    start_date=_P5_START, initial_balance=10_000.0)
            return c, trades, eq

        old, trades_old, eq_old = run(True)
        new, trades_new, eq_new = run(False)
        old_keys = [self._key(i) for i in old.all_pairs]
        new_keys = [self._key(i) for i in new.all_pairs]
        survivors = [k for k in old_keys if "X1" not in k[1:]]
        assert set(new_keys) == set(survivors)          # nothing added
        assert new_keys != survivors                    # ... but reordered
        assert [k for k in new_keys if k[1].startswith("Y")] == \
            [k for k in survivors if k[1].startswith("Y")]
        assert [k for k in new_keys if k[1].startswith("X")] == \
            [k for k in survivors if k[1].startswith("X")]
        assert new_keys.index(survivors[-1]) < new_keys.index(survivors[0])

        old_entries = self._every_entry(old)
        new_entries = self._every_entry(new)
        assert all(old_entries[k] is None for k in old_keys if k not in set(new_keys))
        assert new_entries == {k: old_entries[k] for k in new_keys}
        assert sum(e is not None for e in new_entries.values()) == 2

        assert len(trades_new) == 2
        assert sorted(map(astuple, trades_new)) == sorted(map(astuple, trades_old))
        pd.testing.assert_frame_equal(eq_new, eq_old)

    @pytest.mark.parametrize("seed", [0, 1])
    def test_run_backtest_is_unchanged(self, monkeypatch, seed):
        results = {}
        for old in (True, False):
            with monkeypatch.context() as mp:
                self._patch(mp, seed, old)
                results[old] = run_backtest(
                    hist_client=MagicMock(), live_client=MagicMock(),
                    start_date=_P5_START, initial_balance=10_000.0)
        (trades_old, eq_old), (trades_new, eq_new) = results[True], results[False]
        assert trades_old
        assert [astuple(t) for t in trades_new] == [astuple(t) for t in trades_old]
        pd.testing.assert_frame_equal(eq_new, eq_old)


def _ts_member(ticker: str, event_ticker: str, close_d: date | None) -> dict:
    """Minimal time-series group member for _extract_pairs windowing tests.

    The title names this member's OWN close date as a cumulative deadline, the
    shape a real two-deadline family has. That is load-bearing, not decoration:
    _extract_pairs refuses any pair whose wording does not state two different
    "by <date>" deadlines, so members with no wording would make every
    windowing test below assert emptiness against emptiness — the performance
    smoke test would fail loudly, but the oracle-equivalence tests would go
    VACUOUS, which is worse. All titles still normalize to "q " so the members
    stay in ONE group, which is what these tests are about.
    """
    m = {"ticker": ticker, "event_ticker": event_ticker}
    if close_d is not None:
        m["close_time"] = datetime(close_d.year, close_d.month, close_d.day,
                                    tzinfo=UTC).isoformat()
        m["title"] = f"Q by {close_d:%B %d, %Y}"
    return m


def _naive_series(event_ticker: object) -> str:
    """Oracle-local restatement of the series identity the one-series rule
    compares on: the prefix before the first hyphen, stripped and upper-cased,
    with every KXMVE* combo prefix collapsed onto the one family (DR-55).

    The family collapse reads config.MVE_SERIES_FAMILY_PREFIX — a CONSTANT,
    not the implementation — so the oracle still cannot inherit a bug from
    scanner.event_series while staying in step with the rule it claims to
    apply. Omitting the collapse would silently diverge from _extract_pairs on
    any KXMVE fixture (two literal prefixes, one series), which is the
    "oracle replays the old rule" failure mode this file's parity tests exist
    to avoid.
    """
    if not isinstance(event_ticker, str):
        return ""
    prefix = event_ticker.split("-", 1)[0].strip().upper()
    return (MVE_SERIES_FAMILY_PREFIX if prefix.startswith(MVE_SERIES_FAMILY_PREFIX)
            else prefix)


def _naive_cumulative_deadline(m: dict) -> str | None:
    """Oracle-local restatement of the cumulative-deadline spans a member's
    wording states, or None when it states none.

    Deliberately NOT scanner.deadline_profile: an oracle that reuses the
    implementation cannot falsify it. _ts_member builds exactly one shape —
    "Q by <Month> <day>, <year>" — so the oracle only has to recognise that
    shape, and any drift between it and the real tables shows up as a
    disagreement rather than being silently inherited.
    """
    title = m.get("title") or ""
    match = re.search(
        r"\bby\s+(?:January|February|March|April|May|June|July|August|September"
        r"|October|November|December)\s+\d{1,2},\s+\d{4}",
        title, re.IGNORECASE,
    )
    return match.group(0).lower() if match else None


def _naive_stated_deadline(m: dict) -> date | None:
    """Oracle-local restatement of the calendar day a member's deadline names.

    Deliberately NOT scanner.stated_deadline, for the same reason
    _naive_cumulative_deadline is not scanner.deadline_profile: an oracle that
    reuses the implementation cannot falsify it. _ts_member builds exactly one
    shape — "Q by <Month> <day>, <year>" — and "by" INCLUDES the named day, so
    the last included day is that date itself.
    """
    span = _naive_cumulative_deadline(m)
    if span is None:
        return None
    try:
        return datetime.strptime(span, "by %B %d, %Y").date()
    except ValueError:
        return None


def _naive_time_series_pairs(members: list[dict], margin_days: int,
                             same_event_ladders: bool = False) -> set[frozenset]:
    """Independent oracle: naive O(n^2) double loop over the same group,
    filtering by the same margin-inclusive close-time gap, the same
    event_ticker rule, the same one-series rule (DR-02/DR-54/DR-55), the same
    cumulative-deadline rule AND, when same_event_ladders is set, the same
    same-event deadline-ladder rule (DR-73) that _extract_pairs applies, but
    without any sorting/windowing. Written standalone (no backtester internals
    besides plain dict/date arithmetic, _naive_series' restatement of the
    series identity, _naive_cumulative_deadline's of the deadline spans and
    _naive_stated_deadline's of the calendar day one names) so it can serve as
    ground truth for the windowed implementation.

    The one-series conjunct is spelled out here rather than imported, for the
    same reason the rest is: an oracle that reuses the implementation cannot
    falsify it. It is not dead weight on the current fixtures only by
    accident — _build_synthetic_group's members carry no wording keys, so the
    wording half is True for every pair in them and the distinct hyphen-less
    event tickers are the only thing keeping the conjunct from firing. Add one
    hyphenated shared-prefix ticker to that fixture and an oracle without this
    clause diverges silently, which is exactly the "oracle replays the old
    rule" failure CLAUDE.md records for the archive-walk parity tests.

    The ladder clause must move in LOCKSTEP with _extract_pairs' sub-pass or
    test_matches_naive_oracle_exactly fails: it is the whole point of an
    oracle that it restates the rule rather than inheriting it. Note it
    deliberately skips the close-time margin — the sub-pass is unwindowed,
    because a ladder is capped on its STATED gap.
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
            if a["event_ticker"] == b["event_ticker"]:
                # DR-73: two rungs of ONE event, admitted only with ladders
                # on, ordered and capped on their STATED deadlines and NOT
                # subject to the close-time margin above.
                if not (same_event_ladders and a["event_ticker"]):
                    continue
                if ((a.get("title") or "", a.get("subtitle") or "",
                     a.get("event_title") or "")
                        == (b.get("title") or "", b.get("subtitle") or "",
                            b.get("event_title") or "")):
                    continue
                sda = _naive_stated_deadline(a)
                sdb = _naive_stated_deadline(b)
                if sda is None or sdb is None or sda == sdb:
                    continue
                if abs((sdb - sda).days) > MAX_DEADLINE_GAP_DAYS:
                    continue
                result.add(frozenset([a["ticker"], b["ticker"]]))
                continue
            if abs((db - da).days) > margin_days:
                continue
            # Both legs must state a cumulative deadline, and two DIFFERENT
            # ones — the restatement of _extract_pairs' cumulative-deadline
            # conjunct.
            deadline_a = _naive_cumulative_deadline(a)
            deadline_b = _naive_cumulative_deadline(b)
            if deadline_a is None or deadline_b is None or deadline_a == deadline_b:
                continue
            if ((a.get("title") or "", a.get("subtitle") or "",
                 a.get("event_title") or "")
                    == (b.get("title") or "", b.get("subtitle") or "",
                        b.get("event_title") or "")):
                sa = _naive_series(a.get("event_ticker"))
                sb = _naive_series(b.get("event_ticker"))
                if not sa or not sb or sa == sb:
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
        # The switch-OFF pin for DR-73: SAMEEVT-1/-2 are a genuine two-rung
        # ladder (one event, dated cumulative titles one day apart), so this
        # is effective rather than vacuous — the row below admits exactly this
        # pair with the switch on, which is what proves it.
        members = self._build_synthetic_group()
        windowed = _extract_pairs({"synthetic": members})
        pair_tickers = {frozenset([a["ticker"], b["ticker"]]) for a, b, _, _ in windowed}
        assert frozenset(["SAMEEVT-1", "SAMEEVT-2"]) not in pair_tickers

    def test_matches_naive_oracle_exactly_with_ladders_on(self):
        # DR-73: the oracle carries the ladder rule too, so the sub-pass is
        # checked against an independent restatement and not only against its
        # own switch-off behaviour.
        members = self._build_synthetic_group()
        windowed = _extract_pairs({"synthetic": members}, same_event_ladders=True)
        windowed_set = {frozenset([a["ticker"], b["ticker"]]) for a, b, _, _ in windowed}
        naive_set = _naive_time_series_pairs(members, MAX_DEADLINE_GAP_DAYS + 1,
                                             same_event_ladders=True)
        assert windowed_set == naive_set
        # The switch-on set is exactly the switch-off set plus the one ladder,
        # so this row cannot pass by both sides being wrong the same way.
        off_set = {frozenset([a["ticker"], b["ticker"]])
                   for a, b, _, _ in _extract_pairs({"synthetic": members})}
        assert windowed_set - off_set == {frozenset(["SAMEEVT-1", "SAMEEVT-2"])}

    def test_missing_close_time_member_produces_no_pairs(self):
        members = self._build_synthetic_group()
        windowed = _extract_pairs({"synthetic": members})
        for a, b, _, _ in windowed:
            assert a["ticker"] != "NOCLOSE"
            assert b["ticker"] != "NOCLOSE"


_BT_LADDER_TITLE = "Will SpaceX launch another Starship %s?"


def _ladder_member(ticker, deadline_text, *, close, event="KXSTARSHIP-14",
                   event_title="", subtitle=""):
    """One rung of a same-event cumulative deadline ladder, in cache-dict form.

    The dict-world mirror of tests/test_scanner.py's _ladder_rung, down to the
    title template, so the two paths' DR-73 cases can be read side by side.
    Every rung of one ladder shares an event_ticker and a title that differs
    ONLY in its deadline, so _group_by_normalized_title collapses them onto one
    key — the shape KXSPACEXSTARSHIP-14 has live.
    """
    return {
        "ticker": ticker, "event_ticker": event, "event_title": event_title,
        "title": _BT_LADDER_TITLE % deadline_text, "subtitle": subtitle,
        "close_time": close.isoformat(),
    }


def _assert_one_ladder_group_dicts(mA, mB):
    """Every ladder fixture must actually BE a ladder before its rule is tested.

    Two rungs the group key separates, or one the wording screen does not call
    cumulative, would make a test pass for a reason that has nothing to do
    with DR-73 — the way test_same_event_ticker_never_pairs went vacuous on
    the live path.
    """
    groups = _group_by_normalized_title([mA, mB])
    assert len(groups) == 1, groups
    assert backtester._deadline_profile_dict(mA)[0] == scanner.DEADLINE_CUMULATIVE
    assert backtester._deadline_profile_dict(mB)[0] == scanner.DEADLINE_CUMULATIVE
    assert mA["event_ticker"] == mB["event_ticker"]


class TestExtractPairsSameEventLadders:
    """DR-73c: backtester._extract_pairs mirrors the live ladder branch.

    The dict-world mirror of tests/test_scanner.py::TestSameEventDeadlineLadders.
    Both paths must apply the rule or a ladder-enabled live run would be
    measured by a backtest that still refuses every same-event pair — the
    both-paths requirement the wording and one-series rules already carry.
    """

    def _extract(self, members, *, on=True):
        return _extract_pairs({"ladder": members}, same_event_ladders=on)

    def _two_rungs(self, *, early="by March 1, 2026", late="by March 20, 2026",
                   close_a=datetime(2026, 3, 1, tzinfo=UTC),
                   close_b=datetime(2026, 3, 20, tzinfo=UTC),
                   event_title=""):
        mA = _ladder_member("RUNG-EARLY", early, close=close_a, event_title=event_title)
        mB = _ladder_member("RUNG-LATE", late, close=close_b, event_title=event_title)
        _assert_one_ladder_group_dicts(mA, mB)
        return mA, mB

    # ── the admitted case, and the switch ───────────────────────────────────

    def test_two_dated_rungs_of_one_event_pair(self):
        mA, mB = self._two_rungs()
        pairs = self._extract([mA, mB])
        assert len(pairs) == 1
        assert (pairs[0][0]["ticker"], pairs[0][1]["ticker"]) == ("RUNG-EARLY", "RUNG-LATE")

    def test_the_same_fixture_is_refused_with_the_switch_off(self):
        # control: the ONLY thing standing between this fixture and a pair is
        # the switch.
        mA, mB = self._two_rungs()
        assert self._extract([mA, mB], on=False) == []

    def test_the_flag_resolves_true_false_and_none(self, monkeypatch):
        # Resolved at CALL time: None must read the constant as it stands
        # NOW, so a monkeypatched MODULE constant and a run-level override
        # both take effect. A def-time default would freeze the import-time
        # value and silently ignore both. The name resolved is
        # backtester.TIME_SERIES_SAME_EVENT_LADDERS — this module's by-value
        # binding — which is why the patch below targets `backtester` and not
        # `config`, whose attribute this module never reads.
        mA, mB = self._two_rungs()
        monkeypatch.setattr(backtester, "TIME_SERIES_SAME_EVENT_LADDERS", True)
        assert len(_extract_pairs({"ladder": [mA, mB]})) == 1
        assert len(_extract_pairs({"ladder": [mA, mB]}, same_event_ladders=None)) == 1
        # An explicit False overrides a switched-ON config, and vice versa.
        assert _extract_pairs({"ladder": [mA, mB]}, same_event_ladders=False) == []
        monkeypatch.setattr(backtester, "TIME_SERIES_SAME_EVENT_LADDERS", False)
        assert _extract_pairs({"ladder": [mA, mB]}, same_event_ladders=None) == []
        assert len(_extract_pairs({"ladder": [mA, mB]}, same_event_ladders=True)) == 1

    def test_the_candidate_count_is_always_logged_while_on(self, caplog):
        mA, mB = self._two_rungs()
        with caplog.at_level(logging.INFO):
            self._extract([mA, mB])
        # DR-66: a switch that produces nothing must be distinguishable from a
        # broken rule, so the count is logged at zero too.
        assert "Same-event ladder candidates among the time-series candidates: 1" \
            in caplog.text
        # The ZERO case is the whole DR-66 property and is pinned separately —
        # asserting only the `1` above leaves `if ladders_on and
        # saw_time_series_group and ladder_pairs:` alive, which silences the
        # line on exactly the run an operator most needs it on. Two rungs of
        # one event with IDENTICAL wording: a real time-series group (so the
        # `saw_time_series_group` gate opens), refused by the ladder rule (so
        # the count is 0).
        same = "by March 1, 2026"
        z1 = _ladder_member("Z1", same, close=datetime(2026, 3, 1, tzinfo=UTC))
        z2 = _ladder_member("Z2", same, close=datetime(2026, 3, 20, tzinfo=UTC))
        _assert_one_ladder_group_dicts(z1, z2)
        caplog.clear()
        with caplog.at_level(logging.INFO):
            assert self._extract([z1, z2]) == []
        assert "Same-event ladder candidates among the time-series candidates: 0" \
            in caplog.text

    def test_nothing_is_logged_with_the_switch_off(self, caplog):
        mA, mB = self._two_rungs()
        with caplog.at_level(logging.INFO):
            self._extract([mA, mB], on=False)
        assert "Same-event ladder" not in caplog.text

    # ── ordering: the stated deadline, never close_time ─────────────────────

    def test_legs_are_ordered_by_stated_deadline_not_close_time(self):
        # The Mar 1 rung closes a MONTH after the Mar 20 one, so the close_time
        # sort hands this candidate to the sub-pass the wrong way round and
        # only the stated-deadline swap can fix it.
        mA, mB = self._two_rungs(
            close_a=datetime(2026, 4, 1, tzinfo=UTC),
            close_b=datetime(2026, 3, 5, tzinfo=UTC),
        )
        pairs = self._extract([mA, mB])
        assert len(pairs) == 1
        assert pairs[0][0]["ticker"] == "RUNG-EARLY"

    def test_rungs_that_close_at_one_instant_still_pair(self):
        # A settled event closes every rung at once — 681 of 1,821 dated
        # same-event pairs in the archive have a close gap of ZERO days.
        one_instant = datetime(2026, 3, 20, tzinfo=UTC)
        mA, mB = self._two_rungs(close_a=one_instant, close_b=one_instant)
        pairs = self._extract([mA, mB])
        assert len(pairs) == 1
        assert (pairs[0][0]["ticker"], pairs[0][1]["ticker"]) == ("RUNG-EARLY", "RUNG-LATE")

    # ── the sub-pass is UNWINDOWED ──────────────────────────────────────────

    def test_a_35_day_close_gap_with_a_20_day_stated_gap_is_still_formed(self):
        # The cross-event sweep's MAX_DEADLINE_GAP_DAYS + 1 window would drop
        # this pair before ever looking at it, which is exactly why the ladder
        # sub-pass is separate and unwindowed: 463 archive pairs sit the other
        # way round (close gap inside the cap, stated gap beyond it) and 1 sits
        # this way, and a windowed sub-pass would silently lose it.
        mA, mB = self._two_rungs(
            early="by March 1, 2026", late="by March 21, 2026",
            close_a=datetime(2026, 3, 1, tzinfo=UTC),
            close_b=datetime(2026, 4, 5, tzinfo=UTC),
        )
        close_gap = (_parse_iso_date(mB["close_time"]) - _parse_iso_date(mA["close_time"])).days
        assert close_gap == 35 > MAX_DEADLINE_GAP_DAYS + 1
        pairs = self._extract([mA, mB])
        assert len(pairs) == 1
        assert (pairs[0][0]["ticker"], pairs[0][1]["ticker"]) == ("RUNG-EARLY", "RUNG-LATE")

    # ── the gap cap, measured on the stated gap ─────────────────────────────

    def test_a_31_day_stated_gap_is_refused_although_the_closes_are_30_apart(self, caplog):
        mA, mB = self._two_rungs(
            early="by March 1, 2026", late="by April 1, 2026",
            close_a=datetime(2026, 3, 1, tzinfo=UTC),
            close_b=datetime(2026, 3, 31, tzinfo=UTC),
        )
        with caplog.at_level(logging.INFO):
            assert self._extract([mA, mB]) == []
        assert "refused at the 30-day STATED gap cap" in caplog.text

    def test_a_30_day_stated_gap_is_admitted(self):
        # control for the row above: one day narrower and the same fixture pairs.
        mA, mB = self._two_rungs(
            early="by March 1, 2026", late="by March 31, 2026",
            close_a=datetime(2026, 3, 1, tzinfo=UTC),
            close_b=datetime(2026, 3, 31, tzinfo=UTC),
        )
        assert len(self._extract([mA, mB])) == 1

    # ── the refusals, each with its control ─────────────────────────────────

    def test_a_rung_with_no_readable_year_is_refused(self, caplog):
        mA, mB = self._two_rungs(early="by November 4", late="by December 4")
        with caplog.at_level(logging.INFO):
            assert self._extract([mA, mB]) == []
        assert "states no placeable calendar day" in caplog.text

    def test_the_same_rungs_dated_are_admitted(self):
        # control: the refusal above is the missing YEAR, not the wording shape.
        mA, mB = self._two_rungs(early="by November 4, 2026", late="by December 4, 2026",
                                 close_a=datetime(2026, 11, 4, tzinfo=UTC),
                                 close_b=datetime(2026, 12, 4, tzinfo=UTC))
        assert len(self._extract([mA, mB])) == 1

    def test_two_rungs_naming_one_calendar_day_are_refused(self, caplog):
        # "by December 2026" and "by December 31, 2026" are ONE deadline
        # spelled two ways — the SAME_DAY outcome. Their spans differ, so
        # cumulative_deadline_pair admits them and only the date reader can tell.
        mA, mB = self._two_rungs(early="by December 2026", late="by December 31, 2026",
                                 close_a=datetime(2026, 12, 1, tzinfo=UTC),
                                 close_b=datetime(2026, 12, 31, tzinfo=UTC))
        assert scanner.cumulative_deadline_pair(
            backtester._deadline_profile_dict(mA),
            backtester._deadline_profile_dict(mB))
        with caplog.at_level(logging.INFO):
            assert self._extract([mA, mB]) == []
        assert "both rungs name one calendar day" in caplog.text

    def test_a_rung_whose_own_fields_disagree_is_refused(self, caplog):
        # The KXSTARSHIPFL shape: a stale event title naming an irreconcilable
        # deadline. stated_deadline's cross-check refuses the rung rather than
        # trusting the deciding field alone.
        mA, mB = self._two_rungs(
            early="by October 1, 2026", late="by October 21, 2026",
            close_a=datetime(2026, 10, 1, tzinfo=UTC),
            close_b=datetime(2026, 10, 21, tzinfo=UTC),
            event_title="Starship flights before 2026",
        )
        with caplog.at_level(logging.INFO):
            assert self._extract([mA, mB]) == []
        assert "name irreconcilable days" in caplog.text

    def test_the_same_rungs_under_a_dateless_event_title_are_admitted(self):
        # control: the ONLY difference from the row above is the stale event
        # title, so that row pins the cross-check and not the gap cap.
        mA, mB = self._two_rungs(
            early="by October 1, 2026", late="by October 21, 2026",
            close_a=datetime(2026, 10, 1, tzinfo=UTC),
            close_b=datetime(2026, 10, 21, tzinfo=UTC),
            event_title="Starship flights",
        )
        assert len(self._extract([mA, mB])) == 1

    def test_identical_wording_in_one_event_never_pairs(self, caplog):
        same = "by March 1, 2026"
        mA = _ladder_member("R1", same, close=datetime(2026, 3, 1, tzinfo=UTC))
        mB = _ladder_member("R2", same, close=datetime(2026, 3, 20, tzinfo=UTC))
        _assert_one_ladder_group_dicts(mA, mB)
        with caplog.at_level(logging.INFO):
            assert self._extract([mA, mB]) == []
        # Refused HERE, not downstream: cumulative_deadline_pair would refuse
        # it as "the same deadline", a different finding (DR-72's whole point).
        assert "the two rungs' wording is identical" in caplog.text
        assert "state the same deadline" not in caplog.text

    def test_an_empty_shared_event_ticker_never_pairs(self, caplog):
        # THREE rungs, not two, because the counter is incremented per
        # CANDIDATE PAIR (n*(n-1)//2) so it means the same thing as the live
        # finder's, which increments once per pair inside its double loop. At
        # n=2 that arithmetic and a bare `+= 1` are indistinguishable; at n=3
        # they are 3 against 1, and an operator comparing this funnel arrow
        # with the live one would be reading incomparable numbers.
        mA = _ladder_member("E1", "by March 1, 2026", event="",
                            close=datetime(2026, 3, 1, tzinfo=UTC))
        mB = _ladder_member("E2", "by March 20, 2026", event="",
                            close=datetime(2026, 3, 20, tzinfo=UTC))
        mC = _ladder_member("E3", "by April 5, 2026", event="",
                            close=datetime(2026, 4, 5, tzinfo=UTC))
        _assert_one_ladder_group_dicts(mA, mB)
        _assert_one_ladder_group_dicts(mB, mC)
        with caplog.at_level(logging.INFO):
            assert self._extract([mA, mB, mC]) == []
        assert ("the shared event ticker is empty (same-event sub-pass, before "
                "price filters): 3") in caplog.text

    def test_a_snapshot_rung_in_one_event_never_pairs(self, caplog):
        mA = {"ticker": "S1", "event_ticker": "KXSNAP-1", "event_title": "",
              "title": "Starship flights on March 1, 2026", "subtitle": "",
              "close_time": "2026-03-01T00:00:00+00:00"}
        mB = {"ticker": "S2", "event_ticker": "KXSNAP-1", "event_title": "",
              "title": "Starship flights on March 20, 2026", "subtitle": "",
              "close_time": "2026-03-20T00:00:00+00:00"}
        assert len(_group_by_normalized_title([mA, mB])) == 1
        assert backtester._deadline_profile_dict(mA)[0] == scanner.DEADLINE_SNAPSHOT
        with caplog.at_level(logging.INFO):
            assert self._extract([mA, mB]) == []
        # Refused BY the wording screen, on its own counter — without it the
        # pair still fails, but as a parser problem rather than a
        # wrong-kind-of-market one.
        assert "deciding field is snapshot wording" in caplog.text
        assert "states no placeable calendar day" not in caplog.text

    def test_a_large_same_event_bucket_warns(self, monkeypatch, caplog):
        # The sub-pass is O(B^2) and unwindowed, so the only thing standing
        # between a future corpus and a quadratic blow-up is this canary.
        # Measured maxima are 26 live and 75 on a strike-blind legacy slice, so
        # the real threshold cannot fire on anything on disk today — the
        # threshold is lowered here rather than building a 1,000-rung fixture.
        monkeypatch.setattr(backtester, "LARGE_GROUP_WARN_THRESHOLD", 2)
        rungs = [_ladder_member(f"R{i}", f"by March {i}, 2026",
                                close=datetime(2026, 3, i, tzinfo=UTC))
                 for i in range(1, 5)]
        with caplog.at_level(logging.WARNING):
            self._extract(rungs)
        assert "Same-event ladder bucket 'KXSTARSHIP-14'" in caplog.text
        assert "holds 4 dated rungs" in caplog.text

    def test_a_small_bucket_does_not_warn(self, monkeypatch, caplog):
        # control: the canary is a size threshold, not a "ladders are on" notice.
        monkeypatch.setattr(backtester, "LARGE_GROUP_WARN_THRESHOLD", 2)
        mA, mB = self._two_rungs()
        with caplog.at_level(logging.WARNING):
            self._extract([mA, mB])
        assert "Same-event ladder bucket" not in caplog.text

    # ── the cross-event sweep is untouched ──────────────────────────────────

    def test_cross_event_pairs_are_unchanged_by_the_switch(self):
        # The sweep keeps its own `continue` on equal event tickers, so the two
        # populations cannot overlap and no cross-event pair is re-ordered.
        m1 = _ladder_member("M1", "by March 1, 2026", event="EVA-1",
                            close=datetime(2026, 3, 1, tzinfo=UTC))
        m2 = _ladder_member("M2", "by March 20, 2026", event="EVB-1",
                            close=datetime(2026, 3, 20, tzinfo=UTC))
        m3 = _ladder_member("M3", "by March 15, 2026", event="EVA-1",
                            close=datetime(2026, 3, 15, tzinfo=UTC))
        off = [(a["ticker"], b["ticker"]) for a, b, _, _ in self._extract([m1, m2, m3], on=False)]
        on = [(a["ticker"], b["ticker"]) for a, b, _, _ in self._extract([m1, m2, m3])]
        assert off == [("M1", "M2"), ("M3", "M2")]
        # Every cross-event pair survives in the same order, and the ladder is
        # ADDED after them — this function keeps all pairs, so unlike the live
        # finder's one-best contest nothing is displaced here.
        assert on == off + [("M1", "M3")]


class TestPrepareEntriesThreadsTheLadderFlag:
    """DR-73c: _prepare_entries hands the SAME unresolved flag to both
    _extract_pairs and _find_entry.

    Load-bearing rather than cosmetic: if the two resolved it separately, or
    one of them never received it, a pair one rule admitted would be replayed
    under the other rule's ordering — a ladder formed on stated deadlines and
    then entered on close_time, which is the inversion DR-73 exists to stop.
    """

    @pytest.mark.parametrize("passed", [None, True, False])
    def test_both_callees_receive_the_same_sentinel(self, monkeypatch, passed):
        seen: dict = {"extract": [], "entry": []}

        def _fake_extract(groups, *, same_event_ladders=None):
            seen["extract"].append(same_event_ladders)
            return []

        def _fake_entry(*args, **kwargs):
            seen["entry"].append(kwargs.get("same_event_ladders", "MISSING"))
            return None

        monkeypatch.setattr(backtester, "fetch_all_settled_markets",
                            lambda *a, **k: [
                                {"ticker": "T1", "event_ticker": "E1", "title": "Q",
                                 "subtitle": "", "event_title": "EV", "result": "yes",
                                 "close_time": "2026-02-01T00:00:00+00:00",
                                 "settlement_ts": "2026-02-01T12:00:00+00:00"},
                            ])
        monkeypatch.setattr(backtester, "_extract_pairs", _fake_extract)
        monkeypatch.setattr(backtester, "_find_entry", _fake_entry)
        backtester._prepare_entries(MagicMock(), MagicMock(), date(2026, 1, 1),
                                    True, None, same_event_ladders=passed)
        # Both groupings go through _extract_pairs; the 3-tuple call takes the
        # flag too, where it is inert.
        assert seen["extract"] == [passed, passed]

    def test_find_entry_receives_it_too(self, monkeypatch):
        seen: list = []

        def _fake_entry(*args, **kwargs):
            seen.append(kwargs.get("same_event_ladders", "MISSING"))
            return None

        markets = [
            {"ticker": "SA", "event_ticker": "EA", "event_title": "EV",
             "title": "Q by March 1, 2026", "subtitle": "", "result": "yes",
             "close_time": "2026-03-01T00:00:00+00:00",
             "settlement_ts": "2026-03-01T12:00:00+00:00"},
            {"ticker": "SB", "event_ticker": "EB", "event_title": "EV",
             "title": "Q by March 15, 2026", "subtitle": "", "result": "yes",
             "close_time": "2026-03-15T00:00:00+00:00",
             "settlement_ts": "2026-03-15T12:00:00+00:00"},
        ]
        monkeypatch.setattr(backtester, "fetch_all_settled_markets",
                            lambda *a, **k: markets)
        monkeypatch.setattr(backtester, "fetch_candlesticks", lambda *a, **k: [])
        monkeypatch.setattr(backtester, "_find_entry", _fake_entry)
        backtester._prepare_entries(MagicMock(), MagicMock(), date(2026, 1, 1),
                                    True, None, same_event_ladders=True)
        assert seen and set(seen) == {True}


class TestFindEntrySameEventLadders:
    """DR-73c: _find_entry orders and gaps a same-event ladder on its STATED
    deadlines, gated on the same flag as _extract_pairs.

    Gated on the FLAG, never on event_ticker equality alone: _extract_pairs
    only proposes a same-event pair while the switch is on, but any other
    caller must not get ladder semantics from a switched-off tree.
    """

    def _rungs(self, *, early="by March 1, 2026", late="by March 20, 2026",
               close_a=datetime(2026, 3, 20, tzinfo=UTC),
               close_b=datetime(2026, 3, 20, tzinfo=UTC)):
        return (_ladder_member("RUNG-EARLY", early, close=close_a),
                _ladder_member("RUNG-LATE", late, close=close_b))

    @staticmethod
    def _candles(pA=0.20, pB=0.60, nA=0.80, nB=0.40):
        return ([_candle(_MONDAY_TS, pA, nA)], [_candle(_MONDAY_TS, pB, nB)])

    def _entry(self, mA, mB, *, on=True, prices=None):
        ca, cb = prices if prices is not None else self._candles()
        return _find_entry(ca, cb, mA, mB, "time_series", date(2026, 1, 1),
                           same_event_ladders=on)

    def test_rungs_closing_at_one_instant_enter_on_the_stated_gap(self):
        # close_time gives a gap of 0 here, which would pick the SHORT tier;
        # the stated deadlines are 19 days apart, which is the LONG one.
        mA, mB = self._rungs()
        entry = self._entry(mA, mB)
        assert entry is not None
        assert entry["gap_days"] == 19
        assert entry["mA"]["ticker"] == "RUNG-EARLY"
        assert entry["mB"]["ticker"] == "RUNG-LATE"

    def test_the_stated_gap_chooses_a_tier_the_close_gap_would_not(self):
        # control, and the reason the ladder gap must reach the tier: a 0.20
        # spread clears the SHORT tier a 0-day close gap selects and fails the
        # LONG tier the 19-day stated gap demands.
        mA, mB = self._rungs()
        prices = self._candles(pA=0.20, pB=0.40, nA=0.80, nB=0.60)
        assert self._entry(mA, mB, prices=prices) is None
        off = self._entry(mA, mB, on=False, prices=prices)
        assert off is not None and off["gap_days"] == 0

    def test_the_switch_off_reads_close_time_for_the_same_fixture(self):
        # The gate is the FLAG, not the shared event ticker.
        mA, mB = self._rungs()
        off = self._entry(mA, mB, on=False)
        assert off is not None and off["gap_days"] == 0

    def test_a_ladder_whose_later_rung_closed_first_is_oriented_by_deadline(self):
        # The Mar 1 rung closes a month AFTER the Mar 20 one — the
        # early-resolution shape CLAUDE.md records as a backtest residual for
        # cross-event pairs, and the normal case inside one event.
        mA, mB = self._rungs(close_a=datetime(2026, 4, 1, tzinfo=UTC),
                             close_b=datetime(2026, 3, 5, tzinfo=UTC))
        entry = self._entry(mA, mB)
        assert entry is not None
        assert entry["mA"]["ticker"] == "RUNG-EARLY"
        assert entry["gap_days"] == 19
        # control: on close_time the SAME two markets are ordered the other way
        # round and gapped at 27 days, which is what the ladder rule exists to
        # prevent. Their quotes are mirrored here only because the inverted
        # ordering needs the inverted price direction to enter at all — no
        # single quote pair can enter under both orderings, since each demands
        # the opposite sign of pB - pA.
        off = _find_entry([_candle(_MONDAY_TS, 0.60, 0.40)],
                          [_candle(_MONDAY_TS, 0.20, 0.80)],
                          mA, mB, "time_series", date(2026, 1, 1),
                          same_event_ladders=False)
        assert off is not None
        assert off["mA"]["ticker"] == "RUNG-LATE"
        assert off["gap_days"] == 27

    def test_a_reversed_pair_is_swapped_and_its_candles_move_with_it(self):
        # Passed LATE-first, and both rungs close at one instant, so close_time
        # cannot re-order them at all — only the stated-deadline swap can. The
        # swap must move candles_a/candles_b with mA/mB, or the entry reports
        # the wrong market's prices; the quotes below are distinct enough that
        # a swap of the markets alone would fail the direction test.
        early, late = self._rungs()
        entry = _find_entry([_candle(_MONDAY_TS, 0.60, 0.40)],
                            [_candle(_MONDAY_TS, 0.20, 0.80)],
                            late, early, "time_series", date(2026, 1, 1),
                            same_event_ladders=True)
        assert entry is not None
        assert entry["mA"]["ticker"] == "RUNG-EARLY"
        assert entry["mB"]["ticker"] == "RUNG-LATE"
        assert entry["gap_days"] == 19
        assert entry["pA"] == pytest.approx(0.20)
        assert entry["pB"] == pytest.approx(0.60)
        assert entry["nB"] == pytest.approx(0.40)
        # control: with the switch off nothing re-orders them, so the pair is
        # read the wrong way round and its direction test goes negative.
        assert _find_entry([_candle(_MONDAY_TS, 0.60, 0.40)],
                           [_candle(_MONDAY_TS, 0.20, 0.80)],
                           late, early, "time_series", date(2026, 1, 1),
                           same_event_ladders=False) is None

    def test_a_31_day_stated_gap_returns_none(self):
        mA, mB = self._rungs(early="by March 1, 2026", late="by April 1, 2026")
        assert self._entry(mA, mB) is None
        # control: the close gap is 0, so only the STATED cap can refuse it.
        assert self._entry(mA, mB, on=False) is not None

    def test_an_unreadable_deadline_returns_none(self):
        # Fails CLOSED: a year-less rung states no placeable day, so the pair
        # has no order and no gap and is not replayed at all.
        mA, mB = self._rungs(early="by March 1", late="by March 20")
        assert self._entry(mA, mB) is None

    def test_two_rungs_naming_one_day_return_none(self):
        # SAME_DAY, branched on with `is` — it is a non-empty string, so a
        # truthiness test would fall through and unpacking it would raise.
        mA, mB = self._rungs(early="by December 2026", late="by December 31, 2026",
                             close_a=datetime(2026, 12, 31, tzinfo=UTC),
                             close_b=datetime(2026, 12, 31, tzinfo=UTC))
        assert self._entry(mA, mB) is None

    def test_a_cross_event_pair_is_untouched_by_the_flag(self):
        mA = _ladder_member("X1", "by March 1, 2026", event="EVA-1",
                            close=datetime(2026, 3, 1, tzinfo=UTC))
        mB = _ladder_member("X2", "by March 20, 2026", event="EVB-1",
                            close=datetime(2026, 3, 20, tzinfo=UTC))
        on = self._entry(mA, mB)
        off = self._entry(mA, mB, on=False)
        assert on is not None and off is not None
        assert on == off
        # Ordered and gapped on close_time, as it always was.
        assert on["gap_days"] == 19


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

    Every member gets its own event_ticker, so this guards the CROSS-EVENT
    sweep only — the DR-73 same-event sub-pass is not exercised here at all
    (every bucket holds one member, so it does no pairwise work). Its cost is
    bounded by the measurement recorded in _extract_pairs' docstring instead.
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
        # With no open_time (the shape of a cache record written before
        # _market_to_dict carried it) every request opens at start_date
        # midnight, as it always did; close_ts is per-ticker close + one day.
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

    @staticmethod
    def _requested_open(monkeypatch, market, start=date(2026, 1, 1)):
        """The open_ts _fetch_candles_parallel hands fetch_candlesticks for one market."""
        seen = {}

        def _record(_c, ticker, open_ts, close_ts, use_cache):
            seen[ticker] = open_ts
            return []

        monkeypatch.setattr(backtester, "fetch_candlesticks", _record)
        _fetch_candles_parallel(MagicMock(), {"M": {"ticker": "M", **market}}, start, False)
        return seen["M"]

    def test_a_market_opened_after_the_window_starts_at_its_own_open_hour(self, monkeypatch):
        # A market has no candles before it opens, so its request starts at
        # its own open_time — floored to the hour, so the candle covering the
        # opening hour is inside the request however the endpoint reads a
        # mid-hour start_ts.
        opened = self._requested_open(monkeypatch, {
            "open_time": "2026-03-10T14:30:00Z", "close_time": "2026-04-01T00:00:00Z"})
        assert opened == int(datetime(2026, 3, 10, 14, 0, tzinfo=UTC).timestamp())

    def test_an_on_the_hour_open_is_kept_as_is(self, monkeypatch):
        opened = self._requested_open(monkeypatch, {
            "open_time": "2026-03-10T14:00:00+00:00", "close_time": "2026-04-01T00:00:00Z"})
        assert opened == int(datetime(2026, 3, 10, 14, 0, tzinfo=UTC).timestamp())

    def test_a_market_opened_before_the_window_starts_at_the_window(self, monkeypatch):
        opened = self._requested_open(monkeypatch, {
            "open_time": "2025-06-01T12:00:00Z", "close_time": "2026-02-01T00:00:00Z"})
        assert opened == int(datetime(2026, 1, 1, tzinfo=UTC).timestamp())

    @pytest.mark.parametrize("open_time", [None, "", "not-a-timestamp"])
    def test_an_unreadable_open_time_keeps_the_window_start(self, monkeypatch, open_time):
        # Every cache record written before _market_to_dict carried open_time
        # reads it back as None: those requests are exactly what they were.
        opened = self._requested_open(monkeypatch, {
            "open_time": open_time, "close_time": "2026-02-01T00:00:00Z"})
        assert opened == int(datetime(2026, 1, 1, tzinfo=UTC).timestamp())

    def test_an_open_time_past_the_request_end_keeps_the_window_start(self, monkeypatch):
        # A data defect: the market "opens" after its own request ends (close
        # plus a day). The only safe request is the one always sent.
        opened = self._requested_open(monkeypatch, {
            "open_time": "2026-02-05T00:00:00Z", "close_time": "2026-02-01T00:00:00Z"})
        assert opened == int(datetime(2026, 1, 1, tzinfo=UTC).timestamp())

    def test_a_long_window_gets_every_market_its_candles_end_to_end(
        self, monkeypatch, tmp_path, caplog,
    ):
        # The whole fix through the REAL historical.fetch_candlesticks, against
        # a fake endpoint that refuses any request that could hold more than
        # 5,000 hourly candles, as Kalshi does. On the CLI's default
        # --start-date 2024-01-01 the old code sent ONE request per ticker from
        # that date, which this endpoint refuses for RECENT and LONGLIVED —
        # both came back with no candles and could never enter.
        import json

        from kalshi_betting import historical

        cap = historical.CANDLESTICK_MAX_CANDLES_PER_REQUEST
        hour = 3600

        def ts(text):
            return int(datetime.fromisoformat(text).timestamp())

        needed = {
            # Opened long after the window began; its own life is 3 weeks.
            "RECENT": {"open_time": "2026-03-10T14:30:00+00:00",
                       "close_time": "2026-04-01T00:00:00+00:00"},
            # Open for two years: no single request can hold its own life.
            "LONGLIVED": {"open_time": "2024-06-01T00:00:00+00:00",
                          "close_time": "2026-06-01T00:00:00+00:00"},
            # Opened before the window: requested from the window's start.
            "OLD": {"open_time": "2023-06-01T00:00:00+00:00",
                    "close_time": "2024-03-01T00:00:00+00:00"},
        }
        # One candle per hour from the first full hour of trading to the close
        series = {
            t: range(ts(m["open_time"]) - ts(m["open_time"]) % hour + hour,
                     ts(m["close_time"]) + 1, hour)
            for t, m in needed.items()
        }
        calls: dict[str, list[tuple[int, int]]] = {t: [] for t in needed}

        class _Refused(Exception):
            status = 400
            reason = "max candlesticks: 5000"

        def endpoint(_client, path, **params):
            ticker = path.split("/")[-2]
            start, end = params["start_ts"], params["end_ts"]
            calls[ticker].append((start, end))
            if (end - start) // hour + 1 > cap:
                raise _Refused()
            body = {"candlesticks": [
                {"end_period_ts": t, "yes_ask": {"close": "0.40"},
                 "yes_bid": {"close": "0.38"}}
                for t in series[ticker] if start <= t <= end]}
            return SimpleNamespace(status=200, data=json.dumps(body).encode())

        window_open = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp())
        # The premise: one request from the window's start is refused
        for ticker in ("RECENT", "LONGLIVED"):
            with pytest.raises(_Refused):
                endpoint(None, f"/x/{ticker}/candlesticks", start_ts=window_open,
                         end_ts=ts(needed[ticker]["close_time"]) + 86_400)
            calls[ticker].clear()

        monkeypatch.setattr(historical, "_CANDLES_DIR", tmp_path / "candles")
        monkeypatch.setattr(historical, "_signed_raw_get", endpoint)
        monkeypatch.setattr(historical.time, "sleep", lambda _s: None)
        with caplog.at_level(logging.WARNING):
            result = _fetch_candles_parallel(
                MagicMock(), {t: {"ticker": t, **m} for t, m in needed.items()},
                date(2024, 1, 1), False)

        # Every candle from the later of the window's start and the market's
        # open through its close — nothing lost at either end or in between.
        for ticker in needed:
            expected = [t for t in series[ticker] if t >= window_open]
            assert [c["ts"] for c in result[ticker]] == expected, ticker
        # First candle: the one covering the 14:00 opening hour, ending 15:00
        assert len(result["RECENT"]) == (ts("2026-04-01T00:00:00+00:00")
                                         - ts("2026-03-10T15:00:00+00:00")) // hour + 1
        assert len(calls["RECENT"]) == 1   # its own three weeks: one request
        assert len(calls["OLD"]) == 1
        assert calls["OLD"][0][0] == window_open
        assert len(calls["LONGLIVED"]) >= 2  # paged, every request within the cap
        assert all((e - s) // hour + 1 <= cap for s, e in calls["LONGLIVED"])
        assert not any("returned no candles" in r.getMessage() for r in caplog.records)

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

    @staticmethod
    def _empty_summaries(caplog):
        return [r.getMessage() for r in caplog.records
                if "returned no candles" in r.getMessage()]

    def test_empty_series_are_summarized_once(self, monkeypatch, caplog):
        # TS-02: on a post-cutoff window every ticker 404s and each failure
        # logged its own warning. The count is the useful signal, and it is
        # taken off the RESULT dict — fetch_candlesticks fail-softs a failure
        # to [] internally, so there is no exception here to count.
        needed = self._needed(3)
        empties = {"T00", "T01"}
        monkeypatch.setattr(
            backtester, "fetch_candlesticks",
            lambda _c, ticker, *_a, **_k: (
                [] if ticker in empties else [_candle(_MONDAY_TS, 0.70, 0.32)]),
        )

        with caplog.at_level("WARNING"):
            result = _fetch_candles_parallel(MagicMock(), needed,
                                             date(2026, 1, 1), False)

        assert sum(1 for s in result.values() if not s) == 2
        msgs = self._empty_summaries(caplog)
        assert len(msgs) == 1
        assert "2 of 3 tickers returned no candles" in msgs[0]

    def test_no_empty_series_logs_nothing(self, monkeypatch, caplog):
        # Summary-warning idiom: silent at zero.
        monkeypatch.setattr(backtester, "fetch_candlesticks",
                            lambda *_a, **_k: [_candle(_MONDAY_TS, 0.70, 0.32)])
        with caplog.at_level("WARNING"):
            _fetch_candles_parallel(MagicMock(), self._needed(3),
                                    date(2026, 1, 1), False)
        assert self._empty_summaries(caplog) == []

    def test_summary_counts_tickers_that_never_reached_a_worker(
        self, monkeypatch, caplog,
    ):
        # The summary sits OUTSIDE the `if work:` block on purpose: a ticker
        # resolved to [] for a missing or unparseable close_time never enters
        # `work`, but it is every bit as much a ticker with no prices.
        needed = {
            "GOOD": {"ticker": "GOOD", "close_time": "2026-02-01T00:00:00+00:00"},
            "NOCLOSE": {"ticker": "NOCLOSE", "close_time": None},
        }
        monkeypatch.setattr(backtester, "fetch_candlesticks",
                            lambda *_a, **_k: [_candle(_MONDAY_TS, 0.70, 0.32)])

        with caplog.at_level("WARNING"):
            _fetch_candles_parallel(MagicMock(), needed, date(2026, 1, 1), False)

        msgs = self._empty_summaries(caplog)
        assert len(msgs) == 1
        assert "1 of 2 tickers returned no candles" in msgs[0]

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


class TestLogRss:
    """_log_rss: one INFO line, same meaning on macOS and Linux (TS-07)."""

    @staticmethod
    def _rss_records(caplog):
        return [r for r in caplog.records if r.getMessage().startswith("Peak RSS")]

    def test_reports_a_positive_mib_value(self, caplog):
        with caplog.at_level("INFO"):
            _log_rss("before grouping")
        records = self._rss_records(caplog)
        assert len(records) == 1
        label, mib = records[0].args
        assert label == "before grouping"
        assert mib > 0

    @pytest.mark.parametrize(
        ("platform", "ru_maxrss", "expected_mib"),
        [
            # macOS reports ru_maxrss in BYTES ...
            ("darwin", 6 * 1024 * 1024 * 1024, 6144.0),
            # ... and Linux in KILOBYTES. Same peak, same line.
            ("linux", 6 * 1024 * 1024, 6144.0),
        ],
    )
    def test_units_are_normalized_per_platform(
        self, monkeypatch, caplog, platform, ru_maxrss, expected_mib,
    ):
        monkeypatch.setattr(backtester, "sys", SimpleNamespace(platform=platform))
        monkeypatch.setattr(backtester, "resource", SimpleNamespace(
            RUSAGE_SELF=0,
            getrusage=lambda _who: SimpleNamespace(ru_maxrss=ru_maxrss),
        ))
        with caplog.at_level("INFO"):
            _log_rss("after pair extraction")
        assert self._rss_records(caplog)[0].args[1] == pytest.approx(expected_mib)


class TestPrepareEntriesMemoryInstrumentation:
    """TS-07: the grouping/pairing step holds the groupable subset (SS-1: the
    eligible records that share a grouping key), two group maps and two pair
    lists live at once. It is bracketed by RSS lines, the first of which
    precedes a RAM-budget warning carrying only this run's own numbers, and
    the maps and the subset are released together before the candlestick
    pool runs. The measured figures behind all of this live in config.py
    beside BACKTEST_RECORD_BYTES_ESTIMATE, not here.
    """

    @staticmethod
    def _markets() -> list[dict]:
        return [
            {"ticker": "EA", "event_ticker": "EVA", "event_title": "EV",
             "title": "Team wins by February 1, 2026", "subtitle": "",
             "result": "yes",
             "open_time": "2026-01-01T00:00:00+00:00",
             "close_time": "2026-02-01T00:00:00+00:00",
             "settlement_ts": "2026-02-01T12:00:00+00:00"},
            {"ticker": "EB", "event_ticker": "EVB", "event_title": "EV",
             "title": "Team wins by February 14, 2026", "subtitle": "",
             "result": "yes",
             "open_time": "2026-01-01T00:00:00+00:00",
             "close_time": "2026-02-14T00:00:00+00:00",
             "settlement_ts": "2026-02-14T12:00:00+00:00"},
        ]

    def _run(self, monkeypatch):
        candles = {
            "EA": [_candle(_MONDAY_TS, 0.30, 0.70)],
            "EB": [_candle(_MONDAY_TS, 0.60, 0.40)],
        }
        monkeypatch.setattr(backtester, "fetch_all_settled_markets",
                            lambda *a, **k: self._markets())
        monkeypatch.setattr(backtester, "fetch_candlesticks",
                            lambda _c, ticker, *a, **k: candles[ticker])
        return run_backtest(
            hist_client=MagicMock(), live_client=MagicMock(),
            start_date=date(2026, 1, 1), initial_balance=10_000.0,
        )

    @staticmethod
    def _ram_warnings(caplog):
        # SS-1 reworded the warning: it now counts the GROUPABLE records it
        # says are resident, beside the eligible count they were chosen from.
        return [r.getMessage() for r in caplog.records
                if "are materialized for grouping: their records alone are"
                in r.getMessage()]

    def test_rss_lines_bracket_the_grouping_step(self, monkeypatch, caplog):
        with caplog.at_level("INFO"):
            self._run(monkeypatch)
        labels = [r.args[0] for r in caplog.records
                  if r.getMessage().startswith("Peak RSS")]
        # In order, and exactly the two that bracket grouping/pairing — the
        # window between the existing "Markets to analyze" and "Potential
        # pairs" lines, where the peak lives and is otherwise invisible.
        assert labels == ["before grouping", "after pair extraction"]

    def test_ram_warning_fires_above_the_threshold(self, monkeypatch, caplog):
        monkeypatch.setattr(backtester, "BACKTEST_MARKETS_RAM_WARN", 1)
        with caplog.at_level("WARNING"):
            self._run(monkeypatch)
        warnings = self._ram_warnings(caplog)
        assert len(warnings) == 1
        assert warnings[0].startswith("2 groupable markets (of 2 eligible)")

    def test_ram_warning_is_silent_at_the_threshold(self, monkeypatch, caplog):
        # Strictly greater-than: a run exactly at the threshold is not warned.
        monkeypatch.setattr(backtester, "BACKTEST_MARKETS_RAM_WARN", 2)
        with caplog.at_level("WARNING"):
            self._run(monkeypatch)
        assert self._ram_warnings(caplog) == []

    def test_ram_warning_is_silent_at_the_configured_threshold(
        self, monkeypatch, caplog,
    ):
        # The real constant, unpatched: an ordinary small run says nothing.
        with caplog.at_level("WARNING"):
            self._run(monkeypatch)
        assert self._ram_warnings(caplog) == []

    def test_trades_are_unchanged_by_the_instrumentation(self, monkeypatch):
        # The `del` of the group maps and the record list must not change what
        # the run produces: nothing below pair extraction reads either.
        trades, _ = self._run(monkeypatch)
        assert len(trades) == 1
        assert (trades[0].ticker_a, trades[0].ticker_b) == ("EA", "EB")

    def test_peak_rss_is_logged_before_the_ram_warning(self, monkeypatch, caplog):
        # Causal order: the RSS line reports what the fetch or cache load has
        # already cost, and the warning budgets the records inside it. The
        # warning's own text points at "the peak RSS line above", so the order
        # is part of what it means.
        monkeypatch.setattr(backtester, "BACKTEST_MARKETS_RAM_WARN", 1)
        with caplog.at_level("INFO"):
            self._run(monkeypatch)
        messages = [r.getMessage() for r in caplog.records]
        rss_at = next(i for i, m in enumerate(messages)
                      if m.startswith("Peak RSS before grouping"))
        warn_at = next(i for i, m in enumerate(messages)
                       if "are materialized for grouping: their records alone are" in m)
        assert rss_at < warn_at

    def test_ram_warning_quotes_only_this_runs_numbers(self, monkeypatch, caplog):
        # A line emitted on every run must not carry another run's
        # measurements: those live in config.py's comment, where a reader is
        # prompted to keep them current. The only numbers here are this run's
        # own groupable and eligible counts and the footprint derived from the
        # first.
        monkeypatch.setattr(backtester, "BACKTEST_MARKETS_RAM_WARN", 1)
        with caplog.at_level("WARNING"):
            self._run(monkeypatch)
        message = self._ram_warnings(caplog)[0]
        expected_gb = 2 * backtester.BACKTEST_RECORD_BYTES_ESTIMATE / 1e9
        assert message.startswith("2 groupable markets (of 2 eligible)")
        assert f"{expected_gb:.1f} GB" in message
        # Every numeric token in the line is derived from this run.
        numbers = re.findall(r"\d+(?:\.\d+)?", message)
        assert numbers == ["2", "2", f"{expected_gb:.1f}"]

    def test_unpaired_records_are_released_before_the_candlestick_fetch(
        self, monkeypatch, caplog,
    ):
        """TS-07: deleting the group maps alone frees no record dicts, because
        a list of records still references every one of them. Two such lists
        exist since SS-1, released by two different statements, and each is
        guarded by its own records here:

        - EC shares no grouping key with anything, so it is never copied into
          the groupable subset; only the CORPUS list holds it, and it goes with
          `del markets` right after the second walk.
        - ED and EE share both keys (identical wording on two events of ONE
          series), so they ARE materialized into the groupable subset, yet the
          one-series rule (DR-02) keeps them out of every candidate pair; only
          the post-extraction `del ts_groups, same_groups, groupable` frees
          them.

        Residency, not peak: the process high-water mark is already set by
        this point. The probe is a weakref taken inside the patched fetch, so
        the test itself never holds a record alive.
        """
        candles = {
            "EA": [_candle(_MONDAY_TS, 0.30, 0.70)],
            "EB": [_candle(_MONDAY_TS, 0.60, 0.40)],
        }
        # A third eligible market with a title that groups with nothing else,
        # so it survives the prefilter but is never groupable.
        lonely = {"ticker": "EC", "event_ticker": "EVC", "event_title": "EVC",
                  "title": "Unrelated question by March 1, 2026", "subtitle": "",
                  "result": "no",
                  "open_time": "2026-01-01T00:00:00+00:00",
                  "close_time": "2026-03-01T00:00:00+00:00",
                  "settlement_ts": "2026-03-01T12:00:00+00:00"}
        # Two eligible, groupable markets that no finder may pair: one series
        # (EVD), identical wording, two fixtures.
        fixture = {"event_title": "EVD", "title": "Recurring fixture question",
                   "subtitle": "", "result": "yes",
                   "open_time": "2026-01-01T00:00:00+00:00",
                   "close_time": "2026-03-01T00:00:00+00:00",
                   "settlement_ts": "2026-03-01T12:00:00+00:00"}
        fixtures = [{**fixture, "ticker": "ED", "event_ticker": "EVD-1"},
                    {**fixture, "ticker": "EE", "event_ticker": "EVD-2"}]
        probes: dict[str, weakref.ref] = {}

        def _fetch(*_a, **_k):
            # Built and weak-referenced HERE so the only strong references are
            # the ones the backtester itself keeps.
            records = [_WeakrefDict(m)
                       for m in self._markets() + [lonely] + fixtures]
            for rec in records:
                probes[rec["ticker"]] = weakref.ref(rec)
            return records

        real_pool = backtester._fetch_candles_parallel
        observed: dict[str, bool] = {}

        def _spy(*a, **k):
            gc.collect()
            observed.update({t: probes[t]() is not None for t in probes})
            return real_pool(*a, **k)

        monkeypatch.setattr(backtester, "fetch_all_settled_markets", _fetch)
        monkeypatch.setattr(backtester, "fetch_candlesticks",
                            lambda _c, ticker, *a, **k: candles.get(ticker, []))
        monkeypatch.setattr(backtester, "_fetch_candles_parallel", _spy)
        with caplog.at_level("INFO"):
            trades, _ = run_backtest(
                hist_client=MagicMock(), live_client=MagicMock(),
                start_date=date(2026, 1, 1), initial_balance=10_000.0)

        # Not vacuous: ED and EE really were materialized into the subset (EA,
        # EB, ED, EE of 5 eligible), and really formed no pair.
        assert ("Groupable subset: materializing 4 of 5 eligible markets"
                in caplog.text)
        assert "Potential pairs: 1 time-series, 0 same-title" in caplog.text
        assert [(t.ticker_a, t.ticker_b) for t in trades] == [("EA", "EB")]
        # The unpaired records are gone; the two that a candidate pair holds
        # are still alive, because the pair lists legitimately reference them.
        assert observed == {"EA": True, "EB": True,
                            "EC": False, "ED": False, "EE": False}


class TestOutcomeLabelCoverageCensus:
    """DR-66: a cache written before the 2026-08-14 yes_sub_title ingest fix
    carries subtitle=None on nearly every record, so the time-series key
    collapses to the pre-DR-01 strike-blind title-only form. The run still
    reports pair counts, trades, a return and an empirical k-hat for the
    real-money TIME_SERIES_INTERVAL_PROB_DISCOUNT, and nothing said the numbers
    describe a different strategy. The census is that missing signal.

    The measured coverages behind the threshold live in config.py beside
    BACKTEST_OUTCOME_LABEL_WARN_FRACTION, never in these tests and never in an
    emitted string.
    """

    CENSUS = "Outcome-label coverage"
    WARN_MARK = "below the"

    @staticmethod
    def _census_records(caplog, level):
        return [r.getMessage() for r in caplog.records
                if r.levelname == level
                and r.getMessage().startswith(
                    TestOutcomeLabelCoverageCensus.CENSUS)]

    @staticmethod
    def _labelled(n: int) -> list[dict]:
        return [_md(f"T{i}", f"EV{i}", title="Q", subtitle=f"strike {i}",
                    event_title="Event") for i in range(n)]

    @staticmethod
    def _unlabelled(n: int) -> list[dict]:
        # Exactly the shape of a pre-fix cache record: subtitle null, and the
        # event_title that the bounded fallback never resolved.
        return [_md(f"T{i}", f"EV{i}", title="Q") | {"subtitle": None,
                                                     "event_title": ""}
                for i in range(n)]

    def test_warning_fires_on_a_label_less_market_list(self, caplog):
        with caplog.at_level("INFO"):
            backtester._log_outcome_label_coverage(self._unlabelled(100))
        warnings = self._census_records(caplog, "WARNING")
        assert len(warnings) == 1
        # The consequence: which key degrades, and that the headline numbers
        # are therefore about a different strategy.
        assert "strike-blind" in warnings[0]
        assert "different strategy" in warnings[0]
        # The remedy, exactly as CLAUDE.md's subtitle-drift gotcha states it:
        # both slice stores, plus --no-cache, and the explicit note that
        # --no-cache alone does not refresh the slices.
        assert "backtest_cache/archive_days/" in warnings[0]
        assert "backtest_cache/live_days/" in warnings[0]
        assert "--no-cache ALONE does not refresh the day slices" in warnings[0]

    def test_no_warning_on_a_fully_labelled_list(self, caplog):
        with caplog.at_level("INFO"):
            backtester._log_outcome_label_coverage(self._labelled(100))
        assert self._census_records(caplog, "WARNING") == []
        # The INFO census is unconditional — it is the only signal, so it must
        # be present on a healthy run too.
        info = self._census_records(caplog, "INFO")
        assert len(info) == 1
        assert "subtitle on 100 (100.00%)" in info[0]

    def test_blank_event_title_alone_does_not_warn(self, caplog):
        """event_title coverage is reported, never escalated: it is near zero
        on a HEALTHY cache, because the corpus is overwhelmingly MVE combos
        whose titles the bulk listings exclude and whose per-ticker fallback is
        capped. Warning on it would fire every run."""
        markets = [m | {"event_title": ""} for m in self._labelled(100)]
        with caplog.at_level("INFO"):
            backtester._log_outcome_label_coverage(markets)
        assert self._census_records(caplog, "WARNING") == []
        assert "event_title on 0 (0.00%)" in self._census_records(caplog, "INFO")[0]

    @pytest.mark.parametrize(("labelled", "total", "warns"), [
        # Strictly below the floor warns; exactly at it does not.
        (49, 100, True),
        (50, 100, False),
        (51, 100, False),
    ])
    def test_threshold_is_strictly_below(self, caplog, labelled, total, warns):
        assert backtester.BACKTEST_OUTCOME_LABEL_WARN_FRACTION == 0.50
        markets = self._labelled(labelled) + self._unlabelled(total - labelled)
        with caplog.at_level("INFO"):
            backtester._log_outcome_label_coverage(markets)
        assert bool(self._census_records(caplog, "WARNING")) is warns

    def test_empty_list_does_not_raise_and_does_not_warn(self, caplog):
        """Coverage over an empty list is undefined, not zero: dividing would
        raise, and warning would manufacture a drift alarm out of a corpus with
        no records at all. One INFO line still goes out, so the census line's
        absence always means the helper did not run."""
        with caplog.at_level("INFO"):
            backtester._log_outcome_label_coverage([])
        assert self._census_records(caplog, "WARNING") == []
        info = self._census_records(caplog, "INFO")
        assert info == ["Outcome-label coverage: no eligible markets to census"]

    def test_census_quotes_only_this_runs_numbers(self, caplog):
        # A line emitted on every run must carry no measurement from another
        # run (TS-07). Every numeric token here is this call's own count, a
        # percentage derived from it, or the configured floor.
        with caplog.at_level("INFO"):
            backtester._log_outcome_label_coverage(
                self._labelled(1) + self._unlabelled(3))
        info = self._census_records(caplog, "INFO")[0]
        warning = self._census_records(caplog, "WARNING")[0]
        assert re.findall(r"\d+(?:\.\d+)?", info) == [
            "4", "1", "25.00", "1", "25.00",
        ]
        floor = backtester.BACKTEST_OUTCOME_LABEL_WARN_FRACTION * 100.0
        assert re.findall(r"\d+(?:\.\d+)?", warning) == [
            "25.00", f"{floor:.2f}",
        ]

    def test_it_reads_the_list_once_without_materializing_another(self):
        """The list can be millions of records and the surrounding code is
        memory-tuned (TS-07), so the census must not build a second list. A
        one-shot iterable stands in for the real list: a second pass over it
        would see nothing and miscount."""
        markets = self._labelled(4) + self._unlabelled(6)

        class _OnePassList(list):
            passes = 0

            def __iter__(self):
                _OnePassList.passes += 1
                return super().__iter__()

        probe = _OnePassList(markets)
        backtester._log_outcome_label_coverage(probe)
        assert _OnePassList.passes == 1

    # ── The census is advisory: it must move no number the run reports ──

    @staticmethod
    def _markets() -> list[dict]:
        # A genuine time-series pair, label-less in exactly the way a pre-fix
        # cache is, so the census fires on the run below.
        return [
            {"ticker": "EA", "event_ticker": "EVA", "event_title": "",
             "title": "Team wins by February 1, 2026", "subtitle": None,
             "result": "yes",
             "open_time": "2026-01-01T00:00:00+00:00",
             "close_time": "2026-02-01T00:00:00+00:00",
             "settlement_ts": "2026-02-01T12:00:00+00:00"},
            {"ticker": "EB", "event_ticker": "EVB", "event_title": "",
             "title": "Team wins by February 14, 2026", "subtitle": None,
             "result": "yes",
             "open_time": "2026-01-01T00:00:00+00:00",
             "close_time": "2026-02-14T00:00:00+00:00",
             "settlement_ts": "2026-02-14T12:00:00+00:00"},
        ]

    def _run(self, monkeypatch):
        candles = {
            "EA": [_candle(_MONDAY_TS, 0.30, 0.70)],
            "EB": [_candle(_MONDAY_TS, 0.60, 0.40)],
        }
        monkeypatch.setattr(backtester, "fetch_all_settled_markets",
                            lambda *a, **k: self._markets())
        monkeypatch.setattr(backtester, "fetch_candlesticks",
                            lambda _c, ticker, *a, **k: candles[ticker])
        return run_backtest(
            hist_client=MagicMock(), live_client=MagicMock(),
            start_date=date(2026, 1, 1), initial_balance=10_000.0,
        )

    def test_census_changes_no_backtest_result(self, monkeypatch):
        """Advisory only. Running with the census replaced by a no-op must
        produce the same trades and the same equity curve — if it does not, the
        census is filtering or consuming something it only meant to read.

        SS-1 moved the seam: _prepare_candidates no longer hands a list to
        _log_outcome_label_coverage, it feeds an _OutcomeLabelTally inside its
        first pass and reports through _report_outcome_label_coverage. Both
        halves are replaced here, and the spy proves the run really went
        through them — patching the old name alone would now be a no-op and
        this comparison vacuous."""
        with_census_trades, with_census_equity = self._run(monkeypatch)
        used: list[str] = []

        class _NoCensusTally:
            def add(self, _m):
                used.append("add")

        def _no_report(_tally):
            used.append("report")

        monkeypatch.setattr(backtester, "_OutcomeLabelTally", _NoCensusTally)
        monkeypatch.setattr(backtester, "_report_outcome_label_coverage", _no_report)
        without_trades, without_equity = self._run(monkeypatch)
        assert used == ["add", "add", "report"]

        assert [astuple(t) for t in with_census_trades] == \
               [astuple(t) for t in without_trades]
        pd.testing.assert_frame_equal(with_census_equity, without_equity)
        # Guard against the comparison being vacuous.
        assert len(with_census_trades) == 1

    def test_census_is_logged_inside_the_grouping_window(self, monkeypatch, caplog):
        # After the RSS/RAM-budget lines and before the pair counts, i.e. while
        # the groupable subset is still alive — it is del'd right after pair
        # extraction (the corpus itself was released after the second walk,
        # before the RSS line; SS-1).
        with caplog.at_level("INFO"):
            self._run(monkeypatch)
        messages = [r.getMessage() for r in caplog.records]
        rss_at = next(i for i, m in enumerate(messages)
                      if m.startswith("Peak RSS before grouping"))
        census_at = next(i for i, m in enumerate(messages)
                         if m.startswith(self.CENSUS))
        pairs_at = next(i for i, m in enumerate(messages)
                        if m.startswith("Potential pairs:"))
        assert rss_at < census_at < pairs_at

    def test_a_real_run_on_a_label_less_cache_warns(self, monkeypatch, caplog):
        # End to end: the run still produces its trade and its numbers, and the
        # operator is now told those numbers describe a different strategy.
        with caplog.at_level("INFO"):
            trades, _ = self._run(monkeypatch)
        assert len(trades) == 1
        assert len(self._census_records(caplog, "WARNING")) == 1


class TestOutcomeLabelCoverageIsCarried:
    """DR-66b: the census must also cross out of _prepare_entries, because that
    is the only scope the market records it censused are held in (the corpus,
    and since SS-1 the groupable subset), and their lifetime must not be
    extended (TS-07).

    The carrier is a RETURN VALUE rather than an optional sink precisely
    because a sink can be forgotten — which would reproduce, in the mechanism
    built to close DR-66's silence, exactly that silence.
    """

    def test_the_census_returns_what_it_logged(self, caplog):
        # A shared corpus with pairwise-distinct, non-zero phrasing counts
        # (DR-71): the 3 subtitled/event-titled records also carry a
        # cumulative deadline, 2 are snapshots, 5 state no deadline at all.
        cumulative = [{"subtitle": "Yes", "event_title": "E",
                       "title": "Will X happen by Dec 31, 2026?"}
                      for _ in range(3)]
        snapshot = [{"title": "Bitcoin price on Sep 15, 2026?"}
                    for _ in range(2)]
        unknown = [{"title": "Q"} for _ in range(5)]
        markets = cumulative + snapshot + unknown
        with caplog.at_level("INFO"):
            coverage = backtester._log_outcome_label_coverage(markets)

        assert coverage.total == 10
        assert coverage.with_subtitle == 3
        assert coverage.with_event_title == 3
        assert coverage.subtitle_fraction == pytest.approx(0.30)
        assert coverage.event_title_fraction == pytest.approx(0.30)
        # The same number reached the log, so page and log cannot disagree.
        assert any("30.00%" in r.getMessage() for r in caplog.records)

        # DR-71: the three phrasing fields equal the per-record
        # scanner.deadline_phrasing tally...
        tally = {"cumulative": 0, "snapshot": 0, "unknown": 0}
        for m in markets:
            verdict = scanner.deadline_phrasing(
                m.get("event_title", ""), m.get("title", ""),
                m.get("subtitle", ""))
            tally[verdict] += 1
        assert coverage.cumulative_markets == tally["cumulative"] == 3
        assert coverage.snapshot_markets == tally["snapshot"] == 2
        assert coverage.unknown_deadline_markets == tally["unknown"] == 5
        # ...sum to the corpus...
        assert (coverage.cumulative_markets + coverage.snapshot_markets
                + coverage.unknown_deadline_markets) == coverage.total
        # ...and equal the numbers in the logged INFO line, so the page (which
        # reads these same fields) and the log can never disagree.
        phrasing_lines = [
            r.getMessage() for r in caplog.records
            if r.levelname == "INFO"
            and r.getMessage().startswith("Deadline phrasing over")
        ]
        assert phrasing_lines == [
            "Deadline phrasing over 10 eligible markets: 3 worded as a "
            "cumulative deadline, 2 snapshot, 5 with no deadline wording the "
            "classifier recognises"
        ]

    def test_the_below_floor_flag_is_the_warning_s_own_condition(self):
        floor = backtester.BACKTEST_OUTCOME_LABEL_WARN_FRACTION
        assert floor == 0.50
        # One market either side of the floor, and exactly on it.
        low = backtester._log_outcome_label_coverage(
            [{"subtitle": "Y"}] * 49 + [{}] * 51)
        exact = backtester._log_outcome_label_coverage(
            [{"subtitle": "Y"}] * 50 + [{}] * 50)
        high = backtester._log_outcome_label_coverage(
            [{"subtitle": "Y"}] * 51 + [{}] * 49)
        assert low.below_floor is True
        # Strictly below: a run sitting exactly on the floor is not warned on,
        # matching the `<` the WARNING branches on.
        assert exact.below_floor is False
        assert high.below_floor is False

    def test_an_empty_corpus_reports_undefined_not_zero(self):
        coverage = backtester._log_outcome_label_coverage([])
        assert coverage.total == 0
        # None, not 0.0 — the fraction is undefined on an empty corpus, and a
        # 0.0 here would render as "0% coverage" and warn.
        assert coverage.subtitle_fraction is None
        assert coverage.event_title_fraction is None
        assert coverage.below_floor is False
        # DR-71: the phrasing fields are a genuine measurement (zero markets
        # of every kind), not an omission that happened to default to 0.
        assert coverage.cumulative_markets == 0
        assert coverage.snapshot_markets == 0
        assert coverage.unknown_deadline_markets == 0

    @pytest.mark.parametrize("omit", [
        "cumulative_markets", "snapshot_markets", "unknown_deadline_markets",
    ])
    def test_the_phrasing_fields_are_required(self, omit):
        # DR-71: before this, a forgotten phrasing keyword silently defaulted
        # to 0, which would render the dashboard's strongest sentence ("could
        # not have produced a time-series trade at all") as if it had been
        # measured. A forgotten keyword is now a TypeError at construction —
        # the same "cannot be forgotten" reasoning DR-66b used for returning
        # the carrier in the first place. Parametrized over each of the three
        # fields individually (C5-ADV-2): omitting all three at once only
        # pins "at least one is required", and would survive a mutant that
        # restored a default on just the trailing field(s).
        kwargs = {
            "total": 1, "with_subtitle": 1, "with_event_title": 1,
            "subtitle_fraction": 1.0, "event_title_fraction": 1.0,
            "below_floor": False,
            "cumulative_markets": 1, "snapshot_markets": 0,
            "unknown_deadline_markets": 0,
        }
        del kwargs[omit]
        with pytest.raises(TypeError):
            backtester.OutcomeLabelCoverage(**kwargs)

    def test_the_carrier_holds_no_market_reference(self):
        # _prepare_candidates del's the corpus right after its second walk and
        # the groupable subset right after pair extraction, to lower residency
        # across grouping and the candlestick fetch (TS-07, SS-1). A carrier
        # holding examples would pin those records alive past both statements.
        coverage = backtester._log_outcome_label_coverage(
            [{"subtitle": "Y", "event_title": "E"}])
        for value in astuple(coverage):
            assert isinstance(value, (int, float, bool, type(None)))

    def test_the_sweep_carries_the_census(self, monkeypatch):
        # It hangs off BacktestSweep, not off a SweepPoint: it is k-independent,
        # exactly like the calibration beside it — one measurement, valid at
        # every swept discount.
        census = backtester.OutcomeLabelCoverage(
            total=4, with_subtitle=1, with_event_title=1,
            subtitle_fraction=0.25, event_title_fraction=0.25,
            below_floor=True,
            cumulative_markets=1, snapshot_markets=1, unknown_deadline_markets=2,
        )
        monkeypatch.setattr(backtester, "_prepare_candidates",
                            lambda *a, **k: backtester._Candidates(
                                all_pairs=[], candles_by_ticker={},
                                label_coverage=census, start_date=date(2026, 1, 1),
                                max_horizon_days=None,
                                same_event_ladders=k.get("same_event_ladders")))
        monkeypatch.setattr(backtester, "_interval_calibration", lambda *a, **k: None)
        result = backtester.run_backtest_sweep(
            MagicMock(), MagicMock(), date(2026, 1, 1), 1000.0, sweep=False)
        assert result.label_coverage is census

    def test_the_infeasible_window_carries_no_census(self, monkeypatch):
        # The fetch never ran, so nothing was censused — None, distinct from a
        # censused corpus that held zero records.
        monkeypatch.setattr(backtester, "_prepare_candidates",
                            lambda *a, **k: None)
        result = backtester.run_backtest_sweep(
            MagicMock(), MagicMock(), date(2026, 1, 1), 1000.0, sweep=False)
        assert result.label_coverage is None
        assert result.calibration is None

    def test_run_backtest_keeps_its_two_tuple(self, monkeypatch):
        # Constraint: run_backtest's signature and return type are unchanged —
        # it unpacks and discards the census, which has already logged itself.
        monkeypatch.setattr(backtester, "_prepare_entries",
                            lambda *a, **k: ([], None))
        out = run_backtest(hist_client=MagicMock(), live_client=MagicMock(),
                           start_date=date(2026, 1, 1), initial_balance=1000.0)
        assert isinstance(out, tuple) and len(out) == 2
        trades, equity = out
        assert trades == []
        assert list(equity.columns) == ["date", "portfolio_value", "daily_return"]


class TestCorpusProvenanceIsCarried:
    """DR-13 / M2 (P2): what the fetched corpus says about itself — when it was
    assembled, whether it came from an earlier run's cache, and the archive
    cutoff and post-cutoff verdict as of that assembly — rides from the
    SettledCorpus through _Candidates onto BacktestSweep.corpus_provenance,
    exactly the way label_coverage travels, so the dashboard header can render
    it. None whenever it was never recorded."""

    PROV = historical.CorpusProvenance(
        from_cache=True, assembled_at=datetime(2026, 9, 24, 12, 37, tzinfo=UTC),
        archive_cutoff=datetime(2026, 7, 25, tzinfo=UTC), post_cutoff=True)

    def test_the_sweep_carries_the_provenance(self, monkeypatch):
        monkeypatch.setattr(backtester, "_prepare_candidates",
                            lambda *a, **k: backtester._Candidates(
                                all_pairs=[], candles_by_ticker={},
                                label_coverage=None, start_date=date(2026, 1, 1),
                                max_horizon_days=None,
                                same_event_ladders=k.get("same_event_ladders"),
                                corpus_provenance=self.PROV))
        monkeypatch.setattr(backtester, "_interval_calibration", lambda *a, **k: None)
        result = backtester.run_backtest_sweep(
            MagicMock(), MagicMock(), date(2026, 1, 1), 1000.0, sweep=False)
        assert result.corpus_provenance is self.PROV

    def test_the_infeasible_window_carries_none(self, monkeypatch):
        # No fetch ran, so there is no corpus to describe.
        monkeypatch.setattr(backtester, "_prepare_candidates", lambda *a, **k: None)
        result = backtester.run_backtest_sweep(
            MagicMock(), MagicMock(), date(2026, 1, 1), 1000.0, sweep=False)
        assert result.corpus_provenance is None

    @staticmethod
    def _point(n_trades, k=0.75):
        return backtester.SweepPoint(k=k, trades=[object()] * n_trades,
                                     equity_df=pd.DataFrame())

    @pytest.mark.parametrize("where, expected", [
        ("none", 0), ("primary", 3), ("points", 5), ("scenarios", 7),
        ("same_title_point", 2), ("tier_off_scenarios", 9),
    ])
    def test_max_trades_simulated_reads_every_point_the_page_can_show(
            self, where, expected):
        # The one test both renderers apply to a carried post-cutoff verdict:
        # a trade at ANY simulated point proves it stale, since the k dropdown
        # and the scenario explorer put every point on the same page — a
        # tier-floors-off scenario included.
        primary = self._point(3 if where == "primary" else 0)
        off_point = self._point(9 if where == "tier_off_scenarios" else 0, k=0.9)
        off_point.tier_floors = False
        sweep = backtester.BacktestSweep(
            primary=primary,
            points=[primary, self._point(5 if where == "points" else 0, k=0.5)],
            calibration=None,
            scenarios=[self._point(7 if where == "scenarios" else 0, k=0.9)],
            same_title_point=(self._point(2) if where == "same_title_point"
                              else None),
            tier_off_scenarios=[off_point])
        assert backtester.max_trades_simulated(sweep) == expected

    def test_existing_constructions_default_to_none(self):
        # Defaulted, like label_coverage: a hand-built sweep or candidates
        # object needs no change and reads as "not recorded".
        point = backtester.SweepPoint(k=0.75, trades=[], equity_df=pd.DataFrame())
        assert backtester.BacktestSweep(primary=point, points=[point],
                                        calibration=None).corpus_provenance is None
        assert backtester._Candidates(
            all_pairs=[], candles_by_ticker={}, label_coverage=None,
            start_date=date(2026, 1, 1), max_horizon_days=None,
            same_event_ladders=None).corpus_provenance is None


class TestRunBacktestFeasibilityPreCheck:
    """BS-11: no Monday 09:00 UTC checkpoint in the window means no trade can
    ever be entered, so run_backtest must skip the fetch entirely rather than
    discover that only after paying for it.

    `backtester.datetime` (not the stdlib one) is patched with a thin subclass
    whose `.now(tz)` is frozen, since backtester.py imports `datetime` by name
    (`from datetime import ... datetime ...`) and the feasibility window is
    `datetime.now(UTC).date()`. It used to be `date.today()`, and this helper
    used to freeze `backtester.date` accordingly — the switch to UTC (TS-13)
    is what moved the seam.
    """

    class _FrozenDateTime(datetime):
        _fixed: datetime

        @classmethod
        def now(cls, tz=None):
            return cls._fixed if tz is None else cls._fixed.astimezone(tz)

    def _freeze(self, monkeypatch, today: date, hour: int = 12):
        """Freeze UTC now at `hour` on `today`. The default noon is
        deliberately mid-day so a test that does not care about the boundary
        cannot accidentally straddle one."""
        fixed = datetime(today.year, today.month, today.day, hour, tzinfo=UTC)
        frozen = type("FrozenDateTime", (self._FrozenDateTime,), {"_fixed": fixed})
        monkeypatch.setattr(backtester, "datetime", frozen)

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


class TestEquityCurveFutureStartDate:
    """A start_date after today (UTC) leaves _build_equity_curve's day span
    zero or negative. It used to build pd.DataFrame([]) — a frame with no
    columns at all — and then raise KeyError: 'portfolio_value' on the very
    next line, so the documented empty-result shape was unreachable through
    the public API.

    TestRunBacktestFeasibilityPreCheck cannot catch this: it freezes
    `backtester.date.today()`, which drives the Monday pre-check but NOT
    _build_equity_curve's own `datetime.now(UTC).date()`, so its windows are
    always non-empty. These tests use a genuinely future start_date instead of
    freezing anything, so both clocks agree it is ahead of today.
    """

    # Comfortably ahead of both `date.today()` (local tz) and UTC today, so
    # the window is empty no matter which side of midnight the suite runs on.
    _FUTURE_START = datetime.now(UTC).date() + timedelta(days=30)

    def _fetch_should_not_be_called(self, monkeypatch):
        monkeypatch.setattr(
            backtester, "fetch_all_settled_markets",
            lambda *a, **k: pytest.fail("fetch must be skipped for a future window"),
        )

    def test_build_equity_curve_emits_two_rows_for_a_future_start(self):
        """Re-pinned from "one row" (DR-03): _build_equity_curve now opens every
        curve one day BEFORE start_date at the untouched initial balance, so the
        floored span produces the leading row plus start_date's own — two rows,
        not one. The old single-row expectation described the curve that hid a
        day-0 entry's outflow from pct_change and cummax; nothing about the
        future-window guarantee this class exists for changed (the frame still
        has its three columns and is still readable by .iloc).
        """
        df = backtester._build_equity_curve([], self._FUTURE_START, 1234.0)

        assert list(df.columns) == ["date", "portfolio_value", "daily_return"]
        assert len(df) == 2
        assert df["date"].iloc[0] == self._FUTURE_START - timedelta(days=1)
        assert df["date"].iloc[1] == self._FUTURE_START
        # Nothing can enter before start_date, and start_date is in the future,
        # so both rows sit at the initial balance and neither moves.
        assert df["portfolio_value"].iloc[0] == pytest.approx(1234.0)
        assert df["portfolio_value"].iloc[1] == pytest.approx(1234.0)
        assert df["daily_return"].iloc[0] == pytest.approx(0.0)
        assert df["daily_return"].iloc[1] == pytest.approx(0.0)

    def test_run_backtest_returns_the_empty_shape_for_a_future_start(self, monkeypatch):
        self._fetch_should_not_be_called(monkeypatch)

        trades, equity = run_backtest(
            hist_client=MagicMock(), live_client=MagicMock(),
            start_date=self._FUTURE_START, initial_balance=2000.0,
        )

        assert trades == []
        assert list(equity.columns) == ["date", "portfolio_value", "daily_return"]
        # Flat at initial_balance, as run_backtest's docstring promises — and
        # readable by .iloc, which dashboard.py depends on.
        assert equity["portfolio_value"].iloc[0] == pytest.approx(2000.0)
        assert equity["portfolio_value"].iloc[-1] == pytest.approx(2000.0)

    def test_run_backtest_sweep_returns_the_empty_shape_for_a_future_start(self, monkeypatch):
        # run_backtest_sweep reaches the same curve through its own
        # short-circuit (one empty point via _simulate_at_discount).
        self._fetch_should_not_be_called(monkeypatch)

        result = run_backtest_sweep(
            hist_client=MagicMock(), live_client=MagicMock(),
            start_date=self._FUTURE_START, initial_balance=2000.0,
        )

        assert result.points == [result.primary]
        assert result.primary.trades == []
        assert result.calibration is None
        assert list(result.primary.equity_df.columns) == [
            "date", "portfolio_value", "daily_return"]
        assert result.primary.equity_df["portfolio_value"].iloc[-1] == pytest.approx(2000.0)


class TestEquityCurveOpensAtTheInitialBalance:
    """A trade entering on start_date itself must show its day-0 charges as a
    real daily return and a real drawdown (DR-03).

    The default backtest window starts on a Monday (--start-date 2024-01-01),
    which is exactly the kind of day _find_entry can open a trade on, and the
    curve used to apply that day's charges to its FIRST row. pct_change and
    cummax both read the first row as the baseline, so day 0 was invisible to
    both: on this fixture the dashboard reported max drawdown 0.0% on a run that
    ended day one holding $1.84 of CASH out of its $10,000, and the per-k sweep
    table divided by the depleted opening (+133,804.3%) while the performance card
    divided by initial_balance (-75.4%) — one run reported two ways on one page.

    DR-61 changed WHAT day 0 costs without touching that guarantee. The curve is
    now a portfolio value rather than a cash balance (open positions are carried
    at cost), so this fixture's day-0 step is its $476.56 of taker fees rather
    than the whole $9,998.16 stake, and the stake's actual LOSS lands on the
    exit date where it is realized. The leading row is still what keeps the
    cummax peak at $10,000 instead of at the already-charged $9,523.44, and
    still what makes the two report bases agree — so this class keeps pinning
    the leading row, at the figures the current accounting produces.

    The fixture reproduces the 2026-09-15 dry-run sweep's shape: $10,000 in,
    everything committed on start_date ($9,998.16 all-in across both legs of two
    pairs), $2,462.00 back 29 days later, a closing portfolio of $2,463.84. It
    takes two trades because one cannot do it: a BacktestTrade's winning
    settlement pays its own contract count, and a pair's two leg prices sum to
    less than $1, so a single winning trade always pays back MORE than it cost.
    The big pair therefore settles in the time-series in-between cell (A=NO,
    B=YES — both legs worthless, the whole stake lost) and the small one in the
    never-by-B win cell (A=NO, B=NO — the NO on B pays n).
    """

    _START = date(2026, 6, 1)      # a Monday, comfortably in the past
    _EXIT = date(2026, 6, 30)
    _HOLDING_DAYS = (_EXIT - _START).days
    _INITIAL = 10_000.0

    # Winner: YES on the earlier leg at 0.15, NO on the later at 0.40.
    _WIN_N, _WIN_PA, _WIN_NB = 2462, 0.15, 0.40
    # Loser: YES at 0.25, NO at 0.30 — settles in the in-between cell.
    _LOSS_N, _LOSS_PA, _LOSS_NB = 14850, 0.25, 0.30

    def _trade(self, n: int, pA: float, nB: float, outcome_b: str,
               payoff: float) -> backtester.BacktestTrade:
        """One coherent time-series BacktestTrade.

        The loss cell (outcome_b="yes") pays nothing; the never-by-B win cell
        (outcome_b="no") pays n, the count of NO contracts held on market B.
        Fees are the real two-leg taker fees at these prices, so the two steps
        the equity curve takes — the entry-day fees and the exit-day realized
        P&L — are the ones the backtester would have recorded. (Under DR-61 the
        curve no longer subtracts the whole outflow on the entry date: the
        contracts bought with it are carried at cost until settlement.)
        """
        cost = n * (pA + nB)
        fees = fee_leg_exact(n, pA) + fee_leg_exact(n, nB)
        profit = payoff - cost - fees
        expected_payoff = n * (1.0 - pA - nB) - fees
        return backtester.BacktestTrade(
            pair_type="time_series",
            ticker_a="TICK-A", ticker_b="TICK-B",
            title_a="Will BTC exceed $80k by June?",
            title_b="Will BTC exceed $80k by July?",
            category="Crypto",
            entry_date=self._START, exit_date=self._EXIT,
            entry_pA=pA, entry_pB=0.60, entry_nA=1.0 - pA, entry_nB=nB,
            n=n,
            total_cost=cost, fees=fees,
            outcome_a="no", outcome_b=outcome_b,
            actual_payoff=payoff,
            profit=profit,
            profit_ratio=profit / (cost + fees),
            monthly_profit_ratio=profit / (cost + fees) * 30 / self._HOLDING_DAYS,
            kelly_fraction=0.2,
            expected_payoff=expected_payoff,
            slippage=profit - expected_payoff,
            holding_days=self._HOLDING_DAYS,
            balance_at_entry=self._INITIAL,
            deadline_gap_days=7,
        )

    def _trades(self) -> list[backtester.BacktestTrade]:
        return [
            self._trade(self._LOSS_N, self._LOSS_PA, self._LOSS_NB,
                        outcome_b="yes", payoff=0.0),
            self._trade(self._WIN_N, self._WIN_PA, self._WIN_NB,
                        outcome_b="no", payoff=float(self._WIN_N)),
        ]

    def _curve(self) -> pd.DataFrame:
        return backtester._build_equity_curve(
            self._trades(), self._START, self._INITIAL)

    def test_the_fixture_commits_the_whole_balance_on_day_zero(self):
        # Guards the numbers every other test in this class reads: both pairs
        # enter on start_date for $9,998.16 all-in out of $10,000.
        outflow = sum(t.total_cost + t.fees for t in self._trades())
        assert outflow == pytest.approx(9998.16)
        assert all(t.entry_date == self._START for t in self._trades())

        # ...and that each leg's payoff is the cell the backtester would
        # actually have paid, so the class docstring's settlement claims are
        # checked rather than asserted: hand-written literals would stay green
        # through a change to _settlement_receipt's time-series table.
        for t in self._trades():
            assert t.actual_payoff == backtester._settlement_receipt(
                t.n, t.outcome_a, t.outcome_b, t.pair_type)

    def test_leading_row_is_the_untouched_initial_balance(self):
        eq = self._curve()

        assert eq["date"].iloc[0] == self._START - timedelta(days=1)
        assert eq["portfolio_value"].iloc[0] == pytest.approx(self._INITIAL)
        assert eq["daily_return"].iloc[0] == pytest.approx(0.0)

    def test_day_zero_charge_is_a_real_daily_return(self):
        """Re-pinned for DR-61 (was: "outflow", -99.98% on day 0).

        Row 1 is start_date. Both pairs commit $9,998.16 that day, but $9,521.60
        of it buys contracts that are still held, so the only value that LEAVES
        the portfolio is the $476.56 of taker fees. That step is still a real,
        visible negative return — it just measures a real cost instead of
        measuring deployment.
        """
        fees = sum(t.fees for t in self._trades())
        assert fees == pytest.approx(476.56)

        eq = self._curve()

        assert eq["date"].iloc[1] == self._START
        assert eq["portfolio_value"].iloc[1] == pytest.approx(
            self._INITIAL - fees)
        assert eq["portfolio_value"].iloc[1] == pytest.approx(9523.44)
        assert eq["daily_return"].iloc[1] < 0
        assert eq["daily_return"].iloc[1] == pytest.approx(-0.047656, abs=1e-6)

    def test_max_drawdown_sees_the_realized_loss(self):
        """Re-pinned for DR-61 (was: a -99.98% trough on start_date).

        The trough is now the EXIT date, where the losing pair's stake is
        actually written off, and its magnitude is the run's realized loss —
        which for a monotonically declining run equals the total return. The old
        -99.98% trough on start_date was the deployment artefact DR-61 removed.
        """
        from kalshi_betting.dashboard import _max_drawdown

        eq = self._curve()
        max_dd, trough = _max_drawdown(
            eq["portfolio_value"].set_axis(eq["date"]))

        assert trough == self._EXIT
        assert max_dd == pytest.approx(-0.753616, abs=1e-6)
        # Nothing ever rose above the opening, so the deepest drawdown and the
        # total return are the same number — a coherence the cash-only curve
        # could not produce (-99.98% drawdown against a -75.4% return).
        final = float(eq["portfolio_value"].iloc[-1])
        assert max_dd == pytest.approx(
            (final - self._INITIAL) / self._INITIAL, abs=1e-9)

    def test_the_leading_row_is_what_keeps_the_day_zero_fee_in_the_drawdown(self):
        """DR-03's mechanism, re-pinned under DR-61's accounting.

        Day 0 is a decline (the fees), so without the leading row the cummax
        peak would be the already-charged $9,523.44 and the reported drawdown
        would be shallower by exactly that fee. Dropping the leading row from
        the same curve reproduces the understatement, which is what makes the
        leading row measurable rather than merely asserted.
        """
        from kalshi_betting.dashboard import _max_drawdown

        eq = self._curve()
        with_leading, _ = _max_drawdown(
            eq["portfolio_value"].set_axis(eq["date"]))
        without_leading, _ = _max_drawdown(
            eq["portfolio_value"].iloc[1:].set_axis(eq["date"].iloc[1:]))

        assert with_leading < without_leading
        assert with_leading == pytest.approx(-0.753616, abs=1e-6)
        # Same trough, shallower peak: 2463.84 / 9523.44 - 1.
        assert without_leading == pytest.approx(-0.741287, abs=1e-6)

    def test_sweep_row_and_performance_card_report_one_return(self):
        # _srow (inside _section_interval_discount) divides by the curve's
        # iloc[0]; _section_performance divides by initial_balance. With the
        # leading row those bases are the same number, so the two cells on one
        # page can no longer disagree.
        from kalshi_betting import dashboard

        eq = self._curve()
        final = float(eq["portfolio_value"].iloc[-1])
        opening = float(eq["portfolio_value"].iloc[0])

        assert final == pytest.approx(2463.84)
        assert opening == pytest.approx(self._INITIAL)
        assert (final - opening) / opening == pytest.approx(
            (final - self._INITIAL) / self._INITIAL, abs=1e-9)

        trades = self._trades()
        point = backtester.SweepPoint(
            k=TIME_SERIES_INTERVAL_PROB_DISCOUNT, trades=trades, equity_df=eq)
        sweep = backtester.BacktestSweep(
            primary=point, points=[point], calibration=None)

        # Both render the same headline percentage, -75.4%.
        assert "-75.4%" in dashboard._section_interval_discount(sweep)
        assert "-75.4%" in dashboard._section_performance(
            eq, trades, self._START, self._INITIAL)


class TestEquityCurveCarriesOpenPositionsAtCost:
    """The equity curve must not report capital DEPLOYMENT as loss (DR-61).

    _build_equity_curve used to accumulate cash alone, so an open position was
    carried at ZERO for its whole holding period and the curve dived on the
    entry date and recovered on the exit date whether the trade won or lost.
    Every risk figure on the dashboard reads that series — the "Max Drawdown"
    KPI, the "Drawdown (%)" chart, _sharpe/_sortino via the derived
    "daily_return" column, the per-k sweep table's drawdown and Sharpe columns,
    and the benchmark row that sits in the same column as ^GSPC's genuine
    mark-to-market drawdown — so all of them measured peak deployment.

    The real 2026-05-01 run is the proof: its k=1.00 point had THREE trades, all
    three profitable and a +4.8% return, and the rendered table reported a max
    drawdown of -60.0%; its k=0.40 point reported -100.0% (total ruin) against a
    final balance of $4,655.87. This fixture reproduces that shape — three
    winning time-series pairs all entering on one Monday, committing $6,227.91
    of $10,000, which the cash-only curve read as a -62.3% drawdown.

    An open position is now carried at its COST BASIS, so the only moves left
    are the entry-day fees and the realized P&L at settlement.
    """

    _START = date(2026, 5, 25)     # a Monday, comfortably in the past
    _INITIAL = 10_000.0
    # (n, pA, nB, exit_date) — the never-by-B win cell (A=NO, B=NO) pays n on
    # the NO leg held against market B, so every one of these is profitable.
    _WINNERS = [
        (4800, 0.15, 0.40, date(2026, 6, 15)),
        (3600, 0.20, 0.35, date(2026, 6, 29)),
        (2400, 0.25, 0.30, date(2026, 7, 13)),
    ]

    def _trade(self, n: int, pA: float, nB: float, exit_date: date,
               outcome_b: str, payoff: float) -> backtester.BacktestTrade:
        """One coherent time-series BacktestTrade entering on _START.

        outcome_b="no" is the never-by-B win cell and pays n; outcome_b="yes"
        is the in-between cell, where both legs expire worthless. Fees are the
        real two-leg taker fees at these prices.
        """
        cost = n * (pA + nB)
        fees = fee_leg_exact(n, pA) + fee_leg_exact(n, nB)
        profit = payoff - cost - fees
        holding_days = (exit_date - self._START).days
        return backtester.BacktestTrade(
            pair_type="time_series",
            ticker_a="TICK-A", ticker_b="TICK-B",
            title_a="Will BTC exceed $80k by June?",
            title_b="Will BTC exceed $80k by July?",
            category="Crypto",
            entry_date=self._START, exit_date=exit_date,
            entry_pA=pA, entry_pB=0.60, entry_nA=1.0 - pA, entry_nB=nB,
            n=n,
            total_cost=cost, fees=fees,
            outcome_a="no", outcome_b=outcome_b,
            actual_payoff=payoff,
            profit=profit,
            profit_ratio=profit / (cost + fees),
            monthly_profit_ratio=profit / (cost + fees) * 30 / holding_days,
            kelly_fraction=0.2,
            expected_payoff=n * (1.0 - pA - nB) - fees,
            slippage=profit - (n * (1.0 - pA - nB) - fees),
            holding_days=holding_days,
            balance_at_entry=self._INITIAL,
            deadline_gap_days=7,
        )

    def _all_winners(self) -> list[backtester.BacktestTrade]:
        return [self._trade(n, pA, nB, exit_date, outcome_b="no",
                            payoff=float(n))
                for n, pA, nB, exit_date in self._WINNERS]

    @staticmethod
    def _cash_only_final(trades, initial: float) -> float:
        """The pre-DR-61 curve's closing value, computed the old way.

        Cash-only accounting and cost-basis carry differ only in the PATH
        between entry and settlement, so this is what lets the tests below
        assert the endpoint is untouched without needing the old code.
        """
        return initial + sum(t.actual_payoff - t.total_cost - t.fees
                             for t in trades)

    def test_the_fixture_is_the_real_runs_shape(self):
        # Guards every number the tests below read: three trades, all
        # profitable, all entering on one day, committing 62.3% of the balance.
        trades = self._all_winners()
        assert len(trades) == 3
        assert all(t.profit > 0 for t in trades)
        assert all(t.entry_date == self._START for t in trades)
        assert sum(t.total_cost + t.fees for t in trades) == pytest.approx(6227.91)
        # ...and each payoff is the cell _settlement_receipt would have paid,
        # so "all profitable" is checked rather than asserted.
        for t in trades:
            assert t.actual_payoff == backtester._settlement_receipt(
                t.n, t.outcome_a, t.outcome_b, t.pair_type)

    def test_an_all_profitable_run_has_no_deployment_drawdown(self):
        """THE headline pin. Fails on the cash-only curve, which reports
        -62.3% here (the real run reported -60.0% on the same shape)."""
        from kalshi_betting.dashboard import _max_drawdown

        trades = self._all_winners()
        eq = backtester._build_equity_curve(trades, self._START, self._INITIAL)
        max_dd, trough = _max_drawdown(
            eq["portfolio_value"].set_axis(eq["date"]))

        fees = sum(t.fees for t in trades)
        # The only realized cost a winning run can carry is its taker fees, so
        # the deepest drawdown is exactly those, on the day they were charged.
        assert max_dd == pytest.approx(-fees / self._INITIAL, abs=1e-12)
        assert max_dd == pytest.approx(-0.028791, abs=1e-6)
        assert trough == self._START
        # Nowhere near the deployment artefact this replaced.
        assert max_dd > -0.05

    def test_the_curve_is_flat_while_the_positions_are_open(self):
        """Shape documentation, NOT a DR-61 discriminator — it passes on the
        cash-only builder too (measured in the negative control), since cash is
        also flat between the entry day and the first settlement, just at a
        lower level, and a run of winners is monotone under both accountings.
        The discriminating pins are the drawdown and entry-step tests below.
        """
        trades = self._all_winners()
        eq = backtester._build_equity_curve(trades, self._START, self._INITIAL)
        by_date = dict(zip(eq["date"], eq["portfolio_value"], strict=True))

        entry_value = by_date[self._START]
        # Every day between the entry and the first settlement holds three open
        # positions at cost and sees no cash move at all.
        for offset in range(1, (date(2026, 6, 15) - self._START).days):
            assert by_date[self._START + timedelta(days=offset)] == pytest.approx(
                entry_value)

        # After day 0 a run of winners can only climb: each settlement returns
        # more than the position it writes off.
        post = eq["portfolio_value"].iloc[1:].tolist()
        assert all(b >= a - 1e-9 for a, b in zip(post, post[1:], strict=False))

    def test_the_entry_day_step_is_the_fees_and_nothing_else(self):
        """Fees are NOT capitalised into the carrying value: they buy nothing
        that can be sold on, so they hit the day they are charged."""
        trades = self._all_winners()
        eq = backtester._build_equity_curve(trades, self._START, self._INITIAL)

        fees = sum(t.fees for t in trades)
        assert eq["portfolio_value"].iloc[0] == pytest.approx(self._INITIAL)
        assert eq["portfolio_value"].iloc[1] == pytest.approx(
            self._INITIAL - fees)

    def test_a_losing_trade_still_produces_a_real_drawdown(self):
        """The fix must not flatten genuine losses — only deployment."""
        from kalshi_betting.dashboard import _max_drawdown

        exit_date = date(2026, 6, 15)
        loser = self._trade(4800, 0.15, 0.40, exit_date,
                            outcome_b="yes", payoff=0.0)
        assert loser.profit < 0

        eq = backtester._build_equity_curve([loser], self._START, self._INITIAL)
        max_dd, trough = _max_drawdown(
            eq["portfolio_value"].set_axis(eq["date"]))

        # The whole stake is written off on the settlement date, not on entry.
        assert trough == exit_date
        final = self._INITIAL + loser.profit
        assert eq["portfolio_value"].iloc[-1] == pytest.approx(final)
        assert max_dd == pytest.approx(
            (final - self._INITIAL) / self._INITIAL, abs=1e-12)
        assert max_dd == pytest.approx(-0.276348, abs=1e-6)

    def test_total_return_and_final_balance_are_unchanged(self):
        """Only the PATH moves: the endpoint must match the cash-only curve's
        to the last cent, on a MIXED fixture (two winners and a loser)."""
        winners = self._all_winners()[:2]
        loser = self._trade(2400, 0.25, 0.30, date(2026, 7, 13),
                            outcome_b="yes", payoff=0.0)
        trades = [*winners, loser]

        eq = backtester._build_equity_curve(trades, self._START, self._INITIAL)
        final = float(eq["portfolio_value"].iloc[-1])

        assert final == pytest.approx(
            self._cash_only_final(trades, self._INITIAL), abs=1e-9)
        # ...and therefore so does the total return both report bases divide out
        # (the leading row is the untouched initial balance — DR-03).
        opening = float(eq["portfolio_value"].iloc[0])
        assert opening == pytest.approx(self._INITIAL)
        assert (final - opening) / opening == pytest.approx(
            (self._cash_only_final(trades, self._INITIAL) - self._INITIAL)
            / self._INITIAL, abs=1e-12)


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
    """The cross-type collision the dedup (C4) exists for can no longer arise
    from the two finders, and this pins WHY.

    Fixture shape: two markets sharing an identical (event_title, title,
    subtitle) — so _group_by_exact_title groups them as same_title — whose
    titles therefore also normalize to one key, so _group_by_normalized_title
    GROUPS the same two tickers for time_series too.

    That used to yield both copies, which is what _drop_cross_type_duplicates
    was built to resolve. It no longer can: a time-series pair must state two
    DIFFERENT cumulative deadlines, and identical wording cannot state two of
    anything. The two conditions are mutually exclusive on one ticker pair,
    so Pass 1 produces at most the same-title copy and the dedup has nothing
    to drop. The helper stays — it is still the right thing to do if a
    collision ever arises another way — and TestDropCrossTypeDuplicates
    unit-tests it directly; what changed is that the FINDERS no longer
    manufacture one.

    Two versions of the fixture, since DR-74. _ALIGNED_MARKETS close at one
    instant, so the same-title close gate admits the pair and the same-title
    copy is the only one. _MARKETS close a week apart — two fixtures, not one
    question — so the close gate refuses the same-title copy, and because
    identical wording can never form a time-series pair (DR-67) the refused
    copy does not come back relabelled as one: NOTHING reaches the dedup.
    That second version is the relabel guard for a same-title-only gate.
    """

    # Closes a week apart: DA Feb 1, DB Feb 8 — two fixtures of one
    # identically worded question on two series. The same-title close gate
    # (DR-74) refuses the pair and the time-series branch refuses identical
    # wording, so no candidate of either type forms.
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
    # The same two markets closing at ONE instant (both Feb 1) — the
    # flow-through fixture. DA yes 0.30 / no 0.70; DB yes 0.60 / no 0.40.
    # Same-title copy: DB is the pricier side, gap 0.30 >= 0.05, legs
    # nA+pB = 0.40+0.30 = 0.70 <= 0.95. A time-series copy would clear the
    # 15% short-gap tier on price alone (DB priced 0.30 above DA, legs
    # pA+nB = 0.30+0.40 = 0.70 <= 0.85), but it is refused earlier than any
    # of that, at extraction: both legs are worded "Q", so they state no
    # deadline at all, let alone two different ones.
    _ALIGNED_MARKETS = [
        {**_MARKETS[0]},
        {**_MARKETS[1], "close_time": "2026-02-01T00:00:00+00:00",
         "settlement_ts": "2026-02-01T12:00:00+00:00"},
    ]
    _CANDLES = {
        "DA": [_candle(_MONDAY_TS, 0.30, 0.70)],
        "DB": [_candle(_MONDAY_TS, 0.60, 0.40)],
    }

    def test_fixture_lands_in_both_groupings_but_only_one_yields_a_candidate(self):
        # GROUPING is untouched by the cumulative-deadline rule, so the pair is
        # still discovered by both keys...
        assert len(_group_by_exact_title(self._ALIGNED_MARKETS)) == 1
        assert len(_group_by_normalized_title(self._ALIGNED_MARKETS)) == 1
        # ...but only the same-title branch produces a candidate. The legs are
        # worded "Q", so they state no deadline at all, and the time-series
        # branch refuses them.
        assert len(_extract_pairs(_group_by_exact_title(self._ALIGNED_MARKETS))) == 1
        assert _extract_pairs(_group_by_normalized_title(self._ALIGNED_MARKETS)) == []

    def test_differing_closes_yield_no_candidate_of_either_type(self):
        # regression (DR-74) — the relabel guard. The same two tickers, a week
        # apart: still discovered by both keys, and now refused by both
        # branches — the same-title one by the close gate, the time-series
        # one by the wording rule, as before.
        assert len(_group_by_exact_title(self._MARKETS)) == 1
        assert len(_group_by_normalized_title(self._MARKETS)) == 1
        assert _extract_pairs(_group_by_exact_title(self._MARKETS)) == []
        assert _extract_pairs(_group_by_normalized_title(self._MARKETS)) == []

    def _run_with_dedup_spy(self, monkeypatch, markets):
        monkeypatch.setattr(backtester, "fetch_all_settled_markets",
                            lambda *a, **k: markets)
        monkeypatch.setattr(backtester, "fetch_candlesticks",
                            lambda _c, ticker, *a, **k: self._CANDLES[ticker])

        real = backtester._drop_cross_type_duplicates
        seen: dict = {"in": [], "out": []}

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
        return trades, seen

    @staticmethod
    def _typed(cands, pair_type):
        key = frozenset({"DA", "DB"})
        return [c for c in cands
                if c["pair_type"] == pair_type
                and frozenset({c["mA"]["ticker"], c["mB"]["ticker"]}) == key]

    def test_no_time_series_duplicate_reaches_the_dedup(self, monkeypatch):
        trades, seen = self._run_with_dedup_spy(monkeypatch, self._ALIGNED_MARKETS)

        # Pass 1 produces the same-title copy ONLY: identical wording cannot
        # state two different deadlines, so the time-series copy is refused at
        # extraction and never reaches the dedup at all.
        assert len(self._typed(seen["in"], "same_title")) == 1
        assert self._typed(seen["in"], "time_series") == []
        # The dedup is therefore a no-op here, and the same-title copy is
        # carried into Pass 2 unchanged.
        assert len(self._typed(seen["out"], "same_title")) == 1
        assert self._typed(seen["out"], "time_series") == []

        assert len(trades) == 1
        assert trades[0].pair_type == "same_title"
        # The same-title copy canonicalizes A as the pricier side (DB)
        assert (trades[0].ticker_a, trades[0].ticker_b) == ("DB", "DA")

    def test_nothing_reaches_the_dedup_when_the_closes_differ(self, monkeypatch):
        # regression (DR-74) — the relabel guard end to end: with the closes a
        # week apart the same-title copy is refused and no time-series copy
        # replaces it, so the dedup sees no candidate for these tickers and no
        # trade is made.
        trades, seen = self._run_with_dedup_spy(monkeypatch, self._MARKETS)
        assert self._typed(seen["in"], "same_title") == []
        assert self._typed(seen["in"], "time_series") == []
        assert trades == []

    def test_dated_identical_wording_is_same_title_only(self):
        # control — kills a mutant that drops the spans-differ requirement
        # (`return True` in place of `spans_a != spans_b`). The class
        # docstring's exclusivity claim was cited to
        # test_fixture_lands_in_both_groupings_but_only_one_yields_a_candidate
        # above, whose fixture states NO deadline at all ("Q"), so it only
        # ever exercises the spans-differ conjunct's "both empty" branch. This
        # fixture states a deadline IDENTICALLY on two DIFFERENT series, so
        # the refusal comes from the spans-differ conjunct alone (the two
        # spans are equal) — the one-series rule does not fire here, since
        # _same_series_dicts is False on two different series. Both legs close
        # at one instant, so the same-title close gate (DR-74) admits the
        # same-title copy. Mirror of test_scanner.py::TestDeadlineGuardFinders::
        # test_dated_identical_wording_is_same_title_only.
        title = "Will X happen by Dec 31, 2026?"
        mA = _md("A1", "EVA-1", title=title, event_title="EV")
        mA["close_time"] = "2026-12-01T00:00:00Z"
        mB = _md("B1", "EVB-1", title=title, event_title="EV")
        mB["close_time"] = "2026-12-01T00:00:00Z"
        assert len(_group_by_normalized_title([mA, mB])) == 1
        assert backtester._deadline_profile_dict(mA) == (
            scanner.DEADLINE_CUMULATIVE, ("by dec 31, 2026",),
        )
        assert backtester._deadline_profile_dict(mB) == (
            scanner.DEADLINE_CUMULATIVE, ("by dec 31, 2026",),
        )
        assert backtester._identical_wording_dicts(mA, mB) is True
        assert backtester._same_series_dicts(mA, mB) is False
        assert backtester._comparable_closes_dicts(mA, mB) is not None
        assert len(_extract_pairs(_group_by_exact_title([mA, mB]))) == 1
        assert _extract_pairs(_group_by_normalized_title([mA, mB])) == []

        # Positive control: B's wording states a DIFFERENT deadline on the
        # same two series, at the same prices and the same close instant — the
        # WORDING is the only thing that changed. The spans now differ, so the
        # time-series candidate forms — proving the [] above comes from the
        # spans-differ conjunct rather than from the close times or the two
        # series being distinct.
        mB3 = _md("B1", "EVB-1", title="Will X happen by Dec 20, 2026?", event_title="EV")
        mB3["close_time"] = mA["close_time"]
        assert len(_extract_pairs(_group_by_normalized_title([mA, mB3]))) == 1

    @pytest.mark.parametrize("title", ["Q", "Will X happen by Dec 31, 2026?"])
    def test_identical_wording_closing_apart_forms_no_pair_of_either_type(self, title):
        # regression (DR-74) — the relabel guard through both _extract_pairs
        # branches, undated and dated. Identical wording on two DIFFERENT
        # series closing 19 days apart: the same-title branch refuses it at
        # the close gate, and the time-series branch refuses identical wording
        # whether it states no deadline ("Q") or one deadline twice. Mirror of
        # test_scanner.py::TestDeadlineGuardFinders::
        # test_identical_wording_closing_apart_forms_no_pair_of_either_type.
        mA = _md("A1", "EVA-1", title=title, event_title="EV")
        mA["close_time"] = "2026-12-01T00:00:00Z"
        mB = _md("B1", "EVB-1", title=title, event_title="EV")
        mB["close_time"] = "2026-12-20T00:00:00Z"
        assert backtester._same_series_dicts(mA, mB) is False
        assert len(_group_by_exact_title([mA, mB])) == 1
        assert len(_group_by_normalized_title([mA, mB])) == 1
        assert _extract_pairs(_group_by_exact_title([mA, mB])) == []
        assert _extract_pairs(_group_by_normalized_title([mA, mB])) == []
        # Positive control: B on A's close instant — the same-title pair forms.
        mB["close_time"] = mA["close_time"]
        assert len(_extract_pairs(_group_by_exact_title([mA, mB]))) == 1


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
    b 0.3671 (= net / (0.70 + 0.0315) — DR-62 put the fee in Kelly's
    denominator; it was 0.3836 over the fee-less 0.70), p = 1 - 0.75*0.30 =
    0.775, f* = 0.1620 (was 0.1884; still below the 0.20 cap, so Kelly sizes
    it). Pass 2: budget 1620.11 => raw n 2314, shrunk to 2214 by the fee loop
    (cost 1549.80, exact fees 69.75, cash out 1619.55), win profit 594.45.
    Settlement: event by EA => +594.45; never by EB => +594.45; in between =>
    -1619.55; EA yes / EB no is a premise violation and is excluded with a
    counted WARNING.
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
        # model THROUGH run_backtest rather than a hardcoded number. b's
        # denominator carries the fee: the losing cell loses cost + fees, so
        # that is the capital actually at risk (DR-62).
        fee = fee_per_pair_approx(pA, nB)
        net = (1.0 - pA - nB) - fee
        b = net / (pA + nB + fee)
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
        # 0.1884 before DR-62 put the fee in Kelly's denominator; the gate is
        # strictly tighter now, so this pair sizes smaller than it used to.
        assert expected_f == pytest.approx(0.1620, abs=5e-4)
        assert expected_f < BUDGET_FRACTION  # Kelly, not the cap, sized this pair
        assert t.kelly_fraction == pytest.approx(expected_f)
        assert t.balance_at_entry == pytest.approx(10_000.0)
        assert t.n == 2214
        # Sized on the LEG prices (pA + nB), never on (nA + pB)
        assert t.total_cost == pytest.approx(2214 * (self._PA + self._NB))
        assert t.total_cost == pytest.approx(1549.80)
        assert t.fees == pytest.approx(
            fee_leg_exact(2214, self._PA) + fee_leg_exact(2214, self._NB))
        assert t.fees == pytest.approx(69.75)
        assert t.total_cost + t.fees <= 10_000.0 * t.kelly_fraction + 1e-9
        assert t.expected_payoff == pytest.approx(594.45)

    def test_event_by_earlier_deadline_wins(self, monkeypatch):
        # EA yes, EB yes: YES on EA pays n, NO on EB worthless
        trades, equity = self._run(monkeypatch, "yes", "yes")
        assert len(trades) == 1
        t = trades[0]
        self._assert_entry_and_sizing(t)
        assert t.actual_payoff == pytest.approx(2214.0)
        assert t.profit == pytest.approx(594.45)
        assert t.slippage == pytest.approx(0.0, abs=1e-9)
        assert float(equity["portfolio_value"].iloc[-1]) == pytest.approx(10_594.45)

    def test_event_never_by_later_deadline_wins(self, monkeypatch):
        # EA no, EB no: NO on EB pays n, YES on EA worthless
        trades, equity = self._run(monkeypatch, "no", "no")
        assert len(trades) == 1
        t = trades[0]
        self._assert_entry_and_sizing(t)
        assert t.actual_payoff == pytest.approx(2214.0)
        assert t.profit == pytest.approx(594.45)
        assert t.slippage == pytest.approx(0.0, abs=1e-9)
        assert float(equity["portfolio_value"].iloc[-1]) == pytest.approx(10_594.45)

    def test_event_in_between_loses_the_full_stake(self, monkeypatch):
        # EA no, EB yes: both legs worthless — the loss cell
        trades, equity = self._run(monkeypatch, "no", "yes")
        assert len(trades) == 1
        t = trades[0]
        self._assert_entry_and_sizing(t)
        assert t.actual_payoff == pytest.approx(0.0)
        assert t.profit == pytest.approx(-1619.55)
        assert t.profit == pytest.approx(-(t.total_cost + t.fees))
        assert t.slippage == pytest.approx(-1619.55 - 594.45)
        assert float(equity["portfolio_value"].iloc[-1]) == pytest.approx(10_000.0 - 1619.55)

    def test_a_tier_off_simulation_labels_its_premise_warning(self, monkeypatch, caplog):
        # The premise check runs inside every simulation, the backtest-only
        # tier-floors-off family's included: a count from that family says
        # so, and the tier-on WARNING is exactly the text it always was.
        candles = {"EA": [_candle(_MONDAY_TS, self._PA, self._NA)],
                   "EB": [_candle(_MONDAY_TS, self._PB, self._NB)]}
        markets = self._markets("yes", "no")
        monkeypatch.setattr(backtester, "fetch_all_settled_markets",
                            lambda *a, **k: markets)
        monkeypatch.setattr(backtester, "fetch_candlesticks",
                            lambda _c, ticker, *a, **k: candles[ticker])
        entries, _ = backtester._prepare_entries(MagicMock(), MagicMock(), date(2026, 1, 1),
                                                 True, None)
        assert len(entries) == 1

        def premise_lines(**kw):
            caplog.clear()
            backtester._simulate_at_discount(entries, date(2026, 1, 1), 10_000.0, **kw)
            return [r.getMessage() for r in caplog.records
                    if r.levelname == "WARNING" and "cumulative-deadline premise" in r.getMessage()]

        with caplog.at_level(logging.WARNING):
            on = premise_lines()
            off = premise_lines(tier_floors=False)
        assert len(on) == len(off) == 1
        assert on[0].startswith("Excluded 1 time-series candidate(s)")
        assert on[0].endswith("(see the outcome-label coverage line)")
        assert off[0] == on[0] + " [tier floors off]"

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
        # DR-72: three named causes, not one guessed one.
        assert "snapshot markets" in premise[0]
        assert "recurring windows" in premise[0]
        assert "REALIZED close" in premise[0]
        assert "outcome-label coverage" in premise[0]
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
        # Later candle 0.85 / 0.15: gap 0.55, legs 0.45, p = 0.5875 — the
        # uncapped Kelly fraction is ~0.216, so BUDGET_FRACTION binds. The gap
        # had to widen from 0.40 to 0.55 when DR-62 put the fee into Kelly's
        # denominator: at the old 0.70 / 0.30 candle f* is now 0.1905, just
        # under the cap, so that fixture no longer exercises the cap at all.
        trades, _ = self._run(monkeypatch, "yes", "yes", eb_yes=0.85, eb_no=0.15)
        assert len(trades) == 1
        t = trades[0]
        assert t.pair_type == "time_series"
        assert t.entry_nB == pytest.approx(0.15)
        uncapped = self._expected_kelly(self._PA, 0.15, 0.85)
        assert uncapped == pytest.approx(0.216, abs=1e-3)
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
        raw_entries, _coverage = backtester._prepare_entries(
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
        # The sentinel lives on ELEMENT 0 of the returned pair: a caller that
        # forgot to unpack would hold a 2-tuple, which is never None, so its
        # guard would silently go false. Assert the shape explicitly.
        raw_entries, coverage = backtester._prepare_entries(
            MagicMock(), MagicMock(), today + timedelta(days=1), True, None
        )
        assert raw_entries is None
        # No census either: the fetch never ran, so there was no corpus.
        assert coverage is None

    def _prepared(self, monkeypatch, result_a, result_b, eb_yes=None, eb_no=None):
        # _run installs the fixture's fetch monkeypatches and leaves them in
        # force, so the prologue below replays exactly the same markets and
        # candles run_backtest just consumed (the idiom
        # test_default_k_equals_explicit_config_k already uses).
        self._run(monkeypatch, result_a, result_b, eb_yes=eb_yes, eb_no=eb_no)
        raw_entries, _coverage = backtester._prepare_entries(
            MagicMock(), MagicMock(), date(2026, 1, 1), True, None
        )
        return raw_entries

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


class TestRunBacktestSameDateLegOrder:
    """End-to-end proof of TS-06 through run_backtest.

    Both legs close on 2026-02-09 — EARLY at 09:00Z, LATE at 21:00Z — and the
    market list holds LATE first, which is the order _extract_pairs preserves
    (its close-time sort is by DATE and stable, so an equal-date pair keeps
    group order). Before TS-06 _find_entry decided its swap on those same
    dates, so the pair was a tie and "market A" was simply LATE.
    """

    @staticmethod
    def _markets(result_early: str, result_late: str) -> list[dict]:
        # LATE first on purpose — this is the ordering the defect needed.
        # Titles normalize to one key (the date text is stripped) but are not
        # exact-title equal, so only the time-series grouping forms the pair
        # and the cross-type dedup has nothing to drop.
        return [
            {"ticker": "LATE", "event_ticker": "EVL", "event_title": "EV",
             "title": "Team wins by February 10, 2026", "subtitle": "",
             "result": result_late,
             "open_time": "2026-01-01T00:00:00+00:00",
             "close_time": "2026-02-09T21:00:00+00:00",
             "settlement_ts": "2026-02-09T23:00:00+00:00"},
            {"ticker": "EARLY", "event_ticker": "EVE", "event_title": "EV",
             "title": "Team wins by February 9, 2026", "subtitle": "",
             "result": result_early,
             "open_time": "2026-01-01T00:00:00+00:00",
             "close_time": "2026-02-09T09:00:00+00:00",
             "settlement_ts": "2026-02-09T23:00:00+00:00"},
        ]

    def _run(self, monkeypatch, result_early, result_late,
             early_quotes, late_quotes):
        candles = {
            "EARLY": [_candle(_MONDAY_TS, *early_quotes)],
            "LATE":  [_candle(_MONDAY_TS, *late_quotes)],
        }
        markets = self._markets(result_early, result_late)
        monkeypatch.setattr(backtester, "fetch_all_settled_markets",
                            lambda *a, **k: markets)
        monkeypatch.setattr(backtester, "fetch_candlesticks",
                            lambda _c, ticker, *a, **k: candles[ticker])
        return run_backtest(
            hist_client=MagicMock(), live_client=MagicMock(),
            start_date=date(2026, 1, 1), initial_balance=10_000.0,
        )

    def test_fixture_is_a_time_series_group_only(self):
        markets = self._markets("yes", "yes")
        assert len(_group_by_normalized_title(markets)) == 1
        assert _group_by_exact_title(markets) == {}
        # And the pair really does reach _find_entry LATE-first.
        pairs = _extract_pairs(_group_by_normalized_title(markets))
        assert len(pairs) == 1
        assert pairs[0][0]["ticker"] == "LATE"

    def test_same_date_pair_enters_with_the_earlier_leg_as_a(self, monkeypatch):
        # EARLY cheap (0.30) / LATE dear (0.60): the anomaly the strategy
        # disputes. Before TS-06 the untaken swap made pB − pA read as −0.30
        # and the pair was silently dropped.
        trades, _ = self._run(monkeypatch, "yes", "yes",
                              early_quotes=(0.30, 0.70), late_quotes=(0.60, 0.40))
        assert len(trades) == 1
        t = trades[0]
        assert t.pair_type == "time_series"
        assert (t.ticker_a, t.ticker_b) == ("EARLY", "LATE")
        assert t.entry_pA == pytest.approx(0.30)
        assert t.entry_nB == pytest.approx(0.40)
        assert t.deadline_gap_days == 0

    def test_same_date_inverted_pricing_is_not_a_premise_violation(
        self, monkeypatch, caplog,
    ):
        # EARLY dear (0.60) / LATE cheap (0.30) is never a candidate, and the
        # settlement is the genuine in-between (EARLY no, LATE yes). Before
        # TS-06 the pair entered with its legs inverted, so that settlement
        # read as the impossible A=YES/B=NO cell and was booked as a premise
        # violation — excluded from P&L and dropped from the interval-discount
        # calibration's denominator.
        with caplog.at_level("WARNING"):
            trades, equity = self._run(monkeypatch, "no", "yes",
                                       early_quotes=(0.60, 0.40),
                                       late_quotes=(0.30, 0.70))
        assert trades == []
        assert not any("cumulative-deadline premise" in r.getMessage()
                       for r in caplog.records)
        assert equity["portfolio_value"].min() == pytest.approx(10_000.0)


def _cal_entry(gap_days, pA, pB, result_a, result_b, pair_type="time_series",
               event_ticker=None):
    """Build one _prepare_entries record shaped as _interval_calibration reads it.

    event_ticker, when given, is market A's event ticker; by default mA carries
    no event_ticker key at all, the shape every older fixture here uses."""
    mA = {"ticker": "A", "result": result_a}
    if event_ticker is not None:
        mA["event_ticker"] = event_ticker
    return {
        "pair_type": pair_type,
        "canon": "canon",
        "group_key": "group",
        "entry": {
            "entry_date": date(2026, 1, 5),
            "pA": pA, "pB": pB, "nA": 1.0 - pA, "nB": 1.0 - pB,
            "mA": mA,
            "mB": {"ticker": "B", "result": result_b},
            "gap_days": gap_days,
        },
    }


class TestIntervalCalibration:
    """_interval_calibration: the k-independent empirical-discount measurement."""

    def test_returns_none_without_time_series_candidates(self):
        # None means there is no table to print (nothing measurable) — but
        # since DR-72 the caller (_log_interval_calibration) is no longer
        # silent on None: it logs one explanatory line instead of an
        # all-zero table. See TestLogIntervalCalibration for that line.
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


class TestCalibrationObservationsAreCarried:
    """IntervalCalibration.observations: the population the pooled row was
    reduced from, carried out so a report (the dashboard's k-hat by category,
    tag and spread band) can regroup it through _calibration_bucket rather than
    re-derive which entries count."""

    def _mixed_entries(self):
        return [
            _cal_entry(3, 0.10, 0.70, "no", "yes", event_ticker="KXSPACEX-14"),  # in-between
            _cal_entry(None, 0.60, 0.50, "yes", "no", pair_type="same_title",
                       event_ticker="KXST-1"),                                     # same-title
            _cal_entry(5, 0.20, 0.60, "yes", "yes", event_ticker="KXFED-26"),     # by A
            _cal_entry(3, 0.10, 0.70, "", "yes", event_ticker="KXVOID-1"),         # unsettled
            _cal_entry(20, 0.25, 0.75, "yes", "no", event_ticker="KXBAD-1"),       # violation
            _cal_entry(20, 0.30, 0.80, "no", "no", event_ticker="KXFED-27"),      # never by B
            # No gap: it lands in no gap band but IS part of the pooled row,
            # so it must be carried too
            _cal_entry(None, 0.10, 0.70, "no", "yes", event_ticker="KXNOGAP-1"),
        ]

    def test_observations_are_exactly_the_measured_population_in_order(self):
        calib = _interval_calibration(self._mixed_entries())
        # Same-title, unsettled and premise-violating entries never become one
        assert [o.event_ticker for o in calib.observations] == [
            "KXSPACEX-14", "KXFED-26", "KXFED-27", "KXNOGAP-1"]
        assert len(calib.observations) == calib.pooled.n == 4
        assert calib.excluded_premise_violations == 1
        assert [(o.gap_days, o.in_between) for o in calib.observations] == [
            (3, True), (5, False), (20, False), (None, True)]
        assert [o.implied for o in calib.observations] == pytest.approx(
            [0.60, 0.40, 0.50, 0.60])

    def test_reducing_the_observations_reproduces_the_pooled_row_exactly(self):
        # Exact equality, not approx: the same list in the same order through
        # the same arithmetic, which is what lets a report's "all" group —
        # reduced from the carried tuple itself, never from its groups put
        # back together — match the pooled k-hat printed beside it
        calib = _interval_calibration(self._mixed_entries(), spread_min=0.30)
        assert backtester._calibration_bucket(
            calib.pooled.label, 0.0, calib.observations) == calib.pooled
        # ...and every gap band is a sub-population of it
        for bucket in calib.buckets:
            lo, hi = (int(x) for x in bucket.label.rstrip("d").split("-"))
            members = [o for o in calib.observations
                       if o.gap_days is not None and lo <= o.gap_days <= hi]
            assert backtester._calibration_bucket(
                bucket.label, bucket.tier, members) == bucket

    def test_each_observation_names_its_event_and_ticker_prefix_category(self):
        calib = _interval_calibration([
            _cal_entry(3, 0.10, 0.70, "no", "yes", event_ticker="KXBTCMAXY-26DEC31"),
            _cal_entry(3, 0.10, 0.70, "no", "no"),   # mA carries no event ticker
        ])
        named, unnamed = calib.observations
        assert named.event_ticker == "KXBTCMAXY-26DEC31"
        # The same fallback label BacktestTrade.category carries
        assert named.category == historical.infer_category("KXBTCMAXY-26DEC31")
        assert unnamed.event_ticker == ""
        assert unnamed.category == historical.infer_category("")

    def test_the_band_floor_labels_but_never_changes_the_population(self):
        entries = self._mixed_entries()
        assert (_interval_calibration(entries, spread_min=0.30).observations
                == _interval_calibration(entries).observations)

    def test_a_violation_only_window_carries_no_observation(self):
        calib = _interval_calibration([_cal_entry(3, 0.10, 0.70, "yes", "no")])
        assert calib.observations == ()
        assert calib.pooled.n == 0

    def test_a_non_string_event_ticker_reads_as_absent(self):
        # Only a hand-edited cache can carry one; it must not raise out of
        # infer_category and end the sweep (the parent commit never read it)
        calib = _interval_calibration([
            _cal_entry(3, 0.10, 0.70, "no", "yes", event_ticker=12345)])
        (obs,) = calib.observations
        assert (obs.event_ticker, obs.category) == ("", historical.infer_category(""))

    def test_an_observation_is_filed_like_the_trade_of_its_entry(self, monkeypatch):
        # Both read market A of the SAME entry (_find_entry's canonicalized
        # leg), so a report files a trade and the k-hat observation of its
        # entry under one series. The golden fixture holds a ladder and a
        # cross-event time-series entry, both traded.
        golden = TestPrepareEntriesGolden()
        golden._patch(monkeypatch)
        entries, _ = backtester._prepare_entries(
            MagicMock(), MagicMock(), golden._START, True, None,
            same_event_ladders=True,
        )
        point = backtester._simulate_at_discount(entries, golden._START, 10_000.0)
        observed = {(o.event_ticker, o.category)
                    for o in _interval_calibration(entries).observations}
        ts_trades = [t for t in point.trades if t.pair_type == "time_series"]
        assert {t.event_ticker for t in ts_trades} == {"EVA", "KXSTARSHIP-14"}
        for t in ts_trades:
            assert (t.event_ticker, t.category) in observed

    def test_the_carrier_is_immutable_and_hand_built_calibrations_still_build(self):
        obs = backtester.CalibrationObservation(3, 0.5, True, "KX-1", "Other")
        with pytest.raises(AttributeError):
            obs.implied = 0.1  # frozen
        # event_ticker and category are required (DR-71's reasoning): a
        # construction that forgot them must fail, not file under ""
        with pytest.raises(TypeError):
            backtester.CalibrationObservation(3, 0.5, True)
        bucket = backtester.IntervalCalibrationBucket("POOLED", 0.0, 1, 0.0, 0.5, 0.0)
        # observations is appended with a default, so "not carried" is told
        # from "no candidate" by len(observations) != pooled.n
        hand_built = backtester.IntervalCalibration(bucket, [], 0)
        assert hand_built.observations == ()
        assert len(hand_built.observations) != hand_built.pooled.n
        assert isinstance(_interval_calibration(self._mixed_entries()).observations, tuple)


class TestLogIntervalCalibration:
    """The report's presentation: one line even when there is nothing to
    measure (DR-72); sub-counts inside the report stay silent at zero."""

    @staticmethod
    def _messages(caplog):
        return [r.getMessage() for r in caplog.records]

    def test_none_logs_one_explanatory_line(self, caplog):
        with caplog.at_level("INFO"):
            _log_interval_calibration(None)
        msgs = self._messages(caplog)
        assert len(msgs) == 1
        # "with a readable settlement" rather than a bare "no entry": the None
        # branch is also taken when time-series entries DID exist but every one
        # was dropped for an unreadable outcome, so the line must not claim
        # more than the branch proves.
        assert msgs[0] == (
            "Interval-discount calibration: no time-series candidate entry "
            "with a readable settlement in this window — empirical k_hat is "
            "not measurable"
        )

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


class TestRunBacktestSweep:
    """run_backtest_sweep: one preparation pass, many discounts, one calibration.

    Drives TestRunBacktestTimeSeriesFlow's pinned fixture (EA/EB, 13-day gap,
    YES asks 0.30/0.60, NO asks 0.70/0.40) rather than copying its numbers, so
    the sweep can never be measured against a second, drifting copy of them.
    """

    _START = date(2026, 1, 1)

    @staticmethod
    def _patch(monkeypatch, result_a="yes", result_b="yes"):
        flow = TestRunBacktestTimeSeriesFlow
        markets = flow._markets(result_a, result_b)
        candles = {
            "EA": [_candle(_MONDAY_TS, flow._PA, flow._NA)],
            "EB": [_candle(_MONDAY_TS, flow._PB, flow._NB)],
        }
        monkeypatch.setattr(backtester, "fetch_all_settled_markets",
                            lambda *a, **k: markets)
        monkeypatch.setattr(backtester, "fetch_candlesticks",
                            lambda _c, ticker, *a, **k: candles[ticker])

    def _sweep(self, monkeypatch, result_a="yes", result_b="yes", **kwargs):
        self._patch(monkeypatch, result_a, result_b)
        return run_backtest_sweep(
            hist_client=MagicMock(), live_client=MagicMock(),
            start_date=self._START, initial_balance=10_000.0, **kwargs,
        )

    def test_grid_always_contains_the_effective_k(self, monkeypatch):
        # No override: the primary sits at the config discount, which is also
        # a standard grid point, so the grid is the standard one.
        result = self._sweep(monkeypatch)
        assert result.primary.k == TIME_SERIES_INTERVAL_PROB_DISCOUNT
        assert [p.k for p in result.points] == sorted(
            set(INTERVAL_DISCOUNT_SWEEP) | {TIME_SERIES_INTERVAL_PROB_DISCOUNT})
        # points is ascending, and primary is the SAME object in it — not an
        # equal copy that could drift from it
        assert [p.k for p in result.points] == sorted(p.k for p in result.points)
        assert any(p is result.primary for p in result.points)

    def test_off_grid_override_gets_its_own_exact_point(self, monkeypatch):
        # 0.62 is not on the standard grid: it must appear at exactly 0.62,
        # not be rounded to the nearest standard point.
        result = self._sweep(monkeypatch, interval_discount=0.62)
        assert result.primary.k == 0.62
        assert 0.62 in [p.k for p in result.points]
        assert len(result.points) == len(INTERVAL_DISCOUNT_SWEEP) + 1
        assert any(p is result.primary for p in result.points)

    def test_on_grid_override_is_not_duplicated(self, monkeypatch):
        # An override that already sits on the grid must not add a second
        # point at the same k — the union is a set, not a concatenation.
        on_grid = INTERVAL_DISCOUNT_SWEEP[0]
        result = self._sweep(monkeypatch, interval_discount=on_grid)
        assert result.primary.k == on_grid
        assert len(result.points) == len(INTERVAL_DISCOUNT_SWEEP)
        assert [p.k for p in result.points].count(on_grid) == 1

    def test_sweep_false_yields_one_point(self, monkeypatch):
        result = self._sweep(monkeypatch, sweep=False)
        assert result.points == [result.primary]
        assert result.primary.k == TIME_SERIES_INTERVAL_PROB_DISCOUNT
        assert len(result.primary.trades) == 1

    def test_sweep_false_still_honours_the_override(self, monkeypatch):
        result = self._sweep(monkeypatch, interval_discount=0.50, sweep=False)
        assert [p.k for p in result.points] == [0.50]

    def test_primary_reproduces_run_backtest(self, monkeypatch):
        # The whole blast-radius promise: run_backtest is unchanged, and the
        # sweep's primary is the same simulation by a different door.
        self._patch(monkeypatch)
        trades, equity = run_backtest(
            hist_client=MagicMock(), live_client=MagicMock(),
            start_date=self._START, initial_balance=10_000.0,
        )
        result = self._sweep(monkeypatch, sweep=False)
        assert [astuple(t) for t in result.primary.trades] == [
            astuple(t) for t in trades]
        pd.testing.assert_frame_equal(result.primary.equity_df, equity)

    def test_preparation_and_calibration_run_once(self, monkeypatch):
        # The expensive, network-bound half must not be repeated per discount;
        # the calibration must not either, because it is k-independent.
        self._patch(monkeypatch)
        calls = {"prepare": 0, "calibrate": 0, "simulate": 0}
        real = {
            "prepare": backtester._prepare_candidates,
            "calibrate": backtester._interval_calibration,
            "simulate": backtester._simulate_at_discount,
        }

        def counted(name):
            def wrapper(*a, **kw):
                calls[name] += 1
                return real[name](*a, **kw)
            return wrapper

        monkeypatch.setattr(backtester, "_prepare_candidates", counted("prepare"))
        monkeypatch.setattr(backtester, "_interval_calibration", counted("calibrate"))
        monkeypatch.setattr(backtester, "_simulate_at_discount", counted("simulate"))

        result = run_backtest_sweep(
            hist_client=MagicMock(), live_client=MagicMock(),
            start_date=self._START, initial_balance=10_000.0,
        )
        assert calls["prepare"] == 1
        assert calls["calibrate"] == 1
        # Once per grid point — the primary is simulated once and then reused,
        # never re-run at its own k.
        assert calls["simulate"] == len(result.points)

    def test_override_actually_bites_across_the_grid(self, monkeypatch):
        # k = 1.00 takes the market's in-between mass at face value, which
        # drives Kelly <= 0 for every time-series pair — the sharpest proof
        # that k is threaded all the way through the simulation.
        result = self._sweep(monkeypatch)
        by_k = {p.k: p for p in result.points}
        assert len(by_k[TIME_SERIES_INTERVAL_PROB_DISCOUNT].trades) == 1
        assert by_k[1.00].trades == []

    def test_calibration_is_attached_and_kelly_independent(self, monkeypatch):
        # The in-between cell. The pair is counted in the calibration whatever
        # any point's Kelly gate did with it.
        result = self._sweep(monkeypatch, result_a="no", result_b="yes")
        calib = result.calibration
        assert calib is not None
        assert calib.pooled.n == 1
        assert calib.pooled.realised_rate == pytest.approx(1.0)
        assert calib.excluded_premise_violations == 0
        assert [b.label for b in calib.buckets] == ["8-15d"]

    def test_calibration_is_none_without_time_series_candidates(self, monkeypatch):
        # A premise-violating pair is excluded from the denominator, but the
        # exclusion itself is still worth reporting.
        result = self._sweep(monkeypatch, result_a="yes", result_b="no")
        assert result.calibration.excluded_premise_violations == 1
        assert result.calibration.pooled.n == 0
        assert result.primary.trades == []

    # An infeasible window, frozen exactly as TestRunBacktestFeasibilityPreCheck
    # does it: Tuesday start, same-week Friday "today", so [Tue, Fri] holds no
    # Monday checkpoint at all.
    _INFEASIBLE_START = date(2026, 8, 25)   # Tuesday
    _INFEASIBLE_TODAY = date(2026, 8, 28)   # Friday, same week

    def _infeasible(self, monkeypatch, **kwargs):
        fixed = datetime(self._INFEASIBLE_TODAY.year, self._INFEASIBLE_TODAY.month,
                         self._INFEASIBLE_TODAY.day, 12, tzinfo=UTC)
        frozen = type("FrozenDateTime",
                      (TestRunBacktestFeasibilityPreCheck._FrozenDateTime,),
                      {"_fixed": fixed})
        monkeypatch.setattr(backtester, "datetime", frozen)
        monkeypatch.setattr(backtester, "fetch_all_settled_markets",
                            lambda *a, **k: pytest.fail("fetch must be skipped"))
        return run_backtest_sweep(
            hist_client=MagicMock(), live_client=MagicMock(),
            start_date=self._INFEASIBLE_START, initial_balance=10_000.0, **kwargs,
        )

    def test_feasibility_short_circuit_yields_one_empty_point(self, monkeypatch):
        # No Monday checkpoint exists, so nothing can be simulated at ANY
        # discount. The caller still gets the normal shape — one point, a flat
        # curve — so no special case is needed downstream.
        result = self._infeasible(monkeypatch)
        assert result.points == [result.primary]
        assert result.primary.trades == []
        assert result.primary.k == TIME_SERIES_INTERVAL_PROB_DISCOUNT
        assert result.calibration is None
        assert list(result.primary.equity_df.columns) == [
            "date", "portfolio_value", "daily_return"]
        assert result.primary.equity_df["portfolio_value"].max() == pytest.approx(10_000.0)
        assert result.primary.equity_df["portfolio_value"].min() == pytest.approx(10_000.0)

    def test_feasibility_short_circuit_reports_the_override(self, monkeypatch):
        # Even with nothing to simulate, the empty point must carry the k the
        # caller asked for — a report reading primary.k must not be told 0.75.
        result = self._infeasible(monkeypatch, interval_discount=0.62)
        assert result.primary.k == 0.62
        assert result.points == [result.primary]


class TestSimulationsAreLabelledWithTheirDiscount:
    """
    TS-21: "Backtest complete: N trades" appeared 13 times per default run with
    nothing distinguishing them, the primary's copy printed BEFORE the sweep
    was announced, and the "Sweeping i/13" counter started at 2 because the
    primary's slot was never numbered.
    """

    def test_completion_line_names_the_resolved_discount(self, caplog):
        with caplog.at_level(logging.INFO):
            backtester._simulate_at_discount([], date(2026, 1, 1), 1000.0, k=0.62)
        lines = [r.getMessage() for r in caplog.records
                 if "Backtest complete" in r.getMessage()]
        assert len(lines) == 1
        assert "k=0.620" in lines[0]

    def test_completion_line_resolves_the_none_sentinel(self, caplog):
        # The sentinel must never reach the log — a reader needs the number
        # that was actually priced, not "None".
        with caplog.at_level(logging.INFO):
            backtester._simulate_at_discount([], date(2026, 1, 1), 1000.0, k=None)
        line = next(r.getMessage() for r in caplog.records
                    if "Backtest complete" in r.getMessage())
        assert f"k={TIME_SERIES_INTERVAL_PROB_DISCOUNT:.3f}" in line
        assert "None" not in line

    def test_every_swept_point_is_distinguishable(self, monkeypatch, caplog):
        monkeypatch.setattr(backtester, "INTERVAL_DISCOUNT_SWEEP", [0.50, 0.75])
        monkeypatch.setattr(backtester, "_prepare_candidates",
                            lambda *a, **k: backtester._Candidates(
                                all_pairs=[], candles_by_ticker={},
                                label_coverage=None, start_date=date(2026, 1, 1),
                                max_horizon_days=None,
                                same_event_ladders=k.get("same_event_ladders")))
        monkeypatch.setattr(backtester, "_interval_calibration", lambda *a, **k: None)
        with caplog.at_level(logging.INFO):
            backtester.run_backtest_sweep(
                MagicMock(), MagicMock(), date(2026, 1, 1), 1000.0,
            )
        completions = [r.getMessage() for r in caplog.records
                       if "Backtest complete" in r.getMessage()]
        # One per grid point, each naming a different k
        assert len(completions) == len({c.split(":")[0] for c in completions})

    def test_the_primary_slot_is_announced(self, monkeypatch, caplog):
        monkeypatch.setattr(backtester, "_prepare_candidates",
                            lambda *a, **k: backtester._Candidates(
                                all_pairs=[], candles_by_ticker={},
                                label_coverage=None, start_date=date(2026, 1, 1),
                                max_horizon_days=None,
                                same_event_ladders=k.get("same_event_ladders")))
        monkeypatch.setattr(backtester, "_interval_calibration", lambda *a, **k: None)
        with caplog.at_level(logging.INFO):
            backtester.run_backtest_sweep(
                MagicMock(), MagicMock(), date(2026, 1, 1), 1000.0, sweep=False,
            )
        assert any("Simulating the primary interval discount" in r.getMessage()
                   for r in caplog.records)

    @pytest.mark.parametrize("passed", [None, True, False])
    def test_the_ladder_flag_reaches_prepare_entries_unresolved(self, monkeypatch,
                                                                passed):
        # DR-73c: the sentinel must survive the hand-off, or a run-level
        # override and a monkeypatched constant would both be silently
        # pre-resolved here instead of at the one place that reads them.
        seen: dict = {}

        def _fake(*args, **kwargs):
            seen["ladders"] = kwargs.get("same_event_ladders", "MISSING")
            return backtester._Candidates(
                all_pairs=[], candles_by_ticker={}, label_coverage=None,
                start_date=date(2026, 1, 1), max_horizon_days=None,
                same_event_ladders=kwargs.get("same_event_ladders"))

        monkeypatch.setattr(backtester, "_prepare_candidates", _fake)
        monkeypatch.setattr(backtester, "_interval_calibration", lambda *a, **k: None)
        backtester.run_backtest_sweep(
            MagicMock(), MagicMock(), date(2026, 1, 1), 1000.0, sweep=False,
            same_event_ladders=passed,
        )
        assert seen["ladders"] is passed

    @pytest.mark.parametrize("configured,expected", [(True, "on"), (False, "off")])
    def test_the_resolved_ladder_setting_is_logged_with_its_source(
        self, monkeypatch, caplog, configured, expected,
    ):
        monkeypatch.setattr(backtester, "TIME_SERIES_SAME_EVENT_LADDERS", configured)
        monkeypatch.setattr(backtester, "_prepare_candidates",
                            lambda *a, **k: backtester._Candidates(
                                all_pairs=[], candles_by_ticker={},
                                label_coverage=None, start_date=date(2026, 1, 1),
                                max_horizon_days=None,
                                same_event_ladders=k.get("same_event_ladders")))
        monkeypatch.setattr(backtester, "_interval_calibration", lambda *a, **k: None)
        with caplog.at_level(logging.INFO):
            backtester.run_backtest_sweep(
                MagicMock(), MagicMock(), date(2026, 1, 1), 1000.0, sweep=False,
            )
        assert (f"Same-event deadline ladders (DR-73): {expected} "
                "(config.TIME_SERIES_SAME_EVENT_LADDERS)") in caplog.text
        # And an override says so, rather than looking like the configured value
        caplog.clear()
        with caplog.at_level(logging.INFO):
            backtester.run_backtest_sweep(
                MagicMock(), MagicMock(), date(2026, 1, 1), 1000.0, sweep=False,
                same_event_ladders=not configured,
            )
        other = "off" if expected == "on" else "on"
        assert (f"Same-event deadline ladders (DR-73): {other} "
                "(run-level override)") in caplog.text

    def test_run_backtest_leaves_the_ladder_flag_to_the_config(self, monkeypatch):
        # run_backtest keeps its exact pre-DR-73 signature, so it must pass no
        # ladder argument at all — the None default is what resolves the
        # constant at call time, i.e. the value live sizing uses.
        seen: dict = {}

        def _fake(*args, **kwargs):
            seen["kwargs"] = kwargs
            return None, None

        monkeypatch.setattr(backtester, "_prepare_entries", _fake)
        run_backtest(hist_client=MagicMock(), live_client=MagicMock(),
                     start_date=date(2026, 1, 1), initial_balance=1000.0)
        assert "same_event_ladders" not in seen["kwargs"]


class TestFeasibilityWindowIsMeasuredInUTC:
    """
    TS-13: the feasibility pre-check used date.today() — a LOCAL date — while
    _monday_timestamps builds 09:00 UTC checkpoints and _build_equity_curve
    already used datetime.now(UTC).date(). West of UTC the local date lags the
    UTC one for the first hours of each UTC day (7 of every 24 on a PDT host),
    so a window whose only Monday is the current UTC day short-circuited to
    zero trades and reported the run as structurally impossible when it was not.
    """

    # 2026-08-31 is a Monday. At 02:00 UTC that day it is still Sunday
    # 2026-08-30 in PDT (UTC-7) — the exact instant old and new disagree.
    _UTC_INSTANT = datetime(2026, 8, 31, 2, 0, tzinfo=UTC)
    _START = date(2026, 8, 30)   # Sunday

    def _freeze(self, monkeypatch, moment: datetime):
        """Freeze BOTH date seams at one real instant.

        backtester.datetime.now(UTC) is what the code reads now; backtester
        .date.today() is what it read before TS-13. Freezing only the new one
        would leave pre-fix code on the real clock, and these tests would then
        pass or fail for reasons unrelated to the fix.
        """
        class _Frozen(datetime):
            @classmethod
            def now(cls, tz=None):
                return moment if tz is None else moment.astimezone(tz)

        class _FrozenDate(date):
            @classmethod
            def today(cls):
                return moment.astimezone().date()   # the LOCAL date, as before

        monkeypatch.setattr(backtester, "datetime", _Frozen)
        monkeypatch.setattr(backtester, "date", _FrozenDate)

    def test_the_window_includes_the_current_utc_day(self, monkeypatch):
        # Under the old local-date rule feasibility_end was Sunday 08-30 and
        # [Sun, Sun] holds no Monday, so the fetch was skipped and the run
        # returned None. In UTC the end is Monday 08-31 and the checkpoint
        # exists, so the fetch must actually be reached.
        self._freeze(monkeypatch, self._UTC_INSTANT)
        reached = []
        monkeypatch.setattr(backtester, "fetch_all_settled_markets",
                            lambda *a, **k: reached.append(True) or [])
        out, coverage = backtester._prepare_entries(
            MagicMock(), MagicMock(), self._START, False, None,
        )
        assert reached == [True]
        assert out == []
        # The fetch ran, so a census was taken — over an empty corpus.
        assert coverage is not None and coverage.total == 0

    def test_a_genuinely_infeasible_window_still_short_circuits(self, monkeypatch):
        # GUARD: moving to UTC must not disarm the check. Tuesday to Friday
        # holds no Monday in either timezone.
        self._freeze(monkeypatch, datetime(2026, 8, 28, 12, 0, tzinfo=UTC))
        monkeypatch.setattr(
            backtester, "fetch_all_settled_markets",
            lambda *a, **k: pytest.fail("fetch must be skipped"),
        )
        # Element 0 carries the sentinel; element 1 is None because no corpus
        # was ever censused on this path.
        assert backtester._prepare_entries(
            MagicMock(), MagicMock(), date(2026, 8, 25), False, None,
        ) == (None, None)

    def test_the_local_date_is_not_what_is_measured(self, monkeypatch):
        # Pins the seam itself: the frozen instant's LOCAL date is behind its
        # UTC date on any timezone west of UTC, and it is the UTC one the
        # window must use. Skipped where the two agree, so the test is honest
        # about only being meaningful west of UTC.
        local_date = self._UTC_INSTANT.astimezone().date()
        if local_date == self._UTC_INSTANT.date():
            pytest.skip("host is at or east of UTC; the two dates agree here")
        assert local_date.weekday() != 0        # Sunday locally
        assert self._UTC_INSTANT.date().weekday() == 0   # Monday in UTC


# ─── PB2: band-aware entry detection, split at the band ──────────────────────

# The Monday after _MONDAY_TS (2026-01-12 09:00 UTC) — the second checkpoint a
# scan starting 2026-01-01 visits.
_MONDAY2_TS = _MONDAY_TS + 7 * 86_400


class TestPrepareEntriesGolden:
    """_prepare_entries, now composed of _prepare_candidates and one
    _entries_for_band pass at the default band, must produce EXACTLY the
    entries it produced before the split.

    The expected rows below are LITERALS captured by running main's
    _prepare_entries (fe0a758, before the split existed) over this fixture —
    not re-derived from the code under test, which would make the check
    tautological. The fixture is the TestRunBacktestSweep EA/EB time-series
    pair; a same-event ladder whose rungs close at one instant (so the stated
    gap, not close_time, orders the legs and picks the 0.30 tier — Monday 1's
    0.25 spread would clear the 0.15 tier a close gap of 0 picks, so the
    ladder entering on Monday 2 is what proves the stated gap was used), with
    the later rung listed first; a same-title pair whose B leg is the pricier
    one (canonicalized by price); a cross-event time-series pair that never
    qualifies (a pricier earlier contract on both Mondays); and two short-gap
    cross-event pairs at the two extremes a band can act on — TA/TB at a
    spread of exactly the 0.15 short tier and WA/WB at 0.98, the widest two
    live [0.01, 0.99] YES asks can make. Those two are what let the capture
    SEE a band: without them every grid band with floor <= 0.30 and ceiling
    >= 0.40 reproduced the rows, so _prepare_entries could have silently
    started banding with the golden still green. With them, any floor above
    the short tier or ceiling below 0.98 moves the rows — i.e. every band
    that could change an entry on any data (the band tests below pin both
    directions). Both are voided (result ""), so they never trade. It is run
    with ladders
    on AND off. The fetch and candle seams are mocked exactly as
    TestRunBacktestSweep mocks them.
    """

    _START = date(2026, 1, 1)

    @staticmethod
    def _markets() -> list[dict]:
        def mk(ticker, event_ticker, event_title, title, result, close):
            return {"ticker": ticker, "event_ticker": event_ticker,
                    "event_title": event_title, "title": title, "subtitle": "",
                    "result": result,
                    "open_time": "2026-01-01T00:00:00+00:00",
                    "close_time": f"{close}T00:00:00+00:00",
                    "settlement_ts": f"{close}T12:00:00+00:00"}
        return [
            mk("EA", "EVA", "EV", "Team wins by February 1, 2026", "yes", "2026-02-01"),
            mk("EB", "EVB", "EV", "Team wins by February 14, 2026", "yes", "2026-02-14"),
            mk("RUNG-LATE", "KXSTARSHIP-14", "",
               "Will SpaceX launch another Starship by March 20, 2026?", "yes", "2026-03-20"),
            mk("RUNG-EARLY", "KXSTARSHIP-14", "",
               "Will SpaceX launch another Starship by March 1, 2026?", "no", "2026-03-20"),
            mk("SA", "SERA-1", "EVS", "Q", "yes", "2026-02-01"),
            mk("SB", "SERB-1", "EVS", "Q", "yes", "2026-02-01"),
            mk("FA", "RAINA", "RAIN", "Rain falls by March 1, 2026", "no", "2026-03-01"),
            mk("FB", "RAINB", "RAIN", "Rain falls by March 10, 2026", "no", "2026-03-10"),
            mk("TA", "SNOWA", "SNOW", "Snow falls by February 1, 2026", "", "2026-02-01"),
            mk("TB", "SNOWB", "SNOW", "Snow falls by February 10, 2026", "", "2026-02-10"),
            mk("WA", "HAILA", "HAIL", "Hail falls by February 1, 2026", "", "2026-02-01"),
            mk("WB", "HAILB", "HAIL", "Hail falls by February 10, 2026", "", "2026-02-10"),
        ]

    _CANDLES = {
        "EA": [_candle(_MONDAY_TS, 0.30, 0.70)],
        "EB": [_candle(_MONDAY_TS, 0.60, 0.40)],
        "RUNG-EARLY": [_candle(_MONDAY_TS, 0.20, 0.80), _candle(_MONDAY2_TS, 0.20, 0.80)],
        "RUNG-LATE": [_candle(_MONDAY_TS, 0.45, 0.55), _candle(_MONDAY2_TS, 0.60, 0.40)],
        "SA": [_candle(_MONDAY_TS, 0.35, 0.65)],
        "SB": [_candle(_MONDAY_TS, 0.60, 0.40)],
        "FA": [_candle(_MONDAY_TS, 0.50, 0.50), _candle(_MONDAY2_TS, 0.55, 0.45)],
        "FB": [_candle(_MONDAY_TS, 0.40, 0.60), _candle(_MONDAY2_TS, 0.45, 0.55)],
        # 9-day gap (the 0.15 tier): spread 0.45 - 0.30, exactly the tier;
        # pA + nB = 0.80 <= 0.85
        "TA": [_candle(_MONDAY_TS, 0.30, 0.70)],
        "TB": [_candle(_MONDAY_TS, 0.45, 0.50)],
        # 9-day gap: spread 0.99 - 0.01 = 0.98, the widest possible;
        # pA + nB = 0.02
        "WA": [_candle(_MONDAY_TS, 0.01, 0.99)],
        "WB": [_candle(_MONDAY_TS, 0.99, 0.01)],
    }

    # (pair_type, canon, group_key, entry_date, pA, pB, nA, nB, gap_days,
    #  ticker_a, ticker_b) — captured on main @ fe0a758.
    _EA_EB = ("time_series", "ev | team wins by", "ev | team wins by",
              date(2026, 1, 5), 0.3, 0.6, 0.7, 0.4, 13, "EA", "EB")
    _LADDER = ("time_series", "will spacex launch another starship by ?",
               "will spacex launch another starship by ?",
               date(2026, 1, 12), 0.2, 0.6, 0.8, 0.4, 19, "RUNG-EARLY", "RUNG-LATE")
    _TA_TB = ("time_series", "snow | snow falls by", "snow | snow falls by",
              date(2026, 1, 5), 0.3, 0.45, 0.7, 0.5, 9, "TA", "TB")
    _WA_WB = ("time_series", "hail | hail falls by", "hail | hail falls by",
              date(2026, 1, 5), 0.01, 0.99, 0.99, 0.01, 9, "WA", "WB")
    _SAME_TITLE = ("same_title", "Q", ("EVS", "Q", ""),
                   date(2026, 1, 5), 0.6, 0.35, 0.4, 0.65, None, "SB", "SA")
    _GOLDEN = {True: [_EA_EB, _LADDER, _TA_TB, _WA_WB, _SAME_TITLE],
               False: [_EA_EB, _TA_TB, _WA_WB, _SAME_TITLE]}
    # The census total main reported over the same fixture: every market is
    # eligible (each spans a Monday on/after the start date).
    _GOLDEN_CENSUS_TOTAL = 12

    def _patch(self, monkeypatch):
        markets = self._markets()
        monkeypatch.setattr(backtester, "fetch_all_settled_markets",
                            lambda *a, **k: markets)
        monkeypatch.setattr(backtester, "fetch_candlesticks",
                            lambda _c, ticker, *a, **k: self._CANDLES[ticker])

    @staticmethod
    def _rows(entries: list[dict]) -> list[tuple]:
        rows = []
        for rec in entries:
            # The record and entry shapes are part of the contract too.
            assert set(rec) == {"pair_type", "canon", "group_key", "entry"}
            e = rec["entry"]
            assert set(e) == {"entry_date", "pA", "pB", "nA", "nB", "mA", "mB", "gap_days"}
            rows.append((rec["pair_type"], rec["canon"], rec["group_key"],
                         e["entry_date"], e["pA"], e["pB"], e["nA"], e["nB"],
                         e["gap_days"], e["mA"]["ticker"], e["mB"]["ticker"]))
        return rows

    def _prepare(self, monkeypatch, ladders):
        self._patch(monkeypatch)
        return backtester._prepare_entries(
            MagicMock(), MagicMock(), self._START, True, None,
            same_event_ladders=ladders,
        )

    @pytest.mark.parametrize("ladders", [True, False])
    def test_prepare_entries_reproduces_the_main_capture(self, monkeypatch, ladders):
        entries, coverage = self._prepare(monkeypatch, ladders)
        # Exact equality, floats included: every price here is a candle value
        # passed through untouched, never arithmetic.
        assert self._rows(entries) == self._GOLDEN[ladders]
        assert coverage.total == self._GOLDEN_CENSUS_TOTAL

    @pytest.mark.parametrize("ladders", [True, False])
    def test_the_two_halves_compose_to_the_capture(self, monkeypatch, ladders):
        self._patch(monkeypatch)
        candidates = backtester._prepare_candidates(
            MagicMock(), MagicMock(), self._START, True, None,
            same_event_ladders=ladders,
        )
        assert self._rows(backtester._entries_for_band(candidates)) == self._GOLDEN[ladders]
        assert candidates.label_coverage.total == self._GOLDEN_CENSUS_TOTAL

    @pytest.mark.parametrize("band", [None, (0.0, 1.0), BACKTEST_DEFAULT_SPREAD_BAND,
                                      (0, 1), (-0.0, 1.0)])
    def test_every_spelling_of_no_band_reproduces_the_capture(self, monkeypatch, band):
        self._patch(monkeypatch)
        candidates = backtester._prepare_candidates(
            MagicMock(), MagicMock(), self._START, True, None, same_event_ladders=True,
        )
        rows = self._rows(backtester._entries_for_band(candidates, spread_band=band))
        assert rows == self._GOLDEN[True]

    def test_splitting_by_pair_type_concatenates_to_the_default(self, monkeypatch):
        # The band sweep computes the same-title entries once and the
        # time-series entries per band; ts + st must be the default call.
        self._patch(monkeypatch)
        candidates = backtester._prepare_candidates(
            MagicMock(), MagicMock(), self._START, True, None, same_event_ladders=True,
        )
        ts = backtester._entries_for_band(candidates, pair_types=("time_series",))
        st = backtester._entries_for_band(candidates, pair_types=("same_title",))
        assert self._rows(ts) == [self._EA_EB, self._LADDER, self._TA_TB, self._WA_WB]
        assert self._rows(st) == [self._SAME_TITLE]
        assert self._rows(ts + st) == self._rows(backtester._entries_for_band(candidates))

    @pytest.mark.parametrize("ladders", [True, False])
    def test_every_non_default_grid_band_changes_the_rows(self, monkeypatch, ladders):
        # The capture is an oracle for "_prepare_entries applies NO band" only
        # if every band the sweep can apply would move it. TA/TB (0.15) is
        # refused by every non-zero grid floor and WA/WB (0.98) by every grid
        # ceiling below 1.0, so of the 36 grid bands exactly one — the
        # default — reproduces the rows.
        self._patch(monkeypatch)
        candidates = backtester._prepare_candidates(
            MagicMock(), MagicMock(), self._START, True, None,
            same_event_ladders=ladders,
        )
        reproducing = []
        for lo in SPREAD_BAND_SWEEP_FLOORS:
            for hi in SPREAD_BAND_SWEEP_CEILINGS:
                rows = self._rows(
                    backtester._entries_for_band(candidates, spread_band=(lo, hi)))
                if rows == self._GOLDEN[ladders]:
                    reproducing.append((lo, hi))
        assert reproducing == [BACKTEST_DEFAULT_SPREAD_BAND]

    @pytest.mark.parametrize(
        "band,moves",
        [
            # Any floor above the short tier or ceiling below 0.98 moves the
            # rows, on or off the grid ...
            ((0.16, 1.0), True), ((0.0, 0.97), True), ((0.16, 0.99), True),
            # ... and a band that can change no entry on any data does not: a
            # floor at or below the short tier (every pair's tier is >= it)
            # and a ceiling at or above 0.98 (no two live YES asks are wider).
            ((0.15, 1.0), False), ((0.10, 0.98), False), ((0.0, 0.99), False),
        ],
    )
    def test_the_capture_moves_exactly_for_an_effective_band(self, monkeypatch, band, moves):
        self._patch(monkeypatch)
        candidates = backtester._prepare_candidates(
            MagicMock(), MagicMock(), self._START, True, None, same_event_ladders=True,
        )
        rows = self._rows(backtester._entries_for_band(candidates, spread_band=band))
        assert (rows != self._GOLDEN[True]) is moves

    def test_prepare_entries_runs_its_entry_pass_at_no_band(self, monkeypatch):
        # A direct pin on the composition's arguments, beside the capture that
        # pins their effect: one entry pass, spread_band None (the default
        # band) and pair_types left at both types.
        self._patch(monkeypatch)
        real = backtester._entries_for_band
        calls: list = []

        def _spy(*args, **kwargs):
            bound = inspect.signature(real).bind(*args, **kwargs)
            bound.apply_defaults()
            calls.append(dict(bound.arguments))
            return real(*args, **kwargs)

        monkeypatch.setattr(backtester, "_entries_for_band", _spy)
        entries, _ = backtester._prepare_entries(
            MagicMock(), MagicMock(), self._START, True, None, same_event_ladders=True,
        )
        assert len(calls) == 1
        assert calls[0]["spread_band"] is None
        assert calls[0]["pair_types"] == ("time_series", "same_title")
        assert self._rows(entries) == self._GOLDEN[True]


class TestPrepareCandidates:
    """_prepare_candidates is everything _prepare_entries did through the
    candlestick fetch, and carries the inputs the entry pass must share with
    pair extraction (the DR-73c agreement rule)."""

    def test_the_infeasible_window_returns_none_before_any_fetch(self, monkeypatch):
        monkeypatch.setattr(backtester, "fetch_all_settled_markets",
                            lambda *a, **k: pytest.fail("fetch must be skipped"))
        assert backtester._prepare_candidates(
            MagicMock(), MagicMock(), date(2099, 1, 1), True, None,
        ) is None
        # ... which _prepare_entries still reports as its (None, None) pair
        assert backtester._prepare_entries(
            MagicMock(), MagicMock(), date(2099, 1, 1), True, None,
        ) == (None, None)

    @pytest.mark.parametrize("ladders", [None, True, False])
    def test_it_carries_start_horizon_and_the_unresolved_ladder_flag(
        self, monkeypatch, ladders,
    ):
        TestPrepareEntriesGolden()._patch(monkeypatch)
        c = backtester._prepare_candidates(
            MagicMock(), MagicMock(), date(2026, 1, 1), True, 45,
            same_event_ladders=ladders,
        )
        assert c.start_date == date(2026, 1, 1)
        assert c.max_horizon_days == 45
        # UNRESOLVED: None stays None, so every entry pass hands _find_entry
        # the same ARGUMENT _extract_pairs was handed (each resolves it at its
        # own call time — see _Candidates for the one caveat that implies).
        assert c.same_event_ladders is ladders

    def test_pairs_are_in_scan_order_and_every_ticker_has_candles(self, monkeypatch):
        TestPrepareEntriesGolden()._patch(monkeypatch)
        c = backtester._prepare_candidates(
            MagicMock(), MagicMock(), date(2026, 1, 1), True, None,
            same_event_ladders=True,
        )
        types = [pt for _, pt in c.all_pairs]
        # EA/EB, the ladder, FA/FB, TA/TB and WA/WB, then the one same-title
        # pair
        assert types == ["time_series"] * 5 + ["same_title"]
        for (mA, mB, _canon, _key), _pt in c.all_pairs:
            assert mA["ticker"] in c.candles_by_ticker
            assert mB["ticker"] in c.candles_by_ticker


# ─── SS-1: key every eligible record, materialize only the groupable subset ──

_SS1_START = date(2026, 1, 1)


def _ss1_record(ticker, event_ticker, title, *, subtitle="", event_title="",
                close="2026-02-20", eligible=True):
    """A cached-shape market record; `eligible=False` spans no Monday."""
    if eligible:
        open_t, close_t = "2026-01-01T00:00:00+00:00", f"{close}T00:00:00+00:00"
    else:
        # Opens and closes inside one Tuesday: no Monday 09:00 UTC checkpoint
        # fits, so _can_ever_enter proves it can never enter any pair.
        open_t, close_t = "2026-01-13T00:00:00+00:00", "2026-01-13T02:00:00+00:00"
    return {"ticker": ticker, "event_ticker": event_ticker,
            "event_title": event_title, "title": title, "subtitle": subtitle,
            "result": "yes", "open_time": open_t, "close_time": close_t,
            "settlement_ts": close_t}


# Fill titles that can never form a TIME-SERIES pair (no stated deadline, a
# snapshot wording, a title that normalizes away, or one deadline shared by
# every record that carries it). Their fill records all close at one instant
# (DR-74), so their same-title candidates survive the close gate and keep
# SS-1's exactness tests covering the same-title branch with real pairs; the
# Rain/Snow fill keeps its random closes, so those tests still cover the
# gate's refusals too, and every time-series pair is unchanged.
_SS1_SAME_TITLE_ONLY_TITLES = frozenset({
    "Q", "March 1, 2026", "Bitcoin price on Sep 15, 2026?",
    "Will X happen by March 5, 2026?",
})
_SS1_SHARED_CLOSE = "2026-02-20"


def _ss1_corpus(seed: int) -> list[dict]:
    """Hand-placed anchors for every case the subset must get right, plus a
    seeded random fill whose small title/subtitle/event pools make keys
    collide (shared) about as often as they stay unique (singletons).

    Since DR-74 a same-title candidate forms only when its two markets close
    within SAME_TITLE_MAX_CLOSE_GAP_SECONDS, and a 50-day random close spread
    would refuse nearly every one (measured: 4/5/4 same-title pairs for
    seeds 0/1/2, against 126/117/93 before the gate). So a fill record whose
    title can only ever pair as same-title closes at _SS1_SHARED_CLOSE — the
    anchors' default close, so those records share it with SA/SB and DA/DB —
    giving 78/65/50. The random close is still DRAWN for every record, so the
    rng stream, every other field and every time-series pair are unchanged."""
    anchors = [
        # A cross-event time-series pair: shared time-series key, distinct
        # same-title keys ...
        _ss1_record("RA", "RAINA-1", "Rain falls by March 1, 2026",
                    event_title="RAIN", close="2026-03-01"),
        _ss1_record("RB", "RAINB-1", "Rain falls by March 20, 2026",
                    event_title="RAIN", close="2026-03-20"),
        # ... and an INELIGIBLE member of the same family, which both passes
        # must skip without shifting any position.
        _ss1_record("RX", "RAINX-1", "Rain falls by March 9, 2026",
                    event_title="RAIN", eligible=False),
        # A same-title pair on two series (shares BOTH keys).
        _ss1_record("SA", "SERA-1", "Q", event_title="EVS"),
        _ss1_record("SB", "SERB-1", "Q", event_title="EVS"),
        # A same-event deadline ladder, later rung listed first (DR-73).
        _ss1_record("L2", "KXSTAR-14",
                    "Will SpaceX launch another Starship by March 20, 2026?",
                    close="2026-03-20"),
        _ss1_record("L1", "KXSTAR-14",
                    "Will SpaceX launch another Starship by March 1, 2026?",
                    close="2026-03-20"),
        # Kept ONLY through the same-title key: the title normalizes away, so
        # the time-series key is empty.
        _ss1_record("DA", "DATEA-1", "March 1, 2026"),
        _ss1_record("DB", "DATEB-1", "March 1, 2026"),
        # Singletons of each kind, plus a record with no wording at all.
        _ss1_record("DZ", "DATEZ-1", "April 2, 2026"),
        _ss1_record("NW", "NOWORD-1", ""),
        _ss1_record("LONE", "LONE-1", "A question nobody else asks by May 1, 2026"),
        # A strike family: the subtitle is the outcome discriminator (DR-01).
        _ss1_record("K1", "KXBTC-1", "Bitcoin price by March 1, 2026?",
                    subtitle="$80,000 or above", close="2026-03-01"),
        _ss1_record("K2", "KXBTC-2", "Bitcoin price by March 9, 2026?",
                    subtitle="$80,000 or above", close="2026-03-09"),
        _ss1_record("K3", "KXBTC-2", "Bitcoin price by March 9, 2026?",
                    subtitle="$90,000 or above", close="2026-03-09"),
        # A pre-fix cache record: subtitle null, event title never resolved.
        {**_ss1_record("U1", "UNL-1", "Unlabelled question"),
         "subtitle": None, "event_title": None},
    ]
    rng = random.Random(seed)
    titles = ["Rain falls by March 1, 2026", "Rain falls by March 20, 2026", "Q",
              "Snow falls by February 1, 2026", "Snow falls by February 10, 2026",
              "Bitcoin price on Sep 15, 2026?", "Will X happen by March 5, 2026?",
              "March 1, 2026"]
    fill = []
    for i in range(300):
        title = (f"Unique question {seed}-{i}" if rng.random() < 0.4
                 else rng.choice(titles))
        close = (date(2026, 2, 1) + timedelta(days=rng.randint(0, 50))).isoformat()
        if title in _SS1_SAME_TITLE_ONLY_TITLES:
            close = _SS1_SHARED_CLOSE
        rec = _ss1_record(
            f"R{seed}-{i}",
            f"{rng.choice('ABCDE')}SER-{rng.randint(1, 4)}",
            title,
            subtitle=rng.choice(["", "Yes", "$80,000 or above"]),
            event_title=rng.choice(["", "RAIN", "EVS", "SNOW"]),
            close=close,
            eligible=rng.random() > 0.1,
        )
        if rng.random() < 0.1:
            rec["subtitle"] = None
        fill.append(rec)
    return anchors + fill


def _old_group_by_exact_title(markets):
    """The pre-SS-1 _group_by_exact_title, verbatim — an oracle that does not
    route through the _st_group_key it is checking."""
    groups = defaultdict(list)
    for m in markets:
        event_title = m.get("event_title") or ""
        title = m.get("title") or ""
        subtitle = m.get("subtitle") or ""
        if title or subtitle:
            groups[(event_title, title, subtitle)].append(m)
    return {k: v for k, v in groups.items() if len(v) >= 2}


def _old_group_by_normalized_title(markets):
    """The pre-SS-1 _group_by_normalized_title, verbatim (see above)."""
    groups = defaultdict(list)
    for m in markets:
        norm = scanner.time_series_group_key(
            backtester._pair_key(m), m.get("subtitle") or "")
        if norm:
            groups[norm].append(m)
    return {k: v for k, v in groups.items() if len(v) >= 2}


def _group_shape(groups):
    """Keys, insertion order, and member IDENTITY and order, in one value."""
    return [(k, [id(m) for m in v]) for k, v in groups.items()]


def _pair_shape(all_pairs, by=id):
    """A candidate-pair list reduced to comparable fields; `by` picks member
    identity (id) or, for fresh-dict corpora, the ticker."""
    return [(by(mA), by(mB), canon, key, pair_type)
            for (mA, mB, canon, key), pair_type in all_pairs]


def _by_ticker(m):
    return m["ticker"]


class _FreshCorpus:
    """A re-iterable corpus that is NOT a list and has no len(): every walk
    yields FRESH dict objects built from the template (as a corpus streamed off
    disk would), and it counts its walks."""

    def __init__(self, template, factory=dict):
        self._template = template
        self._factory = factory
        self.walks = 0
        self.yielded: list[list] = []

    def __iter__(self):
        self.walks += 1
        this_walk: list = []
        self.yielded.append(this_walk)
        for m in self._template:
            fresh = self._factory(m)
            if self._factory is _WeakrefDict:
                this_walk.append(weakref.ref(fresh))
            yield fresh


class _DriftingCorpus:
    """Yields `first` on its first walk and `second` on every later one."""

    def __init__(self, first, second):
        self._walks = [first, second]
        self.walks = 0

    def __iter__(self):
        walk = self._walks[min(self.walks, 1)]
        self.walks += 1
        return iter(walk)


class TestGroupableSubset:
    """SS-1: a 7-day window measured 7,260,952 eligible records of which only
    184,255 share either grouping key with another eligible record. Both
    grouping functions drop every single-member group, so _prepare_candidates
    keys every eligible record in a first pass (hashes only) and materializes
    only the records whose key is shared in a second. That must be EXACT: the
    same groups, pairs, census and log numbers as grouping the whole eligible
    list, which the old code did.
    """

    @staticmethod
    def _eligible(markets):
        return [m for m in markets if _can_ever_enter(m, _SS1_START)]

    @staticmethod
    def _subset(markets):
        index = backtester._index_eligible_keys(
            markets, _SS1_START, backtester._OutcomeLabelTally())
        return backtester._materialize_groupable(markets, _SS1_START, index)

    @staticmethod
    def _patch(monkeypatch, corpus):
        monkeypatch.setattr(backtester, "fetch_all_settled_markets",
                            lambda *a, **k: corpus)
        monkeypatch.setattr(backtester, "fetch_candlesticks",
                            lambda *a, **k: [])

    @staticmethod
    def _prepare(ladders=True):
        return backtester._prepare_candidates(
            MagicMock(), MagicMock(), _SS1_START, True, None,
            same_event_ladders=ladders)

    @staticmethod
    def _reference_pairs(eligible, ladders):
        ts_pairs = _extract_pairs(_old_group_by_normalized_title(eligible),
                                  same_event_ladders=ladders)
        same_pairs = _extract_pairs(_old_group_by_exact_title(eligible),
                                    same_event_ladders=ladders)
        return ([(p, "time_series") for p in ts_pairs]
                + [(p, "same_title") for p in same_pairs])

    # ── (1) exactness ────────────────────────────────────────────────────

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_groups_over_the_subset_equal_groups_over_the_eligible_list(self, seed):
        markets = _ss1_corpus(seed)
        eligible = self._eligible(markets)
        subset = self._subset(markets)

        old_ts = _old_group_by_normalized_title(eligible)
        old_st = _old_group_by_exact_title(eligible)
        # Keys, insertion order, and member identity and order — all equal.
        assert _group_shape(_group_by_normalized_title(subset)) == _group_shape(old_ts)
        assert _group_shape(_group_by_exact_title(subset)) == _group_shape(old_st)

        # And the subset is EXACTLY the records some group of two or more
        # holds, in corpus order (no hash collision is plausible at this size;
        # the collision case has its own test below).
        grouped = {id(m) for g in (old_ts, old_st) for v in g.values() for m in v}
        assert [id(m) for m in subset] == [id(m) for m in eligible if id(m) in grouped]

        # Not vacuous: the fixture really exercises every case.
        assert len(eligible) < len(markets)          # prefiltered records
        assert len(subset) < len(eligible)           # singletons dropped
        by_ticker = {m["ticker"]: m for m in markets}
        assert backtester._ts_group_key(by_ticker["DA"]) == ""
        assert by_ticker["DA"] in subset             # kept via same-title only
        for lone in ("DZ", "NW", "LONE", "RX"):
            assert all(m is not by_ticker[lone] for m in subset)

    @pytest.mark.parametrize("ladders", [True, False])
    @pytest.mark.parametrize("seed", [0, 1])
    def test_prepare_candidates_equals_grouping_the_whole_eligible_list(
        self, monkeypatch, seed, ladders,
    ):
        markets = _ss1_corpus(seed)
        eligible = self._eligible(markets)
        self._patch(monkeypatch, markets)
        c = self._prepare(ladders)

        expected = self._reference_pairs(eligible, ladders)
        assert _pair_shape(c.all_pairs) == _pair_shape(expected)
        assert c.label_coverage == backtester._log_outcome_label_coverage(eligible)
        # Not vacuous: both pair types present, and the ladder with ladders on.
        types = {pt for _, pt in c.all_pairs}
        assert types == {"time_series", "same_title"}
        # ... and the same-title branch with real pairs, not the anchors' token
        # few: the fill's same-title-only titles share one close (DR-74), so
        # their candidates survive the close gate (78/65 for seeds 0/1).
        assert sum(pt == "same_title" for _, pt in c.all_pairs) >= 40
        ladder = {"L1", "L2"}
        assert any({mA["ticker"], mB["ticker"]} == ladder
                   for (mA, mB, _c, _k), _pt in c.all_pairs) is ladders

    def test_log_numbers_and_order_are_unchanged_plus_one_line(
        self, monkeypatch, caplog,
    ):
        markets = _ss1_corpus(0)
        eligible = self._eligible(markets)
        subset = self._subset(markets)
        self._patch(monkeypatch, markets)
        monkeypatch.setattr(backtester, "BACKTEST_MARKETS_RAM_WARN", 1)
        with caplog.at_level("INFO"):
            self._prepare()
        messages = [r.getMessage() for r in caplog.records]

        def at(prefix):
            hits = [i for i, m in enumerate(messages) if m.startswith(prefix)]
            assert len(hits) == 1, prefix
            return hits[0]

        total = f"Markets to analyze: {len(markets)}"
        prefilter = (f"Eligibility prefilter: skipping "
                     f"{len(markets) - len(eligible)}/{len(markets)} markets")
        groupable = (f"Groupable subset: materializing {len(subset)} of "
                     f"{len(eligible)} eligible markets")
        ram = f"{len(subset)} groupable markets (of {len(eligible)} eligible)"
        order = [at(total), at(prefilter), at(groupable),
                 at("Peak RSS before grouping"), at(ram),
                 at(f"Outcome-label coverage over {len(eligible)} eligible markets"),
                 at(f"Deadline phrasing over {len(eligible)} eligible markets"),
                 at("Peak RSS after pair extraction"), at("Potential pairs:")]
        assert order == sorted(order)

    def test_a_hash_collision_can_only_keep_extra_records(self, monkeypatch, caplog):
        """Every key hashes to ONE value here, so every eligible record with a
        key of either kind reads as "shared" and is materialized. The exact
        grouping then drops the true singletons, so the pairs and census are
        unchanged — the whole safety argument for keying on hash()."""
        markets = _ss1_corpus(1)
        eligible = self._eligible(markets)
        expected = self._reference_pairs(eligible, True)
        self._patch(monkeypatch, markets)
        monkeypatch.setattr(backtester, "hash", lambda _value: 7, raising=False)
        with caplog.at_level("INFO"):
            c = self._prepare()
        assert _pair_shape(c.all_pairs) == _pair_shape(expected)
        assert c.label_coverage == backtester._log_outcome_label_coverage(eligible)
        # Every eligible record has a key of some kind here, so all are kept.
        assert all(backtester._ts_group_key(m) or backtester._st_group_key(m)
                   for m in eligible)
        assert (f"Groupable subset: materializing {len(eligible)} of "
                f"{len(eligible)} eligible markets") in caplog.text

    def test_the_shared_mask_never_counts_an_invalid_placeholder(self):
        # Position 1 has NO key; its placeholder hash happens to equal the
        # valid key at position 0. That must not make position 0 "shared".
        hashes = array("q", [0, 0, 5, 5, 9])
        valid = bytearray([1, 0, 1, 1, 1])
        assert backtester._shared_key_mask(hashes, valid).tolist() == [
            False, False, True, True, False]
        assert backtester._shared_key_mask(array("q"), bytearray()).tolist() == []

    # ── (2) the census over a one-shot iterator ──────────────────────────

    @pytest.mark.parametrize("corpus", [
        _ss1_corpus(0),
        TestOutcomeLabelCoverageCensus._unlabelled(10),
        [],
    ])
    def test_the_census_over_a_one_shot_iterator_equals_the_census_over_the_list(
        self, caplog, corpus,
    ):
        with caplog.at_level("INFO"):
            from_list = backtester._log_outcome_label_coverage(corpus)
        list_lines = [(r.levelname, r.getMessage()) for r in caplog.records]
        caplog.clear()
        with caplog.at_level("INFO"):
            from_iter = backtester._log_outcome_label_coverage(m for m in corpus)
        assert from_iter == from_list
        assert [(r.levelname, r.getMessage()) for r in caplog.records] == list_lines

    # ── (3) a non-list re-iterable of fresh dicts ────────────────────────

    @pytest.mark.parametrize("ladders", [True, False])
    def test_a_fresh_dict_reiterable_reproduces_the_golden_capture(
        self, monkeypatch, ladders,
    ):
        golden = TestPrepareEntriesGolden()
        corpus = _FreshCorpus(golden._markets())
        monkeypatch.setattr(backtester, "fetch_all_settled_markets",
                            lambda *a, **k: corpus)
        monkeypatch.setattr(backtester, "fetch_candlesticks",
                            lambda _c, ticker, *a, **k: golden._CANDLES[ticker])
        c = self._prepare(ladders)
        rows = TestPrepareEntriesGolden._rows(backtester._entries_for_band(c))
        assert rows == golden._GOLDEN[ladders]
        assert c.label_coverage.total == golden._GOLDEN_CENSUS_TOTAL
        # Exactly two walks: the census rides the first, never a third.
        assert corpus.walks == 2

    def test_a_fresh_dict_reiterable_matches_the_list_on_the_synthetic_corpus(
        self, monkeypatch,
    ):
        template = _ss1_corpus(2)
        self._patch(monkeypatch, template)
        from_list = self._prepare()
        corpus = _FreshCorpus(template)
        self._patch(monkeypatch, corpus)
        from_fresh = self._prepare()
        assert (_pair_shape(from_fresh.all_pairs, by=_by_ticker)
                == _pair_shape(from_list.all_pairs, by=_by_ticker))
        # Value-equal records, not merely equal tickers.
        assert [p for p, _ in from_fresh.all_pairs] == [p for p, _ in from_list.all_pairs]
        assert from_fresh.label_coverage == from_list.label_coverage
        assert corpus.walks == 2

    # ── (4) a corpus that does not re-iterate identically ────────────────

    @pytest.mark.parametrize("drift", ["fewer", "more", "reordered",
                                       "title", "subtitle", "event_title"])
    def test_a_second_walk_that_disagrees_raises(self, monkeypatch, drift):
        first = _ss1_corpus(0)
        if drift == "fewer":
            second = [m for m in first if m["ticker"] != "SB"]
        elif drift == "more":
            second = first + [_ss1_record("EXTRA", "EXTRA-1", "Q", event_title="EVS")]
        elif drift == "reordered":
            # Same eligible count, two eligible records swapped.
            second = list(first)
            i = next(k for k, m in enumerate(second) if m["ticker"] == "SA")
            j = next(k for k, m in enumerate(second) if m["ticker"] == "RB")
            second[i], second[j] = second[j], second[i]
        else:
            # Same tickers in the same order, but one eligible record's
            # grouping field changed between the walks: the keep flags were
            # chosen on keys this record no longer has. One row per field
            # either key reads, so dropping any one of them from the identity
            # is caught.
            second = [dict(m) for m in first]
            i = next(k for k, m in enumerate(second) if m["ticker"] == "LONE")
            second[i][drift] = "changed between walks"
        self._patch(monkeypatch, _DriftingCorpus(first, second))
        with pytest.raises(RuntimeError, match="did not iterate identically twice"):
            self._prepare()

    def test_a_key_field_drift_that_would_silently_drop_a_record_raises(self):
        """B-ADV-1's shape: walk 2 repeats walk 1's tickers in order, but C's
        subtitle now matches A and B's, so C belongs in their group. The keep
        flags were chosen on walk 1, where C's key was unique, so applying
        them to walk 2 would drop C from a group of the corpus actually being
        grouped — with nothing raised, if only tickers were compared."""
        def rec(ticker, subtitle):
            return _ss1_record(ticker, f"{ticker}SER-1", "Will X happen by March 5, 2026?",
                               subtitle=subtitle, close="2026-03-05")

        first = [rec("A", "s1"), rec("B", "s1"), rec("C", "s2")]
        second = [rec("A", "s1"), rec("B", "s1"), rec("C", "s1")]
        # The premise: grouped as a list, walk 2 genuinely holds all three.
        assert [m["ticker"] for g in _old_group_by_normalized_title(second).values()
                for m in g] == ["A", "B", "C"]
        corpus = _DriftingCorpus(first, second)
        index = backtester._index_eligible_keys(
            corpus, _SS1_START, backtester._OutcomeLabelTally())
        assert index.keep == bytes([1, 1, 0])
        with pytest.raises(RuntimeError, match="eligible record 2 is 'C'"):
            backtester._materialize_groupable(corpus, _SS1_START, index)

    def test_the_count_mismatch_names_both_counts(self):
        first = [_ss1_record("A", "EA-1", "Q"), _ss1_record("B", "EB-1", "Q")]
        corpus = _DriftingCorpus(first, first[:1])
        index = backtester._index_eligible_keys(
            corpus, _SS1_START, backtester._OutcomeLabelTally())
        with pytest.raises(RuntimeError, match="found 2 eligible markets and the second 1"):
            backtester._materialize_groupable(corpus, _SS1_START, index)

    def test_drift_among_ineligible_records_is_not_a_misalignment(self, monkeypatch):
        # Positions are counted among ELIGIBLE records only, so a second walk
        # that differs solely in records the prefilter drops is the same
        # corpus as far as the subset is concerned — and must not raise.
        first = _ss1_corpus(0)
        second = [m for m in first if m["ticker"] != "RX"]      # RX is ineligible
        assert not _can_ever_enter(next(m for m in first if m["ticker"] == "RX"),
                                   _SS1_START)
        self._patch(monkeypatch, _DriftingCorpus(first, second))
        c = self._prepare()
        expected = self._reference_pairs(self._eligible(first), True)
        assert _pair_shape(c.all_pairs) == _pair_shape(expected)

    # ── (5) singletons are never materialized ─────────────────────────────

    def test_singletons_are_never_materialized(self, monkeypatch):
        """Instrumented: every walk yields fresh weak-referenceable dicts, so
        at the moment grouping starts the ONLY live records are the ones the
        backtester chose to hold. Those must be exactly the groupable ones —
        not one singleton, and nothing left over from the first walk."""
        template = _ss1_corpus(0)
        eligible = self._eligible(template)
        old_ts = _old_group_by_normalized_title(eligible)
        old_st = _old_group_by_exact_title(eligible)
        grouped = {m["ticker"] for g in (old_ts, old_st) for v in g.values() for m in v}
        expected = [m["ticker"] for m in eligible if m["ticker"] in grouped]
        assert 0 < len(expected) < len(eligible)

        corpus = _FreshCorpus(template, factory=_WeakrefDict)
        self._patch(monkeypatch, corpus)
        real_group = backtester._group_by_normalized_title
        seen: dict = {}

        def _spy(markets):
            gc.collect()
            seen["input"] = [m["ticker"] for m in markets]
            seen["walk1_alive"] = sum(r() is not None for r in corpus.yielded[0])
            seen["walk2_alive"] = sorted(
                r()["ticker"] for r in corpus.yielded[1] if r() is not None)
            return real_group(markets)

        monkeypatch.setattr(backtester, "_group_by_normalized_title", _spy)
        self._prepare()
        assert seen["input"] == expected
        assert seen["walk1_alive"] == 0
        assert seen["walk2_alive"] == sorted(expected)

    def test_the_ram_warning_counts_the_groupable_records(self, monkeypatch, caplog):
        # Three eligible markets, two of which share a key: the warning is
        # keyed on the 2 it says are resident, not the 3 eligible ones.
        markets = TestPrepareEntriesMemoryInstrumentation._markets() + [
            _ss1_record("EC", "EVC", "Unrelated question by March 1, 2026",
                        close="2026-03-01")]
        self._patch(monkeypatch, markets)
        monkeypatch.setattr(backtester, "BACKTEST_MARKETS_RAM_WARN", 2)
        with caplog.at_level("WARNING"):
            self._prepare()
        assert "groupable markets" not in caplog.text
        caplog.clear()
        monkeypatch.setattr(backtester, "BACKTEST_MARKETS_RAM_WARN", 1)
        with caplog.at_level("WARNING"):
            self._prepare()
        assert "2 groupable markets (of 3 eligible) are materialized" in caplog.text


class TestPrepareCandidatesOverASettledCorpus:
    """SS-1 Commit C: fetch_all_settled_markets now hands _prepare_candidates a
    disk-backed historical.SettledCorpus — the assembled .jsonl.gz cache,
    streamed afresh on every walk — instead of one list. _prepare_candidates'
    two passes must produce exactly what they produce over the list, and walk
    the file exactly twice (the census rides the first pass)."""

    @staticmethod
    def _corpus(tmp_path, records):
        path = tmp_path / "settled_markets_test.jsonl.gz"
        meta = historical._assembled_cache_meta(_SS1_START, "t")
        with historical._DayStreamWriter(path, meta) as writer:
            for m in records:
                writer.write_record(m)
            count = writer.commit()
        return historical.SettledCorpus(path, meta, count)

    @pytest.mark.parametrize("ladders", [True, False])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_the_streamed_corpus_prepares_exactly_what_the_list_does(
        self, tmp_path, monkeypatch, seed, ladders,
    ):
        template = _ss1_corpus(seed)
        TestGroupableSubset._patch(monkeypatch, template)
        from_list = TestGroupableSubset._prepare(ladders)

        corpus = self._corpus(tmp_path, template)
        assert len(corpus) == len(template)
        walks: list[int] = []
        real_iter = historical._day_store_iter

        def counting(*args, **kwargs):
            walks.append(1)
            return real_iter(*args, **kwargs)

        monkeypatch.setattr(historical, "_day_store_iter", counting)
        TestGroupableSubset._patch(monkeypatch, corpus)
        from_corpus = TestGroupableSubset._prepare(ladders)

        assert (_pair_shape(from_corpus.all_pairs, by=_by_ticker)
                == _pair_shape(from_list.all_pairs, by=_by_ticker))
        # Value-equal records (a JSON round trip changes nothing here), not
        # merely equal tickers.
        assert [p for p, _ in from_corpus.all_pairs] == [p for p, _ in from_list.all_pairs]
        assert from_corpus.label_coverage == from_list.label_coverage
        assert len(walks) == 2

    def test_the_corpus_provenance_rides_out_on_the_candidates(
        self, tmp_path, monkeypatch,
    ):
        # Taken off the corpus BY TYPE before it is released: a SettledCorpus
        # hands over its provenance, and so does the LegacySettledCorpus list
        # a legacy-cache hit returns (its file time, no cutoff); a plain list
        # (a test stub) has none; and a MagicMock — which would answer
        # .provenance with a truthy auto-attribute — is not mistaken for one.
        template = _ss1_corpus(0)
        corpus = self._corpus(tmp_path, template)
        prov = historical.CorpusProvenance(
            from_cache=True, assembled_at=datetime(2026, 1, 5, 9, tzinfo=UTC),
            archive_cutoff=datetime(2025, 12, 1, tzinfo=UTC), post_cutoff=True)
        with_prov = historical.SettledCorpus(corpus.path, historical._assembled_cache_meta(
            _SS1_START, "t"), len(corpus), provenance=prov)
        TestGroupableSubset._patch(monkeypatch, with_prov)
        assert TestGroupableSubset._prepare().corpus_provenance is prov
        TestGroupableSubset._patch(monkeypatch, template)
        assert TestGroupableSubset._prepare().corpus_provenance is None
        legacy_prov = historical.CorpusProvenance(
            from_cache=True, assembled_at=datetime(2026, 8, 3, 19, 5, tzinfo=UTC),
            archive_cutoff=None, post_cutoff=None, legacy=True)
        legacy = historical.LegacySettledCorpus(template, legacy_prov)
        TestGroupableSubset._patch(monkeypatch, legacy)
        prepared = TestGroupableSubset._prepare()
        assert prepared.corpus_provenance is legacy_prov
        # ...and the legacy list prepares exactly what the plain list does.
        TestGroupableSubset._patch(monkeypatch, template)
        plain = TestGroupableSubset._prepare()
        assert prepared.all_pairs == plain.all_pairs
        assert prepared.label_coverage == plain.label_coverage
        stub = MagicMock()
        stub.__iter__.return_value = iter([])
        stub.provenance = prov
        TestGroupableSubset._patch(monkeypatch, stub)
        assert backtester._prepare_candidates(
            MagicMock(), MagicMock(), _SS1_START, True, None).corpus_provenance is None

    def test_a_cache_replaced_between_the_two_passes_is_refused(
        self, tmp_path, monkeypatch,
    ):
        # Pass 2 re-opens the file; if another run replaced it in between
        # with a different corpus, the run must stop, not group records the
        # first pass never keyed.
        template = _ss1_corpus(0)
        corpus = self._corpus(tmp_path, template)
        real_index = backtester._index_eligible_keys

        def index_then_replace(markets, start_date, census):
            index = real_index(markets, start_date, census)
            self._corpus(tmp_path, [m for m in template if m["ticker"] != "SB"])
            return index

        monkeypatch.setattr(backtester, "_index_eligible_keys", index_then_replace)
        TestGroupableSubset._patch(monkeypatch, corpus)
        # The first eligible position after the removed record no longer
        # matches, so pass 2's own identity check stops the run there.
        with pytest.raises(RuntimeError, match="did not iterate identically"):
            TestGroupableSubset._prepare()


class TestPrefilterLinesSayItRanDuringAssembly:
    """M9 of the 2026-09-24 7-day-run review: _prepare_candidates re-applies
    the eligibility prefilter to a corpus the fetch already prefiltered, so
    "Eligibility prefilter: skipping 0/7274215" was printed beside "Total
    settled markets to analyze: 7274215" although the assembly had kept
    those 7,274,215 of about 24.6M settled records (the review's estimate).
    A corpus that carries provenance (it came from the fetch) is now
    reported as ELIGIBLE markets, with the assembly's own rejections when it
    recorded them, and its re-check is named as one; a re-check that rejects
    anything is a WARNING, since only a predicate changed without a tag bump
    (or an altered cache) can do that."""

    TAG = backtester.SETTLED_PREFILTER_CACHE_TAG
    COUNTS = historical.AssemblyCounts(settled=40, rejected=25, duplicates=3)

    @staticmethod
    def _prov(**fields):
        base = {"from_cache": False, "assembled_at": datetime(2026, 9, 24, 12, 37, tzinfo=UTC),
                "archive_cutoff": datetime(2026, 7, 25, tzinfo=UTC), "post_cutoff": True}
        base.update(fields)
        return historical.CorpusProvenance(**base)

    @staticmethod
    def _lines(caplog):
        return [(r.levelname, r.getMessage()) for r in caplog.records]

    def test_a_fresh_prefiltered_corpus_quotes_what_the_assembly_rejected(self, caplog):
        with caplog.at_level(logging.INFO):
            backtester._log_corpus_prefilter(12, 12, self._prov(assembly_counts=self.COUNTS))
        assert self._lines(caplog) == [
            ("INFO", f"Markets to analyze: 12 eligible — the eligibility prefilter "
                     f"({self.TAG}) ran during assembly and rejected 25 of the 40 "
                     f"records settled in the window (3 more were duplicate or blank "
                     f"tickers; counted by this run)"),
            ("INFO", "Eligibility prefilter re-check: 0 of 12 markets rejected — "
                     "none expected, since it already ran during assembly"),
        ]
        assert "skipping 0/" not in caplog.text

    def test_a_cache_hit_says_the_counts_are_as_of_its_assembly(self, caplog):
        with caplog.at_level(logging.INFO):
            backtester._log_corpus_prefilter(
                12, 12, self._prov(from_cache=True, assembly_counts=self.COUNTS))
        assert "(3 more were duplicate or blank tickers; counted at this cache's " \
            "assembly)" in caplog.text

    @pytest.mark.parametrize("legacy, noun", [(False, "cache"), (True, "legacy cache")])
    def test_a_corpus_without_counts_says_it_records_none(self, caplog, legacy, noun):
        with caplog.at_level(logging.INFO):
            backtester._log_corpus_prefilter(
                7, 7, self._prov(from_cache=True, legacy=legacy,
                                 archive_cutoff=None, post_cutoff=None))
        assert self._lines(caplog)[0] == (
            "INFO", f"Markets to analyze: 7 eligible — the eligibility prefilter "
                    f"({self.TAG}) ran during assembly, but this {noun} records no "
                    f"count of the records it rejected")
        assert not [r for r in caplog.records if r.levelname == "WARNING"]

    @pytest.mark.parametrize("counts", [None, COUNTS])
    def test_a_re_check_that_rejects_anything_is_a_warning(self, caplog, counts):
        with caplog.at_level(logging.INFO):
            backtester._log_corpus_prefilter(12, 9, self._prov(assembly_counts=counts))
        lines = self._lines(caplog)
        warned = [m for level, m in lines if level == "WARNING"]
        assert len(warned) == 1
        assert warned[0].startswith(
            f"Eligibility prefilter re-check: 3 of 12 markets rejected, although "
            f"the prefilter ({self.TAG}) already ran during this corpus's assembly")
        assert "config.SETTLED_PREFILTER_CACHE_TAG" in warned[0]
        # The INFO line above that WARNING must not call all 12 eligible when
        # the WARNING says 3 of them are not: it names both counts.
        assert lines[0][0] == "INFO"
        assert lines[0][1].startswith(
            f"Markets to analyze: 12 assembled as eligible, 9 still eligible "
            f"after the re-check below — the eligibility prefilter ({self.TAG}) "
            f"ran during assembly")
        assert "12 eligible" not in lines[0][1]

    def test_a_plain_list_keeps_the_old_skip_line(self, caplog):
        # No provenance, so no assembly is known to have filtered it: the
        # re-check IS its prefilter, and says so in the words it always used.
        with caplog.at_level(logging.INFO):
            backtester._log_corpus_prefilter(10, 6, None)
        assert self._lines(caplog) == [
            ("INFO", "Markets to analyze: 10 (no assembly record — whether a "
                     "prefilter ran while this corpus was assembled is unknown)"),
            ("INFO", "Eligibility prefilter: skipping 4/10 markets that cannot "
                     "appear in any tradeable pair"),
        ]

    def test_prepare_candidates_reports_a_fetched_corpus_through_it(
        self, tmp_path, monkeypatch, caplog,
    ):
        # End to end over a real streamed corpus that WAS prefiltered, as the
        # fetch returns it: the corpus is its eligible markets, the assembly's
        # counts are quoted, the re-check rejects nothing, nothing warns.
        eligible = TestGroupableSubset._eligible(_ss1_corpus(0))
        written = TestPrepareCandidatesOverASettledCorpus._corpus(tmp_path, eligible)
        counts = historical.AssemblyCounts(
            settled=len(eligible) + 30, rejected=30, duplicates=0)
        corpus = historical.SettledCorpus(
            written.path, historical._assembled_cache_meta(_SS1_START, "t"),
            len(written), provenance=self._prov(assembly_counts=counts))
        TestGroupableSubset._patch(monkeypatch, corpus)
        with caplog.at_level(logging.INFO):
            TestGroupableSubset._prepare()
        messages = [r.getMessage() for r in caplog.records]
        assert (f"Markets to analyze: {len(eligible)} eligible — the eligibility "
                f"prefilter ({self.TAG}) ran during assembly and rejected 30 of "
                f"the {len(eligible) + 30} records settled in the window") in caplog.text
        assert (f"Eligibility prefilter re-check: 0 of {len(eligible)} markets "
                f"rejected") in caplog.text
        # (the stubbed candle fetch returns nothing, which warns on its own)
        assert not [r for r in caplog.records if r.levelname == "WARNING"
                    and "prefilter" in r.getMessage()]
        # Still before the groupable line, as the old pair of lines was.
        first = next(i for i, m in enumerate(messages) if m.startswith("Markets to analyze"))
        group = next(i for i, m in enumerate(messages) if m.startswith("Groupable subset"))
        assert first < group

    def test_prepare_candidates_warns_on_a_fetched_corpus_it_can_still_filter(
        self, monkeypatch, caplog,
    ):
        # A corpus claiming to come from the fetch that still holds records
        # the predicate rejects: the stale-tag case. Dropped AND warned about.
        template = _ss1_corpus(0)
        legacy = historical.LegacySettledCorpus(
            template, self._prov(from_cache=True, legacy=True,
                                 archive_cutoff=None, post_cutoff=None))
        TestGroupableSubset._patch(monkeypatch, legacy)
        with caplog.at_level(logging.INFO):
            prepared = TestGroupableSubset._prepare()
        rejected = len(template) - len(TestGroupableSubset._eligible(template))
        assert rejected > 0
        warned = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"
                  and r.getMessage().startswith("Eligibility prefilter re-check")]
        assert len(warned) == 1 and warned[0].startswith(
            f"Eligibility prefilter re-check: {rejected} of {len(template)} markets rejected")
        # ...and the line above it names both counts, never all of them eligible.
        assert (f"Markets to analyze: {len(template)} assembled as eligible, "
                f"{len(template) - rejected} still eligible after the re-check "
                f"below — the eligibility prefilter ({self.TAG}) ran during "
                f"assembly, but this legacy cache records no count") in caplog.text
        # ...and the dropped records are dropped exactly as before.
        TestGroupableSubset._patch(monkeypatch, template)
        assert prepared.all_pairs == TestGroupableSubset._prepare().all_pairs


class TestEntriesForBand:
    """_entries_for_band reads the start date, horizon and ladder flag FROM the
    candidates — never as arguments — and hands the band verbatim to every
    _find_entry call."""

    @staticmethod
    def _candidates(**overrides):
        pair = ({"ticker": "A"}, {"ticker": "B"}, "canon", "key")
        fields = {
            "all_pairs": [(pair, "time_series"), (pair, "same_title")],
            "candles_by_ticker": {"A": ["ca"], "B": ["cb"]},
            "label_coverage": None,
            "start_date": date(2026, 3, 2),
            "max_horizon_days": 21,
            "same_event_ladders": None,
        }
        fields.update(overrides)
        return backtester._Candidates(**fields)

    def test_the_entry_pass_takes_no_start_horizon_or_ladder_argument(self):
        # Risk 2 of the plan: a second copy of any of these could disagree
        # with the one extraction and the candle fetch used.
        params = inspect.signature(backtester._entries_for_band).parameters
        assert list(params) == ["candidates", "spread_band", "pair_types", "tier_floors",
                                "_pairs"]
        assert params["pair_types"].kind is inspect.Parameter.KEYWORD_ONLY
        # PB7's private subset of pairs to scan: keyword-only, defaulting to
        # the full candidates.all_pairs scan
        assert params["_pairs"].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["_pairs"].default is None

    @pytest.mark.parametrize("ladders", [None, True, False])
    def test_find_entry_receives_the_candidates_own_inputs(self, monkeypatch, ladders):
        seen: list = []

        def _spy(ca, cb, mA, mB, pair_type, start_date, **kwargs):
            seen.append((ca, cb, pair_type, start_date, kwargs))
            return None

        monkeypatch.setattr(backtester, "_find_entry", _spy)
        backtester._entries_for_band(
            self._candidates(same_event_ladders=ladders), spread_band=(0.3, 0.6),
        )
        assert seen == [
            (["ca"], ["cb"], pt, date(2026, 3, 2),
             {"max_horizon_days": 21, "same_event_ladders": ladders,
              "spread_band": (0.3, 0.6), "tier_floors": True})
            for pt in ("time_series", "same_title")
        ]

    def test_pair_types_filters_without_reordering(self, monkeypatch):
        seen: list = []
        monkeypatch.setattr(backtester, "_find_entry",
                            lambda *a, **k: seen.append(a[4]))
        backtester._entries_for_band(self._candidates(), pair_types=("same_title",))
        assert seen == ["same_title"]
        seen.clear()
        # Order of the pair_types tuple is irrelevant: scan order wins
        backtester._entries_for_band(self._candidates(),
                                     pair_types=("same_title", "time_series"))
        assert seen == ["time_series", "same_title"]

    @pytest.mark.parametrize("bad", [("time-series",), "time_series", ("same_title", "x")])
    def test_an_unknown_pair_type_is_refused(self, bad):
        with pytest.raises(ValueError, match="pair_types"):
            backtester._entries_for_band(self._candidates(), pair_types=bad)

    @pytest.mark.parametrize("empty", [(), ""])
    def test_an_empty_pair_types_is_refused(self, monkeypatch, empty):
        # An empty selection scans nothing and would read as a band with no
        # tradeable pair.
        monkeypatch.setattr(backtester, "_find_entry",
                            lambda *a, **k: pytest.fail("nothing may be scanned"))
        with pytest.raises(ValueError, match="pair_types"):
            backtester._entries_for_band(self._candidates(), pair_types=empty)

    @pytest.mark.parametrize(
        "band,exc",
        [((0.6, 0.3), ValueError), ((0.1, 0.2, 0.3), ValueError), (0.3, TypeError)],
    )
    @pytest.mark.parametrize(
        "all_pairs,pair_types",
        [
            # No candidate at all
            ([], ("time_series", "same_title")),
            # Candidates exist, but none of the requested type
            ([(({"ticker": "A"}, {"ticker": "B"}, "c", "k"), "same_title")],
             ("time_series",)),
        ],
    )
    def test_an_invalid_band_is_refused_even_with_nothing_to_scan(
        self, monkeypatch, band, exc, all_pairs, pair_types,
    ):
        # Validated up front, not left to _find_entry: a pass that happens to
        # scan no pair must not return [] for a band that could never apply.
        monkeypatch.setattr(backtester, "_find_entry",
                            lambda *a, **k: pytest.fail("nothing may be scanned"))
        with pytest.raises(exc):
            backtester._entries_for_band(
                self._candidates(all_pairs=all_pairs), spread_band=band,
                pair_types=pair_types,
            )


class TestFindEntrySpreadBand:
    """_find_entry's backtest-only spread band: the floor is layered on the
    deadline-gap tier (and the leg-price-sum ceiling stays 1 - that floor), the
    ceiling refuses one Monday and lets the scan continue, and same-title
    pairs never read the band."""

    _START = date(2026, 1, 1)

    @staticmethod
    def _ts_markets():
        # 13-day gap: the SHORT (0.15) tier, so any band floor above 0.15 is
        # what binds.
        mA = {"ticker": "EARLY", "event_ticker": "E1",
              "close_time": "2026-02-01T00:00:00+00:00"}
        mB = {"ticker": "LATE", "event_ticker": "E2",
              "close_time": "2026-02-14T00:00:00+00:00"}
        return mA, mB

    def _ts(self, a_candles, b_candles, band):
        mA, mB = self._ts_markets()
        return _find_entry(a_candles, b_candles, mA, mB, "time_series", self._START,
                           spread_band=band)

    def _ts_one_monday(self, pA, pB, nB, band):
        return self._ts([_candle(_MONDAY_TS, pA, 1.0 - pA)],
                        [_candle(_MONDAY_TS, pB, nB)], band)

    # ── band cases ──────────────────────────────────────────────────────────

    def test_a_monday_above_the_ceiling_is_skipped_and_a_later_one_enters(self):
        # Monday 1 spread 0.70 is above the 0.60 ceiling; Monday 2's 0.50 is
        # inside 0.30-0.60. `continue`, never `return None`: the scan must
        # reach Monday 2.
        a = [_candle(_MONDAY_TS, 0.20, 0.80), _candle(_MONDAY2_TS, 0.20, 0.80)]
        b = [_candle(_MONDAY_TS, 0.90, 0.10), _candle(_MONDAY2_TS, 0.70, 0.30)]
        entry = self._ts(a, b, (0.30, 0.60))
        assert entry is not None
        assert entry["entry_date"] == date(2026, 1, 12)
        assert entry["pB"] - entry["pA"] == pytest.approx(0.50)
        # CONTROL: with no band Monday 1 itself enters, so the skip above is
        # the ceiling's doing and nothing else's.
        assert self._ts(a, b, None)["entry_date"] == date(2026, 1, 5)

    def test_a_raised_floor_refuses_a_short_gap_spread_the_tier_admits(self):
        # 0.30 - 0.10 = 0.20 clears the short tier (0.15) but not a 0.30 band
        # floor. The candle is deliberately CROSSED (B's NO ask 0.55 is below
        # 1 - its YES ask 0.30): candle closes are independent series, and
        # only a crossed quote keeps pA + nB (0.65) inside the raised floor's
        # own sum ceiling (0.70), so the refusal below is the floor's alone.
        assert 0.10 + 0.55 <= 1.0 - 0.30
        assert self._ts_one_monday(0.10, 0.30, 0.55, (0.30, 1.0)) is None
        assert self._ts_one_monday(0.10, 0.30, 0.55, None) is not None

    def test_the_raised_floor_also_lowers_the_leg_price_sum_ceiling(self):
        # Short gap, spread 0.40 (above the 0.30 floor), pA + nB = 0.75: kept
        # under the tier's 0.85 ceiling, refused under the raised floor's
        # 1 - 0.30 = 0.70. Kills a sum ceiling left on the tier.
        assert 0.20 + 0.55 == pytest.approx(0.75)
        assert self._ts_one_monday(0.20, 0.60, 0.55, (0.30, 1.0)) is None
        kept = self._ts_one_monday(0.20, 0.60, 0.55, (0.0, 1.0))
        assert kept is not None and kept["entry_date"] == date(2026, 1, 5)

    # ── epsilon on every band comparison (TS-09) ────────────────────────────

    @pytest.mark.parametrize(
        "label,pA,pB,nB,band,evaluates_to,bound",
        [
            # floor: 0.47 - 0.17 evaluates a hair UNDER the 0.30 floor
            ("floor", 0.17, 0.47, 0.50, (0.30, 1.0), 0.47 - 0.17, 0.30),
            # sum ceiling: 0.15 + 0.55 evaluates a hair OVER 1 - 0.30
            ("sum ceiling", 0.15, 0.60, 0.55, (0.30, 1.0), 0.15 + 0.55, 0.70),
            # ceiling: 0.90 - 0.30 evaluates a hair OVER the 0.60 ceiling
            ("ceiling", 0.30, 0.90, 0.10, (0.0, 0.60), 0.90 - 0.30, 0.60),
        ],
    )
    def test_a_value_exactly_on_a_band_bound_is_kept(
        self, label, pA, pB, nB, band, evaluates_to, bound,
    ):
        # Guard the fixture first: each row must genuinely sit off its bound
        # by float noise, or the row stops testing the epsilon at all.
        assert evaluates_to != bound
        assert evaluates_to == pytest.approx(bound, abs=1e-12)
        assert {"floor": 0.29999999999999993, "sum ceiling": 0.7000000000000001,
                "ceiling": 0.6000000000000001}[label] == evaluates_to
        entry = self._ts_one_monday(pA, pB, nB, band)
        assert entry is not None, f"{label} row was refused for float noise"

    def test_the_default_ceiling_never_fires(self):
        # The widest spread two live [0.01, 0.99] YES asks can make is 0.98;
        # the default ceiling of 1.0 must keep it.
        assert self._ts_one_monday(0.01, 0.99, 0.01, None) is not None

    # ── the band on a ladder, and on nothing but time-series ────────────────

    def test_a_same_event_ladder_honours_the_band_ceiling(self):
        # Rungs closing at one instant, stated gap 19 (the 0.30 tier); spread
        # 0.50 enters with no band and is refused above a 0.40 ceiling.
        mA = _ladder_member("RUNG-EARLY", "by March 1, 2026",
                            close=datetime(2026, 3, 20, tzinfo=UTC))
        mB = _ladder_member("RUNG-LATE", "by March 20, 2026",
                            close=datetime(2026, 3, 20, tzinfo=UTC))
        ca, cb = [_candle(_MONDAY_TS, 0.20, 0.80)], [_candle(_MONDAY_TS, 0.70, 0.35)]

        def entry(band):
            return _find_entry(ca, cb, mA, mB, "time_series", self._START,
                               same_event_ladders=True, spread_band=band)

        assert entry(None)["gap_days"] == 19
        assert entry((0.0, 0.40)) is None

    @staticmethod
    def _st_entry(pa_yes, pb_yes, band):
        mA = {"ticker": "SA", "event_ticker": "SERA-1",
              "close_time": "2026-02-01T00:00:00+00:00"}
        mB = {"ticker": "SB", "event_ticker": "SERB-1",
              "close_time": "2026-02-01T00:00:00+00:00"}
        return _find_entry([_candle(_MONDAY_TS, pa_yes, 1.0 - pa_yes)],
                           [_candle(_MONDAY_TS, pb_yes, 1.0 - pb_yes)],
                           mA, mB, "same_title", date(2026, 1, 1), spread_band=band)

    @pytest.mark.parametrize(
        "band",
        [(lo, hi) for lo in SPREAD_BAND_SWEEP_FLOORS for hi in SPREAD_BAND_SWEEP_CEILINGS]
        + [(0.40, 0.45)],
    )
    @pytest.mark.parametrize("pa_yes,pb_yes", [(0.85, 0.15), (0.55, 0.45)])
    def test_same_title_is_untouched_at_any_band(self, band, pa_yes, pb_yes):
        # A 0.70 gap (above every ceiling) and a 0.10 gap (below every floor
        # above 0.10) enter exactly as they do with no band.
        default = self._st_entry(pa_yes, pb_yes, None)
        assert default is not None
        assert self._st_entry(pa_yes, pb_yes, band) == default

    @pytest.mark.parametrize("pair_type", ["time_series", "same_title"])
    def test_an_invalid_band_is_refused_before_any_data_check(self, pair_type):
        # Resolved before the close_time early return, so a caller bug
        # surfaces even on a pair that could never enter.
        with pytest.raises(ValueError, match="spread band"):
            _find_entry([], [], {}, {}, pair_type, date(2026, 1, 1),
                        spread_band=(0.6, 0.3))

    @pytest.mark.parametrize("band,exc", [((0.1, 0.2, 0.3), ValueError), (0.3, TypeError)])
    def test_a_malformed_band_raises_what_the_docstring_names(self, band, exc):
        with pytest.raises(exc):
            _find_entry([], [], {}, {}, "time_series", date(2026, 1, 1),
                        spread_band=band)


class TestBacktestTradePopulationLabels:
    """BacktestTrade.event_ticker / same_event_ladder: reporting-only labels
    set by _simulate_at_discount, so a report can split the ladder,
    cross-event and same-title populations and group P&L by event."""

    def test_a_ladder_a_cross_event_pair_and_a_same_title_pair(self, monkeypatch):
        golden = TestPrepareEntriesGolden()
        golden._patch(monkeypatch)
        entries, _ = backtester._prepare_entries(
            MagicMock(), MagicMock(), golden._START, True, None,
            same_event_ladders=True,
        )
        point = backtester._simulate_at_discount(entries, golden._START, 10_000.0)
        by_a = {t.ticker_a: t for t in point.trades}
        assert set(by_a) == {"EA", "RUNG-EARLY", "SB"}
        assert (by_a["EA"].event_ticker, by_a["EA"].same_event_ladder) == ("EVA", False)
        assert (by_a["RUNG-EARLY"].event_ticker,
                by_a["RUNG-EARLY"].same_event_ladder) == ("KXSTARSHIP-14", True)
        # Market A of a same-title trade is the pricier side AFTER
        # canonicalization (SB here), and a same-title pair is never a ladder.
        assert (by_a["SB"].event_ticker, by_a["SB"].same_event_ladder) == ("SERB-1", False)

    def test_two_missing_event_tickers_are_not_one_event(self):
        # "" == "" must not read as a ladder: an unknown event cannot be
        # shown to be one event.
        def mk(ticker, close):
            return {"ticker": ticker, "result": "yes",
                    "close_time": f"{close}T00:00:00+00:00",
                    "settlement_ts": f"{close}T12:00:00+00:00"}
        entry = {"entry_date": date(2026, 1, 5), "pA": 0.30, "pB": 0.60,
                 "nA": 0.70, "nB": 0.40, "gap_days": 13,
                 "mA": mk("XA", "2026-02-01"), "mB": mk("XB", "2026-02-14")}
        point = backtester._simulate_at_discount(
            [{"pair_type": "time_series", "canon": "c", "group_key": "c",
              "entry": entry}],
            date(2026, 1, 1), 10_000.0,
        )
        assert len(point.trades) == 1
        assert point.trades[0].event_ticker == ""
        assert point.trades[0].same_event_ladder is False

    def test_the_fields_default_for_a_hand_built_trade(self):
        t = TestEquityCurveOpensAtTheInitialBalance()._trade(10, 0.30, 0.40, "yes", 0.0)
        assert t.event_ticker == ""
        assert t.same_event_ladder is False
        # The per-leg report fields default too
        assert (t.subtitle_a, t.subtitle_b) == ("", "")
        assert (t.close_date_a, t.close_date_b, t.settled_date_a, t.settled_date_b) == (
            None, None, None, None)

    def test_each_legs_label_close_and_settlement_dates_are_recorded(self):
        # Market A closes and settles before market B, and each carries its own
        # outcome label: the trade must keep them per leg, not collapse them
        # into exit_date (which is the LATER settlement).
        mA = {"ticker": "XA", "result": "yes", "subtitle": "Label A",
              "close_time": "2026-02-01T00:00:00+00:00",
              "settlement_ts": "2026-02-02T12:00:00+00:00"}
        mB = {"ticker": "XB", "result": "yes", "subtitle": None,
              "close_time": "2026-02-14T00:00:00+00:00",
              "settlement_ts": "2026-02-15T12:00:00+00:00"}
        entry = {"entry_date": date(2026, 1, 5), "pA": 0.30, "pB": 0.60,
                 "nA": 0.70, "nB": 0.40, "gap_days": 13, "mA": mA, "mB": mB}
        point = backtester._simulate_at_discount(
            [{"pair_type": "time_series", "canon": "c", "group_key": "c", "entry": entry}],
            date(2026, 1, 1), 10_000.0,
        )
        (t,) = point.trades
        assert (t.subtitle_a, t.subtitle_b) == ("Label A", "")
        assert (t.close_date_a, t.close_date_b) == (date(2026, 2, 1), date(2026, 2, 14))
        assert (t.settled_date_a, t.settled_date_b) == (date(2026, 2, 2), date(2026, 2, 15))
        assert t.exit_date == date(2026, 2, 15)


# ─── PB3: the band x k x population sweep ────────────────────────────────────

# The 36 grid bands, resolved exactly as the sweep resolves them.
_GRID_BANDS = sorted({(float(lo), float(hi)) for lo in SPREAD_BAND_SWEEP_FLOORS
                      for hi in SPREAD_BAND_SWEEP_CEILINGS})


class _LogCapture(logging.Handler):
    """Collect every record's message; a class-scoped fixture cannot use caplog."""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _completion_prefixes(messages: list[str]) -> list[str]:
    """The text up to the ':' of every "Backtest complete" line — the part
    TS-21 requires to be unique within one sweep."""
    return [m.split(":")[0] for m in messages if m.startswith("Backtest complete")]


@pytest.fixture(scope="class")
def golden_band_sweep():
    """ONE full band sweep (36 bands x 13 k, ladders on) over the
    TestPrepareEntriesGolden fixture, with spies on every seam the tests
    below read, plus a band_sweep=False run of the same fixture. Class-scoped
    so the whole class pays for one sweep, not one per test."""
    mp = pytest.MonkeyPatch()
    handler = _LogCapture()
    root = logging.getLogger()
    old_level = root.level
    try:
        golden = TestPrepareEntriesGolden()
        golden._patch(mp)
        calls: dict = {"prepare": 0, "entries": [], "simulate": [], "candidates": [],
                       "scanned": []}
        real_prepare = backtester._prepare_candidates
        real_entries = backtester._entries_for_band
        real_simulate = backtester._simulate_at_discount

        def prepare_spy(*a, **kw):
            calls["prepare"] += 1
            c = real_prepare(*a, **kw)
            calls["candidates"].append(c)
            return c

        def entries_spy(candidates, spread_band=None, *, pair_types=("time_series", "same_title"),
                        **private):
            out = real_entries(candidates, spread_band, pair_types=pair_types, **private)
            calls["entries"].append((spread_band, pair_types, out))
            # PB7: which pairs each pass scanned (None = candidates.all_pairs)
            calls["scanned"].append((spread_band, pair_types, private.get("_pairs"),
                                     candidates.all_pairs))
            return out

        def simulate_spy(raw_entries, start_date, initial_balance, k=None,
                         spread_band=None, population="all", *, tier_floors=True):
            # Were the candles and the pair list already released when this
            # simulation ran?
            released = all(not hasattr(c, "candles_by_ticker") and not hasattr(c, "all_pairs")
                           for c in calls["candidates"])
            point = real_simulate(raw_entries, start_date, initial_balance, k=k,
                                  spread_band=spread_band, population=population,
                                  tier_floors=tier_floors)
            calls["simulate"].append({"entries": list(raw_entries), "band": spread_band,
                                      "population": population, "k": k,
                                      "tier_floors": tier_floors,
                                      "point": point, "released": released})
            return point

        mp.setattr(backtester, "_prepare_candidates", prepare_spy)
        mp.setattr(backtester, "_entries_for_band", entries_spy)
        mp.setattr(backtester, "_simulate_at_discount", simulate_spy)
        root.setLevel(logging.INFO)
        root.addHandler(handler)
        result = run_backtest_sweep(
            hist_client=MagicMock(), live_client=MagicMock(),
            start_date=golden._START, initial_balance=10_000.0,
            same_event_ladders=True, band_sweep=True,
        )
        root.removeHandler(handler)
        mp.undo()

        # The same fixture with the band sweep OFF — the pre-band single-band
        # k sweep, the oracle for points/primary.
        golden._patch(mp)
        single = run_backtest_sweep(
            hist_client=MagicMock(), live_client=MagicMock(),
            start_date=golden._START, initial_balance=10_000.0,
            same_event_ladders=True, band_sweep=False,
        )
        yield SimpleNamespace(result=result, single=single, calls=calls,
                              messages=handler.messages, start=golden._START)
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
        mp.undo()


@pytest.mark.usefixtures("golden_band_sweep")
class TestBandSweep:
    """run_backtest_sweep(band_sweep=True): every band of the config grid x
    every swept k over ONE fetch, with standalone ladder / cross-event /
    same-title populations, a split-half check and a concentration check.

    The fixture (TestPrepareEntriesGolden, ladders on) is small enough to
    reason about by hand: the ladder (spread 0.40, 19-day stated gap) enters
    at every grid band; the cross-event pairs are EA/EB (0.30, 13-day gap —
    kept at floors <= 0.30), TA/TB (0.15 — floor 0 only) and WA/WB (0.98 —
    ceiling 1.0 only), so the cross-event population is empty at exactly the
    ten bands with floor 0.35 or 0.40 and a ceiling below 1.0; and the
    same-title pair trades at every band and k."""

    # Floors 0 / 0.20 / 0.25 / 0.30 keep EA/EB at every ceiling; floors 0.35
    # and 0.40 keep only WA/WB, and only at the 1.0 ceiling.
    _CROSS_BANDS = sorted(
        [(lo, hi) for lo in (0.0, 0.20, 0.25, 0.30) for hi in SPREAD_BAND_SWEEP_CEILINGS]
        + [(0.35, 1.0), (0.40, 1.0)])

    def test_scenario_counts(self, golden_band_sweep):
        res = golden_band_sweep.result
        by_pop: dict = {}
        for p in res.scenarios:
            by_pop.setdefault(p.population, []).append(p)
        assert set(by_pop) == {"all", "time_series", "ladder", "cross"}
        # 36 bands x 13 k "all" scenarios, on a rectangular grid
        assert len(_GRID_BANDS) == 36 and len(INTERVAL_DISCOUNT_SWEEP) == 13
        assert sorted((p.spread_band, p.k) for p in by_pop["all"]) == sorted(
            (b, k) for b in _GRID_BANDS for k in INTERVAL_DISCOUNT_SWEEP)
        # ... each with both robustness checks
        assert all(p.halves is not None and p.ex_top_event is not None
                   for p in by_pop["all"])
        # PB7: a time-series point at every cell (the ladder enters at every
        # band), each with its own split-half check, and an ex-top check
        # exactly where it traded an event (at k = 1.00 nothing trades).
        assert sorted((p.spread_band, p.k) for p in by_pop["time_series"]) == sorted(
            (b, k) for b in _GRID_BANDS for k in INTERVAL_DISCOUNT_SWEEP)
        assert all(p.halves is not None for p in by_pop["time_series"])
        assert all((p.ex_top_event is not None) == any(t.event_ticker for t in p.trades)
                   for p in by_pop["time_series"])
        # A ladder point at every band (the ladder enters everywhere) and a
        # cross point exactly where the cross population is non-empty — an
        # empty population is skipped, never simulated as an empty scenario.
        assert sorted({p.spread_band for p in by_pop["ladder"]}) == _GRID_BANDS
        assert len(by_pop["ladder"]) == 36 * 13
        assert sorted({p.spread_band for p in by_pop["cross"]}) == self._CROSS_BANDS
        assert len(by_pop["cross"]) == 26 * 13
        # ... plus ONE same-title point, outside the scenarios
        assert res.same_title_point is not None
        assert res.same_title_point.population == "same_title"
        assert [t.pair_type for t in res.same_title_point.trades] == ["same_title"]

    def test_scenarios_are_ordered_band_then_k_then_population(self, golden_band_sweep):
        order = {"all": 0, "time_series": 1, "ladder": 2, "cross": 3}
        keys = [(p.spread_band, p.k, order[p.population])
                for p in golden_band_sweep.result.scenarios]
        assert keys == sorted(keys)

    def test_points_and_primary_equal_a_single_band_run(self, golden_band_sweep):
        res, single = golden_band_sweep.result, golden_band_sweep.single
        assert single.scenarios == [] and single.same_title_point is None
        assert single.split_date is None
        assert [p.k for p in res.points] == [p.k for p in single.points]
        for swept, alone in zip(res.points, single.points, strict=True):
            assert (swept.spread_band, swept.population) == (alone.spread_band,
                                                             alone.population)
            assert [astuple(t) for t in swept.trades] == [astuple(t) for t in alone.trades]
            pd.testing.assert_frame_equal(swept.equity_df, alone.equity_df)
        assert res.primary.k == single.primary.k
        assert [astuple(t) for t in res.primary.trades] == [
            astuple(t) for t in single.primary.trades]
        pd.testing.assert_frame_equal(res.primary.equity_df, single.primary.equity_df)
        assert res.calibration == single.calibration
        # The primary band's points are the default band's, and ARE scenarios
        assert all(p.spread_band == BACKTEST_DEFAULT_SPREAD_BAND for p in res.points)
        assert any(p is res.primary for p in res.points)
        for p in res.points:
            assert any(s is p for s in res.scenarios)

    def test_preparation_once_and_one_time_series_pass_per_band(self, golden_band_sweep):
        calls = golden_band_sweep.calls
        assert calls["prepare"] == 1
        ts_bands = [band for band, types, _ in calls["entries"] if types == ("time_series",)]
        st_passes = [band for band, types, _ in calls["entries"] if types == ("same_title",)]
        assert sorted(ts_bands) == _GRID_BANDS           # each band exactly once
        assert len(st_passes) == 1                       # same-title entries once
        assert len(calls["entries"]) == 36 + 1           # and nothing else

    def test_the_candles_are_released_before_any_simulation(self, golden_band_sweep):
        # ... and the pair list with them: nothing after the entry passes
        # reads either, so neither may stay resident through the simulations.
        calls = golden_band_sweep.calls
        assert calls["simulate"] and all(c["released"] for c in calls["simulate"])

    def test_every_completion_prefix_is_unique_and_468_are_all(self, golden_band_sweep):
        # The band_sweep twin of TS-21's uniqueness test.
        prefixes = _completion_prefixes(golden_band_sweep.messages)
        assert len(prefixes) == len(set(prefixes))
        labels = [p.rsplit(", ", 1)[1] for p in prefixes]
        assert labels.count("all") == 468
        assert labels.count("time_series") == 468
        # The time-series point re-simulates without its top event only where
        # it traded one (an ex-top run needs an event to drop).
        ts_ex_top = sum(1 for p in golden_band_sweep.result.scenarios
                        if p.population == "time_series" and p.ex_top_event is not None)
        assert labels.count("time_series/ex-top") == ts_ex_top
        # 468 x (all + ladder + H1 + H2 + ex-top) + 26 x 13 cross + 1 same-title
        # + 468 x (time_series + its H1 + its H2) + its ex-top runs
        assert len(prefixes) == 468 * 5 + 26 * 13 + 1 + 468 * 3 + ts_ex_top
        assert labels.count("same_title") == 1

    def test_every_scenario_s_entries_lie_within_its_stamped_band(self, golden_band_sweep):
        eps = backtester.PRICE_EPSILON
        by_point = {id(c["point"]): c for c in golden_band_sweep.calls["simulate"]}
        sizes = set()
        for point in golden_band_sweep.result.scenarios:
            call = by_point[id(point)]
            # The stamp IS the band the entries were detected under
            assert call["band"] == point.spread_band
            lo, hi = point.spread_band
            ts = [rec["entry"] for rec in call["entries"] if rec["pair_type"] == "time_series"]
            sizes.add(len(ts))
            for e in ts:
                spread = e["pB"] - e["pA"]
                assert spread <= hi + eps
                assert spread >= min_price_diff_for_gap(e["gap_days"], spread_min=lo) - eps
            for t in point.trades:
                if t.pair_type == "time_series":
                    assert lo - eps <= t.entry_pB - t.entry_pA <= hi + eps
        # Not vacuous: the bands really did select different entry sets
        assert len(sizes) > 1

    def test_robustness_checks_live_only_on_band_sweep_all_points(self, golden_band_sweep):
        res, single = golden_band_sweep.result, golden_band_sweep.single
        for p in res.scenarios:
            if p.population == "all":
                assert p.halves is not None and p.ex_top_event is not None
            elif p.population == "time_series":
                # PB7: the population the dashboard's banner reads carries
                # its own checks too (ex-top only where it traded an event)
                assert p.halves is not None
                assert (p.ex_top_event is not None) == any(t.event_ticker for t in p.trades)
            else:
                assert p.halves is None and p.ex_top_event is None
        assert res.same_title_point.halves is None
        assert res.same_title_point.ex_top_event is None
        assert all(p.halves is None and p.ex_top_event is None for p in single.points)
        # No retained point carries a split-half / ex-top run label
        assert all("/" not in p.population for p in res.scenarios)

    def test_the_same_title_point_is_simulated_once(self, golden_band_sweep):
        sims = [c for c in golden_band_sweep.calls["simulate"]
                if c["population"] == "same_title"]
        assert len(sims) == 1
        assert sims[0]["point"] is golden_band_sweep.result.same_title_point
        # Band- and k-independent: its band stamp is None, its k the primary's
        assert golden_band_sweep.result.same_title_point.spread_band is None
        assert golden_band_sweep.result.same_title_point.k == TIME_SERIES_INTERVAL_PROB_DISCOUNT
        assert all(rec["pair_type"] == "same_title" for rec in sims[0]["entries"])

    def test_populations_are_standalone_simulations_of_their_entries(self, golden_band_sweep):
        start = golden_band_sweep.start
        by_point = {id(c["point"]): c for c in golden_band_sweep.calls["simulate"]}
        for p in golden_band_sweep.result.scenarios:
            entries = by_point[id(p)]["entries"]
            if p.population == "ladder":
                assert entries and all(
                    rec["entry"]["mA"]["event_ticker"] == rec["entry"]["mB"]["event_ticker"]
                    for rec in entries)
                assert all(t.same_event_ladder for t in p.trades)
            elif p.population == "cross":
                assert entries and all(
                    rec["pair_type"] == "time_series"
                    and rec["entry"]["mA"]["event_ticker"] != rec["entry"]["mB"]["event_ticker"]
                    for rec in entries)
                assert not any(t.same_event_ladder for t in p.trades)
        # A population's result is its own run from the initial balance, not a
        # slice of the "all" run: at the default band and k it re-simulates to
        # the same trades on a fresh balance.
        ladder = next(p for p in golden_band_sweep.result.scenarios
                      if p.population == "ladder" and p.k == TIME_SERIES_INTERVAL_PROB_DISCOUNT
                      and p.spread_band == BACKTEST_DEFAULT_SPREAD_BAND)
        again = backtester._simulate_at_discount(
            by_point[id(ladder)]["entries"], start, 10_000.0,
            k=ladder.k, spread_band=ladder.spread_band, population="ladder")
        assert [astuple(t) for t in again.trades] == [astuple(t) for t in ladder.trades]
        assert again.trades[0].balance_at_entry == pytest.approx(10_000.0)

    def test_halves_and_ex_top_event_are_resimulations(self, golden_band_sweep):
        res, start = golden_band_sweep.result, golden_band_sweep.start
        by_point = {id(c["point"]): c for c in golden_band_sweep.calls["simulate"]}
        entries = by_point[id(res.primary)]["entries"]
        # split_date is the median_low of the primary band's entry dates: four
        # of its five entries enter on Monday 1, the ladder on Monday 2.
        assert sorted(rec["entry"]["entry_date"] for rec in entries) == [
            date(2026, 1, 5)] * 4 + [date(2026, 1, 12)]
        assert res.split_date == date(2026, 1, 5)
        # H1 (strictly before the split) is empty, so H2 is every entry and
        # must reproduce the primary's own return and trade count.
        h = res.primary.halves
        assert (h.h1_return, h.h1_trades) == (0.0, 0)
        final = float(res.primary.equity_df["portfolio_value"].iloc[-1])
        assert h.h2_return == pytest.approx((final - 10_000.0) / 10_000.0)
        assert h.h2_trades == len(res.primary.trades)
        # ex_top_event: the event with the largest summed profit, and the
        # return of a re-simulation WITHOUT that event's entries.
        pnl: dict = {}
        for t in res.primary.trades:
            pnl[t.event_ticker] = pnl.get(t.event_ticker, 0.0) + t.profit
        top = max(pnl, key=pnl.get)
        assert res.primary.ex_top_event[0] == top
        rest = [rec for rec in entries if rec["entry"]["mA"]["event_ticker"] != top]
        again = backtester._simulate_at_discount(rest, start, 10_000.0,
                                                 k=res.primary.k,
                                                 spread_band=res.primary.spread_band)
        final_without = float(again.equity_df["portfolio_value"].iloc[-1])
        assert res.primary.ex_top_event[1] == pytest.approx(
            (final_without - 10_000.0) / 10_000.0)

    @staticmethod
    def _return(point):
        return (float(point.equity_df["portfolio_value"].iloc[-1]) - 10_000.0) / 10_000.0

    def test_every_all_point_s_checks_read_its_own_entries(self, golden_band_sweep):
        # The test above re-simulates the PRIMARY cell by hand; this one pins
        # the wiring at EVERY one of the 468 "all" cells: each split-half and
        # ex-top run is handed that point's OWN entries (never the primary
        # band's), split at the one split date, at the point's own k and
        # band — and the numbers on the point are those runs' results.
        res = golden_band_sweep.result
        calls = golden_band_sweep.calls["simulate"]
        by_point = {id(c["point"]): c for c in calls}
        by_key: dict = {}
        for c in calls:
            key = (c["band"], c["point"].k, c["population"])
            assert key not in by_key          # one simulation per (band, k, label)
            by_key[key] = c
        split = res.split_date
        alls = [p for p in res.scenarios if p.population == "all"]
        assert len(alls) == 468
        contents = set()
        for p in alls:
            own = by_point[id(p)]["entries"]
            contents.add(tuple(sorted((r["entry"]["mA"]["ticker"], r["entry"]["mB"]["ticker"])
                                      for r in own)))
            h1 = by_key[(p.spread_band, p.k, "all/H1")]
            h2 = by_key[(p.spread_band, p.k, "all/H2")]
            assert [id(r) for r in h1["entries"]] == [
                id(r) for r in own if r["entry"]["entry_date"] < split]
            assert [id(r) for r in h2["entries"]] == [
                id(r) for r in own if r["entry"]["entry_date"] >= split]
            assert p.halves == backtester.HalfSplit(
                h1_return=self._return(h1["point"]), h2_return=self._return(h2["point"]),
                h1_trades=len(h1["point"].trades), h2_trades=len(h2["point"].trades),
                h1_entries=len(h1["entries"]), h2_entries=len(h2["entries"]))
            pnl: dict = {}
            for t in p.trades:
                if t.event_ticker:
                    pnl[t.event_ticker] = pnl.get(t.event_ticker, 0.0) + t.profit
            top = min(pnl, key=lambda ev: (-pnl[ev], ev))
            ex = by_key[(p.spread_band, p.k, "all/ex-top")]
            assert [id(r) for r in ex["entries"]] == [
                id(r) for r in own if r["entry"]["mA"]["event_ticker"] != top]
            assert p.ex_top_event == (top, self._return(ex["point"]))
        # Not vacuous: the bands hand their checks different entry SETS, not
        # merely different copies of one set.
        assert len(contents) > 1

    def test_a_non_primary_cell_s_checks_re_simulate_independently(self, golden_band_sweep):
        # (0.35, 1.0) keeps a different entry set from the primary band's, so
        # a check computed from the primary's entries would not match here.
        res, start = golden_band_sweep.result, golden_band_sweep.start
        by_point = {id(c["point"]): c for c in golden_band_sweep.calls["simulate"]}
        p = next(s for s in res.scenarios if s.population == "all"
                 and s.spread_band == (0.35, 1.0) and s.k == TIME_SERIES_INTERVAL_PROB_DISCOUNT)
        own = by_point[id(p)]["entries"]
        primary_own = by_point[id(res.primary)]["entries"]
        assert sorted(r["entry"]["mA"]["ticker"] for r in own) != sorted(
            r["entry"]["mA"]["ticker"] for r in primary_own)
        for half, ret, n in (
            ([r for r in own if r["entry"]["entry_date"] < res.split_date],
             p.halves.h1_return, p.halves.h1_trades),
            ([r for r in own if r["entry"]["entry_date"] >= res.split_date],
             p.halves.h2_return, p.halves.h2_trades),
        ):
            alone = backtester._simulate_at_discount(half, start, 10_000.0, k=p.k)
            assert ret == pytest.approx(self._return(alone))
            assert n == len(alone.trades)
        top, without = p.ex_top_event
        rest = [r for r in own if r["entry"]["mA"]["event_ticker"] != top]
        again = backtester._simulate_at_discount(rest, start, 10_000.0, k=p.k)
        assert without == pytest.approx(self._return(again))

    def test_every_band_has_its_own_calibration(self, golden_band_sweep):
        res = golden_band_sweep.result
        assert sorted(res.calibrations_by_band) == _GRID_BANDS
        assert res.calibration is res.calibrations_by_band[BACKTEST_DEFAULT_SPREAD_BAND]
        # At a 0.35 floor only the ladder (19-day stated gap, the 0.30 tier)
        # is measurable, and its row is labelled with the floor it cleared.
        raised = res.calibrations_by_band[(0.35, 1.0)]
        assert [(b.label, b.tier) for b in raised.buckets] == [("16-30d", 0.35)]
        # ... while the default band labels the tiers alone
        assert {b.label: b.tier for b in res.calibration.buckets} == {
            "8-15d": min_price_diff_for_gap(15), "16-30d": min_price_diff_for_gap(30)}
        # Only the primary band's table is logged
        headers = [m for m in golden_band_sweep.messages
                   if m.startswith("Interval-discount calibration")]
        assert len(headers) == 1

    def test_phase_one_announces_every_band(self, golden_band_sweep):
        announced = [m for m in golden_band_sweep.messages if m.startswith("Spread band ")]
        assert len(announced) == 36
        assert announced[0] == "Spread band 1/36: 0-0.5"
        assert "Spread band 6/36: 0-1 (primary)" in announced
        assert sum(m.endswith("(primary)") for m in announced) == 1

    def test_the_sweep_records_the_resolved_ladder_setting(self, golden_band_sweep):
        assert golden_band_sweep.result.same_event_ladders is True
        assert golden_band_sweep.single.same_event_ladders is True


class TestBandSweepEdges:
    """The band sweep's other shapes: --no-sweep, an off-grid primary band, an
    infeasible window, and the argument handling at the top of
    run_backtest_sweep."""

    def _run(self, monkeypatch, **kwargs):
        golden = TestPrepareEntriesGolden()
        golden._patch(monkeypatch)
        return run_backtest_sweep(hist_client=MagicMock(), live_client=MagicMock(),
                                  start_date=golden._START, initial_balance=10_000.0,
                                  same_event_ladders=True, **kwargs)

    def test_no_sweep_with_the_band_sweep_gives_one_k_per_band(self, monkeypatch):
        res = self._run(monkeypatch, sweep=False, band_sweep=True)
        alls = [p for p in res.scenarios if p.population == "all"]
        assert sorted(p.spread_band for p in alls) == _GRID_BANDS     # 36 x 1
        assert {p.k for p in alls} == {TIME_SERIES_INTERVAL_PROB_DISCOUNT}
        assert res.points == [res.primary]

    def test_an_off_grid_primary_band_is_its_own_exact_band(self, monkeypatch):
        res = self._run(monkeypatch, sweep=False, band_sweep=True, spread_band=(0.33, 0.66))
        alls = [p for p in res.scenarios if p.population == "all"]
        assert len(alls) == 37
        assert res.primary.spread_band == (0.33, 0.66)
        assert any(p is res.primary for p in alls)
        assert (0.33, 0.66) in res.calibrations_by_band

    @pytest.mark.parametrize("band", [(0.3000001, 0.6), (0.1 + 0.2, 0.6)])
    def test_a_near_grid_primary_band_is_labelled_distinctly(self, monkeypatch, caplog, band):
        # %g prints both of these as "0.3-0.6", the label of a grid band the
        # sweep also runs; the announcement and every completion prefix must
        # still tell the 37 bands apart (TS-21).
        with caplog.at_level(logging.INFO):
            res = self._run(monkeypatch, sweep=False, band_sweep=True, spread_band=band)
        assert len([p for p in res.scenarios if p.population == "all"]) == 37
        messages = [r.getMessage() for r in caplog.records]
        labels = [m.split(": ", 1)[1].removesuffix(" (primary)")
                  for m in messages if m.startswith("Spread band ")]
        assert len(labels) == len(set(labels)) == 37
        assert "0.3-0.6" in labels
        prefixes = _completion_prefixes(messages)
        assert len(prefixes) == len(set(prefixes))

    def test_a_near_grid_primary_k_is_labelled_distinctly(self, monkeypatch, caplog):
        # "%.3f" prints 0.75 and 0.7500001 alike; the grid holds both.
        with caplog.at_level(logging.INFO):
            res = self._run(monkeypatch, interval_discount=0.7500001)
        assert len(res.points) == 14
        prefixes = _completion_prefixes([r.getMessage() for r in caplog.records])
        assert len(prefixes) == len(set(prefixes)) == 14
        assert "Backtest complete at k=0.750, band 0-1, all" in prefixes
        assert "Backtest complete at k=0.7500001, band 0-1, all" in prefixes

    def test_a_single_band_run_keeps_the_pre_band_log_wording(self, monkeypatch, caplog):
        # Band sweep off: the entry and re-simulation lines read as they did
        # before the band existed, and the band-sweep-only lines are absent.
        with caplog.at_level(logging.INFO):
            res = self._run(monkeypatch)
        assert len(res.points) == 13 and res.scenarios == []
        messages = [r.getMessage() for r in caplog.records]
        # The golden fixture prepares five entries: three cross-event pairs
        # and the same-title pair on Monday 1, the ladder on Monday 2.
        prepared = [m for m in messages if m.startswith("Prepared ")]
        assert prepared == ["Prepared 5 candidate entries for sizing"]
        assert ("Re-simulating 5 prepared entries at 13 interval discounts "
                "(primary k = 0.750)") in messages
        assert not any(m.startswith(("Same-title candidate entries", "Split-half check",
                                     "Simulating spread band", "Simulating the same-title"))
                       for m in messages)
        assert [m for m in messages if m.startswith("Spread band ")] == [
            "Spread band 1/1: 0-1 (primary)"]

    def test_the_infeasible_window_has_no_scenarios(self, monkeypatch):
        res = TestRunBacktestSweep()._infeasible(monkeypatch, band_sweep=True,
                                                 same_event_ladders=True)
        assert res.scenarios == []
        assert res.calibrations_by_band == {}
        assert res.same_title_point is None and res.split_date is None
        assert res.label_coverage is None
        # The resolved ladder setting is still recorded, and the empty point
        # still carries the band it would have been simulated at
        assert res.same_event_ladders is True
        assert res.primary.spread_band == BACKTEST_DEFAULT_SPREAD_BAND

    def test_the_infeasible_window_has_no_tier_off_family_either(self, monkeypatch):
        # The CLI's own flags (the band sweep AND its tier-floors-off family)
        # on a window with no Monday checkpoint: the family comes back empty,
        # like every other band-sweep payload, and nothing was fetched
        res = TestRunBacktestSweep()._infeasible(monkeypatch, band_sweep=True,
                                                 tier_off_sweep=True, same_event_ladders=True)
        assert res.tier_off_scenarios == []
        assert res.tier_off_calibrations_by_band == {}
        assert res.scenarios == [] and res.calibrations_by_band == {}
        assert res.primary.tier_floors is True
        assert backtester.max_trades_simulated(res) == 0

    def test_a_single_band_run_keeps_its_calibration_keyed_by_band(self, monkeypatch):
        res = self._run(monkeypatch, sweep=False)
        assert res.calibrations_by_band == {BACKTEST_DEFAULT_SPREAD_BAND: res.calibration}
        assert res.scenarios == [] and res.same_title_point is None
        assert res.primary.spread_band == BACKTEST_DEFAULT_SPREAD_BAND

    def test_an_invalid_band_fails_before_anything_is_logged_or_fetched(
        self, monkeypatch, caplog,
    ):
        monkeypatch.setattr(backtester, "fetch_all_settled_markets",
                            lambda *a, **k: pytest.fail("fetch must not run"))
        with caplog.at_level(logging.INFO), pytest.raises(ValueError, match="spread band"):
            run_backtest_sweep(MagicMock(), MagicMock(), date(2026, 1, 1), 1000.0,
                               spread_band=(0.6, 0.3))
        assert caplog.records == []

    @pytest.mark.parametrize("band,source", [
        (None, "config.BACKTEST_DEFAULT_SPREAD_BAND"),
        ((0.3, 0.6), "run-level override"),
    ])
    def test_the_resolved_band_is_logged_with_its_source(self, monkeypatch, caplog,
                                                         band, source):
        monkeypatch.setattr(backtester, "_prepare_candidates", lambda *a, **k: None)
        with caplog.at_level(logging.INFO):
            run_backtest_sweep(MagicMock(), MagicMock(), date(2026, 1, 1), 1000.0,
                               spread_band=band, band_sweep=True)
        label = "0-1" if band is None else "0.3-0.6"
        assert (f"Time-series spread band (backtest only): {label} ({source}); "
                "band sweep on") in caplog.text

    def test_a_harness_may_pass_no_band_to_the_second_half(self, monkeypatch):
        # _sweep_from_candidates resolves the band itself too, so a caller
        # holding its own _Candidates gets the default band.
        golden = TestPrepareEntriesGolden()
        golden._patch(monkeypatch)
        candidates = backtester._prepare_candidates(
            MagicMock(), MagicMock(), golden._START, True, None, same_event_ladders=True)
        res = backtester._sweep_from_candidates(
            candidates, 10_000.0, interval_discount=None, sweep=False,
            spread_band=None, band_sweep=False)
        assert res.primary.spread_band == BACKTEST_DEFAULT_SPREAD_BAND
        # ... and the candidates are consumed: a second sweep fails loudly
        # rather than running on no candles
        assert not hasattr(candidates, "candles_by_ticker")
        assert not hasattr(candidates, "all_pairs")
        with pytest.raises(AttributeError):
            backtester._sweep_from_candidates(
                candidates, 10_000.0, interval_discount=None, sweep=False,
                spread_band=None, band_sweep=False)


class TestSweepHelpers:
    """The split-date, split-half and excluding-top-event helpers on their own."""

    def test_split_date_is_the_median_low(self):
        recs = [{"entry": {"entry_date": d}} for d in
                (date(2026, 1, 19), date(2026, 1, 5), date(2026, 1, 12), date(2026, 1, 26))]
        # Even length: median_low takes the lower middle value (median would
        # try to average two dates and raise).
        assert backtester._split_date(recs, date(2026, 1, 1)) == date(2026, 1, 12)

    def test_split_date_falls_back_to_the_window_midpoint(self, monkeypatch):
        fixed = datetime(2026, 1, 21, 12, tzinfo=UTC)
        frozen = type("FrozenDateTime",
                      (TestRunBacktestFeasibilityPreCheck._FrozenDateTime,),
                      {"_fixed": fixed})
        monkeypatch.setattr(backtester, "datetime", frozen)
        # [Jan 1, Jan 21] is 20 days; its midpoint is Jan 11
        assert backtester._split_date([], date(2026, 1, 1)) == date(2026, 1, 11)

    def test_half_split_simulates_each_half_alone(self, monkeypatch):
        golden = TestPrepareEntriesGolden()
        entries, _ = golden._prepare(monkeypatch, True)
        first = [r for r in entries if r["entry"]["entry_date"] < date(2026, 1, 12)]
        second = [r for r in entries if r["entry"]["entry_date"] >= date(2026, 1, 12)]
        assert first and second
        split = backtester._half_split((first, second), golden._START, 10_000.0, 0.75,
                                       BACKTEST_DEFAULT_SPREAD_BAND)
        for half, ret, n in ((first, split.h1_return, split.h1_trades),
                             (second, split.h2_return, split.h2_trades)):
            alone = backtester._simulate_at_discount(half, golden._START, 10_000.0, k=0.75)
            final = float(alone.equity_df["portfolio_value"].iloc[-1])
            assert ret == pytest.approx((final - 10_000.0) / 10_000.0)
            assert n == len(alone.trades)

    @staticmethod
    def _trade(event, profit):
        t = TestEquityCurveOpensAtTheInitialBalance()._trade(10, 0.30, 0.40, "yes", 0.0)
        t.event_ticker, t.profit = event, profit
        return t

    def test_ex_top_event_picks_the_largest_summed_profit(self, monkeypatch):
        seen: list = []

        def _fake(raw_entries, *a, **k):
            seen.append(([r["entry"]["mA"]["event_ticker"] for r in raw_entries], k))
            return SimpleNamespace(equity_df=pd.DataFrame({"portfolio_value": [100.0, 90.0]}))

        monkeypatch.setattr(backtester, "_simulate_at_discount", _fake)
        point = backtester.SweepPoint(
            k=0.6, equity_df=pd.DataFrame(),
            trades=[self._trade("E1", 5.0), self._trade("E2", 4.0), self._trade("E2", 3.0),
                    self._trade("", 50.0)])   # no event ticker: never "an event"
        entries = [{"entry": {"mA": {"event_ticker": ev}}} for ev in ("E1", "E2", "E2", "")]
        top, ret = backtester._ex_top_event(point, entries, date(2026, 1, 1), 100.0,
                                            (0.3, 0.6))
        # E2's 4 + 3 beats E1's 5; the event-less 50 is ignored
        assert top == "E2"
        assert ret == pytest.approx(-0.10)
        # ONE re-simulation, of every entry not on E2, at the point's k and band
        assert seen == [(["E1", ""], {"k": 0.6, "spread_band": (0.3, 0.6),
                                      "population": "all/ex-top", "tier_floors": True})]

    def test_ex_top_event_ties_go_to_the_first_ticker_and_losses_count(self, monkeypatch):
        monkeypatch.setattr(
            backtester, "_simulate_at_discount",
            lambda *a, **k: SimpleNamespace(
                equity_df=pd.DataFrame({"portfolio_value": [100.0]})))
        point = backtester.SweepPoint(
            k=0.6, equity_df=pd.DataFrame(),
            trades=[self._trade("EB", -2.0), self._trade("EA", -2.0), self._trade("EC", -3.0)])
        # Every event lost; the least-losing is still the largest P&L
        assert backtester._ex_top_event(point, [], date(2026, 1, 1), 100.0,
                                        (0.0, 1.0))[0] == "EA"

    def test_ex_top_event_is_none_without_an_event(self, monkeypatch):
        monkeypatch.setattr(backtester, "_simulate_at_discount",
                            lambda *a, **k: pytest.fail("nothing to re-simulate"))
        point = backtester.SweepPoint(k=0.6, equity_df=pd.DataFrame(),
                                      trades=[self._trade("", 1.0)])
        assert backtester._ex_top_event(point, [], date(2026, 1, 1), 100.0,
                                        (0.0, 1.0)) is None


class TestSimulationLabelsAndStamps:
    """_simulate_at_discount's spread_band and population only label and
    stamp — they change no trade."""

    def test_the_completion_line_names_k_band_and_population(self, caplog):
        with caplog.at_level(logging.INFO):
            backtester._simulate_at_discount([], date(2026, 1, 1), 1000.0, k=0.75,
                                             spread_band=(0.3, 0.6), population="cross")
        assert [r.getMessage() for r in caplog.records
                if r.getMessage().startswith("Backtest complete")] == [
            "Backtest complete at k=0.750, band 0.3-0.6, cross: 0 trades, 0 profitable"]

    def test_a_none_band_renders_the_default_and_stamps_none(self, caplog):
        with caplog.at_level(logging.INFO):
            point = backtester._simulate_at_discount([], date(2026, 1, 1), 1000.0)
        assert "band 0-1, all:" in caplog.text
        assert point.spread_band is None and point.population == "all"

    @pytest.mark.parametrize("band", [(0, 1), (-0.0, 1.0), (0.0, 1.0)])
    def test_a_given_band_is_stamped_resolved(self, band):
        point = backtester._simulate_at_discount([], date(2026, 1, 1), 1000.0,
                                                 spread_band=band)
        assert point.spread_band == (0.0, 1.0)
        assert all(type(x) is float for x in point.spread_band)

    def test_labels_change_no_trade(self, monkeypatch):
        golden = TestPrepareEntriesGolden()
        entries, _ = golden._prepare(monkeypatch, True)
        plain = backtester._simulate_at_discount(entries, golden._START, 10_000.0)
        labelled = backtester._simulate_at_discount(
            entries, golden._START, 10_000.0, spread_band=(0.4, 0.5), population="all/H2")
        assert [astuple(t) for t in plain.trades] == [astuple(t) for t in labelled.trades]
        pd.testing.assert_frame_equal(plain.equity_df, labelled.equity_df)

    @pytest.mark.parametrize("population", ["All", "ladders", "H1", ""])
    def test_an_unknown_population_is_refused(self, population):
        with pytest.raises(ValueError, match="population"):
            backtester._simulate_at_discount([], date(2026, 1, 1), 1000.0,
                                             population=population)


class TestIntervalCalibrationBandFloor:
    """_interval_calibration(spread_min=...) labels each gap band with the
    floor its entries were actually detected under."""

    def test_the_0_7d_tier_reads_the_floor_above_it(self):
        entries = [_cal_entry(3, 0.10, 0.70, "no", "yes")]
        assert _interval_calibration(entries).buckets[0].tier == min_price_diff_for_gap(3)
        raised = _interval_calibration(entries, spread_min=0.30)
        assert [(b.label, b.tier) for b in raised.buckets] == [("0-7d", 0.30)]
        # A floor below the tier is inert
        assert _interval_calibration(entries, spread_min=0.10).buckets[0].tier == \
            min_price_diff_for_gap(3)
        # Labelling only: the measurement itself does not move
        assert raised.pooled == _interval_calibration(entries).pooled


class TestBandSweepSplitAndPopulationWiring:
    """_sweep_from_candidates over hand-built entries whose dates and event
    tickers DEPEND on the band — a shape the golden fixture cannot produce
    (its entry dates never move with the band, so a per-band split date or a
    population taken from the wrong band reads identically there). Pins the
    ONE split date — the primary band's median_low, never a per-band or
    first-band one — and every band's populations and halves coming from
    that band's own entries."""

    _D1, _D2, _D3, _D4 = (date(2026, 1, 5), date(2026, 1, 12), date(2026, 1, 19),
                          date(2026, 1, 26))

    @staticmethod
    def _rec(i, when, ladder):
        # A time-series record carrying only what the sweep reads before
        # simulating: its entry date and both legs' event tickers.
        event = f"EV{i}"
        return {"pair_type": "time_series", "canon": f"c{i}", "group_key": f"g{i}",
                "entry": {"entry_date": when,
                          "mA": {"ticker": f"A{i}", "event_ticker": event},
                          "mB": {"ticker": f"B{i}",
                                 "event_ticker": event if ladder else f"X{i}"}}}

    def test_split_date_halves_and_populations_follow_each_band(self, monkeypatch):
        D1, D2, D3, D4 = self._D1, self._D2, self._D3, self._D4
        primary = BACKTEST_DEFAULT_SPREAD_BAND
        by_band: dict = {}

        def entries_for_band(candidates, spread_band=None, *, pair_types=(), **_private):
            if pair_types == ("same_title",):
                return []
            if spread_band == primary:
                spec = [(D1, True), (D2, False), (D3, True)]            # median_low D2
            else:
                # Its own median_low is D3 — including at bands[0], (0, 0.5)
                # — and its ladder rungs differ from the primary's.
                spec = [(D1, False), (D2, True), (D3, True), (D4, False), (D4, True)]
            recs = [self._rec(i, when, ladder) for i, (when, ladder) in enumerate(spec)]
            by_band[spread_band] = recs
            return recs

        sims: list = []

        def fake_simulate(raw_entries, start_date, initial_balance, k=None,
                          spread_band=None, population="all", *, tier_floors=True):
            point = backtester.SweepPoint(
                k=TIME_SERIES_INTERVAL_PROB_DISCOUNT if k is None else k, trades=[],
                equity_df=pd.DataFrame({"portfolio_value": [initial_balance]}),
                spread_band=spread_band, population=population)
            sims.append((spread_band, population, list(raw_entries)))
            return point

        monkeypatch.setattr(backtester, "_entries_for_band", entries_for_band)
        monkeypatch.setattr(backtester, "_simulate_at_discount", fake_simulate)
        monkeypatch.setattr(backtester, "_interval_calibration", lambda *a, **k: None)
        candidates = backtester._Candidates(
            all_pairs=[], candles_by_ticker={}, label_coverage=None,
            start_date=date(2026, 1, 1), max_horizon_days=None, same_event_ladders=True)
        res = backtester._sweep_from_candidates(
            candidates, 10_000.0, interval_discount=None, sweep=False,
            spread_band=None, band_sweep=True)

        # ONE split date, the primary band's median_low
        assert res.split_date == D2
        assert sorted(by_band) == _GRID_BANDS
        seen = {(band, label): entries for band, label, entries in sims}
        for band, own in by_band.items():
            ids = [id(r) for r in own]
            ladder = [id(r) for r in own
                      if r["entry"]["mA"]["event_ticker"] == r["entry"]["mB"]["event_ticker"]]
            cross = [i for i in ids if i not in ladder]
            assert [id(r) for r in seen[(band, "all")]] == ids
            assert [id(r) for r in seen[(band, "all/H1")]] == [
                id(r) for r in own if r["entry"]["entry_date"] < D2]
            assert [id(r) for r in seen[(band, "all/H2")]] == [
                id(r) for r in own if r["entry"]["entry_date"] >= D2]
            assert [id(r) for r in seen[(band, "ladder")]] == ladder
            assert [id(r) for r in seen[(band, "cross")]] == cross
        # Not vacuous: a non-primary band's own median_low (D3) would have
        # split its entries differently from the one split date (D2).
        other = by_band[(0.0, 0.5)]
        assert statistics.median_low(r["entry"]["entry_date"] for r in other) == D3
        assert ([r for r in other if r["entry"]["entry_date"] < D3]
                != [r for r in other if r["entry"]["entry_date"] < D2])


# ─── PB7: the no-band pre-pass, the time-series population, split-half ──────

@pytest.mark.usefixtures("golden_band_sweep")
class TestBandSweepPhaseOneSubset:
    """Phase 1 scans every time-series pair ONCE, at the no-band band (0, 1),
    and every other band rescans only the pairs that produced an entry there.
    Safe because a band only tightens _find_entry's per-Monday tests (the
    floor only rises, the sum ceiling 1 - threshold only falls, the spread
    ceiling only drops), so every band's accepted Mondays are a subset of the
    no-band band's — pinned here as "every band's entries equal a full scan's",
    which is the claim the proof exists to support."""

    @staticmethod
    def _fresh(monkeypatch):
        golden = TestPrepareEntriesGolden()
        golden._patch(monkeypatch)
        return backtester._prepare_candidates(
            MagicMock(), MagicMock(), golden._START, True, None, same_event_ladders=True)

    def test_every_band_s_entries_equal_a_full_scan(self, golden_band_sweep, monkeypatch):
        rows = TestPrepareEntriesGolden._rows
        swept = {band: out for band, types, out in golden_band_sweep.calls["entries"]
                 if types == ("time_series",)}
        assert sorted(swept) == _GRID_BANDS
        distinct = set()
        for band, out in swept.items():
            full = backtester._entries_for_band(self._fresh(monkeypatch), band,
                                                pair_types=("time_series",))
            assert rows(out) == rows(full), band
            distinct.add(tuple(rows(full)))
        # Not vacuous: the bands really do select different entry lists
        assert len(distinct) > 1

    def test_the_no_band_band_is_the_one_full_scan(self, golden_band_sweep):
        scanned = golden_band_sweep.calls["scanned"]
        ts = [(band, pairs, all_pairs) for band, types, pairs, all_pairs in scanned
              if types == ("time_series",)]
        # The pre-pass comes FIRST, is the (0, 1) band, and scans all_pairs
        assert ts[0][0] == (0.0, 1.0) and ts[0][1] is None
        # ... and is never run twice: every other band is handed the subset
        assert [band for band, pairs, _ in ts if pairs is None] == [(0.0, 1.0)]
        rescans = [pairs for band, pairs, _ in ts[1:]]
        assert len(rescans) == 35
        subset = rescans[0]
        assert all(pairs is subset for pairs in rescans)
        # FA/FB (a pricier earlier contract on both Mondays) never enters, so
        # the rescan is the four time-series pairs that did, in scan order
        assert [(item[0][0]["ticker"], item[0][1]["ticker"]) for item in subset] == [
            ("EA", "EB"), ("RUNG-EARLY", "RUNG-LATE"), ("TA", "TB"), ("WA", "WB")]
        # all_pairs itself is never mutated: still the six candidates, and
        # the subset is a separate list
        all_pairs = ts[0][2]
        assert subset is not all_pairs
        assert [(item[0][0]["ticker"], item[1]) for item in all_pairs] == [
            ("EA", "time_series"), ("RUNG-EARLY", "time_series"), ("FA", "time_series"),
            ("TA", "time_series"), ("WA", "time_series"), ("SA", "same_title")]

    def test_the_pre_pass_is_announced_with_its_count(self, golden_band_sweep):
        messages = golden_band_sweep.messages
        assert "No-band pre-pass: scanning all 5 time-series pairs at 0-1" in messages
        assert ("No-band pre-pass: 4 of 5 time-series pairs produced an entry; every "
                "other band rescans only those") in messages
        # Not a "Spread band i/N" line: the announcement count stays 36
        assert not any(m.startswith("Spread band ") and "pre-pass" in m for m in messages)

    def test_a_single_band_run_scans_all_pairs_once(self, monkeypatch):
        candidates = self._fresh(monkeypatch)
        seen: list = []
        real = backtester._entries_for_band

        def spy(c, spread_band=None, *, pair_types=("time_series", "same_title"), **private):
            seen.append((spread_band, pair_types, private))
            return real(c, spread_band, pair_types=pair_types, **private)

        monkeypatch.setattr(backtester, "_entries_for_band", spy)
        backtester._sweep_from_candidates(candidates, 10_000.0, interval_discount=None,
                                          sweep=False, spread_band=(0.3, 0.6),
                                          band_sweep=False)
        # No pre-pass and no subset: one same-title pass and one full
        # time-series pass at the primary band
        assert seen == [((0.3, 0.6), ("same_title",), {}),
                        ((0.3, 0.6), ("time_series",), {})]

    def test_pairs_narrows_the_scan_and_keeps_its_order(self, monkeypatch):
        candidates = self._fresh(monkeypatch)
        rows = TestPrepareEntriesGolden._rows
        full = backtester._entries_for_band(candidates, pair_types=("time_series",))
        # Hand the scan a reordered subset: it scans exactly those items, in
        # the order given, and all_pairs is untouched
        before = list(candidates.all_pairs)
        subset = [candidates.all_pairs[3], candidates.all_pairs[0]]
        narrowed = backtester._entries_for_band(candidates, pair_types=("time_series",),
                                                _pairs=subset)
        assert [r[-2] for r in rows(narrowed)] == ["TA", "EA"]
        assert set(rows(narrowed)) <= set(rows(full))
        assert candidates.all_pairs == before

    # ── A fixture of its own, for the two shapes the golden one lacks ───────
    # The golden fixture's entered pairs are all listed in entry order and it
    # has no pair a band could admit that the no-band rule refuses, so a
    # pre-pass that matched entered pairs on an ORDERED ticker pair, or a band
    # whose floor LOOSENED the tier, would both keep every golden band equal
    # to its full scan. These two pairs make each of those fail:
    #   * ICE: a cross-event pair listed LATER-leg first. Both legs close on
    #     2026-02-01, so _extract_pairs' close-DATE sort keeps the listed order
    #     while _find_entry's close-DATETIME order swaps the legs — the entry
    #     names (ICE-EARLY, ICE-LATE), the pair item (ICE-LATE, ICE-EARLY). A
    #     0-day gap (the 0.15 tier) at a 0.55 spread, pA + nB = 0.45: it enters
    #     at every grid band whose ceiling is at least 0.60 (30 of 36).
    #   * FOG: a 19-day gap (the 0.30 tier) at a 0.25 spread, pA + nB = 0.70 —
    #     refused at every band by the tier alone, and admitted at the 0.20 and
    #     0.25 floors by any rule that let a band floor fall below the tier.
    @staticmethod
    def _own_markets() -> list[dict]:
        def mk(ticker, event_ticker, event_title, title, close):
            return {"ticker": ticker, "event_ticker": event_ticker,
                    "event_title": event_title, "title": title, "subtitle": "",
                    "result": "", "open_time": "2026-01-01T00:00:00+00:00",
                    "close_time": close, "settlement_ts": close}
        return [
            mk("ICE-LATE", "ICEB-1", "ICE", "Ice forms by February 10, 2026",
               "2026-02-01T18:00:00+00:00"),
            mk("ICE-EARLY", "ICEA-1", "ICE", "Ice forms by February 1, 2026",
               "2026-02-01T06:00:00+00:00"),
            mk("FOG-A", "FOGA-1", "FOG", "Fog lifts by February 1, 2026",
               "2026-02-01T00:00:00+00:00"),
            mk("FOG-B", "FOGB-1", "FOG", "Fog lifts by February 20, 2026",
               "2026-02-20T00:00:00+00:00"),
        ]

    _OWN_CANDLES = {
        "ICE-EARLY": [_candle(_MONDAY_TS, 0.20, 0.80)],
        "ICE-LATE": [_candle(_MONDAY_TS, 0.75, 0.25)],
        "FOG-A": [_candle(_MONDAY_TS, 0.20, 0.80)],
        "FOG-B": [_candle(_MONDAY_TS, 0.45, 0.50)],
    }

    def _own_candidates(self, monkeypatch):
        markets = self._own_markets()
        monkeypatch.setattr(backtester, "fetch_all_settled_markets",
                            lambda *a, **k: markets)
        monkeypatch.setattr(backtester, "fetch_candlesticks",
                            lambda _c, ticker, *a, **k: self._OWN_CANDLES[ticker])
        return backtester._prepare_candidates(
            MagicMock(), MagicMock(), date(2026, 1, 1), True, None, same_event_ladders=True)

    def _own_band_sweep(self, monkeypatch):
        """Phase 1 of a real band sweep over the fixture (one k, simulations
        stubbed). Returns (every time-series pass as (band, _pairs, entries),
        the real _entries_for_band) — the first pass is the pre-pass."""
        candidates = self._own_candidates(monkeypatch)
        real = backtester._entries_for_band
        passes: list = []

        def spy(c, spread_band=None, *, pair_types=("time_series", "same_title"), **private):
            out = real(c, spread_band, pair_types=pair_types, **private)
            if pair_types == ("time_series",):
                passes.append((spread_band, private.get("_pairs"), out))
            return out

        def fake_simulate(raw_entries, start_date, initial_balance, k=None,
                          spread_band=None, population="all", *, tier_floors=True):
            return backtester.SweepPoint(
                k=TIME_SERIES_INTERVAL_PROB_DISCOUNT if k is None else k, trades=[],
                equity_df=pd.DataFrame({"portfolio_value": [initial_balance]}),
                spread_band=spread_band, population=population)

        monkeypatch.setattr(backtester, "_entries_for_band", spy)
        monkeypatch.setattr(backtester, "_simulate_at_discount", fake_simulate)
        monkeypatch.setattr(backtester, "_interval_calibration", lambda *a, **k: None)
        backtester._sweep_from_candidates(candidates, 10_000.0, interval_discount=None,
                                          sweep=False, spread_band=None, band_sweep=True)
        return passes, real

    def _assert_every_band_equals_a_full_scan(self, monkeypatch, passes, real) -> dict:
        rows = TestPrepareEntriesGolden._rows
        by_band = {band: out for band, _pairs, out in passes}
        assert sorted(by_band) == _GRID_BANDS
        for band, out in by_band.items():
            full = real(self._own_candidates(monkeypatch), band, pair_types=("time_series",))
            assert rows(out) == rows(full), band
        return by_band

    def test_a_pair_entered_with_swapped_legs_is_still_rescanned(self, monkeypatch):
        passes, real = self._own_band_sweep(monkeypatch)
        # The swap path is real: the pair item lists ICE later-leg first ...
        items = [(i[0][0]["ticker"], i[0][1]["ticker"])
                 for i in self._own_candidates(monkeypatch).all_pairs]
        assert ("ICE-LATE", "ICE-EARLY") in items
        # ... while the no-band entry names the legs the other way round
        pre_band, pre_pairs, pre_out = passes[0]
        assert pre_band == (0.0, 1.0) and pre_pairs is None
        assert [(r["entry"]["mA"]["ticker"], r["entry"]["mB"]["ticker"])
                for r in pre_out] == [("ICE-EARLY", "ICE-LATE")]
        # The rescan still carries the pair (an ORDERED match would drop it)
        rescans = [pairs for _band, pairs, _out in passes[1:]]
        assert len(rescans) == 35
        assert all([(i[0][0]["ticker"], i[0][1]["ticker"]) for i in pairs]
                   == [("ICE-LATE", "ICE-EARLY")] for pairs in rescans)
        by_band = self._assert_every_band_equals_a_full_scan(monkeypatch, passes, real)
        # Not vacuous: it enters at the 30 bands whose ceiling is >= 0.60
        entered = sorted(band for band, out in by_band.items() if out)
        assert entered == [b for b in _GRID_BANDS if b[1] >= 0.60]

    def test_a_pair_only_a_loosened_floor_could_admit_enters_nowhere(self, monkeypatch):
        passes, real = self._own_band_sweep(monkeypatch)
        self._assert_every_band_equals_a_full_scan(monkeypatch, passes, real)
        assert not any(r["entry"]["mA"]["ticker"].startswith("FOG")
                       for _band, _pairs, out in passes for r in out)
        # Not vacuous: a band floor that REPLACED the 0.30 tier instead of
        # being layered on it (max(tier, floor)) would admit FOG at the 0.20
        # and 0.25 floors — exactly what the no-band pre-pass would then miss
        tier_rule = backtester.min_price_diff_for_gap
        with monkeypatch.context() as m:
            m.setattr(backtester, "min_price_diff_for_gap",
                      lambda gap, spread_min=None, **_kw: spread_min or tier_rule(gap))
            loosened = real(self._own_candidates(m), (0.20, 1.0),
                            pair_types=("time_series",))
        assert "FOG-A" in [r["entry"]["mA"]["ticker"] for r in loosened]

    def test_the_tier_off_rescan_comes_from_its_own_pre_pass(self, monkeypatch):
        # FOG is exactly the pair a tier-floors-off family must not lose: the
        # 0.30 tier refuses its 0.25 spread at every band, so the TIER-ON
        # pre-pass never enters it, while with the tiers off it clears every
        # binding floor (0, 0.20, 0.25) and every binding band's sum ceiling
        # (pA + nB = 0.70 <= 1 - 0.25). A tier-off rescan built from the
        # tier-on pre-pass would drop it at every tier-off band but (0, 1) —
        # the golden fixture cannot see that, since the pairs it enters are
        # the same with the tiers on or off.
        candidates = self._own_candidates(monkeypatch)
        real = backtester._entries_for_band
        passes: list = []

        def spy(c, spread_band=None, *, pair_types=("time_series", "same_title"), **private):
            out = real(c, spread_band, pair_types=pair_types, **private)
            if pair_types == ("time_series",):
                passes.append((spread_band, private.get("tier_floors", True),
                               private.get("_pairs"), out))
            return out

        def fake_simulate(raw_entries, start_date, initial_balance, k=None,
                          spread_band=None, population="all", *, tier_floors=True):
            return backtester.SweepPoint(
                k=TIME_SERIES_INTERVAL_PROB_DISCOUNT if k is None else k, trades=[],
                equity_df=pd.DataFrame({"portfolio_value": [initial_balance]}),
                spread_band=spread_band, population=population,
                tier_floors=tier_floors is not False)

        monkeypatch.setattr(backtester, "_entries_for_band", spy)
        monkeypatch.setattr(backtester, "_simulate_at_discount", fake_simulate)
        monkeypatch.setattr(backtester, "_interval_calibration", lambda *a, **k: None)
        backtester._sweep_from_candidates(candidates, 10_000.0, interval_discount=None,
                                          sweep=False, spread_band=None, band_sweep=True,
                                          tier_off_sweep=True)

        def tickers(pairs):
            return [(item[0][0]["ticker"], item[0][1]["ticker"]) for item in pairs]

        on = [p for p in passes if p[1] is True]
        off = [p for p in passes if p[1] is False]
        assert len(on) == 36 and len(off) == 18
        # The tier-on sweep: its one full scan admits ICE alone, and every
        # tier-on rescan carries just that pair
        assert on[0][0] == (0.0, 1.0) and on[0][2] is None
        assert all(tickers(pairs) == [("ICE-LATE", "ICE-EARLY")]
                   for _band, _tf, pairs, _out in on[1:])
        # The tier-off family: ONE full scan of its own, at (0, 1), and its 17
        # other bands rescan what entered THERE — ICE and FOG
        assert off[0][0] == (0.0, 1.0) and off[0][2] is None
        assert [band for band, _tf, pairs, _out in off if pairs is None] == [(0.0, 1.0)]
        assert all(tickers(pairs) == [("ICE-LATE", "ICE-EARLY"), ("FOG-A", "FOG-B")]
                   for _band, _tf, pairs, _out in off[1:])
        # Every tier-off pass is a full tier-off scan's entries
        rows = TestPrepareEntriesGolden._rows
        for band, _tf, _pairs, out in off:
            full = real(self._own_candidates(monkeypatch), band, pair_types=("time_series",),
                        tier_floors=False)
            assert rows(out) == rows(full), band

        # Not vacuous: FOG-A enters at every tier-off band, and at no tier-on one
        def fog_bands(group):
            return sorted(band for band, _tf, _pairs, out in group
                          if any(r["entry"]["mA"]["ticker"] == "FOG-A" for r in out))

        assert fog_bands(off) == _TIER_BOUND_BANDS
        assert fog_bands(on) == []


@pytest.mark.usefixtures("golden_band_sweep")
class TestTimeSeriesPopulation:
    """The standalone "time_series" population — ladders and cross-event
    together, same-title excluded — which the dashboard's heatmap and
    fragility banner read, so a same-title result (band- and k-independent)
    cannot dilute them. It carries its own split-half and ex-top checks."""

    def test_it_is_the_band_s_time_series_entries_simulated_alone(self, golden_band_sweep):
        calls = golden_band_sweep.calls["simulate"]
        by_point = {id(c["point"]): c for c in calls}
        by_key = {(c["band"], c["point"].k, c["population"]): c for c in calls}
        ts_points = [p for p in golden_band_sweep.result.scenarios
                     if p.population == "time_series"]
        assert len(ts_points) == 468
        for p in ts_points:
            own = by_point[id(p)]["entries"]
            everything = by_key[(p.spread_band, p.k, "all")]["entries"]
            # Exactly the time-series entries of the same band's "all" run,
            # in order — the same-title entry is the one left out
            assert [id(r) for r in own] == [
                id(r) for r in everything if r["pair_type"] == "time_series"]
            assert len(everything) == len(own) + 1
            assert all(t.pair_type == "time_series" for t in p.trades)
        # A standalone run from the initial balance, not a slice of "all"
        p = next(p for p in ts_points if p.spread_band == BACKTEST_DEFAULT_SPREAD_BAND
                 and p.k == TIME_SERIES_INTERVAL_PROB_DISCOUNT)
        assert p.trades and p.trades[0].balance_at_entry == pytest.approx(10_000.0)

    def test_its_checks_read_its_own_entries(self, golden_band_sweep):
        res = golden_band_sweep.result
        calls = golden_band_sweep.calls["simulate"]
        by_point = {id(c["point"]): c for c in calls}
        by_key = {(c["band"], c["point"].k, c["population"]): c for c in calls}

        def ret(point):
            return (float(point.equity_df["portfolio_value"].iloc[-1]) - 10_000.0) / 10_000.0

        for p in (s for s in res.scenarios if s.population == "time_series"):
            own = by_point[id(p)]["entries"]
            h1 = by_key[(p.spread_band, p.k, "time_series/H1")]
            h2 = by_key[(p.spread_band, p.k, "time_series/H2")]
            assert [id(r) for r in h1["entries"]] == [
                id(r) for r in own if r["entry"]["entry_date"] < res.split_date]
            assert [id(r) for r in h2["entries"]] == [
                id(r) for r in own if r["entry"]["entry_date"] >= res.split_date]
            assert p.halves == backtester.HalfSplit(
                h1_return=ret(h1["point"]), h2_return=ret(h2["point"]),
                h1_trades=len(h1["point"].trades), h2_trades=len(h2["point"].trades),
                h1_entries=len(h1["entries"]), h2_entries=len(h2["entries"]))
            if p.ex_top_event is not None:
                top = p.ex_top_event[0]
                ex = by_key[(p.spread_band, p.k, "time_series/ex-top")]
                assert [id(r) for r in ex["entries"]] == [
                    id(r) for r in own if r["entry"]["mA"]["event_ticker"] != top]
                assert p.ex_top_event == (top, ret(ex["point"]))

    def test_the_golden_split_leaves_h1_empty_and_says_so(self, golden_band_sweep):
        # Four of the primary band's five entries (three of its four
        # time-series ones) enter on Monday 1, so the median_low split date IS
        # Monday 1 and H1 (strictly before it) is empty in every cell.
        res = golden_band_sweep.result
        assert res.split_date == date(2026, 1, 5)
        assert (res.primary.halves.h1_entries, res.primary.halves.h2_entries) == (0, 5)
        ts = next(p for p in res.scenarios if p.population == "time_series"
                  and p.spread_band == res.primary.spread_band and p.k == res.primary.k)
        assert (ts.halves.h1_entries, ts.halves.h2_entries) == (0, 4)
        warnings = [m for m in golden_band_sweep.messages
                    if m.startswith("Split-half check: split date")]
        assert warnings == [
            "Split-half check: split date 2026-01-05 leaves H1 empty; the split-half "
            "check is not measurable for this window (primary band 0-1: 0 time-series "
            "entries before it, 4 on or after it)"]

    @staticmethod
    def _rec(i, when, pair_type="time_series"):
        return {"pair_type": pair_type, "canon": f"c{i}", "group_key": f"g{i}",
                "entry": {"entry_date": when,
                          "mA": {"ticker": f"A{i}", "event_ticker": f"EV{i}"},
                          "mB": {"ticker": f"B{i}", "event_ticker": f"X{i}"}}}

    def _sweep(self, monkeypatch, ts_dates, st_dates):
        ts = [self._rec(i, d) for i, d in enumerate(ts_dates)]
        st = [self._rec(100 + i, d, "same_title") for i, d in enumerate(st_dates)]
        sims: list = []

        def entries_for_band(candidates, spread_band=None, *, pair_types=(), **_private):
            return list(st) if pair_types == ("same_title",) else list(ts)

        def fake_simulate(raw_entries, start_date, initial_balance, k=None,
                          spread_band=None, population="all", *, tier_floors=True):
            sims.append((population, list(raw_entries)))
            return backtester.SweepPoint(
                k=TIME_SERIES_INTERVAL_PROB_DISCOUNT if k is None else k, trades=[],
                equity_df=pd.DataFrame({"portfolio_value": [initial_balance]}),
                spread_band=spread_band, population=population)

        monkeypatch.setattr(backtester, "_entries_for_band", entries_for_band)
        monkeypatch.setattr(backtester, "_simulate_at_discount", fake_simulate)
        monkeypatch.setattr(backtester, "_interval_calibration", lambda *a, **k: None)
        candidates = backtester._Candidates(
            all_pairs=[], candles_by_ticker={}, label_coverage=None,
            start_date=date(2026, 1, 1), max_horizon_days=None, same_event_ladders=True)
        res = backtester._sweep_from_candidates(
            candidates, 10_000.0, interval_discount=None, sweep=False,
            spread_band=None, band_sweep=True)
        return res, sims

    def test_same_title_entries_cannot_move_the_split(self, monkeypatch):
        D1, D2, D3, D4 = (date(2026, 1, 5), date(2026, 1, 12), date(2026, 1, 19),
                          date(2026, 1, 26))
        # All six dates' median_low is D1; the time-series three's is D3.
        res, sims = self._sweep(monkeypatch, [D2, D3, D4], [D1, D1, D1])
        assert statistics.median_low([D2, D3, D4, D1, D1, D1]) == D1
        assert res.split_date == D3
        ts_h1 = [entries for pop, entries in sims if pop == "time_series/H1"]
        assert ts_h1 and all([r["entry"]["entry_date"] for r in e] == [D2] for e in ts_h1)

    def test_no_time_series_entry_means_no_time_series_point(self, monkeypatch, caplog):
        with caplog.at_level(logging.WARNING):
            res, sims = self._sweep(monkeypatch, [], [date(2026, 1, 5)])
        assert {p.population for p in res.scenarios} == {"all"}
        assert not any(pop.startswith("time_series") for pop, _ in sims)
        # With no time-series entry at the primary band both halves are empty
        assert ("leaves H1 and H2 empty; the split-half check is not measurable"
                in caplog.text)

    def test_no_warning_when_both_halves_have_entries(self, monkeypatch, caplog):
        with caplog.at_level(logging.WARNING):
            res, _ = self._sweep(monkeypatch, [date(2026, 1, 5), date(2026, 1, 12),
                                               date(2026, 1, 19)], [])
        assert res.split_date == date(2026, 1, 12)
        assert "not measurable" not in caplog.text
        halves = [p.halves for p in res.scenarios if p.population == "time_series"]
        assert halves and all((h.h1_entries, h.h2_entries) == (1, 2) for h in halves)


class TestExactLabels:
    """_exact_label / _band_label: the short form for every grid value, an
    exact one whenever the short form would merge two different values."""

    @pytest.mark.parametrize("band,label", [
        ((0.0, 1.0), "0-1"),
        ((0.3, 0.6), "0.3-0.6"),
        ((0.35, 0.9), "0.35-0.9"),
        ((0.3000001, 0.6), "0.3000001-0.6"),
        ((0.1 + 0.2, 0.6), "0.30000000000000004-0.6"),
    ])
    def test_band_labels(self, band, label):
        assert backtester._band_label(band) == label

    @pytest.mark.parametrize("k,label", [
        (0.75, "0.750"), (0.62, "0.620"), (1.0, "1.000"), (0.7500001, "0.7500001"),
    ])
    def test_k_labels(self, k, label):
        assert backtester._exact_label(k, ".3f") == label

    def test_every_grid_value_keeps_its_short_form(self):
        for x in SPREAD_BAND_SWEEP_FLOORS + SPREAD_BAND_SWEEP_CEILINGS:
            assert backtester._exact_label(float(x), "g") == format(x, "g")
        for k in INTERVAL_DISCOUNT_SWEEP:
            assert backtester._exact_label(k, ".3f") == format(k, ".3f")


# ─── The tier-floors-off family (backtest only) ─────────────────────────────

# The grid bands a deadline-gap tier floor binds at: every band whose floor
# sits below a tier — floors 0, 0.20 and 0.25, written out here rather than
# derived from the helper under test.
_TIER_BOUND_BANDS = [band for band in _GRID_BANDS if band[0] in (0.0, 0.20, 0.25)]


class TestTierFloorsOff:
    """The backtest-only tier_floors switch under the tier-off family:
    _tier_floors_bind (which bands a tier binds at), _find_entry with the
    deadline-gap tier not applied (the band floor alone gates pB - pA and sets
    the leg-price-sum ceiling, 1 - floor), the completion line and stamp of a
    tier-off simulation, and a tier-off calibration's labels.

    The literals are hand-derived from TestPrepareEntriesGolden's candles,
    never from the code under test. Its ladder (RUNG-EARLY/RUNG-LATE, a
    19-day STATED gap, so the 0.30 tier) quotes pA 0.20 / pB 0.45 / nB 0.55 on
    Monday 1 — a 0.25 spread at pA + nB = 0.75, below the tier — and pA 0.20 /
    pB 0.60 / nB 0.40 on Monday 2 (0.40 at 0.60), so with the tiers it waits
    for Monday 2 at every floor, while with them off a floor of 0, 0.20 or
    0.25 admits Monday 1 (spread >= floor, pA + nB <= 1 - floor)."""

    _START = TestPrepareEntriesGolden._START
    _MONDAY_1 = (date(2026, 1, 5), 0.2, 0.45, 0.8, 0.55, 19, "RUNG-EARLY", "RUNG-LATE")
    _MONDAY_2 = (date(2026, 1, 12), 0.2, 0.6, 0.8, 0.4, 19, "RUNG-EARLY", "RUNG-LATE")

    @staticmethod
    def _market(ticker):
        return next(m for m in TestPrepareEntriesGolden._markets() if m["ticker"] == ticker)

    def _entry(self, first, second, pair_type, band, tier_floors):
        candles = TestPrepareEntriesGolden._CANDLES
        return _find_entry(candles[first], candles[second], self._market(first),
                           self._market(second), pair_type, self._START,
                           same_event_ladders=True, spread_band=band,
                           tier_floors=tier_floors)

    @staticmethod
    def _row(entry):
        return (entry["entry_date"], entry["pA"], entry["pB"], entry["nA"], entry["nB"],
                entry["gap_days"], entry["mA"]["ticker"], entry["mB"]["ticker"])

    def test_the_tiers_bind_exactly_below_the_long_tier(self):
        assert {lo: backtester._tier_floors_bind((lo, 1.0)) for lo in SPREAD_BAND_SWEEP_FLOORS} == {
            0.0: True, 0.2: True, 0.25: True, 0.3: False, 0.35: False, 0.4: False}
        # The boundary is the 0.30 tier itself, to the float
        assert backtester._tier_floors_bind((0.2999999, 0.6)) is True
        assert backtester._tier_floors_bind((0.3000001, 0.6)) is False
        # The ceiling never enters into it: 18 of the 36 grid bands bind
        assert [b for b in _GRID_BANDS if backtester._tier_floors_bind(b)] == _TIER_BOUND_BANDS
        assert len(_TIER_BOUND_BANDS) == 18

    def test_the_ladder_enters_a_week_earlier_with_the_tiers_off(self):
        # Listed later-rung first, as in the golden fixture: the stated
        # deadlines order it, either way.
        for lo in SPREAD_BAND_SWEEP_FLOORS:
            for hi in SPREAD_BAND_SWEEP_CEILINGS:
                on = self._entry("RUNG-LATE", "RUNG-EARLY", "time_series", (lo, hi), True)
                off = self._entry("RUNG-LATE", "RUNG-EARLY", "time_series", (lo, hi), False)
                assert self._row(on) == self._MONDAY_2, (lo, hi)
                expected = self._MONDAY_1 if lo in (0.0, 0.20, 0.25) else self._MONDAY_2
                assert self._row(off) == expected, (lo, hi)

    def test_the_sum_ceiling_follows_the_floor(self):
        # A cross-event pair 20 days apart (the 0.30 tier) at a 0.25 spread
        mA = {"ticker": "EARLY", "event_ticker": "E1", "close_time": "2026-02-01T00:00:00+00:00"}
        mB = {"ticker": "LATE", "event_ticker": "E2", "close_time": "2026-02-21T00:00:00+00:00"}

        def entry(nB, band, tier_floors):
            return _find_entry([_candle(_MONDAY_TS, 0.30, 0.70)],
                               [_candle(_MONDAY_TS, 0.55, nB)], mA, mB, "time_series",
                               self._START, spread_band=band, tier_floors=tier_floors)

        assert 0.30 + 0.55 == pytest.approx(0.85)
        # Tiers off at floor 0.20: the 0.25 spread clears the floor, but
        # pA + nB = 0.85 is above its sum ceiling, 1 - 0.20 = 0.80 ...
        assert entry(0.55, (0.20, 1.0), False) is None
        # ... and that ceiling is the only reason: at pA + nB = 0.75 it enters
        kept = entry(0.45, (0.20, 1.0), False)
        assert kept is not None and (kept["entry_date"], kept["gap_days"]) == (date(2026, 1, 5), 20)
        # Tiers off at floor 0: a sum ceiling of 1, so 0.85 enters
        zero = entry(0.55, (0.0, 1.0), False)
        assert zero is not None and (zero["entry_date"], zero["gap_days"]) == (date(2026, 1, 5), 20)
        # With the tiers on, the 0.30 tier refuses a 0.25 spread at either floor
        assert entry(0.55, (0.0, 1.0), True) is None
        assert entry(0.45, (0.20, 1.0), True) is None

    def test_a_zero_spread_is_refused_with_the_tiers_off(self):
        # At a floor of 0 with the tiers off the threshold is 0.0, so the gap
        # test alone would admit pB == pA: a pair with no in-between mass,
        # which time_series_profit_prob models as riskless (p = 1, Kelly at
        # the cap). The later leg's candle is CROSSED (pB + nB = 0.80 < 1), so
        # the pair clears every other price test — the live quotes, the sum
        # ceiling of 1 and the fee check — and only the refusal of a spread
        # that is not strictly positive stands between it and an entry.
        mA = {"ticker": "EARLY", "event_ticker": "E1", "close_time": "2026-02-01T00:00:00+00:00"}
        mB = {"ticker": "LATE", "event_ticker": "E2", "close_time": "2026-02-10T00:00:00+00:00"}

        def entry(pB, tier_floors=False):
            return _find_entry([_candle(_MONDAY_TS, 0.30, 0.70)],
                               [_candle(_MONDAY_TS, pB, 0.50)], mA, mB, "time_series",
                               self._START, spread_band=(0.0, 1.0), tier_floors=tier_floors)

        assert 1.0 - 0.30 - 0.50 > fee_per_pair_approx(0.30, 0.50)
        assert entry(0.30) is None
        # The same candles two cents apart enter, on Monday 1
        kept = entry(0.32)
        assert kept is not None
        assert (kept["entry_date"], kept["pA"], kept["pB"], kept["nB"], kept["gap_days"]) == (
            date(2026, 1, 5), 0.30, 0.32, 0.50, 9)
        # Inert with the tiers on: the 0.15 tier (a 9-day gap) refuses both
        assert entry(0.30, tier_floors=True) is None
        assert entry(0.32, tier_floors=True) is None

    def test_a_pricier_earlier_contract_never_enters_with_the_tiers_off(self):
        # FA/FB: pB - pA = -0.10 on both Mondays. Even at a floor of 0 the
        # direction test still stands: pB must exceed pA.
        for lo in SPREAD_BAND_SWEEP_FLOORS:
            for hi in SPREAD_BAND_SWEEP_CEILINGS:
                assert self._entry("FA", "FB", "time_series", (lo, hi), False) is None

    def test_same_title_enters_identically_either_way(self):
        default = self._entry("SA", "SB", "same_title", None, True)
        assert default is not None
        for band in _GRID_BANDS:
            for tier_floors in (True, False):
                assert self._entry("SA", "SB", "same_title", band, tier_floors) == default

    def test_a_tier_off_simulation_is_named_and_stamped_as_one(self, caplog, monkeypatch):
        with caplog.at_level(logging.INFO):
            off = backtester._simulate_at_discount([], date(2026, 1, 1), 1000.0, k=0.75,
                                                   spread_band=(0.2, 0.6),
                                                   population="time_series/H1",
                                                   tier_floors=False)
            on = backtester._simulate_at_discount([], date(2026, 1, 1), 1000.0, k=0.75,
                                                  spread_band=(0.2, 0.6),
                                                  population="time_series/H1")
        lines = [r.getMessage() for r in caplog.records
                 if r.getMessage().startswith("Backtest complete")]
        assert lines == [
            "Backtest complete at k=0.750, band 0.2-0.6 with the tier floors off, "
            "time_series/H1: 0 trades, 0 profitable",
            "Backtest complete at k=0.750, band 0.2-0.6, time_series/H1: 0 trades, 0 profitable",
        ]
        # The suffix holds no colon, and the population is still the last
        # ", "-separated field of the prefix
        assert [p.rsplit(", ", 1)[1] for p in _completion_prefixes(lines)] == [
            "time_series/H1"] * 2
        assert off.tier_floors is False and on.tier_floors is True
        # Only an explicit False marks a tier-off point
        assert backtester._simulate_at_discount([], date(2026, 1, 1), 1000.0,
                                                tier_floors=None).tier_floors is True
        # A label: the trades are those of the tier-on simulation of the same entries
        golden = TestPrepareEntriesGolden()
        entries, _ = golden._prepare(monkeypatch, True)
        plain = backtester._simulate_at_discount(entries, golden._START, 10_000.0)
        labelled = backtester._simulate_at_discount(entries, golden._START, 10_000.0,
                                                    tier_floors=False)
        assert [astuple(t) for t in plain.trades] == [astuple(t) for t in labelled.trades]
        pd.testing.assert_frame_equal(plain.equity_df, labelled.equity_df)

    def test_a_tier_off_calibration_is_labelled_with_the_floor_alone(self):
        entries = [_cal_entry(3, 0.10, 0.70, "no", "yes"),
                   _cal_entry(10, 0.10, 0.40, "no", "no"),
                   _cal_entry(20, 0.10, 0.60, "yes", "yes")]
        on = _interval_calibration(entries, spread_min=0.2)
        assert [(b.label, b.tier) for b in on.buckets] == [
            ("0-7d", 0.2), ("8-15d", 0.2), ("16-30d", 0.30)]
        off = _interval_calibration(entries, spread_min=0.2, tier_floors=False)
        assert [(b.label, b.tier) for b in off.buckets] == [
            ("0-7d", 0.2), ("8-15d", 0.2), ("16-30d", 0.2)]
        zero = _interval_calibration(entries, spread_min=0.0, tier_floors=False)
        assert [b.tier for b in zero.buckets] == [0.0, 0.0, 0.0]
        # Labels only: the measurement itself does not move
        assert off.pooled == on.pooled == zero.pooled
        assert off.observations == on.observations == zero.observations

    def test_a_floor_zero_tier_off_bucket_prints_no_floor(self, caplog):
        # The documented reading of the tier <= 0 sentinel: a tier-off bucket
        # at floor 0 cleared no floor at all, so it prints "-" like the pooled
        # row rather than a 0.00 threshold
        zero = _interval_calibration([_cal_entry(3, 0.10, 0.70, "no", "yes")],
                                     spread_min=0.0, tier_floors=False)
        with caplog.at_level(logging.INFO):
            _log_interval_calibration(zero)
        msgs = [r.getMessage() for r in caplog.records]
        assert msgs[2].split()[:2] == ["0-7d", "-"]
        assert msgs[3].split()[:2] == ["POOLED", "-"]


@pytest.fixture(scope="class")
def golden_tier_off_sweep():
    """ONE band sweep WITH the tier-floors-off family (36 bands x 13 k, ladders
    on) over the TestPrepareEntriesGolden fixture, through spies that tolerate
    any keyword (tier_floors included), plus the same run with
    tier_off_sweep=False — the oracle for the tier-on payload, which the
    family must leave exactly as it was. Class-scoped, like golden_band_sweep."""
    mp = pytest.MonkeyPatch()
    handler = _LogCapture()
    root = logging.getLogger()
    old_level = root.level
    try:
        golden = TestPrepareEntriesGolden()
        golden._patch(mp)
        calls: dict = {"entries": [], "simulate": [], "candidates": [], "populations": []}
        real_prepare = backtester._prepare_candidates
        real_entries = backtester._entries_for_band
        real_simulate = backtester._simulate_at_discount
        real_populations = backtester._band_populations

        def prepare_spy(*a, **kw):
            c = real_prepare(*a, **kw)
            calls["candidates"].append(c)
            return c

        def entries_spy(candidates, spread_band=None, **kw):
            out = real_entries(candidates, spread_band, **kw)
            calls["entries"].append({
                "band": spread_band,
                "pair_types": kw.get("pair_types", ("time_series", "same_title")),
                "tier_floors": kw.get("tier_floors", True),
                "pairs": kw.get("_pairs"), "out": out})
            return out

        def simulate_spy(raw_entries, start_date, initial_balance, k=None,
                         spread_band=None, population="all", **kw):
            released = all(not hasattr(c, "candles_by_ticker") and not hasattr(c, "all_pairs")
                           for c in calls["candidates"])
            point = real_simulate(raw_entries, start_date, initial_balance, k=k,
                                  spread_band=spread_band, population=population, **kw)
            calls["simulate"].append({
                "entries": list(raw_entries), "raw": raw_entries, "band": spread_band,
                "population": population, "tier_floors": kw.get("tier_floors", True),
                "point": point, "released": released})
            return point

        def populations_spy(entries, split_date):
            # The list object itself (not a copy), so a call can be matched to
            # the simulations of the same entries by identity
            calls["populations"].append({"entries": entries, "split_date": split_date})
            return real_populations(entries, split_date)

        mp.setattr(backtester, "_prepare_candidates", prepare_spy)
        mp.setattr(backtester, "_entries_for_band", entries_spy)
        mp.setattr(backtester, "_simulate_at_discount", simulate_spy)
        mp.setattr(backtester, "_band_populations", populations_spy)
        root.setLevel(logging.INFO)
        root.addHandler(handler)
        result = run_backtest_sweep(
            hist_client=MagicMock(), live_client=MagicMock(),
            start_date=golden._START, initial_balance=10_000.0,
            same_event_ladders=True, band_sweep=True, tier_off_sweep=True,
        )
        root.removeHandler(handler)
        mp.undo()

        # The same run without the family: what the tier-on payload must equal
        golden._patch(mp)
        plain = run_backtest_sweep(
            hist_client=MagicMock(), live_client=MagicMock(),
            start_date=golden._START, initial_balance=10_000.0,
            same_event_ladders=True, band_sweep=True,
        )
        yield SimpleNamespace(result=result, plain=plain, calls=calls,
                              messages=handler.messages, start=golden._START)
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
        mp.undo()


@pytest.mark.usefixtures("golden_tier_off_sweep")
class TestTierOffSweep:
    """run_backtest_sweep(band_sweep=True, tier_off_sweep=True): every band a
    deadline-gap tier floor binds at (floors 0, 0.20 and 0.25 — 18 of the 36)
    entered and simulated again with the tiers off, the band floor alone
    gating the spread.

    On the golden fixture the ladder is the only pair the tiers hold back
    (see TestTierFloorsOff): with them off it enters on Monday 1 at every
    binding band. EA/EB (0.30, 13 days), TA/TB (exactly 0.15, 9 days) and
    WA/WB (0.98, 9 days) are short-gap pairs whose 0.15 tier binds only at
    floor 0, where each clears it anyway, and FA/FB (a pricier earlier
    contract) never enters."""

    _LADDER_OFF = ("time_series", "will spacex launch another starship by ?",
                   "will spacex launch another starship by ?",
                   date(2026, 1, 5), 0.2, 0.45, 0.8, 0.55, 19, "RUNG-EARLY", "RUNG-LATE")

    @staticmethod
    def _same_point(a, b):
        assert (a.k, a.spread_band, a.population, a.tier_floors) == (
            b.k, b.spread_band, b.population, b.tier_floors)
        assert [astuple(t) for t in a.trades] == [astuple(t) for t in b.trades]
        pd.testing.assert_frame_equal(a.equity_df, b.equity_df)
        assert (a.halves, a.ex_top_event) == (b.halves, b.ex_top_event)

    def test_the_tier_on_payload_does_not_change(self, golden_tier_off_sweep):
        res, plain = golden_tier_off_sweep.result, golden_tier_off_sweep.plain
        assert plain.tier_off_scenarios == [] and plain.tier_off_calibrations_by_band == {}
        self._same_point(res.primary, plain.primary)
        for mine, theirs in ((res.points, plain.points), (res.scenarios, plain.scenarios)):
            assert len(mine) == len(theirs)
            for a, b in zip(mine, theirs, strict=True):
                self._same_point(a, b)
        assert res.calibration == plain.calibration
        assert res.calibrations_by_band == plain.calibrations_by_band
        self._same_point(res.same_title_point, plain.same_title_point)
        assert res.split_date == plain.split_date == date(2026, 1, 5)
        assert res.same_event_ladders is plain.same_event_ladders is True

    def test_only_the_binding_bands_get_a_twin(self, golden_tier_off_sweep):
        res = golden_tier_off_sweep.result
        off = res.tier_off_scenarios
        assert sorted({p.spread_band for p in off}) == _TIER_BOUND_BANDS
        assert sorted(res.tier_off_calibrations_by_band) == _TIER_BOUND_BANDS
        assert all(p.tier_floors is False for p in off)
        assert all(p.tier_floors is True
                   for p in [res.primary, *res.points, *res.scenarios, res.same_title_point])
        # Every bound band x k has all four populations (the ladder enters at
        # every bound band, and EA/EB keeps the cross population non-empty)
        cells = sorted((b, k) for b in _TIER_BOUND_BANDS for k in INTERVAL_DISCOUNT_SWEEP)
        for population in ("all", "time_series", "ladder", "cross"):
            assert sorted((p.spread_band, p.k) for p in off if p.population == population) == cells
        assert len(off) == 18 * 13 * 4
        # The checks sit where the tier-on sweep puts them: halves on every
        # "all" and "time_series" point, ex-top wherever a trade names an
        # event (every "all" point trades the same-title pair's)
        for p in off:
            if p.population in ("all", "time_series"):
                assert p.halves is not None
                assert (p.ex_top_event is not None) == any(t.event_ticker for t in p.trades)
            else:
                assert p.halves is None and p.ex_top_event is None
        assert all(p.ex_top_event is not None for p in off if p.population == "all")
        # Same order as scenarios: band, then k, then population
        order = {"all": 0, "time_series": 1, "ladder": 2, "cross": 3}
        keys = [(p.spread_band, p.k, order[p.population]) for p in off]
        assert keys == sorted(keys)

    def test_the_tier_off_pre_pass_matches_a_full_scan(self, golden_tier_off_sweep, monkeypatch):
        passes = [c for c in golden_tier_off_sweep.calls["entries"] if c["tier_floors"] is False]
        assert all(c["pair_types"] == ("time_series",) for c in passes)
        # One full scan, of (0, 1), first; every other tier-off band rescans
        # one shared subset — the four pairs that entered there
        assert passes[0]["band"] == (0.0, 1.0) and passes[0]["pairs"] is None
        assert [c["band"] for c in passes if c["pairs"] is None] == [(0.0, 1.0)]
        rescans = [c["pairs"] for c in passes[1:]]
        assert len(rescans) == 17 and all(pairs is rescans[0] for pairs in rescans)
        assert [(i[0][0]["ticker"], i[0][1]["ticker"]) for i in rescans[0]] == [
            ("EA", "EB"), ("RUNG-EARLY", "RUNG-LATE"), ("TA", "TB"), ("WA", "WB")]
        # Every tier-off band's entries are a full tier-off scan's
        rows = TestPrepareEntriesGolden._rows
        by_band = {c["band"]: c["out"] for c in passes}
        assert sorted(by_band) == _TIER_BOUND_BANDS
        for band, out in by_band.items():
            full = backtester._entries_for_band(TestBandSweepPhaseOneSubset._fresh(monkeypatch),
                                                band, pair_types=("time_series",),
                                                tier_floors=False)
            assert rows(out) == rows(full), band

    def test_the_ladder_row_moves_to_monday_1(self, golden_tier_off_sweep):
        golden, rows = TestPrepareEntriesGolden, TestPrepareEntriesGolden._rows
        entries = golden_tier_off_sweep.calls["entries"]
        on = next(c["out"] for c in entries if c["tier_floors"] is True
                  and c["band"] == (0.0, 1.0) and c["pair_types"] == ("time_series",))
        off = next(c["out"] for c in entries if c["tier_floors"] is False
                   and c["band"] == (0.0, 1.0))
        assert rows(on) == [golden._EA_EB, golden._LADDER, golden._TA_TB, golden._WA_WB]
        assert rows(off) == [golden._EA_EB, self._LADDER_OFF, golden._TA_TB, golden._WA_WB]
        assert self._LADDER_OFF != golden._LADDER
        # ... and the trade moves with it
        res = golden_tier_off_sweep.result
        off_point = next(p for p in res.tier_off_scenarios if p.population == "all"
                         and p.spread_band == (0.0, 1.0)
                         and p.k == TIME_SERIES_INTERVAL_PROB_DISCOUNT)

        def ladder(point):
            return next(t for t in point.trades if t.ticker_a == "RUNG-EARLY")

        assert ladder(off_point).entry_date == date(2026, 1, 5)
        assert ladder(res.primary).entry_date == date(2026, 1, 12)

    def test_every_completion_prefix_is_unique_across_both_families(self, golden_tier_off_sweep):
        prefixes = _completion_prefixes(golden_tier_off_sweep.messages)
        assert len(prefixes) == len(set(prefixes))
        off = [p for p in prefixes if " with the tier floors off" in p]
        n_off = sum(1 for c in golden_tier_off_sweep.calls["simulate"] if c["tier_floors"] is False)
        assert len(off) == n_off > 0
        for p in off:
            assert re.fullmatch(r"Backtest complete at k=\S+, band \S+ with the tier floors "
                                r"off, \S+", p), p
        # rsplit still reads the population, on both families
        labels = [p.rsplit(", ", 1)[1] for p in prefixes]
        assert set(labels) <= set(backtester._SIMULATION_LABELS)
        off_labels = [p.rsplit(", ", 1)[1] for p in off]
        off_set = set(off)
        on_labels = [p.rsplit(", ", 1)[1] for p in prefixes if p not in off_set]
        assert off_labels.count("all") == 18 * 13
        assert on_labels.count("all") == 468

    def test_every_tier_off_entry_clears_the_floor_alone(self, golden_tier_off_sweep):
        # The tier-off twin of TestBandSweep's in-band check
        eps = backtester.PRICE_EPSILON
        by_point = {id(c["point"]): c for c in golden_tier_off_sweep.calls["simulate"]}
        below_tier = 0
        for point in golden_tier_off_sweep.result.tier_off_scenarios:
            call = by_point[id(point)]
            assert call["band"] == point.spread_band and call["tier_floors"] is False
            lo, hi = point.spread_band
            for rec in call["entries"]:
                if rec["pair_type"] != "time_series":
                    continue
                e = rec["entry"]
                spread = e["pB"] - e["pA"]
                assert spread <= hi + eps
                assert spread >= min_price_diff_for_gap(e["gap_days"], spread_min=lo,
                                                        tier_floors=False) - eps
                below_tier += spread < min_price_diff_for_gap(e["gap_days"], spread_min=lo) - eps
        # Not vacuous: some tier-off entry sits below the tier the tier-on rule
        # demands (the ladder's 0.25 against its 0.30 tier)
        assert below_tier > 0

    def test_every_tier_off_check_reads_its_own_entries(self, golden_tier_off_sweep):
        res = golden_tier_off_sweep.result
        calls = [c for c in golden_tier_off_sweep.calls["simulate"] if c["tier_floors"] is False]
        by_point = {id(c["point"]): c for c in calls}
        by_key: dict = {}
        for c in calls:
            key = (c["band"], c["point"].k, c["population"])
            assert key not in by_key          # one simulation per (band, k, label)
            by_key[key] = c
        for p in res.tier_off_scenarios:
            if p.population not in ("all", "time_series"):
                continue
            own = by_point[id(p)]["entries"]
            # Split at the ONE split date, like every tier-on cell
            h1 = by_key[(p.spread_band, p.k, f"{p.population}/H1")]
            h2 = by_key[(p.spread_band, p.k, f"{p.population}/H2")]
            assert [id(r) for r in h1["entries"]] == [
                id(r) for r in own if r["entry"]["entry_date"] < res.split_date]
            assert [id(r) for r in h2["entries"]] == [
                id(r) for r in own if r["entry"]["entry_date"] >= res.split_date]
            assert (p.halves.h1_entries, p.halves.h2_entries) == (
                len(h1["entries"]), len(h2["entries"]))
            if p.ex_top_event is not None:
                ex = by_key[(p.spread_band, p.k, f"{p.population}/ex-top")]
                assert [id(r) for r in ex["entries"]] == [
                    id(r) for r in own
                    if r["entry"]["mA"]["event_ticker"] != p.ex_top_event[0]]

    def test_every_band_splits_at_the_one_split_date(self, golden_tier_off_sweep):
        # The tier-off halves split at the SAME date object as every tier-on
        # cell's — pinned by identity, not value (see the non-vacuity check
        # at the end)
        res, calls = golden_tier_off_sweep.result, golden_tier_off_sweep.calls
        off_all = [c for c in calls["simulate"]
                   if c["tier_floors"] is False and c["population"] == "all"]
        off_calls = [c for c in calls["populations"]
                     if any(c["entries"] is sim["raw"] for sim in off_all)]
        # One per binding band, each handed the tier-off entries it simulates
        assert len(off_calls) == 18
        assert all(c["split_date"] is res.split_date for c in off_calls)
        # ... as is every tier-on band's (36 calls, 54 in all)
        assert len(calls["populations"]) == 36 + 18
        assert all(c["split_date"] is res.split_date for c in calls["populations"])
        # Not vacuous: a split date recomputed from the tier-off (0, 1)
        # entries would EQUAL the one split date (the ladder moves onto
        # Monday 1 beside the other three) without BEING it, so a value-only
        # check could not tell the two apart
        off_primary = next(sim["raw"] for sim in off_all if sim["band"] == (0.0, 1.0))
        recomputed = backtester._split_date(
            [rec for rec in off_primary if rec["pair_type"] == "time_series"],
            golden_tier_off_sweep.start)
        assert recomputed == res.split_date and recomputed is not res.split_date

    def test_the_candles_are_released_before_any_tier_off_simulation(self, golden_tier_off_sweep):
        sims = [c for c in golden_tier_off_sweep.calls["simulate"] if c["tier_floors"] is False]
        assert sims and all(c["released"] for c in sims)

    def test_every_tier_off_calibration_is_labelled_with_the_floor_alone(self, golden_tier_off_sweep):
        res = golden_tier_off_sweep.result
        for (lo, _hi), cal in res.tier_off_calibrations_by_band.items():
            # EA/EB (13 days) and the ladder (19 days) at every bound band;
            # TA/TB and WA/WB are voided, so never measured
            assert [(b.label, b.tier) for b in cal.buckets] == [("8-15d", lo), ("16-30d", lo)]
        # The ladder is measured at its Monday-1 spread with the tiers off
        off = res.tier_off_calibrations_by_band[(0.0, 1.0)]
        on = res.calibrations_by_band[(0.0, 1.0)]
        assert sorted(o.implied for o in off.observations) == pytest.approx([0.25, 0.30])
        assert sorted(o.implied for o in on.observations) == pytest.approx([0.30, 0.40])

    def test_no_tier_off_line_is_counted_as_a_tier_on_one(self, golden_tier_off_sweep):
        messages = golden_tier_off_sweep.messages
        # The two line families other tests count are exactly the tier-on sweep's
        announced = [m for m in messages if m.startswith("Spread band ")]
        assert len(announced) == 36 and not any("Tier floors" in m for m in announced)
        assert [m for m in messages if m.startswith("Split-half check: split date")] == [
            "Split-half check: split date 2026-01-05 leaves H1 empty; the split-half "
            "check is not measurable for this window (primary band 0-1: 0 time-series "
            "entries before it, 4 on or after it)"]
        tier_off = [m for m in messages if m.startswith("Tier floors off")]
        assert tier_off[:3] == [
            "Tier floors off (backtest only): 18 of 36 spread bands have a floor below a "
            "deadline-gap tier and are entered again without the tiers; the other 18 enter "
            "the same pairs either way",
            "Tier floors off: no-band pre-pass: scanning all 5 time-series pairs at 0-1",
            "Tier floors off: no-band pre-pass: 4 of 5 time-series pairs produced an entry; "
            "every other tier-off band rescans only those"]
        assert sum(m.startswith("Tier floors off: spread band ") for m in messages) == 18
        assert sum(m.startswith("Tier floors off: simulating spread band ") for m in messages) == 18

    def test_the_tier_off_sweep_needs_the_band_sweep(self, monkeypatch, caplog):
        monkeypatch.setattr(backtester, "fetch_all_settled_markets",
                            lambda *a, **k: pytest.fail("fetch must not run"))
        with caplog.at_level(logging.INFO), pytest.raises(ValueError, match="band_sweep"):
            run_backtest_sweep(MagicMock(), MagicMock(), date(2026, 1, 1), 1000.0,
                               band_sweep=False, tier_off_sweep=True)
        assert caplog.records == []
        # A direct caller of the second half is refused too, before its
        # candidates are consumed
        candidates = backtester._Candidates(
            all_pairs=[], candles_by_ticker={}, label_coverage=None,
            start_date=date(2026, 1, 1), max_horizon_days=None, same_event_ladders=True)
        with pytest.raises(ValueError, match="band_sweep"):
            backtester._sweep_from_candidates(
                candidates, 1000.0, interval_discount=None, sweep=False,
                spread_band=None, band_sweep=False, tier_off_sweep=True)
        assert hasattr(candidates, "all_pairs") and hasattr(candidates, "candles_by_ticker")
