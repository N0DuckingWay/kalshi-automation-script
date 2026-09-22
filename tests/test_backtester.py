"""Tests for backtester.py — grouping helpers, P&L math, and entry direction."""
import gc
import logging
import re
import time
import weakref
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
from kalshi_betting import backtester, scanner
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
    BUDGET_FRACTION,
    INTERVAL_DISCOUNT_SWEEP,
    MAX_DEADLINE_GAP_DAYS,
    MVE_SERIES_FAMILY_PREFIX,
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
        mA = _md("A1", "EVA-1", title="Republicans win majority",
                 event_title="2026 Senate Control")
        mB = _md("B1", "EVB-1", title="Republicans win majority",
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
        mA = self._old_style_dict("A1", "EVA-1", "Republicans win majority",
                                   "2026-01-01T00:00:00Z", event_title="2026 Senate Control")
        mB = self._old_style_dict("B1", "EVB-1", "Republicans win majority",
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
    discount (p = 1 - 0.75*0.35) the Kelly fraction is ~0.180 — positive, so
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


def _naive_time_series_pairs(members: list[dict], margin_days: int) -> set[frozenset]:
    """Independent oracle: naive O(n^2) double loop over the same group,
    filtering by the same margin-inclusive close-time gap, the same
    event_ticker rule, the same one-series rule (DR-02/DR-54/DR-55) AND the
    same cumulative-deadline rule that _extract_pairs applies, but without any
    sorting/windowing. Written standalone (no backtester internals besides
    plain dict/date arithmetic, _naive_series' restatement of the series
    identity and _naive_cumulative_deadline's of the deadline spans) so it can
    serve as ground truth for the windowed implementation.

    The one-series conjunct is spelled out here rather than imported, for the
    same reason the rest is: an oracle that reuses the implementation cannot
    falsify it. It is not dead weight on the current fixtures only by
    accident — _build_synthetic_group's members carry no wording keys, so the
    wording half is True for every pair in them and the distinct hyphen-less
    event tickers are the only thing keeping the conjunct from firing. Add one
    hyphenated shared-prefix ticker to that fixture and an oracle without this
    clause diverges silently, which is exactly the "oracle replays the old
    rule" failure CLAUDE.md records for the archive-walk parity tests.
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
    """TS-07: the grouping/pairing step holds the whole record list, two group
    maps and two pair lists live at once. It is bracketed by RSS lines, the
    first of which precedes a RAM-budget warning carrying only this run's own
    numbers, and the maps and the record list are released together before the
    candlestick pool runs. The measured figures behind all of this live in
    config.py beside BACKTEST_RECORD_BYTES_ESTIMATE, not here.
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
        return [r.getMessage() for r in caplog.records
                if "eligible markets: their records alone are" in r.getMessage()]

    def test_rss_lines_bracket_the_grouping_step(self, monkeypatch, caplog):
        with caplog.at_level("INFO"):
            self._run(monkeypatch)
        labels = [r.args[0] for r in caplog.records
                  if r.getMessage().startswith("Peak RSS")]
        # In order, and exactly the two that bracket grouping/pairing — the
        # window between the existing "Total settled markets" and "Potential
        # pairs" lines, where the peak lives and is otherwise invisible.
        assert labels == ["before grouping", "after pair extraction"]

    def test_ram_warning_fires_above_the_threshold(self, monkeypatch, caplog):
        monkeypatch.setattr(backtester, "BACKTEST_MARKETS_RAM_WARN", 1)
        with caplog.at_level("WARNING"):
            self._run(monkeypatch)
        warnings = self._ram_warnings(caplog)
        assert len(warnings) == 1
        assert warnings[0].startswith("2 eligible markets")

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
                       if "eligible markets: their records alone are" in m)
        assert rss_at < warn_at

    def test_ram_warning_quotes_only_this_runs_numbers(self, monkeypatch, caplog):
        # A line emitted on every run must not carry another run's
        # measurements: those live in config.py's comment, where a reader is
        # prompted to keep them current. The only numbers here are this run's
        # own market count and the footprint derived from it.
        monkeypatch.setattr(backtester, "BACKTEST_MARKETS_RAM_WARN", 1)
        with caplog.at_level("WARNING"):
            self._run(monkeypatch)
        message = self._ram_warnings(caplog)[0]
        expected_gb = 2 * backtester.BACKTEST_RECORD_BYTES_ESTIMATE / 1e9
        assert message.startswith("2 eligible markets")
        assert f"{expected_gb:.1f} GB" in message
        # Every numeric token in the line is derived from this run.
        numbers = re.findall(r"\d+(?:\.\d+)?", message)
        assert numbers == ["2", f"{expected_gb:.1f}"]

    def test_unpaired_records_are_released_before_the_candlestick_fetch(
        self, monkeypatch,
    ):
        """TS-07: deleting the group maps alone frees no record dicts, because
        `markets` still references every one of them. Deleting the list too is
        what lets a market that landed in no candidate pair be collected
        before the candlestick pool and the _find_entry sweep run.

        Residency, not peak: the process high-water mark is already set by
        this point. The probe is a weakref taken inside the patched fetch, so
        the test itself never holds the record alive.
        """
        candles = {
            "EA": [_candle(_MONDAY_TS, 0.30, 0.70)],
            "EB": [_candle(_MONDAY_TS, 0.60, 0.40)],
        }
        # A third eligible market with a title that groups with nothing else,
        # so it survives the prefilter but appears in no candidate pair.
        lonely = {"ticker": "EC", "event_ticker": "EVC", "event_title": "EVC",
                  "title": "Unrelated question by March 1, 2026", "subtitle": "",
                  "result": "no",
                  "open_time": "2026-01-01T00:00:00+00:00",
                  "close_time": "2026-03-01T00:00:00+00:00",
                  "settlement_ts": "2026-03-01T12:00:00+00:00"}
        probes: dict[str, weakref.ref] = {}

        def _fetch(*_a, **_k):
            # Built and weak-referenced HERE so the only strong references are
            # the ones the backtester itself keeps.
            records = [_WeakrefDict(m) for m in self._markets() + [lonely]]
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
        run_backtest(hist_client=MagicMock(), live_client=MagicMock(),
                     start_date=date(2026, 1, 1), initial_balance=10_000.0)

        # The unpaired record is gone; the two that a candidate pair holds are
        # still alive, because the pair lists legitimately reference them.
        assert observed == {"EA": True, "EB": True, "EC": False}


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
        census is filtering or consuming something it only meant to read."""
        with_census_trades, with_census_equity = self._run(monkeypatch)
        monkeypatch.setattr(backtester, "_log_outcome_label_coverage",
                            lambda _markets: None)
        without_trades, without_equity = self._run(monkeypatch)

        assert [astuple(t) for t in with_census_trades] == \
               [astuple(t) for t in without_trades]
        pd.testing.assert_frame_equal(with_census_equity, without_equity)
        # Guard against the comparison being vacuous.
        assert len(with_census_trades) == 1

    def test_census_is_logged_inside_the_grouping_window(self, monkeypatch, caplog):
        # After the RSS/RAM-budget lines and before the pair counts, i.e. while
        # `markets` is still alive — it is del'd right after pair extraction.
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
    is the only scope the eligible-market list exists in and its lifetime must
    not be extended (TS-07).

    The carrier is a RETURN VALUE rather than an optional sink precisely
    because a sink can be forgotten — which would reproduce, in the mechanism
    built to close DR-66's silence, exactly that silence.
    """

    def test_the_census_returns_what_it_logged(self, caplog):
        markets = [{"subtitle": "Yes", "event_title": "E"}] * 3 + [{}] * 7
        with caplog.at_level("INFO"):
            coverage = backtester._log_outcome_label_coverage(markets)

        assert coverage.total == 10
        assert coverage.with_subtitle == 3
        assert coverage.with_event_title == 3
        assert coverage.subtitle_fraction == pytest.approx(0.30)
        assert coverage.event_title_fraction == pytest.approx(0.30)
        # The same number reached the log, so page and log cannot disagree.
        assert any("30.00%" in r.getMessage() for r in caplog.records)

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

    def test_the_carrier_holds_no_market_reference(self):
        # _prepare_entries del's the record list right after pair extraction to
        # lower residency across the candlestick fetch. A carrier holding
        # examples would pin every record alive past that statement.
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
        )
        monkeypatch.setattr(backtester, "_prepare_entries",
                            lambda *a, **k: ([], census))
        monkeypatch.setattr(backtester, "_interval_calibration", lambda *a, **k: None)
        result = backtester.run_backtest_sweep(
            MagicMock(), MagicMock(), date(2026, 1, 1), 1000.0, sweep=False)
        assert result.label_coverage is census

    def test_the_infeasible_window_carries_no_census(self, monkeypatch):
        # The fetch never ran, so nothing was censused — None, distinct from a
        # censused corpus that held zero records.
        monkeypatch.setattr(backtester, "_prepare_entries",
                            lambda *a, **k: (None, None))
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
    subtitle) — so _group_by_exact_title pairs them as same_title — whose
    titles therefore also normalize to one key, so _group_by_normalized_title
    GROUPS the same two tickers for time_series too.

    That used to yield both copies, which is what _drop_cross_type_duplicates
    was built to resolve. It no longer can: a time-series pair must state two
    DIFFERENT cumulative deadlines, and identical wording cannot state two of
    anything. The two conditions are now mutually exclusive on one ticker pair,
    so Pass 1 produces the same-title copy alone and the dedup has nothing to
    drop. The helper stays — it is still the right thing to do if a collision
    ever arises another way — and TestDropCrossTypeDuplicates unit-tests it
    directly; what changed is that the FINDERS no longer manufacture one.
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
    # The time-series copy would once have formed too — the later contract is
    # priced 0.30 higher, clearing the 15% short-gap tier, with legs
    # pA+nB = 0.30+0.40 = 0.70 <= 0.85 and a positive Kelly fraction. It is now
    # refused earlier than any of that, at extraction: both legs are worded
    # "Q", so they state no deadline at all, let alone two different ones.
    _CANDLES = {
        "DA": [_candle(_MONDAY_TS, 0.30, 0.70)],
        "DB": [_candle(_MONDAY_TS, 0.60, 0.40)],
    }

    def test_fixture_lands_in_both_groupings_but_only_one_yields_a_candidate(self):
        # GROUPING is untouched by the cumulative-deadline rule, so the pair is
        # still discovered by both keys...
        assert len(_group_by_exact_title(self._MARKETS)) == 1
        assert len(_group_by_normalized_title(self._MARKETS)) == 1
        # ...but only the same-title branch produces a candidate. The legs are
        # worded identically, so they state one deadline, not two, and the
        # time-series branch refuses them.
        assert len(_extract_pairs(_group_by_exact_title(self._MARKETS))) == 1
        assert _extract_pairs(_group_by_normalized_title(self._MARKETS)) == []

    def test_no_time_series_duplicate_reaches_the_dedup(self, monkeypatch):
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

        # Pass 1 produces the same-title copy ONLY: identical wording cannot
        # state two different deadlines, so the time-series copy is refused at
        # extraction and never reaches the dedup at all.
        assert len(_typed(seen["in"], "same_title")) == 1
        assert _typed(seen["in"], "time_series") == []
        # The dedup is therefore a no-op here, and the same-title copy is
        # carried into Pass 2 unchanged.
        assert len(_typed(seen["out"], "same_title")) == 1
        assert _typed(seen["out"], "time_series") == []

        assert len(trades) == 1
        assert trades[0].pair_type == "same_title"
        # The same-title copy canonicalizes A as the pricier side (DB)
        assert (trades[0].ticker_a, trades[0].ticker_b) == ("DB", "DA")

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
        # _same_series_dicts is False on two different series. Mirror of
        # test_scanner.py::TestDeadlineGuardFinders::
        # test_dated_identical_wording_is_same_title_only.
        title = "Will X happen by Dec 31, 2026?"
        mA = _md("A1", "EVA-1", title=title, event_title="EV")
        mA["close_time"] = "2026-12-01T00:00:00Z"
        mB = _md("B1", "EVB-1", title=title, event_title="EV")
        mB["close_time"] = "2026-12-20T00:00:00Z"
        assert len(_group_by_normalized_title([mA, mB])) == 1
        assert backtester._deadline_profile_dict(mA) == (
            scanner.DEADLINE_CUMULATIVE, ("by dec 31, 2026",),
        )
        assert backtester._deadline_profile_dict(mB) == (
            scanner.DEADLINE_CUMULATIVE, ("by dec 31, 2026",),
        )
        assert backtester._identical_wording_dicts(mA, mB) is True
        assert backtester._same_series_dicts(mA, mB) is False
        assert len(_extract_pairs(_group_by_exact_title([mA, mB]))) == 1
        assert _extract_pairs(_group_by_normalized_title([mA, mB])) == []

        # Positive control: B's wording states a DIFFERENT deadline on the
        # same two series, same close times. The spans now differ, so the
        # time-series pair forms — proving the [] above comes from the
        # spans-differ conjunct rather than from the price tier, the deadline
        # gap, or the two series being distinct.
        mB3 = _md("B1", "EVB-1", title="Will X happen by Dec 20, 2026?", event_title="EV")
        mB3["close_time"] = "2026-12-20T00:00:00Z"
        assert len(_extract_pairs(_group_by_normalized_title([mA, mB3]))) == 1


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
            "prepare": backtester._prepare_entries,
            "calibrate": backtester._interval_calibration,
            "simulate": backtester._simulate_at_discount,
        }

        def counted(name):
            def wrapper(*a, **kw):
                calls[name] += 1
                return real[name](*a, **kw)
            return wrapper

        monkeypatch.setattr(backtester, "_prepare_entries", counted("prepare"))
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
        monkeypatch.setattr(backtester, "_prepare_entries", lambda *a, **k: ([], None))
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
        monkeypatch.setattr(backtester, "_prepare_entries", lambda *a, **k: ([], None))
        monkeypatch.setattr(backtester, "_interval_calibration", lambda *a, **k: None)
        with caplog.at_level(logging.INFO):
            backtester.run_backtest_sweep(
                MagicMock(), MagicMock(), date(2026, 1, 1), 1000.0, sweep=False,
            )
        assert any("Simulating the primary interval discount" in r.getMessage()
                   for r in caplog.records)


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
