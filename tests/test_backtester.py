"""Tests for backtester.py grouping helpers — combined-key MVE handling."""
import pytest

# backtester.py transitively imports historical.py, which in turn imports
# kalshi_python_sync.api.historical_api (only present in SDK >= 3.13.0).
# Skip cleanly so older-SDK environments don't fail test collection.
pytest.importorskip("kalshi_python_sync.api.historical_api")

from kalshi_betting.backtester import (  # noqa: E402
    _extract_pairs,
    _group_by_exact_title,
    _group_by_normalized_title,
    _pair_key,
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
        _, _, canon = pairs[0]
        assert canon == "Republicans win majority"
