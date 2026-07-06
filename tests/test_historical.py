"""Tests for historical.py — event-title cache and dict serialization."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# historical.py imports from kalshi_python_sync.api.historical_api which only
# exists in SDK >= 3.13.0. Skip cleanly in environments with an older SDK so
# this test module doesn't break collection of the rest of the suite.
pytest.importorskip("kalshi_python_sync.api.historical_api")

from kalshi_betting import historical  # noqa: E402


def _make_client_with_event_pages(non_mve_pages: list[list[tuple[str, str]]],
                                  mve_pages: list[list[tuple[str, str]]] | None = None,
                                  single_lookups: dict[str, str] | None = None,
                                  single_failures: set[str] | None = None):
    """Build a MagicMock client whose get_events / get_multivariate_events / get_event
    return paginated event listings as configured.

    Each page is a list of (event_ticker, title) tuples. The mock walks the pages
    in order until exhausted, then returns an empty page with cursor=None.
    """
    mve_pages = mve_pages or []
    single_lookups = single_lookups or {}
    single_failures = single_failures or set()

    def _build_resp(page):
        events = [SimpleNamespace(event_ticker=tkr, title=title) for tkr, title in page]
        return SimpleNamespace(events=events, cursor=None)

    client = MagicMock()
    # Each call returns the next page; pad with empty page when exhausted.
    non_mve_iter = iter(non_mve_pages + [[]])
    mve_iter = iter(mve_pages + [[]])

    def get_events(status=None, limit=None, cursor=None):
        try:
            return _build_resp(next(non_mve_iter))
        except StopIteration:
            return _build_resp([])

    def get_multivariate_events(limit=None, cursor=None):
        try:
            return _build_resp(next(mve_iter))
        except StopIteration:
            return _build_resp([])

    def get_event(event_ticker=None):
        if event_ticker in single_failures:
            raise RuntimeError(f"simulated 404 for {event_ticker}")
        title = single_lookups.get(event_ticker, "")
        return SimpleNamespace(event=SimpleNamespace(title=title))

    client.get_events = MagicMock(side_effect=get_events)
    client.get_multivariate_events = MagicMock(side_effect=get_multivariate_events)
    client.get_event = MagicMock(side_effect=get_event)
    return client


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Redirect _EVENT_TITLES_CACHE to a temp file so tests don't touch real cache."""
    cache_file = tmp_path / "event_titles.json"
    monkeypatch.setattr(historical, "_EVENT_TITLES_CACHE", cache_file)
    return cache_file


class TestEventTitlesCache:
    def test_resolves_from_bulk_events_endpoint(self, isolated_cache):
        # Single page with both tickers, no MVE, no fallback needed
        client = _make_client_with_event_pages(
            non_mve_pages=[[("E1", "Event One Title"), ("E2", "Event Two Title")]],
        )
        result = historical._load_or_build_event_titles(client, {"E1", "E2"})
        assert result == {"E1": "Event One Title", "E2": "Event Two Title"}
        # get_event must NOT be called when bulk satisfies all misses
        assert client.get_event.call_count == 0

    def test_falls_back_to_multivariate_endpoint(self, isolated_cache):
        # First-tier (get_events) misses E2; it shows up in the MVE endpoint
        client = _make_client_with_event_pages(
            non_mve_pages=[[("E1", "Event One Title")]],
            mve_pages=[[("E2", "MVE Event Two")]],
        )
        result = historical._load_or_build_event_titles(client, {"E1", "E2"})
        assert result["E1"] == "Event One Title"
        assert result["E2"] == "MVE Event Two"
        # Both bulk endpoints used; single lookup not needed
        assert client.get_event.call_count == 0

    def test_falls_back_to_single_event_lookup(self, isolated_cache):
        # Bulk endpoints return nothing; per-ticker fallback resolves the title
        client = _make_client_with_event_pages(
            non_mve_pages=[],
            mve_pages=[],
            single_lookups={"OLD-1": "Archived Event Title"},
        )
        result = historical._load_or_build_event_titles(client, {"OLD-1"})
        assert result == {"OLD-1": "Archived Event Title"}
        # Exactly one per-ticker fallback call
        assert client.get_event.call_count == 1

    def test_poison_pill_on_lookup_failure(self, isolated_cache):
        # When the per-ticker fallback raises, ticker maps to "" and is persisted
        client = _make_client_with_event_pages(
            non_mve_pages=[],
            mve_pages=[],
            single_failures={"BAD-1"},
        )
        result = historical._load_or_build_event_titles(client, {"BAD-1"})
        assert result == {"BAD-1": ""}

    def test_cache_hit_skips_api(self, isolated_cache):
        # First call populates the cache; second call should not touch the API
        client1 = _make_client_with_event_pages(non_mve_pages=[[("E1", "Title One")]])
        historical._load_or_build_event_titles(client1, {"E1"})

        client2 = _make_client_with_event_pages(non_mve_pages=[])
        result = historical._load_or_build_event_titles(client2, {"E1"})
        assert result == {"E1": "Title One"}
        # No API calls on the cache hit
        assert client2.get_events.call_count == 0
        assert client2.get_multivariate_events.call_count == 0

    def test_use_cache_false_bypasses_disk(self, isolated_cache):
        # Pre-populate disk cache with a stale value
        client1 = _make_client_with_event_pages(non_mve_pages=[[("E1", "Old Title")]])
        historical._load_or_build_event_titles(client1, {"E1"})

        # use_cache=False should re-fetch — assert get_events is called
        client2 = _make_client_with_event_pages(non_mve_pages=[[("E1", "Fresh Title")]])
        result = historical._load_or_build_event_titles(client2, {"E1"}, use_cache=False)
        assert result["E1"] == "Fresh Title"
        assert client2.get_events.call_count >= 1


class TestMarketToDict:
    def test_carries_event_title(self):
        from datetime import UTC, datetime
        m = SimpleNamespace(
            ticker="T1",
            event_ticker="E1",
            _event_title="Some Event Title",
            title="Market Title",
            subtitle="Sub",
            result="yes",
            yes_ask_dollars="0.45",
            no_ask_dollars="0.55",
            yes_bid_dollars="0.43",
            close_time=datetime(2025, 1, 1, tzinfo=UTC),
            settlement_ts=datetime(2025, 1, 2, tzinfo=UTC),
            status="settled",
        )
        d = historical._market_to_dict(m)
        assert d["event_title"] == "Some Event Title"
        assert d["event_ticker"] == "E1"
        assert d["title"] == "Market Title"

    def test_event_title_defaults_to_empty_when_missing(self):
        from datetime import UTC, datetime
        m = SimpleNamespace(
            ticker="T1",
            event_ticker="E1",
            title="Market Title",
            subtitle="Sub",
            result="yes",
            yes_ask_dollars="0.45",
            no_ask_dollars="0.55",
            yes_bid_dollars="0.43",
            close_time=datetime(2025, 1, 1, tzinfo=UTC),
            settlement_ts=datetime(2025, 1, 2, tzinfo=UTC),
            status="settled",
        )
        # No _event_title attached
        d = historical._market_to_dict(m)
        assert d["event_title"] == ""
