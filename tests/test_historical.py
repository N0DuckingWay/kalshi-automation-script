"""Tests for historical.py — event-title cache and dict serialization."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kalshi_betting import historical


def _raw_resp(payload: dict) -> SimpleNamespace:
    """RESTResponse stand-in for the raw signed-GET path (_signed_raw_get)."""
    return SimpleNamespace(status=200, data=json.dumps(payload).encode("utf-8"))


def _make_client_with_event_pages(non_mve_pages: list[list[tuple[str, str]]],
                                  mve_pages: list[list[tuple[str, str]]] | None = None):
    """Build a MagicMock client whose get_events / get_multivariate_events
    return paginated event listings as configured.

    Each page is a list of (event_ticker, title) tuples. The mock walks the pages
    in order until exhausted, then returns an empty page with cursor=None.
    (These two bulk listings still use the modeled SDK calls — Event models
    without nested markets deserialize fine.)
    """
    mve_pages = mve_pages or []

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

    client.get_events = MagicMock(side_effect=get_events)
    client.get_multivariate_events = MagicMock(side_effect=get_multivariate_events)
    return client


def _patch_single_event_lookups(monkeypatch,
                                single_lookups: dict[str, str] | None = None,
                                single_failures: set[str] | None = None) -> MagicMock:
    """Patch the raw signed-GET seam used by the per-ticker /events/{tkr}
    fallback (the modeled get_event can no longer deserialize live payloads,
    so historical.py fetches raw JSON via _signed_raw_get)."""
    single_lookups = single_lookups or {}
    single_failures = single_failures or set()

    def fake_signed_get(client, path, **params):
        tkr = path.rsplit("/", 1)[-1]
        if tkr in single_failures:
            raise RuntimeError(f"simulated 404 for {tkr}")
        return _raw_resp({"event": {"title": single_lookups.get(tkr, "")}})

    mock = MagicMock(side_effect=fake_signed_get)
    monkeypatch.setattr(historical, "_signed_raw_get", mock)
    return mock


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Redirect _EVENT_TITLES_CACHE to a temp file so tests don't touch real cache."""
    cache_file = tmp_path / "event_titles.json"
    monkeypatch.setattr(historical, "_EVENT_TITLES_CACHE", cache_file)
    return cache_file


class TestEventTitlesCache:
    def test_resolves_from_bulk_events_endpoint(self, isolated_cache, monkeypatch):
        # Single page with both tickers, no MVE, no fallback needed
        client = _make_client_with_event_pages(
            non_mve_pages=[[("E1", "Event One Title"), ("E2", "Event Two Title")]],
        )
        fallback = _patch_single_event_lookups(monkeypatch)
        result = historical._load_or_build_event_titles(client, {"E1", "E2"})
        assert result == {"E1": "Event One Title", "E2": "Event Two Title"}
        # The per-ticker fallback must NOT be called when bulk satisfies all misses
        assert fallback.call_count == 0

    def test_falls_back_to_multivariate_endpoint(self, isolated_cache, monkeypatch):
        # First-tier (get_events) misses E2; it shows up in the MVE endpoint
        client = _make_client_with_event_pages(
            non_mve_pages=[[("E1", "Event One Title")]],
            mve_pages=[[("E2", "MVE Event Two")]],
        )
        fallback = _patch_single_event_lookups(monkeypatch)
        result = historical._load_or_build_event_titles(client, {"E1", "E2"})
        assert result["E1"] == "Event One Title"
        assert result["E2"] == "MVE Event Two"
        # Both bulk endpoints used; single lookup not needed
        assert fallback.call_count == 0

    def test_falls_back_to_single_event_lookup(self, isolated_cache, monkeypatch):
        # Bulk endpoints return nothing; per-ticker fallback resolves the title
        client = _make_client_with_event_pages(non_mve_pages=[], mve_pages=[])
        fallback = _patch_single_event_lookups(
            monkeypatch, single_lookups={"OLD-1": "Archived Event Title"},
        )
        result = historical._load_or_build_event_titles(client, {"OLD-1"})
        assert result == {"OLD-1": "Archived Event Title"}
        # Exactly one per-ticker fallback call
        assert fallback.call_count == 1

    def test_poison_pill_on_lookup_failure(self, isolated_cache, monkeypatch):
        # When the per-ticker fallback raises, ticker maps to "" and is persisted
        client = _make_client_with_event_pages(non_mve_pages=[], mve_pages=[])
        _patch_single_event_lookups(monkeypatch, single_failures={"BAD-1"})
        result = historical._load_or_build_event_titles(client, {"BAD-1"})
        assert result == {"BAD-1": ""}

    def test_mve_bulk_scan_is_page_capped(self, isolated_cache, monkeypatch):
        # The MVE listing is effectively unbounded — a ticker that never
        # appears must not page forever. The bulk scan stops after
        # MVE_TITLE_LOOKUP_MAX_PAGES and the per-ticker fallback resolves it.
        from kalshi_betting.config import MVE_TITLE_LOOKUP_MAX_PAGES

        client = _make_client_with_event_pages(non_mve_pages=[])

        def endless_mve(limit=None, cursor=None):
            # Cursor always set, ticker never found — an unbounded listing
            return SimpleNamespace(
                events=[SimpleNamespace(event_ticker="OTHER", title="x")],
                cursor="NEXT",
            )
        client.get_multivariate_events = MagicMock(side_effect=endless_mve)
        fallback = _patch_single_event_lookups(
            monkeypatch, single_lookups={"DEEP-1": "Deep Title"},
        )
        result = historical._load_or_build_event_titles(client, {"DEEP-1"})
        assert result == {"DEEP-1": "Deep Title"}
        assert client.get_multivariate_events.call_count == MVE_TITLE_LOOKUP_MAX_PAGES
        assert fallback.call_count == 1

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


def _raw_market_dict(**overrides) -> dict:
    """A raw market JSON dict as the current API sends it (ISO time strings,
    *_dollars price strings, no legacy integer fields)."""
    m = {
        "ticker": "T1",
        "event_ticker": "E1",
        "title": "Market Title",
        "result": "yes",
        "yes_ask_dollars": "0.45",
        "no_ask_dollars": "0.55",
        "yes_bid_dollars": "0.43",
        "open_time": "2024-12-01T00:00:00Z",
        "close_time": "2025-01-01T00:00:00Z",
        "settlement_ts": "2025-01-02T00:00:00Z",
        "status": "finalized",
    }
    m.update(overrides)
    return m


class TestMarketToDict:
    def test_carries_event_title(self):
        d = historical._market_to_dict(_raw_market_dict(), "Some Event Title")
        assert d["event_title"] == "Some Event Title"
        assert d["event_ticker"] == "E1"
        assert d["title"] == "Market Title"
        assert d["close_time"] == "2025-01-01T00:00:00Z"

    def test_event_title_defaults_to_empty_when_missing(self):
        # No event title resolved
        d = historical._market_to_dict(_raw_market_dict())
        assert d["event_title"] == ""

    def test_missing_subtitle_maps_to_none(self):
        # The current API omits subtitle entirely — the cached dict must carry
        # None, matching what the old SDK-model path produced
        d = historical._market_to_dict(_raw_market_dict())
        assert d["subtitle"] is None

    def test_carries_open_time(self):
        # open_time feeds backtester._can_ever_enter()'s eligibility prefilter
        d = historical._market_to_dict(_raw_market_dict())
        assert d["open_time"] == "2024-12-01T00:00:00Z"

    def test_missing_open_time_maps_to_none(self):
        # Pre-existing cache files / payloads without open_time must not crash
        # the pipeline — _can_ever_enter treats None as "can't prove ineligibility"
        raw = _raw_market_dict()
        del raw["open_time"]
        d = historical._market_to_dict(raw)
        assert d["open_time"] is None


class TestFetchAllSettledMarkets:
    def test_archive_early_stop_and_dict_output(self, tmp_path, monkeypatch):
        # The /historical/markets archive ignores settlement-time filters and
        # is ordered newest-first; pagination must stop once a page predates
        # start_date instead of walking the multi-million-market archive.
        # Also verifies raw dicts flow through to the cached output format.
        monkeypatch.setattr(historical, "CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr(historical, "INCLUDE_MVE_MARKETS", False)
        from datetime import date

        def market(tkr, settled):
            return {"ticker": tkr, "event_ticker": f"EV-{tkr}", "title": f"Q {tkr}",
                    "result": "yes", "yes_ask_dollars": "0.40", "no_ask_dollars": "0.60",
                    "yes_bid_dollars": "0.38", "close_time": settled,
                    "settlement_ts": settled, "status": "finalized"}

        # Archive pages newest-first: page1 in-window, page2 predates start_date
        # entirely (cursor still set — the early stop must ignore it), then the
        # live endpoint returns one post-cutoff market.
        pages = {
            "cutoff": {"market_settled_ts": "2026-03-01T00:00:00Z"},
            "hist1": {"markets": [market("IN-WINDOW", "2026-02-15T00:00:00Z")], "cursor": "C2"},
            "hist2": {"markets": [market("TOO-OLD", "2026-01-01T00:00:00Z")], "cursor": "C3"},
        }
        calls = {"hist": 0}

        def fake_signed_get(client, path, **params):
            if path.endswith("/historical/cutoff"):
                return _raw_resp(pages["cutoff"])
            assert path.endswith("/historical/markets")
            calls["hist"] += 1
            return _raw_resp(pages["hist1"] if calls["hist"] == 1 else pages["hist2"])

        monkeypatch.setattr(historical, "_signed_raw_get", fake_signed_get)
        live = MagicMock()
        live.get_markets_without_preload_content = MagicMock(return_value=_raw_resp(
            {"markets": [market("RECENT", "2026-03-05T00:00:00Z")], "cursor": None}
        ))

        out = historical.fetch_all_settled_markets(
            MagicMock(), live, start_date=date(2026, 2, 1), use_cache=False,
        )
        # Early stop: page 2 (all pre-start) is fetched, detected, and pagination
        # halts even though its cursor points at a page 3
        assert calls["hist"] == 2
        tickers = {m["ticker"] for m in out}
        assert tickers == {"IN-WINDOW", "RECENT"}
        # Output dicts carry the exact cached format
        m = next(mm for mm in out if mm["ticker"] == "IN-WINDOW")
        assert m["yes_ask_dollars"] == "0.40"
        assert m["settlement_ts"] == "2026-02-15T00:00:00Z"
        assert m["event_title"] == ""


def _patch_candle_fetch(monkeypatch, ts, yes_ask="0.55", yes_bid="0.53",
                        legacy_format=False) -> MagicMock:
    """Patch _signed_raw_get to serve one raw candlestick page.

    legacy_format=True emits the pre-drift shape (close_dollars string beside
    an integer-cent close); the default emits the current shape (the dollar
    string IS close). Both must parse to dollars.
    """
    if legacy_format:
        ya = {"close": 55, "close_dollars": yes_ask}
        yb = {"close": 53, "close_dollars": yes_bid}
    else:
        ya = {"close": yes_ask}
        yb = {"close": yes_bid}
    payload = {"candlesticks": [{"end_period_ts": ts, "yes_ask": ya, "yes_bid": yb}]}
    mock = MagicMock(return_value=_raw_resp(payload))
    monkeypatch.setattr(historical, "_signed_raw_get", mock)
    return mock


class TestFetchDailyCandlesticks:
    def test_parses_current_dollar_string_close(self, tmp_path, monkeypatch):
        # Current wire format: yes_ask.close is the fixed-point DOLLAR string
        monkeypatch.setattr(historical, "_CANDLES_DIR", tmp_path / "candles")
        _patch_candle_fetch(monkeypatch, 1_700_000_000)
        out = historical.fetch_daily_candlesticks(
            MagicMock(), "T1", open_ts=0, close_ts=2, use_cache=False, rate_limit_sleep=0.0,
        )
        assert out == [{
            "ts": 1_700_000_000,
            "yes_ask_close": pytest.approx(0.55),
            "no_ask_close": pytest.approx(0.47),  # 1 - yes_bid 0.53
        }]

    def test_legacy_close_dollars_still_preferred(self, tmp_path, monkeypatch):
        # Pre-drift shape: close_dollars string beside integer-cent close.
        # Regression: reading the cent int fed 1–99 values into the
        # dollar-denominated backtest filters, silently rejecting every candle.
        monkeypatch.setattr(historical, "_CANDLES_DIR", tmp_path / "candles")
        _patch_candle_fetch(monkeypatch, 1_700_000_000, legacy_format=True)
        out = historical.fetch_daily_candlesticks(
            MagicMock(), "T1", open_ts=0, close_ts=2, use_cache=False, rate_limit_sleep=0.0,
        )
        assert out[0]["yes_ask_close"] == pytest.approx(0.55)
        assert out[0]["no_ask_close"] == pytest.approx(0.47)

    def test_cache_hit_skips_api_when_window_covered(self, tmp_path, monkeypatch):
        # A second request for a window already covered by the cached window
        # must not hit the API again.
        monkeypatch.setattr(historical, "_CANDLES_DIR", tmp_path / "candles")
        _patch_candle_fetch(monkeypatch, 1_700_000_000)
        historical.fetch_daily_candlesticks(
            MagicMock(), "T1", open_ts=100, close_ts=200, rate_limit_sleep=0.0,
        )
        fetch2 = _patch_candle_fetch(monkeypatch, 1_700_000_000)
        out = historical.fetch_daily_candlesticks(
            MagicMock(), "T1", open_ts=150, close_ts=180, rate_limit_sleep=0.0,
        )
        assert fetch2.call_count == 0
        assert out[0]["ts"] == 1_700_000_000

    def test_narrower_cached_window_forces_refetch(self, tmp_path, monkeypatch):
        # Regression: a cache built for a LATER backtest start_date (narrower
        # open_ts window) must not be silently reused for an EARLIER
        # start_date — the cached window doesn't cover the newly requested
        # (wider) range, so it's missing candles the caller actually needs.
        monkeypatch.setattr(historical, "_CANDLES_DIR", tmp_path / "candles")
        _patch_candle_fetch(monkeypatch, 1_700_000_000)
        historical.fetch_daily_candlesticks(
            MagicMock(), "T1", open_ts=500, close_ts=1000, rate_limit_sleep=0.0,
        )
        # New request starts EARLIER than the cached window — must refetch
        fetch2 = _patch_candle_fetch(monkeypatch, 1_700_000_000)
        historical.fetch_daily_candlesticks(
            MagicMock(), "T1", open_ts=100, close_ts=1000, rate_limit_sleep=0.0,
        )
        assert fetch2.call_count == 1

    def test_legacy_bare_list_cache_is_migrated(self, tmp_path, monkeypatch):
        # Cache files written before the windowed-cache fix are a bare list
        # with no window metadata. They must not be trusted blindly (we can't
        # confirm what range they cover) — the next fetch should refetch and
        # migrate the file to the tagged dict format.
        candles_dir = tmp_path / "candles"
        monkeypatch.setattr(historical, "_CANDLES_DIR", candles_dir)
        candles_dir.mkdir(parents=True)
        (candles_dir / "T1.json").write_text(
            '[{"ts": 1, "yes_ask_close": 0.5, "no_ask_close": 0.5}]'
        )
        fetch = _patch_candle_fetch(monkeypatch, 1_700_000_000)
        out = historical.fetch_daily_candlesticks(
            MagicMock(), "T1", open_ts=100, close_ts=200, rate_limit_sleep=0.0,
        )
        assert fetch.call_count == 1
        assert out[0]["ts"] == 1_700_000_000

    def test_use_cache_false_still_persists_fetch_to_disk(self, tmp_path, monkeypatch):
        # Regression: use_cache=False (--no-cache) must still refresh the disk
        # cache file, or the whole point of forcing a fresh pull is defeated —
        # the very next default (cached) run would keep loading stale data.
        monkeypatch.setattr(historical, "_CANDLES_DIR", tmp_path / "candles")
        _patch_candle_fetch(monkeypatch, 1_700_000_000)
        historical.fetch_daily_candlesticks(
            MagicMock(), "T1", open_ts=100, close_ts=200, use_cache=False, rate_limit_sleep=0.0,
        )
        fetch2 = _patch_candle_fetch(monkeypatch, 9_999_999_999)
        out = historical.fetch_daily_candlesticks(
            MagicMock(), "T1", open_ts=100, close_ts=200, use_cache=True, rate_limit_sleep=0.0,
        )
        assert fetch2.call_count == 0
        assert out[0]["ts"] == 1_700_000_000
