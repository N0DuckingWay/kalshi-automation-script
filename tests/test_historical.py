"""Tests for historical.py — event-title cache and dict serialization."""
import gzip
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
        # is paged newest-first; the sequential walk must stop once a page
        # predates start_date instead of walking the multi-million-market
        # archive. These fake pages use opaque cursors and records without
        # created_time, so the sharded path's synthesis check fails and the
        # fetch exercises the sequential fallback — the path this early-stop
        # rule lives on. Also verifies dicts flow through to the cached format.
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
            return _raw_resp(pages["hist2"] if params.get("cursor") == "C2" else pages["hist1"])

        monkeypatch.setattr(historical, "_signed_raw_get", fake_signed_get)
        live = MagicMock()
        live.get_markets_without_preload_content = MagicMock(return_value=_raw_resp(
            {"markets": [market("RECENT", "2026-03-05T00:00:00Z")], "cursor": None}
        ))

        out = historical.fetch_all_settled_markets(
            MagicMock(), live, start_date=date(2026, 2, 1), use_cache=False,
        )
        # 3 archive calls: 1 synthesis probe (fails — opaque cursor), then the
        # sequential walk's page 1 and page 2. Early stop: page 2 (all
        # pre-start) is fetched, detected, and pagination halts even though
        # its cursor points at a page 3.
        assert calls["hist"] == 3
        tickers = {m["ticker"] for m in out}
        assert tickers == {"IN-WINDOW", "RECENT"}
        # Output dicts carry the exact cached format
        m = next(mm for mm in out if mm["ticker"] == "IN-WINDOW")
        assert m["yes_ask_dollars"] == "0.40"
        assert m["settlement_ts"] == "2026-02-15T00:00:00Z"
        assert m["event_title"] == ""

    def test_live_sweep_bounds_min_settled_ts_to_start_date(self, tmp_path, monkeypatch):
        # Regression: min_settled_ts used to be hardcoded to cutoff_ts, so a
        # narrow recent start_date still forced the live endpoint to walk the
        # ENTIRE [cutoff_ts, now) range server-side (observed: 20k+ pages,
        # 20M+ records scanned just to reach a one-week window) even though
        # the server honors min_settled_ts and could narrow it directly. When
        # start_date is LATER than the API cutoff, no live window may reach
        # below start_ts (the windowed sweep issues one call per settled-day,
        # so the assertion covers the minimum across all calls).
        from datetime import UTC, date, datetime

        monkeypatch.setattr(historical, "CACHE_DIR", tmp_path / "cache")

        def fake_signed_get(client, path, **params):
            if path.endswith("/historical/cutoff"):
                return _raw_resp({"market_settled_ts": "2026-03-01T00:00:00Z"})
            assert path.endswith("/historical/markets")
            return _raw_resp({"markets": [], "cursor": None})

        monkeypatch.setattr(historical, "_signed_raw_get", fake_signed_get)
        live = MagicMock()
        live.get_markets_without_preload_content = MagicMock(
            return_value=_raw_resp({"markets": [], "cursor": None})
        )

        start_date = date(2026, 7, 6)
        historical.fetch_all_settled_markets(
            MagicMock(), live, start_date=start_date, use_cache=False,
        )
        expected_min_ts = int(datetime(2026, 7, 6, tzinfo=UTC).timestamp())
        seen_min_ts = [kwargs["min_settled_ts"] for _, kwargs
                       in live.get_markets_without_preload_content.call_args_list]
        assert min(seen_min_ts) == expected_min_ts

    def test_live_sweep_min_settled_ts_falls_back_to_cutoff_when_later(self, tmp_path, monkeypatch):
        # When start_date is EARLIER than the API cutoff (the common case —
        # default start_date is 2024-01-01), the live sweep must still start
        # at cutoff_ts, not start_date — the archive already covers everything
        # before cutoff, and the live endpoint does not even serve pre-cutoff
        # settlements (they migrate to the archive; live-verified 2026-07-13).
        from datetime import UTC, date, datetime

        monkeypatch.setattr(historical, "CACHE_DIR", tmp_path / "cache")

        def fake_signed_get(client, path, **params):
            if path.endswith("/historical/cutoff"):
                return _raw_resp({"market_settled_ts": "2026-03-01T00:00:00Z"})
            assert path.endswith("/historical/markets")
            return _raw_resp({"markets": [], "cursor": None})

        monkeypatch.setattr(historical, "_signed_raw_get", fake_signed_get)
        live = MagicMock()
        live.get_markets_without_preload_content = MagicMock(
            return_value=_raw_resp({"markets": [], "cursor": None})
        )

        historical.fetch_all_settled_markets(
            MagicMock(), live, start_date=date(2024, 1, 1), use_cache=False,
        )
        expected_min_ts = int(datetime(2026, 3, 1, tzinfo=UTC).timestamp())
        seen_min_ts = [kwargs["min_settled_ts"] for _, kwargs
                       in live.get_markets_without_preload_content.call_args_list]
        assert min(seen_min_ts) == expected_min_ts


# ─── Sharded-fetch fakes ──────────────────────────────────────────────────────
#
# _FakeArchive reproduces the real /historical/markets contract established by
# live probing (2026-07-13): records paged in (created_time DESC, ticker DESC)
# order behind a protobuf keyset cursor of the last record's position, with
# every time-filter param ignored. _FakeLive reproduces /markets?status=settled:
# settle-DESC ordering with min/max_settled_ts honored server-side. Together
# they let the sharded fetch (cursor synthesis, day slicing, windowing,
# fallbacks) be exercised entirely offline.

def _mk_raw_market(tkr, created, settled, result="yes", **overrides):
    m = {"ticker": tkr, "event_ticker": f"EV-{tkr}", "title": f"Q {tkr}",
         "result": result, "yes_ask_dollars": "0.40", "no_ask_dollars": "0.60",
         "yes_bid_dollars": "0.38", "created_time": created, "open_time": created,
         "close_time": settled, "settlement_ts": settled, "status": "finalized"}
    m.update(overrides)
    return m


def _created_key(m):
    parts = historical._iso_epoch_parts(m["created_time"])
    return (parts[0] + parts[1] / 1e9, m["ticker"])


class _FakeArchive:
    """In-memory /historical/markets: (created DESC, ticker DESC) keyset pages."""

    def __init__(self, markets, page_size=3, opaque_cursors=False, fail_after=None):
        self.markets = sorted(markets, key=_created_key, reverse=True)
        self.page_size = page_size
        self.opaque = opaque_cursors
        self.fail_after = fail_after  # raise RuntimeError after N page calls
        self.calls = 0

    def page(self, cursor=None, **_):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("simulated network failure")
        idx = 0
        if cursor:
            if self.opaque:
                idx = int(cursor[1:])
            else:
                seconds, nanos, ticker = historical._decode_archive_cursor(cursor)
                cursor_key = (seconds + nanos / 1e9, ticker)
                # Keyset: first record strictly AFTER the cursor position in
                # descending order, i.e. with a smaller (created, ticker) key.
                idx = len(self.markets)
                for i, m in enumerate(self.markets):
                    if _created_key(m) < cursor_key:
                        idx = i
                        break
        page = self.markets[idx: idx + self.page_size]
        nxt = None
        if page and idx + self.page_size < len(self.markets):
            if self.opaque:
                nxt = f"@{idx + self.page_size}"
            else:
                last = page[-1]
                sec, nanos = historical._iso_epoch_parts(last["created_time"])
                nxt = historical._encode_archive_cursor(sec, nanos, last["ticker"])
        return {"markets": page, "cursor": nxt}


class _FakeLive:
    """In-memory /markets?status=settled honoring min/max_settled_ts."""

    def __init__(self, markets, page_size=3, ignore_max=False):
        self.markets = sorted(
            markets, key=lambda m: historical._iso_epoch(m["settlement_ts"]),
            reverse=True,
        )
        self.page_size = page_size
        self.ignore_max = ignore_max
        self.calls = 0

    def get_markets_without_preload_content(self, min_settled_ts=None,
                                            max_settled_ts=None, cursor=None, **_):
        self.calls += 1
        if self.ignore_max:
            max_settled_ts = None
        pool = [
            m for m in self.markets
            if (min_settled_ts is None
                or historical._iso_epoch(m["settlement_ts"]) >= min_settled_ts)
            and (max_settled_ts is None
                 or historical._iso_epoch(m["settlement_ts"]) <= max_settled_ts)
        ]
        idx = int(cursor) if cursor else 0
        page = pool[idx: idx + self.page_size]
        nxt = str(idx + self.page_size) if page and idx + self.page_size < len(pool) else None
        return _raw_resp({"markets": page, "cursor": nxt})


def _install_sharded_fakes(monkeypatch, tmp_path, archive, cutoff_iso):
    """Wire a _FakeArchive behind _signed_raw_get, isolate CACHE_DIR, and stub
    out event-title resolution (network-only concern, tested separately)."""
    monkeypatch.setattr(historical, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(historical, "_load_or_build_event_titles",
                        lambda *a, **k: {})

    def fake_signed_get(client, path, **params):
        if path.endswith("/historical/cutoff"):
            return _raw_resp({"market_settled_ts": cutoff_iso})
        assert path.endswith("/historical/markets")
        return _raw_resp(archive.page(**params))

    monkeypatch.setattr(historical, "_signed_raw_get", fake_signed_get)


def _old_semantics_expected(archive_markets, live_markets, start_ts, cutoff_ts,
                            page_size=3):
    """Oracle: the ticker set the ORIGINAL sequential implementation returns —
    walk the archive newest-first with the settlement early-stop rule, then
    sweep the live endpoint from max(cutoff_ts, start_ts)."""
    expected = set()
    walk = sorted(archive_markets, key=_created_key, reverse=True)
    for page_start in range(0, len(walk), page_size):
        page = walk[page_start: page_start + page_size]
        for m in page:
            settle = historical._iso_epoch(m["settlement_ts"])
            if (m["result"] in ("yes", "no") and settle is not None
                    and start_ts <= settle < cutoff_ts):
                expected.add(m["ticker"])
        newest = historical._iso_epoch(page[0]["settlement_ts"])
        if newest is not None and newest < start_ts:
            break
    live_min = max(cutoff_ts, start_ts)
    for m in live_markets:
        settle = historical._iso_epoch(m["settlement_ts"])
        if m["result"] in ("yes", "no") and settle is not None and settle >= live_min:
            expected.add(m["ticker"])
    return expected


class TestShardedFetch:
    """The parallel day-sliced fetch must return the same market set as the
    original sequential walks, resume from day-slice files, and degrade to the
    sequential paths whenever a runtime self-check fails."""

    START = "2026-06-05"
    CUTOFF = "2026-06-10T00:00:00Z"

    @staticmethod
    def _ts(iso):
        from datetime import datetime
        return int(datetime.fromisoformat(iso).timestamp())

    def _fixture_markets(self):
        # Archive spread over several created-days with edge cases:
        # multiple pages per day, a created_time tie, a non-binary result, a
        # record with no settlement_ts, a long-lived market created BEFORE
        # start_date settling inside the window (tail territory), and
        # fast-settled pre-start markets that trigger the early stop.
        archive = [
            # day 2026-06-09 (top day, 4 records → 2 pages at page_size 3)
            _mk_raw_market("A1", "2026-06-09T20:00:00.500000Z", "2026-06-09T22:00:00Z"),
            _mk_raw_market("A2", "2026-06-09T20:00:00.500000Z", "2026-06-09T21:00:00Z"),
            _mk_raw_market("A3", "2026-06-09T10:00:00Z", "2026-06-09T12:00:00Z"),
            _mk_raw_market("VOID", "2026-06-09T09:00:00Z", "2026-06-09T11:00:00Z",
                           result="void"),
            # day 2026-06-08
            _mk_raw_market("B1", "2026-06-08T15:00:00Z", "2026-06-08T18:00:00Z"),
            _mk_raw_market("NOSETTLE", "2026-06-08T14:00:00Z", "2026-06-08T16:00:00Z",
                           settlement_ts=None),
            # day 2026-06-07 (empty), day 2026-06-06
            _mk_raw_market("C1", "2026-06-06T08:00:00Z", "2026-06-06T09:00:00Z"),
            # day 2026-06-05 (bottom slice, record at the exact day boundary)
            _mk_raw_market("D1", "2026-06-05T00:00:00Z", "2026-06-05T02:00:00Z"),
            # tail: created before start_date but settled inside the window
            _mk_raw_market("LONGLIVED", "2026-06-04T23:00:00Z", "2026-06-06T10:00:00Z"),
            # pre-start fast markets: settle before start → early stop fodder
            _mk_raw_market("OLD1", "2026-06-04T20:00:00Z", "2026-06-04T21:00:00Z"),
            _mk_raw_market("OLD2", "2026-06-04T10:00:00Z", "2026-06-04T11:00:00Z"),
            _mk_raw_market("OLD3", "2026-06-03T10:00:00Z", "2026-06-03T11:00:00Z"),
            _mk_raw_market("OLD4", "2026-06-02T10:00:00Z", "2026-06-02T11:00:00Z"),
        ]
        # Live: post-cutoff settles across two days (frontier day is empty)
        live = [
            _mk_raw_market("L1", "2026-06-10T01:00:00Z", "2026-06-10T03:00:00Z"),
            _mk_raw_market("L2", "2026-06-10T04:00:00Z", "2026-06-10T06:00:00Z"),
            _mk_raw_market("L3", "2026-06-11T01:00:00Z", "2026-06-11T02:00:00Z"),
            _mk_raw_market("LVOID", "2026-06-11T03:00:00Z", "2026-06-11T04:00:00Z",
                           result="void"),
        ]
        return archive, live

    def _run(self, monkeypatch, tmp_path, archive, live):
        from datetime import date
        _install_sharded_fakes(monkeypatch, tmp_path, archive, self.CUTOFF)
        return historical.fetch_all_settled_markets(
            MagicMock(), live, start_date=date(2026, 6, 5), use_cache=False,
        )

    def test_sharded_matches_sequential_semantics(self, tmp_path, monkeypatch):
        archive_markets, live_markets = self._fixture_markets()
        archive = _FakeArchive(archive_markets)
        live = _FakeLive(live_markets)
        out = self._run(monkeypatch, tmp_path, archive, live)

        expected = _old_semantics_expected(
            archive_markets, live_markets,
            self._ts(self.START + "T00:00:00+00:00"), self._ts(self.CUTOFF),
        )
        assert {m["ticker"] for m in out} == expected
        assert "LONGLIVED" in expected  # the tail case is actually exercised
        # No duplicate tickers despite deliberately overlapping slice boundaries
        assert len(out) == len({m["ticker"] for m in out})
        # Compact dict format survives the day store round-trip
        a1 = next(m for m in out if m["ticker"] == "A1")
        assert a1["open_time"] == "2026-06-09T20:00:00.500000Z"
        assert a1["yes_ask_dollars"] == "0.40"

    def test_second_run_reuses_day_slices(self, tmp_path, monkeypatch):
        archive_markets, live_markets = self._fixture_markets()
        archive = _FakeArchive(archive_markets)
        live = _FakeLive(live_markets)
        out1 = self._run(monkeypatch, tmp_path, archive, live)
        cold_archive_calls, cold_live_calls = archive.calls, live.calls

        archive.calls = live.calls = 0
        out2 = self._run(monkeypatch, tmp_path, archive, live)
        assert {m["ticker"] for m in out2} == {m["ticker"] for m in out1}
        # Run 2 skips every stored day slice: archive pays only the synthesis
        # probe + the tail walk; live pays only the frontier window.
        assert archive.calls < cold_archive_calls
        assert live.calls < cold_live_calls
        assert (tmp_path / "cache" / "archive_days").exists()
        assert (tmp_path / "cache" / "live_days").exists()

    def test_interrupted_run_resumes_from_day_slices(self, tmp_path, monkeypatch):
        # Serialize the workers so the interruption point is deterministic:
        # synthesis probe (1 call) + top day slice (2 pages) complete, then
        # the next slice's first call dies.
        monkeypatch.setattr(historical, "SETTLED_FETCH_MAX_WORKERS", 1)
        archive_markets, live_markets = self._fixture_markets()

        # Baseline: how many archive pages a cold, uninterrupted run costs.
        cold_archive = _FakeArchive(archive_markets)
        self._run(monkeypatch, tmp_path / "cold", cold_archive, _FakeLive(live_markets))
        cold_calls = cold_archive.calls

        archive = _FakeArchive(archive_markets, fail_after=3)
        live = _FakeLive(live_markets)
        with pytest.raises(RuntimeError):
            self._run(monkeypatch, tmp_path / "warm", archive, live)
        saved_before_crash = list(
            (tmp_path / "warm" / "cache" / "archive_days").glob("*.json.gz"))
        assert saved_before_crash  # at least one completed slice persisted

        # Retry without the fault: completed slices are reused, result is whole.
        archive.fail_after = None
        archive.calls = 0
        out = self._run(monkeypatch, tmp_path / "warm", archive, live)
        expected = _old_semantics_expected(
            archive_markets, live_markets,
            self._ts(self.START + "T00:00:00+00:00"), self._ts(self.CUTOFF),
        )
        assert {m["ticker"] for m in out} == expected
        # The resumed run skips the slice(s) persisted before the crash, so it
        # refetches strictly fewer archive pages than the cold run did.
        assert archive.calls < cold_calls

    def test_opaque_cursor_falls_back_to_sequential(self, tmp_path, monkeypatch):
        # If the cursor format drifts, the synthesis probe must fail closed:
        # no day slicing, no synthesized jumps — just the original walk.
        archive_markets, live_markets = self._fixture_markets()
        archive = _FakeArchive(archive_markets, opaque_cursors=True)
        live = _FakeLive(live_markets)
        out = self._run(monkeypatch, tmp_path, archive, live)
        expected = _old_semantics_expected(
            archive_markets, live_markets,
            self._ts(self.START + "T00:00:00+00:00"), self._ts(self.CUTOFF),
        )
        assert {m["ticker"] for m in out} == expected
        # The sequential path must not fabricate day-slice files
        assert not (tmp_path / "cache" / "archive_days").exists()

    def test_cutoff_advance_invalidates_archive_slices(self, tmp_path, monkeypatch):
        # When Kalshi advances the archive cutoff, markets that settled in the
        # gap MIGRATE from the live endpoint into the archive (and stop being
        # served by the live endpoint — verified 2026-07-13). Day-slice files
        # stamped with the old cutoff would silently miss them, so they must
        # be refetched, not reused.
        from datetime import date
        archive_markets, live_markets = self._fixture_markets()
        archive = _FakeArchive(archive_markets)
        live = _FakeLive(live_markets)
        out1 = self._run(monkeypatch, tmp_path, archive, live)
        assert "L1" in {m["ticker"] for m in out1}  # served by live pre-advance

        # Advance the cutoff past 2026-06-11: L1/L2/L3 migrate to the archive.
        new_cutoff = "2026-06-12T00:00:00Z"
        migrated = _FakeArchive(archive_markets + live_markets)
        empty_live = _FakeLive([])
        _install_sharded_fakes(monkeypatch, tmp_path, migrated, new_cutoff)
        out2 = historical.fetch_all_settled_markets(
            MagicMock(), empty_live, start_date=date(2026, 6, 5), use_cache=False,
        )
        assert {m["ticker"] for m in out2} == _old_semantics_expected(
            archive_markets + live_markets, [],
            self._ts(self.START + "T00:00:00+00:00"), self._ts(new_cutoff),
        )
        # The migrated markets must come from the refetched archive slices
        assert {"L1", "L2", "L3"} <= {m["ticker"] for m in out2}

    def test_assembly_streams_slices_from_disk(self, tmp_path, monkeypatch):
        # Peak memory must not scale with the number of days fetched, so no
        # phase may retain slice records: every day — including ones fetched
        # moments earlier in this same run — is re-read from its file at
        # assembly. Verified by counting _day_store_load calls per path.
        archive_markets, live_markets = self._fixture_markets()
        real_load = historical._day_store_load
        loads: list[str] = []

        def counting_load(path, expect_meta):
            result = real_load(path, expect_meta)
            if result is not None:
                loads.append(str(path))
            return result

        monkeypatch.setattr(historical, "_day_store_load", counting_load)
        out = self._run(monkeypatch, tmp_path, _FakeArchive(archive_markets),
                        _FakeLive(live_markets))
        assert out  # sanity: the run actually produced records

        # Cold run: every successful load is an assembly read (the prescan
        # found nothing on disk), so each written slice is read exactly once.
        written = {str(p) for p in
                   (tmp_path / "cache").glob("*_days/*.json.gz")}
        assert written, "expected day slices to have been persisted"
        assert set(loads) == written
        assert len(loads) == len(written)

    def test_live_ignoring_max_settled_ts_falls_back(self, tmp_path, monkeypatch):
        # If the live endpoint stops honoring max_settled_ts, every window
        # would silently re-walk the whole range; the first window detects it
        # and the phase degrades to the original single sequential sweep.
        archive_markets, live_markets = self._fixture_markets()
        archive = _FakeArchive(archive_markets)
        live = _FakeLive(live_markets, ignore_max=True)
        out = self._run(monkeypatch, tmp_path, archive, live)
        expected = _old_semantics_expected(
            archive_markets, live_markets,
            self._ts(self.START + "T00:00:00+00:00"), self._ts(self.CUTOFF),
        )
        assert {m["ticker"] for m in out} == expected


class TestArchiveCursorCodec:
    def test_roundtrip(self):
        cursor = historical._encode_archive_cursor(
            1778713796, 165186000, "KXMVECROSSCATEGORY-S2026AED5FC84CB7-70D7C8ECAC6",
        )
        assert historical._decode_archive_cursor(cursor) == (
            1778713796, 165186000, "KXMVECROSSCATEGORY-S2026AED5FC84CB7-70D7C8ECAC6",
        )

    def test_known_bytes(self):
        # Hand-computed protobuf: field 1 = Timestamp{seconds=1} (2-byte nested
        # message 08 01), field 2 = "A" → 0a 02 08 01 12 01 41 → base64url
        # "CgIIARIBQQ" (padding stripped).
        assert historical._encode_archive_cursor(1, 0, "A") == "CgIIARIBQQ"

    def test_zero_nanos_omitted(self):
        # Synthesized boundary cursors carry nanos=0, which protobuf encoders
        # omit; the decoder must default it back to 0.
        cursor = historical._encode_archive_cursor(1_750_000_000, 0, "TICK")
        assert historical._decode_archive_cursor(cursor) == (1_750_000_000, 0, "TICK")

    def test_garbage_cursor_returns_none(self):
        assert historical._decode_archive_cursor("!!not-base64!!") is None

    def test_negative_varint_raises_instead_of_hanging(self):
        # Python's arithmetic right shift never carries a negative value to 0,
        # so without the guard a negative input (e.g. a pre-1970 timestamp)
        # would spin the encoding loop forever inside a fetch worker.
        with pytest.raises(ValueError):
            historical._pb_varint(-1)
        with pytest.raises(ValueError):
            historical._encode_archive_cursor(-100, 0, "T")

    def test_iso_epoch_parts_microsecond_to_nanos(self):
        seconds, nanos = historical._iso_epoch_parts("2026-05-13T23:09:56.165186Z")
        assert nanos == 165186000
        from datetime import UTC, datetime
        assert seconds == int(datetime(2026, 5, 13, 23, 9, 56, tzinfo=UTC).timestamp())


class TestDayStore:
    def test_meta_mismatch_rejects_file(self, tmp_path):
        path = tmp_path / "2026-06-09.json.gz"
        meta = {"kind": "archive_created_day", "cutoff_ts": 100,
                "include_mve": True, "complete": True}
        historical._day_store_save(path, meta, [{"ticker": "T1"}])
        assert historical._day_store_load(path, meta) == [{"ticker": "T1"}]
        # Any drifted expectation (advanced cutoff, flipped MVE flag) → refetch
        assert historical._day_store_load(path, {**meta, "cutoff_ts": 200}) is None
        assert historical._day_store_load(path, {**meta, "include_mve": False}) is None

    def test_missing_or_corrupt_file(self, tmp_path):
        path = tmp_path / "2026-06-09.json.gz"
        assert historical._day_store_load(path, {}) is None
        path.write_bytes(b"not gzip")
        assert historical._day_store_load(path, {}) is None

    def test_day_store_roundtrip_interoperates_with_stdlib_json(self, tmp_path):
        # The store may be written by orjson (optional `perf` extra) or the
        # stdlib. Both must produce files the other can read, or installing /
        # removing orjson would silently invalidate every cached day slice.
        meta = {"kind": "archive_created_day", "cutoff_ts": 100,
                "include_mve": True, "complete": True}
        markets = [{"ticker": "T1", "result": "yes", "settlement_ts": "2026-06-09T00:00:00Z"},
                   {"ticker": "T2", "result": "no", "open_time": None}]

        # Whatever _day_store_save used, plain stdlib json must read it back.
        written = tmp_path / "written.json.gz"
        historical._day_store_save(written, meta, markets)
        with gzip.open(written, "rt", encoding="utf-8") as fh:
            assert json.load(fh) == {"meta": meta, "markets": markets}

        # And a slice hand-written by the stdlib must load through the shim —
        # this is the contract that keeps pre-existing caches on disk usable.
        legacy = tmp_path / "legacy.json.gz"
        with gzip.open(legacy, "wt", encoding="utf-8") as fh:
            json.dump({"meta": meta, "markets": markets}, fh)
        assert historical._day_store_load(legacy, meta) == markets


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


class TestFetchCandlesticks:
    def test_parses_current_dollar_string_close(self, tmp_path, monkeypatch):
        # Current wire format: yes_ask.close is the fixed-point DOLLAR string
        monkeypatch.setattr(historical, "_CANDLES_DIR", tmp_path / "candles")
        _patch_candle_fetch(monkeypatch, 1_700_000_000)
        out = historical.fetch_candlesticks(
            MagicMock(), "T1", open_ts=0, close_ts=2, use_cache=False, rate_limit_sleep=0.0,
        )
        assert out == [{
            "ts": 1_700_000_000,
            "yes_ask_close": pytest.approx(0.55),
            "no_ask_close": pytest.approx(0.47),  # 1 - yes_bid 0.53
        }]

    def test_requests_hourly_period_interval(self, tmp_path, monkeypatch):
        # Regression: daily granularity (period_interval=1440) only emits a
        # candle for markets whose lifespan crosses a UTC midnight boundary,
        # which silently produced zero data for most short-lived Kalshi
        # markets. Must request CANDLESTICK_PERIOD_INTERVAL_MINUTES (60).
        monkeypatch.setattr(historical, "_CANDLES_DIR", tmp_path / "candles")
        mock = _patch_candle_fetch(monkeypatch, 1_700_000_000)
        historical.fetch_candlesticks(
            MagicMock(), "T1", open_ts=0, close_ts=2, use_cache=False, rate_limit_sleep=0.0,
        )
        _, kwargs = mock.call_args
        assert kwargs["period_interval"] == historical.CANDLESTICK_PERIOD_INTERVAL_MINUTES
        assert historical.CANDLESTICK_PERIOD_INTERVAL_MINUTES == 60

    def test_legacy_close_dollars_still_preferred(self, tmp_path, monkeypatch):
        # Pre-drift shape: close_dollars string beside integer-cent close.
        # Regression: reading the cent int fed 1–99 values into the
        # dollar-denominated backtest filters, silently rejecting every candle.
        monkeypatch.setattr(historical, "_CANDLES_DIR", tmp_path / "candles")
        _patch_candle_fetch(monkeypatch, 1_700_000_000, legacy_format=True)
        out = historical.fetch_candlesticks(
            MagicMock(), "T1", open_ts=0, close_ts=2, use_cache=False, rate_limit_sleep=0.0,
        )
        assert out[0]["yes_ask_close"] == pytest.approx(0.55)
        assert out[0]["no_ask_close"] == pytest.approx(0.47)

    def test_cache_hit_skips_api_when_window_covered(self, tmp_path, monkeypatch):
        # A second request for a window already covered by the cached window
        # must not hit the API again.
        monkeypatch.setattr(historical, "_CANDLES_DIR", tmp_path / "candles")
        _patch_candle_fetch(monkeypatch, 1_700_000_000)
        historical.fetch_candlesticks(
            MagicMock(), "T1", open_ts=100, close_ts=200, rate_limit_sleep=0.0,
        )
        fetch2 = _patch_candle_fetch(monkeypatch, 1_700_000_000)
        out = historical.fetch_candlesticks(
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
        historical.fetch_candlesticks(
            MagicMock(), "T1", open_ts=500, close_ts=1000, rate_limit_sleep=0.0,
        )
        # New request starts EARLIER than the cached window — must refetch
        fetch2 = _patch_candle_fetch(monkeypatch, 1_700_000_000)
        historical.fetch_candlesticks(
            MagicMock(), "T1", open_ts=100, close_ts=1000, rate_limit_sleep=0.0,
        )
        assert fetch2.call_count == 1

    def test_stale_interval_cache_forces_refetch(self, tmp_path, monkeypatch):
        # A cache written under a different period_interval (e.g. an older
        # daily-granularity cache from before this change) must not be reused
        # as if it were hourly — even though its [open_ts, close_ts] window
        # covers the request, the underlying candle spacing doesn't match.
        candles_dir = tmp_path / "candles"
        monkeypatch.setattr(historical, "_CANDLES_DIR", candles_dir)
        candles_dir.mkdir(parents=True)
        (candles_dir / "T1.json").write_text(json.dumps({
            "open_ts": 0, "close_ts": 1000, "period_interval": 1440,
            "candles": [{"ts": 500, "yes_ask_close": 0.5, "no_ask_close": 0.5}],
        }))
        fetch = _patch_candle_fetch(monkeypatch, 1_700_000_000)
        out = historical.fetch_candlesticks(
            MagicMock(), "T1", open_ts=100, close_ts=200, rate_limit_sleep=0.0,
        )
        assert fetch.call_count == 1
        assert out[0]["ts"] == 1_700_000_000

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
        out = historical.fetch_candlesticks(
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
        historical.fetch_candlesticks(
            MagicMock(), "T1", open_ts=100, close_ts=200, use_cache=False, rate_limit_sleep=0.0,
        )
        fetch2 = _patch_candle_fetch(monkeypatch, 9_999_999_999)
        out = historical.fetch_candlesticks(
            MagicMock(), "T1", open_ts=100, close_ts=200, use_cache=True, rate_limit_sleep=0.0,
        )
        assert fetch2.call_count == 0
        assert out[0]["ts"] == 1_700_000_000
