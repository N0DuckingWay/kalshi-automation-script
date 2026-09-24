"""Tests for historical.py — event-title cache and dict serialization."""
import copy
import gzip
import json
import logging
import os
import threading
import weakref
import zlib
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kalshi_betting import historical


def _read_slice_file(path) -> list[dict]:
    """Read a day-slice file's records with no help from historical.py.

    Understands both on-disk formats so slice-content assertions stay
    independent of which writer produced the file (legacy single JSON document
    vs. streamed "jsonl-v1" lines).
    """
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    head = json.loads(lines[0])
    if "markets" in head:
        return json.loads("\n".join(lines))["markets"]
    return [json.loads(line) for line in lines[1:] if line.strip()]


def _raw_resp(payload: dict) -> SimpleNamespace:
    """RESTResponse stand-in for the raw signed-GET path (_signed_raw_get)."""
    return SimpleNamespace(status=200, data=json.dumps(payload).encode("utf-8"))


class _FakeApiException(Exception):
    """Stand-in for the SDK's ApiException, including its five-line __str__.

    The real one renders the status, the reason, the ENTIRE HTTP header dict
    and the body across five lines (~900 bytes) and its own FIRST line is only
    "(404)" — the reason lives on line two. That shape is the whole point of
    TS-02, so it is reproduced here rather than approximated.
    """

    def __init__(self, status: int = 404, reason: str = "Not Found"):
        self.status = status
        self.reason = reason
        self.headers = {"X-Big-Header": "x" * 500}
        self.body = '{"error": {"code": "not_found"}}'
        super().__init__(f"({status})")

    def __str__(self) -> str:
        return (f"({self.status})\n"
                f"Reason: {self.reason}\n"
                f"HTTP response headers: {self.headers}\n"
                f"HTTP response body: {self.body}\n")


def _make_client_with_event_pages(non_mve_pages: list[list[tuple[str, str]]],
                                  mve_pages: list[list[tuple[str, str]]] | None = None):
    """Build a MagicMock client whose raw get_events /
    get_multivariate_events variants return paginated event listings.

    Each page is a list of (event_ticker, title) tuples. The mock walks the pages
    in order until exhausted, then returns an empty page with cursor=None.

    Both listings go through the `*_without_preload_content` raw variants: the
    live API sends `category: null` on some events, which the pinned SDK's
    EventData model (category typed as a required str) rejects with a pydantic
    ValidationError. The `category` key below is deliberately null so these
    fixtures carry the shape that broke the modeled calls.
    """
    mve_pages = mve_pages or []

    def _build_resp(page):
        events = [{"event_ticker": tkr, "title": title, "category": None}
                  for tkr, title in page]
        return _raw_resp({"events": events, "cursor": None})

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

    client.get_events_without_preload_content = MagicMock(side_effect=get_events)
    client.get_multivariate_events_without_preload_content = MagicMock(
        side_effect=get_multivariate_events)
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
    """Redirect the event-title accumulator (event_titles_v2.json) and its
    legacy file (event_titles.json) to temp files so tests don't touch the real
    cache, and return the accumulator's path. tests/conftest.py already applies
    the same redirect to every test; this fixture names the path for the tests
    that seed or read the file."""
    cache_file = tmp_path / "event_titles_v2.json"
    monkeypatch.setattr(historical, "_EVENT_TITLES_CACHE", cache_file)
    monkeypatch.setattr(historical, "_LEGACY_EVENT_TITLES_CACHE",
                        tmp_path / "event_titles.json")
    return cache_file


@pytest.fixture(autouse=True)
def _unpaced_title_lookups(monkeypatch):
    """Zero the per-lookup pause of the event-title fallback (DR-51) for every
    test in this module, so lookups cost no wall time; the pacing test sets its
    own value."""
    monkeypatch.setattr(historical, "EVENT_TITLE_FALLBACK_RATE_LIMIT_SLEEP_SECONDS", 0)


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

    def test_lookup_failure_logs_one_line_without_header_dump(
        self, isolated_cache, monkeypatch, caplog,
    ):
        # TS-02: this warning fires once per unresolved ticker, up to
        # EVENT_TITLE_FALLBACK_MAX_LOOKUPS (5000) of them per run. Logging the
        # SDK exception whole would dump the entire HTTP header dict each time.
        client = _make_client_with_event_pages(non_mve_pages=[], mve_pages=[])

        def fake_signed_get(_client, _path, **_params):
            raise _FakeApiException()

        monkeypatch.setattr(historical, "_signed_raw_get",
                            MagicMock(side_effect=fake_signed_get))
        with caplog.at_level(logging.WARNING):
            result = historical._load_or_build_event_titles(client, {"BAD-1"})

        assert result == {"BAD-1": ""}          # poison pill unchanged
        msgs = [r.getMessage() for r in caplog.records
                if "Could not resolve event title" in r.getMessage()]
        assert len(msgs) == 1
        assert "BAD-1" in msgs[0]
        assert "HTTP 404 Not Found" in msgs[0]
        assert "\n" not in msgs[0]
        assert "X-Big-Header" not in msgs[0]
        assert "HTTP response headers" not in msgs[0]

    def test_mve_bulk_scan_bails_out_on_barren_pages(self, isolated_cache, monkeypatch):
        # The MVE listing is effectively unbounded — a ticker that never
        # appears must not page forever. Two bounds apply: the productivity
        # bail-out (stop once N consecutive pages resolve nothing) and the hard
        # MVE_TITLE_LOOKUP_MAX_PAGES backstop. The bail-out fires first, which
        # matters because live-measured 2026-08-03 the listings kept scanning
        # long after they had stopped resolving anything.
        from kalshi_betting.config import (
            EVENT_TITLE_LISTING_MAX_BARREN_PAGES,
            MVE_TITLE_LOOKUP_MAX_PAGES,
        )

        client = _make_client_with_event_pages(non_mve_pages=[])

        def endless_mve(limit=None, cursor=None):
            # Cursor always set, ticker never found — an unbounded listing
            return _raw_resp({
                "events": [{"event_ticker": "OTHER", "title": "x", "category": None}],
                "cursor": "NEXT",
            })
        client.get_multivariate_events_without_preload_content = MagicMock(
            side_effect=endless_mve)
        fallback = _patch_single_event_lookups(
            monkeypatch, single_lookups={"DEEP-1": "Deep Title"},
        )
        result = historical._load_or_build_event_titles(client, {"DEEP-1"})
        assert result == {"DEEP-1": "Deep Title"}
        pages = client.get_multivariate_events_without_preload_content.call_count
        assert pages == EVENT_TITLE_LISTING_MAX_BARREN_PAGES
        assert pages < MVE_TITLE_LOOKUP_MAX_PAGES  # backstop never needed here
        # Still resolved, via the per-ticker fallback.
        assert fallback.call_count == 1

    def test_status_listing_bails_out_on_barren_pages(self, isolated_cache, monkeypatch):
        # Same bail-out on the settled/closed/open listings. These are an
        # O(all events) scan for a specific ticker set, and get_events EXCLUDES
        # MVE events by API design — so for an MVE-heavy miss set they can
        # never resolve anything and must not page indefinitely (live-measured
        # 2026-08-03: 80,000 events scanned resolved 9 tickers).
        from kalshi_betting.config import EVENT_TITLE_LISTING_MAX_BARREN_PAGES

        client = _make_client_with_event_pages(non_mve_pages=[], mve_pages=[])

        def endless_events(status=None, limit=None, cursor=None):
            return _raw_resp({
                "events": [{"event_ticker": "OTHER", "title": "x", "category": None}],
                "cursor": "NEXT",
            })
        client.get_events_without_preload_content = MagicMock(side_effect=endless_events)
        fallback = _patch_single_event_lookups(
            monkeypatch, single_lookups={"DEEP-1": "Deep Title"},
        )
        result = historical._load_or_build_event_titles(client, {"DEEP-1"})
        assert result == {"DEEP-1": "Deep Title"}
        # One bail-out per status (settled, closed, open) — bounded, not endless.
        assert (client.get_events_without_preload_content.call_count
                == 3 * EVENT_TITLE_LISTING_MAX_BARREN_PAGES)
        assert fallback.call_count == 1

    def test_bulk_listings_use_raw_variants_not_modeled_calls(self, isolated_cache,
                                                              monkeypatch):
        # Regression: the bulk event listings used the MODELED get_events /
        # get_multivariate_events. The live API now sends `category: null`,
        # which the pinned SDK's EventData model (category: required str)
        # rejects with a pydantic ValidationError — observed 2026-08-03 killing
        # a backtest after a 28-minute fetch had already succeeded. Both must
        # go through the raw *_without_preload_content variants, which never
        # touch the response models.
        client = _make_client_with_event_pages(
            non_mve_pages=[[("E1", "Event One")]], mve_pages=[[("E2", "MVE Two")]],
        )
        # Modeled calls are booby-trapped: touching either is the bug.
        def _modeled_is_broken(*_a, **_k):
            raise AssertionError("modeled SDK call used; it cannot parse live events")

        client.get_events = MagicMock(side_effect=_modeled_is_broken)
        client.get_multivariate_events = MagicMock(side_effect=_modeled_is_broken)
        _patch_single_event_lookups(monkeypatch)

        result = historical._load_or_build_event_titles(client, {"E1", "E2"})
        assert result == {"E1": "Event One", "E2": "MVE Two"}
        assert client.get_events_without_preload_content.call_count >= 1
        assert client.get_multivariate_events_without_preload_content.call_count >= 1

    def test_per_ticker_fallback_is_capped(self, isolated_cache, monkeypatch, caplog):
        # The per-ticker fallback costs one HTTP round trip each. It was
        # written for a handful of stragglers, but a 21-day window measured
        # 289,235 unresolved tickers live (2026-08-03) — uncapped and
        # sequential that is hours of silent grinding. Past the cap, tickers
        # are "" for this run, exactly as after a failed lookup — but since
        # DR-51 they are DEFERRED, not stored (see TestEventTitleAccumulatorBound).
        monkeypatch.setattr(historical, "EVENT_TITLE_FALLBACK_MAX_LOOKUPS", 3)
        client = _make_client_with_event_pages(non_mve_pages=[], mve_pages=[])
        wanted = {f"E{i:02d}" for i in range(10)}
        fallback = _patch_single_event_lookups(
            monkeypatch, single_lookups={t: f"Title {t}" for t in wanted},
        )

        with caplog.at_level(logging.WARNING):
            result = historical._load_or_build_event_titles(client, wanted)

        # Every requested ticker is present — deferred ones as "" this run.
        assert set(result) == wanted
        assert fallback.call_count == 3
        resolved = {t for t, v in result.items() if v}
        assert len(resolved) == 3
        assert all(result[t] == "" for t in wanted - resolved)
        # The cap is deterministic (sorted), so a re-run can't shuffle coverage.
        assert resolved == {"E00", "E01", "E02"}
        # And it must never be silent about what it skipped — these are
        # non-combo tickers, whose titles can change a pair, so it WARNs.
        assert any("deferring the other 7 non-combo tickers" in r.getMessage()
                   and r.levelno == logging.WARNING for r in caplog.records)
        # DR-51: the deferred seven were never looked up, so nothing is stored
        # for them — only the three genuine answers reach the accumulator.
        assert json.loads(isolated_cache.read_text()) == {
            t: f"Title {t}" for t in ("E00", "E01", "E02")
        }

    def test_per_ticker_fallback_runs_in_parallel(self, isolated_cache, monkeypatch):
        # Each lookup is an independent read-only GET, so they must overlap
        # rather than run one-at-a-time.
        import threading

        monkeypatch.setattr(historical, "EVENT_TITLE_FALLBACK_MAX_WORKERS", 4)
        client = _make_client_with_event_pages(non_mve_pages=[], mve_pages=[])
        wanted = {f"E{i:02d}" for i in range(8)}

        concurrent = 0
        peak = 0
        lock = threading.Lock()
        barrier_wait = threading.Event()

        def slow_get(_client, path, **_params):
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
            tkr = path.rsplit("/", 1)[-1]
            return _raw_resp({"event": {"title": f"Title {tkr}"}})

        monkeypatch.setattr(historical, "_signed_raw_get", MagicMock(side_effect=slow_get))
        result = historical._load_or_build_event_titles(client, wanted)

        assert set(result) == wanted
        assert all(v.startswith("Title ") for v in result.values())
        assert peak > 1, f"lookups ran sequentially (peak concurrency {peak})"

    def test_cache_hit_skips_api(self, isolated_cache):
        # First call populates the cache; second call should not touch the API
        client1 = _make_client_with_event_pages(non_mve_pages=[[("E1", "Title One")]])
        historical._load_or_build_event_titles(client1, {"E1"})

        client2 = _make_client_with_event_pages(non_mve_pages=[])
        result = historical._load_or_build_event_titles(client2, {"E1"})
        assert result == {"E1": "Title One"}
        # No API calls on the cache hit
        assert client2.get_events_without_preload_content.call_count == 0
        assert client2.get_multivariate_events_without_preload_content.call_count == 0

    def test_use_cache_false_bypasses_disk(self, isolated_cache):
        # Pre-populate disk cache with a stale value
        client1 = _make_client_with_event_pages(non_mve_pages=[[("E1", "Old Title")]])
        historical._load_or_build_event_titles(client1, {"E1"})

        # use_cache=False should re-fetch — assert get_events is called
        client2 = _make_client_with_event_pages(non_mve_pages=[[("E1", "Fresh Title")]])
        result = historical._load_or_build_event_titles(client2, {"E1"}, use_cache=False)
        assert result["E1"] == "Fresh Title"
        assert client2.get_events_without_preload_content.call_count >= 1

    def test_corrupt_cache_is_treated_as_miss(self, isolated_cache, monkeypatch, caplog):
        # A truncated accumulator (interrupted write from an older build, OOM
        # kill mid-run) must read back as "no cache", not crash the backtest
        # before it has issued a single request.
        isolated_cache.write_text('{"E1": "Half A Titl')
        client = _make_client_with_event_pages(non_mve_pages=[[("E1", "Recovered Title")]])
        _patch_single_event_lookups(monkeypatch)

        with caplog.at_level(logging.WARNING):
            result = historical._load_or_build_event_titles(client, {"E1"})

        assert result == {"E1": "Recovered Title"}
        assert any("Corrupt JSON cache" in r.getMessage() for r in caplog.records)
        # And the damaged file is replaced by a well-formed one.
        assert json.loads(isolated_cache.read_text()) == {"E1": "Recovered Title"}

    def test_no_cache_run_preserves_unrelated_disk_titles(self, isolated_cache, monkeypatch):
        # BS-09: the on-disk map is a cross-run ACCUMULATOR. A --no-cache run
        # resolves its own tickers from scratch, but must not wipe titles other
        # runs paid a round trip each for.
        _patch_single_event_lookups(monkeypatch)
        client1 = _make_client_with_event_pages(non_mve_pages=[[("E1", "Title One")]])
        historical._load_or_build_event_titles(client1, {"E1"})

        client2 = _make_client_with_event_pages(non_mve_pages=[[("E2", "Title Two")]])
        result = historical._load_or_build_event_titles(client2, {"E2"}, use_cache=False)

        # This run's return value covers only this run's tickers...
        assert result == {"E2": "Title Two"}
        # ...but the accumulator on disk keeps both.
        assert json.loads(isolated_cache.read_text()) == {
            "E1": "Title One", "E2": "Title Two",
        }

    def test_fresh_non_empty_title_wins_over_disk(self, isolated_cache, monkeypatch):
        # A re-resolved title is the newer truth — it must overwrite the disk
        # value, otherwise --no-cache could never correct a stale title.
        _patch_single_event_lookups(monkeypatch)
        isolated_cache.write_text(json.dumps({"E1": "Stale Title"}))
        client = _make_client_with_event_pages(non_mve_pages=[[("E1", "Fresh Title")]])

        result = historical._load_or_build_event_titles(client, {"E1"}, use_cache=False)

        assert result["E1"] == "Fresh Title"
        assert json.loads(isolated_cache.read_text()) == {"E1": "Fresh Title"}

    def test_fresh_poison_pill_does_not_clobber_disk_title(self, isolated_cache, monkeypatch):
        # A failed lookup / cap-skipped ticker resolves to "" for THIS run, but
        # "" is an absence of information — it must never overwrite a real
        # title an earlier run resolved.
        isolated_cache.write_text(json.dumps({"E1": "Good Title"}))
        client = _make_client_with_event_pages(non_mve_pages=[], mve_pages=[])
        _patch_single_event_lookups(monkeypatch, single_failures={"E1"})

        result = historical._load_or_build_event_titles(client, {"E1"}, use_cache=False)

        # TS-11: this RUN could not resolve it, but an earlier one did, and the
        # accumulator exists precisely so that answer is not thrown away. The
        # return is the MERGED view — returning "" here is what collapsed the
        # same-title grouping key toward the bare title under --no-cache.
        assert result == {"E1": "Good Title"}
        assert json.loads(isolated_cache.read_text()) == {"E1": "Good Title"}

    def test_fresh_poison_pill_stored_for_ticker_unknown_to_disk(self, isolated_cache,
                                                                 monkeypatch):
        # Poison-pill semantics survive the merge: an unresolvable ticker disk
        # has never seen is still recorded as "", so later runs don't re-pay
        # the failing round trip.
        isolated_cache.write_text(json.dumps({"E1": "Good Title"}))
        client = _make_client_with_event_pages(non_mve_pages=[], mve_pages=[])
        fallback = _patch_single_event_lookups(monkeypatch, single_failures={"NEW-1"})

        result = historical._load_or_build_event_titles(client, {"NEW-1"}, use_cache=False)

        assert result == {"NEW-1": ""}
        assert fallback.call_count == 1
        assert json.loads(isolated_cache.read_text()) == {
            "E1": "Good Title", "NEW-1": "",
        }
        # The pill is honored on the next cached run — no repeat lookup.
        client2 = _make_client_with_event_pages(non_mve_pages=[], mve_pages=[])
        fallback2 = _patch_single_event_lookups(monkeypatch, single_failures={"NEW-1"})
        # Restricted to the tickers ASKED about, not the whole accumulator:
        # that file holds every ticker every past run resolved (hundreds of
        # thousands), and a caller asking about one must not receive them all.
        assert historical._load_or_build_event_titles(client2, {"NEW-1"}) == {"NEW-1": ""}
        assert fallback2.call_count == 0

    def test_listing_pages_request_market_page_size_limit(self, isolated_cache, monkeypatch):
        # BS-22: both bulk listing loops used to hardcode limit=200 rather than
        # importing the shared MARKET_PAGE_SIZE constant. Assert the kwargs
        # actually sent match the constant itself (not just today's value of
        # 200), so a future change to MARKET_PAGE_SIZE stays honored here.
        client = _make_client_with_event_pages(
            non_mve_pages=[[("E1", "Event One")]], mve_pages=[[("E2", "MVE Two")]],
        )
        historical._load_or_build_event_titles(client, {"E1", "E2"})

        events_calls = client.get_events_without_preload_content.call_args_list
        mve_calls = client.get_multivariate_events_without_preload_content.call_args_list
        assert events_calls, "status listing was never called"
        assert mve_calls, "MVE listing was never called"
        for call in events_calls:
            assert call.kwargs["limit"] == historical.MARKET_PAGE_SIZE
        for call in mve_calls:
            assert call.kwargs["limit"] == historical.MARKET_PAGE_SIZE


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

    def test_yes_sub_title_maps_to_subtitle_key(self):
        # 2026-08 drift: the archive now carries the outcome label in
        # yes_sub_title. It must land on the `subtitle` cache key so the
        # backtester's (event_title, title, subtitle) grouping keeps its
        # intra-title discriminator without a cache-shape change.
        d = historical._market_to_dict(
            _raw_market_dict(yes_sub_title="Pierbattista Pizzaballa")
        )
        assert d["subtitle"] == "Pierbattista Pizzaballa"

    def test_explicit_subtitle_wins_over_yes_sub_title(self):
        d = historical._market_to_dict(
            _raw_market_dict(subtitle="Legacy Label", yes_sub_title="New Label")
        )
        assert d["subtitle"] == "Legacy Label"

    def test_no_sub_title_is_not_used_as_subtitle(self):
        # no_sub_title is the negated phrasing — using it would produce a
        # grouping key that differs between the YES and NO framings.
        d = historical._market_to_dict(_raw_market_dict(no_sub_title="Someone else"))
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

    def test_tick_structure_fields_pass_through_raw(self):
        # price_level_structure/price_ranges (2026-08 groundwork) must be
        # stored RAW — not parsed into scanner.PriceRange objects — since this
        # dict is JSON-serialized straight into the backtest cache.
        raw = _raw_market_dict(
            price_level_structure="deci_cent",
            price_ranges=[{"start": "0.0000", "end": "1.0000", "step": "0.0010"}],
        )
        d = historical._market_to_dict(raw)
        assert d["price_level_structure"] == "deci_cent"
        assert d["price_ranges"] == [{"start": "0.0000", "end": "1.0000", "step": "0.0010"}]

    def test_missing_tick_structure_fields_map_to_none(self):
        d = historical._market_to_dict(_raw_market_dict())
        assert d["price_level_structure"] is None
        assert d["price_ranges"] is None

    def test_market_to_dict_passes_exchange_index_through(self):
        # Shard fidelity (2026-08): stored raw so backtest data can
        # distinguish exchange shards; never filtered on the backtest path.
        d = historical._market_to_dict(_raw_market_dict(exchange_index=1))
        assert d["exchange_index"] == 1

    def test_missing_exchange_index_maps_to_none(self):
        # Pre-existing cache records lack the key and must read back as None
        d = historical._market_to_dict(_raw_market_dict())
        assert d["exchange_index"] is None

    def test_stored_record_is_json_serializable(self):
        # The produced dict is gzip+json-dumped straight into the cache — a
        # non-JSON-native value here (e.g. an accidentally-parsed PriceRange
        # dataclass) would blow up at cache-write time, not at read time.
        raw = _raw_market_dict(
            price_level_structure="tapered_deci_cent",
            price_ranges=[
                {"start": "0.0000", "end": "0.1000", "step": "0.0010"},
                {"start": "0.1000", "end": "0.9000", "step": "0.0100"},
            ],
        )
        d = historical._market_to_dict(raw, "Some Event Title")
        round_tripped = json.loads(json.dumps(d))
        assert round_tripped == d


class TestFetchAllSettledMarkets:
    def test_archive_pages_past_barren_page_and_dict_output(self, tmp_path, monkeypatch):
        # The /historical/markets archive ignores settlement-time filters, so
        # the sequential walk needs SOME stop rule or it walks the whole
        # multi-million-market archive. BS-02: that rule is no longer "the
        # first page whose newest-created record predates start_date" — the
        # archive is created-ordered, so such a page proves nothing about
        # deeper ones. The walk pages past it (up to ARCHIVE_MAX_BARREN_PAGES
        # consecutive unproductive pages) and still collects the long-lived
        # market behind it. These fake pages use opaque cursors and records
        # without created_time, so the sharded path's synthesis check fails and
        # the fetch exercises the sequential fallback — the path this stop rule
        # lives on. Also verifies dicts flow through to the cached format.
        monkeypatch.setattr(historical, "CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr(historical, "INCLUDE_MVE_MARKETS", False)
        # Title resolution is not under test here (live is a bare MagicMock).
        monkeypatch.setattr(historical, "_load_or_build_event_titles", lambda *a, **k: {})
        from datetime import date

        def market(tkr, settled):
            return {"ticker": tkr, "event_ticker": f"EV-{tkr}", "title": f"Q {tkr}",
                    "result": "yes", "yes_ask_dollars": "0.40", "no_ask_dollars": "0.60",
                    "yes_bid_dollars": "0.38", "close_time": settled,
                    "settlement_ts": settled, "status": "finalized"}

        # Archive pages newest-CREATED first: page1 in-window, page2 predates
        # start_date entirely (barren — but not a stop signal), page3 holds a
        # long-lived in-window settler and ends the chain; then the live
        # endpoint returns one post-cutoff market.
        pages = {
            "cutoff": {"market_settled_ts": "2026-03-01T00:00:00Z"},
            "hist1": {"markets": [market("IN-WINDOW", "2026-02-15T00:00:00Z")], "cursor": "C2"},
            "hist2": {"markets": [market("TOO-OLD", "2026-01-01T00:00:00Z")], "cursor": "C3"},
            "hist3": {"markets": [market("LONGLIVED", "2026-02-20T00:00:00Z")], "cursor": None},
        }
        calls = {"hist": 0}
        by_cursor = {"C2": pages["hist2"], "C3": pages["hist3"]}

        def fake_signed_get(client, path, **params):
            if path.endswith("/historical/cutoff"):
                return _raw_resp(pages["cutoff"])
            assert path.endswith("/historical/markets")
            calls["hist"] += 1
            return _raw_resp(by_cursor.get(params.get("cursor"), pages["hist1"]))

        monkeypatch.setattr(historical, "_signed_raw_get", fake_signed_get)
        live = MagicMock()
        live.get_markets_without_preload_content = MagicMock(return_value=_raw_resp(
            {"markets": [market("RECENT", "2026-03-05T00:00:00Z")], "cursor": None}
        ))

        out = historical.fetch_all_settled_markets(
            MagicMock(), live, start_date=date(2026, 2, 1), use_cache=False,
        )
        # 4 archive calls: 1 synthesis probe (fails — opaque cursor), then the
        # sequential walk's three pages. Page 2 is barren but its cursor is
        # followed (one barren page is far below ARCHIVE_MAX_BARREN_PAGES);
        # page 3's null cursor is what ends the walk.
        assert calls["hist"] == 4
        tickers = {m["ticker"] for m in out}
        assert tickers == {"IN-WINDOW", "LONGLIVED", "RECENT"}
        # Output dicts carry the exact cached format
        m = next(mm for mm in out if mm["ticker"] == "IN-WINDOW")
        assert m["yes_ask_dollars"] == "0.40"
        assert m["settlement_ts"] == "2026-02-15T00:00:00Z"
        assert m["event_title"] == ""

    def test_corrupt_assembled_cache_falls_through_to_refetch(self, tmp_path, monkeypatch,
                                                              caplog):
        # BS-08: a truncated assembled cache (the multi-hour fetch's final
        # write, historically interrupted by OOM kills) must read back as a
        # miss and refetch, not raise before a single request is issued. This
        # one is a LEGACY (pre-SS-1) single-document cache, which is still
        # read through _load_json_cache; the streamed format's own corruption
        # tests are in TestStreamedAssembledCache.
        from datetime import date

        cache_dir = tmp_path / "cache"
        monkeypatch.setattr(historical, "CACHE_DIR", cache_dir)
        monkeypatch.setattr(historical, "INCLUDE_MVE_MARKETS", False)
        # Title resolution is not under test here (live is a bare MagicMock).
        monkeypatch.setattr(historical, "_load_or_build_event_titles", lambda *a, **k: {})
        cache_dir.mkdir(parents=True)
        # INCLUDE_MVE_MARKETS is False above, so the assembled cache this run
        # looks for carries the DR-57 _nomve marker; seeding the unmarked name
        # would be a plain cache miss and would not exercise the corrupt-cache
        # fall-through this test is named for.
        corrupt = cache_dir / "settled_markets_2026-02-01_nomve.json"
        corrupt.write_text('[{"ticker": "T1"')

        def fake_signed_get(client, path, **params):
            if path.endswith("/historical/cutoff"):
                return _raw_resp({"market_settled_ts": "2026-03-01T00:00:00Z"})
            return _raw_resp({"markets": [], "cursor": None})

        monkeypatch.setattr(historical, "_signed_raw_get", fake_signed_get)
        live = MagicMock()
        live.get_markets_without_preload_content = MagicMock(return_value=_raw_resp({
            "markets": [{"ticker": "RECENT", "event_ticker": "EV", "title": "Q",
                         "result": "yes", "yes_ask_dollars": "0.40",
                         "no_ask_dollars": "0.60", "yes_bid_dollars": "0.38",
                         "close_time": "2026-03-05T00:00:00Z",
                         "settlement_ts": "2026-03-05T00:00:00Z",
                         "status": "finalized"}],
            "cursor": None,
        }))

        with caplog.at_level(logging.WARNING):
            out = historical.fetch_all_settled_markets(
                MagicMock(), live, start_date=date(2026, 2, 1), use_cache=True,
            )

        assert {m["ticker"] for m in out} == {"RECENT"}
        assert any("Corrupt JSON cache" in r.getMessage() for r in caplog.records)
        # A well-formed cache is written for the next run — in the streamed
        # format (SS-1), which the next run prefers — and the damaged legacy
        # file is gone, as the old code's rebuild overwrote it: nothing writes
        # that format any more, and the committed rebuild retires it.
        streamed = cache_dir / "settled_markets_2026-02-01_nomve.jsonl.gz"
        assert _read_slice_file(streamed) == list(out)
        assert not corrupt.exists()

    def test_event_titles_resolved_when_mve_excluded(self, tmp_path, monkeypatch,
                                                     isolated_cache):
        # E2: INCLUDE_MVE_MARKETS=False must narrow WHICH markets are fetched,
        # never how the remaining ones are grouped. The live scanner attaches
        # every market's parent event title in both modes
        # (scanner._market_from_dict on the binary listing path), so skipping
        # event-title resolution here collapsed the backtester's same-title key
        # from (event_title, title, subtitle) to (title, subtitle) and paired
        # binary markets under different events that live keeps apart.
        # The MVE *listing* phase is the only thing the flag may gate.
        from datetime import date

        monkeypatch.setattr(historical, "CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr(historical, "INCLUDE_MVE_MARKETS", False)

        def fake_signed_get(client, path, **params):
            if path.endswith("/historical/cutoff"):
                return _raw_resp({"market_settled_ts": "2026-03-01T00:00:00Z"})
            return _raw_resp({"markets": [], "cursor": None})

        monkeypatch.setattr(historical, "_signed_raw_get", fake_signed_get)

        # Bulk (non-MVE) events listing resolves EV; the MVE listing must not run.
        live = _make_client_with_event_pages([[("EV", "Event EV")]])
        live.get_markets_without_preload_content = MagicMock(return_value=_raw_resp({
            "markets": [{"ticker": "RECENT", "event_ticker": "EV", "title": "Q",
                         "result": "yes", "yes_ask_dollars": "0.40",
                         "no_ask_dollars": "0.60", "yes_bid_dollars": "0.38",
                         "close_time": "2026-03-05T00:00:00Z",
                         "settlement_ts": "2026-03-05T00:00:00Z",
                         "status": "finalized"}],
            "cursor": None,
        }))

        out = historical.fetch_all_settled_markets(
            MagicMock(), live, start_date=date(2026, 2, 1), use_cache=False,
        )

        rows = list(out)  # a streamed corpus since SS-1: iterate, never index
        assert [m["ticker"] for m in rows] == ["RECENT"]
        assert rows[0]["event_title"] == "Event EV"
        # No MVE ticker can be wanted when every market fetch excluded them.
        assert live.get_multivariate_events_without_preload_content.call_count == 0

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

    def test_start_date_at_or_after_cutoff_warns(self, tmp_path, monkeypatch, caplog):
        # BS-11: a start_date at/after the archive cutoff means every market
        # in the window is live-era, and live-era markets 404 on the historical
        # candlesticks endpoint (see the CLAUDE.md "Backtest windows must
        # start BEFORE the archive cutoff" gotcha) — so the window is
        # structurally 0-trade no matter what this fetch returns. This is a
        # WARN, not an abort: the fetch must still run to completion.
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

        with caplog.at_level(logging.WARNING):
            out = historical.fetch_all_settled_markets(
                MagicMock(), live, start_date=date(2026, 7, 6), use_cache=False,
            )

        # Warn, never abort — the fetch still completes and returns normally.
        assert list(out) == []
        assert len(out) == 0
        assert any("archive cutoff" in r.getMessage() for r in caplog.records
                   if r.levelname == "WARNING")

    def test_start_date_before_cutoff_does_not_warn(self, tmp_path, monkeypatch, caplog):
        # The common case (default start_date 2024-01-01, cutoff far later)
        # must not trip the new warning.
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

        with caplog.at_level(logging.WARNING):
            historical.fetch_all_settled_markets(
                MagicMock(), live, start_date=date(2024, 1, 1), use_cache=False,
            )

        assert not any("archive cutoff" in r.getMessage() for r in caplog.records
                       if r.levelname == "WARNING")


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


def _expected_in_window(archive_markets, live_markets, start_ts, cutoff_ts):
    """Oracle: ground truth computed DIRECTLY from the fixture records — every
    binary market that settled inside the backtest window, whichever endpoint
    serves it.

    Deliberately not a re-implementation of the page walk. The previous oracle
    replayed the walk's own early-stop rule, so it shared BS-02's bug (it
    sorted by created_time DESC but stopped on a settlement comparison) and the
    parity tests could never disagree with the code they were checking. Ground
    truth here is a plain predicate over the fixture, so a walk that drops a
    record now fails the test.

    Archive side: result in ("yes", "no") and start_ts <= settle < cutoff_ts.
    Live side: result in ("yes", "no") and settle >= max(cutoff_ts, start_ts) —
    the live endpoint does not serve pre-cutoff settlements at all.
    """
    def _binary_settles_in(markets, lo, hi):
        out = set()
        for m in markets:
            if m["result"] not in ("yes", "no"):
                continue
            settle = historical._iso_epoch(m["settlement_ts"])
            if settle is None:
                continue
            if lo <= settle and (hi is None or settle < hi):
                out.add(m["ticker"])
        return out

    expected = _binary_settles_in(archive_markets, start_ts, cutoff_ts)
    expected |= _binary_settles_in(live_markets, max(cutoff_ts, start_ts), None)
    return expected


class _PagedArchive:
    """Serves a fixed list of hand-built pages in order, ignoring the cursor.

    Both archive walks are strictly linear (one cursor chain, no jumps once
    started), so call N is page N. Used where the page BOUNDARIES themselves
    are the thing under test — the stop rule counts pages, so the fixture has
    to control exactly which records share one.
    """

    def __init__(self, pages):
        self.pages = pages
        self.calls = 0

    def page(self, **_):
        idx = self.calls
        self.calls += 1
        assert idx < len(self.pages), (
            f"page {idx} requested — the walk paged past every page the "
            f"fixture defines"
        )
        # Always advertise a next page: the stop rule, not exhaustion, is what
        # must end the walk (the last page's cursor still points onward).
        return {"markets": self.pages[idx], "cursor": f"CUR{idx + 1}"}


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
        # fast-settled pre-start markets.
        #
        # BS-02 arrangement (load-bearing): the five PRE* records sit between
        # start_date and LONGLIVED in created order so that — at page_size 3,
        # in BOTH the tail walk and the top-down sequential walk — LONGLIVED
        # lands on the page immediately AFTER a page whose records all settled
        # before the window. The old "stop when page[0] settled pre-window"
        # rule therefore provably drops LONGLIVED on both paths; the
        # barren-page rule finds it. Changing the count or the created_time
        # ordering of the PRE* records breaks that alignment.
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
            # Short-lived pre-window settlers created just below start_date:
            # these fill the barren page that used to end both walks.
            _mk_raw_market("PRE1", "2026-06-04T23:50:00Z", "2026-06-04T23:55:00Z"),
            _mk_raw_market("PRE2", "2026-06-04T23:45:00Z", "2026-06-04T23:50:00Z"),
            _mk_raw_market("PRE3", "2026-06-04T23:40:00Z", "2026-06-04T23:45:00Z"),
            _mk_raw_market("PRE4", "2026-06-04T23:35:00Z", "2026-06-04T23:40:00Z"),
            _mk_raw_market("PRE5", "2026-06-04T23:30:00Z", "2026-06-04T23:35:00Z"),
            # tail: created before start_date but settled inside the window —
            # reachable only by paging PAST the all-pre-window page above
            _mk_raw_market("LONGLIVED", "2026-06-04T23:00:00Z", "2026-06-06T10:00:00Z"),
            # more pre-start fast markets below it
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

        expected = _expected_in_window(
            archive_markets, live_markets,
            self._ts(self.START + "T00:00:00+00:00"), self._ts(self.CUTOFF),
        )
        assert {m["ticker"] for m in out} == expected
        # BS-02: the long-lived tail settler must be in the RESULT, not merely
        # in the oracle's set — it sits behind an all-pre-window page that the
        # old early-stop rule never paged past.
        assert "LONGLIVED" in {m["ticker"] for m in out}
        # No duplicate tickers despite deliberately overlapping slice boundaries
        assert len(out) == len({m["ticker"] for m in out})
        # Compact dict format survives the day store round-trip
        a1 = next(m for m in out if m["ticker"] == "A1")
        assert a1["open_time"] == "2026-06-09T20:00:00.500000Z"
        assert a1["yes_ask_dollars"] == "0.40"

    def test_tail_keeps_longlived_settlement_past_barren_page(self, tmp_path,
                                                              monkeypatch):
        # BS-02, headline regression. The archive is ordered by created_time,
        # so page[0] is only the newest-CREATED record — its settlement time
        # says nothing about the rest of the page, let alone deeper pages. The
        # old rule stopped both walks at the first page whose page[0] settled
        # pre-window, which (most markets being short-lived) fires almost
        # immediately and silently drops long-lived in-window settlers.
        # LONGLIVED is positioned one page BEHIND such a page in both walks.
        archive_markets, live_markets = self._fixture_markets()

        # Sharded path — the tail walk below created_time == start_date.
        out = self._run(monkeypatch, tmp_path / "sharded",
                        _FakeArchive(archive_markets), _FakeLive(live_markets))
        assert "LONGLIVED" in {m["ticker"] for m in out}

        # Sequential fallback — same rule, same fixture, top-down walk.
        # Opaque cursors defeat cursor synthesis, forcing the fallback.
        out_seq = self._run(monkeypatch, tmp_path / "sequential",
                            _FakeArchive(archive_markets, opaque_cursors=True),
                            _FakeLive(live_markets))
        assert "LONGLIVED" in {m["ticker"] for m in out_seq}

    def test_tail_progress_logs_under_its_own_label(self, tmp_path, monkeypatch,
                                                    caplog):
        # CLAUDE.md: progress labels are load-bearing for diagnosis. The tail is
        # a SERIAL walk, but it used to share the day-slice pool's progress
        # object, so its pages logged as "[sharded]" — a stalled tail was
        # indistinguishable in the log from a stalled (parallel) slice pool.
        captured = {}
        real_tail = historical._fetch_archive_tail

        def spy(hist_client, start_ts, cutoff_ts, hist_kwargs, progress):
            captured["progress"] = progress
            return real_tail(hist_client, start_ts, cutoff_ts, hist_kwargs, progress)

        monkeypatch.setattr(historical, "_fetch_archive_tail", spy)
        archive_markets, live_markets = self._fixture_markets()
        self._run(monkeypatch, tmp_path, _FakeArchive(archive_markets),
                  _FakeLive(live_markets))

        progress = captured["progress"]
        # The tail really did page through this progress object...
        assert progress.pages > 0
        # ...and it is not the day-slice pool's counter (a shared object would
        # already be carrying the slice pages).
        with caplog.at_level(logging.INFO):
            # _FetchProgress only logs every 100 pages, so drive it to the next
            # boundary and read the label off the line it emits.
            for _ in range(100 - progress.pages % 100):
                progress.tick(0)
        lines = [r.getMessage() for r in caplog.records if "pages scanned" in r.getMessage()]
        assert lines
        assert all("Historical archive [tail]" in ln for ln in lines)
        assert not any("[sharded]" in ln for ln in lines)

    def test_tail_stops_at_the_absolute_page_cap(self, monkeypatch, caplog):
        # The barren rule only bounds depth PAST the last productive page: every
        # page here holds an in-window settlement, so the counter never rises
        # and only ARCHIVE_TAIL_MAX_PAGES ends this serial, uncached walk.
        monkeypatch.setattr(historical, "ARCHIVE_TAIL_MAX_PAGES", 3)
        pages = [
            # Created before start_date, settling inside the window → every page
            # is "productive", so the barren counter stays at 0 throughout.
            [_mk_raw_market(f"LL{i}", f"2026-06-04T2{i}:00:00Z",
                            "2026-06-06T10:00:00Z")]
            for i in range(3)
        ] + [
            # Beyond the cap: must never be requested (_PagedArchive asserts).
            [_mk_raw_market("DEEP", "2026-06-01T00:00:00Z", "2026-06-07T00:00:00Z")],
        ]
        with caplog.at_level(logging.WARNING):
            kept, calls = self._walk_paged(
                monkeypatch, historical._fetch_archive_tail, pages,
            )
        assert kept == {"LL0", "LL1", "LL2"}
        assert calls == 3
        warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert any("ARCHIVE_TAIL_MAX_PAGES" in w for w in warnings)

    def test_tail_stops_at_the_absolute_record_cap(self, monkeypatch, caplog):
        # TS-15. The tail is the one fetch path with no chunked emit sink, so
        # its whole result is resident at once; the page cap alone allows
        # ~2M records (~5 GB). The record cap composes with it — whichever
        # binds first stops the walk — and here the page cap is left high so
        # only the record cap can fire.
        monkeypatch.setattr(historical, "ARCHIVE_TAIL_MAX_PAGES", 100)
        monkeypatch.setattr(historical, "ARCHIVE_TAIL_MAX_RECORDS", 2)
        pages = [
            [_mk_raw_market(f"LL{i}", f"2026-06-04T2{i}:00:00Z",
                            "2026-06-06T10:00:00Z")]
            for i in range(3)
        ] + [
            # Beyond the cap: must never be requested (_PagedArchive asserts).
            [_mk_raw_market("DEEP", "2026-06-01T00:00:00Z", "2026-06-07T00:00:00Z")],
        ]
        with caplog.at_level(logging.WARNING):
            kept, calls = self._walk_paged(
                monkeypatch, historical._fetch_archive_tail, pages,
            )
        assert kept == {"LL0", "LL1"}
        assert calls == 2
        warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert any("ARCHIVE_TAIL_MAX_RECORDS" in w for w in warnings)
        # GUARD: the page cap must not be what stopped this walk.
        assert not any("ARCHIVE_TAIL_MAX_PAGES" in w for w in warnings)

    def test_record_cap_does_not_fire_on_an_ordinary_walk(self, monkeypatch, caplog):
        # GUARD: the cap is a backstop, silent on any realistic window.
        monkeypatch.setattr(historical, "ARCHIVE_MAX_BARREN_PAGES", 1)
        pages = [
            [_mk_raw_market("LL0", "2026-06-04T20:00:00Z", "2026-06-06T10:00:00Z")],
            [_mk_raw_market("PB1", "2026-06-04T19:00:00Z", "2026-06-04T19:30:00Z")],
        ]
        with caplog.at_level(logging.WARNING):
            self._walk_paged(monkeypatch, historical._fetch_archive_tail, pages)
        assert "ARCHIVE_TAIL_MAX_RECORDS" not in caplog.text

    def _walk_paged(self, monkeypatch, walk, pages):
        """Run one archive walk against hand-built pages; return (kept, calls)."""
        paged = _PagedArchive(pages)
        monkeypatch.setattr(
            historical, "_signed_raw_get",
            lambda client, path, **params: _raw_resp(paged.page(**params)),
        )
        start_ts = self._ts(self.START + "T00:00:00+00:00")
        cutoff_ts = self._ts(self.CUTOFF)
        if walk is historical._fetch_archive_tail:
            kept = walk(MagicMock(), start_ts, cutoff_ts, {"limit": 1000},
                        historical._FetchProgress("test tail"))
        else:
            kept = walk(MagicMock(), start_ts, cutoff_ts, {"limit": 1000})
        return {m["ticker"] for m in kept}, paged.calls

    @pytest.mark.parametrize("walk", [
        historical._fetch_archive_tail,
        historical._fetch_archive_sequential,
    ])
    def test_archive_walk_stops_after_max_barren_pages(self, monkeypatch, walk):
        # No exact stop rule exists on a created-ordered walk, so the walks are
        # bounded by productivity instead: ARCHIVE_MAX_BARREN_PAGES consecutive
        # pages with zero in-window settlements ends the walk. Deeper pages
        # must never be requested (_PagedArchive asserts if they are).
        monkeypatch.setattr(historical, "ARCHIVE_MAX_BARREN_PAGES", 3)
        pages = [
            # Productive page: resets/holds the counter at 0.
            [_mk_raw_market("NEAR", "2026-06-04T23:00:00Z", "2026-06-06T00:00:00Z")],
            # Three consecutive barren pages → counter reaches the cap.
            [_mk_raw_market("PB1", "2026-06-04T22:00:00Z", "2026-06-04T22:30:00Z")],
            [_mk_raw_market("PB2", "2026-06-04T21:00:00Z", "2026-06-04T21:30:00Z")],
            [_mk_raw_market("PB3", "2026-06-04T20:00:00Z", "2026-06-04T20:30:00Z")],
            # Beyond the cap: an in-window settler the walk must NOT reach.
            [_mk_raw_market("DEEP", "2026-06-01T00:00:00Z", "2026-06-07T00:00:00Z")],
        ]
        kept, calls = self._walk_paged(monkeypatch, walk, pages)
        assert kept == {"NEAR"}
        assert calls == 4  # the fourth barren-capped page is the last fetched

    @pytest.mark.parametrize("walk", [
        historical._fetch_archive_tail,
        historical._fetch_archive_sequential,
    ])
    def test_barren_counter_is_consecutive_and_result_agnostic(self, monkeypatch, walk):
        # Two things at once: the counter RESETS on a productive page (so the
        # bound is consecutive, not cumulative), and productivity is judged on
        # ANY in-window settlement — the reset page here holds only a VOIDED
        # market, which neither walk keeps. Pages full of voided (or, in the
        # tail, day-sliced) records must not spuriously trip the counter.
        monkeypatch.setattr(historical, "ARCHIVE_MAX_BARREN_PAGES", 2)
        pages = [
            [_mk_raw_market("NEAR", "2026-06-04T23:00:00Z", "2026-06-06T00:00:00Z")],
            [_mk_raw_market("PB1", "2026-06-04T22:00:00Z", "2026-06-04T22:30:00Z")],
            # Kept by neither walk, but proof the walk is still in productive
            # created-time territory → counter back to 0.
            [_mk_raw_market("VOIDED", "2026-06-04T21:00:00Z", "2026-06-06T05:00:00Z",
                            result="void")],
            [_mk_raw_market("PB2", "2026-06-04T20:00:00Z", "2026-06-04T20:30:00Z")],
            [_mk_raw_market("PB3", "2026-06-04T19:00:00Z", "2026-06-04T19:30:00Z")],
            # Never reached: the cap is hit on the page above.
            [_mk_raw_market("DEEP", "2026-06-01T00:00:00Z", "2026-06-07T00:00:00Z")],
        ]
        kept, calls = self._walk_paged(monkeypatch, walk, pages)
        assert kept == {"NEAR"}
        assert calls == 5

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
        # BS-15: the fetch workers persist through _DayStreamWriter, so every
        # slice they wrote is in the streamed format — and run 2 proves those
        # files are what the reuse prescan accepts.
        for path in (tmp_path / "cache").glob("*_days/*.json.gz"):
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                assert json.loads(fh.readline())["meta"]["format"] == "jsonl-v1"

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
        expected = _expected_in_window(
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
        expected = _expected_in_window(
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
        assert {m["ticker"] for m in out2} == _expected_in_window(
            archive_markets + live_markets, [],
            self._ts(self.START + "T00:00:00+00:00"), self._ts(new_cutoff),
        )
        # The migrated markets must come from the refetched archive slices
        assert {"L1", "L2", "L3"} <= {m["ticker"] for m in out2}

    def test_assembly_streams_slices_from_disk(self, tmp_path, monkeypatch):
        # Peak memory must not scale with the number of days fetched, so no
        # phase may retain slice records: every day — including ones fetched
        # moments earlier in this same run — is re-read from its file at
        # assembly. Verified by counting COMPLETE reads of each path through
        # _day_store_iter, the one reader every slice read goes through (the
        # prescan's _day_store_load included).
        archive_markets, live_markets = self._fixture_markets()
        real_iter = historical._day_store_iter
        reads: list[str] = []

        def counting_iter(path, expect_meta, keep=None):
            yield from real_iter(path, expect_meta, keep)
            reads.append(str(path))  # only reached by a read that completed

        monkeypatch.setattr(historical, "_day_store_iter", counting_iter)
        out = self._run(monkeypatch, tmp_path, _FakeArchive(archive_markets),
                        _FakeLive(live_markets))
        assert len(out) > 0  # sanity: the run actually produced records
        # Nothing iterated the returned corpus, so no read of the assembled
        # cache can be in the log; every completed read is a day slice.
        assert all("_days" in p for p in reads)

        # Cold run: the prescan found nothing on disk (no completed read), so
        # every completed read is an assembly read — and since SS-1 the
        # assembly walks its sources TWICE (count and collect event tickers,
        # then write the cache), each re-reading the slice from disk. Exactly
        # two reads per written slice: never zero (retained in memory) and
        # never more (read eagerly as well).
        written = {str(p) for p in
                   (tmp_path / "cache").glob("*_days/*.json.gz")}
        assert written, "expected day slices to have been persisted"
        assert set(reads) == written
        assert all(reads.count(p) == 2 for p in written)
        assert len(reads) == 2 * len(written)

    def test_prefilter_assembly_equals_postfilter(self, tmp_path, monkeypatch):
        # The result-neutrality proof for pushing the backtester's eligibility
        # filter into the fetch: filtering DURING assembly must produce exactly
        # what filtering the unfiltered result afterwards would — same records,
        # same order. run_backtest still applies the predicate itself, so this
        # equality is what makes the optimization invisible to backtest output.
        from datetime import date

        archive_markets, live_markets = self._fixture_markets()

        def pred(m):
            # Discriminating on purpose: keeps a mix of archive-day, tail, and
            # live records so every code path is exercised, not just one.
            return not m["ticker"].endswith("1")

        _install_sharded_fakes(monkeypatch, tmp_path / "full",
                               _FakeArchive(archive_markets), self.CUTOFF)
        out_full = historical.fetch_all_settled_markets(
            MagicMock(), _FakeLive(live_markets),
            start_date=date(2026, 6, 5), use_cache=False,
        )

        _install_sharded_fakes(monkeypatch, tmp_path / "pref",
                               _FakeArchive(archive_markets), self.CUTOFF)
        out_pref = historical.fetch_all_settled_markets(
            MagicMock(), _FakeLive(live_markets),
            start_date=date(2026, 6, 5), use_cache=False,
            prefilter=pred, prefilter_tag="testpred",
        )

        assert list(out_pref) == [m for m in out_full if pred(m)]
        # Sanity: the predicate actually removed something, and kept something.
        assert 0 < len(out_pref) < len(out_full)
        _assert_counts_cover_the_prefilter(out_full, out_pref)

        # The day-slice FILES must stay complete — they are shared across start
        # dates and other callers, so filtering them would corrupt the cache.
        slices = sorted((tmp_path / "pref" / "cache" / "archive_days").glob("*.json.gz"))
        assert slices
        stored = set()
        for path in slices:
            stored |= {m["ticker"] for m in _read_slice_file(path)}
        dropped = {m["ticker"] for m in out_full if not pred(m)}
        assert dropped & stored, "filtered-out records must still be on disk"

    def test_prefiltered_cache_filename_and_isolation(self, tmp_path, monkeypatch):
        # A prefiltered result is a strict subset, so it must never be served
        # to an unfiltered caller (or to one using different filter semantics).
        from datetime import date

        archive_markets, live_markets = self._fixture_markets()

        def pred(m):
            return not m["ticker"].endswith("1")

        archive = _FakeArchive(archive_markets)
        _install_sharded_fakes(monkeypatch, tmp_path, archive, self.CUTOFF)
        out1 = historical.fetch_all_settled_markets(
            MagicMock(), _FakeLive(live_markets), start_date=date(2026, 6, 5),
            use_cache=False, prefilter=pred, prefilter_tag="testpred",
        )
        cache_dir = tmp_path / "cache"
        assert (cache_dir / "settled_markets_2026-06-05_testpred.jsonl.gz").exists()
        assert not (cache_dir / "settled_markets_2026-06-05.jsonl.gz").exists()
        # SS-1: new runs never write the legacy single-document format.
        assert not list(cache_dir.glob("settled_markets_*.json"))

        # Second prefiltered run hits the tagged cache: zero API calls.
        archive.calls = 0
        out2 = historical.fetch_all_settled_markets(
            MagicMock(), _FakeLive(live_markets), start_date=date(2026, 6, 5),
            use_cache=True, prefilter=pred, prefilter_tag="testpred",
        )
        assert list(out2) == list(out1)
        assert archive.calls == 0

        # An unfiltered caller must NOT read the prefiltered cache.
        out_full = historical.fetch_all_settled_markets(
            MagicMock(), _FakeLive(live_markets), start_date=date(2026, 6, 5),
            use_cache=True,
        )
        assert len(out_full) > len(out1)

        # The tag is what keys the cache, so it can't be omitted.
        with pytest.raises(ValueError):
            historical.fetch_all_settled_markets(
                MagicMock(), _FakeLive(live_markets), start_date=date(2026, 6, 5),
                use_cache=False, prefilter=pred,
            )
        with pytest.raises(ValueError):
            historical.fetch_all_settled_markets(
                MagicMock(), _FakeLive(live_markets), start_date=date(2026, 6, 5),
                use_cache=False, prefilter_tag="testpred",
            )

    def test_mve_included_keeps_the_legacy_unmarked_cache_filename(self, tmp_path,
                                                                   monkeypatch):
        # DR-57, backward-compatibility half: every assembled cache already on
        # disk (there is a multi-GB one) was built with MVE included, so the
        # default INCLUDE_MVE_MARKETS=True filename must stay byte-identical to
        # the pre-DR-57 name. If this ever changes, every cached assembly is
        # silently orphaned and the next run re-pays a multi-hour fetch.
        # This one deliberately passes both BEFORE and AFTER the DR-57 fix — it
        # pins the unchanged half. The two that fail pre-fix (i.e. the DR-57
        # regression tests proper) are test_mve_flag_separates_assembled_cache_
        # filenames and test_cache_written_under_one_mve_setting_is_not_served_
        # to_the_other; don't delete this one as redundant with them.
        from datetime import date

        archive_markets, live_markets = self._fixture_markets()
        monkeypatch.setattr(historical, "INCLUDE_MVE_MARKETS", True)
        _install_sharded_fakes(monkeypatch, tmp_path,
                               _FakeArchive(archive_markets), self.CUTOFF)
        historical.fetch_all_settled_markets(
            MagicMock(), _FakeLive(live_markets), start_date=date(2026, 6, 5),
            use_cache=False,
        )
        cache_dir = tmp_path / "cache"
        # SS-1 changed the EXTENSION (the streamed .jsonl.gz format), not the
        # stem this test pins: the default True case is still unmarked, and a
        # legacy .json of that same stem is still what a hit falls back to.
        assert (cache_dir / "settled_markets_2026-06-05.jsonl.gz").exists()
        assert not (cache_dir / "settled_markets_2026-06-05_nomve.jsonl.gz").exists()

    def test_mve_flag_separates_assembled_cache_filenames(self, tmp_path, monkeypatch):
        # DR-57: the flag changes WHAT IS FETCHED (mve_filter="exclude" on the
        # archive query and on every live page), so the two settings must never
        # share a cache file — and the marker must compose with prefilter_tag,
        # which keys the same file on a different axis.
        from datetime import date

        archive_markets, live_markets = self._fixture_markets()

        def pred(m):
            return not m["ticker"].endswith("1")

        for flag, sub in ((True, "on"), (False, "off")):
            monkeypatch.setattr(historical, "INCLUDE_MVE_MARKETS", flag)
            _install_sharded_fakes(monkeypatch, tmp_path / sub,
                                   _FakeArchive(archive_markets), self.CUTOFF)
            historical.fetch_all_settled_markets(
                MagicMock(), _FakeLive(live_markets), start_date=date(2026, 6, 5),
                use_cache=False, prefilter=pred, prefilter_tag="testpred",
            )

        on_files = {p.name for p in (tmp_path / "on" / "cache").glob("settled_markets_*")}
        off_files = {p.name for p in (tmp_path / "off" / "cache").glob("settled_markets_*")}
        assert on_files == {"settled_markets_2026-06-05_testpred.jsonl.gz"}
        assert off_files == {"settled_markets_2026-06-05_testpred_nomve.jsonl.gz"}
        assert on_files.isdisjoint(off_files)

    def test_cache_written_under_one_mve_setting_is_not_served_to_the_other(
            self, tmp_path, monkeypatch):
        # The real-money-adjacent half of DR-57: before the marker existed, an
        # INCLUDE_MVE_MARKETS=False run loaded the MVE-INCLUSIVE assembly and
        # analysed an almost entirely MVE corpus while believing it had
        # excluded them — with nothing abnormal in the output to show for it.
        from datetime import date

        archive_markets, live_markets = self._fixture_markets()
        archive = _FakeArchive(archive_markets)

        monkeypatch.setattr(historical, "INCLUDE_MVE_MARKETS", True)
        _install_sharded_fakes(monkeypatch, tmp_path, archive, self.CUTOFF)
        historical.fetch_all_settled_markets(
            MagicMock(), _FakeLive(live_markets), start_date=date(2026, 6, 5),
            use_cache=False,
        )

        # Same flag, use_cache=True: the assembled cache is served, zero calls.
        archive.calls = 0
        historical.fetch_all_settled_markets(
            MagicMock(), _FakeLive(live_markets), start_date=date(2026, 6, 5),
            use_cache=True,
        )
        assert archive.calls == 0

        # Flipping the flag must MISS that cache and refetch.
        monkeypatch.setattr(historical, "INCLUDE_MVE_MARKETS", False)
        archive.calls = 0
        historical.fetch_all_settled_markets(
            MagicMock(), _FakeLive(live_markets), start_date=date(2026, 6, 5),
            use_cache=True,
        )
        assert archive.calls > 0
        cache_dir = tmp_path / "cache"
        assert (cache_dir / "settled_markets_2026-06-05.jsonl.gz").exists()
        assert (cache_dir / "settled_markets_2026-06-05_nomve.jsonl.gz").exists()

    def test_probe_exception_falls_back_to_sequential(self, tmp_path, monkeypatch, caplog):
        # The cursor-synthesis probe issues a real request, so it can fail for
        # reasons unrelated to cursor format. Such a failure used to escape the
        # phase and kill the run with no warning; it must degrade to the
        # sequential walk exactly like a format mismatch does.
        archive_markets, live_markets = self._fixture_markets()

        def exploding_probe(*_a, **_k):
            raise RuntimeError("probe blew up")

        monkeypatch.setattr(historical, "_archive_cursor_synthesis_ok", exploding_probe)
        with caplog.at_level(logging.WARNING):
            out = self._run(monkeypatch, tmp_path, _FakeArchive(archive_markets),
                            _FakeLive(live_markets))

        assert {m["ticker"] for m in out} == _expected_in_window(
            archive_markets, live_markets,
            self._ts(self.START + "T00:00:00+00:00"), self._ts(self.CUTOFF),
        )
        assert any("probe blew up" in r.getMessage() for r in caplog.records)
        # Fail closed: the sequential path must not fabricate day-slice files.
        assert not (tmp_path / "cache" / "archive_days").exists()

    def test_worker_failure_cancels_queued_days(self, tmp_path, monkeypatch):
        # A failing worker must abandon the remaining queued days instead of
        # letting the executor drain them (hundreds of days = hours) before the
        # fallback is reached.
        recorded: list[dict] = []
        real_pool_cls = historical.ThreadPoolExecutor

        class RecordingPool(real_pool_cls):
            def shutdown(self, wait=True, *, cancel_futures=False):
                recorded.append({"wait": wait, "cancel_futures": cancel_futures})
                return super().shutdown(wait=wait, cancel_futures=cancel_futures)

        monkeypatch.setattr(historical, "ThreadPoolExecutor", RecordingPool)
        monkeypatch.setattr(historical, "SETTLED_FETCH_MAX_WORKERS", 1)
        archive_markets, live_markets = self._fixture_markets()

        # Fail the first day worker, the way a server that ignores a
        # synthesized cursor would. Patched at the worker seam so the
        # sequential fallback (which pages normally) still works.
        def failing_worker(*_a, **_k):
            raise historical._ShardedFetchUnsupported("synthesized cursor rejected")

        monkeypatch.setattr(historical, "_fetch_and_store_archive_day", failing_worker)
        out = self._run(monkeypatch, tmp_path, _FakeArchive(archive_markets),
                        _FakeLive(live_markets))

        # The pool was torn down with cancel_futures, not drained.
        assert any(c["cancel_futures"] and not c["wait"] for c in recorded), recorded
        # And the fallback still produced the correct, complete result.
        assert {m["ticker"] for m in out} == _expected_in_window(
            archive_markets, live_markets,
            self._ts(self.START + "T00:00:00+00:00"), self._ts(self.CUTOFF),
        )

    def test_slice_progress_reports_position_and_eta(self, tmp_path, monkeypatch, caplog):
        # Each completed slice logs N/M plus a rate and ETA, so a multi-hour
        # fetch reports how far along it actually is. This fixture completes
        # its handful of slices essentially instantly, so the observed rate
        # is always comfortably above the 0.1 slices/min cutoff — the
        # "fast" branch, rendered as slices/min. See
        # test_slow_slice_progress_reports_slices_per_hour for the other branch.
        archive_markets, live_markets = self._fixture_markets()
        with caplog.at_level(logging.INFO):
            self._run(monkeypatch, tmp_path, _FakeArchive(archive_markets),
                      _FakeLive(live_markets))
        slice_lines = [r.getMessage() for r in caplog.records
                       if "Archive day slices:" in r.getMessage() and "complete" in r.getMessage()]
        assert slice_lines
        assert all("ETA" in line and "slices/min" in line for line in slice_lines)
        assert all("slices/hour" not in line for line in slice_lines)
        # Counter runs 1..N over the days actually fetched, never exceeding N.
        total = len(slice_lines)
        assert slice_lines[-1].split("complete")[0].strip().endswith(f"{total}/{total}")

    def test_slow_slice_progress_reports_slices_per_hour(self, monkeypatch, caplog):
        # BS-27: at rate < 0.1 slices/min, "%.1f slices/min" rounds to "0.0"
        # beside a perfectly finite ETA — reads as broken math, not "just
        # slow". Below that threshold the rate must render as slices/hour
        # instead; "0.0 slices/min" must never appear. Call the logger
        # directly with a huge elapsed time so the rate is controlled exactly,
        # rather than relying on real wall-clock slowness in a test.
        started = 0.0
        monkeypatch.setattr(historical.time, "monotonic", lambda: 100_000.0)
        with caplog.at_level(logging.INFO):
            historical._log_slice_progress("Archive day slices", 1, 1000, 0, 5, started)
        lines = [r.getMessage() for r in caplog.records
                 if "Archive day slices:" in r.getMessage() and "complete" in r.getMessage()]
        assert lines
        assert "slices/hour" in lines[0]
        assert "0.0 slices/min" not in lines[0]
        assert "ETA" in lines[0]

    def test_live_ignoring_max_settled_ts_falls_back(self, tmp_path, monkeypatch):
        # If the live endpoint stops honoring max_settled_ts, every window
        # would silently re-walk the whole range; the first window detects it
        # and the phase degrades to the original single sequential sweep.
        archive_markets, live_markets = self._fixture_markets()
        archive = _FakeArchive(archive_markets)
        live = _FakeLive(live_markets, ignore_max=True)
        out = self._run(monkeypatch, tmp_path, archive, live)
        expected = _expected_in_window(
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


class TestProgressLabels:
    """The sharded and sequential paths emit the same progress-line SHAPE, so
    their labels are the only thing telling them apart in a log. When the
    labels were identical, a live run on the sharded path was misdiagnosed as
    having fallen back to the sequential walk."""

    def test_fetch_progress_logs_its_label_every_100_pages(self, caplog):
        progress = historical._FetchProgress("Historical archive [sharded]")
        with caplog.at_level(logging.INFO):
            for _ in range(100):
                progress.tick(2)
        messages = [r.getMessage() for r in caplog.records]
        assert messages == [
            "Historical archive [sharded]: 100 pages scanned, 200 markets kept so far"
        ]

    def test_sharded_and_sequential_labels_differ(self):
        # Guards against a future edit re-converging the two labels.
        import inspect

        source = inspect.getsource(historical)
        assert 'Historical archive [sharded]' in source
        assert 'Historical archive [sequential]' in source
        # The tail is a third distinct phase, serial and uncached — its pages
        # must not be attributed to the parallel day-slice pool.
        assert 'Historical archive [tail]' in source
        assert 'Live settled sweep [windowed]' in source
        assert 'Live settled sweep [sequential]' in source
        # No un-suffixed variant left behind.
        assert '"Historical archive: %d pages' not in source
        assert '"Live settled sweep: %d pages' not in source


class TestFormatDuration:
    def test_renders_compact_units(self):
        assert historical._format_duration(0) == "0s"
        assert historical._format_duration(45.4) == "45s"
        assert historical._format_duration(90) == "1m30s"
        assert historical._format_duration(3600) == "1h00m"
        assert historical._format_duration(13_260) == "3h41m"

    def test_negative_clamps_to_zero(self):
        assert historical._format_duration(-5) == "0s"


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


class TestPruneStaleLiveDays:
    """BS-32: once Kalshi advances the archive cutoff, every day that now
    lies entirely before it is served (and cached) exclusively via
    archive_days/ from then on — the old live_days/ slice for that day is
    never read again by any future run and must be pruned."""

    def _day_ts(self, iso_date: str) -> int:
        d = date.fromisoformat(iso_date)
        return int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp())

    def _write_fake_slice(self, day_lo: int) -> Path:
        path = historical._day_store_path("live_days", day_lo)
        meta = {"kind": "live_settled_day", "cutoff_ts": 0,
                "include_mve": True, "complete": True}
        historical._day_store_save(path, meta, [{"ticker": "T1"}])
        return path

    def test_prunes_only_fully_pre_cutoff_days(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(historical, "CACHE_DIR", tmp_path)
        # Cutoff at noon on the 15th: the 14th is entirely archive-covered,
        # the 15th itself still has an afternoon that's live-only, the 16th
        # is entirely still live.
        cutoff_ts = self._day_ts("2026-08-15") + 12 * 3600

        pre_cutoff = self._write_fake_slice(self._day_ts("2026-08-14"))
        straddling = self._write_fake_slice(self._day_ts("2026-08-15"))
        post_cutoff = self._write_fake_slice(self._day_ts("2026-08-16"))

        with caplog.at_level(logging.INFO):
            pruned = historical._prune_stale_live_days(cutoff_ts)

        assert pruned == 1
        assert not pre_cutoff.exists()
        assert straddling.exists()
        assert post_cutoff.exists()
        assert any("Pruned 1 stale pre-cutoff live day slice" in r.getMessage()
                   for r in caplog.records)

    def test_missing_live_days_dir_is_a_noop(self, tmp_path, monkeypatch, caplog):
        # Fresh cache dir, or nothing fetched into live_days/ yet.
        monkeypatch.setattr(historical, "CACHE_DIR", tmp_path)
        with caplog.at_level(logging.INFO):
            pruned = historical._prune_stale_live_days(cutoff_ts=99_999_999_999)
        assert pruned == 0
        assert not any("Pruned" in r.getMessage() for r in caplog.records)

    def test_no_stale_days_logs_nothing(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(historical, "CACHE_DIR", tmp_path)
        cutoff_ts = self._day_ts("2026-08-15")
        kept = self._write_fake_slice(self._day_ts("2026-08-16"))

        with caplog.at_level(logging.INFO):
            pruned = historical._prune_stale_live_days(cutoff_ts)

        assert pruned == 0
        assert kept.exists()
        assert not any("Pruned" in r.getMessage() for r in caplog.records)

    def test_unlink_failure_is_logged_and_does_not_raise(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(historical, "CACHE_DIR", tmp_path)
        cutoff_ts = self._day_ts("2026-08-20")
        stale = self._write_fake_slice(self._day_ts("2026-08-14"))

        real_unlink = Path.unlink

        def failing_unlink(self, *a, **k):
            if self == stale:
                raise OSError("permission denied")
            return real_unlink(self, *a, **k)

        monkeypatch.setattr(Path, "unlink", failing_unlink)

        with caplog.at_level(logging.WARNING):
            pruned = historical._prune_stale_live_days(cutoff_ts)

        assert pruned == 0
        assert stale.exists()
        assert any("Failed to prune stale live-day slice" in r.getMessage()
                   for r in caplog.records)


class TestDayStreamWriter:
    """BS-15: day slices are streamed out in chunks so no fetch worker ever
    holds a whole UTC day (millions of records at 2026-08 volumes) in memory.

    The streamed "jsonl-v1" format and the legacy single-document format must
    both stay readable — hundreds of MB of legacy slices already sit in
    backtest_cache/ and refetching them costs hours.
    """

    META = {"kind": "archive_created_day", "cutoff_ts": 100,
            "include_mve": True, "complete": True}

    @staticmethod
    def _markets(n: int) -> list[dict]:
        return [{"ticker": f"T{i}", "result": "yes", "open_time": None} for i in range(n)]

    def test_jsonl_roundtrip_preserves_order(self, tmp_path):
        # Record order IS the contract: the server returns each window
        # newest-first and the caller's ticker dedup is first-wins, so a
        # reordering writer would silently change which duplicate survives.
        path = tmp_path / "2026-06-09.json.gz"
        markets = self._markets(5)
        with historical._DayStreamWriter(path, self.META) as writer:
            writer.write_records(markets[:2])
            writer.write_records(markets[2:])
            assert writer.commit() == 5
        assert historical._day_store_load(path, self.META) == markets
        # Line framing is real JSONL, not an implementation detail of ours
        assert _read_slice_file(path) == markets

    def test_jsonl_meta_carries_format_tag(self, tmp_path):
        path = tmp_path / "2026-06-09.json.gz"
        with historical._DayStreamWriter(path, self.META) as writer:
            writer.commit()
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            head = json.loads(fh.readline())
        assert head["meta"]["format"] == "jsonl-v1"
        # The tag rides along inside meta, so identity gating is unaffected
        assert all(head["meta"][k] == v for k, v in self.META.items())

    def test_empty_day_is_meta_only_and_loads_as_zero_records(self, tmp_path):
        # Routing edge: a meta-only file whole-parses as a perfectly good JSON
        # dict, so routing on "did json.loads succeed" would call this legacy
        # and report no markets key. It must route on CONTENT and load as [].
        path = tmp_path / "2026-06-07.json.gz"
        with historical._DayStreamWriter(path, self.META) as writer:
            writer.commit()
        assert historical._day_store_load(path, self.META) == []
        # ...and specifically not None, which would mean "refetch this day"
        assert historical._day_store_load(path, self.META) is not None

    def test_legacy_dict_slice_still_loads(self, tmp_path):
        # Written the way every pre-BS-15 slice on disk was written.
        path = tmp_path / "legacy.json.gz"
        markets = self._markets(3)
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump({"meta": self.META, "markets": markets}, fh)
        assert historical._day_store_load(path, self.META) == markets
        # Legacy empty day too — the other empty-file routing case
        empty = tmp_path / "legacy_empty.json.gz"
        with gzip.open(empty, "wt", encoding="utf-8") as fh:
            json.dump({"meta": self.META, "markets": []}, fh)
        assert historical._day_store_load(empty, self.META) == []

    def test_meta_mismatch_rejects_both_formats(self, tmp_path):
        drifted = {**self.META, "cutoff_ts": 200}
        jsonl = tmp_path / "jsonl.json.gz"
        with historical._DayStreamWriter(jsonl, self.META) as writer:
            writer.write_records(self._markets(2))
            writer.commit()
        legacy = tmp_path / "legacy.json.gz"
        historical._day_store_save(legacy, self.META, self._markets(2))

        assert historical._day_store_load(jsonl, drifted) is None
        assert historical._day_store_load(legacy, drifted) is None
        # Sanity: both load fine under the matching expectation
        assert historical._day_store_load(jsonl, self.META)
        assert historical._day_store_load(legacy, self.META)

    def test_abort_leaves_no_visible_file(self, tmp_path):
        # The atomicity contract: an interrupted worker must not publish a
        # partial slice that a later run would trust as complete.
        path = tmp_path / "2026-06-09.json.gz"
        with pytest.raises(RuntimeError):
            with historical._DayStreamWriter(path, self.META) as writer:
                writer.write_records(self._markets(2))
                raise RuntimeError("worker died mid-day")
        assert not path.exists()
        assert not list(tmp_path.glob("*.tmp"))
        assert historical._day_store_load(path, self.META) is None

    def test_truncated_jsonl_is_a_whole_slice_miss(self, tmp_path):
        # A slice cut mid-record must fail entirely (→ refetch), never return
        # the records that happened to survive.
        path = tmp_path / "2026-06-09.json.gz"
        with historical._DayStreamWriter(path, self.META) as writer:
            writer.write_records(self._markets(4))
            writer.commit()
        with gzip.open(path, "rb") as fh:
            raw = fh.read()
        cut = raw[: raw.rindex(b"\n", 0, len(raw) - 1) + 25]  # mid-line
        with gzip.open(path, "wb") as fh:
            fh.write(cut)
        assert historical._day_store_load(path, self.META) is None

    def test_keep_predicate_filters_during_load(self, tmp_path):
        # Pushed down so a multi-million-record day is never materialized just
        # to be filtered afterwards; must equal filtering the full list.
        path = tmp_path / "2026-06-09.json.gz"
        markets = self._markets(6)
        with historical._DayStreamWriter(path, self.META) as writer:
            writer.write_records(markets)
            writer.commit()

        def keep(m):
            return m["ticker"] in ("T1", "T4")

        assert historical._day_store_load(path, self.META, keep) == [
            m for m in markets if keep(m)
        ]
        # The same equality must hold for legacy slices
        legacy = tmp_path / "legacy.json.gz"
        historical._day_store_save(legacy, self.META, markets)
        assert historical._day_store_load(legacy, self.META, keep) == [
            m for m in markets if keep(m)
        ]
        # _discard_all is the prescan's "parse but retain nothing" probe
        assert historical._day_store_load(path, self.META, historical._discard_all) == []

    def test_worker_chunks_a_large_day(self, tmp_path, monkeypatch):
        # The point of BS-15: the worker's buffer is bounded by the chunk size,
        # not by how big the day is. Verified through the real worker with a
        # small chunk — emit must fire repeatedly, never once at the end.
        from datetime import UTC, datetime

        monkeypatch.setattr(historical, "SETTLED_FETCH_CHUNK_RECORDS", 3)
        monkeypatch.setattr(historical, "CACHE_DIR", tmp_path)

        day_lo = int(datetime(2026, 6, 9, tzinfo=UTC).timestamp())
        page_size = 2
        pages = [
            {"markets": [_mk_raw_market(f"T{i}", "2026-06-09T12:00:00Z",
                                        "2026-06-09T13:00:00Z")
                         for i in range(p * page_size, (p + 1) * page_size)],
             "cursor": "next"}
            for p in range(5)
        ]
        pages.append({"markets": [], "cursor": None})
        calls = iter(pages)
        monkeypatch.setattr(historical, "_historical_get",
                            lambda *a, **k: next(calls))

        batches: list[int] = []
        real_write = historical._DayStreamWriter.write_records

        def spy_write(self, batch):
            batches.append(len(batch))
            real_write(self, batch)

        monkeypatch.setattr(historical._DayStreamWriter, "write_records", spy_write)

        count = historical._fetch_and_store_archive_day(
            MagicMock(), day_lo, {"limit": 1000},
            historical._FetchProgress("test"), self.META,
        )

        assert count == 10
        # More than one flush, and no buffer grew past the chunk size plus the
        # page in flight (flushes land on page boundaries, so that — not the
        # size of the day — is the worker's memory bound).
        assert len(batches) > 1
        assert max(batches) <= 3 + page_size
        assert sum(batches) == 10
        # The published slice is complete and in fetch order
        path = historical._day_store_path("archive_days", day_lo)
        loaded = historical._day_store_load(path, self.META)
        assert [m["ticker"] for m in loaded] == [f"T{i}" for i in range(10)]

    def test_no_emit_returns_the_list_unchanged(self, tmp_path, monkeypatch):
        # Direct callers rely on the list-returning behavior — chunking must be
        # strictly opt-in. (No production path passes None since SS-1: the
        # frontier day streams through a keep-filtering sink, see
        # TestFrontierStreamsThroughKeep; the sequential fallbacks never call
        # this function.)
        pages = [
            {"markets": [_mk_raw_market("F1", "2026-06-12T01:00:00Z",
                                        "2026-06-12T02:00:00Z")],
             "cursor": None},
        ]
        calls = iter(pages)
        monkeypatch.setattr(
            historical, "api_call_with_retry",
            lambda fn, *a, **k: next(calls),
        )
        out = historical._fetch_live_window(
            MagicMock(), 0, None, historical._FetchProgress("test"),
        )
        assert isinstance(out, list)
        assert [m["ticker"] for m in out] == ["F1"]


class TestExtendKept:
    """The frontier's emit-sink body (SS-1): append only keep-passing records,
    in batch order, and never the batch list itself."""

    @staticmethod
    def _keep(m):
        return m["ticker"] != "B"

    def test_rejected_records_are_never_appended_and_order_holds(self):
        dest: list[dict] = []
        historical._extend_kept(dest, self._keep, [{"ticker": "A"}, {"ticker": "B"}])
        historical._extend_kept(dest, self._keep, [{"ticker": "C"}, {"ticker": "B"},
                                                   {"ticker": "D"}])
        assert [m["ticker"] for m in dest] == ["A", "C", "D"]

    def test_keep_none_appends_every_record_in_order(self):
        dest: list[dict] = [{"ticker": "X"}]
        batch = [{"ticker": "B"}, {"ticker": "A"}]
        historical._extend_kept(dest, None, batch)
        assert [m["ticker"] for m in dest] == ["X", "B", "A"]
        # The records are appended, not the batch list: the window drops its
        # buffer after each emit, and dest must not keep that list alive.
        assert all(m is not batch for m in dest)

    def test_the_same_record_objects_are_kept(self):
        # A filter, not a copy: downstream (title patching, first-wins dedup)
        # sees exactly the dicts the window produced.
        rec = {"ticker": "A"}
        dest: list[dict] = []
        historical._extend_kept(dest, self._keep, [rec])
        assert dest[0] is rec


class TestFrontierStreamsThroughKeep:
    """SS-1: the frontier (current UTC, never-persisted) day used to be held
    UNFILTERED in memory and filtered only after the whole pool drained — at
    real volumes a partial day of millions of records, for any window length.
    It now streams through _fetch_live_window's emit contract into a sink that
    keeps only `keep`-passing records as each batch lands. Membership and
    order must be exactly what the old post-hoc filter produced; only the peak
    changes."""

    NOW = "2026-09-24T12:00:00+00:00"
    TODAY = "2026-09-24T00:00:00+00:00"

    @staticmethod
    def _ts(iso):
        return int(datetime.fromisoformat(iso).timestamp())

    @staticmethod
    def _keep(m):
        # Discriminating on purpose: rejects roughly a third of every day's
        # records, on every page boundary pattern the fixtures produce.
        return not m["ticker"].endswith(("0", "3", "6", "9"))

    @staticmethod
    def _frontier_markets(n=11):
        # Settled across the frontier day before NOW, newest-first on the wire
        # (the fake sorts by settlement DESC), never on a midnight boundary.
        return [
            _mk_raw_market(f"F{i:02d}", "2026-09-23T00:00:00Z",
                           f"2026-09-24T{i + 1:02d}:00:00Z")
            for i in range(n)
        ]

    def _oracle(self, markets, keep):
        """The OLD frontier expression, literally: fetch the whole window as a
        list (emit=None), then filter it afterwards."""
        frontier = historical._fetch_live_window(
            _FakeLive(markets, page_size=2), self._ts(self.TODAY), None,
            historical._FetchProgress("oracle"),
        )
        assert isinstance(frontier, list)
        if keep is not None:
            frontier = [m for m in frontier if keep(m)]
        return frontier

    def test_frontier_equals_the_old_post_hoc_filter(self, tmp_path, monkeypatch):
        # Small chunks so the frontier is emitted as SEVERAL batches — a sink
        # that reordered or dropped a batch boundary would show up here.
        monkeypatch.setattr(historical, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(historical, "SETTLED_FETCH_CHUNK_RECORDS", 3)
        markets = self._frontier_markets()
        expected = self._oracle(markets, self._keep)

        # live_min_ts inside today => no past days, so the result IS the frontier.
        out = list(historical._fetch_live_phase(
            _FakeLive(markets, page_size=2), self._ts(self.TODAY) + 60,
            self._ts(self.NOW), self._keep,
        ))
        assert out == expected
        assert [m["ticker"] for m in out] == [m["ticker"] for m in expected]
        # Sanity: the predicate removed something and kept something.
        assert 0 < len(out) < len(markets)

    def test_keep_none_keeps_every_frontier_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(historical, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(historical, "SETTLED_FETCH_CHUNK_RECORDS", 3)
        markets = self._frontier_markets()
        out = list(historical._fetch_live_phase(
            _FakeLive(markets, page_size=2), self._ts(self.TODAY) + 60,
            self._ts(self.NOW), None,
        ))
        assert out == self._oracle(markets, None)
        assert len(out) == len(markets)

    def test_whole_phase_filters_only_the_frontier_and_counts_its_rejections(
            self, tmp_path, monkeypatch):
        # With past days on disk as well: frontier first, then past days
        # newest-first. Only the frontier is filtered by the phase (its records
        # would otherwise be spooled); the past days come back UNFILTERED since
        # M9, because the assembly applies the same predicate to them and
        # counts what it rejects — so filtering the phase's output still
        # equals filtering the unfiltered phase, and the tally holds exactly
        # the frontier's rejections, the only ones the assembly never sees.
        monkeypatch.setattr(historical, "SETTLED_FETCH_CHUNK_RECORDS", 3)
        past = [
            _mk_raw_market(f"P{d}{i}", "2026-09-21T00:00:00Z",
                           f"2026-09-2{d}T{i + 1:02d}:00:00Z")
            for d in (2, 3) for i in range(7)
        ]
        markets = self._frontier_markets() + past
        live_min_ts = self._ts("2026-09-22T00:00:00+00:00")

        monkeypatch.setattr(historical, "CACHE_DIR", tmp_path / "all")
        unfiltered = historical._fetch_live_phase(
            _FakeLive(markets, page_size=2), live_min_ts, self._ts(self.NOW), None,
        )
        monkeypatch.setattr(historical, "CACHE_DIR", tmp_path / "kept")
        tally = historical._AssemblyTally()
        filtered = historical._fetch_live_phase(
            _FakeLive(markets, page_size=2), live_min_ts, self._ts(self.NOW),
            self._keep, tally=tally,
        )

        everything, got = list(unfiltered), list(filtered)
        # What the assembly can keep is unchanged: it re-applies `keep`.
        assert [m for m in got if self._keep(m)] == [
            m for m in everything if self._keep(m)]
        tickers = [m["ticker"] for m in everything]
        # Frontier first, then 09-23, then 09-22 — the contract the merge's
        # first-wins dedup depends on.
        assert tickers[:11] == [f"F{i:02d}" for i in range(10, -1, -1)]
        assert tickers[11:18] == [f"P3{i}" for i in range(6, -1, -1)]
        assert tickers[18:] == [f"P2{i}" for i in range(6, -1, -1)]
        # The frontier is filtered where it is fetched; the past days are not.
        frontier_kept = [m for m in everything[:11] if self._keep(m)]
        assert got == frontier_kept + everything[11:]
        # The predicate bit on the frontier, and the past days still carry
        # records it rejects — they reach the assembly, which counts them.
        rejected_at_fetch = 11 - len(frontier_kept)
        assert rejected_at_fetch > 0
        assert any(not self._keep(m) for m in got[len(frontier_kept):])
        assert tally == historical._AssemblyTally(
            settled=rejected_at_fetch, rejected=rejected_at_fetch, duplicates=0)

    def test_rejected_frontier_record_is_never_retained(self, tmp_path, monkeypatch):
        # The point of SS-1. Every compact record is made weakly referenceable
        # (a plain dict is not), and each time the window requests its next
        # page we count how many REJECTED records are still alive. With one
        # flush per page, a record the predicate rejected must already be
        # garbage by the next request. The old code kept every rejected record
        # alive in the unfiltered frontier list until the pool drained, so this
        # count would climb page by page.
        import weakref

        monkeypatch.setattr(historical, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(historical, "SETTLED_FETCH_CHUNK_RECORDS", 1)

        class _Tracked(dict):
            """A compact record that supports weak references."""

        rejected_refs: list = []
        real_to_dict = historical._market_to_dict

        def tracking_to_dict(m, *args, **kwargs):
            rec = _Tracked(real_to_dict(m, *args, **kwargs))
            if not self._keep(rec):
                rejected_refs.append(weakref.ref(rec))
            return rec

        monkeypatch.setattr(historical, "_market_to_dict", tracking_to_dict)

        consulted: list[str] = []

        def keep(m):
            consulted.append(m["ticker"])
            return self._keep(m)

        alive_at_request: list[int] = []

        class _ObservedLive(_FakeLive):
            def get_markets_without_preload_content(self, *args, **kwargs):
                alive_at_request.append(
                    sum(ref() is not None for ref in rejected_refs))
                return super().get_markets_without_preload_content(*args, **kwargs)

        markets = self._frontier_markets()
        out = historical._fetch_live_phase(
            _ObservedLive(markets, page_size=2), self._ts(self.TODAY) + 60,
            self._ts(self.NOW), keep,
        )

        assert rejected_refs, "the fixture must produce rejected frontier records"
        assert len(alive_at_request) >= 3, "the frontier must span several pages"
        # Never a rejected record alive when the next page is requested...
        assert alive_at_request == [0] * len(alive_at_request)
        # ...and none survives the phase either.
        assert all(ref() is None for ref in rejected_refs)
        # Each frontier record was consulted exactly once, in fetch order, and
        # nothing the predicate rejected reached the result.
        assert consulted == [f"F{i:02d}" for i in range(10, -1, -1)]
        assert [m["ticker"] for m in out] == [t for t in consulted
                                              if self._keep({"ticker": t})]

    # ── Failure propagation (frontier_future.result() is load-bearing) ────────

    class _FailingFrontierLive(_FakeLive):
        """A live fake whose FRONTIER window (no max_settled_ts) raises on its
        Nth request; past-day windows (which send max_settled_ts) are served
        normally."""

        def __init__(self, markets, fail_on, exc, **kwargs):
            super().__init__(markets, **kwargs)
            self.fail_on = fail_on
            self.exc = exc
            self.frontier_calls = 0

        def get_markets_without_preload_content(self, min_settled_ts=None,
                                                max_settled_ts=None, cursor=None,
                                                **kwargs):
            if max_settled_ts is None:
                self.frontier_calls += 1
                if self.frontier_calls == self.fail_on:
                    raise self.exc
            return super().get_markets_without_preload_content(
                min_settled_ts=min_settled_ts, max_settled_ts=max_settled_ts,
                cursor=cursor, **kwargs)

    def _past_days(self):
        return [
            _mk_raw_market(f"P{d}{i}", "2026-09-21T00:00:00Z",
                           f"2026-09-2{d}T{i + 1:02d}:00:00Z")
            for d in (2, 3) for i in range(4)
        ]

    def test_a_frontier_fetch_failure_after_a_batch_is_raised(self, tmp_path,
                                                              monkeypatch):
        # The frontier's records reach `frontier` through the sink, so
        # frontier_future.result() no longer DELIVERS them — it is only there to
        # re-raise the window's failure. The pool's __exit__ waits for the
        # worker either way, so without it a frontier that dies part-way would
        # come back as the batches already appended plus every past day: a
        # silently SHORT corpus. A non-transient error (RuntimeError: no status,
        # not a transport class) is not retried, so it surfaces on its page.
        monkeypatch.setattr(historical, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(historical, "SETTLED_FETCH_CHUNK_RECORDS", 1)
        consulted: list[str] = []

        def keep(m):
            consulted.append(m["ticker"])
            return self._keep(m)

        live = self._FailingFrontierLive(
            self._frontier_markets() + self._past_days(), fail_on=4,
            exc=RuntimeError("frontier page 4 failed"), page_size=2,
        )
        with pytest.raises(RuntimeError, match="frontier page 4 failed"):
            historical._fetch_live_phase(
                live, self._ts("2026-09-22T00:00:00+00:00"), self._ts(self.NOW),
                keep,
            )
        # Batches really had been appended before the failure (pages 1-3 of
        # the frontier, one flush each) — the state a swallowed failure would
        # have returned as if it were the whole frontier.
        frontier_seen = [t for t in consulted if t.startswith("F")]
        assert frontier_seen == [f"F{i:02d}" for i in range(10, 4, -1)]
        assert any(self._keep({"ticker": t}) for t in frontier_seen)

    def test_keep_raising_on_the_frontier_worker_is_raised(self, tmp_path,
                                                           monkeypatch):
        # `keep` now runs on the frontier's WORKER thread, so its failure lands
        # in frontier_future, not in this thread — the same result() call is
        # the only thing that brings it back. Before SS-1 it raised here too,
        # from the post-hoc filter.
        monkeypatch.setattr(historical, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(historical, "SETTLED_FETCH_CHUNK_RECORDS", 1)

        def keep(m):
            if m["ticker"] == "F05":
                raise ValueError("keep failed on F05")
            return self._keep(m)

        with pytest.raises(ValueError, match="keep failed on F05"):
            historical._fetch_live_phase(
                _FakeLive(self._frontier_markets() + self._past_days(),
                          page_size=2),
                self._ts("2026-09-22T00:00:00+00:00"), self._ts(self.NOW), keep,
            )

    # ── Windowed-path fallback ────────────────────────────────────────────────

    def test_fallback_releases_the_partial_frontier_and_applies_keep(
            self, tmp_path, monkeypatch):
        # When the server stops honoring max_settled_ts, every past-day window
        # raises _ShardedFetchUnsupported and the phase falls back to the
        # sequential sweep — which refetches today too. By then the frontier
        # worker has run its whole window (the pool's __exit__ joins it), so
        # the keep-passing frontier used to sit beside the fallback's own copy
        # of the same day for the entire serial walk. It is released first now
        # (since the SS-1 review its records are never resident at all — they
        # are spooled — and the spool itself is closed, freeing its disk), and
        # the fallback gets the same `keep`.
        import weakref

        monkeypatch.setattr(historical, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(historical, "SETTLED_FETCH_CHUNK_RECORDS", 1)
        spools: list = []
        real_init = historical._FrontierSpool.__init__

        def recording_init(spool, directory):
            real_init(spool, directory)
            spools.append(spool)

        monkeypatch.setattr(historical._FrontierSpool, "__init__", recording_init)

        class _Tracked(dict):
            """A compact record that supports weak references."""

        kept_refs: list = []
        real_to_dict = historical._market_to_dict

        def tracking_to_dict(m, *args, **kwargs):
            rec = _Tracked(real_to_dict(m, *args, **kwargs))
            if self._keep(rec):
                kept_refs.append(weakref.ref(rec))
            return rec

        monkeypatch.setattr(historical, "_market_to_dict", tracking_to_dict)

        alive_at_fallback: list[int] = []
        real_sequential = historical._fetch_live_sequential

        def spy_sequential(live_client, live_min_ts, keep=None, **kwargs):
            # Every tracked record made so far came from the frontier window
            # (each past-day window raises on its first record, before
            # building one), so this counts the frontier still resident.
            alive_at_fallback.append(sum(r() is not None for r in kept_refs))
            return real_sequential(live_client, live_min_ts, keep, **kwargs)

        monkeypatch.setattr(historical, "_fetch_live_sequential", spy_sequential)

        markets = self._frontier_markets() + self._past_days()
        live_min_ts = self._ts("2026-09-22T00:00:00+00:00")
        tally = historical._AssemblyTally()
        out = historical._fetch_live_phase(
            _FakeLive(markets, page_size=2, ignore_max=True), live_min_ts,
            self._ts(self.NOW), self._keep, tally=tally,
        )

        # The frontier's first page is F10 and F09, and `keep` rejects F09, so
        # a spooled frontier record means the frontier had rejected one too.
        assert kept_refs, "the frontier must have kept records before the fallback"
        assert alive_at_fallback == [0]
        assert len(spools) == 1 and spools[0]._closed and len(spools[0]) > 0
        # The fallback result is exactly the sequential sweep, filtered.
        unfiltered = real_sequential(_FakeLive(markets, page_size=2), live_min_ts)
        assert out == [m for m in unfiltered if self._keep(m)]
        assert 0 < len(out) < len(unfiltered)
        # M9: only the sequential walk's rejections are counted — the
        # discarded frontier had rejected some of the same day's records too,
        # and counting them as well would report them twice.
        rejected = len(unfiltered) - len(out)
        assert tally == historical._AssemblyTally(
            settled=rejected, rejected=rejected, duplicates=0)


class TestSequentialFallbacksApplyKeep:
    """The two sequential fallbacks have no emit sink and hold their whole
    result in memory. They apply the caller's prefilter per record as each
    page arrives, so a record the merge would discard is never retained —
    and that must be exact: the same records in the same order as filtering
    the unfiltered result afterwards, with byte-identical progress lines (the
    "markets kept so far" count is taken before `keep`)."""

    START = "2026-09-17T00:00:00+00:00"
    CUTOFF = "2026-09-20T00:00:00+00:00"

    @staticmethod
    def _ts(iso):
        return int(datetime.fromisoformat(iso).timestamp())

    @staticmethod
    def _keep(m):
        return not m["ticker"].endswith(("0", "3", "6", "9"))

    @staticmethod
    def _stamp(base_iso, minutes):
        base = datetime.fromisoformat(base_iso)
        return (base + timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _markets(self, prefix, base_iso, n=105, step=1):
        # page_size=1 below, so n pages: past the 100-page progress cadence.
        # One voided record so the walk's own result filter bites as well.
        # `step` spaces the records in minutes (a multi-day spread is what lets
        # an ignored max_settled_ts trip the windowed path's fallback).
        out = [
            _mk_raw_market(f"{prefix}{i:03d}", self._stamp(base_iso, i * step),
                           self._stamp(base_iso, i * step + 30))
            for i in range(n)
        ]
        out[7]["result"] = "void"
        return out

    @staticmethod
    def _progress_lines(caplog, label):
        return [r.getMessage() for r in caplog.records if label in r.getMessage()]

    def test_live_sequential_applies_keep_exactly(self, caplog):
        markets = self._markets("L", "2026-09-22T00:00:00+00:00")
        live_min_ts = self._ts("2026-09-22T00:00:00+00:00")
        label = "Live settled sweep [sequential]"

        with caplog.at_level(logging.INFO):
            unfiltered = historical._fetch_live_sequential(
                _FakeLive(markets, page_size=1), live_min_ts)
        lines_none = self._progress_lines(caplog, label)
        caplog.clear()
        tally = historical._AssemblyTally()
        with caplog.at_level(logging.INFO):
            filtered = historical._fetch_live_sequential(
                _FakeLive(markets, page_size=1), live_min_ts, self._keep, tally=tally)
        lines_keep = self._progress_lines(caplog, label)

        assert filtered == [m for m in unfiltered if self._keep(m)]
        # M9: the assembly never sees what this walk dropped, so the walk
        # reports it — every one a record settled in the window
        dropped = len(unfiltered) - len(filtered)
        assert tally == historical._AssemblyTally(
            settled=dropped, rejected=dropped, duplicates=0)
        # The returned list IS what the walk retained: nothing rejected in it.
        assert all(self._keep(m) for m in filtered)
        assert 0 < len(filtered) < len(unfiltered)
        assert lines_keep == lines_none == [
            f"{label}: 100 pages scanned, 99 markets kept so far"
        ]

    def test_archive_sequential_applies_keep_exactly(self, monkeypatch, caplog):
        markets = self._markets("A", "2026-09-17T06:00:00+00:00")
        label = "Historical archive [sequential]"

        def run(keep, tally=None):
            archive = _FakeArchive(markets, page_size=1)
            monkeypatch.setattr(
                historical, "_signed_raw_get",
                lambda client, path, **params: _raw_resp(archive.page(**params)),
            )
            return historical._fetch_archive_sequential(
                MagicMock(), self._ts(self.START), self._ts(self.CUTOFF),
                {"limit": 1000}, keep, tally=tally)

        with caplog.at_level(logging.INFO):
            unfiltered = run(None)
        lines_none = self._progress_lines(caplog, label)
        caplog.clear()
        tally = historical._AssemblyTally()
        with caplog.at_level(logging.INFO):
            filtered = run(self._keep, tally)
        lines_keep = self._progress_lines(caplog, label)

        assert filtered == [m for m in unfiltered if self._keep(m)]
        # M9: the in-window records `keep` dropped, which the assembly never sees
        dropped = len(unfiltered) - len(filtered)
        assert tally == historical._AssemblyTally(
            settled=dropped, rejected=dropped, duplicates=0)
        assert all(self._keep(m) for m in filtered)
        assert 0 < len(filtered) < len(unfiltered)
        assert lines_keep == lines_none == [
            f"{label}: 100 pages scanned, 99 markets kept so far"
        ]

    def test_archive_phase_fallback_passes_keep(self, monkeypatch):
        # Opaque cursors defeat cursor synthesis, so the phase takes the
        # sequential fallback; its first list must already be keep-filtered.
        markets = self._markets("A", "2026-09-17T06:00:00+00:00", n=12)

        def run(keep):
            archive = _FakeArchive(markets, page_size=2, opaque_cursors=True)
            monkeypatch.setattr(
                historical, "_signed_raw_get",
                lambda client, path, **params: _raw_resp(archive.page(**params)),
            )
            return historical._fetch_archive_phase(
                MagicMock(), self._ts(self.START), self._ts(self.CUTOFF),
                {"limit": 1000}, keep)

        unfiltered, tail_none = run(None)
        filtered, tail_keep = run(self._keep)
        assert tail_none == tail_keep == []
        assert filtered == [m for m in unfiltered if self._keep(m)]
        assert 0 < len(filtered) < len(unfiltered)

    @pytest.mark.parametrize("opaque, ignore_max", [
        (True, False), (False, True), (True, True),
    ])
    def test_prefilter_through_the_fallbacks_equals_postfilter(
            self, tmp_path, monkeypatch, caplog, opaque, ignore_max):
        # End to end: whichever fallback fires, fetch_all_settled_markets with
        # a prefilter returns exactly the unfiltered result filtered afterwards.
        archive_markets = self._markets("A", "2026-09-17T06:00:00+00:00", n=12)
        # 3-hour spacing: 09-20 06:30 through 09-21 15:30, so with max_settled_ts
        # ignored the 09-20 window's first record lands a day past its ceiling.
        live_markets = self._markets("L", "2026-09-20T06:00:00+00:00", n=12,
                                     step=180)

        def fetch(sub, **kwargs):
            _install_sharded_fakes(
                monkeypatch, tmp_path / sub,
                _FakeArchive(archive_markets, page_size=2, opaque_cursors=opaque),
                "2026-09-20T00:00:00Z",
            )
            return historical.fetch_all_settled_markets(
                MagicMock(),
                _FakeLive(live_markets, page_size=2, ignore_max=ignore_max),
                start_date=date(2026, 9, 17), use_cache=False, **kwargs,
            )

        out_full = fetch("full")
        with caplog.at_level(logging.WARNING):
            out_pref = fetch("pref", prefilter=self._keep,
                             prefilter_tag="testpred")
        assert list(out_pref) == [m for m in out_full if self._keep(m)]
        assert 0 < len(out_pref) < len(out_full)
        _assert_counts_cover_the_prefilter(out_full, out_pref)
        assert {m["ticker"][0] for m in out_pref} == {"A", "L"}
        # The fixture really did take the fallback(s) it is parametrized for.
        assert ("Archive fetch: sharded path unavailable" in caplog.text) is opaque
        assert ("Live fetch: windowed path unavailable" in caplog.text) is ignore_max

    @pytest.mark.parametrize("opaque, ignore_max", [
        (False, False), (True, False), (False, True), (True, True),
    ])
    def test_a_frontier_that_rejects_records_is_counted_once_on_every_path(
            self, tmp_path, monkeypatch, caplog, opaque, ignore_max):
        # M9 end to end with a NON-EMPTY frontier. Every other end-to-end
        # count test runs at the real "now", whose frontier day holds no
        # fixture record, so the frontier's own rejections (its window's
        # emitted count minus the spool's length) were pinned only at phase
        # level. "Now" is pinned to 2026-09-21 12:00 UTC here: the 09-20
        # records are a past day (walk A counts their rejections), the 09-21
        # ones the frontier (its sink drops L006, L009 and L010 before the
        # assembly sees them). With max_settled_ts ignored, the past day trips
        # the live fallback AFTER the frontier has spooled — the ordering is
        # forced below — so a frontier whose rejections were counted anyway
        # would show 14 settled records instead of 11.
        archive_markets = self._markets("A", "2026-09-17T06:00:00+00:00", n=12)
        live_markets = self._markets("L", "2026-09-20T06:00:00+00:00", n=12,
                                     step=180)
        now_ts = self._ts("2026-09-21T12:00:00+00:00")
        real_phase = historical._fetch_live_phase
        monkeypatch.setattr(
            historical, "_fetch_live_phase",
            lambda client, lo, _now, keep=None, **kw: real_phase(
                client, lo, now_ts, keep, **kw),
        )
        real_extend = historical._extend_kept
        real_store = historical._fetch_and_store_live_window

        def fetch(sub, **kwargs):
            sink_dropped: list[int] = []
            frontier_done = threading.Event()

            def spy_extend(dest, keep, batch):
                # The frontier's sink: record what it drops, then let the
                # past-day windows run, so the fallback (when it fires) always
                # discards a frontier that has already spooled its batch.
                real_extend(dest, keep, batch)
                sink_dropped.append(0 if keep is None
                                    else sum(1 for m in batch if not keep(m)))
                frontier_done.set()

            def past_day_after_the_frontier(*args, **kw):
                assert frontier_done.wait(timeout=30), "the frontier never emitted"
                return real_store(*args, **kw)

            monkeypatch.setattr(historical, "_extend_kept", spy_extend)
            monkeypatch.setattr(historical, "_fetch_and_store_live_window",
                                past_day_after_the_frontier)
            _install_sharded_fakes(
                monkeypatch, tmp_path / sub,
                _FakeArchive(archive_markets, page_size=2, opaque_cursors=opaque),
                "2026-09-20T00:00:00Z",
            )
            out = historical.fetch_all_settled_markets(
                MagicMock(),
                _FakeLive(live_markets, page_size=2, ignore_max=ignore_max),
                start_date=date(2026, 9, 17), use_cache=False, **kwargs,
            )
            return out, sink_dropped

        with caplog.at_level(logging.INFO):
            out_full, dropped_full = fetch("full")
        full_text = caplog.text
        caplog.clear()
        with caplog.at_level(logging.INFO):
            out_pref, dropped_pref = fetch("pref", prefilter=self._keep,
                                           prefilter_tag="testpred")
        pref_text = caplog.text

        # The frontier really held records, and the prefilter really dropped
        # three of them in its sink (L007, voided, never reaches it).
        assert dropped_full == [0] and dropped_pref == [3]
        assert list(out_pref) == [m for m in out_full if self._keep(m)]
        _assert_counts_cover_the_prefilter(out_full, out_pref)
        # Exact, per endpoint: 11 binary live records, 5 of them rejected —
        # L000/L003 by walk A (past day, or the fallback's own walk) and
        # L006/L009/L010 in the frontier's sink (or, on the fallback, the
        # sequential walk that refetched the day). Counted once either way.
        assert ("Live endpoint: 11 settled markets of 11 recently settled records "
                "in the window (0 duplicate or blank tickers)") in full_text
        assert ("Live endpoint: 6 eligible markets of 11 recently settled records "
                "in the window (5 rejected by the prefilter testpred, 0 duplicate "
                "or blank tickers)") in pref_text
        assert ("Live fetch: windowed path unavailable" in pref_text) is ignore_max
        assert ("Archive fetch: sharded path unavailable" in pref_text) is opaque


class TestJsonCacheDurability:
    """BS-08: the plain-JSON cache helpers must fail safe, not fail loud.

    These caches are written at the end of multi-hour fetches that have
    historically been killed by OOM and SIGKILL, so a half-written file is a
    realistic on-disk state. Reads treat it as a miss; writes can never produce
    it in the first place.
    """

    def test_missing_file_is_a_miss(self, tmp_path):
        assert historical._load_json_cache(tmp_path / "nope.json") is None

    def test_roundtrip(self, tmp_path):
        path = tmp_path / "sub" / "cache.json"  # parent dirs created on save
        historical._save_json_cache(path, {"a": [1, 2]})
        assert historical._load_json_cache(path) == {"a": [1, 2]}

    def test_corrupt_file_is_a_miss_with_warning(self, tmp_path, caplog):
        path = tmp_path / "cache.json"
        path.write_text('[{"ticker": "T1"')  # truncated mid-write
        with caplog.at_level(logging.WARNING):
            assert historical._load_json_cache(path) is None
        assert any("Corrupt JSON cache" in r.getMessage() and "cache.json" in r.getMessage()
                   for r in caplog.records)

    def test_leftover_tmp_without_real_file_is_a_miss(self, tmp_path):
        # A run killed between the tmp write and the rename leaves only the
        # sidecar. The real path is still absent, so this is a plain miss —
        # the partial tmp must never be read as if it were the cache.
        path = tmp_path / "cache.json"
        path.with_name(path.name + ".tmp").write_text('[{"ticker": "T1"')
        assert historical._load_json_cache(path) is None

    def test_serializer_failure_leaves_no_partial_file(self, tmp_path):
        # json.dumps calls default=str for unserializable values; this one
        # explodes there, i.e. mid-save.
        class _Boom:
            def __str__(self):
                raise RuntimeError("serializer blew up")

        path = tmp_path / "cache.json"
        historical._save_json_cache(path, {"good": 1})
        with pytest.raises(RuntimeError):
            historical._save_json_cache(path, {"bad": _Boom()})
        # The previously good file is intact — no truncation in place.
        assert historical._load_json_cache(path) == {"good": 1}

    def test_write_crash_never_reaches_the_real_path(self, tmp_path, monkeypatch):
        # Simulate the disk-full / SIGKILL case: half the bytes land, then the
        # write raises. With tmp+replace, the damaged bytes are confined to the
        # sidecar and the real path never becomes visible-but-truncated.
        real_write_text = Path.write_text

        def half_then_die(self, data, *args, **kwargs):
            real_write_text(self, data[: len(data) // 2])
            raise OSError("no space left on device")

        path = tmp_path / "cache.json"
        monkeypatch.setattr(Path, "write_text", half_then_die)
        with pytest.raises(OSError):
            historical._save_json_cache(path, {"ticker": "T1", "candles": [1, 2, 3]})
        monkeypatch.undo()

        assert not path.exists()
        assert historical._load_json_cache(path) is None


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


class TestCandleClose:
    def test_candle_numeric_zero_close_dollars_is_used_not_fallthrough(self):
        # Regression: `ya.get("close_dollars") or ya.get("close")` treated a
        # valid falsy close_dollars (numeric 0) as absent and silently read
        # the legacy field instead.
        assert historical._candle_close({"close_dollars": 0, "close": "55"}) == 0.0

    def test_candle_empty_string_close_dollars_falls_back_to_close(self):
        # An empty string is absence-of-value, not a price — keep the fallback
        assert historical._candle_close({"close_dollars": "", "close": "0.55"}) == 0.55

    def test_candle_missing_both_closes_is_none(self):
        assert historical._candle_close({}) is None
        assert historical._candle_close({"close_dollars": None, "close": None}) is None

    def test_candle_unparseable_close_dollars_is_none_not_fallthrough(self):
        # A present-but-garbage close_dollars means the payload shape is off;
        # don't guess from the legacy field.
        assert historical._candle_close({"close_dollars": "n/a", "close": "0.55"}) is None

    def test_candle_missing_close_skipped_in_fetch(self, tmp_path, monkeypatch):
        # Through-path: a candle whose sides carry no close at all is skipped
        # rather than raising or emitting a bogus price.
        monkeypatch.setattr(historical, "_CANDLES_DIR", tmp_path / "candles")
        mock = MagicMock(return_value={
            "candlesticks": [
                {"end_period_ts": 1, "yes_ask": {}, "yes_bid": {}},
                {"end_period_ts": 2, "yes_ask": {"close": "0.55"}, "yes_bid": {"close": "0.53"}},
            ]
        })
        monkeypatch.setattr(historical, "_historical_get", mock)
        out = historical.fetch_candlesticks(
            MagicMock(), "T1", open_ts=0, close_ts=2, use_cache=False, rate_limit_sleep=0.0,
        )
        assert [c["ts"] for c in out] == [2]


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

    def test_malformed_candle_dropped_and_logged(self, tmp_path, monkeypatch, caplog):
        # BS-23: a malformed candle used to be silently swallowed by a bare
        # `except: pass`, so a thinned series was cached as if it were
        # complete with no visible signal. One good candle + one unparseable
        # candle (yes_ask.close is not a number) must keep the good candle,
        # drop the bad one, and log exactly what was dropped.
        monkeypatch.setattr(historical, "_CANDLES_DIR", tmp_path / "candles")
        payload = {"candlesticks": [
            {"end_period_ts": 1_700_000_000,
             "yes_ask": {"close": "0.55"}, "yes_bid": {"close": "0.53"}},
            {"end_period_ts": 1_700_003_600,
             "yes_ask": {"close": "not-a-number"}, "yes_bid": {"close": "0.50"}},
        ]}
        monkeypatch.setattr(historical, "_signed_raw_get",
                            MagicMock(return_value=_raw_resp(payload)))

        with caplog.at_level(logging.WARNING):
            out = historical.fetch_candlesticks(
                MagicMock(), "T1", open_ts=0, close_ts=2, use_cache=False,
                rate_limit_sleep=0.0,
            )

        assert len(out) == 1
        assert out[0]["ts"] == 1_700_000_000
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("T1" in w and "dropped 1/2 malformed candles" in w for w in warnings)

    def test_clean_candle_series_logs_no_drop_warning(self, tmp_path, monkeypatch, caplog):
        # The flip side of the malformed-candle test: a fully clean series
        # must never emit the drop warning (dropped == 0 is silent).
        monkeypatch.setattr(historical, "_CANDLES_DIR", tmp_path / "candles")
        _patch_candle_fetch(monkeypatch, 1_700_000_000)

        with caplog.at_level(logging.WARNING):
            out = historical.fetch_candlesticks(
                MagicMock(), "T1", open_ts=0, close_ts=2, use_cache=False,
                rate_limit_sleep=0.0,
            )

        assert len(out) == 1
        assert not any("dropped" in r.getMessage() for r in caplog.records)

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

    def test_corrupt_cache_falls_through_to_refetch(self, tmp_path, monkeypatch, caplog):
        # BS-08: the per-ticker cache read used to be a bare json.loads OUTSIDE
        # the fetch try-block, so one truncated file raised straight out of a
        # candlestick worker thread and killed the whole backtest. A damaged
        # file must behave exactly like a cache miss.
        candles_dir = tmp_path / "candles"
        monkeypatch.setattr(historical, "_CANDLES_DIR", candles_dir)
        candles_dir.mkdir(parents=True)
        (candles_dir / "T1.json").write_text(
            '{"open_ts": 0, "close_ts": 1000, "period_interval": 60, "candles": [{"ts"'
        )
        fetch = _patch_candle_fetch(monkeypatch, 1_700_000_000)

        with caplog.at_level(logging.WARNING):
            out = historical.fetch_candlesticks(
                MagicMock(), "T1", open_ts=100, close_ts=200, rate_limit_sleep=0.0,
            )

        assert fetch.call_count == 1
        assert out[0]["ts"] == 1_700_000_000
        assert any("Corrupt JSON cache" in r.getMessage() for r in caplog.records)
        # The refetch rewrites the file in the current tagged format.
        assert json.loads((candles_dir / "T1.json").read_text())["candles"] == out

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

    def test_fetch_failure_logs_one_line_without_header_dump(
        self, tmp_path, monkeypatch, caplog,
    ):
        # TS-02: post-cutoff tickers 404 by design and those failures are
        # deliberately never cached, so this warning is re-paid every run for
        # every such ticker. Logging the SDK exception whole emitted ~900
        # bytes across five lines each time, which rotated the run's own
        # diagnostics out of kalshi_backtest.log.
        monkeypatch.setattr(historical, "_CANDLES_DIR", tmp_path / "candles")
        monkeypatch.setattr(historical, "_signed_raw_get",
                            MagicMock(side_effect=_FakeApiException()))

        with caplog.at_level(logging.WARNING):
            out = historical.fetch_candlesticks(
                MagicMock(), "T1", open_ts=0, close_ts=2, use_cache=False,
                rate_limit_sleep=0.0,
            )

        assert out == []                        # fail-soft contract unchanged
        msgs = [r.getMessage() for r in caplog.records
                if "Candlestick fetch failed" in r.getMessage()]
        assert len(msgs) == 1
        assert "T1" in msgs[0]
        assert "HTTP 404 Not Found" in msgs[0]
        # One physical line, and none of the header dump the SDK's __str__ emits
        assert "\n" not in msgs[0]
        assert "X-Big-Header" not in msgs[0]
        assert "HTTP response headers" not in msgs[0]
        assert len(msgs[0]) < 200
        # Still not cached — a poisoned empty file would silence this ticker
        assert not (tmp_path / "candles" / "T1.json").exists()


_HOUR = 3600
_CAP = historical.CANDLESTICK_MAX_CANDLES_PER_REQUEST
# The longest span one request may cover: one period short of the cap.
_PAGE = (_CAP - 1) * historical.CANDLESTICK_PERIOD_INTERVAL_MINUTES * 60


class _CandleEndpoint:
    """Fake /historical/markets/{ticker}/candlesticks, cap included.

    Holds one candle per hour over [first_ts, last_ts] and, like the real
    endpoint, refuses (HTTP 400) any request that could hold more than
    CANDLESTICK_MAX_CANDLES_PER_REQUEST candles — counted the strict way, with
    both ends inclusive, so a request that passes here passes whichever way the
    real endpoint counts. `start_inclusive` / `end_inclusive` pick which of the
    four undocumented window conventions it serves.
    """

    def __init__(self, first_ts, last_ts, *, start_inclusive=True, end_inclusive=True,
                 fail_on_call=None, fail_status=404):
        self.series = list(range(first_ts, last_ts + 1, _HOUR))
        self.start_inclusive = start_inclusive
        self.end_inclusive = end_inclusive
        self.fail_on_call = fail_on_call or {}
        self.fail_status = fail_status
        self.calls: list[tuple[int, int]] = []

    def served(self, start, end) -> list[int]:
        """The candle timestamps one request for [start, end] returns, cap aside."""
        lo = (lambda t: t >= start) if self.start_inclusive else (lambda t: t > start)
        hi = (lambda t: t <= end) if self.end_inclusive else (lambda t: t < end)
        return [t for t in self.series if lo(t) and hi(t)]

    def __call__(self, _client, _path, **params):
        start, end = params["start_ts"], params["end_ts"]
        self.calls.append((start, end))
        call_no = len(self.calls)
        if self.fail_on_call.get(call_no, 0) > 0:
            self.fail_on_call[call_no] -= 1
            self.calls.pop()  # a refused attempt is retried under the same number
            raise _FakeApiException(self.fail_status, "Injected")
        period = params["period_interval"] * 60
        if (end - start) // period + 1 > _CAP:
            raise _FakeApiException(400, "Bad Request")
        return _raw_resp({"candlesticks": [
            {"end_period_ts": t, "yes_ask": {"close": "0.55"}, "yes_bid": {"close": "0.53"}}
            for t in self.served(start, end)
        ]})


class TestCandleRequestWindows:
    """historical._candle_request_windows: one request unless the window is
    longer than the endpoint serves, then overlapping requests within the cap."""

    def test_a_window_that_fits_is_one_unchanged_request(self):
        assert historical._candle_request_windows(1_000, 1_000 + _PAGE) == [
            (1_000, 1_000 + _PAGE)]
        assert historical._candle_request_windows(0, 2) == [(0, 2)]

    def test_a_degenerate_window_is_passed_through(self):
        assert historical._candle_request_windows(500, 100) == [(500, 100)]

    @pytest.mark.parametrize("span", [_PAGE + 1, 2 * _PAGE, 400 * 86_400, 15_527 * _HOUR])
    def test_a_longer_window_is_covered_by_overlapping_requests_within_the_cap(self, span):
        open_ts = 1_700_000_000
        windows = historical._candle_request_windows(open_ts, open_ts + span)
        assert len(windows) >= 2
        assert windows[0][0] == open_ts
        assert windows[-1][1] == open_ts + span
        for (s1, e1), (s2, _e2) in zip(windows, windows[1:], strict=False):
            # Each request stays within the cap, and the next one starts one
            # candle period before this one ends.
            assert 0 < e1 - s1 <= _PAGE
            assert s2 == e1 - _HOUR
        assert 0 < windows[-1][1] - windows[-1][0] <= _PAGE


class TestFetchCandlesticksPaging:
    """The candlestick endpoint refuses a request spanning more than 5,000
    candles (HTTP 400), and fetch_candlesticks used to send every window as ONE
    request, so any window longer than ~208 days of hourly candles came back
    as "no candles". A longer window is now paged and merged."""

    OPEN = 1_700_000_000 - 1_700_000_000 % _HOUR

    def _fetch(self, monkeypatch, tmp_path, endpoint, open_ts, close_ts, **kw):
        monkeypatch.setattr(historical, "_CANDLES_DIR", tmp_path / "candles")
        monkeypatch.setattr(historical, "_signed_raw_get", endpoint)
        return historical.fetch_candlesticks(
            MagicMock(), "T1", open_ts=open_ts, close_ts=close_ts,
            rate_limit_sleep=0.0, **kw)

    def test_the_fake_endpoint_refuses_one_request_for_the_whole_window(self):
        # The premise of every test below: one GET for this window is a 400.
        close = self.OPEN + 400 * 86_400
        endpoint = _CandleEndpoint(self.OPEN, close)
        with pytest.raises(_FakeApiException):
            endpoint(None, "", start_ts=self.OPEN, end_ts=close, period_interval=60)

    @pytest.mark.parametrize("start_inclusive", [True, False])
    @pytest.mark.parametrize("end_inclusive", [True, False])
    def test_a_long_window_returns_exactly_what_one_uncapped_request_would(
        self, monkeypatch, tmp_path, start_inclusive, end_inclusive,
    ):
        close = self.OPEN + 400 * 86_400  # 9,600 hourly candles, ~2 caps
        endpoint = _CandleEndpoint(self.OPEN, close, start_inclusive=start_inclusive,
                                   end_inclusive=end_inclusive)
        out = self._fetch(monkeypatch, tmp_path, endpoint, self.OPEN, close,
                          use_cache=False)

        assert len(endpoint.calls) == 2
        # Every candle one request would have served, once each, ascending
        assert [c["ts"] for c in out] == endpoint.served(self.OPEN, close)
        assert out[0]["yes_ask_close"] == pytest.approx(0.55)
        # Cached ONCE, under the whole window
        cached = json.loads((tmp_path / "candles" / "T1.json").read_text())
        assert (cached["open_ts"], cached["close_ts"]) == (self.OPEN, close)
        assert cached["candles"] == out

    def test_a_window_that_fits_is_still_one_request_with_the_same_parameters(
        self, monkeypatch, tmp_path,
    ):
        close = self.OPEN + _PAGE
        endpoint = _CandleEndpoint(self.OPEN, close)
        out = self._fetch(monkeypatch, tmp_path, endpoint, self.OPEN, close,
                          use_cache=False)
        assert endpoint.calls == [(self.OPEN, close)]
        assert len(out) == _CAP

    def test_the_longest_measured_market_span_is_fetched_whole(self, monkeypatch, tmp_path):
        # 15,527 hours: the longest open-to-close-plus-a-day span among the
        # DR-73 calibration corpus's 3,704 time-series legs, and one of the
        # 115 that no single request could serve.
        close = self.OPEN + 15_527 * _HOUR
        endpoint = _CandleEndpoint(self.OPEN, close)
        out = self._fetch(monkeypatch, tmp_path, endpoint, self.OPEN, close,
                          use_cache=False)
        assert len(endpoint.calls) == 4
        assert [c["ts"] for c in out] == endpoint.served(self.OPEN, close)

    def test_a_cached_paged_window_is_served_without_any_request(self, monkeypatch, tmp_path):
        close = self.OPEN + 400 * 86_400
        first = self._fetch(monkeypatch, tmp_path, _CandleEndpoint(self.OPEN, close),
                            self.OPEN, close)
        again = _CandleEndpoint(self.OPEN, close)
        assert self._fetch(monkeypatch, tmp_path, again, self.OPEN, close) == first
        assert again.calls == []

    def test_a_failed_later_request_returns_nothing_and_caches_nothing(
        self, monkeypatch, tmp_path, caplog,
    ):
        close = self.OPEN + 400 * 86_400
        endpoint = _CandleEndpoint(self.OPEN, close, fail_on_call={2: 1})
        with caplog.at_level(logging.WARNING):
            out = self._fetch(monkeypatch, tmp_path, endpoint, self.OPEN, close,
                              use_cache=False)

        # All-or-nothing: the first request's candles are NOT returned or
        # cached as if they were the whole window.
        assert out == []
        assert not (tmp_path / "candles" / "T1.json").exists()
        msgs = [r.getMessage() for r in caplog.records
                if "Candlestick fetch failed" in r.getMessage()]
        assert len(msgs) == 1
        assert "T1" in msgs[0] and "HTTP 404" in msgs[0]
        assert msgs[0].endswith("(request 2 of 2)")

    def test_a_single_request_failure_keeps_its_exact_line(self, monkeypatch, tmp_path, caplog):
        close = self.OPEN + 10 * _HOUR
        endpoint = _CandleEndpoint(self.OPEN, close, fail_on_call={1: 1})
        with caplog.at_level(logging.WARNING):
            assert self._fetch(monkeypatch, tmp_path, endpoint, self.OPEN, close,
                               use_cache=False) == []
        msgs = [r.getMessage() for r in caplog.records
                if "Candlestick fetch failed" in r.getMessage()]
        assert msgs == ["Candlestick fetch failed for T1: HTTP 404 Injected"]

    def test_each_request_is_retried_on_a_transient_error(self, monkeypatch, tmp_path):
        # Every request goes through _historical_get's api_call_with_retry, so
        # a 503 on the SECOND request is backed off and retried rather than
        # failing the whole window.
        from kalshi_betting import _http
        sleeps: list[float] = []
        monkeypatch.setattr(_http.time, "sleep", sleeps.append)
        close = self.OPEN + 400 * 86_400
        endpoint = _CandleEndpoint(self.OPEN, close, fail_on_call={2: 1}, fail_status=503)
        out = self._fetch(monkeypatch, tmp_path, endpoint, self.OPEN, close,
                          use_cache=False)
        assert [c["ts"] for c in out] == endpoint.served(self.OPEN, close)
        assert any(s > 0 for s in sleeps)  # the retry backed off

    def test_malformed_candles_are_counted_across_every_request(
        self, monkeypatch, tmp_path, caplog,
    ):
        close = self.OPEN + 400 * 86_400
        endpoint = _CandleEndpoint(self.OPEN, close)
        bad_ts = {self.OPEN, close}  # one in the first request, one in the last

        def _with_bad(client, path, **params):
            resp = endpoint(client, path, **params)
            payload = json.loads(resp.data)
            for c in payload["candlesticks"]:
                if c["end_period_ts"] in bad_ts:
                    c["yes_ask"] = {"close": "not-a-number"}
            return _raw_resp(payload)

        with caplog.at_level(logging.WARNING):
            out = self._fetch(monkeypatch, tmp_path, _with_bad, self.OPEN, close,
                              use_cache=False)
        served = endpoint.served(self.OPEN, close)
        assert [c["ts"] for c in out] == [t for t in served if t not in bad_ts]
        # One line for the whole window; the raw count includes the overlap's
        # repeats, which is what the requests actually returned.
        raw = sum(len(endpoint.served(s, e)) for s, e in endpoint.calls)
        warnings = [r.getMessage() for r in caplog.records if "malformed" in r.getMessage()]
        assert warnings == [f"T1: dropped 2/{raw} malformed candles"]


class TestMergeCandlePages:
    def test_sorts_ascending_and_keeps_the_first_copy_of_a_repeat(self):
        a = {"ts": 2, "src": "first"}
        b = {"ts": 2, "src": "second"}
        merged = historical._merge_candle_pages(
            [{"ts": 1}, a, {"ts": 3}, b, {"ts": 0}])
        assert [c["ts"] for c in merged] == [0, 1, 2, 3]
        assert merged[2] is a


class TestExceptionSummary:
    """_exception_summary: one line, never the SDK's header dump (TS-02)."""

    def test_prefers_the_reason_attribute(self):
        # The SDK exception's own first line is only "(404)" — the useful half
        # is `reason`, which is why it is preferred over str(exc).
        assert historical._exception_summary(_FakeApiException()) == "Not Found"

    def test_falls_back_to_the_message_for_a_plain_exception(self):
        # A non-API failure (a parse error, a KeyError) carries no `reason`;
        # reducing it to its class name alone would throw away the diagnosis.
        assert historical._exception_summary(ValueError("bad payload")) == "bad payload"

    def test_falls_back_to_the_class_name_when_there_is_no_text(self):
        assert historical._exception_summary(ValueError()) == "ValueError"

    def test_keeps_only_the_first_line(self):
        assert historical._exception_summary(
            ValueError("first line\nsecond line\nthird")) == "first line"

    def test_truncates_at_the_limit(self):
        summary = historical._exception_summary(ValueError("y" * 500))
        assert len(summary) == 120
        assert summary == "y" * 120

    def test_limit_is_configurable(self):
        assert historical._exception_summary(ValueError("y" * 500), limit=10) == "y" * 10

    def test_never_raises_even_on_a_pathological_exception(self):
        # Every caller invokes this from INSIDE an `except` block on a
        # per-ticker path, and a _fetch_candles_parallel worker exception kills
        # the whole backtest by that function's deliberate design — so a
        # rendering failure here must degrade to the class name, never escape.
        class ExplodingStr(Exception):
            def __str__(self):
                raise RuntimeError("__str__ is broken")

        class ExplodingReason(Exception):
            @property
            def reason(self):
                raise RuntimeError("reason is broken")

        assert historical._exception_summary(ExplodingStr()) == "ExplodingStr"
        assert historical._exception_summary(ExplodingReason()) == "ExplodingReason"


class TestArchivePhaseSkippedWhenProvablyEmpty:
    """
    TS-25: the archive holds only markets that settled BEFORE the cutoff, so a
    window starting at or after it cannot contain a single archive record. The
    phase still ran the cursor-synthesis probe and the tail walk to prove that,
    costing ~18 seconds and ~50,000 parsed records on every post-cutoff run.
    """

    @staticmethod
    def _ts(iso: str) -> int:
        from datetime import datetime
        return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())

    def test_no_request_is_made_when_the_window_starts_at_the_cutoff(
        self, monkeypatch, caplog,
    ):
        def _boom(*a, **k):
            raise AssertionError("no archive request may be made")

        monkeypatch.setattr(historical, "_signed_raw_get", _boom)
        cutoff = self._ts("2026-06-04T00:00:00Z")
        with caplog.at_level(logging.INFO):
            slices, tail = historical._fetch_archive_phase(
                MagicMock(), cutoff, cutoff, {"limit": 1000},
            )
        assert slices == []
        assert tail == []
        assert "Historical archive phase skipped" in caplog.text

    def test_no_request_is_made_when_the_window_starts_after_the_cutoff(
        self, monkeypatch, caplog,
    ):
        def _boom(*a, **k):
            raise AssertionError("no archive request may be made")

        monkeypatch.setattr(historical, "_signed_raw_get", _boom)
        with caplog.at_level(logging.INFO):
            slices, tail = historical._fetch_archive_phase(
                MagicMock(),
                self._ts("2026-07-01T00:00:00Z"),
                self._ts("2026-06-04T00:00:00Z"),
                {"limit": 1000},
            )
        assert (slices, tail) == ([], [])
        assert "Historical archive phase skipped" in caplog.text

    def test_a_pre_cutoff_window_still_runs_the_phase(self, monkeypatch, caplog):
        # GUARD: the skip must be exact. A window that genuinely overlaps the
        # archive still does the work — proved by the synthesis probe firing.
        probed = []
        monkeypatch.setattr(
            historical, "_archive_cursor_synthesis_ok",
            lambda *a, **k: probed.append(True) or False,
        )
        monkeypatch.setattr(
            historical, "_fetch_archive_sequential", lambda *a, **k: [],
        )
        with caplog.at_level(logging.INFO):
            historical._fetch_archive_phase(
                MagicMock(),
                self._ts("2026-06-01T00:00:00Z"),
                self._ts("2026-06-04T00:00:00Z"),
                {"limit": 1000},
            )
        assert probed == [True]
        assert "Historical archive phase skipped" not in caplog.text


class TestEventTitlesReturnsMergedView:
    """
    TS-11: _load_or_build_event_titles wrote `merged` to disk but returned
    `cached` — this run's resolution alone. Under --no-cache every ticker the
    listings missed and the EVENT_TITLE_FALLBACK_MAX_LOOKUPS cap skipped came
    back as the "" poison pill even though disk held a real title. Measured:
    771,601 unresolved against a 5,000 cap, so ~99% of stragglers. Those
    markets then group by market title alone, collapsing the same-title key
    (event_title, title, subtitle) toward the bare title — the direction that
    manufactures cross-event false positives under the 95% co-resolution prior.
    """

    def test_accumulator_answers_a_ticker_this_run_could_not_resolve(
        self, isolated_cache, monkeypatch,
    ):
        isolated_cache.write_text(json.dumps({"E1": "Real Title", "E2": "Other"}))
        client = _make_client_with_event_pages(non_mve_pages=[], mve_pages=[])
        _patch_single_event_lookups(monkeypatch, single_failures={"E1"})

        result = historical._load_or_build_event_titles(
            client, {"E1"}, use_cache=False,
        )
        assert result == {"E1": "Real Title"}

    def test_result_is_restricted_to_the_requested_tickers(
        self, isolated_cache, monkeypatch,
    ):
        # GUARD: the accumulator is a cross-run store of every ticker ever
        # resolved. Returning it whole would hand a caller asking about one
        # ticker hundreds of thousands of unrelated entries.
        isolated_cache.write_text(json.dumps({"E1": "Real Title", "E2": "Other"}))
        client = _make_client_with_event_pages(non_mve_pages=[], mve_pages=[])
        _patch_single_event_lookups(monkeypatch, single_failures={"E1"})

        result = historical._load_or_build_event_titles(
            client, {"E1"}, use_cache=False,
        )
        assert set(result) == {"E1"}

    def test_a_ticker_nobody_has_ever_resolved_still_maps_to_the_pill(
        self, isolated_cache, monkeypatch,
    ):
        # GUARD: the merge must not invent titles. An unresolved ticker with no
        # disk entry keeps poison-pill semantics.
        isolated_cache.write_text(json.dumps({"E1": "Real Title"}))
        client = _make_client_with_event_pages(non_mve_pages=[], mve_pages=[])
        _patch_single_event_lookups(monkeypatch, single_failures={"UNKNOWN"})

        result = historical._load_or_build_event_titles(
            client, {"UNKNOWN"}, use_cache=False,
        )
        assert result == {"UNKNOWN": ""}

    def test_the_substitution_is_counted_in_the_log(
        self, isolated_cache, monkeypatch, caplog,
    ):
        isolated_cache.write_text(json.dumps({"E1": "Real Title"}))
        client = _make_client_with_event_pages(non_mve_pages=[], mve_pages=[])
        _patch_single_event_lookups(monkeypatch, single_failures={"E1"})

        with caplog.at_level(logging.INFO):
            historical._load_or_build_event_titles(client, {"E1"}, use_cache=False)
        assert "1 of this run's tickers answered from the accumulator" in caplog.text


class TestEventTitleAccumulatorBound:
    """
    DR-51 / DR-42: the event-title accumulator stores genuine answers only.

    Before DR-51 every ticker the per-ticker fallback's cap skipped was stored
    as "" — tickers nobody ever looked up — so one fresh 7-day run
    (2026-09-24) grew backtest_cache/event_titles.json from 3,996,906 to
    7,986,570 keys, 7,918,449 of them KXMVE combo tickers mapped to "", and
    every later fetch parsed the whole file and held two full copies of it.
    The one sorted cap also spent its 5,000 lookups on whatever sorted first,
    so a non-combo event sorting after "KXMVE" was skipped along with millions
    of combos. And the closing summary counted the whole seeded accumulator as
    "this run" (DR-42). These tests pin what replaced all of it: deferred
    tickers are not stored, a genuine failure still is, the cap goes to
    non-combo tickers first, lookups are paced, the accumulator is merged in
    place and rewritten only when it changed, the summary counts this call, and
    the legacy file is migrated exactly once.
    """

    @staticmethod
    def _combo(n: int) -> str:
        return f"KXMVECROSSCATEGORY-SHARD1-S{n:04d}"

    def test_deferred_tickers_are_not_stored_and_a_later_run_takes_the_next_slice(
        self, isolated_cache, monkeypatch,
    ):
        monkeypatch.setattr(historical, "EVENT_TITLE_FALLBACK_MAX_LOOKUPS", 3)
        client = _make_client_with_event_pages(non_mve_pages=[], mve_pages=[])
        wanted = {f"E{i:02d}" for i in range(7)}
        fallback = _patch_single_event_lookups(
            monkeypatch, single_lookups={t: f"Title {t}" for t in wanted},
        )

        historical._load_or_build_event_titles(client, wanted)
        assert set(json.loads(isolated_cache.read_text())) == {"E00", "E01", "E02"}

        # The deferred four are unknown, not pilled, so the next run looks
        # them up — the sorted cap reaching the next slice.
        second = historical._load_or_build_event_titles(
            _make_client_with_event_pages(non_mve_pages=[], mve_pages=[]), wanted,
        )
        assert fallback.call_count == 6
        assert set(json.loads(isolated_cache.read_text())) == {
            "E00", "E01", "E02", "E03", "E04", "E05",
        }
        assert second["E05"] == "Title E05" and second["E06"] == ""

    def test_a_genuine_failure_is_still_stored_beside_deferred_tickers(
        self, isolated_cache, monkeypatch,
    ):
        # GUARD: only the never-looked-up tickers lose their pill. A lookup
        # that was made and FAILED is a genuine answer and is stored, so it is
        # not re-paid every run.
        monkeypatch.setattr(historical, "EVENT_TITLE_FALLBACK_MAX_LOOKUPS", 2)
        client = _make_client_with_event_pages(non_mve_pages=[], mve_pages=[])
        _patch_single_event_lookups(
            monkeypatch, single_lookups={"E00": "Title E00"},
            single_failures={"E01"},
        )
        historical._load_or_build_event_titles(client, {"E00", "E01", "E02"})
        assert json.loads(isolated_cache.read_text()) == {
            "E00": "Title E00", "E01": "",
        }

    def test_over_the_cap_the_lookups_go_to_non_combo_tickers(
        self, isolated_cache, monkeypatch, caplog,
    ):
        # M5 on the 2026-09-24 run: one sorted list spent the whole cap on the
        # tickers before "KXMVE" plus the head of the combo block, so
        # KXNFLEVERYWEEKCOMPETE-27 (sorting after it) was skipped and its 34
        # markets went untitled. Non-combo tickers now take the cap first, and
        # combos — whose titles have no measured effect on pairing — are
        # deferred whole, under every KXMVE* prefix (scanner.event_series).
        monkeypatch.setattr(historical, "EVENT_TITLE_FALLBACK_MAX_LOOKUPS", 3)
        combos = {self._combo(i) for i in range(4)} | {
            "KXMVESPORTSMULTIGAMEEXTENDED-SHARD1-S0001",
        }
        non_combo = {"AAA-26", "KXNFLEVERYWEEKCOMPETE-27", "KXPRIMARYTURNOUT-26"}
        client = _make_client_with_event_pages(non_mve_pages=[], mve_pages=[])
        looked_up: list[str] = []

        def fake_signed_get(_client, path, **_params):
            tkr = path.rsplit("/", 1)[-1]
            looked_up.append(tkr)
            return _raw_resp({"event": {"title": f"Title {tkr}"}})

        monkeypatch.setattr(historical, "_signed_raw_get",
                            MagicMock(side_effect=fake_signed_get))
        with caplog.at_level(logging.INFO):
            result = historical._load_or_build_event_titles(client, combos | non_combo)

        assert sorted(looked_up) == sorted(non_combo)
        assert all(result[t] == f"Title {t}" for t in non_combo)
        assert all(result[t] == "" for t in combos)
        # Every non-combo fit, so nothing pairing-relevant was lost: INFO, not
        # WARNING — a line that fires on every bulk window must not warn.
        assert "deferring the 5 combo (KXMVE-family) tickers" in caplog.text
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
        # And no combo was stored.
        assert set(json.loads(isolated_cache.read_text())) == non_combo

    def test_under_the_cap_every_ticker_is_looked_up_combos_included(
        self, isolated_cache, monkeypatch,
    ):
        # GUARD: the combo deferral applies only when the miss set does not
        # fit. Under the cap nothing changes from before DR-51.
        monkeypatch.setattr(historical, "EVENT_TITLE_FALLBACK_MAX_LOOKUPS", 10)
        wanted = {self._combo(i) for i in range(3)} | {"AAA-26", "ZZZ-26"}
        client = _make_client_with_event_pages(non_mve_pages=[], mve_pages=[])
        fallback = _patch_single_event_lookups(
            monkeypatch, single_lookups={t: f"Title {t}" for t in wanted},
        )
        result = historical._load_or_build_event_titles(client, wanted)
        assert fallback.call_count == 5
        assert result == {t: f"Title {t}" for t in wanted}

    def test_every_lookup_is_paced_success_or_failure(self, isolated_cache, monkeypatch):
        monkeypatch.setattr(historical, "EVENT_TITLE_FALLBACK_RATE_LIMIT_SLEEP_SECONDS",
                            0.125)
        sleeps: list[float] = []
        monkeypatch.setattr(historical.time, "sleep", sleeps.append)
        client = _make_client_with_event_pages(non_mve_pages=[], mve_pages=[])
        _patch_single_event_lookups(
            monkeypatch, single_lookups={"E1": "One", "E2": "Two", "E3": "Three"},
            single_failures={"E4"},
        )
        historical._load_or_build_event_titles(client, {"E1", "E2", "E3", "E4"})
        assert sleeps == [0.125] * 4

    def test_the_summary_counts_this_call_with_the_cache_on(
        self, isolated_cache, monkeypatch, caplog,
    ):
        # DR-42: with use_cache on, the old line reported the whole seeded
        # accumulator as "N this run" (7,986,570 on the 2026-09-24 run, which
        # resolved at most ~7,525) and "0 answered from the accumulator" by
        # construction. The five counts now partition the request.
        monkeypatch.setattr(historical, "EVENT_TITLE_FALLBACK_MAX_LOOKUPS", 2)
        isolated_cache.write_text(json.dumps(
            {"D1": "Disk Title", "D2": "", "OTHER": "Unrelated"}))
        client = _make_client_with_event_pages(
            non_mve_pages=[[("L1", "Listed Title")]], mve_pages=[])
        _patch_single_event_lookups(
            monkeypatch, single_lookups={"N1": "Looked Up"},
            single_failures={"N2"},
        )
        with caplog.at_level(logging.INFO):
            result = historical._load_or_build_event_titles(
                client, {"D1", "D2", "L1", "N1", "N2", "N3"})

        assert result == {"D1": "Disk Title", "D2": "", "L1": "Listed Title",
                          "N1": "Looked Up", "N2": "", "N3": ""}
        assert (
            "Event titles for 6 requested tickers: 2 resolved by this run (1 from "
            "the bulk listings, 1 from per-ticker lookups), 1 of this run's "
            "tickers answered from the accumulator, 3 untitled (2 recorded as "
            "unresolvable, 1 deferred and not stored). Accumulator: 3 -> 6 "
            "entries."
        ) in caplog.text

    def test_the_accumulator_is_merged_in_place_not_copied(
        self, isolated_cache, monkeypatch,
    ):
        # The old code held the parsed file, a full seed copy and a full merge
        # copy at once — the title phase's peak on a fresh fetch. The object
        # loaded must be the object saved.
        isolated_cache.write_text("{}")
        loaded = {"D1": "Disk Title"}
        monkeypatch.setattr(historical, "_read_title_file",
                            lambda _path: (loaded, historical._TITLE_FILE_OK, ""))
        saved: list = []
        monkeypatch.setattr(historical, "_save_json_cache",
                            lambda path, data: saved.append((path, data)))
        client = _make_client_with_event_pages(
            non_mve_pages=[[("L1", "Listed Title")]], mve_pages=[])
        _patch_single_event_lookups(monkeypatch)

        historical._load_or_build_event_titles(client, {"D1", "L1"})

        assert len(saved) == 1
        assert saved[0][0] == isolated_cache
        assert saved[0][1] is loaded
        assert loaded == {"D1": "Disk Title", "L1": "Listed Title"}

    def test_an_unchanged_accumulator_is_not_rewritten(self, isolated_cache, monkeypatch):
        isolated_cache.write_text(json.dumps({"D1": "Disk Title", "D2": ""}))
        saves: list = []
        monkeypatch.setattr(historical, "_save_json_cache",
                            lambda path, data: saves.append(path))
        client = _make_client_with_event_pages(non_mve_pages=[], mve_pages=[])
        result = historical._load_or_build_event_titles(client, {"D1", "D2"})
        assert result == {"D1": "Disk Title", "D2": ""}
        assert saves == []

    def test_an_empty_request_reads_nothing(self, isolated_cache, monkeypatch):
        legacy = isolated_cache.with_name("event_titles.json")
        legacy.write_text(json.dumps({"A": "Title A", "B": ""}))
        client = _make_client_with_event_pages(non_mve_pages=[], mve_pages=[])
        assert historical._load_or_build_event_titles(client, set()) == {}
        assert legacy.exists() and not isolated_cache.exists()

    def test_the_legacy_file_is_migrated_once_keeping_exactly_its_titles(
        self, isolated_cache, monkeypatch, caplog,
    ):
        legacy = isolated_cache.with_name("event_titles.json")
        legacy.write_text(json.dumps({
            "A": "Title A", "B": "", self._combo(1): "", "D": "Title D", "E": 5,
        }))
        client = _make_client_with_event_pages(
            non_mve_pages=[[("B", "Title B")]], mve_pages=[])
        _patch_single_event_lookups(monkeypatch)

        with caplog.at_level(logging.INFO):
            result = historical._load_or_build_event_titles(client, {"A", "B"})

        # A legacy "" is dropped, so B is unknown again and re-resolved.
        assert result == {"A": "Title A", "B": "Title B"}
        assert json.loads(isolated_cache.read_text()) == {
            "A": "Title A", "D": "Title D", "B": "Title B",
        }
        assert not legacy.exists()
        assert "keeping its 2 titled entries and dropping the other 3" in caplog.text

        # Once the v2 file exists the legacy file is never read again, even if
        # an older build writes one back.
        legacy.write_text(json.dumps({"Z": "Legacy Z"}))
        fallback = _patch_single_event_lookups(monkeypatch, single_failures={"Z"})
        again = historical._load_or_build_event_titles(
            _make_client_with_event_pages(non_mve_pages=[], mve_pages=[]), {"Z"})
        assert again == {"Z": ""}
        assert fallback.call_count == 1
        assert legacy.exists()

    def test_a_migration_whose_run_resolves_nothing_still_commits(
        self, isolated_cache, monkeypatch,
    ):
        legacy = isolated_cache.with_name("event_titles.json")
        legacy.write_text(json.dumps({"A": "Title A", "B": ""}))
        client = _make_client_with_event_pages(non_mve_pages=[], mve_pages=[])
        assert historical._load_or_build_event_titles(client, {"A"}) == {"A": "Title A"}
        assert client.get_events_without_preload_content.call_count == 0
        assert json.loads(isolated_cache.read_text()) == {"A": "Title A"}
        assert not legacy.exists()

    def test_a_legacy_file_with_bad_content_is_left_in_place(
        self, isolated_cache, monkeypatch, caplog,
    ):
        # Content that is not a JSON object will not heal, so the migration
        # gives up: the v2 file is written as the marker (or every run would
        # re-read the whole damaged file and store nothing), the legacy file is
        # kept for inspection, and the WARNING names the way back.
        legacy = isolated_cache.with_name("event_titles.json")
        legacy.write_text('{"A": "Half A Titl')
        client = _make_client_with_event_pages(non_mve_pages=[], mve_pages=[])
        _patch_single_event_lookups(monkeypatch, single_failures={"A"})
        with caplog.at_level(logging.WARNING):
            assert historical._load_or_build_event_titles(client, {"A"}) == {"A": ""}
        assert "could not be read as a JSON object" in caplog.text
        assert "delete " + str(isolated_cache) + " and re-run" in caplog.text
        # Nothing of it was destroyed, and the v2 file now exists as the marker.
        assert legacy.read_text() == '{"A": "Half A Titl'
        assert json.loads(isolated_cache.read_text()) == {"A": ""}

    @staticmethod
    def _unreadable(monkeypatch, path):
        """Make reading `path` raise the OSError an offline iCloud file or a
        permission error raises; every other file reads normally."""
        real_read_text = Path.read_text

        def read_text(self, *args, **kwargs):
            if self == path:
                raise PermissionError(13, "Permission denied", str(path))
            return real_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", read_text)
        return lambda: monkeypatch.setattr(Path, "read_text", real_read_text)

    def test_a_legacy_file_that_cannot_be_read_defers_the_migration(
        self, isolated_cache, monkeypatch, caplog,
    ):
        # P1 review: a READ failure (an OSError) used to be folded into "not a
        # JSON object", so the v2 marker was written with this run's answers
        # alone and the legacy titles were abandoned for good. Now nothing is
        # written, and the next readable run migrates.
        legacy = isolated_cache.with_name("event_titles.json")
        legacy.write_text(json.dumps({"A": "Title A", "B": ""}))
        restore = self._unreadable(monkeypatch, legacy)
        client = _make_client_with_event_pages(
            non_mve_pages=[[("N", "Title N")]], mve_pages=[])
        _patch_single_event_lookups(monkeypatch)
        with caplog.at_level(logging.INFO):
            result = historical._load_or_build_event_titles(client, {"A", "N"})
        assert result == {"A": "", "N": "Title N"}
        assert not isolated_cache.exists()
        assert json.loads(legacy.read_bytes()) == {"A": "Title A", "B": ""}
        assert "the next run retries it" in caplog.text
        assert "PermissionError" in caplog.text
        assert "(NOT written: a file could not be read" in caplog.text

        restore()
        again = historical._load_or_build_event_titles(
            _make_client_with_event_pages(non_mve_pages=[[("N", "Title N")]],
                                          mve_pages=[]), {"A", "N"})
        assert again == {"A": "Title A", "N": "Title N"}
        assert json.loads(isolated_cache.read_text()) == {"A": "Title A", "N": "Title N"}
        assert not legacy.exists()

    def test_a_v2_file_that_cannot_be_read_is_not_overwritten(
        self, isolated_cache, monkeypatch, caplog,
    ):
        # Same rule for the accumulator itself: the old code overwrote it with
        # this run's answers alone, destroying every title it could not see.
        isolated_cache.write_text(json.dumps({"D1": "Disk Title"}))
        before = isolated_cache.read_bytes()
        self._unreadable(monkeypatch, isolated_cache)
        client = _make_client_with_event_pages(
            non_mve_pages=[[("L1", "Listed Title")]], mve_pages=[])
        _patch_single_event_lookups(monkeypatch)
        with caplog.at_level(logging.WARNING):
            result = historical._load_or_build_event_titles(client, {"D1", "L1"})
        assert result == {"D1": "", "L1": "Listed Title"}
        assert isolated_cache.read_bytes() == before
        assert "the next run reads it again" in caplog.text

    def test_a_legacy_file_beside_the_v2_file_is_named_every_call_and_kept(
        self, isolated_cache, monkeypatch, caplog,
    ):
        # P1 review: a legacy file that outlives the migration (a failed or
        # interrupted delete, or an older build writing it back) was ignored
        # in silence forever. It is still never read and never deleted — it
        # may be that older build's live accumulator — but every call names it.
        isolated_cache.write_text(json.dumps({"D1": "Disk Title"}))
        legacy = isolated_cache.with_name("event_titles.json")
        legacy.write_text(json.dumps({"Z": "Legacy Z"}))
        _patch_single_event_lookups(monkeypatch, single_failures={"Z"})
        for _ in range(2):
            caplog.clear()
            with caplog.at_level(logging.WARNING):
                result = historical._load_or_build_event_titles(
                    _make_client_with_event_pages(non_mve_pages=[], mve_pages=[]),
                    {"D1", "Z"})
            assert result == {"D1": "Disk Title", "Z": ""}
            warned = [r.getMessage() for r in caplog.records
                      if "sits beside event_titles_v2.json" in r.getMessage()]
            assert len(warned) == 1 and str(legacy) in warned[0]
            assert json.loads(legacy.read_text()) == {"Z": "Legacy Z"}

        # And no warning at all once it is gone.
        legacy.unlink()
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            historical._load_or_build_event_titles(
                _make_client_with_event_pages(non_mve_pages=[], mve_pages=[]), {"D1"})
        assert "sits beside" not in caplog.text

    def test_repeated_no_cache_runs_advance_through_the_deferred_tail(
        self, isolated_cache, monkeypatch, caplog,
    ):
        # P1 review: under use_cache=False every requested ticker is
        # re-resolved, so a plainly sorted cap looked up the same head on every
        # --no-cache run and never reached the tail. Tickers the accumulator
        # has never answered now take the cap first, then its stored pills
        # (A00 sorts first but was already looked up and failed), then the
        # tickers it already titles.
        monkeypatch.setattr(historical, "EVENT_TITLE_FALLBACK_MAX_LOOKUPS", 3)
        isolated_cache.write_text(json.dumps({"A00": ""}))
        wanted = {f"E{i:02d}" for i in range(7)} | {"A00"}
        looked_up: list[list[str]] = []

        def fake_signed_get(_client, path, **_params):
            looked_up[-1].append(path.rsplit("/", 1)[-1])
            return _raw_resp({"event": {"title": "Title " + path.rsplit("/", 1)[-1]}})

        monkeypatch.setattr(historical, "_signed_raw_get",
                            MagicMock(side_effect=fake_signed_get))
        results = []
        for _ in range(3):
            looked_up.append([])
            caplog.clear()
            with caplog.at_level(logging.WARNING):
                results.append(historical._load_or_build_event_titles(
                    _make_client_with_event_pages(non_mve_pages=[], mve_pages=[]),
                    wanted, use_cache=False))
        assert [sorted(batch) for batch in looked_up] == [
            ["E00", "E01", "E02"], ["E03", "E04", "E05"], ["A00", "E00", "E06"],
        ]
        assert json.loads(isolated_cache.read_text()) == {t: f"Title {t}" for t in wanted}
        # The third run deferred E01..E05, which the accumulator titles, so
        # they keep their titles — the WARNING must not call them untitled.
        assert results[2] == {t: f"Title {t}" for t in wanted}
        assert "is not looked up this run" in caplog.text
        assert "untitled THIS run" not in caplog.text

    def test_the_suite_never_reaches_the_real_accumulator(self, tmp_path):
        # tests/conftest.py's autouse guard: a test that reaches the real
        # function through fetch_all_settled_markets must not read, migrate or
        # delete the operator's real backtest_cache files.
        assert tmp_path in historical._EVENT_TITLES_CACHE.parents
        assert tmp_path in historical._LEGACY_EVENT_TITLES_CACHE.parents


# ─── SS-1 Commit C: the streamed assembly and the streamed assembled cache ────
#
# A 7-day backtest (--start-date 2026-09-17) could not fit on a 16 GB host:
# 7,260,952 of its 18,061,549 fetched past-day records passed the backtester's
# prefilter, at a measured 3,926 B/record (~28 GB), and fetch_all_settled_
# markets assembled them into a dict, a list copy and one whole-list
# json.dumps. The phases now return lazy views, the assembly streams them
# twice through one generator that must reproduce the old merge EXACTLY, and
# the corpus is handed back as a SettledCorpus streaming the new
# settled_markets_*.jsonl.gz cache. These tests pin exactness against the old
# merge, the all-or-nothing cache contract, the no-fallback/no-short-corpus
# rule, and that nothing is materialized.

_SS1C_META = {"kind": "archive_created_day", "cutoff_ts": 100,
              "include_mve": True, "complete": True}


def _write_jsonl(path, meta, records):
    """Publish one jsonl-v1 file through the real writer; return its path."""
    with historical._DayStreamWriter(path, meta) as writer:
        writer.write_records(list(records))
        writer.commit()
    return path


def _day_lo(iso_date):
    d = date.fromisoformat(iso_date)
    return int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp())


class _Tracked(dict):
    """A parsed record that supports weak references (a plain dict does not)."""


def _track_disk_records(monkeypatch):
    """Make every dict _slice_loads parses observable; return an alive-counter.

    Only records READ FROM DISK are tracked (day slices, the live frontier's
    spool and the assembled cache, plus the meta line of whichever slice or
    cache file is open). The archive tail is a list built by _market_to_dict,
    the one documented in-memory residual of these fixtures, so it is
    deliberately not counted.
    """
    refs: list = []
    real = historical._slice_loads

    def tracking(raw):
        value = real(raw)
        if isinstance(value, dict):
            value = _Tracked(value)
            refs.append(weakref.ref(value))
        return value

    monkeypatch.setattr(historical, "_slice_loads", tracking)
    return lambda: sum(ref() is not None for ref in refs)


def _completed_reads(monkeypatch):
    """Record each path whose _day_store_iter walk ran to completion."""
    reads: list[str] = []
    real = historical._day_store_iter

    def counting(path, expect_meta, keep=None):
        yield from real(path, expect_meta, keep)
        reads.append(str(path))

    monkeypatch.setattr(historical, "_day_store_iter", counting)
    return reads


def _old_assembly(day_records, tail_records, live_records, start_ts, cutoff_ts,
                  prefilter, titles_for):
    """fetch_all_settled_markets' assembly BEFORE SS-1, verbatim in substance.

    The ticker-keyed `selected` dict and the three _merge calls, the two log
    counts, the unique-event-ticker set, the list copy and the event_title
    patch gated on `titles` being truthy. Works on deep copies so the caller's
    fixtures are never patched. Returns (markets, archive_count, live_count,
    unique_event_tickers).
    """
    day_records, tail_records, live_records = (
        copy.deepcopy(list(day_records)), copy.deepcopy(list(tail_records)),
        copy.deepcopy(list(live_records)))
    selected: dict = {}

    def _merge(records, max_settle):
        for m in records:
            settle = historical._iso_epoch(m.get("settlement_ts"))
            if settle is None or settle < start_ts:
                continue
            if max_settle is not None and settle >= max_settle:
                continue
            if prefilter is not None and not prefilter(m):
                continue
            ticker = m.get("ticker")
            if ticker and ticker not in selected:
                selected[ticker] = m

    _merge(day_records, cutoff_ts)
    _merge(tail_records, cutoff_ts)
    archive_count = len(selected)
    _merge(live_records, None)
    live_count = len(selected) - archive_count
    unique = {m.get("event_ticker") for m in selected.values() if m.get("event_ticker")}
    titles = titles_for(unique)
    all_markets = list(selected.values())
    if titles:
        for m in all_markets:
            m["event_title"] = titles.get(m.get("event_ticker") or "", "")
    return all_markets, archive_count, live_count, unique


def _assert_counts_cover_the_prefilter(out_full, out_pref):
    """M9's end-to-end invariant, for one fixture assembled twice.

    `out_full` was assembled with no prefilter and `out_pref` with one, from
    the same records. The records settled in the window are the same records
    either way, so the two assemblies must report the same `settled` count —
    whichever path fetched them and wherever the prefilter dropped them: a
    rejection hidden from the count (the old read-back filter) makes it
    smaller, and one counted twice (a discarded frontier beside the fallback
    that refetches its day) makes it larger. Every market the prefilter cost
    is rejected at least once, and each count keeps exactly its corpus.
    """
    full = out_full.provenance.assembly_counts
    pref = out_pref.provenance.assembly_counts
    assert full.settled == pref.settled
    assert (full.rejected, full.kept) == (0, len(out_full))
    assert pref.kept == len(out_pref)
    assert pref.rejected >= len(out_full) - len(out_pref) > 0


def _old_assembly_counts(endpoints, start_ts, prefilter):
    """The old merge's walk, counting instead of keeping (M9's oracle).

    `endpoints` is one [(records, max_settle)] source list per endpoint, in
    merge order; the ticker dedup spans all of them, as the old merge's one
    `selected` dict did. Every record whose settlement lies in
    [start_ts, max_settle) is SETTLED; of those, the ones the prefilter
    refuses are REJECTED, and the ones that pass but whose ticker is blank or
    already kept are DUPLICATES. Returns one historical.AssemblyCounts per
    endpoint.
    """
    seen: set = set()
    out = []
    for sources in endpoints:
        settled = rejected = duplicates = 0
        for records, max_settle in sources:
            for m in records:
                settle = historical._iso_epoch(m.get("settlement_ts"))
                if settle is None or settle < start_ts:
                    continue
                if max_settle is not None and settle >= max_settle:
                    continue
                settled += 1
                if prefilter is not None and not prefilter(m):
                    rejected += 1
                    continue
                ticker = m.get("ticker")
                if ticker and ticker not in seen:
                    seen.add(ticker)
                else:
                    duplicates += 1
        out.append(historical.AssemblyCounts(settled, rejected, duplicates))
    return out


def _mapped_titles(tickers):
    """A title map shaped like the real resolver's: every requested ticker,
    with one left unresolved ("") so the .get(..., "") default path runs."""
    return {t: ("" if t == "EV-B" else f"Title {t}") for t in tickers}


class TestDayStoreIter:
    """_day_store_iter is the one reader of the slice format (day slices and
    the assembled cache); _day_store_load is its all-or-nothing list form and
    keeps its external contract."""

    META = _SS1C_META

    @staticmethod
    def _records(n):
        return [{"ticker": f"T{i}", "result": "yes", "n": i} for i in range(n)]

    def test_load_is_the_iterator_gathered_on_both_formats(self, tmp_path):
        records = self._records(6)
        jsonl = _write_jsonl(tmp_path / "a.json.gz", self.META, records)
        legacy = tmp_path / "b.json.gz"
        historical._day_store_save(legacy, self.META, records)

        def even(m):
            return m["n"] % 2 == 0

        for path in (jsonl, legacy):
            for keep in (None, even):
                expected = [m for m in records if keep is None or keep(m)]
                assert historical._day_store_load(path, self.META, keep) == expected
                assert list(historical._day_store_iter(path, self.META, keep)) == expected

    def test_every_walk_yields_fresh_dicts(self, tmp_path):
        path = _write_jsonl(tmp_path / "a.json.gz", self.META, self._records(3))
        first = list(historical._day_store_iter(path, self.META))
        second = list(historical._day_store_iter(path, self.META))
        assert first == second
        assert all(a is not b for a, b in zip(first, second, strict=True))

    def test_the_meta_is_checked_before_any_record_is_yielded(self, tmp_path):
        path = _write_jsonl(tmp_path / "a.json.gz", self.META, self._records(3))
        walk = historical._day_store_iter(path, {**self.META, "cutoff_ts": 200})
        with pytest.raises(historical._SliceUnreadable, match="cutoff_ts"):
            next(walk)
        with pytest.raises(historical._SliceUnreadable, match="does not exist"):
            next(historical._day_store_iter(tmp_path / "absent.json.gz", self.META))

    def test_damage_raises_at_the_damage_and_the_list_form_is_all_or_nothing(
            self, tmp_path):
        # The iterator cannot take back what it already yielded, so it raises
        # AT the damage; the list form must still return nothing at all.
        path = _write_jsonl(tmp_path / "a.json.gz", self.META, self._records(6))
        with gzip.open(path, "rb") as fh:
            raw = fh.read()
        with gzip.open(path, "wb") as fh:
            fh.write(raw[: raw.index(b'"T4"') + 2])  # cut inside T4's line
        seen = []
        with pytest.raises(historical._SliceUnreadable, match="malformed"):
            for m in historical._day_store_iter(path, self.META):
                seen.append(m["ticker"])
        assert seen == ["T0", "T1", "T2", "T3"]
        assert historical._day_store_load(path, self.META) is None

        # And a stream cut at the gzip level (an interrupted write).
        cut = tmp_path / "cut.json.gz"
        whole = _write_jsonl(tmp_path / "whole.json.gz", self.META, self._records(40))
        cut.write_bytes(whole.read_bytes()[:-12])
        with pytest.raises(historical._SliceUnreadable):
            list(historical._day_store_iter(cut, self.META))
        assert historical._day_store_load(cut, self.META) is None

    def test_a_damaged_deflate_stream_reads_as_unreadable_not_as_a_crash(self, tmp_path):
        # zlib.error is NOT an OSError, and the reader before SS-1 caught only
        # (OSError, EOFError, ValueError): a damaged deflate block escaped the
        # reuse prescan as a crash instead of reading as "refetch this day",
        # contradicting its own all-or-nothing contract.
        payload = (json.dumps({"meta": self.META}) + "\n"
                   + "".join(json.dumps(m) + "\n" for m in self._records(20))).encode()
        blob = bytearray(gzip.compress(payload))
        blob[10] = 0x07  # first deflate block header -> the reserved block type
        path = tmp_path / "a.json.gz"
        path.write_bytes(bytes(blob))
        with pytest.raises(zlib.error):  # the failure being absorbed
            with gzip.open(path, "rb") as fh:
                fh.readline()
        assert historical._day_store_load(path, self.META) is None
        with pytest.raises(historical._SliceUnreadable):
            list(historical._day_store_iter(path, self.META))

    @pytest.mark.parametrize("fmt", ["jsonl", "legacy"])
    def test_an_exception_from_keep_propagates_on_both_formats(self, tmp_path, fmt):
        # A predicate failure is not a damaged file. Before SS-1 a ValueError
        # from `keep` on the JSONL path read as "slice unreadable" (so the day
        # was silently refetched) while on the legacy path it propagated; the
        # reader now keeps the two apart on both formats.
        path = tmp_path / "a.json.gz"
        if fmt == "jsonl":
            _write_jsonl(path, self.META, self._records(4))
        else:
            historical._day_store_save(path, self.META, self._records(4))

        def keep(m):
            if m["n"] == 2:
                raise ValueError("predicate bug")
            return True

        with pytest.raises(ValueError, match="predicate bug"):
            historical._day_store_load(path, self.META, keep)

    @pytest.mark.parametrize("content", [
        '{"meta": META}\n5\n',                            # a record that is not an object
        '{"meta": [1, 2]}\n{"ticker": "T"}\n',            # a meta block that is not an object
        '{"meta": META, "markets": {"a": 1}}',            # legacy "markets" not a list
        '{"meta": META, "markets": [5]}',                 # legacy record not an object
    ])
    def test_a_shape_the_writers_never_produce_is_unreadable(self, tmp_path, content):
        path = tmp_path / "a.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(content.replace("META", json.dumps(self.META)))
        assert historical._day_store_load(path, self.META) is None


class TestDaySliceStream:
    """A phase's day slices are handed back as a lazy _DaySliceStream: nothing
    is read until the assembly walks it, every walk re-reads the files newest
    day first, and a slice that goes bad mid-walk is a loud, named error —
    never a sequential-walk fallback and never a silently short corpus."""

    META = _SS1C_META
    DAYS = ("2026-06-07", "2026-06-08", "2026-06-09")

    def _publish(self, tmp_path, monkeypatch):
        monkeypatch.setattr(historical, "CACHE_DIR", tmp_path)
        by_day = {}
        for d in self.DAYS:
            recs = [{"ticker": f"{d}-{i}", "n": i} for i in range(4)]
            _write_jsonl(historical._day_store_path("archive_days", _day_lo(d)),
                         self.META, recs)
            by_day[_day_lo(d)] = recs
        return by_day

    def test_construction_reads_nothing_and_every_walk_rereads_the_files(
            self, tmp_path, monkeypatch):
        by_day = self._publish(tmp_path, monkeypatch)
        opened: list[str] = []
        real = historical._day_store_iter

        def spy(path, expect_meta, keep=None):
            opened.append(path.name)
            return real(path, expect_meta, keep)

        monkeypatch.setattr(historical, "_day_store_iter", spy)
        stream = historical._assemble_day_slices("archive_days", list(by_day), self.META)
        assert isinstance(stream, historical._DaySliceStream)
        assert opened == []  # lazy: nothing is read until it is walked
        first, second = list(stream), list(stream)
        assert first == second
        assert all(a is not b for a, b in zip(first, second, strict=True))
        newest_first = [f"{d}.json.gz" for d in reversed(self.DAYS)]
        assert opened == newest_first * 2

    def test_it_equals_the_old_eager_assembly_in_order_and_membership(
            self, tmp_path, monkeypatch):
        by_day = self._publish(tmp_path, monkeypatch)

        def keep(m):
            return m["n"] != 1

        # The old _assemble_day_slices, literally: load each day newest-first
        # and extend one list.
        expected: list[dict] = []
        for lo in sorted(by_day, reverse=True):
            expected.extend(historical._day_store_load(
                historical._day_store_path("archive_days", lo), self.META, keep))
        # Day identities arrive in pool-COMPLETION order, not sorted.
        los = [_day_lo("2026-06-08"), _day_lo("2026-06-07"), _day_lo("2026-06-09")]
        stream = historical._assemble_day_slices("archive_days", los, self.META, keep)
        assert list(stream) == expected
        assert [m["ticker"][:10] for m in expected[:3]] == ["2026-06-09"] * 3
        assert all(m["n"] != 1 for m in expected)

    @pytest.mark.parametrize("damage", ["deleted", "truncated", "rewritten"])
    def test_a_slice_that_goes_bad_mid_walk_raises_naming_the_day(
            self, tmp_path, monkeypatch, damage):
        by_day = self._publish(tmp_path, monkeypatch)
        stream = historical._assemble_day_slices("archive_days", list(by_day), self.META)
        middle = historical._day_store_path("archive_days", _day_lo("2026-06-08"))
        if damage == "deleted":
            middle.unlink()
        elif damage == "truncated":
            middle.write_bytes(middle.read_bytes()[:-12])
        else:  # rewritten under different fetch conditions (an advanced cutoff)
            _write_jsonl(middle, {**self.META, "cutoff_ts": 999}, [{"ticker": "X"}])
        got: list[str] = []
        with pytest.raises(historical.SettledCorpusError) as info:
            for m in stream:
                got.append(m["ticker"])
        # The newest day was delivered, then the walk stopped LOUDLY at the
        # damaged day — it never skipped ahead to deliver the oldest day.
        assert got[:4] == [f"2026-06-09-{i}" for i in range(4)]
        assert not any(t.startswith("2026-06-07") for t in got)
        assert "2026-06-08" in str(info.value)
        assert "Re-run" in str(info.value)
        assert isinstance(info.value, RuntimeError)


class TestPhasesReturnLazyViews:
    """The phases hand back re-iterable views, never a materialized list of
    their day slices; the live phase chains its frontier spool in front of
    its past-day stream in the old `frontier + [...]` order."""

    def test_the_archive_phase_returns_an_unread_slice_stream(self, tmp_path, monkeypatch):
        archive_markets, _ = TestShardedFetch()._fixture_markets()
        _install_sharded_fakes(monkeypatch, tmp_path, _FakeArchive(archive_markets),
                               TestShardedFetch.CUTOFF)
        reads = _completed_reads(monkeypatch)
        day, tail = historical._fetch_archive_phase(
            MagicMock(), _day_lo("2026-06-05"),
            TestShardedFetch._ts(TestShardedFetch.CUTOFF), {"limit": 1000}, None)
        assert isinstance(day, historical._DaySliceStream)
        assert isinstance(tail, list)
        # Cold run: the prescan found nothing and nothing was read back.
        assert reads == []
        records = list(day)
        slices = sorted((tmp_path / "cache" / "archive_days").glob("*.json.gz"))
        assert records and sorted(reads) == sorted(str(p) for p in slices)

    def test_the_live_phase_chains_its_frontier_before_a_lazy_stream(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(historical, "CACHE_DIR", tmp_path)
        frontier = [_mk_raw_market(f"F{i}", "2026-09-23T00:00:00Z",
                                   f"2026-09-24T0{i + 1}:00:00Z") for i in range(3)]
        past = [_mk_raw_market(f"P{d}{i}", "2026-09-21T00:00:00Z",
                               f"2026-09-2{d}T0{i + 1}:00:00Z")
                for d in (2, 3) for i in range(3)]
        reads = _completed_reads(monkeypatch)
        out = historical._fetch_live_phase(
            _FakeLive(frontier + past, page_size=2), _day_lo("2026-09-22"),
            _day_lo("2026-09-24") + 12 * 3600, None)
        assert isinstance(out, historical._RecordChain)
        # The frontier is a sealed spool, not a list (SS-1 review).
        assert isinstance(out._parts[0], historical._FrontierSpool)
        assert reads == []  # the past days are on disk, unread
        tickers = [m["ticker"] for m in out]
        assert tickers == ["F2", "F1", "F0", "P32", "P31", "P30", "P22", "P21", "P20"]
        assert [m["ticker"] for m in out] == tickers  # re-iterable, same order
        assert len(reads) == 4  # two past days, two walks
        out.close()


class TestFrontierSpool:
    """SS-1 review: the live frontier used to be returned as a list of its
    keep-passing records, and on the day after a Monday that is most of the
    day (7,190,452 of 9,176,306 records on Tuesday 2026-09-22). It is spooled
    to an anonymous temporary file instead and read back per walk, like a day
    slice: exactly the old records in the old order, none of them resident."""

    NOW = TestFrontierStreamsThroughKeep.NOW
    TODAY = TestFrontierStreamsThroughKeep.TODAY
    _ts = staticmethod(TestFrontierStreamsThroughKeep._ts)
    _keep = staticmethod(TestFrontierStreamsThroughKeep._keep)
    _frontier_markets = staticmethod(TestFrontierStreamsThroughKeep._frontier_markets)

    @staticmethod
    def _sealed(directory, records):
        spool = historical._FrontierSpool(directory)
        spool.extend(records)
        spool.seal()
        return spool

    def test_no_kept_frontier_record_is_resident(self, tmp_path, monkeypatch):
        # The KEPT half of test_rejected_frontier_record_is_never_retained:
        # a list sink kept every one of these alive until the phase's result
        # was dropped, so this count would climb page by page and end at the
        # kept total.
        monkeypatch.setattr(historical, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(historical, "SETTLED_FETCH_CHUNK_RECORDS", 1)

        class _Tracked(dict):
            """A compact record that supports weak references."""

        kept_refs: list = []
        real_to_dict = historical._market_to_dict

        def tracking_to_dict(m, *args, **kwargs):
            rec = _Tracked(real_to_dict(m, *args, **kwargs))
            if self._keep(rec):
                kept_refs.append(weakref.ref(rec))
            return rec

        monkeypatch.setattr(historical, "_market_to_dict", tracking_to_dict)
        alive_at_request: list[int] = []

        class _ObservedLive(_FakeLive):
            def get_markets_without_preload_content(self, *args, **kwargs):
                alive_at_request.append(sum(ref() is not None for ref in kept_refs))
                return super().get_markets_without_preload_content(*args, **kwargs)

        markets = self._frontier_markets()
        out = historical._fetch_live_phase(
            _ObservedLive(markets, page_size=2), self._ts(self.TODAY) + 60,
            self._ts(self.NOW), self._keep,
        )
        assert kept_refs and len(alive_at_request) >= 3
        assert alive_at_request == [0] * len(alive_at_request)
        assert all(ref() is None for ref in kept_refs)
        # ...and still exactly the old post-hoc filter, on every walk.
        expected = TestFrontierStreamsThroughKeep()._oracle(markets, self._keep)
        first, second = list(out), list(out)
        assert first == second == expected
        assert all(a is not b for a, b in zip(first, second, strict=True))
        assert len(out._parts[0]) == len(expected)
        out.close()

    def test_the_spool_is_anonymous(self, tmp_path, monkeypatch):
        # No name, so no later run can find it, reuse it as a complete day,
        # or be left a stale one by a crash.
        monkeypatch.setattr(historical, "CACHE_DIR", tmp_path)
        out = historical._fetch_live_phase(
            _FakeLive(self._frontier_markets(), page_size=2),
            self._ts(self.TODAY) + 60, self._ts(self.NOW), None,
        )
        assert len(list(out)) == 11
        assert [p for p in tmp_path.rglob("*") if p.is_file()] == []
        out.close()

    def test_a_fetch_closes_the_spool_on_success_and_on_failure(self, tmp_path,
                                                                monkeypatch):
        spools: list = []
        real_init = historical._FrontierSpool.__init__

        def recording_init(self, directory):
            real_init(self, directory)
            spools.append(self)

        monkeypatch.setattr(historical._FrontierSpool, "__init__", recording_init)
        archive_markets, live_markets = TestShardedFetch()._fixture_markets()
        _install_sharded_fakes(monkeypatch, tmp_path, _FakeArchive(archive_markets),
                               TestShardedFetch.CUTOFF)
        out = historical.fetch_all_settled_markets(
            MagicMock(), _FakeLive(live_markets), start_date=date(2026, 6, 5),
            use_cache=False)
        assert len(out) > 0 and len(spools) == 1 and spools[0]._closed
        with pytest.raises(historical.SettledCorpusError):
            list(spools[0])

        def titles(live_client, tickers, use_cache=True):
            historical._day_store_path("live_days", _day_lo("2026-06-10")).unlink()
            return {}

        monkeypatch.setattr(historical, "_load_or_build_event_titles", titles)
        with pytest.raises(historical.SettledCorpusError, match="2026-06-10"):
            historical.fetch_all_settled_markets(
                MagicMock(), _FakeLive(live_markets), start_date=date(2026, 6, 5),
                use_cache=False)
        assert len(spools) == 2 and spools[1]._closed

    def test_a_failing_phase_closes_the_spool(self, tmp_path, monkeypatch):
        spools: list = []
        real_init = historical._FrontierSpool.__init__

        def recording_init(self, directory):
            real_init(self, directory)
            spools.append(self)

        monkeypatch.setattr(historical._FrontierSpool, "__init__", recording_init)
        monkeypatch.setattr(historical, "CACHE_DIR", tmp_path)

        def keep(m):
            raise ValueError("keep failed")

        with pytest.raises(ValueError, match="keep failed"):
            historical._fetch_live_phase(
                _FakeLive(self._frontier_markets(), page_size=2),
                self._ts(self.TODAY) + 60, self._ts(self.NOW), keep,
            )
        assert len(spools) == 1 and spools[0]._closed

    def test_lifecycle_misuse_is_loud(self, tmp_path):
        spool = historical._FrontierSpool(tmp_path)
        spool.extend([{"ticker": "A"}])
        with pytest.raises(historical.SettledCorpusError, match="before it was sealed"):
            list(spool)
        spool.seal()
        with pytest.raises(RuntimeError, match="after it was sealed"):
            spool.extend([{"ticker": "B"}])
        assert list(spool) == [{"ticker": "A"}]
        spool.close()
        spool.close()  # idempotent
        with pytest.raises(historical.SettledCorpusError, match="after it was closed"):
            list(spool)

    def test_a_suspended_walk_cannot_resume_after_a_newer_one(self, tmp_path):
        # Both walks share the one file position, so the older must stop
        # rather than read from wherever the newer one left it.
        spool = self._sealed(tmp_path, [{"ticker": t} for t in ("A", "B", "C")])
        older = iter(spool)
        assert next(older) == {"ticker": "A"}
        assert list(spool) == [{"ticker": "A"}, {"ticker": "B"}, {"ticker": "C"}]
        with pytest.raises(historical.SettledCorpusError, match="walks must not interleave"):
            next(older)
        spool.close()

    def test_a_short_spool_is_never_returned_quietly(self, tmp_path):
        spool = self._sealed(tmp_path, [{"ticker": "A"}, {"ticker": "B"}])
        spool._count += 1  # what a lost write would look like
        with pytest.raises(historical.SettledCorpusError, match="yielded 2 records but 3"):
            list(spool)
        spool.close()

    def test_an_empty_spool_walks_empty(self, tmp_path):
        spool = self._sealed(tmp_path, [])
        assert list(spool) == [] and len(spool) == 0
        spool.close()


class TestVanishedSliceDuringAssembly:
    """A slice that disappears after its phase verified or wrote it raises
    SettledCorpusError out of fetch_all_settled_markets: the old eager
    assembly turned this into the sequential fallback, which now would hold
    the whole range in memory and could not take back records already
    yielded. Nothing may be published."""

    @staticmethod
    def _forbid_fallbacks(monkeypatch):
        calls: list[str] = []
        monkeypatch.setattr(historical, "_fetch_archive_sequential",
                            lambda *a, **k: calls.append("archive") or [])
        monkeypatch.setattr(historical, "_fetch_live_sequential",
                            lambda *a, **k: calls.append("live") or [])
        return calls

    @staticmethod
    def _assert_nothing_published(tmp_path):
        cache = tmp_path / "cache"
        assert not list(cache.glob("settled_markets_*"))
        assert not list(cache.rglob("*.tmp"))

    def test_a_live_slice_vanishing_between_the_walks(self, tmp_path, monkeypatch):
        archive_markets, live_markets = TestShardedFetch()._fixture_markets()
        _install_sharded_fakes(monkeypatch, tmp_path, _FakeArchive(archive_markets),
                               TestShardedFetch.CUTOFF)
        fallbacks = self._forbid_fallbacks(monkeypatch)

        def titles(live_client, tickers, use_cache=True):
            # Runs between walk A and walk B.
            historical._day_store_path("live_days", _day_lo("2026-06-10")).unlink()
            return {}

        monkeypatch.setattr(historical, "_load_or_build_event_titles", titles)
        with pytest.raises(historical.SettledCorpusError, match="2026-06-10"):
            historical.fetch_all_settled_markets(
                MagicMock(), _FakeLive(live_markets), start_date=date(2026, 6, 5),
                use_cache=False)
        assert fallbacks == []
        self._assert_nothing_published(tmp_path)

    def test_an_archive_slice_vanishing_before_the_first_walk(self, tmp_path, monkeypatch):
        archive_markets, live_markets = TestShardedFetch()._fixture_markets()
        _install_sharded_fakes(monkeypatch, tmp_path, _FakeArchive(archive_markets),
                               TestShardedFetch.CUTOFF)
        fallbacks = self._forbid_fallbacks(monkeypatch)
        real_phase = historical._fetch_archive_phase

        def phase_then_damage(*a, **k):
            result = real_phase(*a, **k)
            historical._day_store_path("archive_days", _day_lo("2026-06-08")).unlink()
            return result

        monkeypatch.setattr(historical, "_fetch_archive_phase", phase_then_damage)
        live = _FakeLive(live_markets)
        with pytest.raises(historical.SettledCorpusError, match="2026-06-08"):
            historical.fetch_all_settled_markets(
                MagicMock(), live, start_date=date(2026, 6, 5), use_cache=False)
        assert fallbacks == []
        # Raised by walk A's archive half, before the live phase even ran.
        assert live.calls == 0
        self._assert_nothing_published(tmp_path)


class _DriftingRecords:
    """A record source yielding `first` on its first walk and `second` after."""

    def __init__(self, first, second):
        self._first, self._second = first, second
        self.walks = 0

    def __iter__(self):
        walk = self._first if self.walks == 0 else self._second
        self.walks += 1
        return iter(copy.deepcopy(walk))


class TestStreamedAssemblyParity:
    """The streamed assembly must reproduce the old in-memory merge EXACTLY —
    the same records, in the same order, with the same event_title patch and
    the same logged numbers — on every source shape and on the real phases."""

    START = date(2026, 6, 5)
    CUTOFF = "2026-06-10T00:00:00Z"

    @staticmethod
    def _rec(ticker, settle, event_ticker="EV", **extra):
        m = {"ticker": ticker, "event_ticker": event_ticker, "event_title": "",
             "title": f"Q {ticker}", "subtitle": None, "result": "yes",
             "open_time": "2026-06-01T00:00:00Z", "close_time": settle,
             "settlement_ts": settle}
        m.update(extra)
        return m

    def _sources(self):
        r = self._rec
        newer_day = [
            r("A1", "2026-06-09T10:00:00Z", "EV-A"),
            r("DUP", "2026-06-09T09:00:00Z", "EV-D", tag="archive-newer"),
            # Rejected by the prefilter: with it, the TAIL's SHADOW must win
            # (prefilter before dedup); without it, this one wins.
            r("SHADOW", "2026-06-09T08:00:00Z", "EV-S", keep=False, tag="rejected"),
            r("PRE", "2026-06-04T23:59:59Z", "EV-P"),       # settled before the window
            r("ATCUT", "2026-06-10T00:00:00Z", "EV-C", tag="archive"),  # at the ceiling
            r("NOSETTLE", None, "EV-N"),
            r("", "2026-06-09T07:00:00Z", "EV-E"),          # falsy tickers never kept
            r(None, "2026-06-09T07:00:00Z", "EV-E"),
            r("NOEV", "2026-06-09T06:00:00Z", "", event_title="kept-own"),
        ]
        older_day = [
            r("DUP", "2026-06-08T09:00:00Z", "EV-D2", tag="archive-older"),
            r("B1", "2026-06-08T08:00:00Z", "EV-B", event_title="stale"),
            r("B2", "2026-06-05T00:00:00Z", "EV-B"),         # exactly at start: kept
        ]
        tail = [
            r("SHADOW", "2026-06-06T10:00:00Z", "EV-S", tag="tail"),
            r("DUP", "2026-06-07T10:00:00Z", "EV-D", tag="tail"),
            r("T1", "2026-06-06T10:00:00Z", "EV-T"),
            # The tail is never keep-filtered by its phase, so this rejected
            # copy reaches the merge on EVERY source shape: with the prefilter
            # the LIVE copy must win (prefilter before dedup), without it this.
            r("TWIN", "2026-06-07T09:00:00Z", "EV-W", keep=False, tag="tail-rejected"),
        ]
        frontier = [
            r("F1", "2026-09-24T01:00:00Z", "EV-F"),
            r("A1", "2026-09-24T00:30:00Z", "EV-A", tag="live-dup"),
        ]
        live_newer = [
            r("L1", "2026-06-11T10:00:00Z", "EV-L"),
            r("ATCUT", "2026-06-10T00:00:00Z", "EV-C", tag="live"),  # live: kept
        ]
        live_older = [
            r("L1", "2026-06-10T10:00:00Z", "EV-L", tag="older-dup"),
            r("L2", "2026-06-10T09:00:00Z", "EV-L2"),
            r("REJ", "2026-06-10T08:00:00Z", "EV-R", keep=False),
            r("TWIN", "2026-06-10T07:00:00Z", "EV-W", tag="live"),
        ]
        return newer_day, older_day, tail, frontier, live_newer, live_older

    @staticmethod
    def _keep(m):
        return m.get("keep", True)

    def _run(self, tmp_path, monkeypatch, caplog, *, mode, prefilter, titles_for):
        newer_day, older_day, tail, frontier, live_newer, live_older = self._sources()
        monkeypatch.setattr(historical, "CACHE_DIR", tmp_path / "cache")
        if mode == "lists":
            day_src = newer_day + older_day
            live_src = frontier + live_newer + live_older
        else:
            # The real lazy views over real files, as the phases build them.
            meta_a = {"kind": "archive_created_day", "cutoff_ts": 1, "include_mve": True,
                      "complete": True}
            meta_l = {"kind": "live_settled_day", "include_mve": True, "complete": True}
            # Live days must lie after the cutoff, or the real
            # _prune_stale_live_days (which the assembly runs) deletes them.
            archive_los = (_day_lo("2026-06-09"), _day_lo("2026-06-08"))
            live_los = (_day_lo("2026-06-11"), _day_lo("2026-06-10"))
            for store, meta, los, days in (
                    ("archive_days", meta_a, archive_los, (newer_day, older_day)),
                    ("live_days", meta_l, live_los, (live_newer, live_older))):
                for lo, recs in zip(los, days, strict=True):
                    _write_jsonl(historical._day_store_path(store, lo), meta, recs)
            # Handed over oldest first, as pool completion order may; the
            # stream sorts newest first itself. UNFILTERED, as the phases hand
            # their day slices over since M9; only the frontier is filtered
            # where it is fetched (this fixture's frontier has nothing the
            # prefilter rejects, so no fetch-time count is lost by stubbing
            # the phase here).
            day_src = historical._assemble_day_slices(
                "archive_days", sorted(archive_los), meta_a)
            live_src = historical._RecordChain(
                [m for m in frontier if prefilter is None or prefilter(m)],
                historical._assemble_day_slices("live_days", sorted(live_los), meta_l))
        monkeypatch.setattr(historical, "_fetch_archive_phase",
                            lambda *a, **k: (day_src, tail))
        monkeypatch.setattr(historical, "_fetch_live_phase", lambda *a, **k: live_src)
        monkeypatch.setattr(historical, "_historical_get",
                            lambda *a, **k: {"market_settled_ts": self.CUTOFF})
        asked: list[set] = []

        def titles(live_client, tickers, use_cache=True):
            asked.append(set(tickers))
            return titles_for(tickers)

        monkeypatch.setattr(historical, "_load_or_build_event_titles", titles)
        kwargs = ({"prefilter": prefilter, "prefilter_tag": "t"}
                  if prefilter is not None else {})
        with caplog.at_level(logging.INFO):
            out = historical.fetch_all_settled_markets(
                MagicMock(), MagicMock(), start_date=self.START, use_cache=False,
                **kwargs)
        return out, asked

    @pytest.mark.parametrize("mode", ["lists", "streams"])
    @pytest.mark.parametrize("use_prefilter", [False, True])
    @pytest.mark.parametrize("titles_kind", ["empty", "mapped"])
    def test_the_streamed_assembly_equals_the_old_merge(
            self, tmp_path, monkeypatch, caplog, mode, use_prefilter, titles_kind):
        prefilter = self._keep if use_prefilter else None
        titles_for = _mapped_titles if titles_kind == "mapped" else (lambda _t: {})
        out, asked = self._run(tmp_path, monkeypatch, caplog, mode=mode,
                               prefilter=prefilter, titles_for=titles_for)

        newer_day, older_day, tail, frontier, live_newer, live_older = self._sources()
        start_ts = _day_lo("2026-06-05")
        cutoff_ts = TestShardedFetch._ts(self.CUTOFF)
        expected, archive_count, live_count, unique = _old_assembly(
            newer_day + older_day, tail, frontier + live_newer + live_older,
            start_ts, cutoff_ts, prefilter, titles_for)

        assert isinstance(out, historical.SettledCorpus)
        assert list(out) == expected           # records, order and event_title
        assert len(out) == len(expected)
        assert asked == [unique]               # titles resolved for the same set

        # M9: every count line reports the settled records beside the kept
        # ones, split into what the prefilter and the dedup removed.
        archive_counts, live_counts = _old_assembly_counts(
            [[(newer_day + older_day, cutoff_ts), (tail, cutoff_ts)],
             [(frontier + live_newer + live_older, None)]], start_ts, prefilter)
        total = historical._total_counts(archive_counts, live_counts)
        # Not vacuous: the fixture's numbers, worked by hand. The settled
        # counts do not depend on the prefilter; the kept ones match the
        # old merge's.
        if use_prefilter:
            assert (archive_counts, live_counts) == (
                historical.AssemblyCounts(13, 2, 4), historical.AssemblyCounts(8, 1, 2))
        else:
            assert (archive_counts, live_counts) == (
                historical.AssemblyCounts(13, 0, 5), historical.AssemblyCounts(8, 0, 3))
        assert (archive_counts.kept, live_counts.kept, total.kept) == (
            archive_count, live_count, len(expected))

        def said(counts):
            dup = f"{counts.duplicates} duplicate or blank tickers"
            return (f"{counts.rejected} rejected by the prefilter t, {dup}"
                    if use_prefilter else dup)

        noun = "eligible markets" if use_prefilter else "settled markets"
        messages = [r.getMessage() for r in caplog.records]
        lines = [f"Historical endpoint: {archive_count} {noun} of "
                 f"{archive_counts.settled} records settled in the window before "
                 f"the archive cutoff ({said(archive_counts)})",
                 "Fetching recently settled markets (after API cutoff)...",
                 f"Live endpoint: {live_count} {noun} of {live_counts.settled} "
                 f"recently settled records in the window ({said(live_counts)})",
                 f"Resolving event titles for {len(unique)} unique event_tickers",
                 f"Assembled {len(expected)} {noun} of {total.settled} records "
                 f"settled since 2026-06-05 ({said(total)})"]
        positions = [messages.index(line) for line in lines]
        assert positions == sorted(positions)
        # ...and the total rides the corpus out, and into the cache's meta.
        assert out.provenance.assembly_counts == total
        with gzip.open(out.path, "rt", encoding="utf-8") as fh:
            assert json.loads(fh.readline())["meta"]["assembly_counts"] == {
                "settled": total.settled, "rejected": total.rejected,
                "duplicates": total.duplicates}

        # Not vacuous: every rule in the fixture actually decided something.
        by_ticker = {m["ticker"]: m for m in expected}
        assert by_ticker["DUP"]["tag"] == "archive-newer"
        assert by_ticker["ATCUT"]["tag"] == "live"
        assert by_ticker["A1"].get("tag") is None
        assert by_ticker["SHADOW"]["tag"] == ("tail" if use_prefilter else "rejected")
        assert by_ticker["TWIN"]["tag"] == ("live" if use_prefilter else "tail-rejected")
        assert "L1" in by_ticker and by_ticker["L1"].get("tag") is None
        assert not {"PRE", "NOSETTLE", "", None} & set(by_ticker)
        assert ("REJ" in by_ticker) is (not use_prefilter)
        if titles_kind == "empty":
            assert by_ticker["NOEV"]["event_title"] == "kept-own"
            assert by_ticker["B1"]["event_title"] == "stale"
        else:
            assert by_ticker["NOEV"]["event_title"] == ""
            assert by_ticker["B1"]["event_title"] == ""
            assert by_ticker["L2"]["event_title"] == "Title EV-L2"

    @pytest.mark.parametrize("opaque, ignore_max", [
        (False, False), (True, False), (False, True),
    ])
    def test_the_real_phases_assemble_exactly_as_the_old_merge(
            self, tmp_path, monkeypatch, opaque, ignore_max):
        archive_markets, live_markets = TestShardedFetch()._fixture_markets()
        _install_sharded_fakes(
            monkeypatch, tmp_path, _FakeArchive(archive_markets, opaque_cursors=opaque),
            TestShardedFetch.CUTOFF)
        monkeypatch.setattr(historical, "_load_or_build_event_titles",
                            lambda live_client, tickers, use_cache=True: _mapped_titles(tickers))
        captured: dict = {}
        real_archive, real_live = historical._fetch_archive_phase, historical._fetch_live_phase

        def spy_archive(*a, **k):
            captured["archive"] = real_archive(*a, **k)
            return captured["archive"]

        def spy_live(*a, **k):
            result = real_live(*a, **k)
            # Copied NOW, by one extra complete walk: the fetch closes the live
            # phase's frontier spool once its own two walks are done, so the
            # result cannot be walked again after it returns.
            captured["live"] = list(result)
            return result

        monkeypatch.setattr(historical, "_fetch_archive_phase", spy_archive)
        monkeypatch.setattr(historical, "_fetch_live_phase", spy_live)

        def pred(m):
            return not m["ticker"].endswith("2")

        out = historical.fetch_all_settled_markets(
            MagicMock(), _FakeLive(live_markets, ignore_max=ignore_max),
            start_date=self.START, use_cache=False, prefilter=pred, prefilter_tag="t")
        day, tail = captured["archive"]
        expected, _, _, _ = _old_assembly(
            day, tail, captured["live"], _day_lo("2026-06-05"),
            TestShardedFetch._ts(TestShardedFetch.CUTOFF), pred, _mapped_titles)
        assert list(out) == expected
        assert {"A1", "LONGLIVED", "L1", "L3"} <= {m["ticker"] for m in expected}
        assert all(m["event_title"] == f"Title {m['event_ticker']}" for m in expected)

    @pytest.mark.parametrize("drift", ["fewer", "reordered", "event_ticker"])
    def test_a_second_walk_that_disagrees_publishes_nothing(
            self, tmp_path, monkeypatch, drift):
        r = self._rec
        first = [r("L1", "2026-06-11T10:00:00Z", "EV-1"),
                 r("L2", "2026-06-11T09:00:00Z", "EV-2"),
                 r("L3", "2026-06-11T08:00:00Z", "EV-3")]
        if drift == "fewer":
            second = first[:2]
        elif drift == "reordered":
            second = [first[1], first[0], first[2]]
        else:  # same tickers, same order; one record's event_ticker changed
            second = [first[0], r("L2", "2026-06-11T09:00:00Z", "EV-OTHER"), first[2]]
        monkeypatch.setattr(historical, "CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr(historical, "_fetch_archive_phase", lambda *a, **k: ([], []))
        source = _DriftingRecords(first, second)
        monkeypatch.setattr(historical, "_fetch_live_phase", lambda *a, **k: source)
        monkeypatch.setattr(historical, "_historical_get",
                            lambda *a, **k: {"market_settled_ts": self.CUTOFF})
        monkeypatch.setattr(historical, "_load_or_build_event_titles",
                            lambda *a, **k: {})
        with pytest.raises(historical.SettledCorpusError, match="did not reproduce"):
            historical.fetch_all_settled_markets(
                MagicMock(), MagicMock(), start_date=self.START, use_cache=False)
        assert source.walks == 2
        TestVanishedSliceDuringAssembly._assert_nothing_published(tmp_path)


class TestStreamedAssembledCache:
    """The assembled cache is settled_markets_<...>.jsonl.gz, written through
    the atomic streaming writer and served back as a SettledCorpus after a
    full validation walk (all or nothing). A legacy settled_markets_<...>.json
    is still served, whole, as a list — but only when no valid streamed cache
    exists, and nothing writes that format any more."""

    START = date(2026, 6, 5)

    def _fetch(self, tmp_path, monkeypatch, use_cache=False):
        archive_markets, live_markets = TestShardedFetch()._fixture_markets()
        archive = _FakeArchive(archive_markets)
        _install_sharded_fakes(monkeypatch, tmp_path, archive, TestShardedFetch.CUTOFF)
        out = historical.fetch_all_settled_markets(
            MagicMock(), _FakeLive(live_markets), start_date=self.START,
            use_cache=use_cache)
        return out, archive, live_markets

    def _again(self, live_markets, use_cache=True):
        live = _FakeLive(live_markets)
        out = historical.fetch_all_settled_markets(
            MagicMock(), live, start_date=self.START, use_cache=use_cache)
        return out, live

    def test_a_fetch_streams_into_the_cache_and_a_hit_streams_it_back(
            self, tmp_path, monkeypatch, caplog):
        out, archive, live_markets = self._fetch(tmp_path, monkeypatch)
        path = tmp_path / "cache" / "settled_markets_2026-06-05.jsonl.gz"
        assert isinstance(out, historical.SettledCorpus)
        assert out.path == path
        first, second = list(out), list(out)
        assert first == second and len(out) == len(first) > 0
        assert all(a is not b for a, b in zip(first, second, strict=True))
        # The file IS the corpus, in jsonl-v1 framing, and its meta repeats
        # the identity the filename encodes.
        assert _read_slice_file(path) == first
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            meta = json.loads(fh.readline())["meta"]
        expected_meta = historical._assembled_cache_meta(self.START, None)
        assert {k: meta[k] for k in expected_meta} == expected_meta
        assert not list((tmp_path / "cache").glob("settled_markets_*.json"))

        archive.calls = 0
        with caplog.at_level(logging.INFO):
            hit, live = self._again(live_markets)
        assert isinstance(hit, historical.SettledCorpus)
        assert list(hit) == first and len(hit) == len(first)
        assert archive.calls == 0 and live.calls == 0
        assert f"Loaded {len(first)} settled markets from cache" in caplog.text

    @pytest.mark.parametrize("damage", ["truncated", "mid_record", "other_kind"])
    def test_a_damaged_cache_is_a_miss_with_a_warning_and_is_rebuilt(
            self, tmp_path, monkeypatch, caplog, damage):
        out, archive, live_markets = self._fetch(tmp_path, monkeypatch)
        good, path = list(out), out.path
        if damage == "truncated":
            data = path.read_bytes()
            path.write_bytes(data[: len(data) // 2])
        elif damage == "mid_record":
            with gzip.open(path, "rb") as fh:
                raw = fh.read()
            with gzip.open(path, "wb") as fh:
                fh.write(raw[: raw.rindex(b"\n", 0, len(raw) - 1) + 10])
        else:  # a valid jsonl-v1 file of another kind copied onto the name
            _write_jsonl(path, {"kind": "live_settled_day", "include_mve": True,
                                "complete": True}, good)
        archive.calls = 0
        with caplog.at_level(logging.WARNING):
            rebuilt, _ = self._again(live_markets)
        assert "Corrupt or mismatched settled-market cache" in caplog.text
        assert archive.calls > 0  # rebuilt from the API + day slices, not served
        assert list(rebuilt) == good
        assert historical.SettledCorpus.open_validated(
            path, historical._assembled_cache_meta(self.START, None)) is not None

    def test_a_legacy_json_cache_is_still_served_whole_as_a_list(
            self, tmp_path, monkeypatch, caplog):
        archive_markets, live_markets = TestShardedFetch()._fixture_markets()
        archive = _FakeArchive(archive_markets)
        _install_sharded_fakes(monkeypatch, tmp_path, archive, TestShardedFetch.CUTOFF)
        legacy = [{"ticker": "OLD1", "event_title": "x"}, {"ticker": "OLD2"}]
        (tmp_path / "cache").mkdir(parents=True)
        (tmp_path / "cache" / "settled_markets_2026-06-05.json").write_text(json.dumps(legacy))
        with caplog.at_level(logging.INFO):
            out, live = self._again(live_markets)
        # Served whole, as a list: since P2 a LegacySettledCorpus, the list
        # subclass that only adds the file time as provenance (DR-13).
        assert isinstance(out, list) and type(out) is historical.LegacySettledCorpus
        assert not isinstance(out, historical.SettledCorpus) and out == legacy
        assert archive.calls == 0 and live.calls == 0
        assert "Loaded 2 settled markets from cache" in caplog.text
        assert not list((tmp_path / "cache").glob("*.jsonl.gz"))  # a hit writes nothing

    def test_the_streamed_cache_wins_over_a_legacy_one_and_a_rebuild_retires_it(
            self, tmp_path, monkeypatch, caplog):
        out, archive, live_markets = self._fetch(tmp_path, monkeypatch)
        fresh = list(out)
        legacy_path = tmp_path / "cache" / "settled_markets_2026-06-05.json"
        legacy_path.write_text(json.dumps([{"ticker": "STALE"}]))

        hit, _ = self._again(live_markets)
        assert isinstance(hit, historical.SettledCorpus) and list(hit) == fresh
        # A hit changes nothing on disk.
        assert legacy_path.read_text() == '[{"ticker": "STALE"}]'

        archive.calls = 0
        with caplog.at_level(logging.INFO):
            refreshed, _ = self._again(live_markets, use_cache=False)
        assert archive.calls > 0
        assert isinstance(refreshed, historical.SettledCorpus)
        assert list(refreshed) == fresh
        # The committed rebuild supersedes the legacy file of the same
        # identity, as the old code's rebuild overwrote it in place — and says
        # so, since it can be a GB-scale file.
        assert not legacy_path.exists()
        assert ("Removed the superseded legacy settled-market cache "
                "settled_markets_2026-06-05.json") in caplog.text

    @pytest.mark.parametrize("loss", ["damaged", "vanished", "rename_reverted"])
    def test_a_superseded_legacy_cache_is_never_served_again(
            self, tmp_path, monkeypatch, caplog, loss):
        # SS-1 review: a legacy .json of the same identity used to outlive the
        # rebuild that superseded it and come back whenever the .jsonl.gz was
        # damaged (after a WARNING claiming a cache miss) or missing (silently)
        # — including the way iCloud reverted a committed rename while this repo
        # lived in iCloud-synced ~/Documents (until 2026-09-24). The stale
        # corpus must never be served.
        stale = [{"ticker": "STALE1"}, {"ticker": "STALE2"}]
        (tmp_path / "cache").mkdir(parents=True)
        legacy_path = tmp_path / "cache" / "settled_markets_2026-06-05.json"
        legacy_path.write_text(json.dumps(stale))
        # The operator's rebuild (the BS-02 / subtitle-drift remedy).
        out, archive, live_markets = self._fetch(tmp_path, monkeypatch)
        fresh = list(out)
        assert fresh and not legacy_path.exists()
        if loss == "damaged":
            data = out.path.read_bytes()
            out.path.write_bytes(data[: len(data) // 2])
        elif loss == "vanished":
            out.path.unlink()
        else:
            out.path.rename(out.path.with_name(out.path.name + ".tmp"))

        archive.calls = 0
        with caplog.at_level(logging.INFO):
            again, _ = self._again(live_markets)
        assert archive.calls > 0  # rebuilt, not served from any cache
        assert list(again) == fresh
        assert "STALE" not in caplog.text
        assert "Loaded" not in caplog.text

    def test_a_damaged_streamed_cache_rebuilds_rather_than_serving_a_legacy_file(
            self, tmp_path, monkeypatch, caplog):
        # The other half: even if a legacy file of the same identity is
        # present beside a DAMAGED streamed cache (its retirement failed, or
        # it was restored by hand), the brief's rule holds — an invalid
        # streamed cache is a WARNING and a REBUILD, never a fall-through to
        # the older assembly the streamed one superseded.
        out, archive, live_markets = self._fetch(tmp_path, monkeypatch)
        fresh = list(out)
        legacy_path = tmp_path / "cache" / "settled_markets_2026-06-05.json"
        legacy_path.write_text(json.dumps([{"ticker": "STALE"}]))
        data = out.path.read_bytes()
        out.path.write_bytes(data[: len(data) // 2])

        archive.calls = 0
        with caplog.at_level(logging.INFO):
            rebuilt, _ = self._again(live_markets)
        assert "Corrupt or mismatched settled-market cache" in caplog.text
        assert archive.calls > 0
        assert isinstance(rebuilt, historical.SettledCorpus)
        assert list(rebuilt) == fresh
        assert not legacy_path.exists()

    def test_a_rebuild_that_publishes_nothing_retires_nothing(self, tmp_path, monkeypatch):
        # The legacy file is deleted only AFTER the replacement is committed:
        # a rebuild that fails leaves the operator's existing cache in place,
        # exactly as the old atomic overwrite did.
        (tmp_path / "cache").mkdir(parents=True)
        legacy_path = tmp_path / "cache" / "settled_markets_2026-06-05.json"
        legacy_path.write_text(json.dumps([{"ticker": "KEEP"}]))
        archive_markets, live_markets = TestShardedFetch()._fixture_markets()
        _install_sharded_fakes(monkeypatch, tmp_path, _FakeArchive(archive_markets),
                               TestShardedFetch.CUTOFF)

        def titles(live_client, tickers, use_cache=True):
            # Between walk A and walk B: walk B cannot reproduce walk A.
            historical._day_store_path("live_days", _day_lo("2026-06-10")).unlink()
            return {}

        monkeypatch.setattr(historical, "_load_or_build_event_titles", titles)
        with pytest.raises(historical.SettledCorpusError):
            historical.fetch_all_settled_markets(
                MagicMock(), _FakeLive(live_markets), start_date=self.START,
                use_cache=False)
        assert legacy_path.read_text() == '[{"ticker": "KEEP"}]'
        assert not (tmp_path / "cache" / "settled_markets_2026-06-05.jsonl.gz").exists()

    def test_retiring_the_legacy_cache(self, tmp_path, caplog):
        streamed = tmp_path / "settled_markets_2026-06-05.jsonl.gz"
        # Absent: the common case, silently nothing.
        with caplog.at_level(logging.INFO):
            historical._retire_legacy_cache(tmp_path / "absent.json", streamed)
        assert caplog.text == ""
        # Cannot be removed (a directory stands in for a locked file): a
        # WARNING telling the operator to delete it, and the run goes on.
        blocked = tmp_path / "settled_markets_2026-06-05.json"
        blocked.mkdir()
        with caplog.at_level(logging.WARNING):
            historical._retire_legacy_cache(blocked, streamed)
        assert "Could not remove the superseded legacy settled-market cache" in caplog.text
        assert "Delete it by hand" in caplog.text
        assert blocked.exists()

    def test_a_corpus_walk_raises_when_its_file_is_replaced_or_vanishes(
            self, tmp_path, monkeypatch):
        out, _, _ = self._fetch(tmp_path, monkeypatch)
        records, n = list(out), len(out)
        assert n > 1
        # Replaced (atomically, by the same writer) with one record fewer: a
        # COMPLETE walk must not end as if nothing happened.
        _write_jsonl(out.path, historical._assembled_cache_meta(self.START, None),
                     records[:-1])
        with pytest.raises(historical.SettledCorpusError, match=f"held {n}"):
            list(out)
        # An abandoned walk claims nothing and is not checked.
        assert next(iter(out))["ticker"] == records[0]["ticker"]
        out.path.unlink()
        with pytest.raises(historical.SettledCorpusError, match="Re-run"):
            list(out)


class _NoNetwork:
    """A client that fails the test on ANY attribute access — a stronger zero-
    network pin than counting the calls one fake happens to serve."""

    def __getattr__(self, name):
        raise AssertionError(f"a cache hit touched the network client ({name})")


def _forbid_network(monkeypatch):
    """Make every network path of fetch_all_settled_markets fail the test:
    the signed raw GET (the cutoff read and the archive walk) and event-title
    resolution. Clients passed in should be _NoNetwork()."""
    def signed(*_a, **_k):
        raise AssertionError("a cache hit issued a signed GET")

    def titles(*_a, **_k):
        raise AssertionError("a cache hit resolved event titles")

    monkeypatch.setattr(historical, "_signed_raw_get", signed)
    monkeypatch.setattr(historical, "_load_or_build_event_titles", titles)


class TestCorpusProvenance:
    """DR-13 and M2/M3 of the 2026-09-24 7-day-run review. A cache hit used to
    log one "Loaded N" line and return: nothing said the corpus stops at its
    assembly while the window runs to today, the post-cutoff "structurally
    0-trade" WARNING — logged only after the cutoff read, which a hit never
    reaches — vanished from every cached re-run, and an EMPTY cache was a
    permanent hit. Now the archive cutoff is stamped into the streamed cache
    (informational, never part of its identity), every hit announces what it
    covers and repeats the verdict "as of assembly" with ZERO network calls,
    and an empty cache is served only while younger than
    EMPTY_ASSEMBLED_CACHE_MAX_AGE_SECONDS."""

    PRE = date(2026, 6, 5)     # before TestShardedFetch.CUTOFF (2026-06-10)
    POST = date(2026, 6, 10)   # exactly on it: at-or-after is post-cutoff
    CUTOFF_DT = datetime(2026, 6, 10, tzinfo=UTC)

    def _fetch(self, tmp_path, monkeypatch, start):
        archive_markets, live_markets = TestShardedFetch()._fixture_markets()
        archive = _FakeArchive(archive_markets)
        _install_sharded_fakes(monkeypatch, tmp_path, archive, TestShardedFetch.CUTOFF)
        out = historical.fetch_all_settled_markets(
            MagicMock(), _FakeLive(live_markets), start_date=start, use_cache=False)
        return out, archive, live_markets

    @staticmethod
    def _hit(start):
        return historical.fetch_all_settled_markets(
            _NoNetwork(), _NoNetwork(), start_date=start, use_cache=True)

    @staticmethod
    def _meta_of(path):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.loads(fh.readline())["meta"]

    # ── the stamp and the fresh provenance ────────────────────────────────

    def test_a_fresh_assembly_stamps_the_cutoff_and_carries_its_provenance(
            self, tmp_path, monkeypatch):
        before = datetime.now(UTC)
        out, _, _ = self._fetch(tmp_path, monkeypatch, self.PRE)
        after = datetime.now(UTC)
        meta = self._meta_of(out.path)
        assert meta["archive_cutoff_ts"] == int(self.CUTOFF_DT.timestamp())
        assert before <= datetime.fromisoformat(meta["assembled_at"]) <= after
        prov = out.provenance
        # M9: the assembly's counts ride along too — with no prefilter nothing
        # is rejected, and the counts keep exactly the corpus's records
        counts = prov.assembly_counts
        assert counts is not None and counts.rejected == 0
        assert counts.kept == len(out) and counts.settled >= len(out)
        assert meta["assembly_counts"] == {
            "settled": counts.settled, "rejected": 0, "duplicates": counts.duplicates}
        assert prov == historical.CorpusProvenance(
            from_cache=False, assembled_at=datetime.fromisoformat(meta["assembled_at"]),
            archive_cutoff=self.CUTOFF_DT, post_cutoff=False, assembly_counts=counts)

    def test_the_stamp_is_informational_not_identity(self, tmp_path, monkeypatch):
        # A file whose informational keys differ from anything this run would
        # write is still served — they describe WHEN, not WHICH request.
        out, _, _ = self._fetch(tmp_path, monkeypatch, self.PRE)
        records = list(out)
        meta = {**historical._assembled_cache_meta(self.PRE, None),
                "assembled_at": "2026-06-20T08:00:00+00:00",
                "archive_cutoff_ts": int(datetime(2026, 5, 1, tzinfo=UTC).timestamp())}
        _write_jsonl(out.path, meta, records)
        _forbid_network(monkeypatch)
        hit = self._hit(self.PRE)
        assert list(hit) == records
        assert hit.provenance.archive_cutoff == datetime(2026, 5, 1, tzinfo=UTC)
        assert hit.provenance.assembled_at == datetime(2026, 6, 20, 8, tzinfo=UTC)

    # ── the hit announcement ──────────────────────────────────────────────

    def test_a_hit_announces_what_it_covers_with_zero_network_calls(
            self, tmp_path, monkeypatch, caplog):
        out, _, _ = self._fetch(tmp_path, monkeypatch, self.PRE)
        fresh = out.provenance
        _forbid_network(monkeypatch)
        caplog.clear()
        with caplog.at_level(logging.INFO):
            hit = self._hit(self.PRE)
        assert list(hit) == list(out)
        # The same meta block, read back: identical facts, flagged as cached.
        assert hit.provenance == historical.CorpusProvenance(
            from_cache=True, assembled_at=fresh.assembled_at,
            archive_cutoff=fresh.archive_cutoff, post_cutoff=False,
            assembly_counts=fresh.assembly_counts)
        text = caplog.text
        assert f"Loaded {len(out)} settled markets from cache" in text
        # ...with the assembly's counts, as of assembly (M9)
        counts = fresh.assembly_counts
        assert (f"Loaded {len(out)} settled markets from cache — as assembled, of "
                f"{counts.settled} records settled since {self.PRE} "
                f"({counts.duplicates} duplicate or blank tickers)") in text
        assert (f"Assembled cache {out.path.name} was assembled at "
                f"{fresh.assembled_at:%Y-%m-%d %H:%M UTC}") in text
        assert "holds no market settled after that moment" in text
        assert "the window nominally runs to today" in text
        assert "archive cutoff at assembly: 2026-06-10" in text
        assert "Pass --no-cache to extend it" in text
        # The remedy is priced honestly (P2 review): a re-assembly that reuses
        # only still-valid day slices, plus a candlestick and title refetch —
        # never "mainly the current day".
        assert "re-assembles the whole corpus" in text
        assert "archive day slice goes stale whenever the archive cutoff advances" in text
        assert "re-fetches every pair's candlesticks and re-resolves event titles" in text
        assert "close to a full fetch" in text
        assert "mainly the current day" not in text
        # A pre-cutoff window's hit repeats no post-cutoff WARNING.
        assert not [r for r in caplog.records if r.levelname == "WARNING"]

    def test_a_post_cutoff_hit_repeats_the_warning_as_of_assembly_with_zero_network_calls(
            self, tmp_path, monkeypatch, caplog):
        with caplog.at_level(logging.WARNING):
            out, _, _ = self._fetch(tmp_path, monkeypatch, self.POST)
        # The miss path's WARNING is unchanged, word for word.
        miss = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"
                and "archive cutoff" in r.getMessage()]
        assert miss == ["start_date (2026-06-10) is at or after the archive cutoff "
                        "(2026-06-10) — post-cutoff markets 404 on the historical "
                        "candlesticks endpoint, so this window is structurally 0-trade"]
        assert out.provenance.post_cutoff is True

        caplog.clear()
        _forbid_network(monkeypatch)
        with caplog.at_level(logging.WARNING):
            hit = self._hit(self.POST)
        assert hit.provenance.post_cutoff is True and hit.provenance.from_cache is True
        warned = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert len(warned) == 1
        assert "start_date (2026-06-10) is at or after the archive cutoff (2026-06-10)" \
            in warned[0]
        assert "as of this cached corpus's assembly" in warned[0]
        assert "structurally 0-trade unless the cutoff has since moved" in warned[0]
        assert "pass --no-cache to re-check it" in warned[0]

    def test_a_cache_written_before_the_stamp_is_served_and_says_so(
            self, tmp_path, monkeypatch, caplog):
        # The real 2026-09-17 cache on disk carries assembled_at but no
        # archive_cutoff_ts: it must still hit, with an unknown verdict — and
        # no post-cutoff WARNING can be claimed without a network read.
        out, _, _ = self._fetch(tmp_path, monkeypatch, self.POST)
        records = list(out)
        _write_jsonl(out.path, {**historical._assembled_cache_meta(self.POST, None),
                                "assembled_at": "2026-06-11T09:30:00+00:00"}, records)
        _forbid_network(monkeypatch)
        caplog.clear()  # the fresh fetch above logged the miss path's WARNING
        with caplog.at_level(logging.INFO):
            hit = self._hit(self.POST)
        assert list(hit) == records
        assert hit.provenance == historical.CorpusProvenance(
            from_cache=True, assembled_at=datetime(2026, 6, 11, 9, 30, tzinfo=UTC),
            archive_cutoff=None, post_cutoff=None)
        assert "the archive cutoff was not recorded when it was assembled" in caplog.text
        assert not [r for r in caplog.records if r.levelname == "WARNING"]

    def test_a_legacy_hit_names_its_file_time_and_carries_it_as_provenance(
            self, tmp_path, monkeypatch, caplog):
        # Seven of the eight assembled caches on disk on 2026-09-24 were
        # legacy files: their file time must reach the page, not only the log
        # (P2 review, DR-66) — so the list comes back as a LegacySettledCorpus.
        monkeypatch.setattr(historical, "CACHE_DIR", tmp_path / "cache")
        (tmp_path / "cache").mkdir(parents=True)
        legacy = tmp_path / "cache" / "settled_markets_2026-06-10.json"
        legacy.write_text(json.dumps([{"ticker": "OLD1"}]))
        written = datetime(2026, 6, 12, 7, 45, tzinfo=UTC)
        os.utime(legacy, (written.timestamp(), written.timestamp()))
        _forbid_network(monkeypatch)
        with caplog.at_level(logging.INFO):
            out = self._hit(self.POST)
        assert type(out) is historical.LegacySettledCorpus
        assert out == [{"ticker": "OLD1"}] and len(out) == 1
        assert out.provenance == historical.CorpusProvenance(
            from_cache=True, assembled_at=written, archive_cutoff=None,
            post_cutoff=None, legacy=True)
        assert ("Legacy assembled cache settled_markets_2026-06-10.json was last "
                "written at 2026-06-12 07:45 UTC") in caplog.text
        assert "its file time" in caplog.text
        assert "the legacy format records no archive cutoff" in caplog.text
        assert "(and rebuild it in the streamed format)" in caplog.text
        # No cutoff was recorded, so no verdict is claimed on a legacy hit.
        assert not [r for r in caplog.records if r.levelname == "WARNING"]

    def test_a_legacy_file_that_is_not_a_list_is_returned_as_before(
            self, tmp_path, monkeypatch):
        # Only a list is wrapped: whatever else a damaged legacy file holds is
        # handed back exactly as before P2, untouched.
        monkeypatch.setattr(historical, "CACHE_DIR", tmp_path / "cache")
        (tmp_path / "cache").mkdir(parents=True)
        (tmp_path / "cache" / "settled_markets_2026-06-10.json").write_text(
            json.dumps({"ticker": "NOT-A-LIST"}))
        _forbid_network(monkeypatch)
        out = self._hit(self.POST)
        assert type(out) is dict and out == {"ticker": "NOT-A-LIST"}

    # ── an EMPTY assembled cache (DR-13's empty-cache rule) ───────────────

    @pytest.mark.parametrize("age_s, served", [
        (60, True),
        (historical.EMPTY_ASSEMBLED_CACHE_MAX_AGE_SECONDS - 60, True),
        (historical.EMPTY_ASSEMBLED_CACHE_MAX_AGE_SECONDS + 60, False),
        (30 * 86_400, False),
        (None, False),          # no readable assembly time: fail toward a miss
        (-3_600, False),        # a future stamp is not "young" either
    ])
    def test_an_empty_streamed_cache_is_served_only_while_young(
            self, tmp_path, monkeypatch, caplog, age_s, served):
        out, archive, live_markets = self._fetch(tmp_path, monkeypatch, self.PRE)
        fresh = list(out)
        meta = historical._assembled_cache_meta(self.PRE, None)
        if age_s is not None:
            meta["assembled_at"] = (datetime.now(UTC) - timedelta(seconds=age_s)).isoformat()
        _write_jsonl(out.path, meta, [])
        archive.calls = 0
        live = _FakeLive(live_markets)
        with caplog.at_level(logging.WARNING):
            again = historical.fetch_all_settled_markets(
                MagicMock(), live, start_date=self.PRE, use_cache=True)
        if served:
            assert len(again) == 0 and list(again) == []
            assert archive.calls == 0 and live.calls == 0
            assert "is EMPTY" in caplog.text and "serving it" in caplog.text
        else:
            assert archive.calls > 0  # re-assembled from the API + day slices
            assert list(again) == fresh
            assert "is EMPTY and was assembled" in caplog.text
            assert "treating it as a miss and re-assembling" in caplog.text

    def test_an_empty_stale_streamed_cache_rebuilds_and_never_falls_through_to_legacy(
            self, tmp_path, monkeypatch, caplog):
        # P2 review (R4): the stale-empty branch is a miss that REBUILDS, like
        # an invalid streamed cache — a legacy file of the same stem beside it
        # is an older assembly and must not be served, and the rebuild's
        # commit retires it.
        out, archive, live_markets = self._fetch(tmp_path, monkeypatch, self.PRE)
        fresh = list(out)
        meta = {**historical._assembled_cache_meta(self.PRE, None),
                "assembled_at": (datetime.now(UTC) - timedelta(days=30)).isoformat()}
        _write_jsonl(out.path, meta, [])
        legacy = tmp_path / "cache" / "settled_markets_2026-06-05.json"
        legacy.write_text(json.dumps([{"ticker": "LEG"}]))
        archive.calls = 0
        with caplog.at_level(logging.INFO):
            again = historical.fetch_all_settled_markets(
                MagicMock(), _FakeLive(live_markets), start_date=self.PRE,
                use_cache=True)
        assert isinstance(again, historical.SettledCorpus)
        assert archive.calls > 0 and list(again) == fresh
        assert "LEG" not in {m["ticker"] for m in again}
        assert not legacy.exists()
        assert "treating it as a miss and re-assembling" in caplog.text
        assert "Removed the superseded legacy settled-market cache" in caplog.text

    @pytest.mark.parametrize("age_s, served", [(3_600, True), (24 * 86_400, False)])
    def test_an_empty_legacy_cache_is_served_only_while_young(
            self, tmp_path, monkeypatch, caplog, age_s, served):
        # The 2-byte "[]" settled_markets_2026-08-29_*.json on disk was last
        # written 2026-09-01 00:16 UTC (its file time) and was still a hit on
        # 2026-09-24, some 23 days later.
        archive_markets, live_markets = TestShardedFetch()._fixture_markets()
        archive = _FakeArchive(archive_markets)
        _install_sharded_fakes(monkeypatch, tmp_path, archive, TestShardedFetch.CUTOFF)
        (tmp_path / "cache").mkdir(parents=True)
        legacy = tmp_path / "cache" / "settled_markets_2026-06-05.json"
        legacy.write_text("[]")
        stamp = datetime.now(UTC).timestamp() - age_s
        os.utime(legacy, (stamp, stamp))
        with caplog.at_level(logging.INFO):
            out = historical.fetch_all_settled_markets(
                MagicMock(), _FakeLive(live_markets), start_date=self.PRE,
                use_cache=True)
        if served:
            assert type(out) is historical.LegacySettledCorpus and out == []
            assert out.provenance.legacy is True and out.provenance.from_cache is True
            assert archive.calls == 0
            assert legacy.exists()
        else:
            assert archive.calls > 0
            assert isinstance(out, historical.SettledCorpus) and len(out) > 0
            # The rebuild supersedes the empty legacy file like any other.
            assert not legacy.exists()
            assert "Removed the superseded legacy settled-market cache" in caplog.text

    # ── the derivation itself ─────────────────────────────────────────────

    @pytest.mark.parametrize("meta, expected", [
        ({"start_date": "2026-06-10", "archive_cutoff_ts": 1_781_049_600},
         (datetime(2026, 6, 10, tzinfo=UTC), True)),          # start == cutoff
        ({"start_date": "2026-06-09", "archive_cutoff_ts": 1_781_049_600},
         (datetime(2026, 6, 10, tzinfo=UTC), False)),         # a day before
        ({"start_date": "2026-06-10", "archive_cutoff_ts": 1_781_049_601},
         (datetime(2026, 6, 10, 0, 0, 1, tzinfo=UTC), False)),  # a second later
        ({"start_date": "2026-06-10", "archive_cutoff_ts": True}, (None, None)),
        ({"start_date": "2026-06-10", "archive_cutoff_ts": "1781049600"}, (None, None)),
        ({"start_date": "2026-06-10"}, (None, None)),
        ({"start_date": "garbage", "archive_cutoff_ts": 1_781_049_600}, (None, None)),
    ])
    def test_the_verdict_derivation(self, meta, expected):
        prov = historical._corpus_provenance(meta, from_cache=True)
        assert (prov.archive_cutoff, prov.post_cutoff) == expected

    @pytest.mark.parametrize("raw, expected", [
        ("2026-09-24T12:37:49.789667+00:00",
         datetime(2026, 9, 24, 12, 37, 49, 789667, tzinfo=UTC)),
        ("2026-09-24T05:37:49-07:00", datetime(2026, 9, 24, 12, 37, 49, tzinfo=UTC)),
        ("2026-09-24T12:37:49", datetime(2026, 9, 24, 12, 37, 49, tzinfo=UTC)),
        ("yesterday", None),
        (1_790_000_000, None),
        (None, None),
    ])
    def test_the_assembly_stamp_is_read_as_a_utc_instant(self, raw, expected):
        assert historical._parse_assembled_at(raw) == expected


class TestAssemblyCounts:
    """M9 of the 2026-09-24 7-day-run review: the fetch's count lines counted
    only prefilter SURVIVORS while calling them settled markets, and nothing
    reported what the prefilter dropped. The assembly's first walk now counts
    every record settled in the window and what the prefilter and the dedup
    removed; the total is stamped into the assembled cache (informational,
    never identity), read back on a hit, and quoted on the hit's count line —
    which names its records "eligible markets" when a prefilter ran."""

    START = date(2026, 6, 5)

    @staticmethod
    def _pred(m):
        return not m["ticker"].endswith("1")

    def _fetch(self, tmp_path, monkeypatch):
        archive_markets, live_markets = TestShardedFetch()._fixture_markets()
        _install_sharded_fakes(monkeypatch, tmp_path, _FakeArchive(archive_markets),
                               TestShardedFetch.CUTOFF)
        return historical.fetch_all_settled_markets(
            MagicMock(), _FakeLive(live_markets), start_date=self.START,
            use_cache=False, prefilter=self._pred, prefilter_tag="t")

    def _hit(self):
        return historical.fetch_all_settled_markets(
            _NoNetwork(), _NoNetwork(), start_date=self.START, use_cache=True,
            prefilter=self._pred, prefilter_tag="t")

    @staticmethod
    def _rewrite_meta(path, **changes):
        records = list(historical._day_store_iter(
            path, historical._assembled_cache_meta(TestAssemblyCounts.START, "t")))
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            meta = json.loads(fh.readline())["meta"]
        meta.update(changes)
        meta = {k: v for k, v in meta.items() if v is not None}
        _write_jsonl(path, meta, records)
        return records

    def test_a_prefiltered_fresh_fetch_logs_settled_beside_eligible(
            self, tmp_path, monkeypatch, caplog):
        with caplog.at_level(logging.INFO):
            out = self._fetch(tmp_path, monkeypatch)
        counts = out.provenance.assembly_counts
        assert counts.kept == len(out) and counts.rejected > 0
        messages = [r.getMessage() for r in caplog.records]
        total = [m for m in messages if m.startswith("Assembled ")]
        assert total == [
            f"Assembled {len(out)} eligible markets of {counts.settled} records "
            f"settled since {self.START} ({counts.rejected} rejected by the "
            f"prefilter t, {counts.duplicates} duplicate or blank tickers)"]
        # The endpoint lines name eligible markets too, and sum to the total.
        hist = [m for m in messages if m.startswith("Historical endpoint: ")]
        live = [m for m in messages if m.startswith("Live endpoint: ")]
        assert len(hist) == len(live) == 1
        assert " eligible markets of " in hist[0] and " eligible markets of " in live[0]
        assert "rejected by the prefilter t" in hist[0] + live[0]
        # The misstatements M9 names are gone: no line counts the survivors
        # as "settled markets" any more.
        assert not [m for m in messages if m.startswith("Total settled markets")
                    or (m.startswith("Live endpoint: ")
                        and "recently settled markets" in m)]

    def test_a_prefiltered_hit_names_eligible_markets_and_quotes_the_counts(
            self, tmp_path, monkeypatch, caplog):
        out = self._fetch(tmp_path, monkeypatch)
        fresh = out.provenance.assembly_counts
        _forbid_network(monkeypatch)
        caplog.clear()
        with caplog.at_level(logging.INFO):
            hit = self._hit()
        assert list(hit) == list(out)
        assert hit.provenance.assembly_counts == fresh
        assert (f"Loaded {len(out)} eligible markets from cache — as assembled, of "
                f"{fresh.settled} records settled since {self.START} "
                f"({fresh.rejected} rejected by the prefilter t, "
                f"{fresh.duplicates} duplicate or blank tickers)") in caplog.text
        assert "settled markets from cache" not in caplog.text

    def test_a_cache_written_before_the_counts_says_it_records_none(
            self, tmp_path, monkeypatch, caplog):
        # The real 2026-09-17 cache on disk has no assembly_counts: served,
        # with the line saying the counts are missing rather than implying
        # that nothing was rejected.
        out = self._fetch(tmp_path, monkeypatch)
        records = self._rewrite_meta(out.path, assembly_counts=None)
        _forbid_network(monkeypatch)
        caplog.clear()
        with caplog.at_level(logging.INFO):
            hit = self._hit()
        assert list(hit) == records
        assert hit.provenance.assembly_counts is None
        assert (f"Loaded {len(records)} eligible markets from cache — the "
                "prefilter t ran during its assembly, but this cache records no "
                "count of the records it rejected") in caplog.text
        assert not [r for r in caplog.records if r.levelname == "WARNING"]

    def test_counts_that_do_not_keep_the_corpus_are_dropped_loudly(
            self, tmp_path, monkeypatch, caplog):
        # A block whose counts keep a different number of records than the
        # file holds describes some other assembly: served, counts dropped.
        out = self._fetch(tmp_path, monkeypatch)
        records = self._rewrite_meta(
            out.path, assembly_counts={"settled": len(out) + 50, "rejected": 10,
                                       "duplicates": 0})
        _forbid_network(monkeypatch)
        caplog.clear()
        with caplog.at_level(logging.INFO):
            hit = self._hit()
        assert list(hit) == records
        assert hit.provenance.assembly_counts is None
        warned = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert warned == [
            f"Settled-market cache {out.path.name} records assembly counts that "
            f"keep {len(out) + 40} records, but it holds {len(records)} — "
            f"ignoring those counts"]
        assert "records no count of the records it rejected" in caplog.text

    def test_a_legacy_hit_under_a_prefilter_says_it_records_no_counts(
            self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(historical, "CACHE_DIR", tmp_path / "cache")
        (tmp_path / "cache").mkdir(parents=True)
        (tmp_path / "cache" / f"settled_markets_{self.START}_t.json").write_text(
            json.dumps([{"ticker": "OLD1"}]))
        _forbid_network(monkeypatch)
        with caplog.at_level(logging.INFO):
            out = self._hit()
        assert out == [{"ticker": "OLD1"}]
        assert out.provenance.assembly_counts is None
        assert ("Loaded 1 eligible markets from cache — the prefilter t ran "
                "during its assembly, but this cache records no count of the "
                "records it rejected") in caplog.text

    @pytest.mark.parametrize("raw", [
        None, "12", [13, 2, 4], {"settled": 13, "rejected": 2},
        {"settled": 13, "rejected": 2, "duplicates": True},
        {"settled": 13, "rejected": -1, "duplicates": 4},
        {"settled": 13, "rejected": 2.0, "duplicates": 4},
        {"settled": 13, "rejected": "2", "duplicates": 4},
        {"settled": 5, "rejected": 2, "duplicates": 4},
    ])
    def test_unreadable_counts_read_as_none(self, raw):
        assert historical._parse_assembly_counts(raw) is None

    def test_readable_counts_parse(self):
        counts = historical._parse_assembly_counts(
            {"settled": 13, "rejected": 2, "duplicates": 4, "extra": "ignored"})
        assert counts == historical.AssemblyCounts(13, 2, 4)
        assert counts.kept == 7
        assert historical._parse_assembly_counts(
            {"settled": 0, "rejected": 0, "duplicates": 0}).kept == 0


class TestNothingIsMaterialized:
    """The point of SS-1 Commit C. Every record parsed off disk is made weakly
    referenceable and counted while alive: between the two assembly walks (the
    old code held every selected record in a dict right there), at every
    record written into the assembled cache, and at every step of a walk over
    the returned corpus. At most the record in hand plus the open file's meta
    line may be alive — a bound independent of how many records there are."""

    def test_no_record_read_off_disk_outlives_its_turn(self, tmp_path, monkeypatch):
        alive = _track_disk_records(monkeypatch)
        archive_markets = [
            _mk_raw_market(f"A{d}{i:02d}", f"2026-06-0{d}T{i:02d}:10:00Z",
                           f"2026-06-0{d}T{i:02d}:40:00Z")
            for d in (6, 7, 8) for i in range(12)
        ]
        live_markets = [
            _mk_raw_market(f"L{d}{i:02d}", f"2026-06-{d}T00:00:00Z",
                           f"2026-06-{d}T{i:02d}:30:00Z")
            for d in (10, 11) for i in range(12)
        ]
        _install_sharded_fakes(monkeypatch, tmp_path,
                               _FakeArchive(archive_markets, page_size=5),
                               TestShardedFetch.CUTOFF)
        at_titles: list[int] = []

        def titles(live_client, tickers, use_cache=True):
            at_titles.append(alive())
            return {}

        monkeypatch.setattr(historical, "_load_or_build_event_titles", titles)
        at_write: list[int] = []
        real_write = historical._DayStreamWriter.write_record

        def spy_write(self, record):
            at_write.append(alive())
            return real_write(self, record)

        monkeypatch.setattr(historical._DayStreamWriter, "write_record", spy_write)
        out = historical.fetch_all_settled_markets(
            MagicMock(), _FakeLive(live_markets, page_size=5),
            start_date=date(2026, 6, 5), use_cache=False)

        n = len(out)
        assert n == len(archive_markets) + len(live_markets) == 60
        assert at_titles == [0]
        assert len(at_write) == n and max(at_write) <= 2
        assert alive() == 0
        during = [alive() for _record in out]
        assert len(during) == n and max(during) <= 2
        assert alive() == 0
