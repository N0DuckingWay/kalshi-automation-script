"""
File: historical.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Fetches and caches historical Kalshi market data needed by the backtester.
    Two data types are collected: (1) settled market metadata (title, outcome,
    prices, timestamps) from both the /historical/markets endpoint and the
    regular /markets?status=settled endpoint (which covers more recent settlements);
    and (2) hourly candlestick price series for individual markets used to find
    the week when each pair first became tradeable. All data is cached to JSON
    files on disk so re-runs do not re-fetch from the API.

Dependencies:
    Imports build_client from auth.py; api_call_with_retry and fetch_json_page
    from _http.py; and PROJECT_ROOT plus a dozen-plus tuning constants
    (MARKET_PAGE_SIZE, MVE_TITLE_LOOKUP_MAX_PAGES, SETTLED_FETCH_MAX_WORKERS,
    SETTLED_FETCH_CHUNK_RECORDS, ARCHIVE_MAX_BARREN_PAGES, ARCHIVE_TAIL_MAX_PAGES,
    ARCHIVE_TAIL_MAX_RECORDS,
    EVENT_TITLE_FALLBACK_MAX_LOOKUPS, EVENT_TITLE_FALLBACK_MAX_WORKERS,
    EVENT_TITLE_LISTING_MAX_BARREN_PAGES, CANDLESTICK_PERIOD_INTERVAL_MINUTES,
    CANDLESTICK_MAX_CANDLES_PER_REQUEST, INCLUDE_MVE_MARKETS, PROD_URL) from
    config.py. Exports
    build_historical_client() and build_prod_live_client(), both called by
    backtest.py (NOT backtester.py, which never builds its own clients); and
    fetch_all_settled_markets(), fetch_candlesticks(), and infer_category(),
    all called by backtester.py.

Notes:
    Historical market data only exists on the production API — the sandbox does
    not have a historical endpoint. The backtest always uses prod credentials.
    Candlesticks are cached per ticker in backtest_cache/candlesticks/<ticker>.json
    to avoid thousands of API calls on repeated runs. Candles are fetched at
    CANDLESTICK_PERIOD_INTERVAL_MINUTES (hourly) — daily candles only cover
    markets whose lifespan crosses a UTC midnight boundary, which silently
    excludes most Kalshi markets (see config.py for the full explanation).
    The candlestick endpoint serves at most
    CANDLESTICK_MAX_CANDLES_PER_REQUEST candles per request and refuses a
    longer one with HTTP 400, so fetch_candlesticks pages any longer window
    into consecutive requests and merges them in timestamp order.

    The pinned SDK (kalshi-python-sync==3.2.0) ships no historical_api module,
    and 2026-07 API drift broke its Market response model anyway (legacy
    integer-cent fields are no longer sent). All /historical endpoints are
    therefore reached with direct signed GETs (_signed_raw_get) through the
    SDK's own KalshiAuth + rest client, and responses are parsed as raw JSON —
    same pattern as scanner.py/trader.py (see _http.fetch_json_page).

    fetch_all_settled_markets is sharded and incremental (2026-07 rewrite):
    the /historical/markets archive is paged by (created_time DESC, ticker
    DESC) behind a protobuf cursor that this module can synthesize, so the
    archive walk is split into per-day slices fetched in parallel and cached
    to disk one day at a time (backtest_cache/archive_days/). The live
    recently-settled sweep is likewise split into per-settled-day windows
    (backtest_cache/live_days/) using the documented min/max_settled_ts
    params. Both sharded paths runtime-verify their assumptions and fall back
    to the original sequential walks if the API drifts. See
    fetch_all_settled_markets for the full contract.

    Day-slice files come in two interchangeable formats: the legacy single
    gzipped JSON document written by _day_store_save, and the streamed
    "jsonl-v1" line format written by _DayStreamWriter (which is what the
    fetch workers use, so no worker ever holds a whole UTC day — millions of
    records at 2026-08 volumes — in memory). _day_store_load reads both; see
    the day-store section for the routing rule. The live frontier (current,
    partial) day is never persisted; it streams through a sink that keeps only
    the caller's prefilter-passing records as each batch arrives
    (_fetch_live_phase), so when a prefilter is given the unfiltered partial
    day is never resident — its prefilter-passing subset still is, as a list,
    and on the day after a Monday that can be most of the day. The two
    sequential fallbacks apply the same prefilter per record but still hold
    their whole filtered result; the archive tail is the one walk that applies
    no prefilter (its record cap bounds it instead).

    JSON parsing dominates the fetch's CPU time at current Kalshi volumes, so
    orjson is used when installed (optional `perf` extra) and the stdlib json
    module otherwise. This is purely a speed knob: orjson emits plain JSON, so
    day-slice files written under either parser are interchangeable and no
    cache is invalidated by installing or removing it.
"""
import base64
import gzip
import json
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse

from ._http import api_call_with_retry, fetch_json_page
from .auth import build_client
from .config import (
    ARCHIVE_MAX_BARREN_PAGES,
    ARCHIVE_TAIL_MAX_PAGES,
    ARCHIVE_TAIL_MAX_RECORDS,
    CANDLESTICK_MAX_CANDLES_PER_REQUEST,
    CANDLESTICK_PERIOD_INTERVAL_MINUTES,
    EVENT_TITLE_FALLBACK_MAX_LOOKUPS,
    EVENT_TITLE_FALLBACK_MAX_WORKERS,
    EVENT_TITLE_LISTING_MAX_BARREN_PAGES,
    INCLUDE_MVE_MARKETS,
    MARKET_PAGE_SIZE,
    MVE_TITLE_LOOKUP_MAX_PAGES,
    PROD_URL,
    PROJECT_ROOT,
    SETTLED_FETCH_CHUNK_RECORDS,
    SETTLED_FETCH_MAX_WORKERS,
)

# Optional acceleration for the day-slice store, which serializes and re-parses
# tens of millions of compact market dicts per full-history fetch. orjson is an
# optional extra (`pip install -e ".[perf]"`) and emits plain JSON, so slices
# written with it are readable by the stdlib fallback and vice versa — existing
# caches never need refetching over this. See _day_store_save/_day_store_load.
try:
    import orjson

    _HAVE_ORJSON = True
except ImportError:
    orjson = None  # type: ignore[assignment]
    _HAVE_ORJSON = False

# Host and path prefix split out of PROD_URL ("https://host/trade-api/v2") so
# _signed_raw_get can sign the path component and hit arbitrary API routes.
_PROD_PARSED = urlparse(PROD_URL)
_API_HOST = f"{_PROD_PARSED.scheme}://{_PROD_PARSED.netloc}"
_API_PREFIX = _PROD_PARSED.path  # "/trade-api/v2"

CACHE_DIR = PROJECT_ROOT / "backtest_cache"
_CANDLES_DIR = CACHE_DIR / "candlesticks"
# Maps event_ticker → event_title, populated lazily by _load_or_build_event_titles
# so the backtester can construct the same (event_title + market_title) grouping
# key the live scanner uses.
_EVENT_TITLES_CACHE = CACHE_DIR / "event_titles.json"

# Cached-empty candles may be genuinely empty markets, but they are also what
# a fetch failure previously produced. Treat empty cache files as stale after
# this many seconds so a transient API failure self-heals on the next run.
_EMPTY_CANDLE_TTL_SECONDS = 86_400

# Map event_ticker prefixes to human-readable categories
_CATEGORY_PREFIXES = [
    ("KXBTC", "Crypto"), ("KXETH", "Crypto"), ("KXSOL", "Crypto"),
    ("KXNASDAQ", "Finance"), ("KXSP", "Finance"), ("KXGOLD", "Finance"),
    ("KXFED", "Economics"), ("KXINF", "Economics"), ("KXGDP", "Economics"),
    ("NFL", "Sports"), ("NBA", "Sports"), ("MLB", "Sports"), ("NHL", "Sports"),
    ("NCAA", "Sports"), ("EPL", "Sports"),
    ("PRES", "Politics"), ("SENATE", "Politics"), ("HOUSE", "Politics"),
    ("KXPOL", "Politics"),
    ("KXAI", "Tech"), ("KXTECH", "Tech"),
    ("KXWEATHER", "Weather"), ("KXTEMP", "Weather"),
]


def infer_category(event_ticker: str) -> str:
    """
    Map a Kalshi event ticker to a human-readable market category.

    Checks the ticker against a list of known prefixes (e.g. "KXBTC" → "Crypto",
    "NFL" → "Sports"). Returns "Other" if no prefix matches.

    Args:
        event_ticker (str): The event ticker string from a Kalshi market (e.g. "KXBTC-2024").
            May be None or empty, in which case "Other" is returned.

    Returns:
        str: Category label such as "Crypto", "Finance", "Sports", "Politics", or "Other".
    """
    upper = (event_ticker or "").upper()
    for prefix, cat in _CATEGORY_PREFIXES:
        if upper.startswith(prefix):
            return cat
    return "Other"


def _signed_raw_get(client: Any, path: str, **params):
    """
    Perform a signed GET against an arbitrary API path via the SDK's transport.

    The pinned SDK has no historical_api module, so /historical routes cannot
    be reached through generated methods. This helper signs the request the
    same way every SDK call is signed (KalshiAuth: RSA-PSS over
    timestamp+method+path) and executes it with the client's own rest client,
    returning the RESTResponse so fetch_json_page can apply the shared
    status-check + JSON-parse contract.

    Args:
        client (Any): Authenticated KalshiClient from auth.build_client().
        path (str): Full API path including the /trade-api/v2 prefix
            (e.g. "/trade-api/v2/historical/markets").
        **params: Query parameters; None values are omitted.

    Returns:
        RESTResponse: The raw response (status + body bytes), unparsed.
    """
    query = urlencode({k: v for k, v in params.items() if v is not None})
    url = f"{_API_HOST}{path}" + (f"?{query}" if query else "")
    # Query strings are excluded from the signature by Kalshi's auth scheme —
    # only timestamp + method + path are signed
    headers = {"accept": "application/json"}
    headers.update(client.kalshi_auth.create_auth_headers("GET", path))
    return client.rest_client.request("GET", url, headers=headers)


def _historical_get(client: Any, path: str, **params) -> dict:
    """
    Signed GET returning parsed JSON, with 429/5xx retry.

    Composes _signed_raw_get with fetch_json_page (which raises ApiException
    on non-2xx so api_call_with_retry can back off on 429/5xx exactly like the
    modeled SDK calls did).

    Args:
        client (Any): Authenticated KalshiClient from auth.build_client().
        path (str): Full API path including the /trade-api/v2 prefix.
        **params: Query parameters; None values are omitted.

    Returns:
        dict: Parsed JSON response body.

    Raises:
        ApiException: On non-2xx HTTP status after retries are exhausted.
    """
    return api_call_with_retry(fetch_json_page, partial(_signed_raw_get, client, path), **params)


def _exception_summary(exc: BaseException, limit: int = 120) -> str:
    """
    One-line, header-free rendering of an API exception for log lines.

    The SDK's ApiException.__str__ emits the status, the reason, the ENTIRE
    HTTP header dict and the body across five lines (~900 bytes), and its own
    first line is only "(404)" — the reason lives on line two. Logged once per
    failed candlestick fetch on a post-cutoff window (where every ticker 404s
    by design, and those failures are deliberately never cached so they are
    re-paid every run), that alone rotated the run's own diagnostics out of
    kalshi_backtest.log: 99.5% of the file was this one warning and ~419 MB of
    older history was evicted (TS-02).

    So the exception's own `reason` attribute is preferred when it carries
    one — that is the useful half of an ApiException — and a plain exception
    falls back to the first line of its message, then to its class name, so a
    non-API failure (a parse error, a KeyError) is never reduced to nothing.

    Args:
        exc (BaseException): The exception to summarize.
        limit (int): Maximum characters of the resulting text to keep.

    Returns:
        str: A single line, never longer than `limit` characters, and never
            containing the header dump. Never raises.
    """
    try:
        reason = getattr(exc, "reason", None)
        # str() rather than assuming a string: `reason` is whatever the SDK set.
        text = str(reason).strip() if reason else str(exc).strip()
        first = text.splitlines()[0].strip() if text else ""
        return (first or type(exc).__name__)[:limit]
    except Exception:
        # The docstring promises this never raises, and every caller invokes it
        # from INSIDE an `except` block on a per-ticker path — a pathological
        # __str__ or a raising `reason` property must degrade to the class name,
        # not escape a worker and kill the run.
        return type(exc).__name__[:limit]


def build_historical_client():
    """
    Construct the client used for /historical endpoint requests.

    Historical market data (older settled markets) only exists on the production
    endpoint — the sandbox does not expose a /historical/markets endpoint. The
    pinned SDK has no historical_api module, so this returns a standard
    authenticated KalshiClient whose credentials and rest client power the
    direct signed GETs in _signed_raw_get (see module Notes).

    Returns:
        KalshiClient: An authenticated client pointed at the production endpoint.
    """
    # build_client("prod") returns a KalshiClient authenticated via RSA key from secrets.json
    return build_client("prod")


def build_prod_live_client():
    """
    Construct a standard KalshiClient pointed at production for recently-settled markets.

    Used alongside build_historical_client() to cover recently settled markets
    that are not yet in the historical archive (i.e. settled after the API cutoff
    timestamp from the /historical/cutoff endpoint).

    Returns:
        KalshiClient: An authenticated client pointed at the production endpoint.
    """
    # build_client("prod") returns a KalshiClient authenticated via RSA key from secrets.json
    return build_client("prod")


# ─── Serialization helpers ────────────────────────────────────────────────────

def _market_to_dict(m: dict, event_title: str = "") -> dict:
    """
    Normalize a raw market JSON dict to the flat dict format the backtester caches.

    Extracts only the fields needed by the backtester. The raw API already
    sends open_time/close_time/settlement_ts as ISO 8601 strings and prices as
    `*_dollars` strings, so values pass through unchanged; missing fields
    become the same falsy defaults the old SDK-model path produced.

    Args:
        m (dict): One market object from a raw markets-endpoint JSON payload.
        event_title (str): The market's parent event title, resolved by
            `_load_or_build_event_titles` (see `fetch_all_settled_markets`) so
            the backtester can group pairs by combined event+market title.
            Defaults to "" (ungrouped / non-MVE).

    Returns:
        dict: A flat dictionary with keys: ticker, event_ticker, event_title,
            title, subtitle, result, yes_ask_dollars, no_ask_dollars,
            yes_bid_dollars, open_time, close_time, settlement_ts, status,
            price_level_structure, price_ranges, exchange_index.
            Note: open_time is a new field (added alongside the backtester's
            eligibility prefilter) — cache files written before this change
            don't have it and will read back as None until refreshed with
            --no-cache; backtester._can_ever_enter() treats that as "can't
            prove ineligibility" and keeps the market (no speedup, no
            incorrect drop).
            Note: subtitle is sourced from the legacy `subtitle` key with a
            fallback to `yes_sub_title`, which is where the archive now puts
            the outcome label (2026-08 drift). The cache key name is unchanged
            so backtester._group_by_exact_title and the display labels read it
            as before. Day-slice files written before this change carry
            subtitle=None and reproduce the pre-fix (weakened) same-title
            grouping; `--no-cache` alone does NOT refresh them, since day
            slices can't go stale by construction — a full refresh requires
            deleting backtest_cache/archive_days/ and backtest_cache/live_days/
            AND re-running with --no-cache, because a default run loads the
            assembled backtest_cache/settled_markets_*.json first and returns
            before the day slices are consulted at all.
            Note: price_level_structure/price_ranges are new, raw pass-through
            groundwork fields (2026-08) for a future tick-aware order cap —
            nothing in the backtester reads them yet. Cache records written
            before this change lack both and read back as None. They are
            stored raw (not parsed into scanner.PriceRange objects) because
            this dict is JSON-serialized straight into the cache file, and
            parsed dataclasses wouldn't round-trip through json.dump. Size
            impact is negligible (a short string plus a handful of small
            dicts per market) relative to the fields already kept here —
            deliberately considered against the OOM history documented in
            CLAUDE.md, which was caused by caching whole raw payloads, not by
            small additions to this already-compact record.
    """
    return {
        "ticker": m.get("ticker"),
        "event_ticker": m.get("event_ticker") or "",
        # Resolved by fetch_all_settled_markets via _load_or_build_event_titles
        "event_title": event_title or "",
        "title": m.get("title"),
        # `subtitle` was dropped from API payloads (2026-08 drift);
        # yes_sub_title carries the same outcome discriminator. Cache records
        # written before this fix keep subtitle=None — that reproduces the
        # pre-fix (weakened) grouping. Refreshing needs BOTH deleting
        # backtest_cache/{archive_days,live_days}/ and a --no-cache run (which
        # bypasses the assembled settled_markets_*.json that would otherwise
        # short-circuit the fetch before any slice is read).
        "subtitle": m.get("subtitle") or m.get("yes_sub_title"),
        "result": m.get("result"),
        "yes_ask_dollars": m.get("yes_ask_dollars"),
        "no_ask_dollars": m.get("no_ask_dollars"),
        "yes_bid_dollars": m.get("yes_bid_dollars"),
        # Feeds backtester._can_ever_enter()'s eligibility prefilter — needed
        # to prove a market's tradeable window contains no Monday checkpoint
        "open_time": m.get("open_time"),
        "close_time": m.get("close_time"),
        "settlement_ts": m.get("settlement_ts"),
        "status": m.get("status"),
        # Tick-structure groundwork (2026-08): raw pass-through for cache
        # fidelity; no reader yet. Pre-existing cache records read back None.
        "price_level_structure": m.get("price_level_structure"),
        "price_ranges": m.get("price_ranges"),
        # Shard fidelity (2026-08): raw pass-through so backtest data can
        # distinguish exchange shards. Stored, never filtered — all historical
        # markets are shard 0, and filtering would silently change backtest
        # results with no live-safety gain (the live order path is guarded at
        # scanner ingest). Pre-existing cache records read back None.
        "exchange_index": m.get("exchange_index"),
    }


def _load_json_cache(path: Path):
    """
    Load and parse a JSON file from disk if it exists, treating corruption as a miss.

    A truncated or otherwise unreadable file is NOT an error: these caches are
    written by multi-hour fetches that have historically been interrupted by
    OOM kills and SIGKILLs, and the only safe interpretation of a half-written
    file is "no cache" — same guarded-read philosophy as _day_store_load.

    Args:
        path (Path): Filesystem path to the JSON cache file.

    Returns:
        Any: Parsed JSON content (typically a list or dict) if the file exists
            and parses, or None if the file is absent, unreadable, or corrupt.
    """
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        # Truncated/corrupt file (interrupted write, OOM, SIGKILL — all
        # documented past events for these multi-hour fetches) is a cache
        # miss, not a crash.
        logging.warning("Corrupt JSON cache %s — treating as cache miss", path)
        return None


def _save_json_cache(path: Path, data) -> None:
    """
    Atomically serialize data to JSON at path, creating parent directories as needed.

    Uses a default=str serializer to handle datetime objects that may appear in
    the data. The write goes to a sibling temp file that is then renamed over
    the destination, so an interrupted run can never leave a truncated file
    that a later run would read back and trust.

    Args:
        path (Path): Destination file path. Parent directories are created if absent.
        data: JSON-serializable data structure (list, dict, etc.) to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic tmp+replace, same idiom as _day_store_save. The tmp name is derived
    # from the destination, so it is unique because cache paths themselves are
    # unique (per-ticker for candlesticks, per-start-date for assembled settled
    # markets) — the same path-uniqueness invariant that keeps the parallel
    # candlestick fetch safe. Never introduce a fetch whose cache path is shared
    # across workers.
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, default=str))
    tmp.replace(path)


def _load_or_build_event_titles(
    live_client,
    event_tickers: set[str],
    use_cache: bool = True,
) -> dict[str, str]:
    """
    Resolve a set of event_ticker values to their human-readable event titles.

    The backtester needs event titles to construct the same combined
    (event_title + market_title) grouping key the live scanner uses. The Market
    objects returned by the historical endpoint only carry event_ticker, so we
    fetch the title mapping separately via the events endpoint.

    Two-tier resolution to keep API calls bounded:
      1. Bulk pull events (settled, closed, open) and — only when
         INCLUDE_MVE_MARKETS is True — their multivariate counterparts.
         For most backtests this covers nearly every event_ticker in a few hundred
         paginated calls.
      2. For any tickers still unresolved (very old archived events that have aged
         out of the bulk listings), fall back to per-ticker get_event() calls —
         run in parallel and capped at EVENT_TITLE_FALLBACK_MAX_LOOKUPS, since
         each costs a round trip and the miss set can run to six figures at
         current Kalshi volumes.

    Every phase logs progress. This function can legitimately run for many
    minutes, and when it was silent an in-progress run was indistinguishable
    from a hang (observed 2026-08-03).

    Results are persisted to _EVENT_TITLES_CACHE so subsequent runs are essentially
    free. Tickers that cannot be resolved — lookup failed, or skipped by the cap —
    are stored as empty strings (poison pill) so we do not retry them every run.

    The on-disk cache is a cross-run ACCUMULATOR and is written as a MERGE, so a
    single run (in particular a `--no-cache` run, which resolves from scratch)
    can never wipe titles other runs paid for. Merge rule: disk entries are
    preserved; a fresh non-empty title wins over the disk value; a fresh "" —
    the poison pill from a failed lookup or the fallback cap — NEVER clobbers a
    non-empty disk title, but a fresh "" for a ticker unknown to disk IS stored
    (poison-pill semantics preserved).

    An unresolved ticker is not an error: the caller groups those markets by
    market title alone, which is exactly what a failed lookup has always done.

    Args:
        live_client: A KalshiClient with the events API methods available
            (e.g. the client from build_prod_live_client()).
        event_tickers (set[str]): The set of event_ticker values whose titles
            we need. May contain hundreds or thousands of entries.
        use_cache (bool): If True, seed resolution from the on-disk accumulator.
            If False, start empty so this run's tickers are genuinely re-fetched.
            The on-disk accumulator is read (for the merge) and rewritten either
            way — a `--no-cache` run refreshes its own tickers without discarding
            titles it did not ask about.

    Returns:
        dict[str, str]: Mapping event_ticker → event_title, restricted to
            event_tickers, read from the MERGED view — this run's resolution
            layered over the on-disk accumulator. Tickers that could not be
            resolved anywhere map to "". Caller treats those markets as
            ungrouped (effectively MVE-excluded).

            It used to return this run's resolution ALONE, which under
            --no-cache handed back the "" poison pill for every ticker the
            listings missed and the EVENT_TITLE_FALLBACK_MAX_LOOKUPS cap
            skipped — even when disk held a real title fetched by an earlier
            run. Measured: 771,601 unresolved against a 5,000 cap, so ~99% of
            stragglers. Those markets then group by market title alone,
            collapsing the same-title key (event_title, title, subtitle)
            toward the bare title: on a captured payload, 52 groups with
            titles became 133 groups with a largest of 112 without — the
            direction that manufactures cross-event false positives under the
            95% co-resolution prior (TS-11).
    """
    # Always read the accumulator: even when use_cache is False and it must not
    # seed resolution, it is needed at save time so this run's writes MERGE with
    # (rather than replace) titles earlier runs paid round trips for. Corrupt
    # file → {} plus a warning, via the guarded loader.
    disk_titles: dict[str, str] = _load_json_cache(_EVENT_TITLES_CACHE) or {}
    cached: dict[str, str] = dict(disk_titles) if use_cache else {}
    missing = event_tickers - cached.keys()
    if not missing:
        # Restricted to the caller's tickers for the same reason the merged
        # return below is: the accumulator holds every ticker every past run
        # ever resolved, and a caller asking about 40 must not receive 800k.
        return {tkr: cached.get(tkr, "") for tkr in event_tickers}

    # Bulk pull non-MVE events across all statuses. Each get_events call returns
    # up to 200 events; pagination continues until cursor is empty or all misses
    # are resolved (whichever comes first).
    #
    # Raw-response call: the modeled get_events deserializes into EventData,
    # whose `category` field the pinned SDK types as a REQUIRED string. The live
    # API now sends `category: null` on some events (observed 2026-08-03), so
    # the modeled call raises pydantic ValidationError mid-listing. Same drift,
    # and same fix, as the market/order/orderbook endpoints — see module Notes.
    total_missing = len(missing)
    for status in ("settled", "closed", "open"):
        if not missing:
            break
        cursor = None
        pages = 0
        barren = 0  # consecutive pages that resolved nothing
        while True:
            # Events-listing page cap (MARKET_PAGE_SIZE, 200) — a different,
            # smaller cap than the /historical & /markets endpoints' 1000 (see
            # the hist_kwargs comment in fetch_all_settled_markets).
            kwargs: dict = {"status": status, "limit": MARKET_PAGE_SIZE}
            if cursor:
                kwargs["cursor"] = cursor
            data = api_call_with_retry(
                fetch_json_page, live_client.get_events_without_preload_content, **kwargs
            )
            resolved_here = 0
            for ev in data.get("events") or []:
                tkr = ev.get("event_ticker")
                if tkr in missing:
                    cached[tkr] = ev.get("title") or ""
                    missing.discard(tkr)
                    resolved_here += 1
            pages += 1
            # Without this the whole phase is silent for however long it runs —
            # which at current volumes is long enough to look like a hang.
            if pages % 100 == 0:
                logging.info("Event titles [%s listing]: %d pages scanned, "
                             "%d/%d still unresolved",
                             status, pages, len(missing), total_missing)
            # Productivity bail-out: this listing is a full scan looking for a
            # specific ticker set, so once it stops hitting wanted tickers it
            # will not start again — keep paging and it burns minutes finding
            # nothing (see EVENT_TITLE_LISTING_MAX_BARREN_PAGES).
            barren = 0 if resolved_here else barren + 1
            if barren >= EVENT_TITLE_LISTING_MAX_BARREN_PAGES:
                logging.info(
                    "Event titles [%s listing]: no new titles in %d consecutive "
                    "pages after %d scanned — moving on with %d/%d unresolved",
                    status, barren, pages, len(missing), total_missing,
                )
                break
            cursor = data.get("cursor")
            if not cursor or not missing:
                break

    # Bulk pull multivariate events — these are excluded from get_events by API
    # design. The MVE listing is effectively unbounded (hundreds of thousands of
    # auto-generated collection events), so this loop is capped at
    # MVE_TITLE_LOOKUP_MAX_PAGES; tickers not found by then fall through to the
    # bounded per-ticker lookup below instead of paging for hours.
    # Raw-response for the same nullable-category reason as above.
    # Only worth paging when MVE markets can be in the wanted set at all —
    # with INCLUDE_MVE_MARKETS off every market fetch excluded them upstream.
    if missing and INCLUDE_MVE_MARKETS:
        cursor = None
        barren = 0
        for page_no in range(1, MVE_TITLE_LOOKUP_MAX_PAGES + 1):
            # Same events-listing page cap as the status-listing loop above.
            kwargs = {"limit": MARKET_PAGE_SIZE}
            if cursor:
                kwargs["cursor"] = cursor
            data = api_call_with_retry(
                fetch_json_page,
                live_client.get_multivariate_events_without_preload_content,
                **kwargs,
            )
            resolved_here = 0
            for ev in data.get("events") or []:
                tkr = ev.get("event_ticker")
                if tkr in missing:
                    cached[tkr] = ev.get("title") or ""
                    missing.discard(tkr)
                    resolved_here += 1
            if page_no % 100 == 0:
                logging.info("Event titles [MVE listing]: %d/%d pages scanned, "
                             "%d/%d still unresolved", page_no,
                             MVE_TITLE_LOOKUP_MAX_PAGES, len(missing), total_missing)
            # Same productivity bail-out as the status listings above; the MVE
            # listing is the most unbounded of the three.
            barren = 0 if resolved_here else barren + 1
            if barren >= EVENT_TITLE_LISTING_MAX_BARREN_PAGES:
                logging.info(
                    "Event titles [MVE listing]: no new titles in %d consecutive "
                    "pages after %d scanned — moving on with %d/%d unresolved",
                    barren, page_no, len(missing), total_missing,
                )
                break
            cursor = data.get("cursor")
            if not cursor or not missing:
                break

    # Per-ticker fallback for events that aren't in the bulk listings (or fell
    # past the MVE page cap). Uses a raw signed GET because the modeled
    # get_event embeds nested Market models the pinned SDK can no longer
    # deserialize (see module Notes). Failures are recorded as "" so we don't
    # retry on every backtest run.
    #
    # This costs one HTTP round-trip per ticker, so it is BOUNDED and RUN IN
    # PARALLEL: it was written for a handful of stragglers, but at current
    # Kalshi volumes the bulk listings can leave hundreds of thousands
    # unresolved (live-measured 2026-08-03), and sequentially that is hours of
    # silent grinding. Anything past the cap is poison-pilled to "" — the same
    # value a failed lookup produces, so the caller's behaviour is unchanged.
    if missing:
        to_look_up = sorted(missing)  # sorted → deterministic which ones the cap keeps
        capped = to_look_up[EVENT_TITLE_FALLBACK_MAX_LOOKUPS:]
        to_look_up = to_look_up[:EVENT_TITLE_FALLBACK_MAX_LOOKUPS]
        if capped:
            # Never drop coverage silently — say exactly how much was skipped.
            logging.warning(
                "Event titles: %d tickers unresolved after the bulk listings; "
                "looking up %d individually and marking the remaining %d as "
                "untitled (cap EVENT_TITLE_FALLBACK_MAX_LOOKUPS=%d). Those "
                "markets group by market title alone, exactly as a failed "
                "lookup would.",
                len(missing), len(to_look_up), len(capped),
                EVENT_TITLE_FALLBACK_MAX_LOOKUPS,
            )
        else:
            logging.info("Event titles: looking up %d tickers individually",
                         len(to_look_up))
        for tkr in capped:
            cached[tkr] = ""

        def _lookup_one(tkr: str) -> tuple[str, str]:
            """Resolve one event title; "" on any failure (poison pill)."""
            try:
                data = _historical_get(live_client, f"{_API_PREFIX}/events/{tkr}")
                return tkr, (data.get("event") or {}).get("title") or ""
            except Exception as e:
                # Same one-line treatment as the candlestick failure (TS-02):
                # this loop runs once per unresolved ticker, up to
                # EVENT_TITLE_FALLBACK_MAX_LOOKUPS of them per run.
                logging.warning("Could not resolve event title for %s: HTTP %s %s",
                                tkr, getattr(e, "status", "?"), _exception_summary(e))
                return tkr, ""

        done = 0
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=EVENT_TITLE_FALLBACK_MAX_WORKERS) as pool:
            for tkr, title in pool.map(_lookup_one, to_look_up):
                cached[tkr] = title
                done += 1
                if done % 500 == 0:
                    elapsed = max(time.monotonic() - started, 1e-9)
                    remaining = (len(to_look_up) - done) / (done / elapsed)
                    logging.info("Event titles: %d/%d individual lookups done "
                                 "(ETA %s)", done, len(to_look_up),
                                 _format_duration(remaining))

    # Merge into the on-disk accumulator rather than overwriting it: disk entries
    # are preserved, a fresh non-empty title wins over the disk value, and a
    # fresh "" (poison pill from a failed lookup or the fallback cap) never
    # clobbers a non-empty disk title — but a fresh "" for a ticker disk has
    # never seen IS stored, so poison-pill semantics survive.
    merged = dict(disk_titles)
    for tkr, title in cached.items():
        if title or not merged.get(tkr):
            merged[tkr] = title
    _save_json_cache(_EVENT_TITLES_CACHE, merged)
    # Return the MERGED view, restricted to what the caller asked about. The
    # accumulator exists precisely so a ticker resolved by an earlier run need
    # not be re-fetched; returning `cached` threw that away at the last step and
    # substituted the "" poison pill (TS-11).
    result = {tkr: merged.get(tkr, "") for tkr in event_tickers}
    # Count the substitutions so the accumulator's contribution is visible
    # rather than inferred — this is the number that was silently lost.
    from_accumulator = sum(
        1 for tkr in event_tickers if not cached.get(tkr) and merged.get(tkr)
    )
    logging.info(
        "Event titles resolved: %d this run, %d cached entries on disk, "
        "%d of this run's tickers answered from the accumulator",
        len(cached), len(merged), from_accumulator,
    )
    return result


# ─── Market fetching ──────────────────────────────────────────────────────────

class _ShardedFetchUnsupported(Exception):
    """
    Raised when a runtime self-check shows the sharded fetch path cannot be
    trusted: the archive cursor format has drifted, archive records stopped
    carrying created_time, or the live endpoint stopped honoring
    max_settled_ts. fetch_all_settled_markets catches this and falls back to
    the original sequential walks, which rely only on documented behavior.
    """


# One archive/live day slice, in seconds. Slices are UTC calendar days.
_DAY_SECONDS = 86_400

# The /historical/markets archive rejects limit > 1000 and ignores every
# server-side time-filter param (min/max_settled_ts and friends — all
# live-verified 2026-07-13 to return the identical newest-first page), so the
# only way to avoid one serial walk of the multi-million-record archive is its
# pagination cursor. That cursor is a urlsafe-base64 protobuf of the keyset
# position (created_time DESC, ticker DESC):
#   field 1: google.protobuf.Timestamp of the last record's created_time
#   field 2: the last record's ticker
# Verified 2026-07-13: re-encoding a page's last record reproduces the server
# cursor byte-for-byte, and a synthesized cursor continues the record stream
# exactly. _archive_cursor_synthesis_ok re-verifies this at the start of every
# fetch, so format drift degrades to the sequential path instead of silently
# corrupting results.
#
# The sentinel sorts above every real ticker at a created_time tie, so a
# synthesized cursor (T, sentinel) yields every record with created_time <= T
# under the DESC tie-break. Records at exactly T with a ticker sorting above
# the sentinel are still covered, because the slice ABOVE sweeps down through
# its own lower boundary — no record can fall between two adjacent slices.
_CURSOR_TICKER_SENTINEL = "ZZZZZZZZZZ"


def _iso_epoch(ts: str | None) -> float | None:
    """
    Parse an ISO 8601 timestamp string to epoch seconds.

    Args:
        ts (str | None): ISO timestamp as sent by the API (e.g.
            "2026-05-13T23:09:56.165186Z"), or None/empty.

    Returns:
        float | None: Epoch seconds, or None if ts is missing or unparseable.
    """
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts).timestamp()
    except (ValueError, TypeError):
        return None


def _iso_epoch_parts(ts: str | None) -> tuple[int, int] | None:
    """
    Parse an ISO 8601 timestamp into (whole epoch seconds, nanoseconds).

    Used to rebuild archive cursors, whose protobuf Timestamp splits the
    instant into integer seconds + nanos. The API sends microsecond precision,
    so nanos is microsecond * 1000 — this matches the server's own cursor
    encoding (verified byte-for-byte, see _CURSOR_TICKER_SENTINEL block).

    Args:
        ts (str | None): ISO timestamp string, or None.

    Returns:
        tuple[int, int] | None: (seconds, nanos), or None if unparseable.
    """
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None
    # Zero out the microseconds before .timestamp() so float rounding can
    # never bleed a fraction into the integer seconds.
    return int(dt.replace(microsecond=0).timestamp()), dt.microsecond * 1000


def _pb_varint(value: int) -> bytes:
    """
    Encode a non-negative integer as a protobuf base-128 varint.

    Args:
        value (int): Non-negative integer to encode.

    Returns:
        bytes: Varint encoding (little-endian groups of 7 bits).

    Raises:
        ValueError: If value is negative — Python's arithmetic right shift
            never carries a negative number to 0, so the encoding loop below
            would otherwise run forever.
    """
    if value < 0:
        raise ValueError(f"varint requires a non-negative integer, got {value}")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _pb_read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    """
    Decode a protobuf varint from buf starting at pos.

    Args:
        buf (bytes): Buffer containing the varint.
        pos (int): Offset of the varint's first byte.

    Returns:
        tuple[int, int]: (decoded value, offset just past the varint).

    Raises:
        IndexError: If the varint runs past the end of buf.
    """
    value = shift = 0
    while True:
        byte = buf[pos]
        value |= (byte & 0x7F) << shift
        pos += 1
        if not (byte & 0x80):
            return value, pos
        shift += 7


def _encode_archive_cursor(seconds: int, nanos: int, ticker: str) -> str:
    """
    Build an archive pagination cursor for a (created_time, ticker) position.

    Produces the exact protobuf layout the server itself emits (field 1: a
    nested Timestamp message with seconds/nanos varints; field 2: the ticker
    string), base64url-encoded without padding.

    Args:
        seconds (int): created_time whole epoch seconds of the position.
        nanos (int): created_time nanoseconds (0 for synthesized boundaries).
        ticker (str): Ticker of the position; use _CURSOR_TICKER_SENTINEL for
            synthesized slice boundaries.

    Returns:
        str: Cursor accepted by /historical/markets' cursor parameter.
    """
    ts_msg = b"\x08" + _pb_varint(seconds)
    if nanos:
        ts_msg += b"\x10" + _pb_varint(nanos)
    tkr = ticker.encode()
    msg = (b"\x0a" + _pb_varint(len(ts_msg)) + ts_msg
           + b"\x12" + _pb_varint(len(tkr)) + tkr)
    return base64.urlsafe_b64encode(msg).decode().rstrip("=")


def _decode_archive_cursor(cursor: str) -> tuple[int, int, str] | None:
    """
    Decode an archive pagination cursor into its keyset position.

    Args:
        cursor (str): Cursor string returned by /historical/markets.

    Returns:
        tuple[int, int, str] | None: (created_time seconds, nanos, ticker),
            or None if the cursor doesn't parse as the expected protobuf.
    """
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        pos = 0
        seconds = nanos = 0
        ticker = ""
        while pos < len(raw):
            tag, pos = _pb_read_varint(raw, pos)
            field, wire_type = tag >> 3, tag & 7
            if wire_type == 2:  # length-delimited
                length, pos = _pb_read_varint(raw, pos)
                payload = raw[pos:pos + length]
                pos += length
                if field == 1:  # nested Timestamp
                    sub = 0
                    while sub < len(payload):
                        sub_tag, sub = _pb_read_varint(payload, sub)
                        sub_val, sub = _pb_read_varint(payload, sub)
                        if sub_tag >> 3 == 1:
                            seconds = sub_val
                        elif sub_tag >> 3 == 2:
                            nanos = sub_val
                elif field == 2:
                    ticker = payload.decode()
            elif wire_type == 0:
                _, pos = _pb_read_varint(raw, pos)
            else:
                return None
        return seconds, nanos, ticker
    except (ValueError, IndexError, UnicodeDecodeError):
        return None


class _FetchProgress:
    """
    Thread-safe page/market counters shared by parallel fetch workers.

    Emits an INFO log line every 100 pages with the same wording the old
    sequential loops used, so long fetches remain observable in the log.
    """

    def __init__(self, label: str):
        """
        Args:
            label (str): Log-line prefix. MUST distinguish the sharded path
                from the sequential fallback — they emit otherwise identical
                progress lines, and on 2026-08-03 that ambiguity led to a live
                run being misdiagnosed as having fallen back when it had not.
        """
        self._label = label
        self._lock = threading.Lock()
        self.pages = 0
        self.kept = 0

    def tick(self, kept_delta: int) -> None:
        """
        Record one fetched page and how many of its markets were kept.

        Args:
            kept_delta (int): Number of markets kept from this page.
        """
        with self._lock:
            self.pages += 1
            self.kept += kept_delta
            if self.pages % 100 == 0:
                logging.info("%s: %d pages scanned, %d markets kept so far",
                             self._label, self.pages, self.kept)


def _log_slice_progress(label: str, done: int, total: int, day_lo: int,
                        markets: int, started: float) -> None:
    """
    Log completion of one day slice with a running rate and ETA.

    A full-history fetch runs for hours across hundreds of slices; without a
    per-slice line the only feedback is a page counter that says nothing about
    how far along the run actually is.

    Rate units are adaptive: a slow multi-hour fetch (e.g. a handful of
    slices per hour) rendered at "%.1f slices/min" rounds to "0.0" beside a
    perfectly finite ETA, which reads as broken math rather than "just slow".
    Below 0.1 slices/min the rate is shown as slices/hour instead; the ETA
    calculation itself is unchanged either way.

    Args:
        label (str): Phase name, e.g. "Archive day slices".
        done (int): Slices completed so far, including this one.
        total (int): Total slices this run must fetch.
        day_lo (int): UTC-midnight lower bound of the completed slice.
        markets (int): Records persisted for this slice.
        started (float): time.monotonic() when the pool started.
    """
    elapsed = max(time.monotonic() - started, 1e-9)
    rate = done / elapsed  # slices per second
    eta = (total - done) / rate if rate > 0 else 0.0
    rate_per_min = rate * 60
    if rate_per_min >= 0.1:
        rate_str = f"{rate_per_min:.1f} slices/min"
    else:
        rate_str = f"{rate * 3600:.1f} slices/hour"
    logging.info(
        "%s: %d/%d complete (%s: %d markets, %s, ETA %s)",
        label, done, total,
        datetime.fromtimestamp(day_lo, tz=UTC).date().isoformat(),
        markets, rate_str, _format_duration(eta),
    )


def _format_duration(seconds: float) -> str:
    """
    Render a duration as a compact human-readable string.

    Args:
        seconds (float): Non-negative duration in seconds.

    Returns:
        str: e.g. "45s", "12m", "3h41m".
    """
    seconds = int(max(seconds, 0))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


# ─── Day-slice disk store ─────────────────────────────────────────────────────

# Day slices exist in TWO on-disk formats, both first-class and both readable:
#
#   legacy dict  — one gzipped JSON document {"meta": {...}, "markets": [...]}.
#                  Written by _day_store_save. Hundreds of MB of these already
#                  sit in backtest_cache/, so they must keep loading forever.
#   "jsonl-v1"   — gzipped JSON Lines: line 1 is {"meta": {..., "format":
#                  "jsonl-v1"}}, every later line is ONE compact market dict.
#                  Written incrementally by _DayStreamWriter so a fetch worker
#                  never holds a whole UTC day (millions of records) in RAM.
#
# The tag lives inside the meta block, which is otherwise pass-through, so an
# expect_meta identity check is unaffected by its presence.
_SLICE_FORMAT_JSONL = "jsonl-v1"


def _discard_all(_record: dict) -> bool:
    """
    `keep` predicate that retains nothing, for validity probes.

    The phase prescans load every candidate slice purely to decide whether it
    is reusable; passing this means a streamed slice is still fully parsed (so
    a corrupt one is still detected) without materializing a single record.

    Args:
        _record (dict): The record under consideration; ignored.

    Returns:
        bool: Always False.
    """
    return False


def _slice_dumps(obj) -> bytes:
    """
    Serialize one day-slice JSON value to UTF-8 bytes.

    Uses orjson when the optional `perf` extra is installed (this runs over tens
    of millions of records per full-history fetch), otherwise the stdlib. Both
    emit plain JSON with no embedded newlines outside string escapes, which is
    what makes the JSON Lines framing safe either way.

    Args:
        obj: Any JSON-serializable value (a meta block or a market dict).

    Returns:
        bytes: Compact UTF-8 JSON, with no trailing newline.
    """
    if _HAVE_ORJSON:
        return orjson.dumps(obj)
    return json.dumps(obj).encode("utf-8")


def _slice_loads(raw: bytes):
    """
    Parse one day-slice JSON document (a whole legacy payload, or one JSONL line).

    Args:
        raw (bytes): UTF-8 JSON bytes.

    Returns:
        Any: The parsed value.

    Raises:
        ValueError: If the bytes are not valid JSON. orjson raises
            JSONDecodeError, which subclasses ValueError, so callers can catch
            the one exception type regardless of which parser is installed.
    """
    return orjson.loads(raw) if _HAVE_ORJSON else json.loads(raw)


def _day_store_path(store: str, day_lo: int) -> Path:
    """
    Compute the on-disk path for one day-slice file.

    Args:
        store (str): Store subdirectory name ("archive_days" or "live_days").
        day_lo (int): Epoch seconds of the slice's UTC midnight lower bound.

    Returns:
        Path: CACHE_DIR/<store>/<YYYY-MM-DD>.json.gz. CACHE_DIR is resolved at
            call time so tests can monkeypatch it.
    """
    day = datetime.fromtimestamp(day_lo, tz=UTC).date().isoformat()
    return CACHE_DIR / store / f"{day}.json.gz"


def _prune_stale_live_days(cutoff_ts: int) -> int:
    """
    Delete live_days/ slices whose whole UTC day now lies at/before the cutoff.

    When Kalshi advances the archive cutoff, every settlement in a day that
    now lies entirely before the new cutoff migrates from the live endpoint
    into the archive — that day's records are, from then on, only ever served
    (and cached) via backtest_cache/archive_days/. Its old
    backtest_cache/live_days/<day>.json.gz slice is never read again by any
    future run (_fetch_live_phase only ever requests days at/after
    max(cutoff_ts, start_ts)): it is pure dead disk that only grows across
    repeated cutoff advances if nothing cleans it up. This is disk hygiene
    only — deleting a slice loses no data, since the same day is fetched and
    cached again as an archive slice regardless.

    Filenames follow the _day_store_path convention (CACHE_DIR/live_days/
    <YYYY-MM-DD>.json.gz, one file per UTC day); the day's lower bound is
    parsed back out of the filename rather than tracked separately.

    Args:
        cutoff_ts (int): Unix timestamp of the current archive/live boundary
            (market_settled_ts from /historical/cutoff).

    Returns:
        int: Number of slice files actually deleted.
    """
    store_dir = CACHE_DIR / "live_days"
    if not store_dir.exists():
        # Nothing fetched yet (or a fresh cache dir) — nothing to prune.
        return 0
    pruned = 0
    for path in sorted(store_dir.glob("*.json.gz")):
        day_str = path.name[: -len(".json.gz")]
        try:
            day = date.fromisoformat(day_str)
        except ValueError:
            # Not one of our day-slice filenames — leave it alone.
            continue
        day_lo = int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp())
        if day_lo + _DAY_SECONDS > cutoff_ts:
            # Day extends up to or past the cutoff — still (partly) live-only.
            continue
        try:
            path.unlink()
            pruned += 1
        except OSError as e:
            # Best-effort cleanup: a locked/already-gone file isn't fatal to
            # the fetch, just leaves a byte or two of dead disk behind.
            logging.warning("Failed to prune stale live-day slice %s: %s", path, e)
    if pruned:
        logging.info("Pruned %d stale pre-cutoff live day slice(s)", pruned)
    return pruned


def _day_store_load(
    path: Path,
    expect_meta: dict,
    keep: Callable[[dict], bool] | None = None,
) -> list[dict] | None:
    """
    Load a day-slice file if it exists and its metadata matches expectations.

    Reads BOTH slice formats (see the format note above): the legacy single
    JSON document and the streamed "jsonl-v1" line format. Routing is done on
    file CONTENT, never on whether a whole-file parse succeeds — an empty
    "jsonl-v1" day is a lone meta line, which parses perfectly well as a JSON
    document, so a "did json.loads work?" test would misclassify it as legacy
    and silently report every empty day as having no meta match.

    The JSONL path parses one line at a time and applies `keep` per record, so
    a 4.8M-record day is never materialized in full just to be filtered down
    afterwards. Any malformed line, unreadable file, or meta mismatch fails the
    WHOLE slice (returns None → the caller refetches the day); a partially
    decodable slice is never returned, which is the same all-or-nothing
    contract the single-document format always had.

    Args:
        path (Path): File path from _day_store_path().
        expect_meta (dict): Key/value pairs the file's meta block must match
            exactly (e.g. cutoff_ts, include_mve, complete). A mismatch means
            the file was written under different fetch conditions and must be
            ignored (the caller refetches the day).
        keep (Callable[[dict], bool] | None): Optional per-record predicate.
            Records for which it returns False are discarded as they are read
            and never retained. Applying it here is equivalent to filtering the
            returned list afterwards — order and membership are identical.

    Returns:
        list[dict] | None: The slice's compact market dicts (filtered by `keep`
            when given), or None when the file is absent, unreadable, or
            written under different conditions.
    """
    if not path.exists():
        return None
    try:
        # Read as bytes so the orjson and stdlib parsers take the same input.
        # Both accept UTF-8 bytes, and slices are plain JSON either way.
        with gzip.open(path, "rb") as fh:
            first_line = fh.readline()
            try:
                head = _slice_loads(first_line)
            except ValueError:
                # Not a self-contained first line. The only way our own writers
                # produce this is a corrupt file, but a legacy slice written by
                # some other tool could be pretty-printed across lines, so fall
                # through to the whole-document parse below rather than
                # declaring the file dead.
                head = None
            if isinstance(head, dict) and "markets" not in head:
                # JSONL: a dict first line WITHOUT a "markets" key is the meta
                # line (the tag is checked below for the log, not for routing —
                # the absence of "markets" is the discriminator, and it holds
                # for the meta-only empty-day file too).
                meta = head.get("meta") or {}
                if any(meta.get(k) != v for k, v in expect_meta.items()):
                    return None
                records: list[dict] = []
                for line in fh:
                    # Streaming, one record at a time: this is what keeps the
                    # load side of a multi-million-record day bounded.
                    if not line.strip():
                        continue
                    record = _slice_loads(line)
                    if keep is None or keep(record):
                        records.append(record)
                return records
            rest = fh.read()
        payload = head if (head is not None and not rest.strip()) else _slice_loads(
            first_line + rest
        )
    except (OSError, EOFError, ValueError):
        # EOFError: gzip raises it for a stream truncated before its end-of-
        # stream marker — the same "interrupted write" class as a short read.
        return None
    if not isinstance(payload, dict):
        return None
    meta = payload.get("meta") or {}
    if any(meta.get(k) != v for k, v in expect_meta.items()):
        return None
    markets = payload.get("markets") or []
    if keep is None:
        return markets
    return [m for m in markets if keep(m)]


def _day_store_save(path: Path, meta: dict, markets: list[dict]) -> None:
    """
    Atomically persist one completed day slice in the LEGACY single-document format.

    Written gzip-compressed (the settled-market history is tens of millions of
    records; plain JSON would be ~10x larger) via a temp file + rename so an
    interrupted run can never leave a truncated file that a later run would
    trust.

    Serialized with orjson when available, otherwise the stdlib json module.
    Both produce plain JSON, so a slice written by one is readable by the other
    — existing caches stay valid regardless of which is installed.

    Deliberately still writes the legacy format rather than "jsonl-v1": it is
    handed a fully-materialized list anyway, so it gains nothing from
    streaming. It is NOT on any production path — both persisting fetch
    workers use _DayStreamWriter, because they are exactly the callers that
    must not hold a whole day in memory — so this is the legacy format's
    writer of record, exercised by the tests and kept so the format both
    writers must stay compatible with cannot rot. The legacy READ path is
    live: hundreds of MB of legacy slices are already on disk and
    _day_store_load still routes to them (TS-26).

    Args:
        path (Path): Destination path from _day_store_path().
        meta (dict): Metadata block checked by _day_store_load on reuse.
        markets (list[dict]): Compact market dicts (see _market_to_dict).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    payload = {"meta": meta, "markets": markets}
    with gzip.open(tmp, "wb", compresslevel=1) as fh:
        fh.write(_slice_dumps(payload))
    tmp.replace(path)


class _DayStreamWriter:
    """
    Incremental writer for one "jsonl-v1" day slice.

    Exists so a fetch worker never accumulates a whole UTC day of records: at
    2026-08 volumes a single day reaches ~4.4–4.8M compact market dicts, and
    holding one per worker sawtoothed RSS to 7.89 GB on a 16 GB host (live
    telemetry, 2026-08-31). The worker instead hands over batches of at most
    SETTLED_FETCH_CHUNK_RECORDS records, which are serialized and compressed
    straight into the file and then dropped.

    Atomicity contract is identical to _day_store_save: everything is written
    to `<path>.tmp` and only renamed into place by commit(), so the visible
    file is either absent or complete — a later run can never trust a slice
    that was cut short by a crash, an OOM kill, or a cancelled worker.

    Record ORDER is preserved exactly as written, which is load-bearing: the
    server returns each window newest-settled first, and the caller's ticker
    dedup at merge time is first-wins over newest-day-first assembly. Reordering
    or interleaving batches would silently change which duplicate wins.

    Usable as a context manager; leaving the block without a successful
    commit() aborts, so an exception mid-fetch removes the temp file.
    """

    def __init__(self, path: Path, meta: dict):
        """
        Open the temp file and write the meta line.

        Args:
            path (Path): Final destination path from _day_store_path().
            meta (dict): Metadata block from _day_store_meta(); the
                "jsonl-v1" format tag is added here so callers cannot forget it.

        Raises:
            OSError: If the destination directory or temp file cannot be opened.
        """
        self._path = path
        self._tmp = path.with_name(path.name + ".tmp")
        self._committed = False
        self._records = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = gzip.open(self._tmp, "wb", compresslevel=1)
        try:
            # Meta first: a reader routes and gates on this line alone, without
            # touching the (potentially millions of) record lines behind it.
            self._fh.write(
                _slice_dumps({"meta": {**meta, "format": _SLICE_FORMAT_JSONL}}) + b"\n"
            )
        except BaseException:
            self.abort()
            raise

    def write_records(self, batch: list[dict]) -> None:
        """
        Append one batch of records, in order, as JSON lines.

        The batch is serialized line by line rather than joined into one big
        buffer, so peak memory stays bounded by the batch itself.

        Args:
            batch (list[dict]): Compact market dicts (see _market_to_dict).
                The caller may reuse or discard the list immediately after.

        Raises:
            OSError: On a write failure; the caller must abort() the writer.
        """
        write = self._fh.write
        for record in batch:
            write(_slice_dumps(record) + b"\n")
        self._records += len(batch)

    def commit(self) -> int:
        """
        Close the temp file and atomically publish it at the final path.

        Returns:
            int: Total records written to this slice.

        Raises:
            OSError: If the final flush or the rename fails; the slice is then
                left unpublished (the day refetches next run).
        """
        self._fh.close()
        self._tmp.replace(self._path)
        self._committed = True
        return self._records

    def abort(self) -> None:
        """
        Close and delete the temp file, leaving no visible slice behind.

        Safe to call more than once and after a failed commit; never raises,
        because it runs on the error path where the original exception must
        survive.
        """
        try:
            self._fh.close()
        except Exception:
            pass
        try:
            self._tmp.unlink()
        except OSError:
            pass

    def __enter__(self) -> "_DayStreamWriter":
        """
        Returns:
            _DayStreamWriter: This writer.
        """
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """
        Abort unless commit() already published the slice.

        Args:
            exc_type: Exception class, or None on a clean exit.
            exc: Exception instance, or None.
            tb: Traceback, or None.

        Returns:
            bool: Always False — exceptions are never suppressed (a worker
                failure must reach the pool so the phase can fall back).
        """
        if not self._committed:
            self.abort()
        return False


def _day_store_meta(lo: int, expect_meta: dict) -> dict:
    """
    Build the metadata block written alongside one completed day slice.

    Args:
        lo (int): Epoch seconds of the slice's UTC midnight lower bound.
        expect_meta (dict): Reuse-gating keys (kind, cutoff_ts, include_mve,
            complete) that _day_store_load will check on a later run.

    Returns:
        dict: expect_meta plus human-readable `day` and `fetched_at` fields.
    """
    return {
        **expect_meta,
        "day": datetime.fromtimestamp(lo, tz=UTC).date().isoformat(),
        "fetched_at": datetime.now(UTC).isoformat(),
    }


def _assemble_day_slices(
    store: str,
    day_los: list[int],
    expect_meta: dict,
    keep: Callable[[dict], bool] | None = None,
) -> list[dict]:
    """
    Rebuild a phase's record list by streaming completed day slices off disk.

    Slices are read back rather than held in RAM because a full-history fetch
    spans ~900 days at up to ~200k records each — retaining every slice (and
    then flattening it into a second list) was measured at 2.7 GB RSS only 17%
    of the way through a run, and grew superlinearly as GC pressure mounted.
    Reading them back costs one extra decode pass, which is minutes against a
    multi-hour fetch.

    Days are emitted newest-first, matching the order the in-memory version
    produced — record order is part of this module's output contract, since the
    caller's ticker dedup is first-wins.

    Args:
        store (str): Store subdirectory name ("archive_days" or "live_days").
        day_los (list[int]): UTC-midnight lower bounds of every day known to
            have a valid slice on disk (reused or just written).
        expect_meta (dict): Reuse-gating keys the slices must still match.
        keep (Callable[[dict], bool] | None): Optional predicate applied per
            record as slices are read, so records the caller would discard
            anyway are never retained. Pushed down into _day_store_load so that
            for streamed "jsonl-v1" slices the rejected records are never even
            materialized. Purely a memory optimization — the caller re-applies
            the same predicate during merge. The slice FILES are never
            filtered; they stay complete for other start dates.

    Returns:
        list[dict]: Compact market dicts, newest day first.

    Raises:
        _ShardedFetchUnsupported: If a slice that was just verified or written
            no longer loads — the caller then takes the sequential fallback
            rather than silently returning a short result set.
    """
    records: list[dict] = []
    for lo in sorted(day_los, reverse=True):
        path = _day_store_path(store, lo)
        slice_records = _day_store_load(path, expect_meta, keep)
        if slice_records is None:
            raise _ShardedFetchUnsupported(f"day slice disappeared or corrupted: {path}")
        records.extend(slice_records)
    return records


# ─── Archive (pre-cutoff) fetching ────────────────────────────────────────────

def _archive_cursor_synthesis_ok(hist_client: Any, hist_kwargs: dict) -> bool:
    """
    Runtime check that archive cursors still follow the known protobuf format.

    Fetches the archive's first page and re-encodes the last record's
    (created_time, ticker) position; if the result matches the server-returned
    cursor byte-for-byte, synthesized cursors are trustworthy and the sharded
    fetch may proceed. Any mismatch (format drift, missing fields, empty
    archive) disables sharding for this run.

    Args:
        hist_client (Any): Authenticated KalshiClient from build_historical_client().
        hist_kwargs (dict): Base query params (limit, optional mve_filter).

    Returns:
        bool: True when cursor synthesis is verified safe to use.
    """
    data = _historical_get(hist_client, f"{_API_PREFIX}/historical/markets", **hist_kwargs)
    page = data.get("markets") or []
    cursor = data.get("cursor")
    if not page or not cursor:
        return False
    last = page[-1]
    parts = _iso_epoch_parts(last.get("created_time"))
    ticker = last.get("ticker")
    if parts is None or not ticker:
        return False
    seconds, nanos = parts
    return _encode_archive_cursor(seconds, nanos, ticker) == cursor.rstrip("=")


def _fetch_archive_day(
    hist_client: Any,
    day_lo: int,
    day_hi: int,
    hist_kwargs: dict,
    progress: _FetchProgress,
    emit: Callable[[list[dict]], None] | None = None,
) -> list[dict] | int:
    """
    Fetch every settled binary market whose created_time falls in [day_lo, day_hi).

    Jumps into the created-time-ordered archive with a synthesized cursor at
    day_hi and pages downward until the slice's lower bound is crossed.
    Records outside the created window are skipped (boundary records belong to
    the adjacent slice); the settlement window is NOT applied here so the
    stored slice stays valid for any start_date.

    Args:
        hist_client (Any): Authenticated KalshiClient.
        day_lo (int): Slice lower bound, epoch seconds (UTC midnight, inclusive).
        day_hi (int): Slice upper bound, epoch seconds (exclusive).
        hist_kwargs (dict): Base query params (limit, optional mve_filter).
        progress (_FetchProgress): Shared page counter for log output.
        emit (Callable[[list[dict]], None] | None): Optional sink for chunked
            output. When given, records are handed over in batches of at most
            SETTLED_FETCH_CHUNK_RECORDS (plus a final partial batch) and the
            internal buffer is cleared each time, so the day is never held in
            memory; the return value is then the record COUNT. When None, the
            full list is accumulated and returned — the original behavior,
            retained for direct callers that want the list. No production path
            passes None: the day workers stream into a slice file, the live
            frontier window streams through a keep-filtering sink (see
            _fetch_live_phase), and the sequential fallbacks are separate walks
            that never call this function.

    Returns:
        list[dict] | int: Compact market dicts (created within the slice,
            binary result, settlement_ts present), newest created_time first —
            or, when `emit` is given, the number of records emitted.

    Raises:
        _ShardedFetchUnsupported: If the synthesized cursor lands above day_hi
            (server ignored it) or a record has no parseable created_time.
    """
    cursor = _encode_archive_cursor(day_hi, 0, _CURSOR_TICKER_SENTINEL)
    kept: list[dict] = []
    total = 0
    first_page = True
    while True:
        data = _historical_get(hist_client, f"{_API_PREFIX}/historical/markets",
                               cursor=cursor, **hist_kwargs)
        page = data.get("markets") or []
        if first_page and page:
            newest = _iso_epoch(page[0].get("created_time"))
            # An invalid synthesized cursor silently restarts from the top of
            # the archive (observed behavior) — detect the jump landing above
            # the slice and bail out rather than crawling the whole archive.
            if newest is None or newest > day_hi + 1.0:
                raise _ShardedFetchUnsupported(
                    f"synthesized cursor landed at {page[0].get('created_time')}, "
                    f"above slice bound {day_hi}"
                )
        first_page = False
        oldest_created = None
        page_kept = 0
        for m in page:
            created = _iso_epoch(m.get("created_time"))
            if created is None:
                raise _ShardedFetchUnsupported("archive record without created_time")
            oldest_created = created
            if not day_lo <= created < day_hi:
                continue
            if m.get("result") not in ("yes", "no") or not m.get("settlement_ts"):
                continue
            # Normalize immediately — holding raw archive payloads (30+ fields
            # plus nested mve_selected_legs) for millions of records is what
            # caused the multi-minute GC/memory stalls in the old fetch.
            kept.append(_market_to_dict(m))
            page_kept += 1
        total += page_kept
        # Unchanged accounting: one tick per page, carrying that page's keeps.
        progress.tick(page_kept)
        if emit is not None and len(kept) >= SETTLED_FETCH_CHUNK_RECORDS:
            # Hand the buffer over and drop it: this is what bounds a worker's
            # memory at (chunk size) instead of (whole UTC day).
            emit(kept)
            kept = []
        cursor = data.get("cursor")
        # Stop once the page bottom crossed below the slice (deeper pages are
        # older still), the archive is exhausted, or the server sent nothing.
        if not cursor or not page or (oldest_created is not None and oldest_created < day_lo):
            break
    if emit is None:
        return kept
    if kept:
        emit(kept)
    return total


def _fetch_archive_tail(
    hist_client: Any,
    start_ts: int,
    cutoff_ts: int,
    hist_kwargs: dict,
    progress: _FetchProgress,
) -> list[dict]:
    """
    Continue the archive walk below created_time == start_ts.

    The archive is ordered by created_time, not settlement time, so markets
    CREATED before start_date can still SETTLE inside the backtest window
    (long-lived markets). The per-day slices cover everything created on or
    after start_date; this tail is what collects those older-created records,
    paging downward from created_time == start_ts.

    Stop rule: no EXACT one exists on a created-ordered walk, because a market
    created arbitrarily early can still settle in-window. The walk therefore
    stops after ARCHIVE_MAX_BARREN_PAGES CONSECUTIVE pages that contain no
    in-window settlement at all (or when the archive runs out of pages). It
    deliberately does NOT stop at the first page whose newest-created record
    settled pre-window: that record's settlement says nothing about the rest
    of the page or about deeper pages, and since most markets are short-lived
    such a page shows up almost immediately — the old rule killed the tail
    after one or two pages and silently dropped long-lived in-window settlers.

    That barren rule only bounds depth PAST the last productive page, though: a
    single long-dated in-window settlement resets the counter, so the walk is
    additionally capped at ARCHIVE_TAIL_MAX_PAGES total pages. This tail is
    serial (one 1000-record request at a time) and is never persisted as a day
    slice, so it is re-paid on every run — an unbounded version can crawl most
    of created-time history. Hitting the cap logs a WARNING and truncates.

    Args:
        hist_client (Any): Authenticated KalshiClient.
        start_ts (int): Backtest window start, epoch seconds (UTC midnight).
        cutoff_ts (int): Archive/live boundary from /historical/cutoff.
        hist_kwargs (dict): Base query params (limit, optional mve_filter).
        progress (_FetchProgress): Page counter for log output. The caller
            passes a progress object labeled for the TAIL specifically — these
            pages are not day-slice pages, and CLAUDE.md's progress-label rule
            means a shared "[sharded]" label would make a stalled tail
            indistinguishable in the log from a stalled day-slice pool.

    Returns:
        list[dict]: Compact market dicts created before start_ts that settled
            within [start_ts, cutoff_ts).

    Raises:
        _ShardedFetchUnsupported: If the synthesized start cursor lands above
            start_ts (server ignored it).
    """
    cursor = _encode_archive_cursor(start_ts, 0, _CURSOR_TICKER_SENTINEL)
    kept: list[dict] = []
    first_page = True
    barren = 0
    pages = 0
    while True:
        data = _historical_get(hist_client, f"{_API_PREFIX}/historical/markets",
                               cursor=cursor, **hist_kwargs)
        pages += 1
        page = data.get("markets") or []
        if first_page and page:
            newest = _iso_epoch(page[0].get("created_time"))
            if newest is None or newest > start_ts + 1.0:
                raise _ShardedFetchUnsupported(
                    "synthesized tail cursor landed above start_ts"
                )
        first_page = False
        kept_before = len(kept)
        for m in page:
            settle = _iso_epoch(m.get("settlement_ts"))
            if settle is None or m.get("result") not in ("yes", "no"):
                continue
            if not start_ts <= settle < cutoff_ts:
                continue
            created = _iso_epoch(m.get("created_time"))
            # Records created exactly at start_ts belong to the bottom day
            # slice; skipping them here avoids double-collection (the final
            # ticker dedup would drop them anyway).
            if created is not None and created >= start_ts:
                continue
            kept.append(_market_to_dict(m))
        progress.tick(len(kept) - kept_before)
        next_cursor = data.get("cursor")
        if not next_cursor or not page:
            return kept
        # Productivity bail-out (see the stop-rule note in the docstring).
        # Judged on ANY in-window settlement on the page, deliberately
        # independent of the result/created_time keep filters above: a page
        # made entirely of voided markets, or of records skipped because they
        # belong to a day slice, still proves the walk is in productive
        # created-time territory and must not trip the counter.
        page_in_window = any(
            s is not None and start_ts <= s < cutoff_ts
            for s in (_iso_epoch(m.get("settlement_ts")) for m in page)
        )
        barren = 0 if page_in_window else barren + 1
        if barren >= ARCHIVE_MAX_BARREN_PAGES:
            logging.info(
                "Historical archive tail: %d consecutive pages with no in-window "
                "settlements (ARCHIVE_MAX_BARREN_PAGES) — stopping pagination",
                barren,
            )
            return kept
        # Absolute depth backstop: the barren rule above only bounds depth past
        # the last PRODUCTIVE page, so one long-dated in-window settlement
        # resets it and the serial, uncached tail keeps crawling.
        if pages >= ARCHIVE_TAIL_MAX_PAGES:
            logging.warning(
                "Historical archive tail: reached the %d-page cap "
                "(ARCHIVE_TAIL_MAX_PAGES) after walking %d pages below "
                "created_time == start_date — stopping. Very long-lived markets "
                "created deeper than this that settle inside the window may be "
                "missed; raise ARCHIVE_TAIL_MAX_PAGES if a run needs them.",
                ARCHIVE_TAIL_MAX_PAGES, pages,
            )
            return kept
        # Residency backstop, composing with the page cap above: whichever
        # binds first stops the walk. Like the two sequential fallbacks, this
        # walk has no chunked emit sink, so its whole result stays resident;
        # unlike them it is not `keep`-filtered, so this cap counts every
        # in-window record. The page cap alone allows ~2M records (roughly
        # 5 GB), the same OOM shape the sharded fetch exists to avoid (TS-15).
        if len(kept) >= ARCHIVE_TAIL_MAX_RECORDS:
            logging.warning(
                "Historical archive tail: reached the %d-record cap "
                "(ARCHIVE_TAIL_MAX_RECORDS) after walking %d pages below "
                "created_time == start_date — stopping to bound memory. "
                "Long-lived pre-start markets beyond this point that settle "
                "inside the window may be missed; raise the cap if a run needs "
                "them.",
                ARCHIVE_TAIL_MAX_RECORDS, pages,
            )
            return kept
        cursor = next_cursor


def _fetch_archive_sequential(
    hist_client: Any,
    start_ts: int,
    cutoff_ts: int,
    hist_kwargs: dict,
    keep: Callable[[dict], bool] | None = None,
) -> list[dict]:
    """
    Original sequential archive walk — the sharding fallback path.

    Pages the archive from the top with the server-provided cursor chain,
    keeping every binary market that settled within [start_ts, cutoff_ts)
    and passes `keep`. Relies only on documented pagination behavior (never
    on cursor synthesis), so it works even if the cursor format drifts — that
    is what makes it a safe fallback. Slow: one serial request per 1000
    records.

    Completeness is bounded, not exact. The archive is ordered by created_time,
    so a market created arbitrarily early can settle in-window and no page
    proves that deeper pages hold nothing. The walk therefore stops after
    ARCHIVE_MAX_BARREN_PAGES consecutive pages with no in-window settlement —
    the same rule as _fetch_archive_tail, and for the same reason.

    Residency: this walk has no chunked emit sink, so its whole result is held
    in memory. `keep` is applied per record as each page arrives, so a record
    the caller would discard is never retained — the same records in the same
    order as filtering the unfiltered result afterwards (fetch_all_settled_
    markets' merge re-applies the same predicate before its first-wins dedup,
    so dropping them here changes nothing downstream). What stays resident is
    still the WHOLE keep-passing result, which is not bounded by this.

    Args:
        hist_client (Any): Authenticated KalshiClient.
        start_ts (int): Backtest window start, epoch seconds.
        cutoff_ts (int): Archive/live boundary from /historical/cutoff.
        hist_kwargs (dict): Base query params (limit, optional mve_filter).
        keep (Callable[[dict], bool] | None): Optional per-record predicate
            (the caller's prefilter); None keeps every in-window binary
            record, exactly as before it existed. It never affects the walk
            itself — pages requested, the barren-page stop and the progress
            line's "markets kept so far" count are all computed before it.

    Returns:
        list[dict]: Compact market dicts settled within [start_ts, cutoff_ts)
            that pass `keep`, in walk order.
    """
    selected: list[dict] = []
    # Records that passed the walk's OWN filters (result, settlement window),
    # before `keep`: this is what the progress line has always reported, so
    # the line stays identical whether or not a predicate is passed.
    walk_kept = 0
    cursor = None
    page_no = 0
    barren = 0
    while True:
        kwargs: dict = dict(hist_kwargs)
        # Include cursor for pages after the first to continue pagination
        if cursor:
            kwargs["cursor"] = cursor
        data = _historical_get(hist_client, f"{_API_PREFIX}/historical/markets", **kwargs)
        page_markets = data.get("markets") or []
        for m in page_markets:
            # Skip markets without a settlement timestamp or with a non-binary result
            settle_epoch = _iso_epoch(m.get("settlement_ts"))
            if settle_epoch is None or m.get("result") not in ("yes", "no"):
                continue
            # Only include markets that settled within our [start_ts, cutoff_ts) window
            if settle_epoch < start_ts or settle_epoch >= cutoff_ts:
                continue
            walk_kept += 1
            rec = _market_to_dict(m)
            # The caller's prefilter, applied as the page arrives so a record
            # it rejects is never retained (the merge would drop it anyway).
            if keep is None or keep(rec):
                selected.append(rec)
        page_no += 1
        if page_no % 100 == 0:
            logging.info("Historical archive [sequential]: %d pages scanned, "
                         "%d markets kept so far", page_no, walk_kept)
        cursor = data.get("cursor")
        # A None or empty cursor signals the last page
        if not cursor:
            break
        # The walk must be bounded somehow — the archive ignores settlement-time
        # filters server-side, so without a stop rule the loop pages the full
        # multi-million-market archive no matter how recent start_date is. It is
        # ordered by created_time, though, so no page proves deeper pages are
        # unproductive; bail out on sustained unproductiveness instead. Judged
        # on ANY in-window settlement, independent of the result filter above,
        # so a run of all-voided pages can't spuriously trip the counter.
        page_in_window = any(
            s is not None and start_ts <= s < cutoff_ts
            for s in (_iso_epoch(m.get("settlement_ts")) for m in page_markets)
        )
        barren = 0 if page_in_window else barren + 1
        if barren >= ARCHIVE_MAX_BARREN_PAGES:
            logging.info(
                "Historical archive [sequential]: %d consecutive pages with no "
                "in-window settlements (ARCHIVE_MAX_BARREN_PAGES) — stopping "
                "pagination",
                barren,
            )
            break
    return selected


def _fetch_and_store_archive_day(
    hist_client: Any,
    day_lo: int,
    hist_kwargs: dict,
    progress: _FetchProgress,
    expect_meta: dict,
) -> int:
    """
    Worker task: fetch one archive created-day slice and persist it.

    Runs entirely inside the pool thread and returns only a count, so the
    slice's records are freed as soon as they are written and never cross back
    to the main thread. This also moves gzip compression and JSON serialization
    off the main thread — zlib releases the GIL, so that work genuinely
    overlaps other workers.

    Records are STREAMED into the slice file in chunks rather than collected
    and saved at the end: a 2026-08 UTC day holds millions of markets, and one
    full day per worker is what drove RSS to 7.89 GB on a 16 GB host. The
    writer publishes the file only on commit(), so an interrupted worker leaves
    no partial slice for a later run to trust.

    Args:
        hist_client (Any): Authenticated KalshiClient.
        day_lo (int): UTC-midnight lower bound of the created-day to fetch.
        hist_kwargs (dict): Base query params (limit, optional mve_filter).
        progress (_FetchProgress): Shared page counter for log output.
        expect_meta (dict): Reuse-gating keys to stamp into the slice file.

    Returns:
        int: Number of records persisted for this day.

    Raises:
        _ShardedFetchUnsupported: Propagated from _fetch_archive_day when the
            server ignores a synthesized cursor or omits created_time.
    """
    # Persisted incrementally as the day is fetched, so an interrupted run
    # resumes at day granularity instead of refetching, and nothing is retained
    # for the caller — assembly re-reads the file.
    with _DayStreamWriter(
        _day_store_path("archive_days", day_lo),
        _day_store_meta(day_lo, expect_meta),
    ) as writer:
        count = _fetch_archive_day(
            hist_client, day_lo, day_lo + _DAY_SECONDS, hist_kwargs, progress,
            emit=writer.write_records,
        )
        writer.commit()
    return int(count)


def _fetch_archive_phase(
    hist_client: Any,
    start_ts: int,
    cutoff_ts: int,
    hist_kwargs: dict,
    keep: Callable[[dict], bool] | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Fetch the archive's contribution: created-day slices plus the below-start tail.

    Sharded path (preferred): verify cursor synthesis, then fetch one slice
    per UTC created-day in [start_ts, cutoff_ts) — reusing any slice already
    on disk from a previous run — with SETTLED_FETCH_MAX_WORKERS parallel
    workers, then walk the tail below start_ts. Each worker persists its own
    slice as soon as the day completes, so an interrupted fetch resumes at day
    granularity instead of restarting the multi-hour walk.

    Slice records are never accumulated in memory: a worker streams its records
    into its slice file in chunks and returns only a count, and the final list
    is streamed back off disk by _assemble_day_slices (see there for why — a
    ~900-day window otherwise runs to tens of GB, and a single 2026-08 day is
    itself millions of records).

    Slice files are stamped with the cutoff_ts they were fetched under and are
    ONLY reused while the stamp matches the current cutoff: when Kalshi
    advances the cutoff, markets that settled in the gap migrate from the live
    endpoint into the archive (and disappear from the live endpoint —
    verified 2026-07-13), so pre-advance slice files would silently miss them.

    Falls back to _fetch_archive_sequential on any _ShardedFetchUnsupported —
    including one synthesized from an unexpected error in the cursor-synthesis
    probe, so a failure there degrades loudly instead of killing the run. A
    worker failure cancels the remaining queued days rather than draining them.

    Args:
        hist_client (Any): Authenticated KalshiClient.
        start_ts (int): Backtest window start, epoch seconds (UTC midnight).
        cutoff_ts (int): Archive/live boundary from /historical/cutoff.
        hist_kwargs (dict): Base query params (limit, optional mve_filter).
        keep (Callable[[dict], bool] | None): Optional per-record predicate
            applied while slices are read back, and per record by the
            sequential fallback as its pages arrive, so records the caller
            will discard anyway never accumulate. Slice FILES stay unfiltered.
            The tail walk does NOT apply it (its ARCHIVE_TAIL_MAX_RECORDS cap
            counts unfiltered records); the caller's merge filters the tail.

    Returns:
        tuple[list[dict], list[dict]]: (day-slice records newest-day first,
            tail records). Day-slice records are NOT yet settlement-filtered
            (the caller applies the [start_ts, cutoff_ts) window); tail
            records already are. On the sequential fallback, everything is
            returned settlement-filtered (and `keep`-filtered) in the first
            list and the second is empty.
    """
    if start_ts >= cutoff_ts:
        # The archive holds only markets that settled BEFORE the cutoff, so a
        # window starting at or after it cannot contain a single archive
        # record — there is nothing here to fetch, whatever the walk would do.
        # Without this the phase still ran the cursor-synthesis probe and the
        # tail walk to prove that, costing ~18 seconds and ~50,000 parsed
        # records on every post-cutoff run (TS-25). Logged rather than silent
        # so the absence of the usual archive progress lines is explained
        # rather than read as a phase that failed; note that this also skips
        # the synthesis probe, so a run with no archive contribution no longer
        # reports on cursor synthesis at all.
        logging.info(
            "Historical archive phase skipped: the window starts at %s, at or "
            "after the archive cutoff %s, so no archive record can satisfy it",
            datetime.fromtimestamp(start_ts, tz=UTC).isoformat(timespec="seconds"),
            datetime.fromtimestamp(cutoff_ts, tz=UTC).isoformat(timespec="seconds"),
        )
        return [], []

    try:
        # The probe issues a real request, so it can fail for reasons that have
        # nothing to do with cursor format (auth, outage, a body that won't
        # parse). Any such failure is funnelled into the same fail-closed
        # fallback rather than killing the run: an unexpected exception here
        # used to escape the whole phase with no warning logged at all.
        try:
            synthesis_ok = _archive_cursor_synthesis_ok(hist_client, hist_kwargs)
        except _ShardedFetchUnsupported:
            raise
        except Exception as exc:
            raise _ShardedFetchUnsupported(
                f"cursor-synthesis probe failed: {type(exc).__name__}: {exc}"
            ) from exc
        if not synthesis_ok:
            raise _ShardedFetchUnsupported("archive cursor re-encoding mismatch")

        # One slice per UTC day of created_time. All archive records satisfy
        # created < settle < cutoff, so days at/above the cutoff can't exist.
        day_los = list(range(start_ts, cutoff_ts, _DAY_SECONDS))
        expect_meta = {
            "kind": "archive_created_day",
            "cutoff_ts": cutoff_ts,
            "include_mve": INCLUDE_MVE_MARKETS,
            "complete": True,
        }
        # Only day identities are tracked here, never their records: the
        # prescan's loaded slices are discarded and re-read at assembly. That
        # costs one extra decode of the reused days but keeps peak memory
        # independent of how many days the window spans. _discard_all makes
        # that literal — a streamed slice is fully parsed (so corruption is
        # still caught) but nothing is retained from it.
        on_disk: list[int] = []
        to_fetch: list[int] = []
        for lo in day_los:
            if _day_store_load(_day_store_path("archive_days", lo), expect_meta,
                               _discard_all) is not None:
                on_disk.append(lo)
            else:
                to_fetch.append(lo)
        reused = len(on_disk)
        logging.info("Archive day slices: %d reused from disk, %d to fetch",
                     reused, len(to_fetch))

        progress = _FetchProgress("Historical archive [sharded]")
        if to_fetch:
            # Newest days first so the biggest slices (recent volume is far
            # higher) start immediately and the pool drains evenly.
            to_fetch.sort(reverse=True)
            with ThreadPoolExecutor(max_workers=SETTLED_FETCH_MAX_WORKERS) as pool:
                futures = {
                    pool.submit(_fetch_and_store_archive_day, hist_client, lo,
                                hist_kwargs, progress, expect_meta): lo
                    for lo in to_fetch
                }
                started = time.monotonic()
                try:
                    for future in as_completed(futures):
                        lo = futures[future]
                        # The worker already persisted this slice and returns
                        # only its record count, so nothing large is retained.
                        count = future.result()
                        on_disk.append(lo)
                        _log_slice_progress("Archive day slices",
                                            len(on_disk) - reused, len(to_fetch),
                                            lo, count, started)
                except BaseException:
                    # Without this, the executor's __exit__ waits for every
                    # still-queued day (hundreds of them, hours of work) before
                    # the fallback below is even reached.
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise

        # Long-lived markets created before start_date but settling inside the
        # window — same records the sequential walk picked up past start_ts.
        # Its own progress object, labeled "[tail]": these pages are a serial
        # walk, not day-slice pages, and sharing the "[sharded]" counter made a
        # slow tail read in the log as a slow (parallel) slice pool — exactly
        # the label ambiguity CLAUDE.md calls load-bearing for diagnosis.
        tail_progress = _FetchProgress("Historical archive [tail]")
        tail = _fetch_archive_tail(hist_client, start_ts, cutoff_ts, hist_kwargs,
                                   tail_progress)

        return _assemble_day_slices("archive_days", on_disk, expect_meta, keep), tail
    except _ShardedFetchUnsupported as exc:
        logging.warning(
            "Archive fetch: sharded path unavailable (%s) — falling back to the "
            "sequential walk. Any day slices already completed remain on disk "
            "and will be reused by the next run.", exc,
        )
        # Same prefilter as the slice read-back, applied per record as each
        # page arrives; the walk still holds its whole keep-passing result.
        return _fetch_archive_sequential(hist_client, start_ts, cutoff_ts,
                                         hist_kwargs, keep), []


# ─── Live (post-cutoff) fetching ──────────────────────────────────────────────

def _fetch_live_window(
    live_client,
    win_lo: int,
    win_hi: int | None,
    progress: _FetchProgress,
    emit: Callable[[list[dict]], None] | None = None,
) -> list[dict] | int:
    """
    Fetch settled binary markets from the live endpoint for one settled-time window.

    Uses the documented min_settled_ts/max_settled_ts params (both honored
    server-side — verified 2026-07-13), so unlike the archive this needs no
    cursor synthesis. Results are sorted newest-settled first by the server.

    Args:
        live_client: KalshiClient from build_prod_live_client().
        win_lo (int): Window lower bound (min_settled_ts), epoch seconds.
        win_hi (int | None): Window upper bound (max_settled_ts), or None for
            the open-ended frontier window that runs to "now".
        progress (_FetchProgress): Shared page counter for log output.
        emit (Callable[[list[dict]], None] | None): Optional sink for chunked
            output — see _fetch_archive_day for the full contract. When given,
            batches of at most SETTLED_FETCH_CHUNK_RECORDS are handed over and
            the buffer is dropped, and the return value is the record COUNT.
            Both production callers pass one: the past-day worker streams into
            its slice file, and the frontier window streams through a sink that
            keeps only records passing the caller's predicate
            (_fetch_live_phase / _extend_kept), so an unfiltered frontier
            record is resident only while its batch is in flight. None returns
            the accumulated list — the original behavior, retained for direct
            callers; the sequential fallback (_fetch_live_sequential) is a
            separate walk that never calls this function.

    Returns:
        list[dict] | int: Compact market dicts with a binary result and a
            settlement_ts (no settlement-window filtering beyond the server's;
            the caller applies the backtest window) — or, when `emit` is given,
            the number of records emitted.

    Raises:
        _ShardedFetchUnsupported: If ANY record on the first page settles
            more than an hour above win_hi, i.e. the server stopped honoring
            max_settled_ts. The check runs for every record while page 1 is
            being processed (first_page only flips to False once that whole
            page is done), not just the page's first record — but since
            results arrive newest-settled-first, the first record is usually
            the one that actually trips it.
    """
    kept: list[dict] = []
    total = 0
    cursor = None
    first_page = True
    while True:
        kwargs: dict = {"status": "settled", "limit": 1000, "min_settled_ts": win_lo}
        if win_hi is not None:
            kwargs["max_settled_ts"] = win_hi
        if not INCLUDE_MVE_MARKETS:
            kwargs["mve_filter"] = "exclude"
        if cursor:
            kwargs["cursor"] = cursor
        # Raw-response call: the modeled get_markets can no longer deserialize
        # live payloads (see module Notes); retry semantics are unchanged
        data = api_call_with_retry(
            fetch_json_page, live_client.get_markets_without_preload_content, **kwargs
        )
        page = data.get("markets") or []
        page_kept = 0
        for m in page:
            settle = _iso_epoch(m.get("settlement_ts"))
            if settle is None or m.get("result") not in ("yes", "no"):
                continue
            if first_page and win_hi is not None and settle > win_hi + 3600:
                # This fires for ANY record on page 1 that's an hour past the
                # ceiling, not only page[0] (first_page stays True for the
                # whole page, not just its first record) — but results are
                # newest-first, so page[0] is usually the one that trips it.
                # Either way it means max_settled_ts is being ignored and
                # every window would re-walk the whole range.
                raise _ShardedFetchUnsupported("live endpoint ignored max_settled_ts")
            kept.append(_market_to_dict(m))
            page_kept += 1
        first_page = False
        total += page_kept
        # Unchanged accounting: one tick per page, carrying that page's keeps.
        progress.tick(page_kept)
        if emit is not None and len(kept) >= SETTLED_FETCH_CHUNK_RECORDS:
            # Bounded buffer — see _fetch_archive_day for the rationale.
            emit(kept)
            kept = []
        cursor = data.get("cursor")
        if not cursor or not page:
            break
    if emit is None:
        return kept
    if kept:
        emit(kept)
    return total


def _fetch_live_sequential(
    live_client,
    live_min_ts: int,
    keep: Callable[[dict], bool] | None = None,
) -> list[dict]:
    """
    Original single-sweep live fetch — the windowing fallback path.

    One serial cursor walk over [live_min_ts, now), exactly the pre-sharding
    behavior. Used only when _fetch_live_window detects that max_settled_ts is
    no longer honored server-side.

    Residency: this walk has no chunked emit sink, so its whole result — every
    settlement from live_min_ts to now, i.e. every past day of the window plus
    the partial current one — is held in memory. `keep` is applied per record
    as each page arrives, so a record the caller would discard is never
    retained: the same records in the same order as filtering the unfiltered
    result afterwards (fetch_all_settled_markets' merge re-applies the same
    predicate before its first-wins dedup, so dropping them here changes
    nothing downstream). What stays resident is still the WHOLE keep-passing
    result, which is not bounded by this.

    Args:
        live_client: KalshiClient from build_prod_live_client().
        live_min_ts (int): Server-side window start (min_settled_ts).
        keep (Callable[[dict], bool] | None): Optional per-record predicate
            (the caller's prefilter); None keeps every binary record, exactly
            as before it existed. It never affects the walk itself — pages
            requested and the progress line's "markets kept so far" count are
            computed before it.

    Returns:
        list[dict]: Compact market dicts with a binary result and settlement_ts
            that pass `keep`, in walk order.
    """
    kept: list[dict] = []
    # Records that passed the walk's OWN filters (binary result, settlement_ts
    # present), before `keep`: this is what the progress line has always
    # reported, so the line stays identical whether or not a predicate is
    # passed.
    walk_kept = 0
    cursor = None
    page_no = 0
    while True:
        kwargs: dict = {"status": "settled", "limit": 1000, "min_settled_ts": live_min_ts}
        if not INCLUDE_MVE_MARKETS:
            kwargs["mve_filter"] = "exclude"
        if cursor:
            kwargs["cursor"] = cursor
        # Raw-response call: the modeled get_markets can no longer deserialize
        # live payloads (see module Notes); retry semantics are unchanged
        data = api_call_with_retry(
            fetch_json_page, live_client.get_markets_without_preload_content, **kwargs
        )
        for m in data.get("markets") or []:
            settle = _iso_epoch(m.get("settlement_ts"))
            if settle is None or m.get("result") not in ("yes", "no"):
                continue
            walk_kept += 1
            rec = _market_to_dict(m)
            # The caller's prefilter, applied as the page arrives so a record
            # it rejects is never retained (the merge would drop it anyway).
            if keep is None or keep(rec):
                kept.append(rec)
        page_no += 1
        if page_no % 100 == 0:
            logging.info("Live settled sweep [sequential]: %d pages scanned, "
                         "%d markets kept so far", page_no, walk_kept)
        cursor = data.get("cursor")
        if not cursor:
            return kept


def _fetch_and_store_live_window(
    live_client,
    day_lo: int,
    progress: _FetchProgress,
    expect_meta: dict,
) -> int:
    """
    Worker task: fetch one fully-elapsed live settled-day window and persist it.

    The archive-side counterpart is _fetch_and_store_archive_day; same
    rationale (records streamed out in chunks, freed in-thread, serialization
    off the main thread). Only past days go through here — the frontier day is
    fetched directly and deliberately never persisted, since it was captured
    mid-day. It is still streamed in chunks, but into an in-memory list that
    keeps only the records passing the caller's predicate (see
    _fetch_live_phase): with a predicate — the backtester always passes its
    eligibility prefilter — what stays resident is the keep-passing subset of
    the partial day plus one batch in flight; with none, every record passes
    and the whole partial day is held, as it always was.

    Args:
        live_client: KalshiClient from build_prod_live_client().
        day_lo (int): UTC-midnight lower bound of the settled-day window.
        progress (_FetchProgress): Shared page counter for log output.
        expect_meta (dict): Reuse-gating keys to stamp into the slice file.

    Returns:
        int: Number of records persisted for this day.

    Raises:
        _ShardedFetchUnsupported: Propagated from _fetch_live_window when the
            server stops honoring max_settled_ts.
    """
    # Fully-elapsed days are immutable, so persisting here lets later runs and
    # interrupted-run resumes skip the day entirely. Streamed in chunks; the
    # file becomes visible only once the whole window completed.
    with _DayStreamWriter(
        _day_store_path("live_days", day_lo),
        _day_store_meta(day_lo, expect_meta),
    ) as writer:
        count = _fetch_live_window(
            live_client, day_lo, day_lo + _DAY_SECONDS, progress,
            emit=writer.write_records,
        )
        writer.commit()
    return int(count)


def _extend_kept(
    dest: list[dict],
    keep: Callable[[dict], bool] | None,
    batch: list[dict],
) -> None:
    """
    Emit-sink body: append the records of one batch that pass `keep` to `dest`.

    Bound with functools.partial into the `emit` callback of the frontier
    window's _fetch_live_window call (see _fetch_live_phase), so the caller's
    predicate — the backtester's eligibility prefilter — runs on each batch as
    its pages arrive instead of once over the whole accumulated partial day.
    A rejected record is never appended to `dest`; it is resident only while
    its batch (at most SETTLED_FETCH_CHUNK_RECORDS plus one API page) is in
    flight. The frontier used to be accumulated unfiltered and filtered only
    after the whole pool drained, which alone peaked a 7-day backtest at
    ~5.5 GiB RSS (2026-09-24, 1.9M frontier records at 07:30 UTC) and grows
    through the UTC day, for any window length (SS-1).

    Order is preserved exactly — records are appended in batch order and the
    batches arrive in fetch order from one worker thread — so `dest` ends up
    equal, element for element, to filtering the full list afterwards: same
    predicate, same records, same order. Only the peak changes — and only by
    the REJECTED records: every keep-passing record is still retained in
    `dest` until the phase returns, which on the day after a Monday can be
    most of the partial day (see _fetch_live_phase).

    Args:
        dest (list[dict]): List to extend in place. Touched only from the one
            worker thread running the window; the caller reads it only after
            that window's future has completed (Future.result() is the
            synchronization point).
        keep (Callable[[dict], bool] | None): Per-record predicate, or None to
            keep every record. Runs on the worker thread, so it must be
            thread-safe; the backtester's _can_ever_enter is pure.
        batch (list[dict]): One chunk handed over under _fetch_live_window's
            emit contract. Not retained — the window drops it once this
            returns.
    """
    if keep is None:
        dest.extend(batch)
    else:
        dest.extend(m for m in batch if keep(m))


def _fetch_live_phase(
    live_client,
    live_min_ts: int,
    now_ts: int,
    keep: Callable[[dict], bool] | None = None,
) -> list[dict]:
    """
    Fetch the live endpoint's contribution: per-settled-day windows in parallel.

    Splits [live_min_ts, now) into UTC settled-day windows plus an open-ended
    frontier window for the current day. Fully-elapsed days are immutable
    (settlement timestamps never change), so completed day windows are
    persisted to backtest_cache/live_days/ and reused by later runs — only
    the frontier day and any never-fetched days hit the API. The frontier
    window is deliberately never persisted: it was fetched mid-day and would
    otherwise be reused as if complete.

    As on the archive side, past-day records are streamed into the slice file
    in chunks inside the worker and read back off disk at assembly rather than
    accumulated in memory. The frontier window, which is never persisted, is
    streamed in chunks too, through a sink (_extend_kept) that applies `keep`
    to each batch as its pages arrive: only the keep-passing subset of the
    partial day is ever retained, plus one batch in flight. It used to be
    accumulated unfiltered and filtered only once the whole pool had drained —
    the same records in the same order, at a peak that grows through the UTC
    day (up to a full day's settlements, 9.2M records on 2026-09-22) whatever
    the window length (SS-1). With keep=None every record passes and the whole
    partial day is still held, exactly as before.

    What that does NOT bound: the keep-passing subset is itself still held
    whole, as a list, and how large it is depends on the weekday, not on the
    window's length. The backtester's predicate (_can_ever_enter) admits every
    market that was open over a Monday checkpoint on/after start_date and
    closes at least a day later, so a frontier captured the day AFTER a Monday
    can be mostly eligible — 7,190,452 of the 9,176,306 records settled on
    Tuesday 2026-09-22 passed _can_ever_enter(m, 2026-09-17). Bounding that
    requires not holding the frontier as a list at all, which this phase
    cannot do while it returns one materialized list (the past days are
    materialized by _assemble_day_slices too).

    Falls back to _fetch_live_sequential if the server stops honoring
    max_settled_ts (detected per window by _fetch_live_window), passing the
    same `keep`; the partial frontier collected so far is released first,
    since the sequential sweep refetches today anyway.

    Args:
        live_client: KalshiClient from build_prod_live_client().
        live_min_ts (int): Server-side window start — max(cutoff_ts, start_ts).
            Bug fixed 2026-07: this used to always be cutoff_ts, which forced
            the server to walk the entire [cutoff_ts, now) range for a narrow
            recent start_date (observed 20k+ pages discarded client-side).
        now_ts (int): Current epoch seconds; determines the frontier day.
        keep (Callable[[dict], bool] | None): Optional per-record predicate
            applied while past-day slices are read back (main thread), to
            each frontier batch as its pages arrive (on the frontier's worker
            thread, so it must be thread-safe), and per record by the
            sequential fallback (main thread). Slice FILES stay unfiltered.

    Returns:
        list[dict]: Compact market dicts, frontier first then past days
            newest-first (no settlement filtering beyond min_settled_ts; the
            caller applies the backtest window).
    """
    first_lo = live_min_ts - (live_min_ts % _DAY_SECONDS)
    today_lo = now_ts - (now_ts % _DAY_SECONDS)
    frontier_lo = max(today_lo, first_lo)
    past_day_los = list(range(first_lo, frontier_lo, _DAY_SECONDS))

    expect_meta = {
        "kind": "live_settled_day",
        "include_mve": INCLUDE_MVE_MARKETS,
        "complete": True,
    }
    # Filled ONLY by the frontier worker's sink below, and read by this thread
    # only after frontier_future.result() has returned. Bound before the `try`
    # so the fallback branch can always release it (see there).
    frontier: list[dict] = []
    try:
        # As in _fetch_archive_phase: track day identities only, and re-read
        # the slices at assembly so peak memory doesn't scale with the number
        # of days in the sweep.
        on_disk: list[int] = []
        to_fetch: list[int] = []
        for lo in past_day_los:
            if _day_store_load(_day_store_path("live_days", lo), expect_meta,
                               _discard_all) is not None:
                on_disk.append(lo)
            else:
                to_fetch.append(lo)
        reused = len(on_disk)
        logging.info("Live settled-day windows: %d reused from disk, %d to fetch "
                     "(plus the frontier day)", reused, len(to_fetch))

        progress = _FetchProgress("Live settled sweep [windowed]")
        to_fetch.sort(reverse=True)
        with ThreadPoolExecutor(max_workers=SETTLED_FETCH_MAX_WORKERS) as pool:
            # The frontier day is kept in memory rather than persisted — it was
            # captured mid-day and must never be reused as a complete day. It
            # streams through the emit contract like a past day, but into a
            # sink that keeps only `keep`-passing records as each batch lands,
            # so with a predicate the unfiltered partial day is never resident
            # (SS-1).
            frontier_future = pool.submit(
                _fetch_live_window, live_client, frontier_lo, None, progress,
                partial(_extend_kept, frontier, keep),
            )
            futures = {
                pool.submit(_fetch_and_store_live_window, live_client, lo,
                            progress, expect_meta): lo
                for lo in to_fetch
            }
            started = time.monotonic()
            try:
                for future in as_completed(futures):
                    lo = futures[future]
                    # Persisted in the worker; only the day identity is kept.
                    count = future.result()
                    on_disk.append(lo)
                    _log_slice_progress("Live settled-day windows",
                                        len(on_disk) - reused, len(to_fetch),
                                        lo, count, started)
                # LOAD-BEARING, not a leftover: this is the ONLY place a
                # frontier-window failure surfaces — an ApiException that
                # outlived its retries, a non-transient error, or `keep`
                # raising on the worker thread. The pool's __exit__ waits for
                # the worker whether or not this runs, so deleting it would not
                # hang; it would silently return the batches already appended
                # as if they were the whole frontier (a short corpus, pinned by
                # TestFrontierStreamsThroughKeep's failure tests). The records
                # themselves are already in `frontier`, filtered, so the
                # returned count is deliberately discarded (never rebind it).
                frontier_future.result()
            except BaseException:
                # Abandon queued windows immediately rather than draining them
                # on the way out to the sequential fallback.
                pool.shutdown(wait=False, cancel_futures=True)
                raise

        # No post-hoc `keep` pass: the frontier sink already applied it in
        # fetch order, so `frontier` is exactly the list that pass produced.
        return frontier + _assemble_day_slices("live_days", on_disk, expect_meta, keep)
    except _ShardedFetchUnsupported as exc:
        logging.warning(
            "Live fetch: windowed path unavailable (%s) — falling back to the "
            "sequential sweep. Any settled-day windows already completed remain "
            "on disk and will be reused by the next run.", exc,
        )
        # The sequential sweep refetches today as well, so the frontier the
        # windowed path collected is dead weight here. Nothing writes it any
        # more — the pool's __exit__ joined the frontier worker (or its
        # cancel_futures dropped it while still queued) — and a running worker
        # finishes its whole window before that join returns, so without this
        # the keep-passing frontier would sit beside the fallback's own copy of
        # the same day for the whole serial walk. Cleared in place rather than
        # rebound, so the records are released even while another reference
        # to the list survives (the sink's partial stays reachable from
        # frontier_future when the frontier window itself failed).
        frontier.clear()
        # Same prefilter as the windowed path, applied per record as each page
        # arrives; the walk still holds its whole keep-passing result.
        return _fetch_live_sequential(live_client, live_min_ts, keep)


def fetch_all_settled_markets(
    hist_client: Any,
    live_client,
    start_date: date,
    use_cache: bool = True,
    prefilter: Callable[[dict], bool] | None = None,
    prefilter_tag: str | None = None,
) -> list[dict]:
    """
    Fetch all settled Kalshi markets from start_date onward and return them as plain dicts.

    Uses two complementary API endpoints to get full coverage:
    - /historical/markets — settled markets archived before the API cutoff
      timestamp. Fetched as parallel per-created-day slices via synthesized
      pagination cursors (see _fetch_archive_phase), plus a tail walk below
      start_date for long-lived markets; falls back to the original
      sequential walk if the cursor format ever drifts.
    - /markets?status=settled — markets that settled after the API cutoff.
      Fetched as parallel per-settled-day windows bounded server-side by
      min/max_settled_ts (see _fetch_live_phase), starting at
      max(cutoff_ts, start_ts) so a narrow recent start_date doesn't force a
      walk of the entire cutoff→now range.

    Both phases persist each completed day slice to disk
    (backtest_cache/archive_days/, backtest_cache/live_days/), so an
    interrupted fetch resumes at day granularity and later runs only fetch
    days not already covered. Day slices are reused regardless of use_cache
    because they cannot go stale: archive slices are only reused while their
    recorded cutoff matches the current one (a cutoff advance migrates
    markets between endpoints and invalidates them), and live slices cover
    fully-elapsed UTC days whose settlements are immutable — the current
    (frontier) day is always refetched.

    The assembled result is serialized to a JSON cache file named
    settled_markets_<start_date>[_<prefilter_tag>][_nomve].json, so subsequent
    backtests skip fetching entirely. Every component of that name is part of
    the result's identity: the prefilter tag because a filtered result is a
    strict subset, and the _nomve marker because INCLUDE_MVE_MARKETS changes
    which markets are fetched at all (DR-57). Only the False case is marked —
    every assembled cache already on disk was built with MVE included, so the
    default (True) filename is unchanged and no existing cache is orphaned.
    Pass use_cache=False (--no-cache) to rebuild it — thanks to the day stores
    that now only costs the frontier day plus any newly-appeared days. A
    fresh result is always written back regardless of use_cache, so a
    --no-cache run refreshes what the next default run will load.

    Args:
        hist_client (Any): Authenticated KalshiClient from build_historical_client().
        live_client: KalshiClient from build_prod_live_client() for recent settlements.
        start_date (date): Earliest settlement date to include. Markets that settled
            before this date are skipped even if the API returns them.
        use_cache (bool): If True (default), load the assembled per-start_date
            cache if available and skip fetching entirely. If False, always
            re-assemble from the API + day stores. Either way the result is
            saved to disk.
        prefilter (Callable[[dict], bool] | None): Optional per-record
            predicate; records failing it are dropped during assembly and
            never reach the returned list or the assembled cache. Intended for
            a filter the caller would apply immediately anyway (the backtester
            passes _can_ever_enter), which makes it result-neutral while
            keeping peak memory and cache size proportional to the markets
            actually usable rather than to everything Kalshi ever settled. The
            per-day slice FILES are never filtered — they are shared across
            start dates and must stay complete.
        prefilter_tag (str | None): Short name for prefilter's semantics; becomes
            part of the assembled cache's filename so a cache built under one
            predicate is never served to a caller expecting another. Required
            when prefilter is given, and forbidden otherwise.

    Raises:
        ValueError: If exactly one of prefilter / prefilter_tag is provided.

    Returns:
        list[dict]: Flat list of market dicts, each with keys: ticker, event_ticker,
            event_title, title, subtitle, result ("yes" | "no"), yes_ask_dollars,
            no_ask_dollars, yes_bid_dollars, open_time, close_time (ISO str),
            settlement_ts (ISO str), status, price_level_structure,
            price_ranges, exchange_index. Only includes markets with a
            non-null settlement_ts and a binary result; tickers are unique.
            See _market_to_dict for the per-key notes (subtitle now falls
            back to yes_sub_title; price_level_structure/price_ranges are
            unread groundwork that older cache records lack entirely).
    """
    if (prefilter is None) != (prefilter_tag is None):
        raise ValueError(
            "prefilter and prefilter_tag must be provided together — the tag "
            "keys the assembled cache to the predicate that produced it"
        )

    # A prefiltered result is a strict subset, so it gets its own cache file;
    # otherwise a filtered cache could be served to an unfiltered caller (or a
    # cache built under different filter semantics silently reused).
    suffix = f"_{prefilter_tag}" if prefilter_tag else ""
    # DR-57: INCLUDE_MVE_MARKETS changes WHAT IS FETCHED (it adds
    # mve_filter="exclude" to the archive query and to every live page), so it
    # is part of the assembled cache's identity exactly as prefilter_tag is —
    # without it a run configured to EXCLUDE MVE is served an MVE-INCLUSIVE
    # assembly (~99.7% combo markets in practice) and reports results over a
    # universe its own config excludes, with nothing abnormal in the output.
    # Only the False case is marked: every assembled cache currently on disk
    # was built with MVE included, so leaving the True case unmarked keeps
    # those filenames valid and avoids triggering a multi-hour refetch. The
    # per-day slice stores need no such marker — their meta already carries
    # include_mve (see _fetch_archive_phase / _fetch_live_phase), so a flip
    # invalidates them on the existing gate; their PATHS are not flag-keyed,
    # so each flip refetches and overwrites them.
    mve_suffix = "" if INCLUDE_MVE_MARKETS else "_nomve"
    cache_path = (CACHE_DIR /
                  f"settled_markets_{start_date.isoformat()}{suffix}{mve_suffix}.json")
    if use_cache:
        cached = _load_json_cache(cache_path)
        if cached is not None:
            logging.info("Loaded %d settled markets from cache", len(cached))
            return cached

    # Convert start_date to a unix timestamp for filtering individual market records
    start_ts = int(datetime(start_date.year, start_date.month, start_date.day,
                            tzinfo=UTC).timestamp())

    # Get the cutoff timestamp that divides historical archive from live endpoint coverage
    cutoff    = _historical_get(hist_client, f"{_API_PREFIX}/historical/cutoff")
    cutoff_ts = int(datetime.fromisoformat(cutoff["market_settled_ts"]).timestamp())

    # A start_date at/after the archive cutoff means every market this window
    # could ever touch is post-cutoff — i.e. live-era. Live-era markets 404 on
    # /historical/markets/{ticker}/candlesticks (see the CLAUDE.md "Backtest
    # windows must start BEFORE the archive cutoff" gotcha), so _find_entry()
    # can never get candles for either leg and the run is structurally 0-trade
    # no matter how many markets this fetch returns. Warn only — never abort,
    # since the fetch can still be useful (e.g. for cache warming) and a false
    # positive here must not block a legitimate run.
    if start_ts >= cutoff_ts:
        logging.warning(
            "start_date (%s) is at or after the archive cutoff (%s) — "
            "post-cutoff markets 404 on the historical candlesticks endpoint, "
            "so this window is structurally 0-trade",
            start_date, datetime.fromtimestamp(cutoff_ts, tz=UTC).date(),
        )

    # Build the historical-endpoint base kwargs; gate the MVE filter on the config flag.
    # When INCLUDE_MVE_MARKETS is True, omitting mve_filter lets MVE markets through;
    # when False, the legacy "exclude" behaviour is preserved.
    # 1000 is this endpoint family's own hard cap (verified live 2026-07-13),
    # not MARKET_PAGE_SIZE (200) — that constant is the /events listing cap
    # used by _load_or_build_event_titles. Different endpoint family, don't
    # "fix" this to MARKET_PAGE_SIZE in a future sweep.
    hist_kwargs: dict = {"limit": 1000}
    if not INCLUDE_MVE_MARKETS:
        hist_kwargs["mve_filter"] = "exclude"

    # ── Historical endpoint ───────────────────────────────────────────────────
    logging.info("Fetching historical settled markets (settled before API cutoff)...")
    # Sharded parallel fetch with day-level disk reuse; sequential on fallback
    day_records, tail_records = _fetch_archive_phase(
        hist_client, start_ts, cutoff_ts, hist_kwargs, prefilter
    )

    # Assembly: settlement-window filter + first-wins ticker dedup. Adjacent
    # day slices, the tail walk, and live windows deliberately overlap at
    # their boundaries so no record can fall in a gap; dedup collapses those
    # overlaps without changing the set (tickers are unique per market).
    selected: dict[str, dict] = {}

    def _merge(records: list[dict], max_settle: int | None) -> None:
        """Merge compact dicts into `selected`, keeping settlements within
        [start_ts, max_settle) (max_settle None = unbounded above)."""
        for m in records:
            settle = _iso_epoch(m.get("settlement_ts"))
            if settle is None or settle < start_ts:
                continue
            if max_settle is not None and settle >= max_settle:
                continue
            # Single point where the caller's prefilter is GUARANTEED, so it
            # covers every source — day slices, the live frontier, BOTH
            # sequential fallbacks (all of which also take the `keep` fast
            # path) and the tail (which does not). Re-checking records the
            # phases already filtered is idempotent and cheap, and because it
            # runs before the first-wins dedup below, a phase dropping a
            # record early can never change which record wins a ticker.
            if prefilter is not None and not prefilter(m):
                continue
            ticker = m.get("ticker")
            if ticker and ticker not in selected:
                selected[ticker] = m

    _merge(day_records, cutoff_ts)
    _merge(tail_records, cutoff_ts)
    archive_count = len(selected)
    logging.info("Historical endpoint: %d markets from %s", archive_count, start_date)

    # ── Live endpoint (recently settled) ─────────────────────────────────────
    # Prune any live_days/ slices left over from a PRIOR cutoff before the
    # sweep below reuses/repopulates the store — those days are now served by
    # the archive and their old live-endpoint slices are dead disk (see
    # _prune_stale_live_days). Cheap relative to the fetch itself; runs every
    # call, not just when the cutoff has actually advanced.
    _prune_stale_live_days(cutoff_ts)
    # min_settled_ts is honored server-side, so the sweep is bounded to
    # [live_min_ts, now) by the API itself; see _fetch_live_phase for why the
    # lower bound is max(cutoff_ts, start_ts) rather than bare cutoff_ts.
    logging.info("Fetching recently settled markets (after API cutoff)...")
    live_min_ts = max(cutoff_ts, start_ts)
    # Windowed parallel fetch with settled-day disk reuse; sequential on fallback
    live_records = _fetch_live_phase(live_client, live_min_ts, int(time.time()), prefilter)
    _merge(live_records, None)
    logging.info("Live endpoint: %d recently settled markets", len(selected) - archive_count)

    # ── Attach event titles ───────────────────────────────────────────────────
    # Collect unique event_tickers and look up their titles in one batch so the
    # backtester can build (event_title + market_title) grouping keys — for
    # binary markets too: the live scanner attaches _event_title to EVERY
    # market regardless of INCLUDE_MVE_MARKETS (scanner._market_from_dict on
    # the binary listing path of fetch_open_events_with_markets), so the
    # same-title key is (event_title, title, subtitle) live in both modes.
    # Resolving only when MVE is on made the flag silently collapse the
    # backtester's key to (title, subtitle) and pair binary markets live
    # keeps apart. The flag now gates only the MVE-listing phase inside
    # _load_or_build_event_titles (no MVE ticker can be wanted when every
    # market fetch passed mve_filter="exclude").
    unique_tickers = {m.get("event_ticker") for m in selected.values()
                      if m.get("event_ticker")}
    logging.info("Resolving event titles for %d unique event_tickers", len(unique_tickers))
    # Resolve event_ticker -> title so grouping keys match the live scanner's
    titles = _load_or_build_event_titles(live_client, unique_tickers, use_cache=use_cache)

    all_markets: list[dict] = list(selected.values())
    if titles:
        # Day-store records were normalized before titles existed, so the
        # event_title field is patched in here rather than at _market_to_dict time.
        for m in all_markets:
            m["event_title"] = titles.get(m.get("event_ticker") or "", "")
    logging.info("Total settled markets from %s: %d", start_date, len(all_markets))

    # Always persist a fresh fetch — use_cache only controls whether reads are
    # allowed to come from disk. Gating the save on use_cache meant --no-cache
    # runs (whose whole point is to refresh stale data) never updated the file
    # the very next default run would load.
    _save_json_cache(cache_path, all_markets)
    return all_markets


# ─── Candlestick fetching ─────────────────────────────────────────────────────

def _candle_close(side: dict) -> float | None:
    """
    Extract the closing price in dollars from one candle side dict.

    The API sends fixed-point DOLLAR strings (e.g. "0.5500"). Older payloads
    named the field close_dollars alongside an integer-cent close; the current
    format sends the dollar string AS close — prefer close_dollars whenever it
    is actually present so both formats parse to dollars, never cents. The
    presence check is explicit (None / empty string = absent) rather than
    truthiness: a valid falsy close_dollars of numeric 0 must be used, not
    silently fall through to the legacy field.

    Args:
        side (dict): A candle's "yes_ask" or "yes_bid" sub-dict.

    Returns:
        float | None: Closing price in dollars. None when close_dollars is
            absent (None/"") AND close is also absent/unparseable, OR when
            close_dollars IS present but fails to parse as a float — that
            case does NOT fall through to close; a present-but-malformed
            preferred field is treated as a parse failure, not as absence.
    """
    for key in ("close_dollars", "close"):
        raw = side.get(key)
        if raw is None or raw == "":
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None
    return None


def _candle_request_windows(open_ts: int, close_ts: int) -> list[tuple[int, int]]:
    """
    Split one candlestick window into requests the endpoint will serve.

    /historical/markets/{ticker}/candlesticks refuses a request spanning more
    than config.CANDLESTICK_MAX_CANDLES_PER_REQUEST candles with HTTP 400
    ("max candlesticks: 5000"). A window no longer than
    CANDLESTICK_MAX_CANDLES_PER_REQUEST - 1 candle periods is returned
    unchanged as ONE request — exactly what fetch_candlesticks always sent —
    and anything longer is cut into consecutive requests of at most that many
    periods, the first starting at open_ts and the last ending at close_ts.
    One period short of the cap because a request whose span is N periods can
    hold N + 1 period ends when the endpoint counts both ends inclusively, so
    N = cap - 1 stays within the cap whichever way it counts.

    Consecutive requests OVERLAP by one candle period (request k + 1 starts
    one period before request k ends). Whether the endpoint treats a window's
    two ends as inclusive or exclusive is not documented, and with that
    overlap every timestamp strictly inside the overall window lies strictly
    inside at least one request — so, under any of the four conventions, the
    requests together return exactly the candles one uncapped request for the
    whole window would, plus repeats of the few candles two requests share.
    fetch_candlesticks drops those repeats (_merge_candle_pages).

    Args:
        open_ts (int): Unix timestamp the window starts at.
        close_ts (int): Unix timestamp the window ends at.

    Returns:
        list[tuple[int, int]]: (start_ts, end_ts) per request, in time order.
            A single element — (open_ts, close_ts) itself — whenever the window
            fits one request, including a degenerate window with
            close_ts <= open_ts, which is passed through unchanged for the
            endpoint to judge, as it always was.
    """
    period_seconds = CANDLESTICK_PERIOD_INTERVAL_MINUTES * 60
    page_seconds = (CANDLESTICK_MAX_CANDLES_PER_REQUEST - 1) * period_seconds
    if close_ts - open_ts <= page_seconds:
        return [(open_ts, close_ts)]
    # Each request advances by one period less than it spans, which is the
    # one-period overlap described above (positive, since the cap is 5000).
    step_seconds = page_seconds - period_seconds
    windows = []
    start = open_ts
    while start + page_seconds < close_ts:
        windows.append((start, start + page_seconds))
        start += step_seconds
    windows.append((start, close_ts))
    return windows


def _merge_candle_pages(candles: list[dict]) -> list[dict]:
    """
    Order the candles of several requests by timestamp and drop repeats.

    The requests _candle_request_windows builds overlap by one candle period,
    so a candle in an overlap can come back from both; the first copy in
    timestamp order is kept (the sort is stable, so that is the earlier
    request's). Sorting is what makes the result safe for
    backtester._candle_at_or_before, which assumes an ascending series and
    stops at the first candle past its target.

    Only called when a window took more than one request: a single response
    is returned exactly as the endpoint ordered it, as it always was.

    Args:
        candles (list[dict]): Parsed candles (each with an int "ts") from every
            request of one window, concatenated in request order.

    Returns:
        list[dict]: The same candle dicts, ascending by "ts", with at most one
            candle per "ts".
    """
    merged: list[dict] = []
    for candle in sorted(candles, key=lambda c: c["ts"]):
        if merged and merged[-1]["ts"] == candle["ts"]:
            continue
        merged.append(candle)
    return merged


def fetch_candlesticks(
    hist_client: Any,
    ticker: str,
    open_ts: int,
    close_ts: int,
    use_cache: bool = True,
    rate_limit_sleep: float = 0.15,
) -> list[dict]:
    """
    Fetch OHLC candlesticks for one market over its active lifetime.

    Returns one candle per CANDLESTICK_PERIOD_INTERVAL_MINUTES (hourly) with
    the YES ask close price and an approximated NO ask close price. The NO ask
    is computed as the complement of the YES bid close (1 − yes_bid_close),
    which is exact in a binary market with no bid-ask spread and a close
    approximation in practice. Hourly (not daily) granularity is required
    because most Kalshi markets are single-game/few-hour windows that don't
    cross a UTC midnight boundary — daily candles return zero bars for them
    (see config.py CANDLESTICK_PERIOD_INTERVAL_MINUTES for the full explanation).

    The endpoint serves at most CANDLESTICK_MAX_CANDLES_PER_REQUEST candles
    per request and refuses a longer one with HTTP 400, which used to reach
    the except branch below and come back as "no candles" — so every market
    whose window was longer than about 208 days of hourly candles silently
    had no price series. A window that fits one request is still fetched in
    exactly one GET with exactly the same parameters; a longer one is fetched
    as consecutive requests overlapping by one candle period
    (_candle_request_windows), each through the same retried read-only GET,
    and merged ascending by timestamp with the overlap's repeats dropped
    (_merge_candle_pages). The window is
    all-or-nothing: if ANY of its requests fails, the whole ticker returns []
    and nothing is cached, exactly like a single failed request — a partial
    series cached as complete would silently drop the missing span on every
    later run.

    Results are cached per ticker in backtest_cache/candlesticks/<ticker>.json,
    tagged with the [open_ts, close_ts] window and period_interval that were
    actually fetched — the whole window as one entry, however many requests it
    took. A cache hit requires the cached window to COVER the
    requested window AND the cached period_interval to match the current
    CANDLESTICK_PERIOD_INTERVAL_MINUTES — open_ts varies between backtest runs
    with different --start-date values, so a cache built for a later start_date
    must not be reused for an earlier one (it would be silently missing the
    earlier candles), and a cache built under a different granularity (e.g. an
    older daily-interval cache) must not be silently reused as if it were
    hourly. Only successful fetches are cached; a fetch failure returns []
    WITHOUT persisting it, so the ticker is retried on the next run (a cached
    empty file would otherwise silence it forever). Pre-existing empty cache
    files are honored only while younger than _EMPTY_CANDLE_TTL_SECONDS.

    Args:
        hist_client (Any): Authenticated KalshiClient from build_historical_client().
        ticker (str): Kalshi market ticker to fetch candlesticks for.
        open_ts (int): Unix timestamp for the start of the candlestick window
            (typically the backtest start date at 00:00 UTC).
        close_ts (int): Unix timestamp for the end of the candlestick window
            (typically the market's close_time + one day buffer).
        use_cache (bool): If True (default), load from disk cache if available
            and save after fetching. If False, always fetch from the API.
        rate_limit_sleep (float): Seconds to sleep after each API call (each
            request of a paged window included) to stay within the Kalshi rate
            limit. Defaults to 0.15 seconds.

    Returns:
        list[dict]: List of candlestick dicts with keys:
            - "ts" (int): Unix timestamp of the candle's end period.
            - "yes_ask_close" (float): YES ask price at close. Range: [0.01, 0.99].
            - "no_ask_close" (float): Approximated NO ask price at close (1 − yes_bid_close).
              Clamped to [0.01, 0.99]. Returns an empty list on API failure,
              including a failure of any one request of a paged window.
    """
    _CANDLES_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _CANDLES_DIR / f"{ticker}.json"
    if use_cache:
        # Guarded loader: absent OR corrupt (a truncated file from an
        # interrupted run) both read back as None, so a damaged cache falls
        # through to a refetch instead of raising out of the worker thread.
        cached = _load_json_cache(cache_path)
        # Legacy cache files are a bare list with no window metadata (or predate
        # the period_interval tag) — we can't confirm what range/granularity
        # they cover, so fall through and refetch. The refetch below re-saves
        # in the current tagged format, migrating it.
        if isinstance(cached, dict) and cached.get("open_ts", None) is not None:
            covers_window = (
                cached["open_ts"] <= open_ts and cached["close_ts"] >= close_ts
                and cached.get("period_interval") == CANDLESTICK_PERIOD_INTERVAL_MINUTES
            )
            candles = cached.get("candles", [])
            if covers_window:
                # Honor the cache if it has real content, or if a recorded empty
                # is still fresh. Otherwise fall through and retry — a stale
                # empty was likely a previous transient failure, not a
                # genuinely empty market.
                if candles:
                    return candles
                age = time.time() - cache_path.stat().st_mtime
                if age < _EMPTY_CANDLE_TTL_SECONDS:
                    return candles

    # One request unless the window is longer than the endpoint serves in one
    # (CANDLESTICK_MAX_CANDLES_PER_REQUEST), in which case it is paged.
    windows = _candle_request_windows(open_ts, close_ts)
    # 1-based number of the request in flight, read only by the failure line
    # below; reset to 0 once every request has returned, so a failure after
    # that (the cache write) is not blamed on the last request.
    request_no = 0
    try:
        candles = []
        raw_count = 0
        dropped = 0
        for request_open, request_close in windows:
            request_no += 1
            # Raw signed GET — the pinned SDK has no historical_api module and
            # its candlestick models predate the current wire format anyway.
            # Read-only, so _historical_get's api_call_with_retry backoff
            # applies per request.
            data = _historical_get(
                hist_client,
                f"{_API_PREFIX}/historical/markets/{ticker}/candlesticks",
                start_ts=request_open,
                end_ts=request_close,
                period_interval=CANDLESTICK_PERIOD_INTERVAL_MINUTES,
            )
            raw_candlesticks = data.get("candlesticks") or []
            raw_count += len(raw_candlesticks)
            for c in raw_candlesticks:
                try:
                    ya = c.get("yes_ask") or {}
                    yb = c.get("yes_bid") or {}
                    # Dollar-string extraction with explicit presence checks —
                    # see _candle_close for why truthiness fallthrough is wrong
                    yes_ask = _candle_close(ya)
                    yes_bid = _candle_close(yb)
                    if yes_ask is None or yes_bid is None:
                        # Counts as a DROP, not a silent skip: _candle_close
                        # signals an unparseable/absent close by returning None
                        # rather than raising, so without this the candle would
                        # bypass the counter below and a thinned series would be
                        # cached with no visible signal at all (BS-23).
                        dropped += 1
                        continue
                    # NO ask ≈ 1 - YES bid (binary market complement); clamp to avoid 0 or 1
                    no_ask  = 1.0 - yes_bid
                    candles.append({
                        "ts": c["end_period_ts"],
                        "yes_ask_close": yes_ask,
                        "no_ask_close": max(0.01, min(0.99, no_ask)),
                    })
                except (ValueError, TypeError, AttributeError, KeyError):
                    dropped += 1
            # Rate limit: sleep briefly after each call to avoid 429 responses
            time.sleep(rate_limit_sleep)
        request_no = 0
        if len(windows) > 1:
            # Paged: put the requests' candles in timestamp order and drop the
            # repeats the one-period overlaps return twice.
            candles = _merge_candle_pages(candles)
        if dropped:
            # The drop happens before the cache write, so a thinned series is
            # otherwise cached as if it were complete with no visible signal.
            logging.warning("%s: dropped %d/%d malformed candles",
                            ticker, dropped, raw_count)
        # Only successful fetches are cached (tagged with the window and
        # granularity just fetched); failures fall through the except branch
        # and return [] without persisting so the next run retries. Saved
        # unconditionally — use_cache only controls whether reads may come
        # from disk, mirroring fetch_all_settled_markets — otherwise a
        # --no-cache run would never actually refresh the file the next
        # default run loads.
        _save_json_cache(cache_path, {
            "open_ts": open_ts, "close_ts": close_ts,
            "period_interval": CANDLESTICK_PERIOD_INTERVAL_MINUTES,
            "candles": candles,
        })
        return candles
    except Exception as e:
        # ONE line, no header dump (TS-02): post-cutoff tickers 404 by design
        # and are deliberately never cached, so this warning is re-paid on
        # every run for every such ticker. The per-run count is summarized
        # once by backtester._fetch_candles_parallel, which sees every ticker.
        # A paged window names the request that failed; a single-request one
        # keeps the exact line it always logged.
        where = (f" (request {request_no} of {len(windows)})"
                 if len(windows) > 1 and request_no else "")
        logging.warning(
            "Candlestick fetch failed for %s: HTTP %s %s%s",
            ticker, getattr(e, "status", "?"), _exception_summary(e), where,
        )
        time.sleep(rate_limit_sleep)
        # Deliberately DO NOT cache — a poisoned empty file would silence this
        # ticker on every subsequent run until manually deleted. That holds for
        # a paged window too: the requests that DID succeed are discarded
        # rather than cached as if they were the whole window.
        return []
