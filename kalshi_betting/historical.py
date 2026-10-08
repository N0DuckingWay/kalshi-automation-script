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
    (or gzipped JSON-lines) files on disk so re-runs do not re-fetch from the
    API. It also loads Kalshi's series categories and holds series_labels, the
    one category/tag filing rule the dashboard and the live filter share.

Dependencies:
    Imports build_client from auth.py; api_call_with_retry and fetch_json_page
    from _http.py; event_series from scanner.py (the one definition of an
    event's series identity, so the event-title fallback's combo test — DR-51 —
    can never disagree with the one-series rule about what a combo is); and
    PROJECT_ROOT plus a dozen-plus tuning constants
    (MARKET_PAGE_SIZE, MVE_TITLE_LOOKUP_MAX_PAGES, SETTLED_FETCH_MAX_WORKERS,
    SETTLED_FETCH_CHUNK_RECORDS, ARCHIVE_MAX_BARREN_PAGES,
    ARCHIVE_FIRST_CREATED_DATE, EVENT_TITLE_FALLBACK_MAX_LOOKUPS, EVENT_TITLE_FALLBACK_MAX_WORKERS,
    EVENT_TITLE_FALLBACK_RATE_LIMIT_SLEEP_SECONDS,
    EVENT_TITLE_LISTING_MAX_BARREN_PAGES, MVE_SERIES_FAMILY_PREFIX,
    CANDLESTICK_PERIOD_INTERVAL_MINUTES,
    CANDLESTICK_MAX_CANDLES_PER_REQUEST, CANDLE_NO_ASK_CEILING,
    RECENT_CANDLES_RATE_LIMIT_SLEEP_SECONDS,
    INCLUDE_MVE_MARKETS, PROD_URL, PRICE_EPSILON) from config.py. Exports
    build_historical_client() and build_prod_live_client(), both called by
    backtest.py (NOT backtester.py, which never builds its own clients); and
    fetch_all_settled_markets(), fetch_candlesticks(), and infer_category(),
    all called by backtester.py (infer_category() by main.py too).
    load_series_categories() is called by backtest.py and main.py,
    series_labels() by dashboard.py and main.py.
    Also exports SettledCorpus — the
    disk-backed, re-iterable corpus fetch_all_settled_markets returns — which
    carries a CorpusProvenance (when, and under which archive cutoff, the
    corpus was assembled or last extended, and — as AssemblyCounts — how many
    records settled in its window and how many of them the prefilter
    rejected; backtester.py carries it to the dashboard header and reports
    the counts on its own prefilter line); and SettledCorpusError, which
    walking a SettledCorpus raises when its file cannot be read.
    And the candle bid rule a sale reads: usable_candle_ask, which
    backtester.py reads the asks that value an open leg through (its
    _usable_ask, in _leg_quotes); candle_sale_bids, which backtester.py
    records its sale bids with; and bid_before, which reads one side's bid at
    one moment the same way, for live selling's checks on the days before a
    sale; and recent_candles, which fetches a held market's last few days of
    candles for those checks, parsed as fetch_candlesticks parses them but
    never read from or written to the candle cache.

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
    records at 2026-08 volumes — in memory). _day_store_iter reads both (and
    _day_store_load is its all-or-nothing list form); see the day-store
    section for the routing rule. The live frontier (current, partial) day is
    never persisted as a slice; it streams through a sink that keeps only the
    caller's prefilter-passing records as each batch arrives and spools them
    to an anonymous temporary file (_fetch_live_phase, _FrontierSpool), so
    neither the partial day nor its prefilter-passing subset — which on the
    day after a Monday can be most of the day — is ever resident. The two
    sequential fallbacks apply the same prefilter per record but still hold
    their whole filtered result. Those three
    fetch-time filters are the only places the prefilter runs before the
    assembly: the day slices are handed back UNFILTERED (M9 of the 2026-09-24
    review), so the assembly's first walk sees every one of their records and
    can count what the prefilter rejects, and the fetch-time filters add the
    records they dropped to the same count (_AssemblyTally, AssemblyCounts).

    Since SS-1 the ASSEMBLED corpus is never held in memory either. The phases
    return lazy views (_DaySliceStream re-reads the day slices on every walk;
    the live phase chains its frontier spool in front of one), the assembly
    walks them twice through one generator that reproduces the old merge
    exactly (_assembled_records), and the second walk streams straight into
    the assembled cache settled_markets_*.jsonl.gz, which is returned as a
    SettledCorpus that streams the file again on every walk. A legacy
    settled_markets_*.json cache is no longer served: it records neither
    when its live part ends nor its archive cutoff, so it cannot be brought
    up to date, and the request is re-assembled in full (whose commit deletes
    it, _retire_legacy_cache). A file that cannot be read once a walk is
    under way raises SettledCorpusError — never a sequential-walk fallback,
    never a short corpus.

    Every backtest runs through the most recent available day. An assembled
    cache records when it was assembled, the archive cutoff it was assembled
    under and the first second of its partial "frontier" day (meta
    assembled_at, archive_cutoff_ts, live_frontier_ts; informational, never
    part of the identity check). A cache whose frontier day is today (UTC)
    is served as it is, with zero network calls, and announced (DR-13). One
    from an earlier day is EXTENDED (_extend_assembled_cache): one
    /historical/cutoff read, then the live endpoint's settled days from its
    frontier day onward (day slices on disk are reused) are spliced in after
    its archive part and before its older live days, in the order a fresh
    assembly would produce, and its partial frontier day is replaced by the
    complete one. A cutoff that moved past the cache's frontier day (its
    newest markets have migrated into the archive), or a cache that records
    no cutoff or assembly time, is re-assembled in full instead (one that
    records no frontier day is extended from the earlier of the day before
    its assembly and the day its newest live record settled on); a failure
    while extending is a WARNING and the cache is served as it was, marked
    stale. A window that
    starts after the cutoff is no longer special: its markets are priced
    from Kalshi's live candlestick endpoint (fetch_candlesticks' 404
    fallback), so the old "structurally 0-trade" WARNING is gone.

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
import math
import tempfile
import threading
import time
import zlib
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from datetime import UTC, date, datetime
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse

from ._http import api_call_with_retry, fetch_json_page
from .auth import build_client
from .config import (
    ARCHIVE_FIRST_CREATED_DATE,
    ARCHIVE_MAX_BARREN_PAGES,
    CANDLE_NO_ASK_CEILING,
    CANDLESTICK_CACHE_FIELDS_VERSION,
    CANDLESTICK_MAX_CANDLES_PER_REQUEST,
    CANDLESTICK_PERIOD_INTERVAL_MINUTES,
    EVENT_TITLE_FALLBACK_MAX_LOOKUPS,
    EVENT_TITLE_FALLBACK_MAX_WORKERS,
    EVENT_TITLE_FALLBACK_RATE_LIMIT_SLEEP_SECONDS,
    EVENT_TITLE_LISTING_MAX_BARREN_PAGES,
    INCLUDE_MVE_MARKETS,
    MARKET_PAGE_SIZE,
    MVE_SERIES_FAMILY_PREFIX,
    MVE_TITLE_LOOKUP_MAX_PAGES,
    PRICE_EPSILON,
    PROD_URL,
    PROJECT_ROOT,
    RECENT_CANDLES_RATE_LIMIT_SLEEP_SECONDS,
    SERIES_CATEGORY_CACHE_MAX_AGE_SECONDS,
    SETTLED_FETCH_CHUNK_RECORDS,
    SETTLED_FETCH_MAX_WORKERS,
)
from .scanner import event_series

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
# key the live scanner uses. The "_v2" file holds GENUINE answers only; the
# legacy file also held "" for every ticker the lookup cap skipped, grew by
# millions of such entries per bulk window, and is migrated into the v2 file
# once, then deleted (DR-51 — see _load_event_title_accumulator).
_EVENT_TITLES_CACHE = CACHE_DIR / "event_titles_v2.json"
_LEGACY_EVENT_TITLES_CACHE = CACHE_DIR / "event_titles.json"

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


def series_ticker(event_ticker: str) -> str:
    """
    The series part of an event ticker (everything before its first hyphen).

    Not scanner.event_series, which collapses the KXMVE* combo family for the
    one-series rule: Kalshi files each literal series under its own category.

    Args:
        event_ticker (str): An event ticker, e.g. "KXNCAAMBGAME-26JAN13WIUEIU".

    Returns:
        str: The series ticker ("KXNCAAMBGAME"), or "" for an empty ticker.
    """
    return (event_ticker or "").split("-", 1)[0]


def series_labels(
    event_ticker: str,
    fallback_category: str,
    series_categories: dict[str, tuple[str, tuple[str, ...]]] | None,
) -> tuple[str, str]:
    """
    Name the Kalshi category and FIRST tag an event's series is filed under.

    The one filing rule behind the backtest dashboard's categories and tags and
    main._filter_by_category's live filter (which states its own matching). First
    tag only, so every breakdown partitions what it breaks down.

    Args:
        event_ticker (str): The event whose series (series_ticker) is looked up.
        fallback_category (str): Category for a series the map lacks (infer_category).
        series_categories (dict | None): load_series_categories' map, or None.

    Returns:
        tuple[str, str]: (category, tag); the tag "General" when the series has none
            or is not in the map, the category "Uncategorised" when Kalshi gives none.
    """
    entry = (series_categories or {}).get(series_ticker(event_ticker))
    if entry is None:
        return fallback_category, "General"
    return entry[0] or "Uncategorised", entry[1][0] if entry[1] else "General"


# Disk copy of Kalshi's /series listing: {"fetched_at": ISO, "series":
# {series_ticker: [category, [tags...]]}}. See load_series_categories.
_SERIES_CATEGORIES_CACHE = CACHE_DIR / "series_categories.json"
# /series served every series in one response when checked (2026-09-25, no
# cursor); should it ever paginate, this bounds the walk.
_SERIES_LIST_MAX_PAGES = 100


def _parse_series_categories(raw: Any) -> dict[str, tuple[str, tuple[str, ...]]]:
    """
    Read a series_categories.json "series" block back into the returned shape.

    Args:
        raw (Any): The cached "series" value; anything but a dict reads as empty.

    Returns:
        dict[str, tuple[str, tuple[str, ...]]]: series ticker -> (category,
            tags), skipping any entry that is not [str, [str, ...]].
    """
    if not isinstance(raw, dict):
        return {}
    out: dict[str, tuple[str, tuple[str, ...]]] = {}
    for ticker, value in raw.items():
        if (isinstance(ticker, str) and isinstance(value, list) and len(value) == 2
                and isinstance(value[0], str) and isinstance(value[1], list)):
            out[ticker] = (value[0], tuple(t for t in value[1] if isinstance(t, str)))
    return out


def load_series_categories(
    client: Any,
    max_age_seconds: int = SERIES_CATEGORY_CACHE_MAX_AGE_SECONDS,
) -> dict[str, tuple[str, tuple[str, ...]]]:
    """
    Map every Kalshi series ticker to its official category and tags.

    Kalshi's /series listing carries, per series, the category the exchange
    files it under ("Sports", "Politics", "Commodities", ... — 20 of them) and
    finer tags ("Basketball", "Congress", "Oil & Gas", ...). The backtest
    dashboard breaks returns down by them, because infer_category's ticker
    prefixes predate the "KX" prefix every current series carries and so file
    nearly every trade under "Other". Nothing is priced, sized or settled on
    these labels; they file the dashboard's trades and decide which pairs a
    live category/tag filter keeps.

    The listing is cached in backtest_cache/series_categories.json and reused
    while younger than max_age_seconds. It is one read-only GET (retried like
    every other historical read). It never raises: a failed fetch logs a WARNING
    and returns the cached copy however old, or {} — on which the dashboard files
    by infer_category and a set live category/tag filter keeps nothing. A fetched
    listing is returned even when writing its cache fails.

    Args:
        client (Any): A KalshiClient pointed at prod (build_prod_live_client() or
            a production run's own), or None to read the cached copy only.
        max_age_seconds (int): Reuse the cached copy while it is younger than
            this. Defaults to config.SERIES_CATEGORY_CACHE_MAX_AGE_SECONDS.
            Ignored when client is None.

    Returns:
        dict[str, tuple[str, tuple[str, ...]]]: series ticker -> (category,
            tags). A series with no category maps to ""; with no tags to ().
    """
    cached = _load_json_cache(_SERIES_CATEGORIES_CACHE)
    cached_map: dict[str, tuple[str, tuple[str, ...]]] = {}
    fetched_at: datetime | None = None
    if isinstance(cached, dict):
        cached_map = _parse_series_categories(cached.get("series"))
        try:
            fetched_at = datetime.fromisoformat(cached.get("fetched_at"))
        except (TypeError, ValueError):
            fetched_at = None
    if client is None:
        # Dev mode: the sandbox key must never sign a production request
        return cached_map
    now = datetime.now(UTC)
    if (cached_map and fetched_at is not None and fetched_at.tzinfo is not None
            and 0 <= (now - fetched_at).total_seconds() < max_age_seconds):
        return cached_map
    try:
        fresh: dict[str, tuple[str, tuple[str, ...]]] = {}
        cursor, seen = None, set()
        for _ in range(_SERIES_LIST_MAX_PAGES):
            params = {"cursor": cursor} if cursor else {}
            data = _historical_get(client, f"{_API_PREFIX}/series", **params)
            if not isinstance(data, dict):
                raise ValueError(f"/series returned a {type(data).__name__}, not an object")
            for s in data.get("series") or []:
                if isinstance(s, dict) and isinstance(s.get("ticker"), str) and s["ticker"]:
                    tags = s.get("tags") if isinstance(s.get("tags"), list) else []
                    fresh[s["ticker"]] = (
                        s.get("category") if isinstance(s.get("category"), str) else "",
                        tuple(t for t in tags if isinstance(t, str)),
                    )
            cursor = data.get("cursor")
            if not cursor or cursor in seen:
                break
            seen.add(cursor)
        if not fresh:
            raise ValueError("/series returned no series")
    except Exception as exc:  # never end a run over it (the live filter fails closed on {})
        logging.warning(
            "Series category listing unavailable (%s) — %s",
            _exception_summary(exc),
            f"using the cached copy of {len(cached_map)} series" if cached_map
            else "every series falls back to ticker-prefix categories (and a live "
                 "category/tag filter, if set, trades nothing)",
        )
        return cached_map
    try:
        _save_json_cache(_SERIES_CATEGORIES_CACHE, {
            "fetched_at": now.isoformat(),
            "series": {t: [cat, list(tags)] for t, (cat, tags) in fresh.items()},
        })
    except Exception as exc:  # the listing is in hand: a failed write costs the cache only
        logging.warning(
            "Series categories: could not cache Kalshi's /series listing (%s) — this run "
            "still files by the %d series it fetched", _exception_summary(exc), len(fresh))
    logging.info("Series categories: %d series from Kalshi's /series listing", len(fresh))
    return fresh


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
    failed candlestick fetch on a post-cutoff window (where every ticker
    404'd on the historical endpoint before fetch_candlesticks learned to ask
    the live one, and those failures are deliberately never cached so they
    are re-paid every run), that alone rotated the run's own diagnostics out of
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
            assembled backtest_cache/settled_markets_* cache (the .jsonl.gz,
            or a legacy .json) first and returns before the day slices are
            consulted at all.
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
        # bypasses the assembled settled_markets_* cache that would otherwise
        # short-circuit the fetch before any slice is read).
        "subtitle": m.get("subtitle") or m.get("yes_sub_title"),
        "result": m.get("result"),
        "yes_ask_dollars": m.get("yes_ask_dollars"),
        "no_ask_dollars": m.get("no_ask_dollars"),
        "yes_bid_dollars": m.get("yes_bid_dollars"),
        # Feeds backtester._can_ever_enter()'s eligibility prefilter — needed
        # to prove no Monday checkpoint of a market's tradeable window falls
        # after its opening instant
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

    The whole file is read into one string and parsed at once, so it is only
    for small caches (candlesticks) — and for LEGACY assembled settled-market
    caches (settled_markets_*.json), which fetch_all_settled_markets still
    serves this way for compatibility. That path materializes the whole
    corpus (TS-07 in CLAUDE.md records what it cost); new assembled caches are
    streamed instead (SettledCorpus). The event-title accumulator does not
    come through here: its reader, _read_title_file, must tell a file that
    could not be READ from one whose content is bad, which this one cannot.

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
    # unique (per-ticker for candlesticks; the one event-title accumulator per run)
    # — the same path-uniqueness invariant that keeps the parallel candlestick
    # fetch safe. Never introduce a fetch whose cache path is shared across
    # workers. (The assembled settled-market cache no longer comes through
    # here: since SS-1 it is streamed by _DayStreamWriter, same idiom.)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, default=str))
    tmp.replace(path)


def _is_combo_event(event_ticker: str) -> bool:
    """
    True when an event ticker belongs to the KXMVE combo (parlay) family.

    Read through scanner.event_series, the one definition of an event's series
    identity, which collapses every KXMVE* prefix onto MVE_SERIES_FAMILY_PREFIX
    (DR-55), so this can never disagree with the one-series rule about what a
    combo is. It decides only which per-ticker title lookups
    _load_or_build_event_titles spends its cap on (DR-51). It never drops or
    filters a market, and the grouping and pairing RULES are unchanged. What
    it does change is one of their INPUTS: a combo ticker the old sorted cap
    happened to reach was looked up and titled, and a deferred one now keeps a
    blank event_title unless the accumulator already holds its title. That
    moves those markets' grouping keys and the backtest's event_title and
    deadline-phrasing census figures, but not which pairs form — see
    config.EVENT_TITLE_FALLBACK_MAX_LOOKUPS for the evidence.

    Args:
        event_ticker (str): An event ticker from the settled corpus.

    Returns:
        bool: True for a combo event ticker; False otherwise, including for an
            unreadable one (event_series reads it as the series "").
    """
    # The one series definition, so "combo" means here what the one-series
    # rule means by it
    return event_series(event_ticker) == MVE_SERIES_FAMILY_PREFIX


# Outcomes of _read_title_file. A failed READ (an OSError: a permission error,
# an iCloud file that cannot be downloaded while offline) may succeed on the
# next run, so nothing may be written over the file; CONTENT that is not a JSON
# object will not heal by itself.
_TITLE_FILE_OK = "ok"
_TITLE_FILE_ABSENT = "absent"
_TITLE_FILE_UNREADABLE = "unreadable"
_TITLE_FILE_CORRUPT = "corrupt"


def _read_title_file(path: Path) -> tuple[dict | None, str, str]:
    """
    Read one event-title accumulator file, telling a failed READ from bad CONTENT.

    _load_json_cache folds every failure into None, which is right for a cache
    that can simply be rebuilt but not for this one: the answer to an
    unusable accumulator is to write a new one, and writing over a file that
    merely could not be read THIS time destroys every title in it — and, for
    the legacy file, writing the v2 file commits the migration marker and
    abandons the legacy titles for good (DR-51). So the two cases are kept
    apart. Nothing is logged here and nothing is raised; the caller logs one
    line per case.

    Args:
        path (Path): The accumulator file to read (the v2 file or the legacy one).

    Returns:
        tuple[dict | None, str, str]: (titles, outcome, detail). outcome is
            _TITLE_FILE_OK (titles is the parsed JSON object), _TITLE_FILE_ABSENT,
            _TITLE_FILE_UNREADABLE (an OSError while reading) or
            _TITLE_FILE_CORRUPT (undecodable bytes, invalid JSON, or JSON that
            is not an object); titles is None for all but the first. detail
            names the exception or the JSON type, for the caller's log line.
    """
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None, _TITLE_FILE_ABSENT, ""
    except OSError as exc:
        return None, _TITLE_FILE_UNREADABLE, f"{type(exc).__name__}: {exc}"
    except ValueError as exc:
        # Undecodable bytes: UnicodeDecodeError is a ValueError, not an OSError
        return None, _TITLE_FILE_CORRUPT, type(exc).__name__
    try:
        titles = json.loads(text)
    except ValueError as exc:
        return None, _TITLE_FILE_CORRUPT, type(exc).__name__
    if not isinstance(titles, dict):
        return None, _TITLE_FILE_CORRUPT, f"it holds a JSON {type(titles).__name__}"
    return titles, _TITLE_FILE_OK, ""


@dataclass
class _TitleAccumulator:
    """
    The loaded event-title accumulator and what the caller may write back.

    Attributes:
        titles (dict[str, str]): The accumulator. The caller merges this call's
            answers into it IN PLACE and saves this same object.
        migrated (bool): True only when titles came from a successfully parsed
            legacy file; the caller retires that file once the v2 file holding
            its titles is committed.
        rewrite (bool): True when the caller must write the v2 file even if its
            own answers change nothing: after a migration — one that imported
            the legacy titles or one that had to give up on a legacy file whose
            content is not a JSON object — so the marker exists, or to replace
            a v2 file whose content is not a JSON object.
        persist (bool): False when a file could not be READ (an OSError). The
            caller then writes nothing this call — neither the v2 file nor,
            therefore, the migration marker — so a read that may succeed next
            run can never overwrite or abandon titles it did not see.
    """
    titles: dict[str, str]
    migrated: bool = False
    rewrite: bool = False
    persist: bool = True


def _warn_if_legacy_lingers() -> None:
    """
    WARN when a legacy event_titles.json sits beside the v2 accumulator.

    While event_titles_v2.json exists the legacy file is never read, so one
    that is still there is dead weight: a migration whose delete failed, or a
    run killed between the v2 save and that delete; a legacy file left in
    place because its content could not be parsed; or one an older build (a
    checkout from before DR-51, sharing this backtest_cache/) wrote back, and
    keeps growing by millions of entries per bulk window. It is NOT deleted
    here, because in the last case it is that build's live accumulator; it is
    named, with its size, on every call instead, so it cannot sit unnoticed on
    disk the way the retire-once design otherwise would let it.
    """
    try:
        size = _LEGACY_EVENT_TITLES_CACHE.stat().st_size
    except OSError:
        # Absent — the normal case — or its metadata cannot be read: nothing to name
        return
    logging.warning(
        "Event-title accumulator: a legacy %s (%.1f MB) sits beside %s and is "
        "never read while that file exists (DR-51). It is left over from a "
        "migration whose delete failed or was interrupted, kept because it "
        "could not be parsed, or written back by an older build (a checkout "
        "from before DR-51), which keeps growing it. Delete it unless such a "
        "build still uses it.",
        _LEGACY_EVENT_TITLES_CACHE, size / 1e6, _EVENT_TITLES_CACHE.name,
    )


def _load_event_title_accumulator() -> _TitleAccumulator:
    """
    Load the cross-run event-title accumulator, migrating the legacy file once.

    The accumulator lives in _EVENT_TITLES_CACHE (event_titles_v2.json). Its
    invariant is that every entry is a GENUINE answer: a title, or "" for a
    per-ticker lookup that failed or an event a bulk listing returned without
    a title. The legacy _LEGACY_EVENT_TITLES_CACHE (event_titles.json) broke
    that invariant: it also stored "" for every ticker the per-ticker
    fallback's cap skipped — tickers nobody ever looked up — and so grew by
    millions of entries per bulk window (3,996,906 -> 7,986,570 keys in the
    2026-09-24 7-day run, 7,918,449 of them KXMVE tickers mapped to "").
    A legacy "" cannot be told apart from a genuine one, so the migration keeps
    EXACTLY the legacy file's non-empty string entries and drops every "" once
    (DR-51). A dropped ticker is simply unknown again: it costs a lookup only if
    a later window asks for it, and a combo ticker is looked up only when a
    run's whole unresolved set fits under EVENT_TITLE_FALLBACK_MAX_LOOKUPS.

    The v2 file's EXISTENCE is the migration marker. While it exists the
    legacy file is never read, so the migration runs once, and a legacy file
    that reappears later (an older build writing it again, a file-sync tool
    restoring it)
    is ignored rather than re-imported with its pills — and named on every
    call by _warn_if_legacy_lingers, never deleted.

    Neither file's failure is ever raised, and the two kinds are told apart
    (_read_title_file). A file that cannot be READ (an OSError) makes the
    call write nothing, so it is read again next run: the v2 file is not
    overwritten, and the migration marker is not committed over legacy titles
    nobody saw. A file whose CONTENT is not a JSON object will not heal: a
    damaged v2 file is replaced, as the pre-DR-51 accumulator was; a damaged
    legacy file is left in place, none of its titles are imported, and the v2
    file is written as the marker, so the WARNING names the recovery (repair
    it, delete event_titles_v2.json, re-run).

    Returns:
        _TitleAccumulator: The accumulator (titles, which the caller merges into
            IN PLACE) and what the caller may write back (migrated, rewrite,
            persist — see the dataclass).
    """
    titles, outcome, detail = _read_title_file(_EVENT_TITLES_CACHE)
    if outcome == _TITLE_FILE_OK:
        _warn_if_legacy_lingers()
        return _TitleAccumulator(titles)
    if outcome == _TITLE_FILE_UNREADABLE:
        logging.warning(
            "Event-title accumulator %s could not be read (%s). Titles are "
            "resolved without it this run and nothing is written back, so the "
            "file is not overwritten; the next run reads it again.",
            _EVENT_TITLES_CACHE, detail,
        )
        return _TitleAccumulator({}, persist=False)
    if outcome == _TITLE_FILE_CORRUPT:
        # "Corrupt JSON cache" is the wording every cache in this module uses
        logging.warning(
            "Corrupt JSON cache %s (the event-title accumulator: %s) — treating "
            "it as empty and replacing it",
            _EVENT_TITLES_CACHE, detail,
        )
        return _TitleAccumulator({}, rewrite=True)

    # No v2 file: the first run ever, or the one-time migration. The legacy
    # file is read whole — it can hold millions of entries (374 MB on
    # 2026-09-24) — exactly once; every later run reads only the v2 file.
    legacy, outcome, detail = _read_title_file(_LEGACY_EVENT_TITLES_CACHE)
    if outcome == _TITLE_FILE_ABSENT:
        return _TitleAccumulator({})
    if outcome == _TITLE_FILE_UNREADABLE:
        logging.warning(
            "Event-title accumulator: the legacy %s could not be read (%s), so "
            "its migration to %s is deferred. Titles are resolved without it "
            "this run and nothing is written, so no %s marks the migration done "
            "and the next run retries it (DR-51).",
            _LEGACY_EVENT_TITLES_CACHE, detail, _EVENT_TITLES_CACHE.name,
            _EVENT_TITLES_CACHE.name,
        )
        return _TitleAccumulator({}, persist=False)
    if outcome == _TITLE_FILE_CORRUPT:
        logging.warning(
            "Event-title accumulator: the legacy %s could not be read as a JSON "
            "object (%s), so none of its titles are imported and %s starts "
            "empty. The legacy file is left in place for inspection; it is "
            "never read again once %s exists (DR-51). To retry the migration "
            "after repairing it, delete %s and re-run.",
            _LEGACY_EVENT_TITLES_CACHE, detail, _EVENT_TITLES_CACHE.name,
            _EVENT_TITLES_CACHE.name, _EVENT_TITLES_CACHE,
        )
        return _TitleAccumulator({}, rewrite=True)
    legacy_entries = len(legacy)
    titles = {tkr: title for tkr, title in legacy.items()
              if isinstance(title, str) and title}
    del legacy
    logging.info(
        "Event-title accumulator: migrating %s to %s — keeping its %d titled "
        "entries and dropping the other %d, none of which holds a title. "
        "Before DR-51 the lookup cap stored \"\" for tickers it never looked "
        "up, and those cannot be told apart from genuine failures; a dropped "
        "ticker is re-resolved only if a later window asks for it.",
        _LEGACY_EVENT_TITLES_CACHE.name, _EVENT_TITLES_CACHE.name,
        len(titles), legacy_entries - len(titles),
    )
    return _TitleAccumulator(titles, migrated=True, rewrite=True)


def _retire_legacy_event_titles() -> None:
    """
    Delete the legacy event_titles.json once the migrated v2 file is committed.

    Called only after _load_event_title_accumulator parsed the legacy file and
    the v2 file holding every one of its titled entries was written, so the
    only thing destroyed is the legacy file's "" entries — exactly what the
    DR-51 migration drops. Left on disk it would be dead weight (374 MB on
    2026-09-24) that is never read again while
    the v2 file exists. Same retire-after-commit idiom as _retire_legacy_cache.
    A failure to delete is logged and never raised: the backtest's own result
    does not depend on it, and _warn_if_legacy_lingers names the file on every
    later run until it is gone.
    """
    try:
        _LEGACY_EVENT_TITLES_CACHE.unlink()
    except FileNotFoundError:
        # A concurrent run migrated first and already removed it
        return
    except OSError as exc:
        logging.warning(
            "Could not remove the legacy event-title accumulator %s (%s). Delete "
            "it by hand: %s now holds every titled entry it had, it is never "
            "read again while that file exists, and every later run warns "
            "about it until it is gone.",
            _LEGACY_EVENT_TITLES_CACHE, exc, _EVENT_TITLES_CACHE.name,
        )
        return
    logging.info(
        "Removed the legacy event-title accumulator %s: %s now holds every "
        "titled entry it had (DR-51).",
        _LEGACY_EVENT_TITLES_CACHE.name, _EVENT_TITLES_CACHE.name,
    )


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
         For most backtests this covers nearly every non-combo event_ticker in
         a few hundred paginated calls.
      2. For any tickers still unresolved (very old archived events that have aged
         out of the bulk listings, and combo events), fall back to per-ticker
         get_event() calls — run in parallel, paced per worker by
         EVENT_TITLE_FALLBACK_RATE_LIMIT_SLEEP_SECONDS, and capped at
         EVENT_TITLE_FALLBACK_MAX_LOOKUPS, since each costs a round trip and
         the miss set runs to millions at 2026-09 Kalshi volumes. When the
         whole miss set fits under the cap every ticker is looked up; when it
         does not, the cap is spent on NON-combo tickers only and every combo
         ticker is deferred, because a combo's event title has no measured
         effect on which pairs form while a non-combo's does (TS-11; see
         config.EVENT_TITLE_FALLBACK_MAX_LOOKUPS for the evidence). The capped
         non-combo slice is taken in ticker order, tickers the accumulator has
         never answered first (then its stored pills, then tickers it already
         titles — only the first group exists with the cache on).

    Every phase logs progress, and one closing line accounts for every ticker
    this CALL was asked about (DR-42): resolved by this run (listings /
    lookups), answered from the accumulator, recorded as unresolvable, or
    deferred. This function can legitimately run for minutes, and when it was
    silent an in-progress run was indistinguishable from a hang (2026-08-03).

    Results persist in the cross-run accumulator (_EVENT_TITLES_CACHE), which
    stores GENUINE answers only (DR-51): a title, or "" — the poison pill — for
    a lookup that failed or an event listed without a title, so a ticker that
    genuinely cannot be resolved is not re-looked-up every run. A ticker the
    cap DEFERRED is not looked up this call and nothing is stored for it: its
    return value is whatever the accumulator already holds for it — "" unless
    an earlier run titled it — and a later run tries it again. For non-combo
    tickers that later run advances past this one's slice either way: with
    the cache on because every ticker this call answered is no longer
    unresolved, and under use_cache=False because tickers the accumulator has
    never answered are looked up first. Storing "" for deferred tickers is
    what grew the legacy event_titles.json by millions of entries per bulk
    window; _load_event_title_accumulator migrates that file once.

    The accumulator is written as a MERGE, so a single run (in particular a
    `--no-cache` run, which resolves from scratch) can never wipe titles other
    runs paid for. Merge rule: disk entries are preserved; a fresh non-empty
    title wins over the disk value; a fresh "" NEVER clobbers a non-empty disk
    title, but a fresh "" for a ticker unknown to disk IS stored (poison-pill
    semantics preserved). The merge is applied to the loaded accumulator IN
    PLACE and this run's answers are kept in their own map: the old code held
    two full copies of the accumulator (seed and merge) beside the parsed file,
    which with the legacy file at 7,986,570 keys put this phase at the top of a
    fresh fetch's peak RSS. The file is rewritten only when this call changed
    it (or migrated or replaced it), and never when an accumulator file could
    not be READ this call (_load_event_title_accumulator), so a read that may
    succeed next run cannot overwrite the titles it did not see.

    An unresolved ticker is not an error: the caller groups those markets by
    market title alone, which is exactly what a failed lookup has always done.

    Args:
        live_client: A KalshiClient with the events API methods available
            (e.g. the client from build_prod_live_client()).
        event_tickers (set[str]): The set of event_ticker values whose titles
            we need. May contain millions of entries. Never mutated.
        use_cache (bool): If True, a ticker the accumulator already answers (a
            title or a stored pill) is not resolved again. If False, every
            requested ticker is genuinely re-resolved, within the lookup cap:
            when the cap binds, the tickers the accumulator has never answered
            take it first and a deferred one keeps whatever it already holds.
            The accumulator is read (for the merge and the return) and updated
            either way — a `--no-cache` run refreshes its own tickers without
            discarding titles it did not ask about.

    Returns:
        dict[str, str]: Mapping event_ticker → event_title, restricted to
            event_tickers, read from the MERGED view — this run's answers
            layered over the accumulator. Tickers that could not be resolved
            anywhere, and deferred tickers the accumulator holds no title for,
            map to "". Caller treats those markets as ungrouped (effectively
            MVE-excluded).

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
    if not event_tickers:
        # Nothing asked, nothing to answer — and no reason to read (or migrate)
        # an accumulator that cannot contribute to an empty result.
        return {}
    # Always read the accumulator: even when use_cache is False and it must not
    # seed resolution, it is needed for the merge and the TS-11 return. A file
    # that cannot be used → {} plus a warning (and, if it could not even be
    # READ, nothing is written back this call); a legacy file is migrated once.
    accumulator = _load_event_title_accumulator()
    disk_titles = accumulator.titles
    accumulator_before = len(disk_titles)
    # This call's answers ONLY — never a copy of the accumulator (DR-51). Every
    # "this run" figure in the closing summary is read off this map (DR-42).
    fresh: dict[str, str] = {}
    by_listing = 0   # titles this run got from the bulk listings
    by_lookup = 0    # titles this run got from per-ticker lookups

    def _unresolved(tkr) -> bool:
        """True while this call still has no answer for tkr."""
        return tkr not in fresh and not (use_cache and tkr in disk_titles)

    # Nothing is copied to track what is missing: `remaining` counts it, and
    # _unresolved answers membership against the caller's set and the two maps.
    remaining = (sum(1 for tkr in event_tickers if tkr not in disk_titles)
                 if use_cache else len(event_tickers))

    # Bulk pull non-MVE events across all statuses. Each get_events call returns
    # up to 200 events; pagination continues until cursor is empty or all misses
    # are resolved (whichever comes first).
    #
    # Raw-response call: the modeled get_events deserializes into EventData,
    # whose `category` field the pinned SDK types as a REQUIRED string. The live
    # API now sends `category: null` on some events (observed 2026-08-03), so
    # the modeled call raises pydantic ValidationError mid-listing. Same drift,
    # and same fix, as the market/order/orderbook endpoints — see module Notes.
    total_missing = remaining
    for status in ("settled", "closed", "open"):
        if not remaining:
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
                if tkr in event_tickers and _unresolved(tkr):
                    title = ev.get("title") or ""
                    fresh[tkr] = title
                    by_listing += bool(title)
                    remaining -= 1
                    resolved_here += 1
            pages += 1
            # Without this the whole phase is silent for however long it runs —
            # which at current volumes is long enough to look like a hang.
            if pages % 100 == 0:
                logging.info("Event titles [%s listing]: %d pages scanned, "
                             "%d/%d still unresolved",
                             status, pages, remaining, total_missing)
            # Productivity bail-out: this listing is a full scan looking for a
            # specific ticker set, so once it stops hitting wanted tickers it
            # will not start again — keep paging and it burns minutes finding
            # nothing (see EVENT_TITLE_LISTING_MAX_BARREN_PAGES).
            barren = 0 if resolved_here else barren + 1
            if barren >= EVENT_TITLE_LISTING_MAX_BARREN_PAGES:
                logging.info(
                    "Event titles [%s listing]: no new titles in %d consecutive "
                    "pages after %d scanned — moving on with %d/%d unresolved",
                    status, barren, pages, remaining, total_missing,
                )
                break
            cursor = data.get("cursor")
            if not cursor or not remaining:
                break

    # Bulk pull multivariate events — these are excluded from get_events by API
    # design. The MVE listing is effectively unbounded (hundreds of thousands of
    # auto-generated collection events), so this loop is capped at
    # MVE_TITLE_LOOKUP_MAX_PAGES; tickers not found by then fall through to the
    # bounded per-ticker lookup below instead of paging for hours.
    # Raw-response for the same nullable-category reason as above.
    # Only worth paging when MVE markets can be in the wanted set at all —
    # with INCLUDE_MVE_MARKETS off every market fetch excluded them upstream.
    if remaining and INCLUDE_MVE_MARKETS:
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
                if tkr in event_tickers and _unresolved(tkr):
                    title = ev.get("title") or ""
                    fresh[tkr] = title
                    by_listing += bool(title)
                    remaining -= 1
                    resolved_here += 1
            if page_no % 100 == 0:
                logging.info("Event titles [MVE listing]: %d/%d pages scanned, "
                             "%d/%d still unresolved", page_no,
                             MVE_TITLE_LOOKUP_MAX_PAGES, remaining, total_missing)
            # Same productivity bail-out as the status listings above; the MVE
            # listing is the most unbounded of the three.
            barren = 0 if resolved_here else barren + 1
            if barren >= EVENT_TITLE_LISTING_MAX_BARREN_PAGES:
                logging.info(
                    "Event titles [MVE listing]: no new titles in %d consecutive "
                    "pages after %d scanned — moving on with %d/%d unresolved",
                    barren, page_no, remaining, total_missing,
                )
                break
            cursor = data.get("cursor")
            if not cursor or not remaining:
                break

    # Per-ticker fallback for events that aren't in the bulk listings (or fell
    # past the MVE page cap). Uses a raw signed GET because the modeled
    # get_event embeds nested Market models the pinned SDK can no longer
    # deserialize (see module Notes). A failed lookup is recorded as "" (a
    # genuine answer) so it is not retried on every backtest run.
    #
    # This costs one HTTP round-trip per ticker, so it is BOUNDED, PACED and
    # RUN IN PARALLEL: it was written for a handful of stragglers, but at
    # current Kalshi volumes the bulk listings leave millions unresolved
    # (3,987,139 on the 2026-09-24 7-day run, almost all combo tickers), and
    # sequentially that is hours of silent grinding.
    if remaining:
        if remaining <= EVENT_TITLE_FALLBACK_MAX_LOOKUPS:
            # Everything fits: look every ticker up, combos included, exactly
            # as before DR-51. Sorted only so progress is reproducible.
            to_look_up = sorted(tkr for tkr in event_tickers if _unresolved(tkr))
            logging.info("Event titles: looking up %d tickers individually",
                         len(to_look_up))
        else:
            # It does not fit: spend the cap where a title can change a pair.
            # Deterministic order, so a re-run cannot shuffle coverage, and —
            # since a deferred ticker is not stored — a later run takes the
            # next slice. Combo tickers are counted, never listed: at
            # bulk-window volume there are millions of them.
            def _lookup_priority(tkr: str) -> tuple[int, str]:
                """
                Order the capped non-combo slice: tickers the accumulator has
                never answered, then its stored pills, then tickers it already
                titles, each group by ticker. With the cache on only the first
                group is unresolved, so this is plain ticker order. Under
                use_cache=False every requested ticker is re-resolved, and a
                deferred ticker the accumulator titles keeps that title, so
                spending the cap on the unanswered ones first is what lets
                repeated --no-cache runs advance through the deferred tail
                instead of re-looking-up the same head every run.
                """
                stored = disk_titles.get(tkr)
                return (0 if stored is None else 1 if not stored else 2), tkr

            non_combo = sorted((tkr for tkr in event_tickers
                                if _unresolved(tkr) and not _is_combo_event(tkr)),
                               key=_lookup_priority)
            to_look_up = non_combo[:EVENT_TITLE_FALLBACK_MAX_LOOKUPS]
            deferred_non_combo = len(non_combo) - len(to_look_up)
            deferred_combo = remaining - len(non_combo)
            if deferred_non_combo:
                # Non-combo coverage is being lost this run — that is a warning.
                logging.warning(
                    "Event titles: %d tickers unresolved after the bulk listings "
                    "(%d non-combo, %d combo), more than the lookup cap "
                    "EVENT_TITLE_FALLBACK_MAX_LOOKUPS=%d. Looking up %d non-combo "
                    "tickers individually and deferring the other %d non-combo "
                    "tickers%s. A deferred ticker is not looked up this run and "
                    "nothing is stored for it, so a later run tries it again; it "
                    "keeps any title the accumulator already holds and is "
                    "otherwise untitled this run, its markets grouping by market "
                    "title alone exactly as after a failed lookup (the closing "
                    "summary counts those).",
                    remaining, len(non_combo), deferred_combo,
                    EVENT_TITLE_FALLBACK_MAX_LOOKUPS, len(to_look_up),
                    deferred_non_combo,
                    f" and all {deferred_combo} combo tickers" if deferred_combo else "",
                )
            else:
                # Only combos deferred — expected on every bulk window, and
                # without pairing effect, so INFO: a WARNING that fires on every
                # run trains the operator to ignore the one above.
                logging.info(
                    "Event titles: %d tickers unresolved after the bulk listings, "
                    "more than the lookup cap EVENT_TITLE_FALLBACK_MAX_LOOKUPS=%d. "
                    "Looking up all %d non-combo tickers individually and "
                    "deferring the %d combo (%s-family) tickers, whose event "
                    "titles have no measured effect on pairing (DR-51); they are "
                    "not looked up this run and nothing is stored for them.",
                    remaining, EVENT_TITLE_FALLBACK_MAX_LOOKUPS, len(to_look_up),
                    deferred_combo, MVE_SERIES_FAMILY_PREFIX,
                )
            del non_combo

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
            finally:
                # Pace every worker after every lookup, success or failure —
                # the fetch_candlesticks idiom (DR-51). Unpaced, eight workers
                # drew 268 HTTP 429s in 2m03s on the 2026-09-24 7-day run.
                time.sleep(EVENT_TITLE_FALLBACK_RATE_LIMIT_SLEEP_SECONDS)

        if to_look_up:
            done = 0
            started = time.monotonic()
            with ThreadPoolExecutor(max_workers=EVENT_TITLE_FALLBACK_MAX_WORKERS) as pool:
                for tkr, title in pool.map(_lookup_one, to_look_up):
                    fresh[tkr] = title
                    by_lookup += bool(title)
                    done += 1
                    if done % 500 == 0:
                        elapsed = max(time.monotonic() - started, 1e-9)
                        eta = (len(to_look_up) - done) / (done / elapsed)
                        logging.info("Event titles: %d/%d individual lookups done "
                                     "(ETA %s)", done, len(to_look_up),
                                     _format_duration(eta))
        del to_look_up

    # Merge this run's answers into the accumulator IN PLACE: a fresh non-empty
    # title wins over the disk value, and a fresh "" (a failed lookup, or an
    # event listed without a title) never clobbers a non-empty disk title — but
    # a fresh "" for a ticker disk has never seen IS stored, so poison-pill
    # semantics survive. A deferred ticker is not in `fresh`, so it is never
    # stored (DR-51). Only real changes are counted, so an unchanged
    # accumulator is not rewritten.
    written = 0
    for tkr, title in fresh.items():
        old = disk_titles.get(tkr)
        if title:
            if old != title:
                disk_titles[tkr] = title
                written += 1
        elif old is None:
            disk_titles[tkr] = ""
            written += 1
    if accumulator.persist and (written or accumulator.rewrite):
        _save_json_cache(_EVENT_TITLES_CACHE, disk_titles)
        if accumulator.migrated:
            # Only now: the v2 file holding every titled legacy entry is on disk.
            _retire_legacy_event_titles()
    if not accumulator.persist:
        stored_note = " (NOT written: a file could not be read, see the WARNING above)"
    elif written or accumulator.rewrite:
        stored_note = ""
    else:
        stored_note = " (unchanged, not rewritten)"

    # Return the MERGED view, restricted to what the caller asked about. The
    # accumulator exists precisely so a ticker resolved by an earlier run need
    # not be re-fetched; returning this run's answers alone threw that away at
    # the last step and substituted the "" poison pill (TS-11). The same pass
    # sorts every requested ticker into exactly one outcome for the summary.
    result: dict[str, str] = {}
    from_accumulator = 0       # titled, but not by this run (disk hit or TS-11)
    recorded_unresolvable = 0  # untitled, with a stored or fresh "" answer
    deferred_untitled = 0      # untitled and never answered: deferred by the cap
    for tkr in event_tickers:
        title = disk_titles.get(tkr, "")
        result[tkr] = title
        if fresh.get(tkr):
            continue  # titled by this run; counted by source above
        if title:
            from_accumulator += 1
        elif tkr in fresh or tkr in disk_titles:
            recorded_unresolvable += 1
        else:
            deferred_untitled += 1
    # One line accounting for every ticker THIS call was asked about (DR-42):
    # the five counts partition the request, and the accumulator's size before
    # and after makes its growth visible run over run (DR-51).
    logging.info(
        "Event titles for %d requested tickers: %d resolved by this run (%d from "
        "the bulk listings, %d from per-ticker lookups), %d of this run's "
        "tickers answered from the accumulator, %d untitled (%d recorded as "
        "unresolvable, %d deferred and not stored). Accumulator: %d -> %d "
        "entries%s.",
        len(event_tickers), by_listing + by_lookup, by_listing, by_lookup,
        from_accumulator, recorded_unresolvable + deferred_untitled,
        recorded_unresolvable, deferred_untitled, accumulator_before,
        len(disk_titles), stored_note,
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


def _utc_now() -> datetime:
    """
    The current instant, in UTC — the one clock the settled-market fetch reads.

    fetch_all_settled_markets reads it to decide whether an assembled cache is
    today's, to stamp when a corpus was assembled and which day its live
    frontier was, and to bound the live phase; a test patches this one name to
    put the whole fetch on any day it likes.

    Returns:
        datetime: datetime.now(UTC).
    """
    return datetime.now(UTC)


def _frontier_day_start(live_min_ts: int, now_ts: int) -> int:
    """
    The first second of the live phase's frontier day: the UTC day still being written.

    The live phase splits [live_min_ts, now) into whole UTC settled-days,
    which it persists, and an open-ended frontier window from the start of
    the current UTC day (never earlier than the day live_min_ts falls in),
    which it fetches every run and never persists. Every market settled
    before this second is therefore in a complete day slice or the archive;
    an assembled cache records it (meta "live_frontier_ts") so a later run
    knows from which day to extend it.

    Args:
        live_min_ts (int): The live phase's lower bound, epoch seconds.
        now_ts (int): The current epoch second.

    Returns:
        int: Epoch seconds of the frontier day's UTC midnight.
    """
    first_lo = live_min_ts - (live_min_ts % _DAY_SECONDS)
    today_lo = now_ts - (now_ts % _DAY_SECONDS)
    return max(today_lo, first_lo)

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
#
# Since SS-1 the assembled settled-market cache (settled_markets_*.jsonl.gz,
# see fetch_all_settled_markets) is written in the same "jsonl-v1" framing by
# the same writer and read by the same reader, _day_store_iter; only its meta
# block differs.
_SLICE_FORMAT_JSONL = "jsonl-v1"

# What reading a gzipped slice can raise when the FILE is at fault, as opposed
# to the caller's `keep` predicate: OSError covers a missing file, a bad gzip
# header and a CRC/length mismatch (gzip.BadGzipFile); EOFError a stream cut
# before its end-of-stream marker (an interrupted write); zlib.error a damaged
# deflate stream, which is NOT an OSError subclass. JSON decoding errors are
# ValueError and are caught separately, around the parse alone.
_SLICE_READ_ERRORS = (OSError, EOFError, zlib.error)


class _SliceUnreadable(Exception):
    """
    Raised by _day_store_iter when a slice-format file cannot be trusted.

    Covers every way the file itself can be at fault: absent, not gzip, cut
    short, damaged, not valid JSON, not the expected shape, or written under
    different fetch conditions (a meta mismatch). Never raised for an
    exception out of the caller's `keep` predicate, which propagates as-is.
    Each caller maps it to its own contract: _day_store_load to None (the day
    is refetched), SettledCorpus.open_validated to a cache miss, and a lazy
    walk that is already under way to SettledCorpusError.
    """


class SettledCorpusError(RuntimeError):
    """
    A disk-backed settled-market corpus could not be read while it was being walked.

    Raised by the lazy views fetch_all_settled_markets builds and returns
    (_DaySliceStream over the day slices and _FrontierSpool over the live
    frontier during assembly, SettledCorpus over the assembled cache
    afterwards) when a file that was verified or written earlier in the run
    disappears, is damaged, or no longer holds the records it held — and by
    the assembly itself when its second walk does not reproduce its first. Loud by design: those records have already been
    partly consumed, so the only alternatives would be a silently SHORT
    corpus or a retry through a sequential walk that holds the whole range in
    memory. The message names the file and the remedy (re-run: the damaged
    file fails its validity check next time and is rebuilt). A RuntimeError
    subclass, so the backtester's documented "the corpus did not iterate
    identically" failures share one base class.
    """


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


def _check_slice_meta(path: Path, meta: Any, expect_meta: dict) -> None:
    """
    Raise _SliceUnreadable unless a slice file's meta block matches expectations.

    Args:
        path (Path): The file being read, named in the error.
        meta (Any): The file's parsed "meta" value. Absent/null reads as an
            empty block (which then fails any non-empty expectation); anything
            that is not a JSON object cannot have been written by this module
            and is refused.
        expect_meta (dict): Key/value pairs the block must match exactly.

    Raises:
        _SliceUnreadable: On a non-object meta block or any mismatched key.
    """
    meta = meta or {}
    if not isinstance(meta, dict):
        raise _SliceUnreadable(f"{path}: meta block is not a JSON object")
    for key, want in expect_meta.items():
        if meta.get(key) != want:
            raise _SliceUnreadable(
                f"{path}: meta {key}={meta.get(key)!r}, expected {want!r} "
                f"(written under different conditions)"
            )


def _copy_meta(meta: Any, meta_out: dict | None) -> None:
    """
    Hand a slice file's (already checked) meta block to _day_store_iter's caller.

    Args:
        meta (Any): The file's parsed "meta" value, which _check_slice_meta has
            just accepted — a JSON object, or absent/null (an empty block).
        meta_out (dict | None): The caller's sink, or None when it asked for
            nothing. Cleared, then filled with the block's keys.
    """
    if meta_out is None:
        return
    meta_out.clear()
    meta_out.update(meta or {})


def _day_store_iter(
    path: Path,
    expect_meta: dict,
    keep: Callable[[dict], bool] | None = None,
    *,
    meta_out: dict | None = None,
) -> Iterator[dict]:
    """
    Yield a slice-format file's records one at a time, validating as it goes.

    The single reader of the on-disk slice format, shared by _day_store_load
    (which gathers it into a list, all or nothing), by _DaySliceStream (the
    lazy per-phase view fetch_all_settled_markets assembles from) and by
    SettledCorpus (the assembled cache). Reads BOTH formats (see the format
    note above): the legacy single JSON document and the streamed "jsonl-v1"
    line format. Routing is done on file CONTENT, never on whether a
    whole-file parse succeeds — an empty "jsonl-v1" day is a lone meta line,
    which parses perfectly well as a JSON document, so a "did json.loads
    work?" test would misclassify it as legacy and silently report every
    empty day as having no meta match.

    The JSONL path parses and yields one line at a time, so neither a
    multi-million-record day nor the assembled corpus is ever materialized by
    this function; `keep` is applied per record before it is yielded. The
    legacy path necessarily parses its whole document first (it is one JSON
    value) and then yields from it.

    The meta block is checked before the first record is yielded. A problem
    found further in — a malformed line, a truncated or damaged stream —
    raises _SliceUnreadable at that point, AFTER the records before it were
    yielded: a caller that must be all-or-nothing collects first (as
    _day_store_load does) and a caller that streams must treat the exception
    as fatal to the whole walk (as _DaySliceStream and SettledCorpus do).
    Nothing here ever returns a short result quietly.

    Args:
        path (Path): File path (a day slice from _day_store_path(), or the
            assembled cache).
        expect_meta (dict): Key/value pairs the file's meta block must match
            exactly (e.g. cutoff_ts, include_mve, complete). A mismatch means
            the file was written under different fetch conditions.
        keep (Callable[[dict], bool] | None): Optional per-record predicate.
            Records for which it returns False are discarded as they are read
            and never yielded. Applying it here is equivalent to filtering the
            output afterwards — order and membership are identical. Its own
            exceptions propagate unchanged (they are not a fault of the file).
        meta_out (dict | None): Optional sink for the file's WHOLE meta block,
            informational keys included (e.g. the assembled cache's
            assembled_at and archive_cutoff_ts, which expect_meta deliberately
            does not name). When given, it is cleared and filled with that
            block as soon as the block has passed the check, before any record
            is yielded — so a reader validating the file by one walk learns
            what the file says about itself from that same walk, never from a
            second read that could see a different file. Untouched when the
            check fails.

    Yields:
        dict: The file's compact market dicts, in file order, filtered by
            `keep` when given. Every record is a fresh object.

    Raises:
        _SliceUnreadable: When the file is absent, not gzip, truncated,
            damaged, not valid JSON, not of the expected shape (a non-object
            record, meta block or document, or a non-list "markets"), or its
            meta does not match `expect_meta`.
    """
    if not path.exists():
        raise _SliceUnreadable(f"{path}: file does not exist")
    try:
        # Read as bytes so the orjson and stdlib parsers take the same input.
        # Both accept UTF-8 bytes, and slices are plain JSON either way.
        fh = gzip.open(path, "rb")
    except OSError as exc:
        raise _SliceUnreadable(f"{path}: cannot open ({exc})") from exc
    with fh:
        try:
            first_line = fh.readline()
        except _SLICE_READ_ERRORS as exc:
            raise _SliceUnreadable(f"{path}: unreadable ({exc!r})") from exc
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
            # line (the absence of "markets" is the discriminator, and it
            # holds for the meta-only empty-day file too).
            _check_slice_meta(path, head.get("meta"), expect_meta)
            _copy_meta(head.get("meta"), meta_out)
            lines = iter(fh)
            while True:
                # Streaming, one record at a time: this is what keeps a
                # multi-million-record file bounded on the read side. The
                # read and the parse are guarded separately from `keep`, so a
                # predicate failure is never mistaken for a damaged file.
                try:
                    line = next(lines)
                except StopIteration:
                    return
                except _SLICE_READ_ERRORS as exc:
                    raise _SliceUnreadable(f"{path}: unreadable ({exc!r})") from exc
                if not line.strip():
                    continue
                try:
                    record = _slice_loads(line)
                except ValueError as exc:
                    raise _SliceUnreadable(f"{path}: malformed record line ({exc})") from exc
                if not isinstance(record, dict):
                    raise _SliceUnreadable(f"{path}: a record line is not a JSON object")
                if keep is None or keep(record):
                    yield record
        try:
            rest = fh.read()
        except _SLICE_READ_ERRORS as exc:
            raise _SliceUnreadable(f"{path}: unreadable ({exc!r})") from exc
    try:
        payload = head if (head is not None and not rest.strip()) else _slice_loads(
            first_line + rest
        )
    except ValueError as exc:
        raise _SliceUnreadable(f"{path}: not valid JSON ({exc})") from exc
    if not isinstance(payload, dict):
        raise _SliceUnreadable(f"{path}: document is not a JSON object")
    _check_slice_meta(path, payload.get("meta"), expect_meta)
    _copy_meta(payload.get("meta"), meta_out)
    markets = payload.get("markets") or []
    if not isinstance(markets, list) or not all(isinstance(m, dict) for m in markets):
        raise _SliceUnreadable(f"{path}: \"markets\" is not a list of JSON objects")
    for m in markets:
        if keep is None or keep(m):
            yield m


def _day_store_load(
    path: Path,
    expect_meta: dict,
    keep: Callable[[dict], bool] | None = None,
) -> list[dict] | None:
    """
    Load a day-slice file if it exists and its metadata matches expectations.

    The all-or-nothing form of _day_store_iter (see there for both formats,
    the content-based routing rule and the per-record `keep`): the records are
    gathered into a list only once the whole file has been read cleanly. Any
    malformed line, unreadable file, or meta mismatch fails the WHOLE slice
    (returns None → the caller refetches the day); a partially decodable slice
    is never returned, which is the same contract the single-document format
    always had. Used by the phases' reuse prescans (with _discard_all, so
    nothing is retained) and by direct callers; assembly streams through
    _DaySliceStream instead.

    Args:
        path (Path): File path from _day_store_path().
        expect_meta (dict): Key/value pairs the file's meta block must match
            exactly (e.g. cutoff_ts, include_mve, complete). A mismatch means
            the file was written under different fetch conditions and must be
            ignored (the caller refetches the day).
        keep (Callable[[dict], bool] | None): Optional per-record predicate.
            Records for which it returns False are discarded as they are read
            and never retained. Applying it here is equivalent to filtering the
            returned list afterwards — order and membership are identical. An
            exception it raises propagates (it is not a fault of the file).

    Returns:
        list[dict] | None: The slice's compact market dicts (filtered by `keep`
            when given), or None when the file is absent, unreadable, or
            written under different conditions.
    """
    try:
        return list(_day_store_iter(path, expect_meta, keep))
    except _SliceUnreadable:
        return None


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
    Incremental writer for one "jsonl-v1" file: a day slice, or the assembled cache.

    Exists so a fetch worker never accumulates a whole UTC day of records: at
    2026-08 volumes a single day reaches ~4.4–4.8M compact market dicts, and
    holding one per worker sawtoothed RSS to 7.89 GB on a 16 GB host (live
    telemetry, 2026-08-31). The worker instead hands over batches of at most
    SETTLED_FETCH_CHUNK_RECORDS records, which are serialized and compressed
    straight into the file and then dropped.

    Since SS-1 fetch_all_settled_markets also writes the assembled
    settled-market cache (settled_markets_*.jsonl.gz) through it, one record
    at a time (write_record) as its second assembly walk produces them, so the
    assembled corpus is never held in memory either; only the meta block it is
    given differs from a day slice's.

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
            path (Path): Final destination path — from _day_store_path(), or
                the assembled cache's path in fetch_all_settled_markets.
            meta (dict): Metadata block from _day_store_meta() (or, for the
                assembled cache, _assembled_cache_meta() plus a timestamp);
                the "jsonl-v1" format tag is added here so callers cannot
                forget it.

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

    def write_record(self, record: dict) -> None:
        """
        Append one record as a JSON line.

        The streamed assembly (fetch_all_settled_markets) produces records one
        at a time from a generator; writing each as it arrives, rather than
        batching, keeps nothing of the corpus resident beyond the record in
        hand.

        Args:
            record (dict): One compact market dict (see _market_to_dict).

        Raises:
            OSError: On a write failure; the caller must abort() the writer
                (leaving a `with` block does so).
        """
        self._fh.write(_slice_dumps(record) + b"\n")
        self._records += 1

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


class _DaySliceStream:
    """
    Lazy, re-iterable view of one phase's completed day slices, newest day first.

    What a phase hands fetch_all_settled_markets instead of a list: every walk
    (`for m in stream`) re-opens the slice files and yields their records one
    at a time through _day_store_iter, so no slice — and no phase — is ever
    held in memory. The assembly walks each stream twice (once to count and
    collect event tickers, once to write the assembled cache), and each walk
    yields fresh dicts in the same order.

    A slice that cannot be read DURING a walk — it disappeared, was damaged,
    or was rewritten under different conditions since the phase verified or
    wrote it — raises SettledCorpusError naming the day and the remedy. That
    is a deliberate change from the old eager assembly, which read every slice
    inside the phase and turned such a failure into _ShardedFetchUnsupported
    and hence the sequential fallback: a walk under way has already handed
    records to its consumer, so falling back would either duplicate them or
    silently leave a SHORT corpus, and the sequential walk holds the whole
    range in memory besides (unbounded at current volumes). The next run's
    reuse prescan finds the damaged day invalid and fetches it again.
    """

    def __init__(
        self,
        store: str,
        day_los: Iterable[int],
        expect_meta: dict,
        keep: Callable[[dict], bool] | None = None,
    ):
        """
        Args:
            store (str): Store subdirectory name ("archive_days" or "live_days").
            day_los (Iterable[int]): UTC-midnight lower bounds of every day
                known to have a valid slice on disk (reused or just written).
                Copied and sorted newest-first here, once, so every walk yields
                the same sequence however the caller's list later changes.
            expect_meta (dict): Reuse-gating keys the slices must still match
                (copied).
            keep (Callable[[dict], bool] | None): Optional predicate applied per
                record as slices are read, so records the caller would discard
                anyway are never yielded. The phases no longer pass one (M9):
                it saved no memory — a walk holds one record at a time either
                way, and the assembly discards a rejected record as soon as it
                has tested it — while hiding every record it dropped from the
                assembly's count of what the prefilter rejected. The slice
                FILES are never filtered; they stay complete for other start
                dates.
        """
        self._store = store
        # Newest day first: the order the old in-memory assembly produced, and
        # part of this module's output contract, since the caller's ticker
        # dedup is first-wins. Paths are resolved here, once, so every walk
        # reads the very files the phase verified or wrote (CACHE_DIR is read
        # at call time by _day_store_path, and tests repoint it).
        self._slices = [(lo, _day_store_path(store, lo))
                        for lo in sorted(day_los, reverse=True)]
        self._expect_meta = dict(expect_meta)
        self._keep = keep

    def __iter__(self) -> Iterator[dict]:
        """
        Walk every slice, newest day first, yielding one record at a time.

        Yields:
            dict: Compact market dicts (fresh objects on every walk), filtered
                by the stream's `keep`.

        Raises:
            SettledCorpusError: If a slice cannot be read during this walk.
        """
        for lo, path in self._slices:
            try:
                yield from _day_store_iter(path, self._expect_meta, self._keep)
            except _SliceUnreadable as exc:
                day = datetime.fromtimestamp(lo, tz=UTC).date().isoformat()
                raise SettledCorpusError(
                    f"The {self._store} slice for {day} could not be read while "
                    f"the settled-market corpus was being assembled ({exc}). It "
                    f"was verified or written earlier in this run, so it "
                    f"disappeared or was damaged since. Re-run the backtest: "
                    f"the reuse prescan will find the day missing or unreadable "
                    f"and fetch it again. Not retried here through the "
                    f"sequential walk, which would hold the whole range in "
                    f"memory, and never skipped, which would leave the corpus "
                    f"short."
                ) from exc


class _RecordChain:
    """
    Re-iterable concatenation of record sources, walked in order on every pass.

    The live phase's return value: its frontier spool (_FrontierSpool)
    followed by its past-day _DaySliceStream — the same records in the same
    order as the old `frontier + [...]` list, without materializing either.
    Each walk walks every part afresh, so the chain is re-iterable exactly
    when its parts are (a list, a _FrontierSpool or a _DaySliceStream).
    close() releases whichever parts hold a resource (the spool).
    """

    def __init__(self, *parts: Iterable[dict]):
        """
        Args:
            *parts (Iterable[dict]): Re-iterable record sources, in the order
                they are to be walked.
        """
        self._parts = parts

    def __iter__(self) -> Iterator[dict]:
        """
        Yields:
            dict: Every record of every part, part by part, in order.
        """
        for part in self._parts:
            yield from part

    def close(self) -> None:
        """
        Close every part that can be closed (see _close_records); idempotent.
        """
        for part in self._parts:
            _close_records(part)


def _close_records(records: Any) -> None:
    """
    Release a phase result's resources, if it holds any.

    A phase hands back a list, a _DaySliceStream, a _RecordChain or a test's
    stand-in; only the live phase's frontier spool (reached through its
    _RecordChain) holds anything that needs releasing. Anything without a
    callable close() is left alone, so every one of those can be passed.

    Args:
        records (Any): The phase result (or any part of one).
    """
    close = getattr(records, "close", None)
    if callable(close):
        close()


def _assemble_day_slices(
    store: str,
    day_los: list[int],
    expect_meta: dict,
    keep: Callable[[dict], bool] | None = None,
) -> _DaySliceStream:
    """
    Return a phase's day-slice records as a lazy stream off disk (never a list).

    Slices are read back rather than held in RAM because the archive phase
    alone spans ~1,900 created-days (every day since the archive's first) at
    up to ~475k records each — retaining every slice (and
    then flattening it into a second list) was measured at 2.7 GB RSS only 17%
    of the way through a run, and grew superlinearly as GC pressure mounted.
    Since SS-1 even the read-back is not collected: a 7-day window's past days
    held 18,061,549 fetched records, of which 7,260,952 passed the backtester's
    prefilter — about 28 GB at the 3,926 B/record measured on those records —
    so this returns a _DaySliceStream that the assembly walks record by record,
    and nothing is read until then.

    Days are emitted newest-first, matching the order the in-memory version
    produced — record order is part of this module's output contract, since the
    caller's ticker dedup is first-wins.

    Args:
        store (str): Store subdirectory name ("archive_days" or "live_days").
        day_los (list[int]): UTC-midnight lower bounds of every day known to
            have a valid slice on disk (reused or just written).
        expect_meta (dict): Reuse-gating keys the slices must still match.
        keep (Callable[[dict], bool] | None): Optional predicate applied per
            record as slices are read (see _DaySliceStream for why the phases
            pass none). The slice FILES are never filtered; they stay complete
            for other start dates.

    Returns:
        _DaySliceStream: Compact market dicts, newest day first, re-read from
            disk on every walk. A slice that cannot be read during a walk
            raises SettledCorpusError there; it no longer takes the sequential
            fallback (see _DaySliceStream for why).
    """
    return _DaySliceStream(store, day_los, expect_meta, keep)


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


def _archive_first_ts() -> int:
    """
    Epoch seconds of ARCHIVE_FIRST_CREATED_DATE's UTC midnight.

    The archive phase reads one slice per created-day from this day (or from
    the window's start, when that is earlier) to the cutoff. The constant is
    read when called, so a test can move the first day by patching
    historical.ARCHIVE_FIRST_CREATED_DATE.

    Returns:
        int: UTC midnight of the archive's first created-day, epoch seconds.
    """
    first = ARCHIVE_FIRST_CREATED_DATE
    return int(datetime(first.year, first.month, first.day, tzinfo=UTC).timestamp())


def _check_archive_floor(hist_client: Any, first_ts: int, hist_kwargs: dict) -> None:
    """
    Warn when the archive holds a market created before the first day read.

    The archive phase reads every created-day from first_ts up, so a market
    created earlier is never read, even if it settles inside the window.
    ARCHIVE_FIRST_CREATED_DATE sits below the oldest market Kalshi's archive
    served when it was set; this one request, with a cursor at first_ts,
    asks whether that still holds: its page starts with the newest market
    created before first_ts, if there is one.

    It never stops the run: a market found below first_ts, or a request that
    fails, is one WARNING, and the fetch goes on with what it read.

    Args:
        hist_client (Any): Authenticated KalshiClient.
        first_ts (int): The first created-day the phase reads, UTC midnight.
        hist_kwargs (dict): Base query params (limit, optional mve_filter), as
            every archive page is asked for; the page below the first day is
            empty while the constant holds, so the limit costs nothing.
    """
    first_day = datetime.fromtimestamp(first_ts, tz=UTC).date().isoformat()
    try:
        data = _historical_get(
            hist_client, f"{_API_PREFIX}/historical/markets",
            cursor=_encode_archive_cursor(first_ts, 0, _CURSOR_TICKER_SENTINEL),
            **hist_kwargs,
        )
        older = [m for m in data.get("markets") or []
                 if isinstance(m, dict)
                 and (created := _iso_epoch(m.get("created_time"))) is not None
                 and created < first_ts]
    except Exception as exc:
        logging.warning(
            "Archive first day: could not check for markets created before %s "
            "(%s) — any such market is not read", first_day, _exception_summary(exc),
        )
        return
    if older:
        logging.warning(
            "Archive first day: the archive holds markets created before %s "
            "(the newest of them, %s, was created %s), and the archive day "
            "slices start at %s, so those markets are never read, even if they "
            "settle inside the window — set config.ARCHIVE_FIRST_CREATED_DATE "
            "earlier", first_day, older[0].get("ticker"),
            older[0].get("created_time"), first_day,
        )


def _fetch_archive_sequential(
    hist_client: Any,
    start_ts: int,
    cutoff_ts: int,
    hist_kwargs: dict,
    keep: Callable[[dict], bool] | None = None,
    *,
    tally: "_AssemblyTally | None" = None,
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
    ARCHIVE_MAX_BARREN_PAGES consecutive pages with no in-window settlement.
    That stop can miss a long-lived market created far below a stretch of
    such pages (the sharded path reads every created-day instead, and is
    complete); walking the whole archive one page at a time would take
    hours, and this walk runs only when cursor synthesis fails.

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
        tally (_AssemblyTally | None): When given, the in-window records
            `keep` rejected are added to it once the walk completes
            (_AssemblyTally.note_rejected) — every record this walk tests is
            already inside [start_ts, cutoff_ts), so each is one the
            assembly's count of settled records must include, and the
            assembly never sees it (M9). Nothing is added if the walk raises.

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
    if tally is not None:
        # walk_kept counts in-window binary records BEFORE `keep`, so the gap
        # is exactly what the prefilter dropped here — records the assembly
        # will never see and so could not count itself (M9).
        tally.note_rejected(walk_kept - len(selected))
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
    *,
    tally: "_AssemblyTally | None" = None,
) -> Iterable[dict]:
    """
    Fetch the archive's contribution: one slice per created-day, from the first.

    Sharded path (preferred): verify cursor synthesis, then fetch one slice
    per UTC created-day from the earlier of ARCHIVE_FIRST_CREATED_DATE and
    start_ts up to cutoff_ts — reusing any slice already on disk from a
    previous run — with SETTLED_FETCH_MAX_WORKERS parallel workers. The days
    before start_ts are what find the long-lived markets: a market created
    in 2023 that settles inside the window sits in its 2023 slice, and the
    caller's assembly keeps it because its settlement is in the window. The
    archive cannot be asked for "created before start_ts and settled after
    it" (it has no time filter), so reading every earlier created-day is the
    only complete way; a walk that stopped after a run of pages with nothing
    in the window missed 60 such markets on 2026-10-01. Each worker persists
    its own slice as soon as the day completes, so an interrupted fetch
    resumes at day granularity instead of restarting the multi-hour walk.

    Slice records are never accumulated in memory: a worker streams its records
    into its slice file in chunks and returns only a count, and the day-slice
    records are handed back as a lazy _DaySliceStream from
    _assemble_day_slices (see there for why — the ~1,900 days back to the
    first one otherwise run to tens of GB, and a single 2026-08 day is itself
    millions of records).
    Nothing is read back here; the caller's assembly walks the stream. A slice
    that cannot be read during that walk raises SettledCorpusError from the
    walk itself — it no longer reaches the sequential fallback below, which
    only covers failures inside this phase (see _DaySliceStream).

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
        keep (Callable[[dict], bool] | None): Optional per-record predicate,
            applied only by the sequential fallback, per record as its pages
            arrive, so the list it holds never accumulates records the caller
            will discard. The day-slice stream is returned UNFILTERED (M9):
            filtering it at read-back saved no memory and hid its rejections
            from the assembly's count; the caller's assembly applies the same
            predicate to every record and counts. Slice FILES stay
            unfiltered either way.
        tally (_AssemblyTally | None): Handed to the sequential fallback,
            which adds the in-window records `keep` dropped (the only records
            of this phase the assembly never sees). The sharded path adds
            nothing: everything it returns reaches the assembly unfiltered.

    Returns:
        Iterable[dict]: The archive's records, newest created-day first. On
            the sharded path, a _DaySliceStream — re-iterable, read lazily off
            disk on every walk, never a list, not `keep`-filtered and NOT yet
            settlement-filtered (the caller applies the [start_ts, cutoff_ts)
            window, which is what drops the earlier days' records that settled
            before the window). On the sequential fallback, a list already
            settlement-filtered and `keep`-filtered; when the window starts at
            or after the cutoff, an empty list.
    """
    if start_ts >= cutoff_ts:
        # The archive holds only markets that settled BEFORE the cutoff, so a
        # window starting at or after it cannot contain a single archive
        # record — there is nothing here to fetch, whatever the walk would do.
        # Without this the phase still ran the cursor-synthesis probe and a
        # walk below start_ts to prove that, costing ~18 seconds and ~50,000
        # parsed records on every post-cutoff run (TS-25). Logged rather than silent
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
        return []

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

        # One slice per UTC day of created_time, from the archive's first day
        # (or start_ts, if a window starts earlier still) to the cutoff. The
        # days before start_ts hold the markets created before the window that
        # settle inside it; the assembly drops the rest of their records,
        # which settled before the window. All archive records satisfy
        # created < settle < cutoff, so days at/above the cutoff can't exist.
        first_ts = min(_archive_first_ts(), start_ts)
        day_los = list(range(first_ts, cutoff_ts, _DAY_SECONDS))
        expect_meta = {
            "kind": "archive_created_day",
            "cutoff_ts": cutoff_ts,
            "include_mve": INCLUDE_MVE_MARKETS,
            "complete": True,
        }
        # Only day identities are tracked here, never their records: the
        # prescan's loaded slices are discarded and re-read at assembly. That
        # costs two extra decodes of the reused days (one per assembly walk)
        # but keeps peak memory independent of how many days the window
        # spans. _discard_all makes that literal — a streamed slice is fully
        # parsed (so corruption is still caught) but nothing is retained from
        # it.
        on_disk: list[int] = []
        to_fetch: list[int] = []
        for lo in day_los:
            if _day_store_load(_day_store_path("archive_days", lo), expect_meta,
                               _discard_all) is not None:
                on_disk.append(lo)
            else:
                to_fetch.append(lo)
        reused = len(on_disk)
        logging.info(
            "Archive day slices (created-days %s to %s): %d reused from disk, "
            "%d to fetch",
            datetime.fromtimestamp(day_los[0], tz=UTC).date().isoformat(),
            datetime.fromtimestamp(day_los[-1], tz=UTC).date().isoformat(),
            reused, len(to_fetch),
        )

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

        # The first day is a fixed date, so say so loudly if Kalshi ever
        # serves a market created before it: such a market is never read.
        _check_archive_floor(hist_client, first_ts, hist_kwargs)

        # A lazy stream, not a list: nothing is read here, and every later
        # walk re-reads the slices one record at a time (SS-1). Unfiltered
        # (M9): the assembly applies `keep` itself and counts what it rejects.
        return _assemble_day_slices("archive_days", on_disk, expect_meta)
    except _ShardedFetchUnsupported as exc:
        logging.warning(
            "Archive fetch: sharded path unavailable (%s) — falling back to the "
            "sequential walk. Any day slices already completed remain on disk "
            "and will be reused by the next run.", exc,
        )
        # The caller's prefilter, applied per record as each page arrives;
        # the walk still holds its whole keep-passing result, and adds what it
        # dropped to the tally, since the assembly never sees those records.
        return _fetch_archive_sequential(hist_client, start_ts, cutoff_ts,
                                         hist_kwargs, keep, tally=tally)


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
            keeps only records passing the caller's predicate and spools them
            to an anonymous temporary file (_fetch_live_phase / _extend_kept /
            _FrontierSpool), so any frontier record, kept or not, is resident
            only while its batch is in flight. None returns
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
    *,
    tally: "_AssemblyTally | None" = None,
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
        tally (_AssemblyTally | None): When given, the records `keep`
            rejected are added to it once the walk completes
            (_AssemblyTally.note_rejected): the assembly never sees them, so
            it could not count them itself (M9). Every one settled at or after
            live_min_ts — the server-side bound this walk requests, which is
            never below the backtest window's start — so each is a record
            settled in the window. Nothing is added if the walk raises.

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
            if tally is not None:
                # walk_kept counts binary settled records BEFORE `keep`, so the
                # gap is exactly what the prefilter dropped here (M9).
                tally.note_rejected(walk_kept - len(kept))
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
    fetched directly and deliberately never persisted as a slice, since it was
    captured mid-day. It is still streamed in chunks, but into a private,
    anonymous spool file (_FrontierSpool) that keeps only the records passing
    the caller's predicate (see _fetch_live_phase): what stays resident is one
    batch in flight, whatever the predicate, and the spool vanishes with the
    run.

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


class _FrontierSpool:
    """
    The frontier day's keep-passing records, spooled to an anonymous temporary file.

    The live frontier (the current, partial UTC day) is never persisted as a
    day slice — it was captured mid-day and must never be reused as complete —
    so it used to be held as a list (filtered as its pages arrived), which the
    assembly walks twice. Such a list is bounded by the prefilter alone, and
    the prefilter's pass rate depends on the WEEKDAY: _can_ever_enter admits
    every market that was open over a Monday checkpoint on/after start_date
    and closes at least a day later, so a frontier captured the day after a
    Monday can be mostly eligible — 7,190,452 of the 9,176,306 records
    settled on Tuesday 2026-09-22 passed _can_ever_enter(m, 2026-09-17),
    ~28 GB at the 3,926 B/record measured on eligible records of that window,
    on a 16 GB host (measured under the date-granular predicate P5 replaced,
    which also admitted markets opened later on the Monday itself; the
    checkpoint-instant one admits a subset, so the bound only improves). So
    the frontier worker's sink (_extend_kept) streams
    each kept batch into this spool instead, and every walk reads it back one
    record at a time, exactly like a day slice.

    The file is created with tempfile.TemporaryFile, which on POSIX unlinks
    it immediately: it has no name, no later run can ever find or reuse it
    (so it can never be mistaken for a complete day), and the operating
    system reclaims it when it is closed or the process dies, however the
    process dies — no stale spool can outlive a crash or an OOM kill. It is
    created in CACHE_DIR, under PROJECT_ROOT, beside the slices it stands in
    for. It carries the day slices' line framing (_slice_dumps, one record per
    line, gzip level 1) and no meta block, since nothing but this object ever
    reads it.

    Lifecycle: extend() from the one frontier worker thread (under
    _fetch_live_window's emit contract); seal() once, by the phase's thread,
    after the frontier future has completed; then any number of walks, one
    at a time (a new walk invalidates a suspended one, which raises if
    resumed, since both would share the one file position); close() when the
    assembly is done (fetch_all_settled_markets closes the live phase's
    result in a `finally`), or on any failure of the phase.
    """

    def __init__(self, directory: Path):
        """
        Create the anonymous spool file and open its compressed writer.

        Args:
            directory (Path): Where the (immediately unlinked) file is
                created — the caller passes CACHE_DIR, resolved at call time
                so tests can repoint it. Created if absent.

        Raises:
            OSError: If the directory or the temporary file cannot be created.
        """
        directory.mkdir(parents=True, exist_ok=True)
        self._file = tempfile.TemporaryFile(dir=directory, prefix="frontier-spool-")
        try:
            self._gz = gzip.GzipFile(fileobj=self._file, mode="wb", compresslevel=1)
        except BaseException:
            self._file.close()
            raise
        self._count = 0
        self._sealed = False
        self._closed = False
        # Bumped by every walk; a walk whose number is no longer current
        # stops rather than read from a file position another walk moved.
        self._walk = 0

    def extend(self, records: Iterable[dict]) -> None:
        """
        Append records, in order, as JSON lines (the list-like sink interface).

        Named for list.extend so _extend_kept can fill either; each record is
        serialized and dropped as it is written, so nothing here retains one.

        Args:
            records (Iterable[dict]): Compact market dicts, in fetch order.

        Raises:
            RuntimeError: If the spool was already sealed or closed (a bug —
                only the frontier worker writes, and only before seal()).
            OSError: On a write failure (e.g. a full disk); it propagates to
                the frontier future and out of the phase.
        """
        if self._sealed or self._closed:
            raise RuntimeError("frontier spool written after it was sealed or closed")
        write = self._gz.write
        for record in records:
            write(_slice_dumps(record) + b"\n")
            self._count += 1

    def seal(self) -> None:
        """
        Finish the compressed stream; from here on the spool is read-only.

        Raises:
            OSError: If the final flush fails.
        """
        self._gz.close()
        self._sealed = True

    def __len__(self) -> int:
        """
        Returns:
            int: How many records have been spooled.
        """
        return self._count

    def __iter__(self) -> Iterator[dict]:
        """
        Read the spool back once, in the order it was written.

        Yields:
            dict: Compact market dicts, fresh objects on every walk.

        Raises:
            SettledCorpusError: If the spool is not sealed or already closed,
                a newer walk started while this one was suspended, it cannot
                be decoded, or a complete walk yields a different number of
                records than were written — each a bug or a failing disk,
                never a condition to paper over with a short frontier.
        """
        if self._closed or not self._sealed:
            raise SettledCorpusError(
                "The frontier spool was walked before it was sealed or after it "
                "was closed — a bug in the live phase's lifecycle.")
        self._walk += 1
        walk = self._walk
        self._file.seek(0)
        walked = 0
        try:
            with gzip.GzipFile(fileobj=self._file, mode="rb") as reader:
                for line in reader:
                    if walk != self._walk:
                        raise SettledCorpusError(
                            "A walk over the frontier spool was resumed after a "
                            "newer walk had started; walks must not interleave.")
                    walked += 1
                    yield _slice_loads(line)
        except (*_SLICE_READ_ERRORS, ValueError) as exc:
            raise SettledCorpusError(
                f"The frontier spool could not be read back ({exc!r}). It is a "
                f"private temporary file, so this is a failing disk or a bug; "
                f"re-run the backtest (the frontier day is always refetched)."
            ) from exc
        if walked != self._count:
            raise SettledCorpusError(
                f"The frontier spool yielded {walked} records but {self._count} "
                f"were written. Re-run the backtest (the frontier day is always "
                f"refetched).")

    def close(self) -> None:
        """
        Release the spool: the file and its disk space go immediately.

        Safe to call more than once, and on a spool that was never sealed.
        Never raises: it runs on error paths where the original exception
        must survive.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._gz.close()
        except Exception:
            pass
        try:
            self._file.close()
        except Exception:
            pass


def _extend_kept(
    dest: "list[dict] | _FrontierSpool",
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
    predicate, same records, same order. In production `dest` is the live
    phase's _FrontierSpool, so a KEPT record is not retained either: it is
    serialized into the spool and dropped with its batch (a list `dest`, as
    the unit tests use, would retain every kept record — which on the day
    after a Monday can be most of the partial day).

    Args:
        dest (list[dict] | _FrontierSpool): Anything with list-style
            extend(); extended in place. Touched only from the one worker
            thread running the window; the caller reads it only after that
            window's future has completed (Future.result() is the
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
    *,
    tally: "_AssemblyTally | None" = None,
) -> Iterable[dict]:
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
    in chunks inside the worker and never accumulated in memory: they are
    returned as a lazy _DaySliceStream that re-reads them off disk on every
    walk of the assembly (SS-1; a slice that cannot be read during a walk
    raises SettledCorpusError there and never reaches the sequential fallback
    below), UNFILTERED — the assembly applies `keep` to them itself and
    counts what it rejects (M9). The frontier window, which is never
    persisted as a slice, is streamed in chunks too, through a sink
    (_extend_kept) that applies `keep` to each batch as its pages arrive and
    writes the survivors into a _FrontierSpool — a private, anonymous
    temporary file that no later run can find, read back one record at a
    time on every walk like a day slice. It
    used to be accumulated unfiltered and filtered only once the whole pool
    had drained — the same records in the same order, at a peak that grows
    through the UTC day (up to a full day's settlements, 9.2M records on
    2026-09-22) whatever the window length (SS-1). Filtering as it arrives
    still left the keep-passing subset resident as a list, and how large that
    is depends on the weekday, not on the window's length: the backtester's
    predicate (_can_ever_enter) admits every market that was open over a
    Monday checkpoint on/after start_date and closes at least a day later, so
    a frontier captured the day AFTER a Monday can be mostly eligible —
    7,190,452 of the 9,176,306 records settled on Tuesday 2026-09-22 passed
    _can_ever_enter(m, 2026-09-17) (under the date-granular predicate P5
    replaced; the checkpoint-instant one admits a subset). Spooled, it holds
    one batch in flight whatever the weekday, and with keep=None as well.
    The caller must close()
    the returned chain once it is done walking it (fetch_all_settled_markets
    does so in a `finally`), which releases the spool; it also vanishes with
    the process.

    Falls back to _fetch_live_sequential if the server stops honoring
    max_settled_ts (detected per window by _fetch_live_window), passing the
    same `keep`; the partial frontier spool is closed first (its disk
    released), since the sequential sweep refetches today anyway. The
    fallback's own result is still one list, as it always was.

    Args:
        live_client: KalshiClient from build_prod_live_client().
        live_min_ts (int): Server-side window start — max(cutoff_ts, start_ts).
            Bug fixed 2026-07: this used to always be cutoff_ts, which forced
            the server to walk the entire [cutoff_ts, now) range for a narrow
            recent start_date (observed 20k+ pages discarded client-side).
        now_ts (int): Current epoch seconds; determines the frontier day.
        keep (Callable[[dict], bool] | None): Optional per-record predicate
            applied only where records would otherwise be held: to each
            frontier batch as its pages arrive (on the frontier's worker
            thread, so it must be thread-safe), and per record by the
            sequential fallback (main thread). The past-day stream is NOT
            filtered (M9): the caller's assembly applies the same predicate
            to every record it reads and counts the rejections, which a
            read-back filter would hide. Slice FILES stay unfiltered.
        tally (_AssemblyTally | None): Receives the records `keep` dropped
            before the assembly could see them — the frontier's (what its
            window emitted minus what the spool kept, added only once the
            windowed path has succeeded, so a frontier discarded by the
            fallback is never counted) or, on the fallback, the sequential
            walk's own. Every such record settled at or after its window's
            server-side min_settled_ts: live_min_ts on the fallback, and on
            the windowed path at worst live_min_ts rounded down to its UTC
            midnight — never below the backtest window's start, itself a UTC
            midnight no later than live_min_ts — so each is a record settled
            in the window. That rests on the server honoring min_settled_ts,
            the same bound every live window already relies on.

    Returns:
        Iterable[dict]: Compact market dicts, frontier first then past days
            newest-first (no settlement filtering beyond min_settled_ts; the
            caller applies the backtest window) — the same records in the same
            order as the old `frontier + [...]` list. On the windowed path a
            re-iterable _RecordChain of the sealed _FrontierSpool and a lazy
            _DaySliceStream over the past days, never one list (close() it
            when done); on the sequential fallback, the fallback's list.

    Raises:
        OSError: If the frontier spool cannot be created or written (e.g. a
            full disk), like a day slice that cannot be written.
    """
    first_lo = live_min_ts - (live_min_ts % _DAY_SECONDS)
    frontier_lo = _frontier_day_start(live_min_ts, now_ts)
    past_day_los = list(range(first_lo, frontier_lo, _DAY_SECONDS))

    expect_meta = {
        "kind": "live_settled_day",
        "include_mve": INCLUDE_MVE_MARKETS,
        "complete": True,
    }
    # Filled ONLY by the frontier worker's sink below, and sealed and read by
    # this thread only after frontier_future.result() has returned. Created
    # before the `try` so every way out of this function can release it. It
    # is an anonymous temporary file under CACHE_DIR (read at call time, so
    # tests repoint it): no other run can ever see it (SS-1).
    frontier = _FrontierSpool(CACHE_DIR)
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
            # The frontier day is never persisted as a slice — it was captured
            # mid-day and must never be reused as a complete day. It streams
            # through the emit contract like a past day, but into a sink that
            # keeps only `keep`-passing records as each batch lands and spools
            # them to the anonymous file, so neither the unfiltered partial day
            # nor its keep-passing subset is ever resident (SS-1).
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
                # outlived its retries, a non-transient error, `keep` raising
                # on the worker thread, or a spool write failing. The pool's
                # __exit__ waits for the worker whether or not this runs, so
                # deleting it would not hang; it would silently return the
                # batches already spooled as if they were the whole frontier
                # (a short corpus, pinned by TestFrontierStreamsThroughKeep's
                # failure tests). The records themselves are already in the
                # spool, filtered; the returned count is how many the window
                # EMITTED before `keep`, so the gap to the spool's length is
                # exactly what the prefilter dropped (M9, counted below).
                frontier_emitted = frontier_future.result()
            except BaseException:
                # Abandon queued windows immediately rather than draining them
                # on the way out to the sequential fallback.
                pool.shutdown(wait=False, cancel_futures=True)
                raise

        # The frontier worker has finished (result() above returned), so no
        # write can race this: finish the compressed stream, read-only from
        # here. No post-hoc `keep` pass: the sink already applied it in fetch
        # order, so the spool holds exactly the list that pass produced.
        frontier.seal()
        if tally is not None:
            # Only now, on success: a frontier the fallback below discards
            # must not have its rejections counted beside the sequential
            # walk's, which refetches the same day.
            tally.note_rejected(frontier_emitted - len(frontier))
        # Chained rather than concatenated (SS-1): the frontier and the past
        # days stay on disk, walked in the same order `frontier + [...]` had.
        # The past days are unfiltered (M9): the assembly applies `keep` to
        # them and counts what it rejects.
        return _RecordChain(
            frontier, _assemble_day_slices("live_days", on_disk, expect_meta),
        )
    except _ShardedFetchUnsupported as exc:
        logging.warning(
            "Live fetch: windowed path unavailable (%s) — falling back to the "
            "sequential sweep. Any settled-day windows already completed remain "
            "on disk and will be reused by the next run.", exc,
        )
        # The sequential sweep refetches today as well, so the frontier the
        # windowed path spooled is dead weight here. Nothing writes it any
        # more — the pool's __exit__ joined the frontier worker (or its
        # cancel_futures dropped it while still queued) — and a running worker
        # finishes its whole window before that join returns, so without this
        # the spool's disk would stay claimed for the whole serial walk. Closed
        # explicitly rather than left to garbage collection, because the
        # sink's partial keeps it reachable from frontier_future when the
        # frontier window itself failed.
        frontier.close()
        # Same prefilter as the windowed path's frontier, applied per record as
        # each page arrives; the walk still holds its whole keep-passing
        # result, and adds what it dropped to the tally.
        return _fetch_live_sequential(live_client, live_min_ts, keep, tally=tally)
    except BaseException:
        # Any other failure (a frontier or past-day window error, a spool that
        # could not be written or sealed): release the spool, then propagate.
        frontier.close()
        raise


# ─── Assembly and the assembled cache ─────────────────────────────────────────

# Identifies an assembled-cache file's "kind" in its meta block, so a day
# slice (or anything else in the jsonl-v1 framing) can never be mistaken for
# one even if it were copied to the assembled cache's filename.
_ASSEMBLED_CACHE_KIND = "settled_markets_assembled"


def _assembled_cache_meta(start_date: date, prefilter_tag: str | None) -> dict:
    """
    The meta block an assembled cache file must carry to be served for a request.

    Every key here is part of the RESULT's identity, exactly as each component
    of the filename is (start date, prefilter tag, the DR-57 MVE flag): the
    filename already separates them, and repeating them in the meta block means
    a file copied or renamed onto another request's name is refused rather
    than trusted. The format tag is the one _DayStreamWriter writes.

    Args:
        start_date (date): The window start the corpus was assembled for.
        prefilter_tag (str | None): The prefilter's tag, or None when no
            prefilter was applied.

    Returns:
        dict: The expected meta block (without the three informational keys
            the writer also records — assembled_at, archive_cutoff_ts and
            assembly_counts, see CorpusProvenance — which describe WHEN and
            UNDER WHICH CUTOFF a corpus was assembled and what its prefilter
            rejected, not WHICH request it answers, so they are never
            compared and a file written before any of them existed is still
            served).
    """
    return {
        "kind": _ASSEMBLED_CACHE_KIND,
        "start_date": start_date.isoformat(),
        "prefilter_tag": prefilter_tag,
        "include_mve": INCLUDE_MVE_MARKETS,
        "format": _SLICE_FORMAT_JSONL,
    }


@dataclass(frozen=True)
class AssemblyCounts:
    """
    What an assembly did with the records settled in its window: how many, how many the prefilter rejected.

    M9 of the 2026-09-24 7-day-run review. The fetch's count lines counted
    only the records that SURVIVED the prefilter while calling them "settled
    markets", and the backtester re-applied that same prefilter to the
    already-prefiltered corpus, so its "Eligibility prefilter: skipping" line
    read 0 on every production path. No line anywhere reported how many of
    that 7-day window's ~24.6M settled records the prefilter dropped (the
    review's estimate: 22,175,942 in the seven past-day slices plus the
    ~2.39M the frontier walked, against 7,274,215 assembled — a gap of about
    17.3M, an upper bound on the prefilter's share, since boundary duplicates
    fall in it too), and a prefilter that rejected nothing would have logged
    exactly the same lines as one that worked. This carries the missing
    numbers: fetch_all_settled_markets counts them during the assembly's
    FIRST walk (no extra walk — see _AssemblyTally), logs them per endpoint
    and in total, and stamps the
    total into the assembled cache's meta block so a later HIT can report it
    too; CorpusProvenance carries it to the backtester, whose prefilter line
    says the filter ran during assembly and quotes it.

    RECORDS, not markets. The day slices and the live windows
    deliberately overlap at their boundaries, so one market can arrive twice.
    Among records the prefilter passes, a repeat is caught by the ticker
    dedup and counted in `duplicates`; a repeat the prefilter rejects is
    counted in `rejected` once per arrival, since telling it apart would mean
    holding every rejected ticker — up to about 17M strings on that window,
    the residency SS-1 removed.

    Attributes:
        settled (int): Records whose settlement lies in the window (below the
            archive cutoff for the archive's sources), before the prefilter
            and before the ticker dedup.
        rejected (int): Of those, the ones the caller's prefilter rejected — at
            the assembly, or earlier by a fetch-time filter (the live
            frontier's sink, a sequential fallback) that had to drop them
            before they could be held. 0 when no prefilter was given.
        duplicates (int): Of those the prefilter passed, the ones not kept
            because their ticker was already kept or is blank.
    """
    settled: int
    rejected: int
    duplicates: int

    @property
    def kept(self) -> int:
        """
        Returns:
            int: The records kept — the assembled corpus's own count (for an
                assembly with a prefilter, its eligible markets).
        """
        return self.settled - self.rejected - self.duplicates


@dataclass
class _AssemblyTally:
    """
    The mutable counter behind AssemblyCounts, filled during the assembly's first walk.

    Two kinds of site add to it, and between them every in-window record is
    counted exactly once per arrival. The fetch-time filters — the live
    frontier's sink and the two sequential fallbacks, which must drop
    rejected records before they are held — add the records they dropped
    (note_rejected), since the assembly never sees those. Every other record
    reaches the assembly's first walk — the day slices unfiltered (the
    day-slice streams are no longer filtered at read-back), the fetch-time
    filters' survivors after their
    filter — and that walk, _count_assembled, counts it there
    (_assembled_records' `tally`). The second walk passes none, so nothing
    is counted twice.

    Attributes:
        settled (int): See AssemblyCounts.
        rejected (int): See AssemblyCounts.
        duplicates (int): See AssemblyCounts.
    """
    settled: int = 0
    rejected: int = 0
    duplicates: int = 0

    def note_rejected(self, count: int) -> None:
        """
        Record `count` in-window records a fetch-time filter rejected.

        Each is a record settled in the window that the prefilter rejected
        and that the assembly will never see, so it adds to both counts.

        Args:
            count (int): How many records the filter dropped (>= 0).
        """
        self.settled += count
        self.rejected += count

    def counts(self) -> AssemblyCounts:
        """
        Returns:
            AssemblyCounts: A frozen copy of the counts so far.
        """
        return AssemblyCounts(self.settled, self.rejected, self.duplicates)


def _total_counts(*parts: AssemblyCounts) -> AssemblyCounts:
    """
    Sum per-endpoint assembly counts into the whole assembly's.

    Args:
        *parts (AssemblyCounts): The archive's and the live endpoint's counts.

    Returns:
        AssemblyCounts: Their field-by-field sum.
    """
    return AssemblyCounts(
        sum(c.settled for c in parts),
        sum(c.rejected for c in parts),
        sum(c.duplicates for c in parts),
    )


def _describe_counts(counts: AssemblyCounts, prefilter_tag: str | None) -> str:
    """
    The parenthetical every count line carries: what became of the records not kept.

    One definition, so the per-endpoint lines, the assembly total and a cache
    hit's line can never word the same numbers two ways.

    Args:
        counts (AssemblyCounts): The counts being reported.
        prefilter_tag (str | None): The prefilter's tag, or None when none
            was applied (its clause is then left out: it rejected nothing).

    Returns:
        str: "<R> rejected by the prefilter <tag>, <D> duplicate or blank
            tickers", or just the duplicate clause with no prefilter.
    """
    parts = []
    if prefilter_tag is not None:
        parts.append(f"{counts.rejected} rejected by the prefilter {prefilter_tag}")
    parts.append(f"{counts.duplicates} duplicate or blank tickers")
    return ", ".join(parts)


def _kept_noun(prefilter_tag: str | None) -> str:
    """
    What the kept records ARE, for a count line: eligible markets, or settled markets.

    With a prefilter the corpus is the settled markets that passed it — the
    "eligible markets" of the backtester's lines and the dashboard's census —
    and calling them "settled markets" is the M9 misstatement. Without one,
    every kept record is a settled market.

    Args:
        prefilter_tag (str | None): The prefilter's tag, or None.

    Returns:
        str: "eligible markets" or "settled markets".
    """
    return "eligible markets" if prefilter_tag is not None else "settled markets"


def _parse_assembly_counts(raw: Any) -> AssemblyCounts | None:
    """
    Read an assembled cache's assembly_counts meta value, fail-safe by type.

    Args:
        raw (Any): The meta block's "assembly_counts" value — a dict of three
            non-negative ints as the writer records it, or anything else an
            older, damaged or hand-edited block might hold.

    Returns:
        AssemblyCounts | None: The counts, or None when the value is absent,
            not a dict, holds a non-int (a bool is refused: it is an int
            subclass), a negative, or counts that do not add up (more rejected
            and duplicate records than settled ones).
    """
    if not isinstance(raw, dict):
        return None
    values = [raw.get(key) for key in ("settled", "rejected", "duplicates")]
    if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in values):
        return None
    settled, rejected, duplicates = values
    if rejected + duplicates > settled:
        return None
    return AssemblyCounts(settled, rejected, duplicates)


@dataclass(frozen=True)
class CorpusProvenance:
    """
    What a settled-market corpus covers: when it was assembled, and under which archive cutoff.

    DR-13 and M3 of the 2026-09-24 7-day-run review. A cache hit makes zero
    network calls, so before this a hit was served with one "Loaded N" line:
    nothing said that the corpus stops at the moment it was assembled while
    the window nominally runs to today. This carries that out, with the
    archive cutoff the corpus was assembled under (reported as information;
    it decides nothing about which markets can trade, since a market settled
    after the cutoff is priced from the live candlestick endpoint) and, for a
    cache an earlier run brought up to date, when it was last assembled in
    full. fetch_all_settled_markets builds it from the assembled cache's meta
    block (_corpus_provenance) — the block it just wrote on a fresh assembly
    or an extension, the block its validating walk read on a same-day hit —
    and hangs it on the SettledCorpus it returns;
    backtester._prepare_candidates carries it to
    BacktestSweep.corpus_provenance, and the dashboard renders it under the
    Period line.

    Attributes:
        from_cache (bool): True when the corpus was served, as it was, from
            an assembled cache an earlier run wrote — today (UTC), or, with
            stale set, an earlier day; False when this call assembled it or
            extended it.
        assembled_at (datetime | None): When the corpus was assembled or last
            extended (UTC, tz-aware). It holds no market that settled after
            this moment; the live fetch that fed it ran in the minutes before
            (the 2026-09-17 cache's first record, from the
            newest-settled-first frontier, settled at 12:18:38Z against an
            assembled_at of 12:37:49Z), so its newest settlement sits at or
            somewhat before it. None when the meta block carries no readable
            timestamp.
        archive_cutoff (datetime | None): The archive cutoff (the
            /historical/cutoff endpoint's market_settled_ts) observed when the
            corpus was assembled or last extended (UTC, tz-aware). None when
            it was not recorded — every streamed cache written before P2 lacks
            it.
        assembly_counts (AssemblyCounts | None): How many records settled in
            the window and what became of the ones not kept — above all, how
            many the prefilter rejected (M9). Recorded at a FULL assembly only:
            an extension drops the old cache's partial frontier day, whose
            counts cannot be separated from the rest, so an extended cache
            records none (its own counts are logged instead). None too for
            every streamed cache written before these counts existed
            (including the 2026-09-17 cache on disk), for a block whose value
            is unreadable, and for a hit whose counts disagree with the
            records its validating walk counted
            (SettledCorpus.open_validated). Defaulted.
        full_assembly_at (datetime | None): When the corpus was last assembled
            IN FULL, for a cache later runs extended day by day
            (_extend_assembled_cache) — the moment its archive part was read;
            None for a corpus assembled in full at assembled_at (never
            extended), or a block whose value is unreadable. Defaulted.
        stale (bool): True when the corpus is a cache from an earlier UTC day
            served as it was, because bringing it up to date failed (the
            archive cutoff could not be read, or the extension raised — each
            with its WARNING). The reports must not call such a corpus
            today's: it holds nothing settled after assembled_at. Set only by
            _up_to_date_corpus; False (the default) everywhere else.
    """
    from_cache: bool
    assembled_at: datetime | None
    archive_cutoff: datetime | None
    assembly_counts: AssemblyCounts | None = None
    full_assembly_at: datetime | None = None
    stale: bool = False


def _window_start_ts(start_date: date) -> int:
    """
    The epoch second a backtest window opens: start_date's midnight, UTC.

    Args:
        start_date (date): The window's first day.

    Returns:
        int: Epoch seconds of start_date 00:00:00 UTC.
    """
    return int(datetime(start_date.year, start_date.month, start_date.day,
                        tzinfo=UTC).timestamp())


def _parse_assembled_at(raw: Any) -> datetime | None:
    """
    Read an assembled cache's assembled_at meta value as a UTC instant.

    Args:
        raw (Any): The meta block's "assembled_at" value — an ISO-8601 string
            as the writer records it, or anything else a damaged or hand-edited
            block might hold.

    Returns:
        datetime | None: The instant, tz-aware in UTC (a naive string is read
            as UTC, the writer's own zone), or None when the value is absent,
            not an ISO-8601 timestamp, or has no UTC instant a datetime holds.
    """
    if not isinstance(raw, str):
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    try:
        return parsed.astimezone(UTC)
    except OverflowError:
        # A stamp with no UTC instant inside datetime's range (a year-1 time
        # east of UTC, say): only a hand-edited block holds one
        return None


def _parse_archive_cutoff(raw: Any) -> datetime | None:
    """
    Read an assembled cache's archive_cutoff_ts meta value as a UTC instant.

    Args:
        raw (Any): The meta block's "archive_cutoff_ts" value — an int epoch
            second as the writer records it, or anything else a damaged or
            hand-edited block might hold.

    Returns:
        datetime | None: The cutoff, tz-aware in UTC, or None when the value
            is absent, not an int (read by TYPE: a bool is an int subclass and
            must not pass for an epoch), or outside the range a datetime holds.
    """
    epoch = _meta_epoch(raw)
    return None if epoch is None else datetime.fromtimestamp(epoch, tz=UTC)


def _meta_epoch(raw: Any) -> int | None:
    """
    Read an epoch-second meta value (archive_cutoff_ts, live_frontier_ts), range-checked.

    Args:
        raw (Any): The meta block's value — an int epoch second as the writer
            records it, or anything else a damaged or hand-edited block
            might hold.

    Returns:
        int | None: The value, or None when it is absent, not an int (read by
            TYPE: a bool is an int subclass and must not pass for an epoch),
            or outside the range a datetime holds — so nothing that decides
            on it, or prints it, can raise.
    """
    if not isinstance(raw, int) or isinstance(raw, bool):
        return None
    try:
        datetime.fromtimestamp(raw, tz=UTC)
    except (OverflowError, ValueError, OSError):
        return None
    return raw


def _corpus_provenance(meta: dict, *, from_cache: bool) -> CorpusProvenance:
    """
    Derive a corpus's provenance from its assembled cache's meta block.

    A pure function of the block, so a fresh assembly (the block just written)
    and a later hit (the block its validating walk read back) describe one
    corpus identically.

    Args:
        meta (dict): The assembled cache's whole meta block.
        from_cache (bool): Whether the corpus is being served from a cache an
            earlier run wrote (True) or was assembled by this call (False).

    Returns:
        CorpusProvenance: Unreadable or absent informational keys read as None
            (see CorpusProvenance); nothing here raises on a bad block.
    """
    assembled_at = _parse_assembled_at(meta.get("assembled_at"))
    return CorpusProvenance(
        from_cache=from_cache, assembled_at=assembled_at,
        archive_cutoff=_parse_archive_cutoff(meta.get("archive_cutoff_ts")),
        # Present only on a cache an extension wrote (informational too)
        full_assembly_at=_parse_assembled_at(meta.get("full_assembly_at")),
        # Informational like the two keys above: by TYPE, None when absent
        assembly_counts=_parse_assembly_counts(meta.get("assembly_counts")),
    )


def _describe_age(assembled_at: datetime, now: datetime) -> str:
    """
    How long ago a corpus was assembled, for a log line.

    Args:
        assembled_at (datetime): The assembly instant (tz-aware).
        now (datetime): The current instant (tz-aware).

    Returns:
        str: "N.N h ago" under two days, "N.N days ago" beyond, or a clock
            warning when the stamp is in the future.
    """
    seconds = (now - assembled_at).total_seconds()
    if seconds < 0:
        return "in the future by this host's clock"
    hours = seconds / 3600
    return f"{hours:.1f} h ago" if hours < 48 else f"{hours / 24:.1f} days ago"


def _log_cache_load(count: int, counts: AssemblyCounts | None, start_date: date,
                    prefilter_tag: str | None) -> None:
    """
    Log a cache hit's record count, naming what the records are and what their assembly rejected.

    The line used to read "Loaded N settled markets from cache" whatever the
    corpus held; with a prefilter those N are the ELIGIBLE markets, the
    settled markets that passed it, and nothing said how many did not (M9 of
    the 2026-09-24 review — run3 of the 7-day window logged "Loaded 7274215
    settled markets from cache" for a corpus assembled from about 24.6M
    settled records, the review's estimate). The noun now says which, and
    when the cache recorded its assembly counts they follow, as of assembly.
    A cache that records none — every legacy file, and every streamed one
    written before the counts existed — says so rather than letting the
    silence read as "nothing was rejected". Without a prefilter the line
    still begins "Loaded N settled markets from cache", as it always did.

    Args:
        count (int): The corpus's record count (its validated len()).
        counts (AssemblyCounts | None): The counts the cache recorded at
            assembly, or None.
        start_date (date): The window's first day.
        prefilter_tag (str | None): The prefilter tag the cache was assembled
            under (part of its identity), or None.
    """
    noun = _kept_noun(prefilter_tag)
    if counts is not None:
        logging.info(
            "Loaded %d %s from cache — as assembled, of %d records settled "
            "since %s (%s)",
            count, noun, counts.settled, start_date,
            _describe_counts(counts, prefilter_tag),
        )
    elif prefilter_tag is not None:
        logging.info(
            "Loaded %d %s from cache — the prefilter %s ran during its "
            "assembly, but this cache records no count of the records it "
            "rejected",
            count, noun, prefilter_tag,
        )
    else:
        logging.info("Loaded %d %s from cache", count, noun)


def _announce_cache_hit(path: Path, provenance: CorpusProvenance, now: datetime) -> None:
    """
    Say what a served assembled cache covers: one assembled earlier today (UTC), served as it is.

    DR-13 / M3: a hit used to log only "Loaded N settled markets from cache",
    so a repeat run silently replayed a corpus truncated at its assembly
    moment while every Period line said the window ran to today. Since a
    cache from an earlier UTC day is now brought up to date before it is
    served (_extend_assembled_cache), a hit — served with zero network calls
    — is a cache written earlier TODAY: it misses only what settled since,
    and the next UTC day's first run extends it. INFO, not WARNING — it fires
    on every healthy cached run. The archive cutoff at assembly is reported
    as information only.

    Args:
        path (Path): The cache file served.
        provenance (CorpusProvenance): What the cache says about itself.
        now (datetime): The current instant (tz-aware).
    """
    if provenance.assembled_at is None:
        when = "records no assembly time"
    else:
        when = (f"was assembled at {provenance.assembled_at:%Y-%m-%d %H:%M UTC} "
                f"({_describe_age(provenance.assembled_at, now)})")
    if provenance.full_assembly_at is not None:
        when += (f" by extending a full assembly of "
                 f"{provenance.full_assembly_at:%Y-%m-%d %H:%M UTC}")
    cutoff = ("the archive cutoff was not recorded" if provenance.archive_cutoff is None
              else f"archive cutoff at assembly: {provenance.archive_cutoff.date()}")
    logging.info(
        "Assembled cache %s %s, earlier today (UTC), so it is served as it is "
        "and holds no market settled after that moment; %s. The first run on "
        "a later UTC day extends it through that day, and --no-cache "
        "re-assembles it in full now",
        path.name, when, cutoff,
    )


def _assembled_records(
    sources: Iterable[tuple[Iterable[dict], int | None]],
    start_ts: int,
    prefilter: Callable[[dict], bool] | None,
    seen: set,
    tally: _AssemblyTally | None = None,
) -> Iterator[dict]:
    """
    Yield the assembled corpus: settlement window, prefilter, first-wins ticker dedup.

    The streaming form of the old in-memory `_merge` (SS-1), with identical
    semantics and order. For each (records, max_settle) source in turn, and
    each record in it in order, a record is yielded when its settlement_ts
    parses, lies in [start_ts, max_settle) (max_settle None = unbounded
    above), it passes `prefilter`, and its ticker is truthy and not yet in
    `seen` — the ticker is then added. The old code kept the same records in
    a ticker-keyed dict, whose insertion order is this yield order.

    Adjacent day slices and live windows deliberately overlap
    at their boundaries so no record can fall in a gap; the dedup collapses
    those overlaps without changing the set (tickers are unique per market).
    Only tickers are held — never a record — so a walk costs one set of
    ticker strings, not the corpus.

    Args:
        sources (Iterable[tuple[Iterable[dict], int | None]]): The record
            sources in precedence order, each with its exclusive settlement
            ceiling (the archive's cutoff_ts, or None for live).
        start_ts (int): Inclusive settlement floor, epoch seconds.
        prefilter (Callable[[dict], bool] | None): The caller's predicate.
            Applied here, before the dedup, as the single point where it is
            GUARANTEED for every source. The day slices reach it
            unfiltered; the live frontier and both sequential fallbacks were
            already filtered where they were fetched, because their records
            would otherwise be held — re-checking those is idempotent. And
            because it runs before the dedup, a phase dropping a record early
            can never change which record wins a ticker.
        seen (set): Tickers already yielded. Pass a FRESH set per walk; a walk
            split across calls (the assembly's first walk, archive then live)
            passes the same one to every call.
        tally (_AssemblyTally | None): When given — by the assembly's FIRST
            walk only, so nothing is counted twice — every in-window record
            is counted in `settled`, and each one not yielded in `rejected`
            (the prefilter refused it) or `duplicates` (its ticker was already
            yielded, or is blank) (M9). Records outside the window are not
            counted at all. Counting changes nothing that is yielded.

    Yields:
        dict: Each record of the assembled corpus, in first-wins order. They
            are the sources' own objects; the caller may patch them.
    """
    for records, max_settle in sources:
        for m in records:
            # The prefilter is tested BEFORE the window: both are pure
            # filters, so the order changes nothing that is yielded, and a
            # walk that counts nothing skips the settlement parse for every
            # record the prefilter rejects — most of them on the review's
            # 7-day window — exactly as the day-slice read-back filter used
            # to (M9 moved that filter here). A counting walk must still read
            # a rejected record's settlement, to know whether it lies in the
            # window at all.
            passed = prefilter is None or prefilter(m)
            if not passed and tally is None:
                continue
            settle = _iso_epoch(m.get("settlement_ts"))
            if settle is None or settle < start_ts:
                continue
            if max_settle is not None and settle >= max_settle:
                continue
            if tally is not None:
                tally.settled += 1
                if not passed:
                    tally.rejected += 1
                    continue
            ticker = m.get("ticker")
            if ticker and ticker not in seen:
                seen.add(ticker)
                yield m
            elif tally is not None:
                tally.duplicates += 1


def _assembly_identity(digest: int, m: dict) -> int:
    """
    Fold one assembled record into a running, order-sensitive identity hash.

    The assembly walks its sources twice — once to count and collect event
    tickers, once (after titles are resolved from those tickers) to write the
    cache — so the second walk must reproduce the first. The count alone
    cannot show that: a day slice rewritten between the walks could swap one
    record for another and keep the count, and a record whose event_ticker
    the first walk never saw would be written with a blank event title. This
    covers exactly what the walks must agree on: the ticker (which record,
    in which position) and the event_ticker (what the titles were resolved
    for). Being a hash, a change could in principle slip through on a
    collision; the guard is against a corpus that drifts, not an adversarial
    one.

    Args:
        digest (int): The identity so far (0 before the first record).
        m (dict): The next assembled record.

    Returns:
        int: The updated identity.
    """
    return hash((digest, m.get("ticker"), m.get("event_ticker")))


def _count_assembled(
    sources: Iterable[tuple[Iterable[dict], int | None]],
    start_ts: int,
    prefilter: Callable[[dict], bool] | None,
    seen: set,
    event_tickers: set[str],
    identity: int,
    tally: _AssemblyTally | None = None,
    title_wanted: Callable[[dict], bool] | None = None,
) -> tuple[int, int]:
    """
    The assembly's first walk over some sources: count, collect event tickers, fold identity.

    A function of its own rather than a loop in fetch_all_settled_markets so
    that no record outlives the walk: a loop variable left bound in the caller
    would keep the last record alive through the whole title resolution that
    follows. Nothing else is retained — only tickers (in `seen`) and event
    tickers. This is also the one walk that fills the assembly's counts (M9)
    — never a second walk. Counting is not free: the day slices now reach
    this walk unfiltered, so each record the prefilter rejects pays the
    settlement parse that decides whether it lies in the window, on top of at
    most two integer increments per record; each record the prefilter passes
    saves the read-back filter's prefilter call. See the prefilter gotcha in
    CLAUDE.md for what that nets to on real records.

    Args:
        sources (Iterable[tuple[Iterable[dict], int | None]]): As for
            _assembled_records.
        start_ts (int): Inclusive settlement floor, epoch seconds.
        prefilter (Callable[[dict], bool] | None): The caller's predicate.
        seen (set): The walk's ticker set, shared by both halves of the walk.
        event_tickers (set[str]): Extended in place with each yielded record's
            truthy event_ticker — the set titles are resolved for, exactly the
            old `{m.get("event_ticker") for m in selected.values() if ...}`.
        identity (int): The walk's running _assembly_identity so far.
        tally (_AssemblyTally | None): The endpoint's counts, extended in
            place with every in-window record of these sources (see
            _assembled_records). None counts nothing.
        title_wanted (Callable[[dict], bool] | None): Which yielded records'
            event tickers to collect. None (the default) collects every
            truthy one, as a full assembly does; an extension passes
            _needs_title for its old records, so only the titles still
            missing there are looked up again.

    Returns:
        tuple[int, int]: (records yielded by these sources, updated identity).
    """
    count = 0
    for m in _assembled_records(sources, start_ts, prefilter, seen, tally):
        count += 1
        identity = _assembly_identity(identity, m)
        event_ticker = m.get("event_ticker")
        if event_ticker and (title_wanted is None or title_wanted(m)):
            event_tickers.add(event_ticker)
    return count, identity


class SettledCorpus:
    """
    The settled-market corpus as a disk-backed, re-iterable sequence of market dicts.

    What fetch_all_settled_markets returns: a view of the assembled cache file
    settled_markets_<start_date>[_<tag>][_nomve].jsonl.gz. Every `for m in
    corpus` re-opens the file and yields its records one at a time, as fresh
    dicts, in the assembled order, so the corpus is never held in memory — the
    whole point of SS-1, where a 7-day window's eligible corpus measured
    7,260,952 records at 3,926 B/record (~28 GB) on a 16 GB host. The
    backtester walks it twice (_prepare_candidates' two passes); nothing needs
    random access.

    len() is the record count established when the corpus was built — written
    by the assembly, or counted by the full validation walk on a cache hit —
    and every walk is held to it: a walk that cannot read the file, or that
    ends on a different count, raises SettledCorpusError rather than returning
    a short or different corpus. A walk abandoned early is not checked.

    provenance says what the corpus covers — when it was assembled and the
    archive cutoff it was assembled under (CorpusProvenance) — read from the
    same meta block the identity check reads, so it describes exactly the
    file the walks stream; meta is that whole block, which an extension
    reads its starting points from (_extend_assembled_cache).
    """

    def __init__(self, path: Path, expect_meta: dict, count: int,
                 provenance: CorpusProvenance | None = None,
                 meta: dict | None = None):
        """
        Args:
            path (Path): The assembled cache file (jsonl-v1 framing).
            expect_meta (dict): The meta block every walk re-checks
                (_assembled_cache_meta()).
            count (int): How many records the file holds, as written or as
                validated. Callers other than fetch_all_settled_markets and
                open_validated should not construct this directly.
            provenance (CorpusProvenance | None): What the file's meta block
                says about when and under which archive cutoff it was
                assembled. Both production constructions (a fresh assembly and
                open_validated) pass it; None (the default) is a hand-built
                corpus, which a report then shows as "not recorded".
            meta (dict | None): The file's whole meta block — the identity
                keys plus the informational ones (assembled_at,
                archive_cutoff_ts, live_frontier_ts, ...). None (the default)
                reads as an empty block.
        """
        self._path = path
        self._expect_meta = dict(expect_meta)
        self._count = count
        self._provenance = provenance
        self._meta = dict(meta or {})

    @classmethod
    def open_validated(cls, path: Path, expect_meta: dict) -> "SettledCorpus | None":
        """
        Validate an assembled cache file by one full streaming walk and wrap it.

        All or nothing, like _day_store_load: the whole file must decode and
        its meta must match, or it is a cache miss. Nothing is retained by the
        walk — each record is parsed, counted and dropped.

        Args:
            path (Path): The assembled cache file to validate.
            expect_meta (dict): The meta block it must carry.

        Returns:
            SettledCorpus | None: A corpus over the file with its validated
                count and its provenance (from_cache=True, read from the meta
                block this same walk validated — never from a second read;
                its assembly_counts are dropped, with a WARNING, when they do
                not keep exactly the records the walk counted), or
                None when the file is absent (silently) or unreadable,
                truncated, damaged or written for a different request (with a
                WARNING naming the reason). Whether it is then served as it is,
                extended, or rebuilt is the caller's decision.
        """
        if not path.exists():
            return None
        count = 0
        # Filled by the validating walk itself with the file's whole meta
        # block — the informational assembled_at / archive_cutoff_ts included,
        # which expect_meta deliberately does not compare.
        meta: dict = {}
        try:
            for _record in _day_store_iter(path, expect_meta, meta_out=meta):
                count += 1
        except _SliceUnreadable as exc:
            logging.warning(
                "Corrupt or mismatched settled-market cache — treating as cache "
                "miss: %s", exc,
            )
            return None
        provenance = _corpus_provenance(meta, from_cache=True)
        counts = provenance.assembly_counts
        if counts is not None and counts.kept != count:
            # The counts describe some OTHER assembly (a hand-edited block, a
            # file rewritten under a stale meta line): quoting them beside
            # this corpus would report numbers that do not add up to it.
            # Dropped, not fatal — they are informational, never identity.
            logging.warning(
                "Settled-market cache %s records assembly counts that keep %d "
                "records, but it holds %d — ignoring those counts",
                path.name, counts.kept, count,
            )
            provenance = dc_replace(provenance, assembly_counts=None)
        return cls(path, expect_meta, count, provenance=provenance, meta=meta)

    def with_provenance(self, provenance: CorpusProvenance | None) -> "SettledCorpus":
        """
        The same corpus over the same file, saying something else about what it covers.

        Args:
            provenance (CorpusProvenance | None): The provenance to carry
                instead of this corpus's own.

        Returns:
            SettledCorpus: A new corpus with this one's file, identity block,
                count and meta block, and that provenance.
        """
        return SettledCorpus(self._path, self._expect_meta, self._count,
                             provenance=provenance, meta=self._meta)

    @property
    def path(self) -> Path:
        """
        Returns:
            Path: The assembled cache file this corpus streams.
        """
        return self._path

    @property
    def provenance(self) -> CorpusProvenance | None:
        """
        Returns:
            CorpusProvenance | None: When and under which archive cutoff the
                corpus was assembled, and whether it came from an earlier
                run's cache; None only on a hand-built corpus.
        """
        return self._provenance

    @property
    def meta(self) -> dict:
        """
        Returns:
            dict: A copy of the file's whole meta block, as validated or
                written ({} on a hand-built corpus).
        """
        return dict(self._meta)

    def __len__(self) -> int:
        """
        Returns:
            int: The corpus's record count (see the class docstring).
        """
        return self._count

    def __iter__(self) -> Iterator[dict]:
        """
        Stream the corpus once, in assembled order.

        Yields:
            dict: Compact market dicts, fresh objects on every walk.

        Raises:
            SettledCorpusError: If the file cannot be read (it disappeared, was
                damaged, or now carries a different meta block), or if a
                complete walk yields a different number of records than len().
        """
        walked = 0
        try:
            for record in _day_store_iter(self._path, self._expect_meta):
                walked += 1
                yield record
        except _SliceUnreadable as exc:
            raise SettledCorpusError(
                f"The assembled settled-market cache could not be read during a "
                f"walk ({exc}). Re-run the backtest: a damaged file fails its "
                f"validation next time and a missing one is simply absent, and "
                f"either way the corpus is rebuilt from the day slices."
            ) from exc
        if walked != self._count:
            raise SettledCorpusError(
                f"The assembled settled-market cache {self._path} yielded "
                f"{walked} records on this walk but held {self._count} when it "
                f"was opened — it was replaced or altered in between. Re-run "
                f"the backtest."
            )

    def __repr__(self) -> str:
        """
        Returns:
            str: The class name, path and record count.
        """
        return f"SettledCorpus({str(self._path)!r}, {self._count} records)"


def _needs_title(m: dict) -> bool:
    """
    Whether an already-assembled record's event title should be looked up again.

    An extension resolves titles for its new records, and for an old record
    only when its event title is still blank and its event is not a combo:
    a combo's title has no measured effect on which pairs form, and there are
    millions of them, while a non-combo's matters (TS-11) and there are few.

    Args:
        m (dict): A record read back from an assembled cache.

    Returns:
        bool: True when it has an event ticker, a blank event_title, and the
            event is not an MVE combo (_is_combo_event).
    """
    event_ticker = m.get("event_ticker")
    return (bool(event_ticker) and not m.get("event_title")
            and not _is_combo_event(event_ticker))


class _CorpusSplit:
    """
    One walk of an assembled cache, cut where its archive part ends.

    A full assembly writes its archive records first (every one settled
    before the archive cutoff it was assembled under), then its live records
    (the partial frontier day, then whole days, newest first). An extension
    must splice the new live records in between — after the archive part,
    before the old live days — to keep a fresh assembly's order, so each
    walk reads the old file ONCE through a shared iterator: head() yields the
    archive part and stops at the first record settled at or after the
    cutoff, holding it; tail() yields that record and everything after it.
    head() must be exhausted before tail() starts, which _assembled_records
    guarantees by walking its sources in turn.
    """

    def __init__(self, records: Iterable[dict], cutoff_ts: int):
        """
        Args:
            records (Iterable[dict]): The old cache's records, in file order
                (one walk of its SettledCorpus).
            cutoff_ts (int): The archive cutoff the old cache was assembled
                under, epoch seconds.
        """
        self._it = iter(records)
        self._cutoff = cutoff_ts
        self._held: dict | None = None

    def head(self) -> Iterator[dict]:
        """
        Yield the archive part: records up to the first one settled at or after the cutoff.

        Yields:
            dict: The old cache's leading records settled before the cutoff.
        """
        for m in self._it:
            settle = _iso_epoch(m.get("settlement_ts"))
            if settle is None or settle >= self._cutoff:
                self._held = m
                return
            yield m

    def tail(self) -> Iterator[dict]:
        """
        Yield the rest of the walk: the record head() stopped at, then every later one.

        Yields:
            dict: The old cache's remaining records, in file order.
        """
        if self._held is not None:
            held, self._held = self._held, None
            yield held
        yield from self._it


def _extend_assembled_cache(
    corpus: SettledCorpus,
    live_client,
    start_date: date,
    prefilter: Callable[[dict], bool] | None,
    prefilter_tag: str | None,
    cache_meta: dict,
    cutoffs: tuple[int, int],
    extend_from_ts: int,
) -> SettledCorpus:
    """
    Bring an assembled cache from an earlier UTC day up to now, in a fresh assembly's order.

    A full assembly holds no market settled after it was assembled, and a
    hit used to serve it unchanged at any age, so a backtest run on a later
    day silently stopped at the cache's day. This extends it instead: the
    live endpoint's settled days from `extend_from_ts` (the old cache's
    frontier day — the partial day it captured) through today are fetched
    (_fetch_live_phase; day slices on disk are reused, and only the missing
    days and today's frontier hit the API), and the corpus is re-assembled
    from three sources, walked twice like a fresh assembly (count and
    collect titles, then patch and write, with the same identity check):
      1. the old cache's archive part (its records settled before the cutoff
         it was assembled under — the leading records of the file);
      2. the new live records;
      3. the old cache's later records settled before `extend_from_ts` (its
         older live days), and any settled exactly at it; its records from
         the frontier day onward are dropped, since the complete day in (2)
         replaces them (one settled at the very midnight is kept because an
         endpoint that reads min_settled_ts as exclusive leaves it out of
         (2); when (2) does hold it, (2)'s copy wins the dedup).
    Each walk reads the old file once (_CorpusSplit). With the archive cutoff
    unchanged that is exactly a fresh assembly's record order; with a cutoff
    that moved forward (but not past `extend_from_ts`, which the caller
    checks) the same markets, the ones that migrated into the archive taken
    from the old live part. Titles are resolved for the new records, and for
    old records whose title is still blank (non-combo only, _needs_title);
    every other old record keeps the title it was assembled with.

    The extended cache replaces the old one atomically (_DayStreamWriter: the
    old file is read while the new one is written beside it, and the rename
    happens only after the second walk reproduced the first). Its meta block
    records the new assembly time, cutoff and frontier day, and when the
    corpus was last assembled in full ("full_assembly_at"); it records no
    assembly_counts — the dropped frontier day's counts cannot be separated
    from the rest — and the extension's own counts are logged instead.

    Args:
        corpus (SettledCorpus): The validated old cache. Its file is
            replaced, so the object must not be walked after this returns.
        live_client: KalshiClient from build_prod_live_client().
        start_date (date): The window's first day (the cache's own).
        prefilter (Callable[[dict], bool] | None): The predicate the cache
            was assembled under (part of its identity); applied to the new
            records and, idempotently, to the old ones.
        prefilter_tag (str | None): Its tag, for the log lines.
        cache_meta (dict): The cache's identity block (_assembled_cache_meta).
        cutoffs (tuple[int, int]): (the cutoff the old cache was assembled
            under, the current one), epoch seconds; the current one must lie
            between the old one and extend_from_ts (the caller checks).
        extend_from_ts (int): The UTC midnight from which every settled day
            is fetched again — the old cache's frontier day.

    Returns:
        SettledCorpus: The extended corpus over the rewritten cache file,
            from_cache False, its provenance read from the block just written.

    Raises:
        SettledCorpusError: If the old file or a day slice cannot be read
            during a walk, or the second walk does not reproduce the first;
            nothing is published.
        OSError: If the frontier spool or the new cache cannot be written.
        Exception: Whatever the live fetch raises once its retries are spent.
            The caller turns every exception into a WARNING and serves the
            old cache as it was.
    """
    old_meta = corpus.meta
    old_cutoff_ts, cutoff_ts = cutoffs
    start_ts = _window_start_ts(start_date)
    # Every market settled from here on comes from the new live fetch; the old
    # cache supplies only what settled before it
    ext_lo = max(extend_from_ts, start_ts)
    old_assembled = _parse_assembled_at(old_meta.get("assembled_at"))
    full_assembly = (_parse_assembled_at(old_meta.get("full_assembly_at"))
                     or old_assembled)
    logging.info(
        "Extending assembled cache %s (assembled %s%s; archive cutoff then %s, "
        "now %s) through today: every market settled from %s onward is fetched "
        "again from the live endpoint, reusing the day slices on disk",
        corpus.path.name,
        "at an unknown time" if old_assembled is None
        else f"{old_assembled:%Y-%m-%d %H:%M UTC}",
        "" if full_assembly is None or full_assembly == old_assembled
        else f", from a full assembly of {full_assembly:%Y-%m-%d %H:%M UTC}",
        datetime.fromtimestamp(old_cutoff_ts, tz=UTC).date(),
        datetime.fromtimestamp(cutoff_ts, tz=UTC).date(),
        datetime.fromtimestamp(ext_lo, tz=UTC).date(),
    )
    # Live-day slices wholly before the current cutoff are dead disk now
    _prune_stale_live_days(cutoff_ts)
    # The old records kept are those settled before the frontier midnight —
    # and AT it: the new fetch asks from that second, and an endpoint that
    # reads min_settled_ts as exclusive would leave a record settled exactly
    # then out of it. A copy the new fetch does hold comes first and wins the
    # dedup, so this never duplicates one
    tail_ceiling = ext_lo + 1
    now_ts = int(_utc_now().timestamp())
    new_tally = _AssemblyTally()
    # Persisted settled days plus today's frontier, as a fresh assembly fetches them
    live_records = _fetch_live_phase(live_client, ext_lo, now_ts, prefilter,
                                     tally=new_tally)
    try:
        # Walk A: count each part, collect the titles to resolve, fold identity
        split_a = _CorpusSplit(corpus, old_cutoff_ts)
        seen_a: set = set()
        event_tickers: set[str] = set()
        head_count, identity_a = _count_assembled(
            ((split_a.head(), None),), start_ts, prefilter, seen_a, event_tickers, 0,
            title_wanted=_needs_title)
        new_count, identity_a = _count_assembled(
            ((live_records, None),), start_ts, prefilter, seen_a, event_tickers,
            identity_a, tally=new_tally)
        tail_count, identity_a = _count_assembled(
            ((split_a.tail(), tail_ceiling),), start_ts, prefilter, seen_a, event_tickers,
            identity_a, title_wanted=_needs_title)
        del seen_a, split_a
        new_counts = new_tally.counts()
        kept_old = head_count + tail_count
        logging.info(
            "Extension of %s: kept %d of its %d %s (dropped %d settled on or "
            "after %s, the partial day it was assembled on, which the complete "
            "day replaces), added %d %s of %d records settled from then on (%s)",
            corpus.path.name, kept_old, len(corpus), _kept_noun(prefilter_tag),
            len(corpus) - kept_old, datetime.fromtimestamp(ext_lo, tz=UTC).date(),
            new_count, _kept_noun(prefilter_tag), new_counts.settled,
            _describe_counts(new_counts, prefilter_tag),
        )
        # The new records' titles, and the old records' still-blank non-combo ones
        logging.info("Resolving event titles for %d unique event_tickers", len(event_tickers))
        titles = _load_or_build_event_titles(live_client, event_tickers, use_cache=True)
        del event_tickers

        expected = kept_old + new_count
        new_meta = {
            **cache_meta,
            "assembled_at": _utc_now().isoformat(),
            "archive_cutoff_ts": cutoff_ts,
            "live_frontier_ts": _frontier_day_start(ext_lo, now_ts),
        }
        if full_assembly is not None:
            new_meta["full_assembly_at"] = full_assembly.isoformat()
        written = 0
        identity_b = 0
        # Walk B: the same three parts, a FRESH seen set, written as they come
        with _DayStreamWriter(corpus.path, new_meta) as writer:
            split_b = _CorpusSplit(corpus, old_cutoff_ts)
            seen_b: set = set()
            parts = (((split_b.head(), None), False), ((live_records, None), True),
                     ((split_b.tail(), tail_ceiling), False))
            for source, new in parts:
                for m in _assembled_records((source,), start_ts, prefilter, seen_b):
                    event_ticker = m.get("event_ticker") or ""
                    if new:
                        # As a full assembly patches every record, when titles resolved at all
                        if titles:
                            m["event_title"] = titles.get(event_ticker, "")
                    elif not m.get("event_title") and titles.get(event_ticker):
                        # An old record whose title was missing and is known now
                        m["event_title"] = titles[event_ticker]
                    identity_b = _assembly_identity(identity_b, m)
                    writer.write_record(m)
                    written += 1
            if written != expected or identity_b != identity_a:
                raise SettledCorpusError(
                    f"The extension of {corpus.path.name} did not reproduce "
                    f"itself: the first walk yielded {expected} markets and the "
                    f"second {written}"
                    + ("" if written != expected else " with a different order or "
                       "different tickers/event_tickers")
                    + ". A file changed between the two walks (another run may "
                      "be writing backtest_cache/ concurrently). Nothing was "
                      "written."
                )
            writer.commit()
        logging.info(
            "Extended assembled cache %s: %d %s, holding every market settled "
            "before %s",
            corpus.path.name, written, _kept_noun(prefilter_tag), new_meta["assembled_at"],
        )
        return SettledCorpus(
            corpus.path, cache_meta, written,
            provenance=_corpus_provenance(new_meta, from_cache=False), meta=new_meta,
        )
    finally:
        # Release the live phase's frontier spool, as a full assembly does
        _close_records(live_records)


def _served_stale(corpus: SettledCorpus) -> SettledCorpus:
    """
    The corpus, marked as an earlier day's cache served as it was because bringing it up to date failed.

    Args:
        corpus (SettledCorpus): The validated cache, from an earlier UTC day.

    Returns:
        SettledCorpus: The same file, its provenance's stale flag set (a
            corpus without a provenance is returned as it is).
    """
    provenance = corpus.provenance
    if provenance is None:
        return corpus
    return corpus.with_provenance(dc_replace(provenance, stale=True))


def _first_live_settle(corpus: SettledCorpus, cutoff_ts: int) -> int | None:
    """
    When the newest record of an assembled cache's live part settled: the first record at or after its cutoff.

    A full assembly writes its archive records first (each settled before the
    cutoff it read), then its live records, its frontier window first (that
    day's records, as the endpoint serves them), then whole days newest
    first; so the first record settled at or after the cutoff belongs to
    the frontier day — or, when that window held nothing, to an earlier day,
    which only makes an extension read one more stored day. Walks the
    archive part once and stops there.

    Args:
        corpus (SettledCorpus): The validated cache.
        cutoff_ts (int): The archive cutoff it was assembled under.

    Returns:
        int | None: That record's settlement, epoch seconds (whole); None
            when no record settled at or after the cutoff.
    """
    walk = iter(corpus)
    try:
        for m in walk:
            settle = _iso_epoch(m.get("settlement_ts"))
            if settle is not None and settle >= cutoff_ts:
                return int(settle)
        return None
    finally:
        # An abandoned walk: close its file now, not when it is collected
        close = getattr(walk, "close", None)
        if close is not None:
            close()


def _up_to_date_corpus(
    corpus: SettledCorpus,
    hist_client: Any,
    live_client,
    start_date: date,
    prefilter: Callable[[dict], bool] | None,
    prefilter_tag: str | None,
    cache_meta: dict,
    now: datetime,
) -> tuple[SettledCorpus, str] | None:
    """
    Decide what to do with a valid assembled cache: serve it, extend it, or rebuild.

    The one place the "always through the most recent available day" rule is
    decided for a cached corpus:
      * its frontier day (meta live_frontier_ts — the partial day it was
        assembled on) is today (UTC): served as it is, with zero network
        calls ("today");
      * an earlier day: one /historical/cutoff read; when the cutoff is the
        one it was assembled under, or has moved forward but not past its
        frontier day, it is extended through today (_extend_assembled_cache,
        "extended");
      * the cutoff has moved past its frontier day (markets it would need
        have migrated into the archive, which the live endpoint no longer
        serves), moved backward, or the cache records no cutoff or assembly
        time: None, and the caller re-assembles in full — each with an INFO
        line giving the reason;
      * the cutoff read, or the extension itself, fails: a WARNING naming the
        failure, and the cache is served as it was ("stale"), its provenance
        marked stale so the closing log line and the dashboard header do not
        call it today's (_served_stale).
    A cache written before live_frontier_ts existed counts as today's when it
    was assembled today, and otherwise is extended from the day BEFORE the
    one it was assembled on, or from the day its newest live record settled
    on when that is earlier (_first_live_settle, one walk of its archive
    part): an assembly that ran across one or more UTC midnights captured
    its frontier day only partly, and a day slice already on disk costs a
    read, not a fetch. A meta value no datetime can hold reads as
    unrecorded (_meta_epoch), so a damaged block costs a re-assembly, never
    an exception.

    Args:
        corpus (SettledCorpus): The validated cache.
        hist_client (Any): Client for the signed /historical/cutoff read.
        live_client: KalshiClient from build_prod_live_client().
        start_date (date): The window's first day.
        prefilter (Callable[[dict], bool] | None): The cache's predicate.
        prefilter_tag (str | None): Its tag.
        cache_meta (dict): The cache's identity block.
        now (datetime): The current instant (tz-aware, _utc_now()).

    Returns:
        tuple[SettledCorpus, str] | None: The corpus to serve and how it was
            obtained ("today", "extended" or "stale"), or None to rebuild.
    """
    meta = corpus.meta
    provenance = corpus.provenance
    name = corpus.path.name
    now_ts = int(now.timestamp())
    today_lo = now_ts - (now_ts % _DAY_SECONDS)
    assembled = provenance.assembled_at if provenance is not None else None
    if assembled is None:
        logging.info("Assembled cache %s records no assembly time, so it cannot be "
                     "brought up to date — re-assembling the window in full", name)
        return None
    # Range-checked: a value no datetime can hold reads as unrecorded, so a
    # damaged meta block costs a re-assembly, never an exception
    frontier = _meta_epoch(meta.get("live_frontier_ts"))
    assembled_lo: int | None = None
    if frontier is not None:
        if frontier == today_lo:
            return corpus, "today"
        if frontier > today_lo:
            logging.info("Assembled cache %s says its frontier day is after today "
                         "by this host's clock — re-assembling the window in full", name)
            return None
    else:
        assembled_ts = int(assembled.timestamp())
        assembled_lo = assembled_ts - (assembled_ts % _DAY_SECONDS)
        if assembled_lo == today_lo:
            return corpus, "today"
        if assembled_lo > today_lo:
            logging.info("Assembled cache %s says it was assembled after today by "
                         "this host's clock — re-assembling the window in full", name)
            return None
    old_cutoff = _meta_epoch(meta.get("archive_cutoff_ts"))
    if old_cutoff is None:
        logging.info("Assembled cache %s records no archive cutoff, so it cannot be "
                     "brought up to date — re-assembling the window in full", name)
        return None
    if frontier is None:
        # No frontier recorded (a cache written before the stamp existed):
        # extend from the day before it was assembled, or from the day its
        # newest live record settled on if that is earlier — an assembly can
        # run across more than one UTC midnight between its live fetch and
        # its stamp, and extending from too late a day would drop the rest
        # of its frontier day for good
        frontier = assembled_lo - _DAY_SECONDS
        newest = _first_live_settle(corpus, old_cutoff)
        if newest is not None:
            frontier = min(frontier, newest - newest % _DAY_SECONDS)
        if _meta_epoch(frontier) is None:
            logging.info("Assembled cache %s records no frontier day, and none can be "
                         "worked out from it — re-assembling the window in full", name)
            return None
    try:
        # The one network read a decision needs
        raw = _historical_get(hist_client, f"{_API_PREFIX}/historical/cutoff")
        cutoff_ts = int(datetime.fromisoformat(raw["market_settled_ts"]).timestamp())
    except Exception as exc:
        logging.warning(
            "Could not read the archive cutoff to extend assembled cache %s (%s): "
            "serving it as assembled at %s — it holds no market settled after that",
            name, _exception_summary(exc), f"{assembled:%Y-%m-%d %H:%M UTC}")
        return _served_stale(corpus), "stale"
    frontier_day = datetime.fromtimestamp(frontier, tz=UTC).date()
    if cutoff_ts < old_cutoff:
        logging.info(
            "The archive cutoff (%s) is earlier than the one assembled cache %s was "
            "assembled under (%s) — re-assembling the window in full",
            datetime.fromtimestamp(cutoff_ts, tz=UTC).date(), name,
            datetime.fromtimestamp(old_cutoff, tz=UTC).date())
        return None
    if cutoff_ts > frontier:
        logging.info(
            "The archive cutoff (%s) has moved past %s, the day assembled cache %s "
            "was last extended from: markets settled since then have migrated into "
            "the archive, which only a full assembly reads — re-assembling the "
            "window in full",
            datetime.fromtimestamp(cutoff_ts, tz=UTC).date(), frontier_day, name)
        return None
    try:
        # Splice in every market settled from its frontier day onward
        extended = _extend_assembled_cache(
            corpus, live_client, start_date, prefilter, prefilter_tag, cache_meta,
            (old_cutoff, cutoff_ts), frontier)
    except Exception as exc:
        logging.warning(
            "Could not extend assembled cache %s (%s): serving it as assembled at "
            "%s — it holds no market settled after that",
            name, _exception_summary(exc), f"{assembled:%Y-%m-%d %H:%M UTC}",
            exc_info=True)
        return _served_stale(corpus), "stale"
    return extended, "extended"


def _retire_legacy_cache(legacy_path: Path, superseded_by: Path) -> None:
    """
    Delete a legacy settled_markets_*.json once a rebuild of the same identity is committed.

    Before SS-1 a rebuild (a --no-cache run, or a miss) overwrote
    settled_markets_<stem>.json in place, so an out-of-date assembly could
    never come back. The rebuild now lands in settled_markets_<stem>.jsonl.gz
    instead, and a legacy file left beside it would be served again the
    moment that file went missing (until 2026-09-24 this repo lived in
    iCloud-synced ~/Documents, where iCloud reverted a committed rename; a
    failed delete or a hand restore can do the same) — silently,
    and typically after the very rebuild meant to replace it (the BS-02 and
    subtitle-drift remedies both end in one). Deleting it once the new file
    is committed restores the old invariant: a rebuild of an identity
    replaces that identity's previous assembly. It destroys exactly what the
    old overwrite destroyed, and only after the replacement is safely on disk.

    Args:
        legacy_path (Path): The legacy cache for this request's name stem.
        superseded_by (Path): The streamed cache just committed for it; named
            in the log lines.
    """
    try:
        legacy_path.unlink()
    except FileNotFoundError:
        # The common case: no legacy cache of this identity ever existed.
        return
    except OSError as exc:
        logging.warning(
            "Could not remove the superseded legacy settled-market cache %s (%s). "
            "Delete it by hand: it holds an older assembly of this same request, "
            "and it would be served again if %s ever went missing.",
            legacy_path, exc, superseded_by.name,
        )
        return
    logging.info(
        "Removed the superseded legacy settled-market cache %s: this run "
        "rebuilt the same request into %s, which replaces it.",
        legacy_path.name, superseded_by.name,
    )


def fetch_all_settled_markets(
    hist_client: Any,
    live_client,
    start_date: date,
    use_cache: bool = True,
    prefilter: Callable[[dict], bool] | None = None,
    prefilter_tag: str | None = None,
) -> SettledCorpus:
    """
    Fetch all settled Kalshi markets from start_date onward, as a re-iterable corpus of dicts.

    Uses two complementary API endpoints to get full coverage:
    - /historical/markets — settled markets archived before the API cutoff
      timestamp. Fetched as parallel per-created-day slices via synthesized
      pagination cursors (see _fetch_archive_phase), every created-day from
      the archive's first (ARCHIVE_FIRST_CREATED_DATE) so that long-lived
      markets created before start_date are read too; falls back to the
      original sequential walk if the cursor format ever drifts.
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

    Assembly is STREAMED and the corpus is never held in memory (SS-1). A
    7-day window's past days held 18,061,549 fetched records of which
    7,260,952 passed the backtester's prefilter, at 3,926 B/record (~28 GB) on
    a 16 GB host, so the old assembly — a ticker-keyed dict of every selected
    record, then a list copy, then one whole-list json.dumps — could never
    finish. The phases now return lazy views (day slices are re-read off disk
    per walk, and so is the live frontier, spooled to an anonymous temporary
    file by _FrontierSpool and released when assembly ends; only a
    sequential fallback's result is a list), and one generator,
    _assembled_records, reproduces the old
    `_merge` exactly: sources in the old order (archive day slices
    newest-first, bounded above by cutoff_ts; then live:
    frontier, then live day slices newest-first, unbounded above),
    settlement >= start_ts, the prefilter, and first-wins ticker dedup. It is
    walked TWICE. Walk A counts the records the "Historical endpoint" and
    "Live endpoint" lines report (the same kept numbers as before) and
    collects the unique event_tickers for title resolution; it also counts,
    per endpoint, every record settled in the window and what the prefilter
    and the dedup removed from them (M9 of the 2026-09-24 review), which
    those lines and the closing "Assembled N ... of M records settled" line
    now report beside the kept number. For that the day slices reach walk A
    UNFILTERED — they used to be prefiltered as they were read back, which
    saved no memory and hid every rejection from any count — while the
    three filters that must drop records before they are held (the live
    frontier's sink and the two sequential fallbacks) report their own
    rejections into the same counts. Walk B counts nothing: it patches
    event_title exactly as before and writes every record straight into the
    assembled cache, which the returned SettledCorpus then streams. Walk B must
    reproduce walk A (same count, same order-sensitive identity over ticker
    and event_ticker) or nothing is published and SettledCorpusError is
    raised. A day slice that cannot be read during either walk raises
    SettledCorpusError — deliberately NOT the sequential fallback the old
    eager assembly took, which would hold the whole range in memory and could
    not un-yield records already consumed (see _DaySliceStream).

    The assembled result is streamed into a gzipped JSON-lines cache file
    named settled_markets_<start_date>[_<prefilter_tag>][_nomve].jsonl.gz
    (the day slices' "jsonl-v1" framing, written atomically through
    _DayStreamWriter), so subsequent backtests skip fetching entirely. Every
    component of that name is part of the result's identity, and its meta
    block repeats them (_assembled_cache_meta): the prefilter tag because a
    filtered result is a strict subset, and the _nomve marker because
    INCLUDE_MVE_MARKETS changes which markets are fetched at all (DR-57). Only
    the False case is marked — every assembled cache already on disk was
    built with MVE included, so the default (True) filename is unchanged and
    no existing cache is orphaned. On use_cache=True, when the .jsonl.gz file
    exists it is validated by one full streaming walk (all or nothing: any
    decode error, truncation or meta mismatch is a WARNING and a miss), and
    then brought up to date (_up_to_date_corpus): served as it is, with ZERO
    network calls, when it was assembled today (UTC); extended through today
    when it is from an earlier day (_extend_assembled_cache: one
    /historical/cutoff read and the live endpoint's settled days from its
    frontier day onward); re-assembled in full when it cannot be extended
    (the cutoff moved past its frontier day, or it records no cutoff or
    assembly time); and served as it was, with a WARNING, when the extension
    fails. A miss REBUILDS: it never falls through to a legacy file, which
    the streamed cache's own commit superseded. A LEGACY
    settled_markets_<...>.json cache of the same identity (the single JSON
    document every run before SS-1 wrote) is no longer served at all — it
    records nothing an extension could start from — so its window is
    re-assembled in full, and once that is committed the legacy file is
    deleted (_retire_legacy_cache, with an INFO line). New runs never write
    the legacy format. Pass use_cache=False (--no-cache) to re-assemble in
    full; thanks to the day stores that costs the frontier day plus any
    archive or live day not stored or no longer valid. A fresh result is
    always written regardless of use_cache.

    Every corpus is announced, never silent (DR-13 and M3 of the 2026-09-24
    review). The streamed cache's meta block records, besides its identity,
    informational keys the identity check never compares: assembled_at (the
    corpus holds no market settled after it), archive_cutoff_ts (the cutoff
    it was assembled under), live_frontier_ts (the first second of its
    partial frontier day, from which a later day's run extends it),
    assembly_counts on a full assembly (how many records settled in the
    window and how many of them the prefilter rejected, M9) and
    full_assembly_at on an extended one (when it was last assembled in
    full). A served cache's count line names its records "eligible markets"
    when a prefilter ran and quotes those counts as of assembly
    (_log_cache_load), and a same-day hit also logs its assembly time and
    the cutoff at assembly (_announce_cache_hit). An extension logs what it
    kept, dropped and added.

    Args:
        hist_client (Any): Authenticated KalshiClient from build_historical_client().
        live_client: KalshiClient from build_prod_live_client() for recent settlements.
        start_date (date): Earliest settlement date to include. Markets that settled
            before this date are skipped even if the API returns them.
        use_cache (bool): If True (default), use the assembled per-start_date
            cache when there is one: served as it is when assembled today,
            otherwise extended through today (or re-assembled in full when
            it cannot be). If False, always re-assemble from the API + day
            stores. Either way the result is saved to disk.
        prefilter (Callable[[dict], bool] | None): Optional per-record
            predicate; records failing it are dropped during assembly and
            never reach the returned corpus or the assembled cache. Intended for
            a filter the caller would apply immediately anyway (the backtester
            passes _can_ever_enter), which makes it result-neutral while
            keeping peak memory and cache size proportional to the markets
            actually usable rather than to everything Kalshi ever settled. The
            per-day slice FILES are never filtered — they are shared across
            start dates and must stay complete. How many in-window records it
            rejected is counted during the assembly's first walk and logged
            beside the kept count (M9); it must be pure and thread-safe,
            since the live frontier applies it on a worker thread.
        prefilter_tag (str | None): Short name for prefilter's semantics; becomes
            part of the assembled cache's filename so a cache built under one
            predicate is never served to a caller expecting another. Required
            when prefilter is given, and forbidden otherwise.

    Raises:
        ValueError: If exactly one of prefilter / prefilter_tag is provided.
        OSError: If the live frontier cannot be spooled (e.g. a full disk) or
            the assembled cache cannot be written; nothing is published.
        SettledCorpusError: If a day slice (or the frontier spool) cannot be
            read during either assembly walk, or the second walk does not
            reproduce the first; nothing is published in either case. Walking the returned
            SettledCorpus later raises it too if the cache file cannot be
            read or no longer holds len() records.

    Returns:
        SettledCorpus: A re-iterable corpus of market dicts that supports
            len(), streaming the assembled .jsonl.gz cache (fresh dicts on
            every walk). Each record has the keys: ticker,
            event_ticker, event_title, title, subtitle, result ("yes" |
            "no"), yes_ask_dollars, no_ask_dollars, yes_bid_dollars,
            open_time, close_time (ISO str), settlement_ts (ISO str), status,
            price_level_structure, price_ranges, exchange_index. Only includes
            markets with a non-null settlement_ts and a binary result; tickers
            are unique. See _market_to_dict for the per-key notes (subtitle
            now falls back to yes_sub_title; price_level_structure/
            price_ranges are unread groundwork that older cache records lack
            entirely). Consumers must only iterate it (as many times as they
            like) and take its len(); nothing indexes it. It also carries
            .provenance (CorpusProvenance: from_cache, assembled_at,
            archive_cutoff, assembly_counts, full_assembly_at), which the
            backtester carries to the dashboard header and quotes on its
            prefilter line.
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
    stem = f"settled_markets_{start_date.isoformat()}{suffix}{mve_suffix}"
    # The streamed format every run writes since SS-1, and the single JSON
    # document every earlier run wrote — same identity, same name stem, so the
    # DR-57 and prefilter-tag semantics above carry over unchanged.
    cache_path = CACHE_DIR / f"{stem}.jsonl.gz"
    legacy_cache_path = CACHE_DIR / f"{stem}.json"
    cache_meta = _assembled_cache_meta(start_date, prefilter_tag)
    if use_cache:
        # Every backtest runs through the most recent available day: a cache
        # assembled today (UTC) is served with ZERO network calls; one from an
        # earlier day is extended through today, or re-assembled in full when
        # it cannot be (pinned by TestCorpusProvenance and TestCacheExtension).
        now = _utc_now()
        if cache_path.exists():
            # The streamed cache, validated by one full walk; the same walk
            # reads the meta block its provenance and frontier day come from.
            corpus = SettledCorpus.open_validated(cache_path, cache_meta)
            if corpus is not None:
                # Serve it, extend it, or rebuild (None)
                decided = _up_to_date_corpus(corpus, hist_client, live_client, start_date,
                                             prefilter, prefilter_tag, cache_meta, now)
                if decided is not None:
                    served, how = decided
                    if how != "extended":
                        # What the corpus is, and what its prefilter rejected
                        # as of assembly (M9) — from its own meta block
                        _log_cache_load(len(served), served.provenance.assembly_counts,
                                        start_date, prefilter_tag)
                    if how == "today":
                        # Coverage line: assembly time, cutoff at assembly
                        _announce_cache_hit(cache_path, served.provenance, now)
                    return served
            # Invalid (its WARNING is already logged) or not to be extended
            # (its INFO line says why): a miss that REBUILDS, never a
            # fall-through to a legacy file, which a streamed cache of this
            # identity superseded when it was committed.
        elif legacy_cache_path.exists():
            # A legacy settled_markets_*.json records neither its archive
            # cutoff nor where its live part ends, so it cannot be brought up
            # to date: re-assembled in full below (it is not even read), and
            # the rebuild's commit deletes it (_retire_legacy_cache).
            logging.info(
                "Legacy assembled cache %s cannot be brought up to date (it records "
                "no archive cutoff and no frontier day) — re-assembling the window "
                "in full in the streamed format, which then replaces it",
                legacy_cache_path.name)

    # Convert start_date to a unix timestamp for filtering individual market records
    start_ts = _window_start_ts(start_date)

    # Get the cutoff timestamp that divides historical archive from live endpoint coverage
    cutoff    = _historical_get(hist_client, f"{_API_PREFIX}/historical/cutoff")
    cutoff_ts = int(datetime.fromisoformat(cutoff["market_settled_ts"]).timestamp())

    # A window may start at or after the cutoff: its markets settled after
    # it, and the backtester prices them from Kalshi's live candlestick
    # endpoint (fetch_candlesticks' 404 fallback), so nothing here is
    # structurally 0-trade any more. The cutoff is stamped into the assembled
    # cache below, as information a later hit reports.

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
    # What the prefilter and the dedup did to each endpoint's in-window
    # records (M9): filled by walk A below and by the fetch-time filters inside
    # the phases (the only filters whose rejections walk A never sees).
    archive_tally = _AssemblyTally()
    live_tally = _AssemblyTally()
    # Sharded parallel fetch with day-level disk reuse; sequential on fallback.
    # Its day slices come back unfiltered; only a sequential fallback applies
    # the prefilter itself, and it reports what it dropped into archive_tally.
    day_records = _fetch_archive_phase(
        hist_client, start_ts, cutoff_ts, hist_kwargs, prefilter, tally=archive_tally,
    )

    # Assembly: settlement-window filter, prefilter and first-wins ticker
    # dedup, streamed (_assembled_records) rather than collected into a dict
    # of every selected record (SS-1). The sources, their order and their
    # ceilings are the old _merge calls': the archive day slices (every
    # created-day from the archive's first, newest first), below cutoff_ts;
    # then live, unbounded above.
    archive_sources = ((day_records, cutoff_ts),)

    # Walk A, archive half: count what the old `len(selected)` reported and
    # collect the event_tickers titles are resolved for, and count every
    # in-window record into the archive's tally (M9). One `seen` set spans
    # both halves of this walk (the dedup is global); only tickers are held.
    seen_a: set = set()
    event_tickers: set[str] = set()
    archive_count, identity_a = _count_assembled(
        archive_sources, start_ts, prefilter, seen_a, event_tickers, 0,
        tally=archive_tally,
    )
    archive_counts = archive_tally.counts()
    # The kept count first, as before, then what the window held (M9): the
    # settled records, and what the prefilter and the dedup removed from them
    logging.info(
        "Historical endpoint: %d %s of %d records settled in the window before "
        "the archive cutoff (%s)",
        archive_count, _kept_noun(prefilter_tag), archive_counts.settled,
        _describe_counts(archive_counts, prefilter_tag),
    )

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
    # Windowed parallel fetch with settled-day disk reuse; sequential on
    # fallback. Past days come back unfiltered; the frontier (and a fallback)
    # apply the prefilter as pages arrive and report what they dropped.
    now_ts = int(_utc_now().timestamp())
    live_records = _fetch_live_phase(live_client, live_min_ts, now_ts,
                                     prefilter, tally=live_tally)
    try:
        live_sources = ((live_records, None),)
        # Walk A, live half: same `seen` set, so a live record whose ticker the
        # archive already supplied is dropped exactly as the old merge dropped it.
        live_count, identity_a = _count_assembled(
            live_sources, start_ts, prefilter, seen_a, event_tickers, identity_a,
            tally=live_tally,
        )
        live_counts = live_tally.counts()
        logging.info(
            "Live endpoint: %d %s of %d recently settled records in the window (%s)",
            live_count, _kept_noun(prefilter_tag), live_counts.settled,
            _describe_counts(live_counts, prefilter_tag),
        )
        # The whole assembly's counts, stamped into the cache below so a
        # later hit can report them without a fetch
        assembly_counts = _total_counts(archive_counts, live_counts)
        # Walk A is done; its ticker set is the one piece of it worth releasing
        # before titles are resolved (walk B builds its own).
        del seen_a

        # ── Attach event titles ───────────────────────────────────────────────
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
        logging.info("Resolving event titles for %d unique event_tickers", len(event_tickers))
        # Resolve event_ticker -> title so grouping keys match the live scanner's
        titles = _load_or_build_event_titles(live_client, event_tickers, use_cache=use_cache)
        del event_tickers

        # Walk B: the same sources in the same order with a FRESH `seen` set, so
        # it yields walk A's records again. Each is patched and streamed straight
        # into the assembled cache — nothing of the corpus is held — through the
        # day slices' atomic writer: the file becomes visible only on commit(),
        # and leaving this block by any exception deletes the temp file, so a
        # failed or disagreeing walk can never publish a partial cache.
        #
        # Always persist a fresh fetch — use_cache only controls whether reads are
        # allowed to come from disk. Gating the save on use_cache meant --no-cache
        # runs (whose whole point is to refresh stale data) never updated the file
        # the very next default run would load.
        expected = archive_count + live_count
        written = 0
        identity_b = 0
        # The identity block plus INFORMATIONAL keys a later run reads back
        # (CorpusProvenance, _up_to_date_corpus) and the identity check never
        # compares: when the corpus was assembled (it holds nothing settled
        # after that), the archive cutoff it was assembled under, and the
        # first second of its partial frontier day, from which a later day's
        # run extends it.
        assembled_meta = {
            **cache_meta,
            "assembled_at": _utc_now().isoformat(),
            "archive_cutoff_ts": cutoff_ts,
            "live_frontier_ts": _frontier_day_start(live_min_ts, now_ts),
            # Informational too (M9): how many records settled in the window
            # and what the prefilter and the dedup removed, as of assembly
            "assembly_counts": {
                "settled": assembly_counts.settled,
                "rejected": assembly_counts.rejected,
                "duplicates": assembly_counts.duplicates,
            },
        }
        with _DayStreamWriter(cache_path, assembled_meta) as writer:
            for m in _assembled_records(archive_sources + live_sources, start_ts,
                                        prefilter, set()):
                if titles:
                    # Day-store records were normalized before titles existed, so
                    # the event_title field is patched in here rather than at
                    # _market_to_dict time — only when titles resolved at all,
                    # exactly as before.
                    m["event_title"] = titles.get(m.get("event_ticker") or "", "")
                identity_b = _assembly_identity(identity_b, m)
                writer.write_record(m)
                written += 1
            if written != expected or identity_b != identity_a:
                # A day slice changed between the walks (rewritten or pruned by a
                # concurrent run). The titles were resolved for walk A's records,
                # so walk B's cannot be trusted to carry the right ones; raising
                # here aborts the writer and publishes nothing.
                raise SettledCorpusError(
                    f"The settled-market assembly did not reproduce itself: the "
                    f"first walk yielded {expected} markets and the second "
                    f"{written}"
                    + ("" if written != expected else " with a different order or "
                       "different tickers/event_tickers")
                    + ". A day slice changed between the two walks (another run "
                      "may be writing or pruning backtest_cache/ concurrently). "
                      "Nothing was cached; re-run the backtest."
                )
            # Named for what it counts (M9): the kept records are the eligible
            # markets when a prefilter ran, beside the settled records they
            # were assembled from — no longer "Total settled markets".
            logging.info(
                "Assembled %d %s of %d records settled since %s (%s)",
                written, _kept_noun(prefilter_tag), assembly_counts.settled,
                start_date, _describe_counts(assembly_counts, prefilter_tag),
            )
            writer.commit()
        # A committed rebuild of this identity supersedes any legacy .json of
        # the same stem, exactly as the old code's rebuild overwrote it; left
        # on disk it would be served again whenever the .jsonl.gz goes missing.
        _retire_legacy_cache(legacy_cache_path, cache_path)
        # Handed back as a stream over the file just written, never as a list,
        # carrying the provenance derived from the block just written — the
        # same derivation a later hit applies to the block it reads back.
        return SettledCorpus(
            cache_path, cache_meta, written,
            provenance=_corpus_provenance(assembled_meta, from_cache=False),
        )
    finally:
        # The live phase's frontier spool (an anonymous temporary file inside
        # its _RecordChain) is walked only by the two assembly walks above;
        # release it now, on success and failure alike, rather than whenever
        # the chain happens to be collected. A no-op for a fallback's list.
        _close_records(live_records)


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


def _candle_count(candle: dict) -> float | None:
    """
    Read the contracts traded in one candle's hour.

    Takes the fixed-point "volume_fp" when the candle has it, else "volume"
    (a number or a numeric string). A value that is present but unreadable is
    not replaced by the other key.

    Args:
        candle (dict): One raw candle from the API.

    Returns:
        float | None: A finite count of zero or more. None when the candle has
            neither key, or the value is not a readable non-negative number.
            Never raises.
    """
    if not isinstance(candle, dict):
        return None
    for key in ("volume_fp", "volume"):
        raw = candle.get(key)
        if raw is None or raw == "":
            continue
        if isinstance(raw, bool):
            return None
        try:
            count = float(raw)
        except (TypeError, ValueError, OverflowError):
            return None
        # "+ 0.0" turns a negative zero ("-0") into a plain 0.0
        return count + 0.0 if math.isfinite(count) and count >= 0 else None
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
    backtester._candles_at_or_before (and its reference,
    _candle_at_or_before), which stop at the first candle past each moment
    and so need an ascending series.

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


def _candle_endpoints(ticker: str, series: str | None,
                      live_first: bool) -> list[tuple[str, str]]:
    """
    The candlestick endpoints to ask for one market, in the order to ask them.

    Kalshi serves a market's candles from one of two places. A market that
    settled before the archive cutoff is in the archive, at
    /historical/markets/{ticker}/candlesticks; a market that settled after it
    (the last couple of months) is still on the live API, at
    /series/{series_ticker}/markets/{ticker}/candlesticks, and the archive
    path answers 404 for it. The live path needs the market's series, so
    without one only the archive is asked.

    Args:
        ticker (str): The market's ticker.
        series (str | None): Its series ticker (historical.series_ticker of its
            event ticker), or None/"" when unknown.
        live_first (bool): Whether to ask the live endpoint first (the market
            settled at or after the archive cutoff).

    Returns:
        list[tuple[str, str]]: (name, path) pairs, "historical" and/or "live",
            first to ask first.
    """
    archive = ("historical", f"{_API_PREFIX}/historical/markets/{ticker}/candlesticks")
    if not isinstance(series, str) or not series:
        return [archive]
    live = ("live", f"{_API_PREFIX}/series/{series}/markets/{ticker}/candlesticks")
    return [live, archive] if live_first else [archive, live]


def _fetch_candle_pages(client: Any, path: str, windows: list[tuple[int, int]],
                        rate_limit_sleep: float, progress: dict) -> tuple[list[dict], int, int]:
    """
    Fetch every request of one window from one candlestick endpoint and parse it.

    Each request is the same retried read-only GET (_historical_get); the
    candles of a paged window are put in timestamp order with the overlaps'
    repeats dropped (_merge_candle_pages). Any failed request raises, so the
    window is all-or-nothing.

    Args:
        client (Any): Authenticated KalshiClient.
        path (str): The endpoint path (_candle_endpoints).
        windows (list[tuple[int, int]]): The window's requests (_candle_request_windows).
        rate_limit_sleep (float): Seconds to sleep after each request.
        progress (dict): Its "request" key is set to the 1-based number of the
            request in flight, for the caller's failure line.

    Returns:
        tuple[list[dict], int, int]: The parsed candles ("ts", "yes_ask_close",
            "no_ask_close", "volume"), how many raw candles came back and how
            many of them could not be parsed.

    Raises:
        Exception: Whatever a request raises (an ApiException carries .status).
    """
    candles: list[dict] = []
    raw_count = 0
    dropped = 0
    progress["request"] = 0
    for request_open, request_close in windows:
        progress["request"] += 1
        # Raw signed GET — the pinned SDK has no historical_api module and
        # its candlestick models predate the current wire format anyway.
        # Read-only, so _historical_get's api_call_with_retry backoff
        # applies per request.
        data = _historical_get(
            client, path,
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
                # NO ask ≈ 1 - YES bid (binary market complement); clamp to
                # avoid 0 or 1. The top is CANDLE_NO_ASK_CEILING, the value
                # usable_candle_ask refuses as no quote
                no_ask = 1.0 - yes_bid
                candles.append({
                    "ts": c["end_period_ts"],
                    "yes_ask_close": yes_ask,
                    "no_ask_close": max(0.01, min(CANDLE_NO_ASK_CEILING, no_ask)),
                    # Contracts traded in this hour; None when not readable
                    "volume": _candle_count(c),
                })
            except (ValueError, TypeError, AttributeError, KeyError):
                dropped += 1
        # Rate limit: sleep briefly after each call to avoid 429 responses
        time.sleep(rate_limit_sleep)
    if len(windows) > 1:
        # Paged: put the requests' candles in timestamp order and drop the
        # repeats the one-period overlaps return twice.
        candles = _merge_candle_pages(candles)
    return candles, raw_count, dropped


def _fetch_candles_from_endpoints(client: Any, endpoints: list[tuple[str, str]],
                                  windows: list[tuple[int, int]], rate_limit_sleep: float,
                                  progress: dict, tried: list[str]
                                  ) -> tuple[list[dict], int, int]:
    """
    Fetch one market's candle window from the first endpoint that holds it.

    The endpoints are asked in order (_candle_endpoints). A 404 means that
    endpoint does not hold the market (the archive cutoff sits on the other
    side of its settlement), so the next one is asked for the whole window;
    any other failure, or a 404 from the last endpoint, is final. Each
    endpoint's window is all-or-nothing (_fetch_candle_pages), so pages one
    endpoint did serve are never joined to another's. fetch_candlesticks
    (the backtest, cached) and recent_candles (live selling, never cached)
    both fetch through it.

    Args:
        client (Any): Authenticated KalshiClient.
        endpoints (list[tuple[str, str]]): (name, path) pairs to ask, first
            to ask first (_candle_endpoints).
        windows (list[tuple[int, int]]): The window's requests (_candle_request_windows).
        rate_limit_sleep (float): Seconds to sleep after each request; a
            failed request that moves on to the next endpoint sleeps once too.
        progress (dict): Passed to _fetch_candle_pages, which keeps the
            1-based number of the request in flight under "request".
        tried (list[str]): Each endpoint's name is appended as it is asked,
            for the caller's failure line.

    Returns:
        tuple[list[dict], int, int]: _fetch_candle_pages' result from the
            endpoint that served the window.

    Raises:
        Exception: What the failed request raised: a failure other than a
            404, or the last endpoint's 404 (an ApiException carries
            .status).
        ValueError: When there is no endpoint to ask.
    """
    for attempt, (name, path) in enumerate(endpoints):
        tried.append(name)
        try:
            return _fetch_candle_pages(client, path, windows, rate_limit_sleep, progress)
        except Exception as e:
            # A 404 means this endpoint does not hold the market (the
            # archive cutoff sits on the other side of its settlement);
            # the other endpoint may. Anything else, or a 404 from the
            # last endpoint, is final.
            if getattr(e, "status", None) != 404 or attempt + 1 == len(endpoints):
                raise
            # The failed request took no sleep of its own
            time.sleep(rate_limit_sleep)
    raise ValueError("no candlestick endpoint to ask")


def fetch_candlesticks(
    hist_client: Any,
    ticker: str,
    open_ts: int,
    close_ts: int,
    use_cache: bool = True,
    rate_limit_sleep: float = 0.15,
    *,
    series: str | None = None,
    live_first: bool = False,
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

    Two endpoints serve candles (_candle_endpoints): the archive
    (/historical/markets/{ticker}/candlesticks) for a market that settled
    before Kalshi's archive cutoff, and the live API
    (/series/{series}/markets/{ticker}/candlesticks) for one that settled
    after it, which the archive answers with 404. Given the market's series,
    the endpoint live_first names is asked first and a 404 from it asks the
    other for the whole window, so a market the cutoff has moved past since
    it was routed is still found. Any other failure, or a 404 from both, is
    final. Without a series only the archive is asked, as before. The live
    endpoint's per-request cap is assumed to be the archive's
    (CANDLESTICK_MAX_CANDLES_PER_REQUEST); it has not been measured.

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
    tagged with the [open_ts, close_ts] window, the period_interval and the
    fields version (CANDLESTICK_CACHE_FIELDS_VERSION) that were actually
    fetched — the whole window as one entry, however many requests it took. A
    cache hit requires the cached window to COVER the
    requested window AND the cached period_interval to match the current
    CANDLESTICK_PERIOD_INTERVAL_MINUTES — open_ts varies between backtest runs
    with different --start-date values, so a cache built for a later start_date
    must not be reused for an earlier one (it would be silently missing the
    earlier candles), and a cache built under a different granularity (e.g. an
    older daily-interval cache) must not be silently reused as if it were
    hourly. It also requires the cached fields version to match the current
    one, so a file written before a candle field was added is fetched again.
    Only successful fetches are cached; a fetch failure returns []
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
        series (str | None): Keyword-only. The market's series ticker
            (series_ticker of its event ticker); with one, the live endpoint is
            asked too. None (default) asks the archive only.
        live_first (bool): Keyword-only. Ask the live endpoint before the
            archive — the market settled at or after the archive cutoff. Read
            only with a series. False (default).

    Returns:
        list[dict]: List of candlestick dicts with keys:
            - "ts" (int): Unix timestamp of the candle's end period.
            - "yes_ask_close" (float): YES ask price at close. Range: [0.01, 0.99].
            - "no_ask_close" (float): Approximated NO ask price at close (1 − yes_bid_close).
              Clamped to [0.01, 0.99].
            - "volume" (float | None): Contracts traded in the candle's hour;
              None when the API gave no readable count.
            Returns an empty list on API failure, including a failure of any
            one request of a paged window.
    """
    _CANDLES_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _CANDLES_DIR / f"{ticker}.json"
    if use_cache:
        # Guarded loader: absent OR corrupt (a truncated file from an
        # interrupted run) both read back as None, so a damaged cache falls
        # through to a refetch instead of raising out of the worker thread.
        cached = _load_json_cache(cache_path)
        # Legacy cache files are a bare list with no window metadata (or predate
        # the period_interval or fields tag) — we can't confirm what
        # range/granularity/fields they cover, so fall through and refetch. The
        # refetch below re-saves in the current tagged format, migrating it.
        if isinstance(cached, dict) and cached.get("open_ts", None) is not None:
            covers_window = (
                cached["open_ts"] <= open_ts and cached["close_ts"] >= close_ts
                and cached.get("period_interval") == CANDLESTICK_PERIOD_INTERVAL_MINUTES
                and cached.get("fields") == CANDLESTICK_CACHE_FIELDS_VERSION
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
    # The archive, the live API, or both, in the order to ask them
    endpoints = _candle_endpoints(ticker, series, live_first)
    # "request": the 1-based number of the request in flight, read only by the
    # failure line below; reset to 0 once every request has returned, so a
    # failure after that (the cache write) is not blamed on the last request.
    progress = {"request": 0}
    # The endpoints asked so far, named in the failure line when there were two
    tried: list[str] = []
    try:
        # The first endpoint that holds the market serves the whole window
        candles, raw_count, dropped = _fetch_candles_from_endpoints(
            hist_client, endpoints, windows, rate_limit_sleep, progress, tried)
        progress["request"] = 0
        if dropped:
            # The drop happens before the cache write, so a thinned series is
            # otherwise cached as if it were complete with no visible signal.
            logging.warning("%s: dropped %d/%d malformed candles",
                            ticker, dropped, raw_count)
        # Only successful fetches are cached (tagged with the window,
        # granularity and fields version just fetched); failures fall
        # through the except branch and return [] without persisting so the
        # next run retries.
        # Saved unconditionally — use_cache only controls whether reads may
        # come from disk, mirroring fetch_all_settled_markets — otherwise a
        # --no-cache run would never actually refresh the file the next
        # default run loads. Which endpoint served the candles is not
        # recorded: the candles are the same market's either way.
        _save_json_cache(cache_path, {
            "open_ts": open_ts, "close_ts": close_ts,
            "period_interval": CANDLESTICK_PERIOD_INTERVAL_MINUTES,
            "fields": CANDLESTICK_CACHE_FIELDS_VERSION,
            "candles": candles,
        })
        return candles
    except Exception as e:
        # ONE line, no header dump (TS-02): a ticker neither endpoint serves
        # is deliberately never cached, so this warning is re-paid on every
        # run for every such ticker. The per-run count is summarized once by
        # backtester._fetch_candles_parallel, which sees every ticker. A
        # paged window names the request that failed, and a fetch that asked
        # both endpoints says so; a single-request, archive-only fetch keeps
        # the exact line it always logged.
        where = (f" (request {progress['request']} of {len(windows)})"
                 if len(windows) > 1 and progress["request"] else "")
        via = f" ({' then '.join(tried)} endpoint)" if tried != ["historical"] else ""
        logging.warning(
            "Candlestick fetch failed for %s: HTTP %s %s%s%s",
            ticker, getattr(e, "status", "?"), _exception_summary(e), where, via,
        )
        time.sleep(rate_limit_sleep)
        # Deliberately DO NOT cache — a poisoned empty file would silence this
        # ticker on every subsequent run until manually deleted. That holds for
        # a paged window too: the requests that DID succeed are discarded
        # rather than cached as if they were the whole window.
        return []


def recent_candles(client: Any, ticker: str, event_ticker: str, start_ts: int,
                   end_ts: int) -> list[dict] | None:
    """
    Fetch one market's hourly candles over [start_ts, end_ts] for a live run, never cached.

    Live selling checks a held position on the days before a sale from each
    held market's recent candles (bid_before reads them). They are fetched
    and parsed exactly as fetch_candlesticks fetches and parses them: the
    live candlestick endpoint first (the market is still open, or settled
    after the archive cutoff), then the archive on a 404
    (_candle_endpoints, _fetch_candles_from_endpoints), with a window too
    long for one request paged (_candle_request_windows), each request a
    retried read-only GET, and each candle parsed by _fetch_candle_pages.
    Unlike fetch_candlesticks it never reads or writes the candle cache: a
    live run touches nothing under backtest_cache/, and its candles must be
    the latest the exchange has.

    Hourly candles are sparse: many hours of a quiet market have no candle
    at all, so a gap is normal. A candle that cannot be parsed is left out,
    with one WARNING naming the count, as fetch_candlesticks does.

    Args:
        client (Any): An authenticated KalshiClient for the production API.
        ticker (str): The market's ticker.
        event_ticker (str): Its event ticker; its series (series_ticker)
            names the live endpoint's path.
        start_ts (int): Unix seconds the window starts at.
        end_ts (int): Unix seconds the window ends at.

    Returns:
        list[dict] | None: The candles ("ts", "yes_ask_close",
            "no_ask_close", "volume"), as fetch_candlesticks returns them;
            [] when the endpoint has none in the window. None, with one
            WARNING line, when any request fails or a candle's end time is
            not an integer number of seconds: the caller must then treat the
            market's earlier days as unknown.
    """
    # The live endpoint's path names the market's series
    series = series_ticker(event_ticker if isinstance(event_ticker, str) else "")
    progress = {"request": 0}
    tried: list[str] = []
    windows: list[tuple[int, int]] = []
    try:
        # One request unless the window is longer than one request serves
        windows = _candle_request_windows(start_ts, end_ts)
        # Live first: a held market's candles are on the live endpoint
        endpoints = _candle_endpoints(ticker, series, live_first=True)
        candles, raw_count, dropped = _fetch_candles_from_endpoints(
            client, endpoints, windows, RECENT_CANDLES_RATE_LIMIT_SLEEP_SECONDS,
            progress, tried)
    except Exception as e:
        # One line, never the SDK's multi-line text with every header
        where = (f" (request {progress['request']} of {len(windows)})"
                 if len(windows) > 1 and progress["request"] else "")
        via = f" ({' then '.join(tried)} endpoint)" if tried else ""
        logging.warning("Recent candles could not be read for %s: HTTP %s %s%s%s",
                        ticker, getattr(e, "status", "?"), _exception_summary(e), where, via)
        return None
    if dropped:
        logging.warning("%s: dropped %d/%d malformed candles", ticker, dropped, raw_count)
    # The shared parser copies each candle's end time as sent; bid_before
    # compares it with a moment, so an end time that is not an integer
    # number of seconds (null, a string) would raise there. Such a reply
    # cannot place the market's earlier days, so it is not used at all.
    unplaced = sum(1 for c in candles
                   if not isinstance(c["ts"], int) or isinstance(c["ts"], bool))
    if unplaced:
        logging.warning("Recent candles could not be read for %s: %d of %d candles have "
                        "an end time that is not an integer number of seconds",
                        ticker, unplaced, len(candles))
        return None
    return candles


# ─── What a sale would fetch, read from candles ───────────────────────────────
#
# A candle carries each side's ASK at the end of its hour (fetch_candlesticks:
# "yes_ask_close", and "no_ask_close", which is 1 - the YES bid). Selling a
# side fetches that side's BID, and a YES bid is the price someone offers for
# YES, which is 1 - the NO ask (and a NO bid 1 - the YES ask). The backtest
# reads its bids this way (backtester._leg_quotes), and live selling will read
# the days before a sale from recent candles with the same functions. The
# highest NO ask a candle can carry, which is no quote, is
# config.CANDLE_NO_ASK_CEILING, the value fetch_candlesticks clamps to.


def usable_candle_ask(raw: Any, side: str) -> float:
    """
    Read one candle close as an ask that can value or sell a leg, or NaN.

    An ask is usable when it reads as a number strictly between 0 and 1 —
    the rule live applies to a held pair's ask (scanner._held_leg_worth) —
    so a sub-cent ask on a fine grid counts, while a YES ask of 1.00 (no one
    offering YES) does not. A NO ask must also be below
    CANDLE_NO_ASK_CEILING: the candle stores the NO ask as 1 - the YES bid,
    clamped to that ceiling at most, so the ceiling is also what a YES-bid
    book with no bids reads as. Each bound is held PRICE_EPSILON inside, so
    float noise on a price sitting on a bound reads as that bound. The
    backtest reads the asks that value an open leg through it
    (backtester._usable_ask, in _leg_quotes), and candle_sale_bids reads both
    asks of a candle through it.

    Args:
        raw: A candle's "yes_ask_close" or "no_ask_close" (normally a float).
        side (str): "yes" for a YES ask, "no" for a NO ask.

    Returns:
        float: The ask when usable; NaN otherwise.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float("nan")
    top = 1.0 if side == "yes" else CANDLE_NO_ASK_CEILING
    # A NaN fails both comparisons, so it stays unusable too
    return value if PRICE_EPSILON < value < top - PRICE_EPSILON else float("nan")


def candle_sale_bids(candle: dict) -> tuple[float, float]:
    """
    What selling one contract of each side would fetch on one candle: (YES bid, NO bid).

    The YES bid is 1 - the candle's usable NO ask and the NO bid is 1 - its
    usable YES ask (usable_candle_ask), each rounded to six decimals so float
    noise (1 - 0.43 is 0.5700000000000001) never reaches a price. An
    unusable ask leaves NaN on the other side: that side has no bid there.
    The backtest records its sale bids with it (backtester._leg_quotes), and
    bid_before reads a recent candle with it.

    Args:
        candle (dict): One candle, as fetch_candlesticks returns it
            ("yes_ask_close", "no_ask_close"; either may be missing).

    Returns:
        tuple[float, float]: (YES bid, NO bid); NaN for a side with no bid.
    """
    return (round(1.0 - usable_candle_ask(candle.get("no_ask_close"), "no"), 6),
            round(1.0 - usable_candle_ask(candle.get("yes_ask_close"), "yes"), 6))


def bid_before(candles: list[dict], moment: int, side: str, *, window: int) -> float:
    """
    One side's bid from the last candle that ended at or before `moment`, if it is recent enough.

    The sell rule checks a position once a day before it sells, and an
    earlier day's check reads that day's last quote: the latest candle that
    ended at or before the check and less than `window` seconds before it
    (the backtest's daily checks use one day, so a candle exactly 24 hours
    old belongs to the check before). An older quote is never carried
    forward. The candle is found the way the backtest finds one
    (backtester._candles_at_or_before for one moment): stepping through the
    candles in their own order and stopping at the first that ends after
    `moment`, the answer being the one before it. So for candles in time
    order it is the latest candle at or before `moment`. Whether the market
    had paid out by then is not read here; the caller handles a payout.
    Live selling will read the days before a sale from a market's recent
    candles with it.

    Args:
        candles (list[dict]): One market's candles, each with "ts" (the end
            of its hour, Unix seconds), normally in time order.
        moment (int): The check, in Unix seconds.
        side (str): The side to sell, "yes" or "no".
        window (int): Keyword-only. How many seconds old the candle may be:
            it must have ended less than this long before `moment`.

    Returns:
        float: The side's bid (candle_sale_bids); NaN when no candle ended
            at or before `moment`, the latest one is `window` seconds old or
            more, or the other side's ask on it is not usable.

    Raises:
        ValueError: For a side other than "yes" or "no".
    """
    if side == "yes":
        index = 0
    elif side == "no":
        index = 1
    else:
        raise ValueError(f"side must be 'yes' or 'no', got {side!r}")
    # Step forward to the first candle that ends after the moment; the one
    # just before it is the latest at or before the moment
    i = 0
    while i < len(candles) and candles[i]["ts"] <= moment:
        i += 1
    if i == 0:
        return float("nan")
    candle = candles[i - 1]
    if not moment - candle["ts"] < window:
        return float("nan")
    return candle_sale_bids(candle)[index]
