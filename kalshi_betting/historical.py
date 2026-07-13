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
    Imports build_client from auth.py and PROJECT_ROOT from config.py. Exports
    build_historical_client(), build_prod_live_client(), fetch_all_settled_markets(),
    fetch_candlesticks(), and infer_category() — all called by backtester.py.

Notes:
    Historical market data only exists on the production API — the sandbox does
    not have a historical endpoint. The backtest always uses prod credentials.
    Candlesticks are cached per ticker in backtest_cache/candlesticks/<ticker>.json
    to avoid thousands of API calls on repeated runs. Candles are fetched at
    CANDLESTICK_PERIOD_INTERVAL_MINUTES (hourly) — daily candles only cover
    markets whose lifespan crosses a UTC midnight boundary, which silently
    excludes most Kalshi markets (see config.py for the full explanation).

    The pinned SDK (kalshi-python-sync==3.2.0) ships no historical_api module,
    and 2026-07 API drift broke its Market response model anyway (legacy
    integer-cent fields are no longer sent). All /historical endpoints are
    therefore reached with direct signed GETs (_signed_raw_get) through the
    SDK's own KalshiAuth + rest client, and responses are parsed as raw JSON —
    same pattern as scanner.py/trader.py (see _http.fetch_json_page).
"""
import json
import logging
import time
from datetime import UTC, date, datetime
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse

from ._http import api_call_with_retry, fetch_json_page
from .auth import build_client
from .config import (
    CANDLESTICK_PERIOD_INTERVAL_MINUTES,
    INCLUDE_MVE_MARKETS,
    MVE_TITLE_LOOKUP_MAX_PAGES,
    PROD_URL,
    PROJECT_ROOT,
)

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
            yes_bid_dollars, open_time, close_time, settlement_ts, status.
            Note: open_time is a new field (added alongside the backtester's
            eligibility prefilter) — cache files written before this change
            don't have it and will read back as None until refreshed with
            --no-cache; backtester._can_ever_enter() treats that as "can't
            prove ineligibility" and keeps the market (no speedup, no
            incorrect drop).
    """
    return {
        "ticker": m.get("ticker"),
        "event_ticker": m.get("event_ticker") or "",
        # Resolved by fetch_all_settled_markets via _load_or_build_event_titles
        "event_title": event_title or "",
        "title": m.get("title"),
        "subtitle": m.get("subtitle"),
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
    }


def _load_json_cache(path: Path):
    """
    Load and parse a JSON file from disk if it exists.

    Args:
        path (Path): Filesystem path to the JSON cache file.

    Returns:
        Any: Parsed JSON content (typically a list or dict) if the file exists,
            or None if the file does not exist.
    """
    if path.exists():
        return json.loads(path.read_text())
    return None


def _save_json_cache(path: Path, data) -> None:
    """
    Serialize data to JSON and write it to path, creating parent directories as needed.

    Uses a default=str serializer to handle datetime objects that may appear in the data.

    Args:
        path (Path): Destination file path. Parent directories are created if absent.
        data: JSON-serializable data structure (list, dict, etc.) to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, default=str))


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
      1. Bulk pull events (settled, closed, open) and their multivariate counterparts.
         For most backtests this covers nearly every event_ticker in a few hundred
         paginated calls.
      2. For any tickers still unresolved (very old archived events that have aged
         out of the bulk listings), fall back to per-ticker get_event() calls.

    Results are persisted to _EVENT_TITLES_CACHE so subsequent runs are essentially
    free. Tickers that cannot be resolved are stored as empty strings (poison pill)
    so we do not retry them every run.

    Args:
        live_client: A KalshiClient with the events API methods available
            (e.g. the client from build_prod_live_client()).
        event_tickers (set[str]): The set of event_ticker values whose titles
            we need. May contain hundreds or thousands of entries.
        use_cache (bool): If True, load existing cache from disk first. If False,
            re-fetch all titles. The on-disk cache is written either way.

    Returns:
        dict[str, str]: Mapping event_ticker → event_title. Tickers that could
            not be resolved map to "". Caller treats those markets as ungrouped
            (effectively MVE-excluded).
    """
    cached: dict[str, str] = (_load_json_cache(_EVENT_TITLES_CACHE) or {}) if use_cache else {}
    missing = event_tickers - cached.keys()
    if not missing:
        return cached

    # Bulk pull non-MVE events across all statuses. Each get_events call returns
    # up to 200 events; pagination continues until cursor is empty or all misses
    # are resolved (whichever comes first).
    for status in ("settled", "closed", "open"):
        if not missing:
            break
        cursor = None
        while True:
            resp = api_call_with_retry(
                live_client.get_events,
                status=status,
                limit=200,
                cursor=cursor,
            )
            for ev in resp.events or []:
                if ev.event_ticker in missing:
                    cached[ev.event_ticker] = ev.title or ""
                    missing.discard(ev.event_ticker)
            cursor = resp.cursor
            if not cursor or not missing:
                break

    # Bulk pull multivariate events — these are excluded from get_events by API
    # design. The MVE listing is effectively unbounded (hundreds of thousands of
    # auto-generated collection events), so this loop is capped at
    # MVE_TITLE_LOOKUP_MAX_PAGES; tickers not found by then fall through to the
    # bounded per-ticker lookup below instead of paging for hours.
    if missing:
        cursor = None
        for _ in range(MVE_TITLE_LOOKUP_MAX_PAGES):
            resp = api_call_with_retry(
                live_client.get_multivariate_events,
                limit=200,
                cursor=cursor,
            )
            for ev in resp.events or []:
                if ev.event_ticker in missing:
                    cached[ev.event_ticker] = ev.title or ""
                    missing.discard(ev.event_ticker)
            cursor = resp.cursor
            if not cursor or not missing:
                break

    # Per-ticker fallback for events that aren't in the bulk listings (or fell
    # past the MVE page cap). Uses a raw signed GET because the modeled
    # get_event embeds nested Market models the pinned SDK can no longer
    # deserialize (see module Notes). Failures are recorded as "" so we don't
    # retry on every backtest run.
    for tkr in list(missing):
        try:
            data = _historical_get(live_client, f"{_API_PREFIX}/events/{tkr}")
            cached[tkr] = (data.get("event") or {}).get("title") or ""
        except Exception as e:
            logging.warning("Could not resolve event title for %s: %s", tkr, e)
            cached[tkr] = ""

    _save_json_cache(_EVENT_TITLES_CACHE, cached)
    return cached


# ─── Market fetching ──────────────────────────────────────────────────────────

def fetch_all_settled_markets(
    hist_client: Any,
    live_client,
    start_date: date,
    use_cache: bool = True,
) -> list[dict]:
    """
    Fetch all settled Kalshi markets from start_date onward and return them as plain dicts.

    Uses two complementary API endpoints to get full coverage:
    - /historical/markets — settled markets archived before the API cutoff timestamp.
      This endpoint holds the bulk of historical data.
    - /markets?status=settled — markets that settled after the API cutoff, fetched
      via the regular live endpoint. This fills the gap between the archive and now.
      Bounded server-side by min_settled_ts=max(cutoff_ts, start_ts) so a narrow
      recent start_date doesn't force a walk of the entire cutoff→now range.

    Results are serialized to a JSON cache file keyed by start_date so subsequent
    backtests do not re-fetch. Pass use_cache=False or use --no-cache to force a
    fresh pull (necessary when new markets have settled since the last cache write).
    A freshly fetched result is always written back to the cache file regardless
    of use_cache, so a --no-cache run actually refreshes what the next default
    (cached) run will load.

    Args:
        hist_client (Any): Authenticated KalshiClient from build_historical_client().
        live_client: KalshiClient from build_prod_live_client() for recent settlements.
        start_date (date): Earliest settlement date to include. Markets that settled
            before this date are skipped even if the API returns them.
        use_cache (bool): If True (default), load from disk cache if available and
            skip the API fetch entirely. If False, always fetch from the API. Either
            way, a fetch that occurs is always saved to disk.

    Returns:
        list[dict]: Flat list of market dicts, each with keys: ticker, event_ticker,
            title, subtitle, result ("yes" | "no"), yes_ask_dollars, no_ask_dollars,
            yes_bid_dollars, close_time (ISO str), settlement_ts (ISO str), status.
            Only includes markets with a non-null settlement_ts and a binary result.
    """
    cache_path = CACHE_DIR / f"settled_markets_{start_date.isoformat()}.json"
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

    # Collect raw market dicts first (deferred normalization) so event titles
    # can be resolved in one batch after both endpoints are drained.
    selected_markets: list[dict] = []

    # Build the historical-endpoint base kwargs; gate the MVE filter on the config flag.
    # When INCLUDE_MVE_MARKETS is True, omitting mve_filter lets MVE markets through;
    # when False, the legacy "exclude" behaviour is preserved.
    hist_base_kwargs: dict = {"limit": 1000}
    if not INCLUDE_MVE_MARKETS:
        hist_base_kwargs["mve_filter"] = "exclude"

    def _settle_epoch(m: dict) -> int | None:
        """Parse a raw market's settlement_ts ISO string to epoch seconds."""
        ts = m.get("settlement_ts")
        if not ts:
            return None
        try:
            return int(datetime.fromisoformat(ts).timestamp())
        except (ValueError, TypeError):
            return None

    # ── Historical endpoint ───────────────────────────────────────────────────
    logging.info("Fetching historical settled markets (settled before API cutoff)...")
    cursor = None
    page_no = 0
    while True:
        kwargs: dict = dict(hist_base_kwargs)
        # Include cursor for pages after the first to continue pagination
        if cursor:
            kwargs["cursor"] = cursor
        data = _historical_get(hist_client, f"{_API_PREFIX}/historical/markets", **kwargs)
        page_markets = data.get("markets") or []
        for m in page_markets:
            # Skip markets without a settlement timestamp or with a non-binary result
            settle_epoch = _settle_epoch(m)
            if settle_epoch is None or m.get("result") not in ("yes", "no"):
                continue
            # Only include markets that settled within our [start_ts, cutoff_ts) window
            if settle_epoch < start_ts or settle_epoch >= cutoff_ts:
                continue
            selected_markets.append(m)
        page_no += 1
        if page_no % 100 == 0:
            logging.info("Historical archive: %d pages scanned, %d markets kept so far",
                         page_no, len(selected_markets))
        cursor = data.get("cursor")
        # A None or empty cursor signals the last page
        if not cursor:
            break
        # The archive is ordered newest-first by settlement time and ignores
        # settlement-time filters server-side, so paging must stop once a page
        # lies ENTIRELY before the backtest window — otherwise the loop pages
        # the full multi-million-market archive (observed ~60k settlements per
        # day) no matter how recent start_date is. The newest market on the
        # page is its first entry; if even that predates start_ts, every later
        # page is older still.
        newest_epoch = _settle_epoch(page_markets[0]) if page_markets else None
        if newest_epoch is not None and newest_epoch < start_ts:
            logging.info(
                "Historical archive page predates start_date %s — stopping pagination early",
                start_date,
            )
            break
    logging.info("Historical endpoint: %d markets from %s", len(selected_markets), start_date)

    # ── Live endpoint (recently settled) ─────────────────────────────────────
    # min_settled_ts is honored server-side here (unlike the archive), so this
    # loop is bounded to the [live_min_ts, now) window by the API itself.
    logging.info("Fetching recently settled markets (after API cutoff)...")
    recent_count = 0
    cursor = None
    page_no = 0
    # Bug fixed 2026-07: this used to always pass cutoff_ts, so a narrow recent
    # start_date (e.g. last 7 days) still forced the server to walk the ENTIRE
    # [cutoff_ts, now) range — observed 20k+ pages (20M+ records) scanned with
    # the client-side start_ts filter silently discarding nearly all of them,
    # because cutoff_ts can trail far behind now. Since the server already
    # honors min_settled_ts, raising it to start_ts whenever start_ts is the
    # tighter (later) bound eliminates that wasted paging entirely — it can
    # only narrow the server-side window, never miss markets the client-side
    # settle_epoch < start_ts check wasn't already going to discard anyway.
    live_min_ts = max(cutoff_ts, start_ts)
    while True:
        kwargs = {"status": "settled", "limit": 1000, "min_settled_ts": live_min_ts}
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
            settle_epoch = _settle_epoch(m)
            if settle_epoch is None or m.get("result") not in ("yes", "no"):
                continue
            # Apply start_date filter here too since the API doesn't filter by settlement date
            if settle_epoch < start_ts:
                continue
            selected_markets.append(m)
            recent_count += 1
        page_no += 1
        if page_no % 100 == 0:
            logging.info("Live settled sweep: %d pages scanned, %d markets kept so far",
                         page_no, recent_count)
        cursor = data.get("cursor")
        if not cursor:
            break
    logging.info("Live endpoint: %d recently settled markets", recent_count)

    # ── Attach event titles ───────────────────────────────────────────────────
    # Collect unique event_tickers and look up their titles in one batch so the
    # backtester can build (event_title + market_title) grouping keys. Skipped
    # entirely when MVE is excluded, since the live scanner's combined-key logic
    # wouldn't change anything for binary-only markets.
    titles: dict[str, str] = {}
    if INCLUDE_MVE_MARKETS:
        unique_tickers = {m.get("event_ticker") for m in selected_markets if m.get("event_ticker")}
        logging.info("Resolving event titles for %d unique event_tickers", len(unique_tickers))
        titles = _load_or_build_event_titles(live_client, unique_tickers, use_cache=use_cache)

    all_markets: list[dict] = [
        _market_to_dict(m, titles.get(m.get("event_ticker") or "", ""))
        for m in selected_markets
    ]
    logging.info("Total settled markets from %s: %d", start_date, len(all_markets))

    # Always persist a fresh fetch — use_cache only controls whether reads are
    # allowed to come from disk. Gating the save on use_cache meant --no-cache
    # runs (whose whole point is to refresh stale data) never updated the file
    # the very next default run would load.
    _save_json_cache(cache_path, all_markets)
    return all_markets


# ─── Candlestick fetching ─────────────────────────────────────────────────────

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

    Results are cached per ticker in backtest_cache/candlesticks/<ticker>.json,
    tagged with the [open_ts, close_ts] window and period_interval that were
    actually fetched. A cache hit requires the cached window to COVER the
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
        rate_limit_sleep (float): Seconds to sleep after each API call to stay
            within the Kalshi rate limit. Defaults to 0.15 seconds.

    Returns:
        list[dict]: List of candlestick dicts with keys:
            - "ts" (int): Unix timestamp of the candle's end period.
            - "yes_ask_close" (float): YES ask price at close. Range: [0.01, 0.99].
            - "no_ask_close" (float): Approximated NO ask price at close (1 − yes_bid_close).
              Clamped to [0.01, 0.99]. Returns an empty list on API failure.
    """
    _CANDLES_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _CANDLES_DIR / f"{ticker}.json"
    if use_cache and cache_path.exists():
        cached = json.loads(cache_path.read_text())
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

    try:
        # Raw signed GET — the pinned SDK has no historical_api module and its
        # candlestick models predate the current wire format anyway.
        data = _historical_get(
            hist_client,
            f"{_API_PREFIX}/historical/markets/{ticker}/candlesticks",
            start_ts=open_ts,
            end_ts=close_ts,
            period_interval=CANDLESTICK_PERIOD_INTERVAL_MINUTES,
        )
        candles = []
        for c in data.get("candlesticks") or []:
            try:
                ya = c.get("yes_ask") or {}
                yb = c.get("yes_bid") or {}
                # The API sends fixed-point DOLLAR strings (e.g. "0.5500").
                # Older payloads named the field close_dollars alongside an
                # integer-cent close; the current format sends the dollar
                # string AS close — prefer close_dollars when present so both
                # formats parse to dollars, never cents.
                yes_ask = float(ya.get("close_dollars") or ya.get("close"))
                # NO ask ≈ 1 - YES bid (binary market complement); clamp to avoid 0 or 1
                no_ask  = 1.0 - float(yb.get("close_dollars") or yb.get("close"))
                candles.append({
                    "ts": c["end_period_ts"],
                    "yes_ask_close": yes_ask,
                    "no_ask_close": max(0.01, min(0.99, no_ask)),
                })
            except (ValueError, TypeError, AttributeError, KeyError):
                pass
        # Rate limit: sleep briefly after each call to avoid 429 responses
        time.sleep(rate_limit_sleep)
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
        logging.warning("Candlestick fetch failed for %s: %s", ticker, e)
        time.sleep(rate_limit_sleep)
        # Deliberately DO NOT cache — a poisoned empty file would silence this
        # ticker on every subsequent run until manually deleted.
        return []
