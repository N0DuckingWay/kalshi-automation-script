"""
File: historical.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Fetches and caches historical Kalshi market data needed by the backtester.
    Two data types are collected: (1) settled market metadata (title, outcome,
    prices, timestamps) from both the /historical/markets endpoint and the
    regular /markets?status=settled endpoint (which covers more recent settlements);
    and (2) daily candlestick price series for individual markets used to find
    the week when each pair first became tradeable. All data is cached to JSON
    files on disk so re-runs do not re-fetch from the API.

Dependencies:
    Imports build_client from auth.py and PROJECT_ROOT from config.py. Exports
    build_historical_client(), build_prod_live_client(), fetch_all_settled_markets(),
    fetch_daily_candlesticks(), and infer_category() — all called by backtester.py.

Notes:
    Historical market data only exists on the production API — the sandbox does
    not have a historical endpoint. The backtest always uses prod credentials.
    Candlesticks are cached per ticker in backtest_cache/candlesticks/<ticker>.json
    to avoid thousands of API calls on repeated runs.
"""
import json
import logging
import time
from datetime import UTC, date, datetime
from pathlib import Path

try:
    from kalshi_python_sync.api.historical_api import HistoricalApi
except ImportError:
    # Some installed SDK builds ship without the historical_api module. Keep
    # the import soft so this module (and backtester.py, which imports from
    # it) stays importable and unit-testable offline — the real data-fetch
    # path fails loudly in build_historical_client() instead.
    HistoricalApi = None

from ._http import api_call_with_retry
from .auth import build_client
from .config import INCLUDE_MVE_MARKETS, PROJECT_ROOT

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


def build_historical_client() -> HistoricalApi:
    """
    Construct a HistoricalApi client backed by production credentials.

    Historical market data (older settled markets) only exists on the production
    endpoint — the sandbox does not expose a /historical/markets endpoint. This
    client is used exclusively by the backtest pipeline.

    Returns:
        HistoricalApi: A Kalshi historical API client authenticated via the prod
            RSA key and pointed at the production endpoint.

    Raises:
        RuntimeError: If the installed kalshi_python_sync build has no
            historical_api module — the backtest data fetch cannot run until
            an SDK version providing it is installed.
    """
    if HistoricalApi is None:
        raise RuntimeError(
            "The installed kalshi_python_sync has no historical_api module — "
            "install an SDK version that provides it to fetch backtest data."
        )
    # build_client("prod") returns a KalshiClient authenticated via RSA key from secrets.json
    return HistoricalApi(api_client=build_client("prod"))


def build_prod_live_client():
    """
    Construct a standard KalshiClient pointed at production for recently-settled markets.

    Used alongside build_historical_client() to cover recently settled markets
    that are not yet in the historical archive (i.e. settled after the API cutoff
    timestamp returned by hist_client.get_historical_cutoff()).

    Returns:
        KalshiClient: An authenticated client pointed at the production endpoint.
    """
    # build_client("prod") returns a KalshiClient authenticated via RSA key from secrets.json
    return build_client("prod")


# ─── Serialization helpers ────────────────────────────────────────────────────

def _market_to_dict(m) -> dict:
    """
    Serialize a Kalshi market API object to a plain dict suitable for JSON caching.

    Extracts only the fields needed by the backtester, converting datetime objects
    to ISO 8601 strings so the result is JSON-serializable.

    Args:
        m: A Kalshi market object returned by the live or historical API client.

    Returns:
        dict: A flat dictionary with keys: ticker, event_ticker, event_title,
            title, subtitle, result, yes_ask_dollars, no_ask_dollars,
            yes_bid_dollars, close_time, settlement_ts, status.

            event_title is populated from the `_event_title` attribute, which
            the caller must attach (see `fetch_all_settled_markets`) so the
            backtester can group pairs by combined event+market title.
    """
    return {
        "ticker": m.ticker,
        "event_ticker": m.event_ticker or "",
        # Populated by fetch_all_settled_markets via _load_or_build_event_titles
        "event_title": getattr(m, "_event_title", "") or "",
        "title": m.title,
        "subtitle": getattr(m, "subtitle", None),
        "result": m.result,
        "yes_ask_dollars": m.yes_ask_dollars,
        "no_ask_dollars": m.no_ask_dollars,
        "yes_bid_dollars": m.yes_bid_dollars,
        "close_time": m.close_time.isoformat() if m.close_time else None,
        "settlement_ts": m.settlement_ts.isoformat() if m.settlement_ts else None,
        "status": m.status,
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

    # Bulk pull multivariate events — these are excluded from get_events by API design.
    if missing:
        cursor = None
        while True:
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

    # Per-ticker fallback for archived events that aren't in the bulk listings.
    # Failures are recorded as "" so we don't retry on every backtest run.
    for tkr in list(missing):
        try:
            resp = api_call_with_retry(live_client.get_event, event_ticker=tkr)
            cached[tkr] = (resp.event.title or "") if resp.event else ""
        except Exception as e:
            logging.warning("Could not resolve event title for %s: %s", tkr, e)
            cached[tkr] = ""

    _save_json_cache(_EVENT_TITLES_CACHE, cached)
    return cached


# ─── Market fetching ──────────────────────────────────────────────────────────

def fetch_all_settled_markets(
    hist_client: HistoricalApi,
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

    Results are serialized to a JSON cache file keyed by start_date so subsequent
    backtests do not re-fetch. Pass use_cache=False or use --no-cache to force a
    fresh pull (necessary when new markets have settled since the last cache write).
    A freshly fetched result is always written back to the cache file regardless
    of use_cache, so a --no-cache run actually refreshes what the next default
    (cached) run will load.

    Args:
        hist_client (HistoricalApi): Historical API client from build_historical_client().
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
    cutoff      = api_call_with_retry(hist_client.get_historical_cutoff)
    cutoff_ts   = int(cutoff.market_settled_ts.timestamp())

    # Collect Market objects first (deferred dict conversion) so we can attach
    # event titles after fetching the event_ticker → event_title mapping in one batch.
    selected_markets: list = []

    # Build the historical-endpoint base kwargs; gate the MVE filter on the config flag.
    # When INCLUDE_MVE_MARKETS is True, omitting mve_filter lets MVE markets through;
    # when False, the legacy "exclude" behaviour is preserved.
    hist_base_kwargs: dict = dict(limit=1000)
    if not INCLUDE_MVE_MARKETS:
        hist_base_kwargs["mve_filter"] = "exclude"

    # ── Historical endpoint ───────────────────────────────────────────────────
    logging.info("Fetching historical settled markets (settled before API cutoff)...")
    cursor = None
    while True:
        kwargs: dict = dict(hist_base_kwargs)
        # Include cursor for pages after the first to continue pagination
        if cursor:
            kwargs["cursor"] = cursor
        resp = api_call_with_retry(hist_client.get_historical_markets, **kwargs)
        for m in resp.markets:
            # Skip markets without a settlement timestamp or with a non-binary result
            if m.settlement_ts is None or m.result not in ("yes", "no"):
                continue
            settle_epoch = int(m.settlement_ts.timestamp())
            # Only include markets that settled within our [start_ts, cutoff_ts) window
            if settle_epoch < start_ts or settle_epoch >= cutoff_ts:
                continue
            selected_markets.append(m)
        cursor = resp.cursor
        # A None or empty cursor signals the last page
        if not cursor:
            break
    logging.info("Historical endpoint: %d markets from %s", len(selected_markets), start_date)

    # ── Live endpoint (recently settled) ─────────────────────────────────────
    logging.info("Fetching recently settled markets (after API cutoff)...")
    recent_count = 0
    cursor = None
    while True:
        # min_settled_ts=cutoff_ts ensures we only get markets not covered by the historical endpoint
        kwargs = dict(status="settled", limit=1000, min_settled_ts=cutoff_ts)
        if not INCLUDE_MVE_MARKETS:
            kwargs["mve_filter"] = "exclude"
        if cursor:
            kwargs["cursor"] = cursor
        resp = api_call_with_retry(live_client.get_markets, **kwargs)
        for m in resp.markets:
            if m.settlement_ts is None or m.result not in ("yes", "no"):
                continue
            # Apply start_date filter here too since the API doesn't filter by settlement date
            if int(m.settlement_ts.timestamp()) < start_ts:
                continue
            selected_markets.append(m)
            recent_count += 1
        cursor = resp.cursor
        if not cursor:
            break
    logging.info("Live endpoint: %d recently settled markets", recent_count)

    # ── Attach event titles ───────────────────────────────────────────────────
    # Collect unique event_tickers and look up their titles in one batch so the
    # backtester can build (event_title + market_title) grouping keys. Skipped
    # entirely when MVE is excluded, since the live scanner's combined-key logic
    # wouldn't change anything for binary-only markets.
    if INCLUDE_MVE_MARKETS:
        unique_tickers = {m.event_ticker for m in selected_markets if m.event_ticker}
        logging.info("Resolving event titles for %d unique event_tickers", len(unique_tickers))
        titles = _load_or_build_event_titles(live_client, unique_tickers, use_cache=use_cache)
        for m in selected_markets:
            # Attach as _event_title so _market_to_dict picks it up
            m._event_title = titles.get(m.event_ticker or "", "")

    all_markets: list[dict] = [_market_to_dict(m) for m in selected_markets]
    logging.info("Total settled markets from %s: %d", start_date, len(all_markets))

    # Always persist a fresh fetch — use_cache only controls whether reads are
    # allowed to come from disk. Gating the save on use_cache meant --no-cache
    # runs (whose whole point is to refresh stale data) never updated the file
    # the very next default run would load.
    _save_json_cache(cache_path, all_markets)
    return all_markets


# ─── Candlestick fetching ─────────────────────────────────────────────────────

def fetch_daily_candlesticks(
    hist_client: HistoricalApi,
    ticker: str,
    open_ts: int,
    close_ts: int,
    use_cache: bool = True,
    rate_limit_sleep: float = 0.15,
) -> list[dict]:
    """
    Fetch daily OHLC candlesticks for one market over its active lifetime.

    Returns one candle per calendar day with the YES ask close price and an
    approximated NO ask close price. The NO ask is computed as the complement
    of the YES bid close (1 − yes_bid_close), which is exact in a binary market
    with no bid-ask spread and a close approximation in practice.

    Results are cached per ticker in backtest_cache/candlesticks/<ticker>.json,
    tagged with the [open_ts, close_ts] window that was actually fetched. A
    cache hit requires the cached window to COVER the requested window —
    open_ts varies between backtest runs with different --start-date values,
    so a cache built for a later start_date must not be reused for an earlier
    one (it would be silently missing the earlier candles). Only successful
    fetches are cached; a fetch failure returns [] WITHOUT persisting it, so
    the ticker is retried on the next run (a cached empty file would otherwise
    silence it forever). Pre-existing empty cache files are honored only
    while younger than _EMPTY_CANDLE_TTL_SECONDS.

    Args:
        hist_client (HistoricalApi): Historical API client from build_historical_client().
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
        list[dict]: List of daily candlestick dicts with keys:
            - "ts" (int): Unix timestamp of the candle's end period.
            - "yes_ask_close" (float): YES ask price at close. Range: [0.01, 0.99].
            - "no_ask_close" (float): Approximated NO ask price at close (1 − yes_bid_close).
              Clamped to [0.01, 0.99]. Returns an empty list on API failure.
    """
    _CANDLES_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _CANDLES_DIR / f"{ticker}.json"
    if use_cache and cache_path.exists():
        cached = json.loads(cache_path.read_text())
        # Legacy cache files are a bare list with no window metadata — we can't
        # confirm what range they cover, so fall through and refetch. The
        # refetch below re-saves in the new tagged format, migrating it.
        if isinstance(cached, dict) and cached.get("open_ts", None) is not None:
            covers_window = (
                cached["open_ts"] <= open_ts and cached["close_ts"] >= close_ts
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
        # period_interval=1440 requests one candle per 1440 minutes (1 day)
        resp = api_call_with_retry(
            hist_client.get_market_candlesticks_historical,
            ticker=ticker,
            start_ts=open_ts,
            end_ts=close_ts,
            period_interval=1440,
        )
        candles = []
        for c in resp.candlesticks:
            try:
                # close_dollars is the fixed-point dollar string; the bare
                # .close field is an integer price in CENTS and must not be
                # fed into the dollar-denominated backtest math
                yes_ask = float(c.yes_ask.close_dollars)
                # NO ask ≈ 1 - YES bid (binary market complement); clamp to avoid 0 or 1
                no_ask  = 1.0 - float(c.yes_bid.close_dollars)
                candles.append({
                    "ts": c.end_period_ts,
                    "yes_ask_close": yes_ask,
                    "no_ask_close": max(0.01, min(0.99, no_ask)),
                })
            except (ValueError, TypeError, AttributeError):
                pass
        # Rate limit: sleep briefly after each call to avoid 429 responses
        time.sleep(rate_limit_sleep)
        # Only successful fetches are cached (tagged with the window just
        # fetched); failures fall through the except branch and return []
        # without persisting so the next run retries. Saved unconditionally —
        # use_cache only controls whether reads may come from disk, mirroring
        # fetch_all_settled_markets — otherwise a --no-cache run would never
        # actually refresh the file the next default run loads.
        _save_json_cache(cache_path, {
            "open_ts": open_ts, "close_ts": close_ts, "candles": candles,
        })
        return candles
    except Exception as e:
        logging.warning("Candlestick fetch failed for %s: %s", ticker, e)
        time.sleep(rate_limit_sleep)
        # Deliberately DO NOT cache — a poisoned empty file would silence this
        # ticker on every subsequent run until manually deleted.
        return []
