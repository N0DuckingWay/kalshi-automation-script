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

from kalshi_python_sync.api.historical_api import HistoricalApi

from ._http import api_call_with_retry
from .auth import build_client
from .config import PROJECT_ROOT

CACHE_DIR = PROJECT_ROOT / "backtest_cache"
_CANDLES_DIR = CACHE_DIR / "candlesticks"

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
    """
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
        dict: A flat dictionary with keys: ticker, event_ticker, title, subtitle,
            result, yes_ask_dollars, no_ask_dollars, yes_bid_dollars, close_time,
            settlement_ts, status.
    """
    return {
        "ticker": m.ticker,
        "event_ticker": m.event_ticker or "",
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

    Args:
        hist_client (HistoricalApi): Historical API client from build_historical_client().
        live_client: KalshiClient from build_prod_live_client() for recent settlements.
        start_date (date): Earliest settlement date to include. Markets that settled
            before this date are skipped even if the API returns them.
        use_cache (bool): If True (default), load from disk cache if available and
            save to disk after fetching. If False, always fetch from the API.

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

    all_markets: list[dict] = []

    # ── Historical endpoint ───────────────────────────────────────────────────
    logging.info("Fetching historical settled markets (settled before API cutoff)...")
    cursor = None
    while True:
        kwargs: dict = dict(limit=1000, mve_filter="exclude")
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
            all_markets.append(_market_to_dict(m))
        cursor = resp.cursor
        # A None or empty cursor signals the last page
        if not cursor:
            break
    logging.info("Historical endpoint: %d markets from %s", len(all_markets), start_date)

    # ── Live endpoint (recently settled) ─────────────────────────────────────
    logging.info("Fetching recently settled markets (after API cutoff)...")
    recent: list[dict] = []
    cursor = None
    while True:
        # min_settled_ts=cutoff_ts ensures we only get markets not covered by the historical endpoint
        kwargs = dict(status="settled", limit=1000, mve_filter="exclude",
                      min_settled_ts=cutoff_ts)
        if cursor:
            kwargs["cursor"] = cursor
        resp = api_call_with_retry(live_client.get_markets, **kwargs)
        for m in resp.markets:
            if m.settlement_ts is None or m.result not in ("yes", "no"):
                continue
            # Apply start_date filter here too since the API doesn't filter by settlement date
            if int(m.settlement_ts.timestamp()) < start_ts:
                continue
            recent.append(_market_to_dict(m))
        cursor = resp.cursor
        if not cursor:
            break
    logging.info("Live endpoint: %d recently settled markets", len(recent))

    all_markets.extend(recent)
    logging.info("Total settled markets from %s: %d", start_date, len(all_markets))

    if use_cache:
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

    Results are cached per ticker in backtest_cache/candlesticks/<ticker>.json.
    On fetch failure, an empty list is cached so the same ticker isn't retried
    on every subsequent run.

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
        # Honor the cache if it has real content, or if a recorded empty is still fresh.
        # Otherwise let the code fall through and retry — a stale empty was likely a
        # previous transient failure, not a genuinely empty market.
        if cached:
            return cached
        age = time.time() - cache_path.stat().st_mtime
        if age < _EMPTY_CANDLE_TTL_SECONDS:
            return cached

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
                yes_ask = float(c.yes_ask.close)
                # NO ask ≈ 1 - YES bid (binary market complement); clamp to avoid 0 or 1
                no_ask  = 1.0 - float(c.yes_bid.close)
                candles.append({
                    "ts": c.end_period_ts,
                    "yes_ask_close": yes_ask,
                    "no_ask_close": max(0.01, min(0.99, no_ask)),
                })
            except (ValueError, TypeError):
                pass
        # Rate limit: sleep briefly after each call to avoid 429 responses
        time.sleep(rate_limit_sleep)
        if use_cache:
            # Only successful fetches are cached; failures fall through the except
            # branch and return [] without persisting so the next run retries.
            _save_json_cache(cache_path, candles)
        return candles
    except Exception as e:
        logging.warning("Candlestick fetch failed for %s: %s", ticker, e)
        time.sleep(rate_limit_sleep)
        # Deliberately DO NOT cache — a poisoned empty file would silence this
        # ticker on every subsequent run until manually deleted.
        return []
