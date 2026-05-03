"""Historical market data fetching with disk caching for backtesting."""
import json
import logging
import time
from datetime import date, datetime, timezone
from pathlib import Path

from kalshi_python_sync.api.historical_api import HistoricalApi

from .auth import build_client
from .config import PROJECT_ROOT

CACHE_DIR = PROJECT_ROOT / "backtest_cache"
_CANDLES_DIR = CACHE_DIR / "candlesticks"

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
    upper = (event_ticker or "").upper()
    for prefix, cat in _CATEGORY_PREFIXES:
        if upper.startswith(prefix):
            return cat
    return "Other"


def build_historical_client() -> HistoricalApi:
    """HistoricalApi backed by prod credentials (historical data only exists on prod)."""
    return HistoricalApi(api_client=build_client("prod"))


def build_prod_live_client():
    """KalshiClient pointed at prod for recently-settled market fetching."""
    return build_client("prod")


# ─── Serialization helpers ────────────────────────────────────────────────────

def _market_to_dict(m) -> dict:
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
    if path.exists():
        return json.loads(path.read_text())
    return None


def _save_json_cache(path: Path, data) -> None:
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
    Return all settled markets from start_date onward as plain dicts.

    Uses two endpoints:
    - /historical/markets — older settled markets (before the API cutoff)
    - /markets?status=settled — recently settled markets (after the cutoff)

    Results are cached so re-runs don't re-fetch. Clear the cache or use
    --no-cache to force a fresh pull.
    """
    cache_path = CACHE_DIR / f"settled_markets_{start_date.isoformat()}.json"
    if use_cache:
        cached = _load_json_cache(cache_path)
        if cached is not None:
            logging.info("Loaded %d settled markets from cache", len(cached))
            return cached

    start_ts = int(datetime(start_date.year, start_date.month, start_date.day,
                            tzinfo=timezone.utc).timestamp())

    cutoff      = hist_client.get_historical_cutoff()
    cutoff_ts   = int(cutoff.market_settled_ts.timestamp())

    all_markets: list[dict] = []

    # ── Historical endpoint ───────────────────────────────────────────────────
    logging.info("Fetching historical settled markets (settled before API cutoff)...")
    cursor = None
    while True:
        kwargs: dict = dict(limit=1000, mve_filter="exclude")
        if cursor:
            kwargs["cursor"] = cursor
        resp = hist_client.get_historical_markets(**kwargs)
        for m in resp.markets:
            if m.settlement_ts is None or m.result not in ("yes", "no"):
                continue
            settle_epoch = int(m.settlement_ts.timestamp())
            if settle_epoch < start_ts or settle_epoch >= cutoff_ts:
                continue
            all_markets.append(_market_to_dict(m))
        cursor = resp.cursor
        if not cursor:
            break
    logging.info("Historical endpoint: %d markets from %s", len(all_markets), start_date)

    # ── Live endpoint (recently settled) ─────────────────────────────────────
    logging.info("Fetching recently settled markets (after API cutoff)...")
    recent: list[dict] = []
    cursor = None
    while True:
        kwargs = dict(status="settled", limit=1000, mve_filter="exclude",
                      min_settled_ts=cutoff_ts)
        if cursor:
            kwargs["cursor"] = cursor
        resp = live_client.get_markets(**kwargs)
        for m in resp.markets:
            if m.settlement_ts is None or m.result not in ("yes", "no"):
                continue
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
    Fetch daily candlesticks for one market over its lifetime.
    Returns list of dicts: {ts, yes_ask_close, no_ask_close}.

    no_ask_close ≈ 1 - yes_bid_close (binary market complement).
    Results cached per ticker.
    """
    _CANDLES_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _CANDLES_DIR / f"{ticker}.json"
    if use_cache and cache_path.exists():
        return json.loads(cache_path.read_text())

    try:
        resp = hist_client.get_market_candlesticks_historical(
            ticker=ticker,
            start_ts=open_ts,
            end_ts=close_ts,
            period_interval=1440,
        )
        candles = []
        for c in resp.candlesticks:
            try:
                yes_ask = float(c.yes_ask.close)
                no_ask  = 1.0 - float(c.yes_bid.close)  # complement approximation
                candles.append({
                    "ts": c.end_period_ts,
                    "yes_ask_close": yes_ask,
                    "no_ask_close": max(0.01, min(0.99, no_ask)),
                })
            except (ValueError, TypeError):
                pass
        time.sleep(rate_limit_sleep)
        if use_cache:
            _save_json_cache(cache_path, candles)
        return candles
    except Exception as e:
        logging.warning("Candlestick fetch failed for %s: %s", ticker, e)
        time.sleep(rate_limit_sleep)
        if use_cache:
            _save_json_cache(cache_path, [])  # cache empty to avoid repeated failures
        return []
