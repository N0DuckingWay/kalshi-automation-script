"""Backtest simulation: replay the arbitrage strategy on historical Kalshi data."""
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from .config import (
    BUDGET_FRACTION,
    MAX_DEADLINE_GAP_DAYS,
    MIN_PRICE_DIFF,
    SAME_TITLE_CO_RESOLVE_PROB,
    SAME_TITLE_MIN_PRICE_DIFF,
)
from .historical import (
    HistoricalApi,
    fetch_all_settled_markets,
    fetch_daily_candlesticks,
    infer_category,
)
from .scanner import normalize_title


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    pair_type: str       # "time_series" | "same_title"
    ticker_a: str
    ticker_b: str
    title_a: str
    title_b: str
    category: str
    entry_date: date
    exit_date: date      # date the last-settling market resolved
    entry_pA: float      # YES ask of A at entry
    entry_pB: float      # YES ask of B at entry
    entry_nA: float      # NO ask of A at entry (≈ 1 - yes_bid_A)
    n: int               # contracts bought on each leg (x = y = n)
    total_cost: float
    outcome_a: str       # "yes" | "no"
    outcome_b: str       # "yes" | "no"
    actual_payoff: float
    profit: float
    profit_ratio: float
    monthly_profit_ratio: float  # profit_ratio * 30 / holding_days
    kelly_fraction: float        # capped Kelly fraction used for sizing
    expected_payoff: float  # n * (1 - nA - pB) — the guaranteed floor
    slippage: float         # actual_payoff - expected_payoff
    holding_days: int


def _compute_actual_payoff(n: int, pA: float, pB: float, nA: float,
                           outcome_a: str, outcome_b: str) -> float:
    """
    Compute actual payoff from settlement outcomes.
    Strategy: n NO on A + n YES on B.

    Payoff table (per n contracts):
      A=YES, B=YES → n*(1-pB) - n*nA
      A=NO,  B=YES → n*(1-nA) + n*(1-pB)
      A=NO,  B=NO  → n*(1-nA) - n*pB
      A=YES, B=NO  → -(n*nA + n*pB)  [impossible for time_series; loss for same_title]
    """
    if outcome_a == "yes" and outcome_b == "yes":
        return n * (1.0 - pB) - n * nA
    if outcome_a == "no" and outcome_b == "yes":
        return n * (1.0 - nA) + n * (1.0 - pB)
    if outcome_a == "no" and outcome_b == "no":
        return n * (1.0 - nA) - n * pB
    # outcome_a == "yes", outcome_b == "no" (loss scenario)
    return -(n * nA + n * pB)


# ─── Pair grouping (metadata only, no prices) ─────────────────────────────────

def _group_by_exact_title(markets: list[dict]) -> dict[tuple, list[dict]]:
    """Group markets by exact (title, subtitle) pair for same-title pair detection."""
    groups: dict = defaultdict(list)
    for m in markets:
        title    = m.get("title") or ""
        subtitle = m.get("subtitle") or ""
        if title or subtitle:
            groups[(title, subtitle)].append(m)
    return {k: v for k, v in groups.items() if len(v) >= 2}


def _group_by_normalized_title(markets: list[dict]) -> dict[str, list[dict]]:
    """Group markets by date-stripped title (for time-series pair detection)."""
    groups: dict = defaultdict(list)
    for m in markets:
        raw = m.get("title") or m.get("subtitle") or m.get("ticker", "")
        norm = normalize_title(raw)
        if norm:
            groups[norm].append(m)
    return {k: v for k, v in groups.items() if len(v) >= 2}


def _extract_pairs(groups: dict, pair_type: str) -> list[tuple[dict, dict, str]]:
    """
    Return list of (market_a, market_b, canonical_title) tuples where the two
    markets have different event_tickers. No price filtering at this stage.
    Keys may be strings (normalized-title groups) or (title, subtitle) tuples.
    """
    pairs = []
    for key, members in groups.items():
        canon = key if isinstance(key, str) else (key[0] or key[1])
        seen: set[frozenset] = set()
        for i, mA in enumerate(members):
            for mB in members[i + 1:]:
                if mA["event_ticker"] == mB["event_ticker"]:
                    continue
                pair_key = frozenset([mA["ticker"], mB["ticker"]])
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                pairs.append((mA, mB, canon))
    return pairs


# ─── Entry point detection from candlestick data ──────────────────────────────

def _monday_timestamps(start_date: date, end_date: date) -> list[int]:
    """Return unix timestamps for every Monday in [start_date, end_date]."""
    d = start_date
    while d.weekday() != 0:  # advance to first Monday
        d += timedelta(days=1)
    ts_list = []
    while d <= end_date:
        ts_list.append(int(datetime(d.year, d.month, d.day, 9, 0, tzinfo=timezone.utc).timestamp()))
        d += timedelta(weeks=1)
    return ts_list


def _candle_at_or_before(candles: list[dict], ts: int) -> Optional[dict]:
    """Return the last candle with end_period_ts <= ts, or None."""
    result = None
    for c in candles:
        if c["ts"] <= ts:
            result = c
        else:
            break
    return result


def _find_entry(
    candles_a: list[dict],
    candles_b: list[dict],
    mA: dict,
    mB: dict,
    pair_type: str,
    start_date: date,
) -> Optional[dict]:
    """
    Scan weekly Monday snapshots to find the first date where the pair was tradeable.
    Returns a dict with entry prices and date, or None.
    """
    # Determine the scan window
    if mA.get("close_time"):
        close_a = datetime.fromisoformat(mA["close_time"]).date()
    else:
        return None
    if mB.get("close_time"):
        close_b = datetime.fromisoformat(mB["close_time"]).date()
    else:
        return None

    scan_end   = min(close_a, close_b) - timedelta(days=1)
    scan_start = max(start_date, scan_end - timedelta(days=365))  # look back at most 1 year

    if scan_start >= scan_end:
        return None

    threshold = MIN_PRICE_DIFF if pair_type == "time_series" else SAME_TITLE_MIN_PRICE_DIFF

    for ts in _monday_timestamps(scan_start, scan_end):
        ca = _candle_at_or_before(candles_a, ts)
        cb = _candle_at_or_before(candles_b, ts)
        if ca is None or cb is None:
            continue

        try:
            pA = float(ca["yes_ask_close"])
            pB = float(cb["yes_ask_close"])
            nA = float(ca["no_ask_close"])
        except (ValueError, TypeError):
            continue

        if not (0.01 <= pA <= 0.99 and 0.01 <= pB <= 0.99):
            continue

        # For time_series: A must be more expensive (earlier deadline = higher YES)
        # For same_title: A must be more expensive
        if pA < pB:
            pA, pB, nA = pB, pA, float(cb["no_ask_close"])
            mA, mB = mB, mA  # swap refs (local only)

        if pA - pB < threshold:
            continue

        if nA + pB >= 1.0:
            continue  # not tradeable

        # For time_series: check deadline gap
        if pair_type == "time_series":
            gap = (close_b - close_a).days if close_b > close_a else (close_a - close_b).days
            if gap > MAX_DEADLINE_GAP_DAYS:
                continue

        entry_date = datetime.fromtimestamp(ts, tz=timezone.utc).date()
        return {
            "entry_date": entry_date,
            "pA": pA, "pB": pB, "nA": nA,
            "mA": mA, "mB": mB,
        }

    return None


# ─── Main backtest loop ───────────────────────────────────────────────────────

def run_backtest(
    hist_client: HistoricalApi,
    live_client,
    start_date: date = date(2024, 1, 1),
    initial_balance: float = 10_000.0,
    use_cache: bool = True,
) -> tuple[list[BacktestTrade], pd.DataFrame]:
    """
    Replay the arbitrage strategy on all settled Kalshi markets from start_date.

    Algorithm:
      1. Fetch all settled markets since start_date.
      2. Group into potential time-series and same-title pairs (metadata only).
      3. For each potential pair, fetch daily candlesticks for both legs.
      4. Find the first Monday where the pair was tradeable at the threshold.
      5. Compute trade size and record actual P&L from settlement outcomes.
      6. Build an equity curve from the trade timeline.

    Returns (trades, equity_df) where equity_df has columns:
      [date, portfolio_value, cash, invested]
    """
    logging.info("Starting backtest from %s with $%.2f", start_date, initial_balance)

    markets = fetch_all_settled_markets(hist_client, live_client, start_date, use_cache)
    logging.info("Total settled markets to analyze: %d", len(markets))

    # Group into potential pairs
    ts_groups    = _group_by_normalized_title(markets)
    same_groups  = _group_by_exact_title(markets)
    ts_pairs     = _extract_pairs(ts_groups,   "time_series")
    same_pairs   = _extract_pairs(same_groups, "same_title")

    logging.info("Potential pairs: %d time-series, %d same-title", len(ts_pairs), len(same_pairs))

    # Tickers that appear in any pair — only fetch candlesticks for these
    needed_tickers: dict[str, dict] = {}
    for mA, mB, _ in ts_pairs + same_pairs:
        needed_tickers[mA["ticker"]] = mA
        needed_tickers[mB["ticker"]] = mB

    logging.info("Fetching candlesticks for %d markets (cached per ticker)...", len(needed_tickers))

    # Fetch candlesticks for all needed tickers
    candles_by_ticker: dict[str, list[dict]] = {}
    for i, (ticker, m) in enumerate(needed_tickers.items()):
        if i % 50 == 0 and i > 0:
            logging.info("  Candlestick progress: %d / %d", i, len(needed_tickers))
        close_time = m.get("close_time")
        if not close_time:
            candles_by_ticker[ticker] = []
            continue
        close_dt  = datetime.fromisoformat(close_time)
        open_ts   = int(datetime(start_date.year, start_date.month, start_date.day,
                                 tzinfo=timezone.utc).timestamp())
        close_ts  = int(close_dt.timestamp()) + 86400  # one day past close
        candles_by_ticker[ticker] = fetch_daily_candlesticks(
            hist_client, ticker, open_ts, close_ts, use_cache
        )

    logging.info("Candlestick fetch complete.")

    # Pass 1: collect all potential entries without ticker-conflict filtering
    all_pairs = [(p, "time_series") for p in ts_pairs] + [(p, "same_title") for p in same_pairs]
    candidates = []

    for (mA_orig, mB_orig, canon), pair_type in all_pairs:
        candles_a = candles_by_ticker.get(mA_orig["ticker"], [])
        candles_b = candles_by_ticker.get(mB_orig["ticker"], [])

        entry = _find_entry(candles_a, candles_b, mA_orig, mB_orig, pair_type, start_date)
        if entry is None:
            continue

        mA = entry["mA"]
        mB = entry["mB"]
        pA, pB, nA = entry["pA"], entry["pB"], entry["nA"]
        entry_date = entry["entry_date"]

        # Kelly sizing: independence model for time_series, fixed prior for same_title
        profit_ratio_entry = (1.0 - nA - pB) / (nA + pB) if (nA + pB) > 0 else 0
        p = (1.0 - pA * (1.0 - pB)) if pair_type == "time_series" else SAME_TITLE_CO_RESOLVE_PROB
        q = 1.0 - p
        kelly_f = (p - q / profit_ratio_entry) if profit_ratio_entry > 0 else -1.0
        if kelly_f <= 0:
            continue  # negative Kelly — skip this pair
        kelly_f_capped = min(BUDGET_FRACTION, kelly_f)

        budget = initial_balance * kelly_f_capped
        n = max(1, int(budget / (nA + pB)))
        total_cost      = n * (nA + pB)
        expected_payoff = n * (1.0 - nA - pB)

        outcome_a = mA.get("result", "")
        outcome_b = mB.get("result", "")
        if outcome_a not in ("yes", "no") or outcome_b not in ("yes", "no"):
            continue

        actual_payoff = _compute_actual_payoff(n, pA, pB, nA, outcome_a, outcome_b)
        profit        = actual_payoff - total_cost
        profit_ratio  = profit / total_cost if total_cost > 0 else 0.0
        slippage      = actual_payoff - expected_payoff

        st_a = mA.get("settlement_ts")
        st_b = mB.get("settlement_ts")
        exit_date_a = datetime.fromisoformat(st_a).date() if st_a else entry_date
        exit_date_b = datetime.fromisoformat(st_b).date() if st_b else entry_date
        exit_date   = max(exit_date_a, exit_date_b)

        holding_days = max(1, (exit_date - entry_date).days)
        monthly_profit_ratio = profit_ratio * 30.0 / holding_days

        title_a = mA.get("title") or mA.get("subtitle") or mA.get("ticker", "")
        title_b = mB.get("title") or mB.get("subtitle") or mB.get("ticker", "")

        candidates.append(dict(
            pair_type=pair_type,
            mA=mA, mB=mB,
            pA=pA, pB=pB, nA=nA,
            entry_date=entry_date,
            exit_date=exit_date,
            n=n,
            total_cost=total_cost,
            expected_payoff=expected_payoff,
            outcome_a=outcome_a,
            outcome_b=outcome_b,
            actual_payoff=actual_payoff,
            profit=profit,
            profit_ratio=profit_ratio,
            monthly_profit_ratio=monthly_profit_ratio,
            kelly_fraction=kelly_f_capped,
            slippage=slippage,
            holding_days=holding_days,
            title_a=title_a,
            title_b=title_b,
        ))

    # Sort by monthly_profit_ratio desc; prefer same_title at equal monthly return
    candidates.sort(
        key=lambda c: (c["monthly_profit_ratio"], c["pair_type"] == "same_title"),
        reverse=True,
    )

    # Pass 2: apply ticker-conflict filter in priority order
    trades: list[BacktestTrade] = []
    active_tickers: set[str] = set()

    for c in candidates:
        mA, mB = c["mA"], c["mB"]
        if mA["ticker"] in active_tickers or mB["ticker"] in active_tickers:
            continue

        trades.append(BacktestTrade(
            pair_type=c["pair_type"],
            ticker_a=mA["ticker"],
            ticker_b=mB["ticker"],
            title_a=c["title_a"],
            title_b=c["title_b"],
            category=infer_category(mA.get("event_ticker", "")),
            entry_date=c["entry_date"],
            exit_date=c["exit_date"],
            entry_pA=c["pA"],
            entry_pB=c["pB"],
            entry_nA=c["nA"],
            n=c["n"],
            total_cost=c["total_cost"],
            outcome_a=c["outcome_a"],
            outcome_b=c["outcome_b"],
            actual_payoff=c["actual_payoff"],
            profit=c["profit"],
            profit_ratio=c["profit_ratio"],
            monthly_profit_ratio=c["monthly_profit_ratio"],
            kelly_fraction=c["kelly_fraction"],
            expected_payoff=c["expected_payoff"],
            slippage=c["slippage"],
            holding_days=c["holding_days"],
        ))

        active_tickers.add(mA["ticker"])
        active_tickers.add(mB["ticker"])

    logging.info(
        "Backtest complete: %d trades, %d profitable",
        len(trades),
        sum(1 for t in trades if t.profit > 0),
    )

    equity_df = _build_equity_curve(trades, start_date, initial_balance)
    return trades, equity_df


# ─── Equity curve construction ────────────────────────────────────────────────

def _build_equity_curve(
    trades: list[BacktestTrade],
    start_date: date,
    initial_balance: float,
) -> pd.DataFrame:
    """Build a daily equity curve from the trade list."""
    today = date.today()
    dates = [start_date + timedelta(days=i) for i in range((today - start_date).days + 1)]

    cash_changes: dict[date, float] = defaultdict(float)
    for t in trades:
        cash_changes[t.entry_date] -= t.total_cost
        cash_changes[t.exit_date]  += t.actual_payoff

    rows = []
    cash = initial_balance
    for d in dates:
        cash += cash_changes.get(d, 0.0)
        rows.append({"date": d, "portfolio_value": cash})

    df = pd.DataFrame(rows)
    df["daily_return"] = df["portfolio_value"].pct_change().fillna(0.0)
    return df
