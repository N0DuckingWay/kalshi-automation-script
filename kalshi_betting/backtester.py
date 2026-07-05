"""
File: backtester.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Replays the arbitrage strategy on the full history of settled Kalshi markets.
    Groups settled markets into potential time-series and same-title pairs, fetches
    daily candlestick price series for each involved market, then scans weekly
    Monday snapshots to find the first date each pair was tradeable at the required
    threshold. Applies Kelly sizing to compute trade size, records actual P&L from
    settlement outcomes, deduplicates overlapping pairs by priority, and builds
    a daily equity curve. Results feed into dashboard.py for visualization.

Dependencies:
    Imports normalize_title from scanner.py; fee helpers from config.py; market
    fetching and candlestick functions from historical.py. Exports BacktestTrade
    (consumed by dashboard.py) and run_backtest() (called by backtest.py).

Notes:
    The backtester uses a two-pass approach: Pass 1 collects all potential entries
    without ticker conflict checking, sorts by priority, then Pass 2 applies a
    greedy ticker-conflict filter so each market ticker appears in at most one trade.
    This mirrors the live bot's behavior of not entering a market already held.
"""
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from .config import (
    BUDGET_FRACTION,
    MAX_DEADLINE_GAP_DAYS,
    MIN_PRICE_DIFF,
    SAME_TITLE_CO_RESOLVE_PROB,
    SAME_TITLE_MIN_PRICE_DIFF,
    fee_leg_exact,
    fee_per_pair_approx,
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
    """
    Complete record of a single simulated arbitrage trade from the backtest.

    Captures both the trade setup (entry prices, sizing, pair metadata) and the
    final outcome (settlement results, P&L, slippage) for use in performance analysis
    and dashboard generation.

    Attributes:
        pair_type (str): Strategy variant used: "time_series" or "same_title".
        ticker_a (str): Kalshi ticker of market A (the NO leg).
        ticker_b (str): Kalshi ticker of market B (the YES leg).
        title_a (str): Display title of market A.
        title_b (str): Display title of market B.
        category (str): Human-readable market category inferred from event_ticker prefix
            (e.g. "Crypto", "Sports", "Politics").
        entry_date (date): The Monday on which the trade was first tradeable and sized.
        exit_date (date): The date the later-settling market resolved; marks when cash returned.
        entry_pA (float): YES ask price of market A at entry. Range: [0.01, 0.99].
        entry_pB (float): YES ask price of market B at entry. Range: [0.01, 0.99].
        entry_nA (float): NO ask price of market A at entry (≈ 1 − yes_bid_A). Range: [0.01, 0.99].
        n (int): Number of contracts bought on each leg (x = y = n). Always >= 1.
        total_cost (float): Total dollar cost of the position: n * (entry_nA + entry_pB).
        outcome_a (str): Settlement result of market A — "yes" or "no".
        outcome_b (str): Settlement result of market B — "yes" or "no".
        actual_payoff (float): Dollar value received at settlement based on outcomes.
        profit (float): actual_payoff − total_cost. May be negative for same_title loss scenarios.
        profit_ratio (float): profit / total_cost. Return on invested capital.
        monthly_profit_ratio (float): profit_ratio scaled to 30 days:
            profit_ratio * 30 / holding_days.
        kelly_fraction (float): Capped Kelly fraction used for sizing, <= BUDGET_FRACTION.
        expected_payoff (float): Guaranteed floor payoff: n * (1 − entry_nA − entry_pB).
        slippage (float): actual_payoff − expected_payoff. Positive means better than the
            guaranteed floor (e.g. both markets resolved favorably).
        holding_days (int): Calendar days between entry_date and exit_date. Always >= 1.
    """
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
    expected_payoff: float  # n * (1 - nA - pB) minus fees — the guaranteed floor
    slippage: float         # actual_payoff - expected_payoff
    holding_days: int


def _compute_actual_payoff(n: int, pA: float, pB: float, nA: float,
                           outcome_a: str, outcome_b: str) -> float:
    """
    Compute the actual dollar payoff for a completed arbitrage trade based on settlement outcomes.

    The strategy is n NO contracts on market A + n YES contracts on market B.
    Kalshi taker fees are charged at execution time regardless of outcome, so
    the fee amount is subtracted from every scenario's payoff.

    Payoff table (n contracts, net of fees):
      A=YES, B=YES: n*(1-pB) - n*nA - fee  [B pays out, A costs nA at entry]
      A=NO,  B=YES: n*(1-nA) + n*(1-pB) - fee  [both legs pay out — best scenario]
      A=NO,  B=NO:  n*(1-nA) - n*pB - fee  [NO on A pays, YES on B expires worthless]
      A=YES, B=NO:  -(n*nA + n*pB) - fee  [loss scenario — both legs expire worthless]

    The A=YES, B=NO scenario is the only loss scenario. For time-series pairs this
    is unlikely but possible; for same-title pairs it should be very rare (same question).

    Args:
        n (int): Number of contracts bought on each leg.
        pA (float): YES ask price of market A at entry (cost to buy YES on A).
        pB (float): YES ask price of market B at entry (cost to buy YES on B).
        nA (float): NO ask price of market A at entry (cost to buy NO on A).
        outcome_a (str): Settlement result of market A — "yes" or "no".
        outcome_b (str): Settlement result of market B — "yes" or "no".

    Returns:
        float: Net dollar payoff received at settlement (after Kalshi taker fees).
            May be negative in the A=YES, B=NO loss scenario.
    """
    # Total taker fee: ceiling-rounded per leg, charged regardless of outcome
    fee = fee_leg_exact(n, nA) + fee_leg_exact(n, pB)
    if outcome_a == "yes" and outcome_b == "yes":
        # B resolves YES (pays 1-pB per contract) but A resolves YES (NO position worthless)
        return n * (1.0 - pB) - n * nA - fee
    if outcome_a == "no" and outcome_b == "yes":
        # Both legs pay out — the maximum profit scenario
        return n * (1.0 - nA) + n * (1.0 - pB) - fee
    if outcome_a == "no" and outcome_b == "no":
        # NO on A pays out; YES on B expires worthless (we lose the pB cost)
        return n * (1.0 - nA) - n * pB - fee
    # outcome_a == "yes", outcome_b == "no" — the loss scenario:
    # NO on A expires worthless; YES on B expires worthless; we lose the full entry cost
    return -(n * nA + n * pB) - fee


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
        norm = normalize_title(raw)  # returns str — lowercased title with all date/time tokens stripped
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
    """
    Generate a list of Unix timestamps for every Monday in the given date range.

    Each timestamp corresponds to 09:00 UTC on the Monday. The backtest scans
    these weekly checkpoints to simulate the bot's Monday morning trading schedule.

    Args:
        start_date (date): First date of the scan range (inclusive). The function
            advances to the first Monday on or after this date.
        end_date (date): Last date of the scan range (inclusive).

    Returns:
        list[int]: Unix timestamps (09:00 UTC) for every Monday in [start_date, end_date].
            Returns an empty list if start_date > end_date or no Monday falls in range.
    """
    d = start_date
    # Advance to the first Monday (weekday() == 0) if start_date is not already one
    while d.weekday() != 0:
        d += timedelta(days=1)
    ts_list = []
    while d <= end_date:
        ts_list.append(int(datetime(d.year, d.month, d.day, 9, 0, tzinfo=UTC).timestamp()))
        d += timedelta(weeks=1)
    return ts_list


def _candle_at_or_before(candles: list[dict], ts: int) -> dict | None:
    """
    Find the most recent candlestick at or before a given Unix timestamp.

    Candles are assumed to be sorted ascending by their "ts" field. Returns the
    last candle whose end_period_ts is <= ts, or None if no such candle exists.
    This is used to read the closing price on or before a given Monday snapshot.

    Args:
        candles (list[dict]): List of candle dicts with a "ts" key (unix timestamp).
            Must be sorted ascending by "ts".
        ts (int): The target Unix timestamp to search at or before.

    Returns:
        Optional[dict]: The last candle dict with ts <= the target, or None if all
            candles are after the target timestamp or the list is empty.
    """
    result = None
    for c in candles:
        if c["ts"] <= ts:
            result = c
        else:
            # Candles are sorted ascending, so once we exceed ts we can stop
            break
    return result


def _find_entry(
    candles_a: list[dict],
    candles_b: list[dict],
    mA: dict,
    mB: dict,
    pair_type: str,
    start_date: date,
) -> dict | None:
    """
    Find the first Monday where a potential pair was tradeable at the required threshold.

    Scans weekly Monday snapshots from the backtest start date up to one calendar
    year before the earlier of the two market close dates. At each Monday, reads
    the candlestick prices, canonicalizes which market is the expensive side (market A),
    applies the price gap and fee filters from the live trading logic, and returns
    the entry data for the first qualifying week.

    Scanning stops at the earlier close date (not the later one) because after
    the first market closes, the pair is no longer open for entry.

    Args:
        candles_a (list[dict]): Daily candlestick dicts for market A (sorted by ts).
        candles_b (list[dict]): Daily candlestick dicts for market B (sorted by ts).
        mA (dict): Market A metadata dict (with "close_time", "result", etc.).
        mB (dict): Market B metadata dict (with "close_time", "result", etc.).
        pair_type (str): "time_series" or "same_title" — controls price gap threshold
            and deadline gap check.
        start_date (date): Backtest start date; no entry is recorded before this date.

    Returns:
        Optional[dict]: A dict with keys "entry_date" (date), "pA" (float), "pB" (float),
            "nA" (float), "mA" (dict), "mB" (dict) for the first qualifying Monday.
            Returns None if no qualifying Monday was found in the scan window.
    """
    # Both markets must have a close_time; without it we can't determine the scan window
    if mA.get("close_time"):
        close_a = datetime.fromisoformat(mA["close_time"]).date()
    else:
        return None
    if mB.get("close_time"):
        close_b = datetime.fromisoformat(mB["close_time"]).date()
    else:
        return None

    # Scan up to (but not including) the day the earlier market closes —
    # after that, the pair is no longer fully open for entry
    scan_end   = min(close_a, close_b) - timedelta(days=1)
    # Look back at most 1 year from scan_end to keep the scan window manageable
    scan_start = max(start_date, scan_end - timedelta(days=365))

    if scan_start >= scan_end:
        return None

    # Use the appropriate minimum price gap for this pair type
    threshold = MIN_PRICE_DIFF if pair_type == "time_series" else SAME_TITLE_MIN_PRICE_DIFF

    for ts in _monday_timestamps(scan_start, scan_end):
        # Read the closing prices at this Monday snapshot
        ca = _candle_at_or_before(candles_a, ts)
        cb = _candle_at_or_before(candles_b, ts)
        if ca is None or cb is None:
            continue

        try:
            p_a_raw = float(ca["yes_ask_close"])
            p_b_raw = float(cb["yes_ask_close"])
            n_a_raw = float(ca["no_ask_close"])
            n_b_raw = float(cb["no_ask_close"])
        except (ValueError, TypeError):
            continue

        # Skip settled or illiquid candles (prices at the extreme ends of the range)
        if not (0.01 <= p_a_raw <= 0.99 and 0.01 <= p_b_raw <= 0.99):
            continue

        # Canonicalize per iteration so the swap never leaks to the next Monday.
        # Market A is the more expensive side; nA is A's NO ask.
        if p_a_raw >= p_b_raw:
            mA_i, mB_i = mA, mB
            pA, pB, nA = p_a_raw, p_b_raw, n_a_raw
        else:
            mA_i, mB_i = mB, mA
            pA, pB, nA = p_b_raw, p_a_raw, n_b_raw

        # Enforce the minimum price gap for this pair type
        if pA - pB < threshold:
            continue

        # Check that the gross spread exceeds the continuous fee estimate
        if (1.0 - nA - pB) <= fee_per_pair_approx(nA, pB):
            continue

        # For time_series pairs: enforce the maximum 30-day deadline gap constraint
        if pair_type == "time_series":
            gap = (close_b - close_a).days if close_b > close_a else (close_a - close_b).days
            if gap > MAX_DEADLINE_GAP_DAYS:
                continue

        entry_date = datetime.fromtimestamp(ts, tz=UTC).date()
        return {
            "entry_date": entry_date,
            "pA": pA, "pB": pB, "nA": nA,
            "mA": mA_i, "mB": mB_i,
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
      [date, portfolio_value, daily_return]
    """
    logging.info("Starting backtest from %s with $%.2f", start_date, initial_balance)

    # Fetch all settled markets from start_date onward (uses disk cache if available)
    markets = fetch_all_settled_markets(hist_client, live_client, start_date, use_cache)
    logging.info("Total settled markets to analyze: %d", len(markets))

    # Group settled markets into potential pairs using the same logic as the live scanner
    ts_groups    = _group_by_normalized_title(markets)
    same_groups  = _group_by_exact_title(markets)
    ts_pairs     = _extract_pairs(ts_groups,   "time_series")
    same_pairs   = _extract_pairs(same_groups, "same_title")

    logging.info("Potential pairs: %d time-series, %d same-title", len(ts_pairs), len(same_pairs))

    # Collect only the tickers that actually appear in a potential pair to avoid
    # fetching candlesticks for thousands of unrelated markets
    needed_tickers: dict[str, dict] = {}
    for mA, mB, _ in ts_pairs + same_pairs:
        needed_tickers[mA["ticker"]] = mA
        needed_tickers[mB["ticker"]] = mB

    logging.info("Fetching candlesticks for %d markets (cached per ticker)...", len(needed_tickers))

    # Fetch daily candlestick price series for all needed tickers.
    # Results are cached per ticker so a second run is much faster.
    candles_by_ticker: dict[str, list[dict]] = {}
    for i, (ticker, m) in enumerate(needed_tickers.items()):
        if i % 50 == 0 and i > 0:
            logging.info("  Candlestick progress: %d / %d", i, len(needed_tickers))
        close_time = m.get("close_time")
        if not close_time:
            candles_by_ticker[ticker] = []
            continue
        close_dt  = datetime.fromisoformat(close_time)
        # open_ts: start of the backtest window
        open_ts   = int(datetime(start_date.year, start_date.month, start_date.day,
                                 tzinfo=UTC).timestamp())
        # close_ts: one day past market close to include the final candle
        close_ts  = int(close_dt.timestamp()) + 86400
        # Returns list[dict] with keys: ts (unix int), yes_ask_close (float), no_ask_close (float)
        candles_by_ticker[ticker] = fetch_daily_candlesticks(
            hist_client, ticker, open_ts, close_ts, use_cache
        )

    logging.info("Candlestick fetch complete.")

    # ── Pass 1: collect all tradeable entries without conflict filtering ──────

    # Combine both pair types for the scan loop
    all_pairs = [(p, "time_series") for p in ts_pairs] + [(p, "same_title") for p in same_pairs]
    candidates = []

    for (mA_orig, mB_orig, canon), pair_type in all_pairs:
        candles_a = candles_by_ticker.get(mA_orig["ticker"], [])
        candles_b = candles_by_ticker.get(mB_orig["ticker"], [])

        # Find the first Monday where this pair was tradeable at the threshold prices
        entry = _find_entry(candles_a, candles_b, mA_orig, mB_orig, pair_type, start_date)
        if entry is None:
            continue

        # Unpack entry — mA/mB may have been swapped inside _find_entry to canonicalize
        mA = entry["mA"]
        mB = entry["mB"]
        pA, pB, nA = entry["pA"], entry["pB"], entry["nA"]
        entry_date = entry["entry_date"]

        # ── Kelly sizing ──────────────────────────────────────────────────────
        # Compute the net spread after the continuous fee approximation
        net_spread = (1.0 - nA - pB) - fee_per_pair_approx(nA, pB)
        profit_ratio_entry = net_spread / (nA + pB) if net_spread > 0 else 0.0

        # Probability model: independence estimate for time_series, fixed prior for same_title
        p = (1.0 - pA * (1.0 - pB)) if pair_type == "time_series" else SAME_TITLE_CO_RESOLVE_PROB
        q = 1.0 - p

        # Kelly formula: f* = p - q/b; negative means no edge
        kelly_f = (p - q / profit_ratio_entry) if profit_ratio_entry > 0 else -1.0
        if kelly_f <= 0:
            # Kelly fraction is non-positive — the pair has no positive expected value
            continue
        # Cap at BUDGET_FRACTION (20%) to avoid over-concentration
        kelly_f_capped = min(BUDGET_FRACTION, kelly_f)

        # Convert the Kelly fraction to an integer contract count
        budget = initial_balance * kelly_f_capped
        n = max(1, int(budget / (nA + pB)))
        total_cost      = n * (nA + pB)
        # Expected guaranteed payoff: gross spread per contract minus exact per-leg fees
        expected_payoff = n * (1.0 - nA - pB) - fee_leg_exact(n, nA) - fee_leg_exact(n, pB)

        # Skip pairs where the settlement result is missing or non-binary
        outcome_a = mA.get("result", "")
        outcome_b = mB.get("result", "")
        if outcome_a not in ("yes", "no") or outcome_b not in ("yes", "no"):
            continue

        # Compute actual P&L based on what the markets resolved to
        actual_payoff = _compute_actual_payoff(n, pA, pB, nA, outcome_a, outcome_b)
        profit        = actual_payoff - total_cost
        profit_ratio  = profit / total_cost if total_cost > 0 else 0.0
        # Slippage = how much better (or worse) the actual payoff was vs. the guaranteed floor
        slippage      = actual_payoff - expected_payoff

        # Determine the exit date as the later of the two settlement timestamps
        st_a = mA.get("settlement_ts")
        st_b = mB.get("settlement_ts")
        exit_date_a = datetime.fromisoformat(st_a).date() if st_a else entry_date
        exit_date_b = datetime.fromisoformat(st_b).date() if st_b else entry_date
        exit_date   = max(exit_date_a, exit_date_b)

        holding_days = max(1, (exit_date - entry_date).days)
        # Normalize profit ratio to a 30-day equivalent for fair comparison across durations
        monthly_profit_ratio = profit_ratio * 30.0 / holding_days

        # Use title > subtitle > ticker as the display label for each market
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

    # Sort by monthly_profit_ratio descending; prefer same_title at equal monthly return
    candidates.sort(
        key=lambda c: (c["monthly_profit_ratio"], c["pair_type"] == "same_title"),
        reverse=True,
    )

    # ── Pass 2: greedy ticker-conflict filter ─────────────────────────────────
    # In priority order, add each candidate only if neither of its tickers is already
    # used by a higher-priority trade — mirrors the live bot's no-re-entry rule
    trades: list[BacktestTrade] = []
    active_tickers: set[str] = set()

    for c in candidates:
        mA, mB = c["mA"], c["mB"]
        # Skip if either ticker is already committed to a higher-priority trade
        if mA["ticker"] in active_tickers or mB["ticker"] in active_tickers:
            continue

        trades.append(BacktestTrade(
            pair_type=c["pair_type"],
            ticker_a=mA["ticker"],
            ticker_b=mB["ticker"],
            title_a=c["title_a"],
            title_b=c["title_b"],
            # infer_category maps the event_ticker prefix to a human-readable label (e.g. "Crypto")
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

        # Mark both tickers as active so no overlapping pair is added below them
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
    """
    Construct a daily equity curve DataFrame from the list of backtest trades.

    For each trade, subtracts the total_cost from cash on the entry_date and
    adds the actual_payoff on the exit_date. This models a simple accounting
    treatment where capital is deployed on entry and returned at settlement.

    Args:
        trades (list[BacktestTrade]): Completed backtest trades with entry_date,
            exit_date, total_cost, and actual_payoff populated.
        start_date (date): The first date of the equity curve (initial balance day).
        initial_balance (float): Starting portfolio value in dollars.

    Returns:
        pd.DataFrame: DataFrame with one row per calendar day from start_date to
            today (UTC), with columns:
            - "date" (date): Calendar date.
            - "portfolio_value" (float): Cumulative portfolio value in dollars.
            - "daily_return" (float): Fractional daily return (pct_change of portfolio_value).
    """
    # entry_date and exit_date come from UTC-derived timestamps, so use UTC today
    # here as well — otherwise `date.today()` in a non-UTC timezone can drop or add
    # a day around the boundary and misalign the equity curve.
    today = datetime.now(UTC).date()
    dates = [start_date + timedelta(days=i) for i in range((today - start_date).days + 1)]

    # Accumulate cash inflows and outflows per date
    cash_changes: dict[date, float] = defaultdict(float)
    for t in trades:
        # Capital leaves the portfolio on entry day (we pay the cost)
        cash_changes[t.entry_date] -= t.total_cost
        # Capital returns to the portfolio on exit day (settlement payoff received)
        cash_changes[t.exit_date]  += t.actual_payoff

    rows = []
    cash = initial_balance
    for d in dates:
        # Apply any net cash change for this day (may be zero if no trades entered/exited)
        cash += cash_changes.get(d, 0.0)
        rows.append({"date": d, "portfolio_value": cash})

    df = pd.DataFrame(rows)
    # Compute fractional daily returns; the first row has no prior day so it gets 0.0
    df["daily_return"] = df["portfolio_value"].pct_change().fillna(0.0)
    return df
