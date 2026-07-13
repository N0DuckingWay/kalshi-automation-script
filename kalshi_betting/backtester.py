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
    (prices, dates, Kelly fraction — no sizing) and keeps only the best entry per
    title group, mirroring the live scanners' one-pair-per-group rule. Pass 2 walks
    entries in chronological order (priority-ordered within a date using the
    ENTRY-TIME expected return, never realized results), maintains a running cash
    balance — sizing each trade against the cash available at entry and releasing
    settlement receipts on exit dates — and applies a greedy ticker-conflict filter
    so each market ticker appears in at most one trade. This mirrors the live bot's
    Kelly sizing against the current balance and its no-re-entry rule.
"""
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from .config import (
    BUDGET_FRACTION,
    MAX_DEADLINE_GAP_DAYS,
    SAME_TITLE_CO_RESOLVE_PROB,
    SAME_TITLE_MIN_PRICE_DIFF,
    fee_leg_exact,
    fee_per_pair_approx,
    min_price_diff_for_gap,
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
        total_cost (float): Dollar cost of the contracts: n * (entry_nA + entry_pB).
            Excludes taker fees (see fees).
        fees (float): Exact ceiling-rounded taker fee for both legs, charged at entry.
        outcome_a (str): Settlement result of market A — "yes" or "no".
        outcome_b (str): Settlement result of market B — "yes" or "no".
        actual_payoff (float): Gross dollar value received at settlement: $1 per
            contract for each leg that pays out (0, n, or 2n). Fees and entry cost
            are NOT deducted here — they are accounted in profit.
        profit (float): actual_payoff − total_cost − fees. Negative in the
            A=YES, B=NO loss scenario.
        profit_ratio (float): profit / (total_cost + fees). Return on the cash
            actually invested.
        monthly_profit_ratio (float): Realized profit_ratio scaled to 30 days:
            profit_ratio * 30 / holding_days. Reporting only — trade selection
            uses the entry-time expected ratio to avoid look-ahead bias.
        kelly_fraction (float): Capped Kelly fraction used for sizing, <= BUDGET_FRACTION.
        expected_payoff (float): Guaranteed NET profit floor:
            n * (1 − entry_nA − entry_pB) − fees. Always > 0 for recorded trades.
        slippage (float): profit − expected_payoff. Positive means better than the
            guaranteed floor (e.g. both markets resolved favorably); negative only
            in the loss scenario.
        holding_days (int): Calendar days between entry_date and exit_date. Always >= 1.
        balance_at_entry (float): Simulated cash balance in dollars immediately
            before this trade's entry deduction — the base the Kelly budget used.
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
    fees: float          # exact both-leg taker fees, charged at entry
    outcome_a: str       # "yes" | "no"
    outcome_b: str       # "yes" | "no"
    actual_payoff: float
    profit: float
    profit_ratio: float
    monthly_profit_ratio: float  # realized profit_ratio * 30 / holding_days (reporting only)
    kelly_fraction: float        # capped Kelly fraction used for sizing
    expected_payoff: float  # n * (1 - nA - pB) minus fees — the guaranteed NET floor
    slippage: float         # profit - expected_payoff
    holding_days: int
    balance_at_entry: float  # simulated cash available when the trade was sized


def _settlement_receipt(n: int, outcome_a: str, outcome_b: str) -> float:
    """
    Compute the gross dollar amount received at settlement for one pair trade.

    The strategy is n NO contracts on market A + n YES contracts on market B.
    Each contract pays exactly $1 when its side wins and $0 otherwise, so the
    receipt is independent of entry prices:

      A=YES, B=YES: n   [YES on B pays; NO on A worthless]
      A=NO,  B=YES: 2n  [both legs pay — best scenario]
      A=NO,  B=NO:  n   [NO on A pays; YES on B worthless]
      A=YES, B=NO:  0   [loss scenario — both legs worthless]

    Entry cost and taker fees are deliberately NOT deducted here — callers
    subtract them exactly once when computing profit and the equity curve.

    Args:
        n (int): Number of contracts bought on each leg.
        outcome_a (str): Settlement result of market A — "yes" or "no".
        outcome_b (str): Settlement result of market B — "yes" or "no".

    Returns:
        float: Gross settlement receipt in dollars: 0.0, n, or 2n.
    """
    receipt = 0.0
    if outcome_a == "no":
        # The NO leg on market A pays $1 per contract
        receipt += n
    if outcome_b == "yes":
        # The YES leg on market B pays $1 per contract
        receipt += n
    return receipt


# ─── Pair grouping (metadata only, no prices) ─────────────────────────────────

def _pair_key(m: dict) -> str:
    """
    Combined grouping key for a market dict — event title joined with market title.

    Mirrors scanner.pair_key() for the backtester's dict-based market representation.
    The event_title prefix prevents cross-event option-label collisions in MVE
    markets — e.g. two markets both titled "Trump" in unrelated events will have
    different event titles and therefore won't be grouped together.

    Falls back to the bare title when event_title is missing (older cache files
    or non-MVE markets).
    """
    event_title = m.get("event_title") or ""
    title = m.get("title") or m.get("subtitle") or m.get("ticker", "")
    if not event_title:
        return title
    return f"{event_title} | {title}"


def _group_by_exact_title(markets: list[dict]) -> dict[tuple, list[dict]]:
    """Group markets by exact (event_title, title, subtitle) tuple for same-title pair detection.

    Three-element key: the event_title component prevents cross-event option-label
    collisions in MVE markets; (title, subtitle) distinguishes markets within an event.
    """
    groups: dict = defaultdict(list)
    for m in markets:
        event_title = m.get("event_title") or ""
        title    = m.get("title") or ""
        subtitle = m.get("subtitle") or ""
        if title or subtitle:
            groups[(event_title, title, subtitle)].append(m)
    return {k: v for k, v in groups.items() if len(v) >= 2}


def _group_by_normalized_title(markets: list[dict]) -> dict[str, list[dict]]:
    """Group markets by date-stripped combined key (event_title + title) for time-series pair detection."""
    groups: dict = defaultdict(list)
    for m in markets:
        # _pair_key combines event_title + market title before normalization so that
        # two MVE markets sharing an option label across unrelated events do not collide.
        norm = normalize_title(_pair_key(m))
        if norm:
            groups[norm].append(m)
    return {k: v for k, v in groups.items() if len(v) >= 2}


def _extract_pairs(groups: dict, pair_type: str) -> list[tuple[dict, dict, str, object]]:
    """
    Return list of (market_a, market_b, canonical_title, group_key) tuples where
    the two markets have different event_tickers. No price filtering at this stage.

    Group keys may be:
      - a string (normalized-title group from _group_by_normalized_title), or
      - a 3-tuple (event_title, title, subtitle) from _group_by_exact_title.
    For the 3-tuple form, the display canonical is taken from title-or-subtitle,
    but the FULL 3-tuple (including event_title) is also returned as group_key —
    the one-pair-per-group dedup in run_backtest must key on the full group, not
    just the display title, or two unrelated events sharing an option label
    (e.g. "Trump" in two different events) would collide into a single group
    and silently drop one of the two legitimate pairs.
    """
    pairs = []
    for key, members in groups.items():
        if isinstance(key, str):
            canon = key
        else:
            # 3-tuple (event_title, title, subtitle) — use title-or-subtitle for display
            canon = key[1] or key[2]
        seen: set[frozenset] = set()
        for i, mA in enumerate(members):
            for mB in members[i + 1:]:
                if mA["event_ticker"] == mB["event_ticker"]:
                    continue
                pair_key = frozenset([mA["ticker"], mB["ticker"]])
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                pairs.append((mA, mB, canon, key))
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
    the candlestick prices, applies the price gap, price-sum, and fee filters from
    the live trading logic, and returns the entry data for the first qualifying week.

    Direction rules mirror the live scanner exactly:
      - time_series: market A is fixed as the EARLIER-closing contract, and an
        entry requires pA − pB >= the deadline-gap-tiered threshold from
        min_price_diff_for_gap (15% for gaps <= 15 days, 30% for 16-30 days) —
        the earlier contract priced higher is the anomaly. A pricier later
        contract is normal term structure and is never traded.
      - same_title: market A is canonicalized per Monday as the more expensive
        side (the two contracts ask the identical question, so direction is
        price-only).

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

    # A single-day window (scan_start == scan_end) is still a valid scan window
    if scan_start > scan_end:
        return None

    if pair_type == "time_series":
        # Live-scanner invariant: market A is the EARLIER-closing contract.
        # Never swap by price — the trade only exists when the earlier contract
        # is priced higher (checked per Monday below).
        if close_b < close_a:
            mA, mB = mB, mA
            candles_a, candles_b = candles_b, candles_a
            close_a, close_b = close_b, close_a
        # Deadline gap is loop-invariant: pairs more than 30 days apart are too
        # weakly correlated for the time-series assumption to hold reliably
        gap_days = (close_b - close_a).days
        if gap_days > MAX_DEADLINE_GAP_DAYS:
            return None
        # Tier the required price gap by deadline distance (15% for gaps
        # <= 15 days, 30% for 16-30 days) — mirrors scanner.find_time_series_pairs
        threshold = min_price_diff_for_gap(gap_days)
    else:
        # same_title pairs have no deadline-gap concept — flat 5% threshold
        threshold = SAME_TITLE_MIN_PRICE_DIFF

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

        if pair_type == "time_series":
            # A stays the earlier-closing market — no price canonicalization
            mA_i, mB_i = mA, mB
            pA, pB, nA = p_a_raw, p_b_raw, n_a_raw
        else:
            # same_title: canonicalize per iteration so the swap never leaks to
            # the next Monday. Market A is the more expensive side this week.
            if p_a_raw >= p_b_raw:
                mA_i, mB_i = mA, mB
                pA, pB, nA = p_a_raw, p_b_raw, n_a_raw
            else:
                mA_i, mB_i = mB, mA
                pA, pB, nA = p_b_raw, p_a_raw, n_b_raw

        # Enforce the minimum price gap for this pair type. For time_series this
        # is directional: the earlier contract must be the expensive one.
        if pA - pB < threshold:
            continue

        # The NO leg's price must also be a live (0.01–0.99) quote — mirrors the
        # live pipeline's (0, 1) price validation in compute_trade
        if not (0.01 <= nA <= 0.99):
            continue

        # Live orderbook-depth parity: enrich_with_orderbook_prices only keeps
        # contracts whose combined price leaves the required gap
        # (nA + pB <= 1 - threshold) — apply the same cut to candle entries
        if nA + pB > 1.0 - threshold:
            continue

        # Check that the gross spread exceeds the continuous fee estimate
        if (1.0 - nA - pB) <= fee_per_pair_approx(nA, pB):
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
      4. Find the first Monday where the pair was tradeable at the threshold;
         keep only the best entry per title group (live one-pair-per-group rule).
      5. Walk entries chronologically with a running cash balance: Kelly-size
         each trade against the cash available at entry, and record actual P&L
         from settlement outcomes.
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
    for mA, mB, _, _ in ts_pairs + same_pairs:
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

    # ── Pass 1: collect all tradeable entries (no sizing, no conflict filter) ──

    # Combine both pair types for the scan loop
    all_pairs = [(p, "time_series") for p in ts_pairs] + [(p, "same_title") for p in same_pairs]
    candidates = []

    for (mA_orig, mB_orig, canon, group_key), pair_type in all_pairs:
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

        # ── Kelly fraction (sizing happens in Pass 2 against running cash) ────
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

        # Skip pairs where the settlement result is missing or non-binary
        outcome_a = mA.get("result", "")
        outcome_b = mB.get("result", "")
        if outcome_a not in ("yes", "no") or outcome_b not in ("yes", "no"):
            continue

        # Determine the exit date as the later of the two settlement timestamps
        st_a = mA.get("settlement_ts")
        st_b = mB.get("settlement_ts")
        exit_date_a = datetime.fromisoformat(st_a).date() if st_a else entry_date
        exit_date_b = datetime.fromisoformat(st_b).date() if st_b else entry_date
        exit_date   = max(exit_date_a, exit_date_b)

        holding_days = max(1, (exit_date - entry_date).days)

        # Entry-time priority metric: the expected return normalized to 30 days
        # using only information available at entry (entry prices and the market
        # close dates). Sorting Pass 2 by REALIZED returns would leak settlement
        # outcomes into trade selection (look-ahead bias).
        close_a_d = datetime.fromisoformat(mA["close_time"]).date()
        close_b_d = datetime.fromisoformat(mB["close_time"]).date()
        expected_days = max(1, (max(close_a_d, close_b_d) - entry_date).days)
        entry_monthly_ratio = profit_ratio_entry * 30.0 / expected_days

        # Use title > subtitle > ticker as the display label for each market
        title_a = mA.get("title") or mA.get("subtitle") or mA.get("ticker", "")
        title_b = mB.get("title") or mB.get("subtitle") or mB.get("ticker", "")

        candidates.append({
            "pair_type": pair_type,
            "canon": canon,
            "group_key": group_key,
            "mA": mA, "mB": mB,
            "pA": pA, "pB": pB, "nA": nA,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "outcome_a": outcome_a,
            "outcome_b": outcome_b,
            "kelly_f_capped": kelly_f_capped,
            "entry_monthly_ratio": entry_monthly_ratio,
            "holding_days": holding_days,
            "title_a": title_a,
            "title_b": title_b,
        })

    # Keep only the single best candidate per title group — mirrors the live
    # scanners, which keep one pair per normalized-title / exact-title group so
    # the portfolio isn't flooded with near-identical correlated positions.
    # Dedup on the FULL group_key (which includes event_title for same_title
    # groups), not the display-only canon — two unrelated events sharing an
    # option label (e.g. "Trump" in two different events) have the same canon
    # but distinct group_keys, and must remain two separate candidates.
    best_by_group: dict = {}
    for c in candidates:
        key = (c["pair_type"], c["group_key"])
        cur = best_by_group.get(key)
        if cur is None or c["entry_monthly_ratio"] > cur["entry_monthly_ratio"]:
            best_by_group[key] = c
    candidates = list(best_by_group.values())

    # Chronological order for the cash simulation; within one entry date, take
    # the best ENTRY-TIME expected return first (same_title preferred at ties,
    # matching strategy.select_portfolio)
    candidates.sort(
        key=lambda c: (c["entry_date"], -c["entry_monthly_ratio"], c["pair_type"] != "same_title"),
    )

    # ── Pass 2: chronological cash-constrained greedy selection ───────────────
    # Walk entries in date order, maintaining a running cash balance: each trade
    # is Kelly-sized against the cash available at its entry (like the live bot
    # sizing against the current account balance), and settlement receipts return
    # to cash on their exit dates. A ticker-conflict filter mirrors the live
    # bot's no-re-entry rule.
    trades: list[BacktestTrade] = []
    active_tickers: set[str] = set()
    cash = initial_balance
    pending_exits: list[tuple[date, float]] = []  # (exit_date, settlement receipt)

    for c in candidates:
        d = c["entry_date"]
        # Release settlement receipts from trades that exited on or before this entry
        cash += sum(amt for ed, amt in pending_exits if ed <= d)
        pending_exits = [(ed, amt) for ed, amt in pending_exits if ed > d]

        mA, mB = c["mA"], c["mB"]
        # Skip if either ticker is already committed to an earlier trade
        if mA["ticker"] in active_tickers or mB["ticker"] in active_tickers:
            continue

        nA, pB = c["nA"], c["pB"]

        # Kelly sizing against the cash available NOW, not the initial balance
        budget = cash * c["kelly_f_capped"]
        n = int(budget / (nA + pB))
        if n < 1:
            # Kelly budget can't afford one contract — live compute_trade skips too
            continue

        total_cost = n * (nA + pB)
        # Exact ceiling-rounded taker fees for both legs, charged at entry
        fees = fee_leg_exact(n, nA) + fee_leg_exact(n, pB)
        # Guaranteed NET profit floor after exact fees — reject if the ceiling
        # rounding ate the margin (mirrors live compute_trade's min_payoff gate)
        expected_payoff = n * (1.0 - nA - pB) - fees
        if expected_payoff <= 0:
            continue
        if total_cost + fees > cash:
            continue

        # Realized P&L from the settlement outcomes
        receipt      = _settlement_receipt(n, c["outcome_a"], c["outcome_b"])
        profit       = receipt - total_cost - fees
        invested     = total_cost + fees
        profit_ratio = profit / invested if invested > 0 else 0.0
        # Normalize realized return to a 30-day equivalent (reporting only)
        monthly_profit_ratio = profit_ratio * 30.0 / c["holding_days"]
        # Slippage = realized profit vs. the guaranteed floor (net vs. net)
        slippage = profit - expected_payoff

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
            n=n,
            total_cost=total_cost,
            fees=fees,
            outcome_a=c["outcome_a"],
            outcome_b=c["outcome_b"],
            actual_payoff=receipt,
            profit=profit,
            profit_ratio=profit_ratio,
            monthly_profit_ratio=monthly_profit_ratio,
            kelly_fraction=c["kelly_f_capped"],
            expected_payoff=expected_payoff,
            slippage=slippage,
            holding_days=c["holding_days"],
            balance_at_entry=cash,
        ))

        # Cash out the door: contracts plus fees; the receipt comes back at exit
        cash -= invested
        pending_exits.append((c["exit_date"], receipt))

        # Mark both tickers as active so no overlapping pair is added later
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

    For each trade, subtracts the full cash outlay (total_cost + fees, both paid
    at execution) from cash on the entry_date and adds the gross settlement
    receipt (actual_payoff) on the exit_date. This models a simple accounting
    treatment where capital is deployed on entry and returned at settlement,
    with each dollar counted exactly once.

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
        # Capital leaves the portfolio on entry day (contract cost + taker fees)
        cash_changes[t.entry_date] -= t.total_cost + t.fees
        # Gross settlement receipt returns to the portfolio on exit day
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
