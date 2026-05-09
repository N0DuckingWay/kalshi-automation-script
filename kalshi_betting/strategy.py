"""
File: strategy.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Converts scanner.py CandidatePair objects into fully sized trade specifications
    using the Kelly criterion, then selects a portfolio subset that fits within
    the available account balance. The Kelly fraction determines how much of the
    balance to allocate to each pair based on the implied edge and the probability
    that the trade is profitable. Both time-series and same-title pairs use
    different probability models to reflect how correlated their outcomes are.

Dependencies:
    Imports BUDGET_FRACTION, SAME_TITLE_CO_RESOLVE_PROB, and fee helpers from
    config.py. Imports CandidatePair from scanner.py. Exports TradeSpec (consumed
    by trader.py and reporter.py) and select_portfolio() (called by main.py and
    backtester.py).
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .config import BUDGET_FRACTION, SAME_TITLE_CO_RESOLVE_PROB, fee_leg_exact, fee_per_pair_approx
from .scanner import CandidatePair


@dataclass
class TradeSpec:
    """
    Fully computed trade specification for an arbitrage pair, ready for execution.

    Encodes both the trade parameters (contract counts, costs, payoff) and the
    Kelly-sizing metadata used to rank and select trades for the portfolio.

    Attributes:
        pair (CandidatePair): The underlying arbitrage candidate this trade is based on.
        x (int): Number of NO contracts to buy on market A. Always equals y.
        y (int): Number of YES contracts to buy on market B. Always equals x.
        total_cost (float): Total dollar cost of the position: x * (nA + pB).
        min_payoff (float): Guaranteed minimum dollar profit if the arbitrage holds:
            x * (1 - nA - pB). Always > 0 for trades that reach execution.
        profit_ratio (float): Return on cost: (1 - nA - pB) / (nA + pB).
        days_to_close (int): Calendar days until the later-closing market resolves. >= 1.
        monthly_profit_ratio (float): Profit ratio normalized to a 30-day period:
            profit_ratio * 30 / days_to_close. Used for portfolio ranking.
        kelly_p (float): Probability of profit used in the Kelly formula. For time_series
            pairs this is the independence-model estimate; for same_title it is the fixed
            SAME_TITLE_CO_RESOLVE_PROB prior. Range: (0, 1).
        kelly_fraction (float): Kelly fraction capped at BUDGET_FRACTION (20%). This is the
            fraction of account balance allocated to this trade.
    """
    pair: CandidatePair
    x: int
    y: int
    total_cost: float
    min_payoff: float
    profit_ratio: float
    days_to_close: int
    monthly_profit_ratio: float
    kelly_p: float            # probability of profit used in Kelly formula
    kelly_fraction: float     # capped Kelly fraction used for sizing


def _kelly_p(pair: CandidatePair) -> float:
    """
    Probability that the pair results in a profit (i.e. NOT the A=YES, B=NO loss scenario).

    time_series: p = 1 - pA*(1-pB) under independence. Markets ask about the same
    underlying variable at different future times — they're correlated but A=YES does NOT
    structurally guarantee B=YES (e.g. price snapshot markets, rate decisions per meeting).
    Market prices are the best available signal for P(A=YES, B=NO).

    same_title: p = SAME_TITLE_CO_RESOLVE_PROB — fixed prior for markets confirmed to ask
    the exact same question (matching title + subtitle). Divergence is an anomaly, not
    an expected outcome, so a fixed base rate is more appropriate than market prices.
    """
    if pair.pair_type == "time_series":
        return 1.0 - pair.pA * (1.0 - pair.pB)
    return SAME_TITLE_CO_RESOLVE_PROB


def compute_trade(pair: CandidatePair, balance_cents: int) -> Optional[TradeSpec]:
    """
    Compute a Kelly-sized trade specification for a guaranteed-arbitrage pair.

    Applies the Kelly criterion to determine the optimal fraction of the account
    balance to allocate, then derives the integer contract count and verifies that
    the exact net profit (after ceiling-rounded per-leg fees) remains positive.

    Kelly formula used:
        b = net_spread / (nA + pB)   [net profit per dollar risked]
        f* = p - (1-p)/b             [optimal Kelly fraction]
        f_capped = min(BUDGET_FRACTION, f*)

    Where net_spread = (1 − nA − pB) − fee_per_pair_approx(nA, pB).

    Args:
        pair (CandidatePair): The candidate arbitrage pair. Must have tradeable=True.
            Uses pair.nA and pair.pB (depth-weighted fill prices from scanner.py)
            and pair.max_contracts (qualifying order book depth, 0 = uncapped).
        balance_cents (int): Current account balance in cents. Used to convert the
            Kelly fraction to a dollar budget for contract sizing.

    Returns:
        Optional[TradeSpec]: A fully specified trade including contract count n,
            total cost, minimum guaranteed payoff, time-normalized monthly return,
            and Kelly metadata. Returns None if:
            - The pair is not tradeable.
            - The prices are out of the valid (0, 1) range.
            - The net spread is zero or negative after fees.
            - The Kelly fraction is zero or negative (no edge).
            - The exact minimum payoff at the computed n is zero or negative.

    Raises:
        None: All error conditions are handled by returning None.
    """
    if not pair.tradeable:
        return None

    nA, pB = pair.nA, pair.pB

    # Validate that both prices are in the open interval (0, 1).
    # Edge cases at 0 or 1 indicate a settled market and would break the fee formula.
    if pB <= 0.0 or pB >= 1.0 or nA <= 0.0 or nA >= 1.0:
        return None

    # Subtract the continuous fee approximation from the gross spread to get the
    # net edge. A zero or negative net_spread means the trade costs more than it pays.
    net_spread = (1.0 - nA - pB) - fee_per_pair_approx(nA, pB)
    if net_spread <= 0:
        return None

    # profit_ratio is the net return per dollar invested — this is "b" in the Kelly formula
    profit_ratio = net_spread / (nA + pB)

    # Compute the probability that the trade is profitable using the appropriate model
    p = _kelly_p(pair)
    q = 1.0 - p
    # b is the net payoff per dollar risked (same as profit_ratio)
    b = profit_ratio

    # Kelly formula: f* = p - q/b. A negative result means negative expected value.
    kelly_fraction = p - q / b
    if kelly_fraction <= 0:
        # Kelly says don't bet — expected value is negative despite the positive spread
        return None

    # Cap at BUDGET_FRACTION (20%) to avoid over-concentrating in a single pair
    kelly_fraction_capped = min(BUDGET_FRACTION, kelly_fraction)

    # Convert the Kelly fraction to a dollar budget, then derive the integer contract count
    budget_dollars = (balance_cents / 100.0) * kelly_fraction_capped
    n = max(1, int(budget_dollars / (nA + pB)))

    # Respect the order book depth limit set by scanner.enrich_with_orderbook_prices()
    if pair.max_contracts > 0:
        n = min(n, pair.max_contracts)

    # Compute exact ceiling-rounded fees for the final integer n
    fee_no  = fee_leg_exact(n, nA)
    fee_yes = fee_leg_exact(n, pB)

    # Verify the guaranteed minimum payoff is positive after exact fees.
    # At very small n the ceiling rounding can eat the entire profit margin.
    min_payoff = n * (1.0 - nA - pB) - fee_no - fee_yes
    if min_payoff <= 0:
        return None

    total_cost = n * (nA + pB)

    # Compute the number of calendar days until the later-closing market resolves.
    # This is used to normalize the profit ratio to a monthly (30-day) figure for ranking.
    now = datetime.now(timezone.utc)
    close_a = pair.market_a.close_time
    close_b = pair.market_b.close_time
    # Add UTC timezone info if the API returned naive datetimes to avoid comparison errors
    if close_a.tzinfo is None:
        close_a = close_a.replace(tzinfo=timezone.utc)
    if close_b.tzinfo is None:
        close_b = close_b.replace(tzinfo=timezone.utc)
    # Use the later close time — capital is tied up until both legs resolve
    days_to_close = max(1, (max(close_a, close_b) - now).days)
    # Scale the profit ratio to a 30-day equivalent to fairly compare short and long positions
    monthly_profit_ratio = profit_ratio * 30.0 / days_to_close

    logging.info(
        "Trade computed: %s [%s] | p=%.2f kelly=%.1f%% n=%d cost=$%.2f "
        "profit_ratio=%.2f%% monthly=%.2f%%",
        pair.canonical_title,
        pair.pair_type,
        p,
        kelly_fraction_capped * 100,
        n,
        total_cost,
        profit_ratio * 100,
        monthly_profit_ratio * 100,
    )
    return TradeSpec(
        pair=pair,
        x=n,
        y=n,
        total_cost=total_cost,
        min_payoff=min_payoff,
        profit_ratio=profit_ratio,
        days_to_close=days_to_close,
        monthly_profit_ratio=monthly_profit_ratio,
        kelly_p=p,
        kelly_fraction=kelly_fraction_capped,
    )


def select_portfolio(specs: list, balance_cents: int) -> list:
    """
    Select a portfolio of trades using a greedy algorithm prioritized by monthly return.

    Sorts all candidate TradeSpec objects by monthly_profit_ratio descending, then
    greedily adds each trade as long as the remaining available balance covers its
    cost. Trade sizes are already Kelly-fraction-weighted by compute_trade(), so
    each entry fits within the budget constraint by construction — this function
    only resolves conflicts between multiple trades competing for the same capital.

    At equal monthly profit ratios, same_title pairs rank above time_series because
    the same-title guarantee is simpler (identical questions must co-resolve) and
    does not depend on an independence-model probability estimate.

    Args:
        specs (list): List of TradeSpec objects produced by compute_trade(), one
            per qualifying CandidatePair.
        balance_cents (int): Total available account balance in cents. The greedy
            selection stops adding trades when the remaining balance is insufficient.

    Returns:
        list: Ordered list of TradeSpec objects selected for execution, sorted by
            monthly_profit_ratio descending. May be empty if no spec fits.
    """
    # Convert balance to dollars for cost comparisons
    available = balance_cents / 100.0

    # Primary sort: monthly_profit_ratio descending (best capital efficiency first).
    # Secondary sort: same_title > time_series at equal monthly return.
    specs_sorted = sorted(
        specs,
        key=lambda s: (s.monthly_profit_ratio, s.pair.pair_type == "same_title"),
        reverse=True,
    )
    selected = []
    for spec in specs_sorted:
        # Skip this trade if it would exceed the remaining available balance
        if spec.total_cost <= available:
            selected.append(spec)
            available -= spec.total_cost
    logging.info(
        "Portfolio: %d trades selected, total cost $%.2f",
        len(selected),
        sum(s.total_cost for s in selected),
    )
    return selected
