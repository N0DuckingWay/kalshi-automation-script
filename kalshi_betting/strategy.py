"""Arbitrage trade computation and portfolio selection."""
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
    Compute Kelly-sized trade for a guaranteed-arbitrage pair.

    b = net profit per dollar risked = profit_ratio = (1 - nA - pB) / (nA + pB)
    Kelly fraction f* = p - (1-p)/b, capped at BUDGET_FRACTION (20%).
    Returns None when Kelly fraction ≤ 0 (negative expected value — don't bet).
    """
    if not pair.tradeable:
        return None

    nA, pB = pair.nA, pair.pB

    if pB <= 0.0 or pB >= 1.0 or nA <= 0.0 or nA >= 1.0:
        return None

    net_spread = (1.0 - nA - pB) - fee_per_pair_approx(nA, pB)
    if net_spread <= 0:
        return None
    profit_ratio = net_spread / (nA + pB)

    p = _kelly_p(pair)
    q = 1.0 - p
    b = profit_ratio  # net profit per dollar risked

    kelly_fraction = p - q / b
    if kelly_fraction <= 0:
        return None  # negative EV — Kelly says don't bet

    kelly_fraction_capped = min(BUDGET_FRACTION, kelly_fraction)

    budget_dollars = (balance_cents / 100.0) * kelly_fraction_capped
    n = max(1, int(budget_dollars / (nA + pB)))
    if pair.max_contracts > 0:
        n = min(n, pair.max_contracts)

    fee_no  = fee_leg_exact(n, nA)
    fee_yes = fee_leg_exact(n, pB)
    min_payoff = n * (1.0 - nA - pB) - fee_no - fee_yes
    if min_payoff <= 0:
        return None

    total_cost = n * (nA + pB)

    now = datetime.now(timezone.utc)
    close_a = pair.market_a.close_time
    close_b = pair.market_b.close_time
    if close_a.tzinfo is None:
        close_a = close_a.replace(tzinfo=timezone.utc)
    if close_b.tzinfo is None:
        close_b = close_b.replace(tzinfo=timezone.utc)
    days_to_close = max(1, (max(close_a, close_b) - now).days)
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
    Greedy portfolio selection sorted by monthly_profit_ratio descending.
    At equal monthly profit, same_title ranks above time_series (simpler guarantee).
    Trade sizes are already Kelly-weighted via compute_trade().
    """
    available = balance_cents / 100.0
    specs_sorted = sorted(
        specs,
        key=lambda s: (s.monthly_profit_ratio, s.pair.pair_type == "same_title"),
        reverse=True,
    )
    selected = []
    for spec in specs_sorted:
        if spec.total_cost <= available:
            selected.append(spec)
            available -= spec.total_cost
    logging.info(
        "Portfolio: %d trades selected, total cost $%.2f",
        len(selected),
        sum(s.total_cost for s in selected),
    )
    return selected
