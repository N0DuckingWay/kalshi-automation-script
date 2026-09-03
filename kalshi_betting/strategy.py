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
from datetime import UTC, datetime

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
        total_cost (float): Total dollar cost of the contracts: x * (nA + pB).
            Excludes taker fees — used for reporting.
        total_cost_with_fees (float): total_cost plus the exact ceiling-rounded
            taker fee for both legs. This is the real cash the trade consumes at
            execution — select_portfolio() budgets against this value.
        min_payoff (float): Guaranteed minimum dollar profit if the arbitrage holds:
            x * (1 - nA - pB). Always > 0 for trades that reach execution.
        profit_ratio (float): Return on cost, net of the continuous fee
            approximation: ((1 - nA - pB) - fee_per_pair_approx(nA, pB)) /
            (nA + pB). This is "b" in the Kelly formula below (see
            compute_trade()'s net_spread/profit_ratio computation).
        days_to_close (int): Calendar days until the later-closing market resolves. >= 1.
        monthly_profit_ratio (float): Profit ratio normalized to a 30-day period:
            profit_ratio * 30 / days_to_close. Used for portfolio ranking.
        kelly_p (float): Probability of profit used in the Kelly formula. For time_series
            pairs this is the independence-model estimate; for same_title it is the fixed
            SAME_TITLE_CO_RESOLVE_PROB prior. Range: (0, 1).
        kelly_fraction (float): Kelly fraction capped at BUDGET_FRACTION (20%). This is the
            fraction of account balance allocated to this trade.
        cost_with_fees_a (float): Leg A's own cash requirement: x * nA + that leg's
            exact ceiling-rounded taker fee (fee_leg_exact(x, nA)). Used by the collateral
            transfer planner to fund leg A's exchange shard. Invariant:
            cost_with_fees_a + cost_with_fees_b == total_cost_with_fees (same terms, same
            fee calls). Defaults to 0.0 for TradeSpec constructions that don't populate it.
        cost_with_fees_b (float): Leg B's own cash requirement: y * pB + that leg's
            exact ceiling-rounded taker fee (fee_leg_exact(y, pB)). Used by the collateral
            transfer planner to fund leg B's exchange shard. Invariant:
            cost_with_fees_a + cost_with_fees_b == total_cost_with_fees (same terms, same
            fee calls). Defaults to 0.0 for TradeSpec constructions that don't populate it.
    """
    pair: CandidatePair
    x: int
    y: int
    total_cost: float
    total_cost_with_fees: float
    min_payoff: float
    profit_ratio: float
    days_to_close: int
    monthly_profit_ratio: float
    kelly_p: float            # probability of profit used in Kelly formula
    kelly_fraction: float     # capped Kelly fraction used for sizing
    # Per-leg cash requirements, used by the collateral transfer planner to fund each
    # leg's shard; invariant: cost_with_fees_a + cost_with_fees_b == total_cost_with_fees
    # (same terms, same fee calls). Defaulted so existing constructions don't break.
    cost_with_fees_a: float = 0.0
    cost_with_fees_b: float = 0.0


def _kelly_p(pair: CandidatePair) -> float:
    """
    Probability that the pair results in a profit (i.e. NOT the A=YES, B=NO loss scenario).

    time_series: p = 1 - pA*(1-pB) under independence. Markets ask about the same
    underlying variable at different future times — they're correlated but A=YES does NOT
    structurally guarantee B=YES (e.g. price snapshot markets, rate decisions per meeting).
    Market prices are the best available signal for P(A=YES, B=NO).

    same_title: p = SAME_TITLE_CO_RESOLVE_PROB — fixed prior for markets confirmed to ask
    the exact same question (matching event_title + title + subtitle, see scanner.pair_key).
    Divergence is an anomaly, not an expected outcome, so a fixed base rate is more
    appropriate than market prices. This prior is calibrated for binary contracts and
    applies equally to MVE option markets once cross-event collisions are eliminated by
    the event_title component of the grouping key.
    """
    if pair.pair_type == "time_series":
        return 1.0 - pair.pA * (1.0 - pair.pB)
    return SAME_TITLE_CO_RESOLVE_PROB


def compute_trade(pair: CandidatePair, balance_cents: int) -> TradeSpec | None:
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
            Kelly metadata, and each leg's own fee-inclusive cash requirement
            (cost_with_fees_a, cost_with_fees_b — used by the collateral transfer
            planner to fund each leg's exchange shard). Returns None if:
            - The pair is not tradeable.
            - The prices are out of the valid (0, 1) range.
            - The net spread is zero or negative after fees.
            - The Kelly fraction is zero or negative (no edge).
            - The Kelly budget cannot afford a single contract pair.
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
    n = int(budget_dollars / (nA + pB))
    if n < 1:
        # The Kelly budget can't afford even one contract pair — forcing n=1
        # would silently exceed both the Kelly fraction and BUDGET_FRACTION
        return None

    # Respect the order book depth limit set by scanner.enrich_with_orderbook_prices()
    if pair.max_contracts > 0:
        n = min(n, pair.max_contracts)

    # Compute exact ceiling-rounded fees for the final integer n. budget_dollars
    # above only covers the contract cost (n * (nA + pB)) — fees are added on
    # top, so the straight n from that division can push total_cost_with_fees
    # slightly past the capped Kelly budget. Shrink n until the fee-inclusive
    # cost actually fits, so the real cash consumed never exceeds what the
    # Kelly fraction (and BUDGET_FRACTION) allowed.
    fee_no  = fee_leg_exact(n, nA)
    fee_yes = fee_leg_exact(n, pB)
    while n > 0 and n * (nA + pB) + fee_no + fee_yes > budget_dollars:
        n -= 1
        fee_no  = fee_leg_exact(n, nA)
        fee_yes = fee_leg_exact(n, pB)
    if n < 1:
        # Fees ate the entire Kelly budget — no contract count fits
        return None

    # Verify the guaranteed minimum payoff is positive after exact fees.
    # At very small n the ceiling rounding can eat the entire profit margin.
    min_payoff = n * (1.0 - nA - pB) - fee_no - fee_yes
    if min_payoff <= 0:
        return None

    total_cost = n * (nA + pB)
    # Fees are cash out the door at execution — the portfolio budget must cover them
    total_cost_with_fees = total_cost + fee_no + fee_yes

    # Per-leg cash requirements — same terms and fee calls as total_cost_with_fees,
    # just not summed together. The collateral transfer planner (later commit) uses
    # these to fund each leg's own exchange shard rather than the pair total.
    cost_with_fees_a = n * nA + fee_no
    cost_with_fees_b = n * pB + fee_yes

    # Compute the number of calendar days until the later-closing market resolves.
    # This is used to normalize the profit ratio to a monthly (30-day) figure for ranking.
    now = datetime.now(UTC)
    close_a = pair.market_a.close_time
    close_b = pair.market_b.close_time
    # Add UTC timezone info if the API returned naive datetimes to avoid comparison errors
    if close_a.tzinfo is None:
        close_a = close_a.replace(tzinfo=UTC)
    if close_b.tzinfo is None:
        close_b = close_b.replace(tzinfo=UTC)
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
        total_cost_with_fees=total_cost_with_fees,
        min_payoff=min_payoff,
        profit_ratio=profit_ratio,
        days_to_close=days_to_close,
        monthly_profit_ratio=monthly_profit_ratio,
        kelly_p=p,
        kelly_fraction=kelly_fraction_capped,
        cost_with_fees_a=cost_with_fees_a,
        cost_with_fees_b=cost_with_fees_b,
    )


def select_portfolio(specs: list, balance_cents: int) -> list:
    """
    Select a portfolio of trades using a greedy algorithm prioritized by monthly return.

    Sorts all candidate TradeSpec objects by monthly_profit_ratio descending and
    walks the list once. Each spec is selected if (a) both of its tickers are still
    free and (b) its total_cost_with_fees — the real cash the trade consumes,
    including both legs' taker fees — fits in the remaining balance. The loop does
    NOT break when a spec doesn't fit — it keeps scanning so a cheaper trade
    further down can still be added.

    Ticker-conflict filter: once a spec is chosen, both of its market tickers are
    marked used and no later spec that touches either ticker is selected. This
    prevents a single market from being a leg in two overlapping pairs within one
    run. Across runs the same invariant is enforced by scanner.get_held_tickers(),
    but only for as long as a position is OPEN: it queries positions with
    count_filter="position", so a ticker drops out of the held set once its market
    settles and may legitimately be entered again afterwards. The backtester's
    Pass-2 filter mirrors exactly that — it blocks a ticker until its trade's exit
    date and releases it there.

    At equal monthly profit ratios, same_title pairs rank above time_series because
    the same-title guarantee is simpler (identical questions must co-resolve) and
    does not depend on an independence-model probability estimate.

    Args:
        specs (list): List of TradeSpec objects produced by compute_trade(), one
            per qualifying CandidatePair.
        balance_cents (int): Total available account balance in cents. The greedy
            selection skips trades that don't fit but keeps scanning cheaper ones.

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
    used_tickers: set[str] = set()
    for spec in specs_sorted:
        ta = spec.pair.market_a.ticker
        tb = spec.pair.market_b.ticker
        # Skip trades that would re-use a ticker already committed to a higher-priority pair
        if ta in used_tickers or tb in used_tickers:
            continue
        # Skip this trade if its full cash requirement (contracts + taker fees)
        # would exceed the remaining available balance
        if spec.total_cost_with_fees > available:
            continue
        selected.append(spec)
        available -= spec.total_cost_with_fees
        used_tickers.add(ta)
        used_tickers.add(tb)
    logging.info(
        "Portfolio: %d trades selected, total cost $%.2f",
        len(selected),
        sum(s.total_cost for s in selected),
    )
    return selected
