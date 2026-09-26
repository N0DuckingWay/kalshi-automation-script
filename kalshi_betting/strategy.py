"""
File: strategy.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Turns scanner CandidatePairs into Kelly-sized TradeSpecs and selects the
    portfolio that fits the balance. Same-title pairs are priced on the fixed
    SAME_TITLE_CO_RESOLVE_PROB co-resolution prior; time-series pairs are a
    directional bet priced on config.time_series_profit_prob.

Dependencies:
    config (constants, fee helpers, the probability model) and scanner
    (CandidatePair, leg_prices/leg_sides, book-pricing helpers). TradeSpec is
    consumed by trader and reporter; main calls compute_trade and
    select_portfolio. backtester and dashboard do NOT import this module: they
    share config's probability model, fee helpers and constants, but
    re-implement the Kelly formula (net spread, b with the fee in its
    denominator, f* = p - q/b), and backtester's Pass 2 also re-implements
    select_portfolio. A change to either must be made in every copy.

Notes:
    All prices here are LEG prices from scanner.leg_prices(pair): (nA, pB) for
    same_title, (pA, nB) for time_series. cost_with_fees_a/_b are per MARKET,
    not per side — trader funds each market's exchange shard from them.
"""
import logging
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from datetime import UTC, datetime
from typing import NamedTuple

from .config import (
    BUDGET_FRACTION,
    ORDER_API_VERSION,
    PRICE_EPSILON,
    SAME_TITLE_CO_RESOLVE_PROB,
    SIZE_SOLVE_MAX_ITERATIONS,
    fee_leg_exact,
    fee_per_pair_approx,
    max_affordable_pairs,
    time_series_profit_prob,
)
from .scanner import (
    CandidatePair,
    leg_prices,
    leg_sides,
    prefix_fill_prices,
    v2_effective_cap,
)


@dataclass
class TradeSpec:
    """
    A sized trade for one candidate pair, ready for execution.

    Attributes:
        pair (CandidatePair): The candidate, re-priced at the solved size when
            it carried a book.
        x (int): Contracts on market_a. Always equals y.
        y (int): Contracts on market_b.
        total_cost (float): x * (price_a + price_b), excluding fees. Reporting.
        total_cost_with_fees (float): total_cost plus both legs' exact fees —
            the cash the trade consumes; select_portfolio budgets on it.
        min_payoff (float): Profit in a win cell after exact fees; > 0 on every
            returned spec. For time_series the in-between cell instead loses
            total_cost_with_fees.
        profit_ratio (float): Net win return on contract cost,
            net_spread / (price_a + price_b). Ranking and reporting only — NOT
            Kelly's b, whose denominator also includes the fee (DR-62).
        days_to_close (int): Days until the later market closes, >= 1.
        monthly_profit_ratio (float): profit_ratio scaled to 30 days; the
            portfolio ranking key.
        kelly_p (float): Probability of profit, in (0, 1].
        kelly_fraction (float): Kelly fraction, capped at BUDGET_FRACTION.
        cost_with_fees_a (float): market_a's leg cost including its exact fee.
        cost_with_fees_b (float): market_b's leg cost including its exact fee.
            The two sum to total_cost_with_fees; both default to 0.0.
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
    kelly_p: float
    kelly_fraction: float
    cost_with_fees_a: float = 0.0
    cost_with_fees_b: float = 0.0


def _depth_levels(pair: CandidatePair) -> tuple:
    """
    Return the pair's stored book levels, or () if it has none.

    Checked by type, not truthiness: a MagicMock pair's auto-attribute is
    truthy, and a pair without a readable book must be priced on its scalar
    leg prices instead.

    Returns:
        tuple: (price_a, price_b, qty) levels in market order, or ().
    """
    levels = getattr(pair, "depth_levels", None)
    return tuple(levels) if isinstance(levels, (tuple, list)) else ()


def _kelly_p(pair: CandidatePair) -> float:
    """
    Probability of profit for the pair, at its stored YES-leg price.

    same_title: the fixed SAME_TITLE_CO_RESOLVE_PROB prior. It is only valid
    for pairs the scanner confirmed ask one question — same wording, different
    series, closing within SAME_TITLE_MAX_CLOSE_GAP_SECONDS (DR-02, DR-74).

    time_series: config.time_series_profit_prob(pA, pB) = 1 - k*(pB - pA).
    Given the cumulative-deadline premise (screened by wording, DR-67), the
    trade loses only if the event lands between the two deadlines; pB - pA is
    the market's price for that, and the model believes the fraction k of it
    (at k = 1 nothing trades on a consistent book). The YES-ask gap is used
    rather than the executable spread because it is the smaller, more
    conservative estimate of that mass.

    The helper clamps pB - pA at zero, which would model a pair as riskless.
    Enrichment drops a pair whose reference is not above the YES fill at its
    affordability-capped size (the full tier when it falls back to the
    scan-time quote), and compute_trade rejects non-tradeable pairs first. With
    a fresh reference on an uncrossed book every qualifying level sits below
    pB, so the clamp can only fire on a crossed book or a stale reference.

    Returns:
        float: p in (0, 1], the "p" in compute_trade's Kelly formula.
    """
    # The pair's own stored YES-leg quote. compute_trade calls _kelly_p_at
    # directly instead, with the price of the quantity it is actually sizing.
    return _kelly_p_at(pair, pair.pA)


def _kelly_p_at(pair: CandidatePair, yes_leg_price: float) -> float:
    """
    _kelly_p with the YES leg's price supplied, for sizing at a given count.

    pA is a leg price, so it moves with the number of contracts; pB is a
    reference quote and does not. The price is ignored for same_title.

    Args:
        pair (CandidatePair): Supplies pair_type and pB.
        yes_leg_price (float): pA at the size being evaluated, in (0, 1).

    Returns:
        float: p in (0, 1].
    """
    if pair.pair_type == "time_series":
        # Single shared definition of the time-series model — backtester and
        # dashboard call the same helper so the three sizers cannot drift
        return time_series_profit_prob(yes_leg_price, pair.pB)
    return SAME_TITLE_CO_RESOLVE_PROB


class _Sizing(NamedTuple):
    """
    One candidate contract count, priced and Kelly-evaluated at its OWN fill price.

    Attributes:
        n (int): The contract count this was evaluated AT. 0 on the no-book
            path, where there is no size-dependent price to evaluate at.
        target (int): The count the capped Kelly budget affords at that price,
            already clamped to the pair's depth. n is "supported" when
            target >= n.
        price_a (float): market_a's leg price at n, dollars in (0, 1).
        price_b (float): market_b's leg price at n, dollars in (0, 1).
        p (float): Probability of profit at that price.
        profit_ratio (float): Reported return on contract cost at that price;
            not Kelly's b (see _evaluate_size).
        kelly_fraction (float): Kelly fraction, capped at BUDGET_FRACTION.
        budget_dollars (float): Contract-only budget the fee shrink measures against.
    """
    n: int
    target: int
    price_a: float
    price_b: float
    p: float
    profit_ratio: float
    kelly_fraction: float
    budget_dollars: float


def _reachable_contracts(
    pair: CandidatePair, levels: tuple, price_a: float, price_b: float,
) -> float:
    """
    Contract pairs a V2 fill-or-kill at these leg prices can actually buy.

    Each leg is one FoK limit and only reaches depth at or below its cap. The
    cap comes from the leg's AVERAGE price, which on a multi-level prefix sits
    below the prefix's top level, so the top can be out of reach (TS-08). Caps
    come from scanner.v2_effective_cap, the arithmetic the order body uses —
    never re-derive them here.

    Args:
        pair (CandidatePair): Supplies pair_type and both markets' tick grids.
        levels (tuple): depth_levels — market order, ascending combined price.
        price_a (float): market_a's leg price for the size being tested.
        price_b (float): market_b's leg price, likewise.

    Returns:
        float: Contract pairs resting at or below both caps.
    """
    # leg_sides maps market to side; levels are already in market order
    side_a, side_b = leg_sides(pair.pair_type)
    cap_a = float(v2_effective_cap(f"buy_{side_a}", price_a, pair.market_a))
    cap_b = float(v2_effective_cap(f"buy_{side_b}", price_b, pair.market_b))
    # PRICE_EPSILON: a level exactly on the cap must count (TS-09)
    return sum(
        qty for pa, pb, qty in levels
        if pa <= cap_a + PRICE_EPSILON and pb <= cap_b + PRICE_EPSILON
    )


def _evaluate_size(
    pair: CandidatePair, levels: tuple, n: int, balance_cents: int,
) -> _Sizing | None:
    """
    Price n contract pairs off the pair's book and return what Kelly then affords.

    Every gate compute_trade applies lives here, so each candidate size is judged
    entirely at ITS OWN fill price rather than at one scalar average computed
    over depth the trade may never reach. With no levels the pair's stored leg
    prices are used and n is ignored — the single-shot path.

    Args:
        pair (CandidatePair): The pair being sized. Supplies pair_type and pB
            for the probability model and max_contracts for the depth clamp.
        levels (tuple): The pair's qualifying depth from _depth_levels(); ()
            means price on the stored scalars instead.
        n (int): Contract count to price at. Ignored when levels is empty.
        balance_cents (int): Account balance in integer cents.

    Returns:
        _Sizing | None: The priced, Kelly-evaluated candidate, or None when any
            gate fails — leg price outside (0, 1), no net spread after the fee
            approximation, a nonpositive Kelly fraction, a budget that cannot
            afford one contract pair, or a book too thin to fill n.
    """
    if levels:
        fills = prefix_fill_prices(levels, n)
        if fills is None:
            # Depth ran out below n — the caller searches smaller sizes
            return None
        price_a, price_b = fills
    else:
        # The two prices the legs actually cost — which of the pair's four quotes
        # they are depends on the pair type; leg_prices is the single source of truth
        price_a, price_b = leg_prices(pair)

    # 0 or 1 means a settled market and breaks the fee formula
    if price_b <= 0.0 or price_b >= 1.0 or price_a <= 0.0 or price_a >= 1.0:
        return None

    if levels and ORDER_API_VERSION == "v2":
        # A V2 FoK only reaches depth at or below its own limit (TS-08); an
        # unreachable n returns None so the search tries smaller sizes. V2 only:
        # the legacy buy_max_cost is a total-cost cap that can sweep a ladder.
        if _reachable_contracts(pair, levels, price_a, price_b) < n:
            return None

    # Net edge after the continuous fee approximation
    fee_approx = fee_per_pair_approx(price_a, price_b)
    net_spread = (1.0 - price_a - price_b) - fee_approx
    if net_spread <= 0:
        return None

    # Reported return on contract cost (ranking + prod log) — not Kelly's b
    profit_ratio = net_spread / (price_a + price_b)

    # p at THIS size's price, not the pair's stored pA
    p = _kelly_p_at(pair, price_a)
    q = 1.0 - p
    # Kelly's b is the payoff per dollar AT RISK, and a losing pair loses its
    # fees too, so the fee is in the denominator (DR-62). Deliberately a
    # different quantity from profit_ratio; do not merge them.
    kelly_b = net_spread / (price_a + price_b + fee_approx)

    # f* > 0 means positive EV under the CONTINUOUS fee approximation. Exact
    # fees can still leave a boundary spec slightly EV-negative at small n;
    # compute_trade's min_payoff > 0 check bounds that, it does not remove it.
    kelly_fraction = p - q / kelly_b
    if kelly_fraction <= 0:
        # No positive EV once fees are counted
        return None

    # Cap at BUDGET_FRACTION (20%) to avoid over-concentrating in a single pair
    kelly_fraction_capped = min(BUDGET_FRACTION, kelly_fraction)

    budget_dollars = (balance_cents / 100.0) * kelly_fraction_capped
    # Same budget -> contracts helper the scanner's depth cap uses, so that cap bounds this count
    target = max_affordable_pairs(balance_cents, price_a + price_b, kelly_fraction_capped)

    # Respect the order book depth limit set by scanner.enrich_with_orderbook_prices()
    if pair.max_contracts > 0:
        target = min(target, pair.max_contracts)
    if target < 1:
        # Can't afford one contract pair; forcing n=1 would exceed the Kelly budget
        return None

    return _Sizing(
        n=n, target=target, price_a=price_a, price_b=price_b, p=p,
        profit_ratio=profit_ratio, kelly_fraction=kelly_fraction_capped,
        budget_dollars=budget_dollars,
    )


def _solve_marginal_size(
    pair: CandidatePair, levels: tuple, balance_cents: int,
) -> _Sizing | None:
    """
    Largest contract count whose own fill price still justifies it.

    n is SUPPORTED when every gate passes at the price of n and Kelly at that
    price affords at least n. This bisects [1, pair.max_contracts]; the upper
    bound is enrichment's affordability cap, so nothing above it can be
    supported.

    The supported set is not downward-closed (a larger n can raise p for a
    time-series pair, and reachability can leave holes), so the search may
    under-size. It never mis-prices: a count is only returned after
    _evaluate_size verified it. It searches rather than checking only the top
    because the gates bite hardest at full depth.

    Returns:
        _Sizing | None: The largest verified count, or None if none is supported.
    """
    lo, hi = 1, pair.max_contracts
    best: _Sizing | None = None
    for _ in range(SIZE_SOLVE_MAX_ITERATIONS):
        if lo > hi:
            return best
        mid = (lo + hi) // 2
        sized = _evaluate_size(pair, levels, mid, balance_cents)
        if sized is None or sized.target < mid:
            # Too big: a gate fails at this price, or Kelly won't fund this many
            hi = mid - 1
        else:
            best = sized
            lo = mid + 1
    # Unreachable (bisection closes in ~log2 passes); best is verified, so return it
    logging.warning(
        "Marginal size for '%s' did not converge in %d passes — using the largest "
        "count verified so far",
        pair.canonical_title, SIZE_SOLVE_MAX_ITERATIONS,
    )
    return best


def compute_trade(pair: CandidatePair, balance_cents: int) -> TradeSpec | None:
    """
    Kelly-size a candidate pair into a TradeSpec.

    With a book (depth_levels), size and price are solved together by
    _solve_marginal_size; without one, the pair's scalar leg prices are used.
    n is then shrunk until the fee-inclusive cost fits the Kelly budget.

    Kelly, on leg prices:
        fee        = fee_per_pair_approx(price_a, price_b)
        net_spread = (1 - price_a - price_b) - fee
        b          = net_spread / (price_a + price_b + fee)   # dollars at risk
        f*         = p - (1 - p) / b, capped at BUDGET_FRACTION

    The fee is in b's denominator because a losing pair loses its fees too
    (DR-62). f* > 0 means positive EV under the continuous fee approximation
    only: exact ceiling-rounded fees can still leave a boundary spec slightly
    EV-negative at small n, which the min_payoff > 0 check bounds but does not
    remove.

    Args:
        pair (CandidatePair): Must be tradeable. max_contracts 0 means not
            enriched (no depth cap).
        balance_cents (int): Account balance in cents.

    Returns:
        TradeSpec | None: None if the pair is not tradeable, a leg price is
            outside (0, 1), no net spread remains after fees, Kelly is <= 0,
            the budget cannot afford one contract pair, or the exact win
            payoff is <= 0.

    Raises:
        AttributeError/TypeError: If a market's close_time is None. Scanner
            pairs never have one (scanner._filter_active_markets drops them);
            only a hand-built pair can.
    """
    if not pair.tradeable:
        return None

    # Book levels from enrichment; () means size on the scalar leg prices
    levels = _depth_levels(pair)

    if levels:
        sized = _solve_marginal_size(pair, levels, balance_cents)
        if sized is None:
            return None
        # The size whose OWN fill price justifies it, and that price
        n = sized.n
    else:
        sized = _evaluate_size(pair, levels, 0, balance_cents)
        if sized is None:
            return None
        # No book (e.g. the bare pair the backtester's Kelly-parity test builds):
        # the single-shot sizing, which must stay unchanged
        n = sized.target

    price_a, price_b = sized.price_a, sized.price_b
    p                = sized.p
    profit_ratio     = sized.profit_ratio
    kelly_fraction_capped = sized.kelly_fraction
    budget_dollars   = sized.budget_dollars

    # budget_dollars covers contracts only; shrink n until the fees fit too
    fee_a = fee_leg_exact(n, price_a)
    fee_b = fee_leg_exact(n, price_b)
    while n > 0 and n * (price_a + price_b) + fee_a + fee_b > budget_dollars:
        n -= 1
        fee_a = fee_leg_exact(n, price_a)
        fee_b = fee_leg_exact(n, price_b)
    if n < 1:
        # Fees ate the entire Kelly budget — no contract count fits
        return None

    if levels and n != sized.n:
        # n moved: re-price at the count actually submitted (trader reads this
        # price for the FoK limit and rollback floor). p, profit_ratio and the
        # Kelly fraction stay at pre-shrink values — reporting/ranking only.
        fills = prefix_fill_prices(levels, n)
        if fills is not None:
            price_a, price_b = fills
            fee_a = fee_leg_exact(n, price_a)
            fee_b = fee_leg_exact(n, price_b)

        # BACKSTOP: the shrink changed n outside _evaluate_size, and
        # reachability is not downward-closed, so re-check it (TS-08, V2 only
        # for the same reason as in _evaluate_size). It has never been observed
        # to fire, but nothing guarantees that: do not delete it as dead code.
        # Terminates: each pass breaks or sets n = int(reachable) < n; if
        # nothing is reachable (n < 1) the pair is dropped.
        if ORDER_API_VERSION == "v2":
            while n >= 1:
                reachable = _reachable_contracts(pair, levels, price_a, price_b)
                if reachable >= n:
                    break
                n = int(reachable)
                if n < 1:
                    break
                fills = prefix_fill_prices(levels, n)
                if fills is None:
                    break
                price_a, price_b = fills
                fee_a = fee_leg_exact(n, price_a)
                fee_b = fee_leg_exact(n, price_b)
            if n < 1:
                # Nothing the cap can reach
                return None

    # Exact-fee win payoff; ceiling rounding can erase it at small n
    min_payoff = n * (1.0 - price_a - price_b) - fee_a - fee_b
    if min_payoff <= 0:
        return None

    total_cost = n * (price_a + price_b)
    # Fees are cash out the door at execution — the portfolio budget must cover them
    total_cost_with_fees = total_cost + fee_a + fee_b

    # Per-MARKET costs, for trader.ensure_shard_collateral's per-shard funding
    cost_with_fees_a = n * price_a + fee_a
    cost_with_fees_b = n * price_b + fee_b

    # Days until the LATER close (capital is tied up until both resolve); naive datetimes are treated as UTC
    now = datetime.now(UTC)
    close_a = pair.market_a.close_time
    close_b = pair.market_b.close_time
    if close_a.tzinfo is None:
        close_a = close_a.replace(tzinfo=UTC)
    if close_b.tzinfo is None:
        close_b = close_b.replace(tzinfo=UTC)
    days_to_close = max(1, (max(close_a, close_b) - now).days)
    # 30-day equivalent so short and long trades rank fairly
    monthly_profit_ratio = profit_ratio * 30.0 / days_to_close

    # Name the sides so a time-series line isn't read as the same-title layout
    side_a, side_b = leg_sides(pair.pair_type)
    logging.info(
        "Trade computed: %s [%s] | %s(A)@%.2f + %s(B)@%.2f | p=%.2f kelly=%.1f%% n=%d "
        "cost=$%.2f profit_ratio=%.2f%% monthly=%.2f%%",
        pair.canonical_title,
        pair.pair_type,
        side_a.upper(),
        price_a,
        side_b.upper(),
        price_b,
        p,
        kelly_fraction_capped * 100,
        n,
        total_cost,
        profit_ratio * 100,
        monthly_profit_ratio * 100,
    )
    if levels:
        # Write the solved prices back through leg_sides, so every reader of
        # leg_prices(spec.pair) — trader, prod log — gets the marginal price.
        leg_updates = (
            {"pA": price_a, "nB": price_b} if side_a == "yes"
            else {"nA": price_a, "pB": price_b}
        )
        pair = dc_replace(pair, max_contracts=n, **leg_updates)

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
    Greedy portfolio: best monthly return first, within balance, no reused tickers.

    Walks specs by monthly_profit_ratio descending (same_title before
    time_series on ties — it is the near-arbitrage). A spec is taken if neither
    ticker is already used this run and its total_cost_with_fees fits the
    remaining balance; a spec that doesn't fit is skipped, not a stop, so a
    cheaper one further down can still be taken. Open positions from earlier
    runs are excluded upstream by scanner.get_held_tickers (prod only).
    backtester's Pass 2 mirrors this selection; change both together.

    Args:
        specs (list): TradeSpecs from compute_trade.
        balance_cents (int): Available balance in cents.

    Returns:
        list: The selected specs in ranking order; may be empty.
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
        # Fee-inclusive: the figure this loop budgets against (TS-12)
        "Portfolio: %d trades selected, total cost $%.2f incl. fees",
        len(selected),
        sum(s.total_cost_with_fees for s in selected),
    )
    return selected
