"""
File: strategy.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Turns scanner CandidatePairs into Kelly-sized TradeSpecs and selects the
    portfolio the cash buys. Same-title pairs are priced on the fixed
    SAME_TITLE_CO_RESOLVE_PROB co-resolution prior; time-series pairs are a
    directional bet priced on config.time_series_profit_prob.

    Kelly fractions are taken of the portfolio value (cash plus what the open
    positions are worth), but only cash buys contracts: each trade's budget is
    config.kelly_budget, min(portfolio value x capped fraction, cash).
    select_portfolio then spends the cash itself, trade by trade in whole
    cents, and shrinks a trade that no longer fits the cash left to the
    largest size that does, skipping it only when not one contract pair fits.

Dependencies:
    config (constants, fee helpers, the probability model, LiveSettings,
    live_settings, pair_size_cap, kelly_budget, leg_cash_cents) and scanner
    (CandidatePair, leg_prices/leg_sides, book-pricing helpers,
    pair_ladder_keys). TradeSpec is consumed by trader and reporter; main
    calls compute_trade and select_portfolio. backtester and dashboard do NOT
    import this module: they share config's probability model, fee helpers
    and constants, but re-implement the Kelly formula (net spread, b with the
    fee in its denominator, f* = p - q/b), and backtester also re-implements
    select_portfolio's ticker and ladder rules. A change to either must be
    made in every copy. The backtester budgets through the same
    config.kelly_budget — each simulated Monday's cash plus its open trades
    at cost, never more than the cash left — so it too shrinks a trade to the
    cash left, but it spends a trade's fee-inclusive cost at its entry prices
    (candle quotes, with no cash_need_cents reserve) and does not repeat
    _spec_at_count's expected-value check on a shrunk trade.

Notes:
    All prices here are LEG prices from scanner.leg_prices(pair): (nA, pB) for
    same_title, (pA, nB) for time_series. cost_with_fees_a/_b are per MARKET,
    not per side — trader funds each market's exchange shard from them.
    TradeSpec.cash_need_cents is never below what that funding asks for, so
    the cash a portfolio select_portfolio admitted always covers every
    shard's funding together. The extra it sets aside for the orders' limit
    prices is held on the total only: a shard topped up by a transfer holds
    its legs' cost at the fill prices.

    k and the per-pair caps come from one config.LiveSettings per call, which
    compute_trade hands to both sizing paths: a live run's, built from the
    saved live defaults, or config.py's toggles for a caller that hands none
    (tests and direct calls).
"""
import logging
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from datetime import UTC, datetime
from typing import NamedTuple

from .config import (
    PRICE_EPSILON,
    SAME_TITLE_CO_RESOLVE_PROB,
    SIZE_SOLVE_MAX_ITERATIONS,
    LiveSettings,
    fee_leg_exact,
    fee_per_pair_approx,
    kelly_budget,
    leg_cash_cents,
    live_settings,
    max_affordable_pairs,
    pair_size_cap,
    time_series_profit_prob,
)
from .scanner import (
    CandidatePair,
    leg_prices,
    leg_sides,
    pair_ladder_keys,
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
            the cash the trade consumes at its fill prices. The prod log and
            the portfolio summary report it; select_portfolio spends
            cash_need_cents instead.
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
        kelly_fraction (float): Kelly fraction, capped by config.pair_size_cap.
        cost_with_fees_a (float): market_a's leg cost including its exact fee.
        cost_with_fees_b (float): market_b's leg cost including its exact fee.
            The two sum to total_cost_with_fees; both default to 0.0.
        cash_need_cents (int | None): Whole cents the two fill-or-kill orders
            can draw from the cash at worst (_order_cash_cents: per leg, the
            count at the higher of its limit and its fill, plus the fee at
            the dearest price between the two, rounded up) — what
            select_portfolio takes from the cash left. compute_trade always
            sets it; None (a spec built by hand) is read as the fee-inclusive
            cost rounded up per leg (_cash_need).
        interval_discount (float | None): The k a time-series spec was sized
            at (the run's LiveSettings.interval_discount); _spec_at_count
            prices its probability of profit at a smaller size with it.
            compute_trade sets it on every time-series spec and leaves it None
            on a same-title one, whose p is the fixed prior. A time-series
            spec without it (one built by hand) is never shrunk.
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
    cash_need_cents: int | None = None
    interval_discount: float | None = None


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


def _kelly_p(pair: CandidatePair, settings: LiveSettings) -> float:
    """
    Probability of profit for the pair, at its stored YES-leg price.

    same_title: the fixed SAME_TITLE_CO_RESOLVE_PROB prior. It is only valid
    for pairs the scanner confirmed ask one question — same wording, different
    series, closing within SAME_TITLE_MAX_CLOSE_GAP_SECONDS (DR-02, DR-74).

    time_series: config.time_series_profit_prob(pA, pB, k) = 1 - k*(pB - pA).
    Given the cumulative-deadline premise (screened by wording, DR-67), the
    trade loses only if the event lands between the two deadlines; pB - pA is
    the market's price for that, and the model believes the fraction k of it
    (at k = 1 nothing trades on a consistent book). The YES-ask gap is used
    rather than the executable spread because it is the smaller, more
    conservative estimate of that mass.

    The helper clamps pB - pA at zero, which would model a pair as riskless.
    Enrichment marks non-tradeable, and compute_trade skips, every time-series
    pair whose fresh pB is missing or not above every YES fill the sizer can
    reach, so the clamp cannot fire on an enriched pair.

    Args:
        pair (CandidatePair): Supplies pair_type, pA and pB.
        settings (LiveSettings): The run's toggles; reads interval_discount (k).

    Returns:
        float: p in (0, 1], the "p" in compute_trade's Kelly formula.
    """
    # The pair's own stored YES-leg quote. compute_trade calls _kelly_p_at
    # directly instead, with the price of the quantity it is actually sizing.
    return _kelly_p_at(pair, pair.pA, settings.interval_discount)


def _kelly_p_at(pair: CandidatePair, yes_leg_price: float, k: float) -> float:
    """
    _kelly_p with the YES leg's price supplied, for sizing at a given count.

    pA is a leg price, so it moves with the number of contracts; pB is a
    reference quote and does not. The price and k are ignored for same_title.

    Args:
        pair (CandidatePair): Supplies pair_type and pB.
        yes_leg_price (float): pA at the size being evaluated, in (0, 1).
        k (float): The run's interval discount, in (0, 1]. Never None, which
            would read config's constant instead.

    Returns:
        float: p in (0, 1].
    """
    if pair.pair_type == "time_series":
        # Single shared definition of the time-series model — backtester and
        # dashboard call the same helper so the three sizers cannot drift
        return time_series_profit_prob(yes_leg_price, pair.pB, k=k)
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
        kelly_fraction (float): Kelly fraction, capped by config.pair_size_cap.
        budget_dollars (float): What this trade may spend, in dollars
            (config.kelly_budget: the capped fraction of the portfolio value,
            never more than the cash). compute_trade's fee shrink fits the
            contracts plus their fees under it.
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
    pair: CandidatePair, levels: tuple, n: int, portfolio_value_cents: int,
    settings: LiveSettings, *, cash_cents: int | None = None,
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
        portfolio_value_cents (int): The value Kelly fractions are taken of,
            in integer cents.
        settings (LiveSettings): The run's toggles (k and the per-pair caps).
        cash_cents (int | None): Keyword-only. The cash on hand, in integer
            cents; the budget never exceeds it. None means the portfolio value
            is all cash.

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

    if levels:
        # A fill-or-kill order only buys depth at or under its own limit
        # (TS-08); if n is out of reach, return None so the search tries less
        if _reachable_contracts(pair, levels, price_a, price_b) < n:
            return None

    # Net edge after the continuous fee approximation
    fee_approx = fee_per_pair_approx(price_a, price_b)
    net_spread = (1.0 - price_a - price_b) - fee_approx
    if net_spread <= 0:
        return None

    # Reported return on contract cost (ranking + prod log) — not Kelly's b
    profit_ratio = net_spread / (price_a + price_b)

    # p at THIS size's price, not the pair's stored pA, and at the run's k
    p = _kelly_p_at(pair, price_a, settings.interval_discount)
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

    # The one cap definition, shared with the backtester and enrichment's bound
    kelly_fraction_capped = min(
        pair_size_cap(pair.pair_type, settings.size_cap, settings.same_title_size_cap),
        kelly_fraction)

    cash = None if cash_cents is None else cash_cents / 100.0
    # A share of the portfolio value, never more than the cash on hand — the
    # one budget rule, shared with enrichment's depth bound
    budget_dollars = kelly_budget(portfolio_value_cents / 100.0, kelly_fraction_capped, cash)
    # Same budget -> contracts helper the scanner's depth cap uses, so that cap bounds this count
    target = max_affordable_pairs(portfolio_value_cents, price_a + price_b,
                                  kelly_fraction_capped, cash_cents=cash_cents)

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
    pair: CandidatePair, levels: tuple, portfolio_value_cents: int, settings: LiveSettings,
    *, cash_cents: int | None = None,
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

    Args:
        pair (CandidatePair): The pair being sized; max_contracts bounds the search.
        levels (tuple): The pair's qualifying depth (non-empty).
        portfolio_value_cents (int): The value Kelly fractions are taken of,
            in integer cents.
        settings (LiveSettings): The run's toggles.
        cash_cents (int | None): Keyword-only. The cash on hand, in integer
            cents, handed to every _evaluate_size. None means the portfolio
            value is all cash.

    Returns:
        _Sizing | None: The largest verified count, or None if none is supported.
    """
    lo, hi = 1, pair.max_contracts
    best: _Sizing | None = None
    for _ in range(SIZE_SOLVE_MAX_ITERATIONS):
        if lo > hi:
            return best
        mid = (lo + hi) // 2
        sized = _evaluate_size(pair, levels, mid, portfolio_value_cents, settings,
                               cash_cents=cash_cents)
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


def _order_cash_cents(pair: CandidatePair, n: int, price_a: float, price_b: float) -> int:
    """
    Whole cents a pair's two fill-or-kill orders can draw from the cash, at worst.

    Each leg is one fill-or-kill order for n contracts whose limit price is
    scanner.v2_effective_cap of its leg price — the price the order body
    carries, in the leg's own side terms — so it can pay up to that limit on
    every contract, plus the taker fee. Per leg this takes n times the higher
    of the limit and the leg price, plus the fee at the price between the two
    that is nearest 50c (the fee is proportional to p(1-p), which peaks
    there), rounded up to the cent by config.leg_cash_cents; the two legs are
    summed. Kalshi may hold cash at the limit rather than at the fill, so
    this, not the fee-inclusive cost at the fill, is what select_portfolio
    fits into the cash left. Per leg it is never below leg_cash_cents of that
    leg's cost at its fill (TradeSpec.cost_with_fees_a/_b), which is what
    trader._required_cents_by_shard asks each shard for, so the cash a
    portfolio fits into always covers every shard's funding together. The
    headroom above the fill is held on the total cash only: a shard topped
    up by a transfer is funded to its legs' cost at the fill.

    Args:
        pair (CandidatePair): Supplies pair_type (which side each leg buys) and
            both markets' tick grids.
        n (int): Contract pairs; each leg buys n. Range: >= 1.
        price_a (float): market_a's leg price, dollars in (0, 1).
        price_b (float): market_b's leg price, dollars in (0, 1).

    Returns:
        int: Whole cents both orders can draw together.
    """
    total = 0
    for side, price, market in zip(leg_sides(pair.pair_type), (price_a, price_b),
                                   (pair.market_a, pair.market_b), strict=True):
        # The one definition of a leg's limit, never re-derived here (TS-08)
        cap = float(v2_effective_cap(f"buy_{side}", price, market))
        low, high = min(cap, price), max(cap, price)
        # The fee at any price the leg can fill at between the two, at its highest
        fee = fee_leg_exact(n, min(max(0.5, low), high))
        total += leg_cash_cents(n * high + fee)
    return total


def _priced_pair(pair: CandidatePair, n: int, price_a: float, price_b: float) -> CandidatePair:
    """
    A copy of a booked pair priced at n contract pairs.

    The two leg prices are written back through leg_sides, so every reader of
    leg_prices(spec.pair) — trader's order limits and rollback floor, the
    prod log — gets the price of the n actually submitted, and max_contracts
    becomes n, the count those prices are valid for. The book (depth_levels)
    and the quotes that are not leg prices are kept. compute_trade calls it
    for the size it solves, and _spec_at_count for a smaller size.

    Args:
        pair (CandidatePair): A pair carrying depth_levels.
        n (int): The contract pairs the prices are for.
        price_a (float): market_a's leg price at n, dollars.
        price_b (float): market_b's leg price at n, dollars.

    Returns:
        CandidatePair: The re-priced copy; the input is not modified.
    """
    side_a, _side_b = leg_sides(pair.pair_type)
    leg_updates = (
        {"pA": price_a, "nB": price_b} if side_a == "yes"
        else {"nA": price_a, "pB": price_b}
    )
    return dc_replace(pair, max_contracts=n, **leg_updates)


def compute_trade(
    pair: CandidatePair, portfolio_value_cents: int, *, settings: LiveSettings | None = None,
    cash_cents: int | None = None,
) -> TradeSpec | None:
    """
    Kelly-size a candidate pair into a TradeSpec.

    With a book (depth_levels), size and price are solved together by
    _solve_marginal_size; without one, the pair's scalar leg prices are used.
    n is then shrunk until the fee-inclusive cost fits the Kelly budget —
    config.kelly_budget: the capped fraction of the portfolio value, never
    more than the cash on hand. The spec also carries cash_need_cents, what
    its two orders can draw at their limit prices, which select_portfolio
    fits into the cash left (shrinking the trade when it no longer fits).

    Kelly, on leg prices:
        fee        = fee_per_pair_approx(price_a, price_b)
        net_spread = (1 - price_a - price_b) - fee
        b          = net_spread / (price_a + price_b + fee)   # dollars at risk
        f*         = p - (1 - p) / b, capped by config.pair_size_cap
    with a time-series p priced at settings.interval_discount.

    The fee is in b's denominator because a losing pair loses its fees too
    (DR-62). f* > 0 means positive EV under the continuous fee approximation
    only: exact ceiling-rounded fees can still leave a boundary spec slightly
    EV-negative at small n, which the min_payoff > 0 check bounds but does not
    remove.

    Args:
        pair (CandidatePair): Must be tradeable. max_contracts 0 means not
            enriched (no depth cap).
        portfolio_value_cents (int): The value Kelly fractions are taken of,
            in cents — cash plus the open positions' value, or the cash alone
            for a caller that holds nothing.
        settings (LiveSettings | None): Keyword-only. The run's toggles; pass
            the object enrichment's affordability bound read. None resolves
            config.live_settings() once (tests and direct calls only).
        cash_cents (int | None): Keyword-only. The cash on hand, in cents; the
            Kelly budget never exceeds it. Pass what enrichment was handed.
            None means the portfolio value is all cash, which sizes exactly as
            a budget of portfolio value x fraction.

    Returns:
        TradeSpec | None: None if the pair is not tradeable, a leg price is
            outside (0, 1), no net spread remains after fees, Kelly is <= 0,
            the budget cannot afford one contract pair, or the exact win
            payoff is <= 0.

    Raises:
        AttributeError/TypeError: If a market's close_time is None. Scanner
            pairs never have one (scanner._filter_active_markets drops them);
            only a hand-built pair can.
        ValueError: If settings is None and a config.py toggle is invalid.
    """
    # Resolved once, so every size this call evaluates reads one k and cap
    settings = live_settings() if settings is None else settings
    if not pair.tradeable:
        return None

    # Book levels from enrichment; () means size on the scalar leg prices
    levels = _depth_levels(pair)

    if levels:
        sized = _solve_marginal_size(pair, levels, portfolio_value_cents, settings,
                                     cash_cents=cash_cents)
        if sized is None:
            return None
        # The size whose OWN fill price justifies it, and that price
        n = sized.n
    else:
        sized = _evaluate_size(pair, levels, 0, portfolio_value_cents, settings,
                               cash_cents=cash_cents)
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

        # BACKSTOP: the shrink changed n outside _evaluate_size, and a smaller
        # n is not always reachable, so check it again (TS-08). Keep it even if
        # it never fires. Each pass stops or lowers n; if nothing is reachable
        # the pair is dropped.
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
    # What the two orders can draw at worst, at the higher of each leg's limit
    # and fill: the cash select_portfolio sets aside for this trade
    cash_need_cents = _order_cash_cents(pair, n, price_a, price_b)

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
        pair = _priced_pair(pair, n, price_a, price_b)

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
        cash_need_cents=cash_need_cents,
        # A same-title spec's model does not read k (its p is the fixed prior)
        interval_discount=(settings.interval_discount
                           if pair.pair_type == "time_series" else None),
    )


def _cash_need(spec: TradeSpec) -> int:
    """
    Whole cents select_portfolio takes from the cash left for one spec.

    A spec from compute_trade (or _spec_at_count) carries it as
    cash_need_cents, what its two orders can draw at their limit prices. A
    spec built by hand carries None, and its fee-inclusive cost stands in,
    rounded up to the cent both as a whole and leg by leg (the rounding the
    shard funder uses), whichever is more.

    Args:
        spec (TradeSpec): The spec to price.

    Returns:
        int: Whole cents, >= 0.
    """
    need = spec.cash_need_cents
    # Read by type, as bool is an int
    if isinstance(need, int) and not isinstance(need, bool):
        return need
    return max(leg_cash_cents(spec.total_cost_with_fees),
               leg_cash_cents(spec.cost_with_fees_a) + leg_cash_cents(spec.cost_with_fees_b))


def _spec_at_count(spec: TradeSpec, n: int) -> TradeSpec | None:
    """
    The same trade at n contract pairs, priced at n's own fill.

    With a book (depth_levels) the legs are priced over the first n contracts
    (scanner.prefix_fill_prices) and the pair is re-priced (_priced_pair);
    without one, the pair's scalar leg prices stand. Every figure that
    depends on n is recomputed exactly as compute_trade computes it — the two
    exact fees, total_cost, total_cost_with_fees, min_payoff, the per-market
    costs and cash_need_cents — while kelly_p, kelly_fraction, profit_ratio,
    monthly_profit_ratio and days_to_close keep the spec's values: they are
    reporting and ranking figures, as they are after compute_trade's own fee
    shrink.

    No Kelly, spread or edge check is repeated at a smaller n, because on a
    book enrichment built both legs' prices only fall as n falls (each leg's
    levels ascend), and a lower price never undoes a check the larger size
    passed. Kelly: with c the cost of a contract pair plus its fee, a
    same-title b = 1/c - 1 rises as c falls, and a time-series
    f* = 1 - k(pB - pA)/(1 - c) rises as pA and nB fall (pB at or above the
    later market's YES bid, which enrichment checks, makes (pB - pA)/(1 - c)
    fall with them); the count Kelly affords at the lower price is therefore
    at least the one it afforded before, which was more than n. Spread:
    pB - pA only widens, so it stays above the entry floor, and the band's
    ceiling was tested at the top of the qualifying book, which no prefix
    exceeds. Edge after the fee: enrichment kept only levels that each keep
    one.

    What is checked, since it can fail at a smaller n: that the book holds n
    contracts; that the n are reachable at their own fill-or-kill limits
    (_reachable_contracts — the supported sizes can have holes); that a win
    still pays after the exact fees; and that the trade's expected value is
    still positive at the exact fees, p x win payoff > (1 - p) x the
    fee-inclusive cost, with p the model's probability of profit at n's own
    price (_kelly_p_at, at the spec's interval_discount for a time-series
    pair). Kelly prices the fee with the continuous approximation, which
    the exact fees, each rounded up to the cent, exceed by up to a cent per
    leg; at the small counts a shrink often lands on, that rounding alone
    can turn a trade the model rates positive into one it rates negative.
    compute_trade's own spec is held only to a positive win payoff, so a
    shrunk spec meets a stricter test than the one it came from.

    Args:
        spec (TradeSpec): A spec from compute_trade.
        n (int): The contract pairs wanted. Range: >= 1.

    Returns:
        TradeSpec | None: A new spec at n, or None if the book holds fewer
            than n contracts, n is not reachable at its own limits, a win no
            longer pays after the exact fees, the expected value at the exact
            fees is not positive, or the spec is a time-series one that does
            not carry its interval_discount. The input is not modified.
    """
    pair = spec.pair
    k = spec.interval_discount
    if pair.pair_type == "time_series" and (
            not isinstance(k, (int, float)) or isinstance(k, bool)):
        # Its probability of profit at n cannot be priced without its k
        return None
    levels = _depth_levels(pair)
    if levels:
        fills = prefix_fill_prices(levels, n)
        if fills is None:
            # The book holds fewer than n contracts
            return None
        price_a, price_b = fills
        # A fill-or-kill only buys depth at or under its own limit (TS-08)
        if _reachable_contracts(pair, levels, price_a, price_b) < n:
            return None
    else:
        # No book: the pair's stored leg prices (leg_prices maps them)
        price_a, price_b = leg_prices(pair)

    fee_a = fee_leg_exact(n, price_a)
    fee_b = fee_leg_exact(n, price_b)
    # Exact-fee win payoff. PRICE_EPSILON: a payoff that is exactly zero (one
    # pair at 0.01 + 0.97 with two one-cent fees) evaluates a hair above it
    min_payoff = n * (1.0 - price_a - price_b) - fee_a - fee_b
    if min_payoff <= PRICE_EPSILON:
        return None
    total_cost = n * (price_a + price_b)
    total_cost_with_fees = total_cost + fee_a + fee_b
    # The model's p at n's own YES price (the same helper the sizer prices with)
    p = _kelly_p_at(pair, price_a, k)
    if p * min_payoff <= (1.0 - p) * total_cost_with_fees:
        # Not a positive expected value once the fees are rounded up
        return None
    return dc_replace(
        spec,
        pair=_priced_pair(pair, n, price_a, price_b) if levels else pair,
        x=n,
        y=n,
        total_cost=total_cost,
        total_cost_with_fees=total_cost_with_fees,
        min_payoff=min_payoff,
        cost_with_fees_a=n * price_a + fee_a,
        cost_with_fees_b=n * price_b + fee_b,
        cash_need_cents=_order_cash_cents(pair, n, price_a, price_b),
    )


def _shrink_to_cash(spec: TradeSpec, available_cents: int) -> TradeSpec | None:
    """
    The largest smaller size of a spec whose orders fit the cash left.

    Scans down from the most contract pairs the cash could possibly buy — at
    the book's cheapest level, fees aside, so no larger count can fit — and
    returns the first count whose own spec (_spec_at_count) exists and whose
    cash_need_cents fits. Every candidate is verified rather than inferred
    from its neighbours: the sizes a book supports can have holes (a count
    below a reachable one can be unreachable at its own, lower, limits), so a
    step down at one fixed price could stop at a hole and under-size. The
    scan prices the book once per candidate, so its cost grows with the count
    the cash could buy: a few milliseconds at this account's size, and
    seconds per shrunk spec on cheap legs with hundreds of thousands of
    dollars left.

    Args:
        spec (TradeSpec): A spec whose cash need exceeds the cash left.
        available_cents (int): The cash left, in whole cents.

    Returns:
        TradeSpec | None: The spec at the largest count below spec.x that
            fits, or None when not one contract pair does (a spec of one
            contract pair has nothing smaller, and no price is read for it).
    """
    if spec.x <= 1:
        # Nothing smaller to try
        return None
    levels = _depth_levels(spec.pair)
    cheapest = (levels[0][0] + levels[0][1]) if levels else sum(leg_prices(spec.pair))
    if cheapest <= 0:
        return None
    # No count above this can fit: even the cheapest level costs more than the cash
    upper = min(spec.x - 1, int((available_cents / 100.0) / cheapest))
    for n in range(upper, 0, -1):
        shrunk = _spec_at_count(spec, n)
        if shrunk is not None and shrunk.cash_need_cents <= available_cents:
            return shrunk
    return None


def select_portfolio(specs: list, cash_cents: int, *,
                     held_ladders: frozenset = frozenset()) -> list:
    """
    Greedy portfolio: best monthly return first, within the cash, no reused
    tickers, and at most one time-series trade per ladder.

    Walks specs by monthly_profit_ratio descending (same_title before
    time_series on ties — it is the near-arbitrage), spending the cash in
    whole cents. A spec is taken if neither ticker is already used this run;
    it takes its cash_need_cents from the cash left (_cash_need: what its two
    orders can draw at their limit prices, rounded up per leg the way the
    shard funder rounds it, so trader._required_cents_by_shard never finds
    the selected portfolio short). A spec whose need exceeds the cash left is
    shrunk to the largest size that fits (_shrink_to_cash) and taken at that
    size, logged on its own line; it is skipped only when not one contract
    pair fits, and a skipped spec is not a stop, so a cheaper one further
    down can still be taken. Open positions from earlier runs are excluded
    upstream by scanner.get_held_tickers (prod only).

    A time-series spec is also skipped when one of its markets is on a ladder
    the account holds, or on the ladder of a spec picked earlier (a ladder is
    one question asked at several deadlines; see scanner.ladder_keys). A
    same-title spec is never skipped this way, but its ladders count once
    picked, and so do a shrunk spec's. A skipped spec spends no cash and
    claims no ladder, so a later spec may then be taken, or taken at a larger
    size. The backtester repeats the ticker and ladder rules (change both
    together) and budgets through the same config.kelly_budget, so it too
    shrinks a trade to the cash left; it spends a trade's fee-inclusive cost
    at its entry prices (candle quotes, with no cash_need_cents reserve) and
    does not repeat _spec_at_count's expected-value check on a shrunk trade.

    Args:
        specs (list): TradeSpecs from compute_trade.
        cash_cents (int): The cash the portfolio may spend, in whole cents
            (every shard's together).
        held_ladders (frozenset): Keyword-only. Ladder labels of the markets we
            hold. Empty means none.

    Returns:
        list: The selected specs in ranking order — a shrunk spec as the new,
            smaller spec in its place; may be empty.
    """
    # Whole cents, at what the orders can draw at worst, so the shard funder
    # (trader._required_cents_by_shard) never finds the portfolio short
    available = cash_cents

    # Primary sort: monthly_profit_ratio descending (best capital efficiency first).
    # Secondary sort: same_title > time_series at equal monthly return.
    specs_sorted = sorted(
        specs,
        key=lambda s: (s.monthly_profit_ratio, s.pair.pair_type == "same_title"),
        reverse=True,
    )
    selected = []
    used_tickers: set[str] = set()
    # Ladders we already hold, plus those of the specs picked earlier in this run
    used_ladders: set = set(held_ladders)
    ladder_skips = 0
    shrinks = 0
    for spec in specs_sorted:
        ta = spec.pair.market_a.ticker
        tb = spec.pair.market_b.ticker
        # Skip trades that would re-use a ticker already committed to a higher-priority pair
        if ta in used_tickers or tb in used_tickers:
            continue
        keys = pair_ladder_keys(spec.pair)
        # At most one open time-series trade per ladder
        if spec.pair.pair_type == "time_series" and keys & used_ladders:
            ladder_skips += 1
            continue
        need = _cash_need(spec)
        if need > available:
            # Shrink to the cash left; skip only when not one contract pair fits
            shrunk = _shrink_to_cash(spec, available)
            if shrunk is None:
                continue
            logging.info(
                "Shrunk '%s' from %d to %d contract pairs to fit the $%.2f of cash left",
                spec.pair.canonical_title, spec.x, shrunk.x, available / 100,
            )
            spec, need = shrunk, shrunk.cash_need_cents
            shrinks += 1
        selected.append(spec)
        available -= need
        used_tickers.add(ta)
        used_tickers.add(tb)
        used_ladders |= keys
    if ladder_skips:
        logging.info(
            "Time-series trades skipped because the account already holds, or this "
            "run already picked, a trade on the same ladder: %d",
            ladder_skips,
        )
    if shrinks:
        logging.info("Trades shrunk to fit the cash left: %d", shrinks)
    logging.info(
        # The cost at the fill prices, fees included (TS-12), and the cash the
        # walk set aside for the orders at their limit prices
        "Portfolio: %d trades selected, total cost $%.2f incl. fees "
        "(up to $%.2f of cash at the orders' limit prices)",
        len(selected),
        sum(s.total_cost_with_fees for s in selected),
        (cash_cents - available) / 100,
    )
    return selected
