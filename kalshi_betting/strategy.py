"""
File: strategy.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Converts scanner.py CandidatePair objects into fully sized trade specifications
    using the Kelly criterion, then selects a portfolio subset that fits within
    the available account balance. The Kelly fraction determines how much of the
    balance to allocate to each pair based on the implied edge and the probability
    that the trade is profitable. The two pair types use different probability
    models: a same-title pair is a near-arbitrage priced on the fixed
    SAME_TITLE_CO_RESOLVE_PROB co-resolution prior, while a time-series pair is
    a directional bet (YES on the earlier contract, NO on the later) priced on
    config.time_series_profit_prob — a discounted version of the market-implied
    probability that the event first happens between the two deadlines, which
    is the trade's single loss scenario.

Dependencies:
    Imports BUDGET_FRACTION, SAME_TITLE_CO_RESOLVE_PROB, time_series_profit_prob
    and the fee helpers from config.py. Imports CandidatePair, leg_prices and
    leg_sides from scanner.py — leg_prices() is the only mapping from a pair's
    four quoted prices to the two prices its legs actually cost, and every
    cost, fee and payoff here is computed on those; leg_sides() only names
    the sides in the log line. Exports TradeSpec (consumed by
    trader.py and reporter.py), compute_trade() and select_portfolio() (both
    called by main.py). backtester.py does NOT import this module — it
    re-implements Kelly sizing and portfolio selection inline against the same
    config.py constants, fee helpers and probability model, so a change to
    either sizing formula must be made in both places to keep live/backtest
    parity (the shared config.time_series_profit_prob is what keeps the
    time-series probability itself from drifting).

Notes:
    compute_trade() returns None for the ordinary no-edge/no-budget cases, but
    see its own Raises section for the one input it does NOT handle by
    returning None: a pair whose market_a or market_b has close_time=None. The
    scanner guarantees that can't happen for pairs it produced — only a
    CandidatePair built outside scanner.py (e.g. in a test) can carry one.

    TradeSpec.cost_with_fees_a is always MARKET_A's leg cost and
    cost_with_fees_b is MARKET_B's, whatever side each leg buys — the
    collateral planner (trader._required_cents_by_shard) pairs them with
    market_a/market_b.exchange_index. The trader submits the NO leg first
    (market_a for same_title, market_b for time_series); that ordering lives
    in trader._ordered_legs, not here.
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
    Fully computed trade specification for a candidate pair, ready for execution.

    Encodes both the trade parameters (contract counts, costs, payoff) and the
    Kelly-sizing metadata used to rank and select trades for the portfolio.
    Throughout, "price_a"/"price_b" are the LEG prices from
    scanner.leg_prices(pair): (nA, pB) for a same-title pair (NO on market_a,
    YES on market_b) and (pA, nB) for a time-series pair (YES on market_a, NO on
    market_b).

    Attributes:
        pair (CandidatePair): The underlying candidate this trade is based on.
        x (int): Number of contracts to buy on market A (NO for same_title, YES
            for time_series — see scanner.leg_sides). Always equals y.
        y (int): Number of contracts to buy on market B (YES for same_title, NO
            for time_series). Always equals x.
        total_cost (float): Total dollar cost of the contracts: x * (price_a + price_b).
            Excludes taker fees — used for reporting.
        total_cost_with_fees (float): total_cost plus the exact ceiling-rounded
            taker fee for both legs. This is the real cash the trade consumes at
            execution — select_portfolio() budgets against this value.
        min_payoff (float): Dollar profit in a win scenario, net of exact
            ceiling-rounded taker fees on both legs:
            x * (1 - price_a - price_b) - fee_leg_exact(x, price_a) - fee_leg_exact(x, price_b).
            For a same_title pair this is the profit floor whenever the two
            questions co-resolve (both legs settle, one pays). For a time_series
            pair it is the profit in EITHER win scenario — event by A's deadline
            (YES-on-A pays) or never by B's (NO-on-B pays) — while the in-between
            scenario (A=NO, B=YES) loses total_cost_with_fees in full. Always > 0
            for trades that reach execution; the field name is kept for the
            reporter/trader consumers.
        profit_ratio (float): Return on the CONTRACTS' cost in a win scenario,
            net of the continuous fee approximation:
            ((1 - price_a - price_b) - fee_per_pair_approx(price_a, price_b)) /
            (price_a + price_b). Reporting and ranking only — it feeds
            monthly_profit_ratio, which select_portfolio sorts on, and the prod
            log's Profit Ratio column. It is NOT "b" in the Kelly formula:
            Kelly's b divides the same numerator by the dollars actually AT RISK
            (price_a + price_b + fee_per_pair_approx(...)), because the losing
            cell loses total_cost_with_fees in full. The two are deliberately
            distinct quantities — see compute_trade().
        days_to_close (int): Calendar days until the later-closing market resolves. >= 1.
        monthly_profit_ratio (float): Profit ratio normalized to a 30-day period:
            profit_ratio * 30 / days_to_close. Used for portfolio ranking.
        kelly_p (float): Probability of profit used in the Kelly formula. For
            time_series pairs this is config.time_series_profit_prob(pA, pB) —
            one minus the discounted market-implied in-between probability; for
            same_title it is the fixed SAME_TITLE_CO_RESOLVE_PROB prior. Range: (0, 1].
        kelly_fraction (float): Kelly fraction capped at BUDGET_FRACTION (20%). This is the
            fraction of account balance allocated to this trade.
        cost_with_fees_a (float): MARKET_A's leg cash requirement, whatever side
            that leg buys: x * price_a + that leg's exact ceiling-rounded taker fee
            (fee_leg_exact(x, price_a)). Used by the collateral transfer planner to
            fund market_a's exchange shard. Invariant: cost_with_fees_a +
            cost_with_fees_b == total_cost_with_fees (same terms, same fee calls).
            Defaults to 0.0 for TradeSpec constructions that don't populate it.
        cost_with_fees_b (float): MARKET_B's leg cash requirement: y * price_b +
            that leg's exact ceiling-rounded taker fee (fee_leg_exact(y, price_b)).
            Used by the collateral transfer planner to fund market_b's exchange
            shard. Same invariant and default as cost_with_fees_a.
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
    # Per-MARKET cash requirements (market_a / market_b, whatever side each leg
    # buys), used by the collateral transfer planner to fund each market's shard;
    # invariant: cost_with_fees_a + cost_with_fees_b == total_cost_with_fees
    # (same terms, same fee calls). Defaulted so existing constructions don't break.
    cost_with_fees_a: float = 0.0
    cost_with_fees_b: float = 0.0


def _depth_levels(pair: CandidatePair) -> tuple:
    """
    Return the pair's qualifying order-book depth, or () when it has no book.

    Read defensively by TYPE, not by truthiness. A MagicMock standing in for a
    pair — which most of this module's tests use — answers any attribute with a
    truthy auto-attribute that iterates into nonsense, and a pair whose depth we
    cannot actually read must be priced on its scalar leg prices rather than on
    a guess. Same fail-safe rule scanner.leg_sides applies to an unknown
    pair_type: when the input is not recognisable, take the conservative branch.

    Args:
        pair (CandidatePair): The candidate pair, or any stand-in for one.

    Returns:
        tuple: The (price_a, price_b, qty) levels enrichment stored, in MARKET
            order; () when the attribute is missing, None, or not a list/tuple.
    """
    levels = getattr(pair, "depth_levels", None)
    return tuple(levels) if isinstance(levels, (tuple, list)) else ()


def _kelly_p(pair: CandidatePair) -> float:
    """
    Probability that the pair results in a profit, per the pair type's model.

    time_series: p = config.time_series_profit_prob(pair.pA, pair.pB), i.e.
    1 - k * (pB - pA) with k = TIME_SERIES_INTERVAL_PROB_DISCOUNT. The trade
    (YES on the earlier contract, NO on the later) has three settlement cells:
    event by A's deadline (A=YES, hence B=YES; YES-on-A pays), never by B's
    (A=NO, B=NO; NO-on-B pays), and in between (A=NO, B=YES; total loss).
    A=YES/B=NO is impossible for a cumulative-deadline pair. The YES-ask gap
    pB - pA is the market-implied probability of the loss cell; the model
    believes only the fraction k of it. At k = 1 (take the market at face
    value) Kelly is <= 0 for every pair and nothing trades — the edge exists
    only if the market systematically overstates the in-between probability.

    Inputs after enrichment: pair.pA is the depth-weighted YES fill on A and
    pair.pB is B's best YES ask re-read from the book fetched in the same pass
    (scanner._reference_yes_ask), so pB - pA is a single-snapshot gap whenever
    B's NO side carried resting bids; when it did not, pB keeps its scan-time
    value. That matters because time_series_profit_prob clamps the gap at zero:
    a pA that had risen to or past pB would clamp to zero and return p = 1.0,
    modelling the pair as riskless. Enrichment therefore drops any time-series
    pair whose reference is not above the YES fill: strictly above when the
    reference was re-read fresh, and clearing the whole deadline-gap tier when
    it fell back to the scan-time quote, since a mixed-snapshot gap of a
    thousandth would otherwise pass. compute_trade returns None on a
    non-tradeable pair before reaching here (strategy.py:58), so every pair
    this prices from the live pipeline satisfies pB > pA and the clamp is
    unreachable. The gap is kept on YES asks (pB - pA) rather than on
    the executable spread because the YES-ask gap is the smaller, more
    conservative estimate of the mass the market assigns to the loss cell.

    same_title: p = SAME_TITLE_CO_RESOLVE_PROB — fixed prior for markets confirmed to ask
    the exact same question (matching event_title + title + subtitle, see scanner.pair_key).
    Divergence is an anomaly, not an expected outcome, so a fixed base rate is more
    appropriate than market prices. This prior is calibrated for binary contracts and
    applies equally to MVE option markets once cross-event collisions are eliminated by
    the event_title component of the grouping key.

    Args:
        pair (CandidatePair): The candidate pair. Uses pair.pair_type to select
            the model and, for time_series, pair.pA/pair.pB as the probability
            inputs.

    Returns:
        float: Probability of profit, in (0, 1], used as "p" in compute_trade()'s
            Kelly formula.
    """
    # The pair's own stored YES-leg quote. compute_trade calls _kelly_p_at
    # directly instead, with the price of the quantity it is actually sizing.
    return _kelly_p_at(pair, pair.pA)


def _kelly_p_at(pair: CandidatePair, yes_leg_price: float) -> float:
    """
    _kelly_p with the YES leg's price supplied rather than read off the pair.

    For a time-series pair the probability model is a function of the YES-ask
    gap pB - pA, and pA is a LEG price — so it depends on how many contracts are
    being bought. compute_trade's marginal-price descent re-prices the legs at
    each candidate size and must re-derive p from that price, not from the
    single scalar enrichment happened to write. For a same-title pair p is the
    fixed co-resolution prior and the argument is ignored.

    Args:
        pair (CandidatePair): The candidate pair. pair_type selects the model
            and, for time_series, pair.pB supplies the reference quote — which
            is NOT a leg price and so does not move with the size.
        yes_leg_price (float): Price of the YES leg at the size being sized,
            dollars in (0, 1). For a time-series pair this is pA (market_a's
            leg); ignored for same_title.

    Returns:
        float: Probability of profit, in (0, 1], used as "p" in compute_trade()'s
            Kelly formula.
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
        profit_ratio (float): REPORTED return on the contracts' cost at that
            price — net_spread / (price_a + price_b). Not Kelly's "b", whose
            denominator also carries the fee; see _evaluate_size.
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
    How many contract pairs a V2 fill-or-kill priced at these legs can actually buy.

    A V2 taker order is ONE fill-or-kill limit per leg, so it buys only the
    depth resting at or below its own limit price — and that limit is a
    PER-CONTRACT cap derived from the leg's price (scanner.v2_limit_price),
    while the price itself is the quantity-weighted average of a MULTI-LEVEL
    prefix. An average over a ladder sits BELOW the ladder's top level, so the
    top of the very prefix being priced can rest above the cap it produces,
    and the order is killed (TS-08). This counts what the order really reaches.

    The caps come from scanner.v2_effective_cap — the same arithmetic the order
    body will carry, including the NO leg's complement round-trip, which can
    TIGHTEN the effective NO cap on a sub-cent grid. Never re-derive it here:
    the whole finding is that the size and the cap disagree.

    Both comparisons carry PRICE_EPSILON for the same reason every other price
    comparison does — a level sitting exactly ON the cap must not be dropped by
    float representation noise (TS-09).

    Args:
        pair (CandidatePair): The pair being sized; supplies pair_type (for
            which side each market buys) and both markets' tick grids.
        levels (tuple): The pair's qualifying depth as (price_a, price_b, qty)
            in MARKET order, ascending by combined price — depth_levels.
        price_a (float): market_a's leg price for the size being tested, in
            dollars — the price that leg's cap is derived from.
        price_b (float): market_b's leg price, likewise.

    Returns:
        float: Total contract pairs resting at or below BOTH legs' caps. A size
            larger than this cannot fill.
    """
    # leg_sides is the only source of truth for which side each market buys,
    # and levels are already oriented to market order, so this is a direct index
    side_a, side_b = leg_sides(pair.pair_type)
    cap_a = float(v2_effective_cap(f"buy_{side_a}", price_a, pair.market_a))
    cap_b = float(v2_effective_cap(f"buy_{side_b}", price_b, pair.market_b))
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

    # Validate that both leg prices are in the open interval (0, 1).
    # Edge cases at 0 or 1 indicate a settled market and would break the fee formula.
    if price_b <= 0.0 or price_b >= 1.0 or price_a <= 0.0 or price_a >= 1.0:
        return None

    if levels and ORDER_API_VERSION == "v2":
        # A V2 FoK buys only the depth resting at or below its own per-contract
        # limit, and that limit comes from the prefix AVERAGE, which on a ladder
        # sits below the prefix's own top level. Size n is unfillable when it
        # reaches past what its own price can pay for (TS-08). Returning None
        # makes _solve_marginal_size treat n as too big and search lower, so the
        # search itself converges on the largest self-consistent size — no
        # separate fixed-point loop is needed.
        #
        # Gated on the V2 path only: the legacy cap is a TOTAL-COST cap
        # (buy_max_cost) which CAN sweep a ladder, so applying this there would
        # shrink legacy sizes for no reason.
        if _reachable_contracts(pair, levels, price_a, price_b) < n:
            return None

    # Subtract the continuous fee approximation from the gross spread to get the
    # net edge. A zero or negative net_spread means the trade costs more than it pays.
    fee_approx = fee_per_pair_approx(price_a, price_b)
    net_spread = (1.0 - price_a - price_b) - fee_approx
    if net_spread <= 0:
        return None

    # REPORTED return on the contracts' cost, net of the fee approximation. This
    # is what feeds monthly_profit_ratio (select_portfolio's ranking key) and the
    # prod log's Profit Ratio column. It is deliberately NOT Kelly's "b" — its
    # denominator excludes the fee; see kelly_b below.
    profit_ratio = net_spread / (price_a + price_b)

    # Probability of profit at the price of THIS size — not at the pair's stored
    # pA, which is only one point on the book
    p = _kelly_p_at(pair, price_a)
    q = 1.0 - p
    # Kelly's "b" is the win payoff per dollar AT RISK, and the dollars at risk
    # include the fee: fees are cash out the door at execution, and the losing
    # cell loses total_cost_with_fees in full (see TradeSpec.min_payoff). Using
    # the fee-less cost as the denominator made f* > 0 whenever
    # p*net_spread > q*(price_a + price_b), while true positive EV needs
    # p*net_spread > q*(price_a + price_b + fee) — the gate overstated EV by
    # exactly q*fee on every pair, which for a time-series bet (q = k*(pB - pA),
    # routinely > 0.5) admitted marginal pairs with negative true EV (DR-62).
    # A DIFFERENT quantity from profit_ratio above; do not collapse the two.
    kelly_b = net_spread / (price_a + price_b + fee_approx)

    # Kelly formula: f* = p - q/b. With the fee-inclusive risk above, f* > 0 is
    # exactly the condition p*net_spread > q*(cost + fee) — positive expected
    # value under the CONTINUOUS fee approximation. It is not the whole
    # guarantee: fee_per_pair_approx UNDERESTIMATES the ceiling-rounded
    # fee_leg_exact that min_payoff and total_cost_with_fees are actually built
    # from, so a spec sitting within roughly q*(exact - approx fee)/net_spread of
    # the boundary can still be marginally EV-negative on its own exact-fee
    # fields at single-digit n. That residual is the small-n rounding regime the
    # min_payoff > 0 check below exists for; it bounds the residual without
    # eliminating it.
    kelly_fraction = p - q / kelly_b
    if kelly_fraction <= 0:
        # Kelly says don't bet — expected value is not positive once the fee is
        # counted on both sides of the wager
        return None

    # Cap at BUDGET_FRACTION (20%) to avoid over-concentrating in a single pair
    kelly_fraction_capped = min(BUDGET_FRACTION, kelly_fraction)

    # Convert the Kelly fraction to a dollar budget, then derive the integer contract count
    budget_dollars = (balance_cents / 100.0) * kelly_fraction_capped
    # Cross-module: the single definition of the budget -> contracts step. The
    # scanner calls the same helper with BUDGET_FRACTION and the best level's
    # price sum, so its depth cap is provably an upper bound on this count.
    target = max_affordable_pairs(balance_cents, price_a + price_b, kelly_fraction_capped)

    # Respect the order book depth limit set by scanner.enrich_with_orderbook_prices()
    if pair.max_contracts > 0:
        target = min(target, pair.max_contracts)
    if target < 1:
        # The Kelly budget can't afford even one contract pair — forcing n=1
        # would silently exceed both the Kelly fraction and BUDGET_FRACTION
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
    Find the largest contract count whose own marginal fill price still justifies it.

    Size and price are mutually dependent: buying more means eating further down
    the book into worse-priced levels, and a worse average price both shrinks
    Kelly's "b" and shrinks what the budget affords. A count n is SUPPORTED when
    every gate passes at the price of n contracts AND the Kelly budget at that
    price still affords at least n. This binary-searches [1, pair.max_contracts]
    for the largest supported count.

    The upper bound is enrichment's own affordability cap — BUDGET_FRACTION over
    the best level's price sum, i.e. the largest fraction over the cheapest
    possible prefix — so no count above it can ever be supported and the search
    range is exhaustive.

    Safety does not rest on the predicate being perfectly downward-closed. It
    very nearly is (a smaller count fills at a better price, which relaxes every
    gate, and b rises faster than p falls), but the guarantee here is simpler:
    a count is only ever returned after being DIRECTLY verified, so a search that
    lands low under-sizes rather than mis-prices. That also makes the search the
    right shape for the gates themselves — evaluating only the top of the range,
    as a plain descent would, would abandon a pair whose Kelly fraction is
    negative at full depth but comfortably positive at a realistic size, which is
    exactly the pair this whole mechanism exists to rescue.

    Args:
        pair (CandidatePair): The pair being sized; max_contracts bounds the search.
        levels (tuple): The pair's qualifying depth, non-empty.
        balance_cents (int): Account balance in integer cents.

    Returns:
        _Sizing | None: The largest verified-supported candidate, or None when
            no count in the range is supported.
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
    # Unreachable: a bisection of [1, max_contracts] closes in about
    # log2(max_contracts) passes, ~20 even for a million-contract book. Kept so
    # an edit that breaks the halving costs one under-sized pair and a WARNING
    # rather than a hung weekly run — best is already verified, so returning it
    # is safe.
    logging.warning(
        "Marginal size for '%s' did not converge in %d passes — using the largest "
        "count verified so far",
        pair.canonical_title, SIZE_SOLVE_MAX_ITERATIONS,
    )
    return best


def compute_trade(pair: CandidatePair, balance_cents: int) -> TradeSpec | None:
    """
    Compute a Kelly-sized trade specification for a candidate pair.

    Applies the Kelly criterion to determine the optimal fraction of the account
    balance to allocate, then derives the integer contract count and verifies that
    the exact win-scenario profit (after ceiling-rounded per-leg fees) remains
    positive. All costs, fees and payoffs are computed on the LEG prices
    (price_a, price_b) = scanner.leg_prices(pair): (nA, pB) for same_title,
    (pA, nB) for time_series.

    Kelly formula used:
        fee = fee_per_pair_approx(price_a, price_b)
        net_spread = (1 − price_a − price_b) − fee   [win payoff per contract pair]
        b = net_spread / (price_a + price_b + fee)   [payoff per dollar AT RISK]
        f* = p - (1-p)/b                             [optimal Kelly fraction]
        f_capped = min(BUDGET_FRACTION, f*)

    The fee sits in the DENOMINATOR as well as the numerator, and that is the
    whole point: fees are cash out the door at execution, so a losing pair loses
    total_cost_with_fees, not price_a + price_b. With the fee-less denominator
    the gate passed whenever p*net_spread > q*(price_a + price_b) while true
    positive EV needs p*net_spread > q*(price_a + price_b + fee), overstating EV
    by exactly q*fee on every pair (DR-62). f* > 0 now means positive expected
    value under the CONTINUOUS fee approximation. It does NOT mean positive EV
    on the spec's own exact-fee fields: fee_per_pair_approx underestimates the
    ceiling-rounded fee_leg_exact that min_payoff and total_cost_with_fees carry,
    so a spec within roughly q*(exact - approx fee)/net_spread of the boundary
    can still be marginally EV-negative at single-digit n (measured: at a
    $10,000 balance, pA=0.12 / pB=0.33 / nB=0.70 accepts at n=30 with
    p*min_payoff - q*total_cost_with_fees = -$0.005). The min_payoff > 0 check
    below bounds that small-n residual; it does not eliminate it. Note also that
    b is NOT the reported TradeSpec.profit_ratio, which keeps the fee-less
    denominator as a return-on-contract-cost figure.

    p comes from _kelly_p (the co-resolution prior for same_title; the
    discounted in-between model for time_series). For a time-series pair the
    result is a directional bet: f* is only positive when the modelled loss
    probability k * (pB - pA) is small enough relative to b, and a wide book
    (large price_a + price_b for the same YES-ask gap) drives it negative.
    Same-title pairs are barely moved by the fee-inclusive denominator (q is the
    fixed 1 − SAME_TITLE_CO_RESOLVE_PROB = 0.05, so q*fee is small — though not
    zero: on a whole-cent grid over the same-title admissible region 12 of 4,465
    price points, 0.27%, change verdict, always accept -> reject); the
    time-series bet is moved far more, since q = k*(pB − pA) routinely exceeds
    0.5 on exactly the wide-gap pairs the strategy targets.

    Args:
        pair (CandidatePair): The candidate pair. Must have tradeable=True.
            Uses the leg prices from scanner.leg_prices() (depth-weighted fill
            prices after scanner.enrich_with_orderbook_prices), pair.pA/pair.pB
            for the time-series probability model, and pair.max_contracts
            (qualifying order book depth, 0 = uncapped).
        balance_cents (int): Current account balance in cents. Used to convert the
            Kelly fraction to a dollar budget for contract sizing.

    Returns:
        Optional[TradeSpec]: A fully specified trade including contract count n,
            total cost, win-scenario payoff (min_payoff), time-normalized monthly
            return, Kelly metadata, and each market's own fee-inclusive cash
            requirement (cost_with_fees_a for market_a, cost_with_fees_b for
            market_b — used by the collateral transfer planner to fund each
            leg's exchange shard). Returns None if:
            - The pair is not tradeable.
            - Either leg price is out of the valid (0, 1) range.
            - The net spread is zero or negative after fees.
            - The Kelly fraction is zero or negative (no edge).
            - The Kelly budget cannot afford a single contract pair.
            - The exact win-scenario payoff at the computed n is zero or negative.

    Raises:
        AttributeError/TypeError: If either market_a.close_time or
            market_b.close_time is None (a market whose close_time failed to
            parse). Every other invalid or unprofitable input is handled by
            returning None; this one case is not, because it cannot arise from
            scanner-produced pairs: scanner._filter_active_markets drops every
            market with a missing/unparseable close_time, and both
            find_time_series_pairs and find_same_title_pairs call it first. A
            None close_time can therefore only reach here on a CandidatePair
            constructed outside the scanner (e.g. in a test).
    """
    if not pair.tradeable:
        return None

    # The qualifying order-book depth enrichment left on the pair. With it, the
    # size and the price are solved together below; without it — a pair built
    # outside the scanner, or the bare pair the backtester's Kelly-parity test
    # constructs — the pair's scalar leg prices are all there is.
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
        # No book to re-price against: the single-shot sizing this always did
        n = sized.target

    price_a, price_b = sized.price_a, sized.price_b
    p                = sized.p
    profit_ratio     = sized.profit_ratio
    kelly_fraction_capped = sized.kelly_fraction
    budget_dollars   = sized.budget_dollars

    # Compute exact ceiling-rounded fees for the final integer n. budget_dollars
    # above only covers the contract cost (n * (price_a + price_b)) — fees are
    # added on top, so the straight n from that division can push
    # total_cost_with_fees slightly past the capped Kelly budget. Shrink n until
    # the fee-inclusive cost actually fits, so the real cash consumed never
    # exceeds what the Kelly fraction (and BUDGET_FRACTION) allowed.
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
        # The shrink loop moved n, so re-price over exactly the contracts this
        # trade will submit — that price is what trader._ordered_legs reads for
        # the FoK limit and the rollback floor. A smaller n can only reach
        # fewer, cheaper levels, so the fee-inclusive cost stays inside
        # budget_dollars and the loop above needs no second pass. p,
        # profit_ratio and kelly_fraction_capped are deliberately left at their
        # pre-shrink values: they are reporting/ranking figures, recomputing
        # them would reopen the non-monotone Kelly question above for no gain,
        # and the pre-shrink price is the conservative side of the estimate.
        fills = prefix_fill_prices(levels, n)
        if fills is not None:
            price_a, price_b = fills
            fee_a = fee_leg_exact(n, price_a)
            fee_b = fee_leg_exact(n, price_b)

        # BACKSTOP, not a live path. The shrink decrements n OUTSIDE
        # _evaluate_size, so the count above is no longer one the reachability
        # gate has verified, and reachability is genuinely NOT downward-closed:
        # a cheaper prefix means a LOWER cap, and the cap can fall a tick while
        # the deepest level the prefix still touches does not. On NO 0.30@1000
        # then 0.33@600 the supported set is [1,1000] and [1501,1600] with a
        # HOLE between, because n=1600 prices at 0.31125 for a 0.33 cap while
        # n=1500 prices at 0.31000 for a 0.32 cap.
        #
        # That hole is why the snap has never been observed to fire, and the
        # reason is worth stating so nobody "simplifies" it away on the grounds
        # that it looks dead: _solve_marginal_size BISECTS [1, max_contracts],
        # so on a set like the above it converges to the top of the LOWER
        # island (1000 here) and cannot land in the upper one at all. Every
        # count it can return therefore sits in a region where the cap covers
        # the whole reachable prefix and shrinking keeps that true. A search
        # that landed differently — a changed seed, a third level, a future
        # non-bisecting solver — would not have that property, and the
        # invariant this enforces is the one the money depends on.
        #
        # Snap n down to what the order actually reaches and re-price until the
        # two agree. It terminates for the same reason a level-filtering loop
        # would: the reachable set is a price-capped PREFIX of a ladder ordered
        # by combined price, so each pass strictly shrinks it, and the cheapest
        # level alone is always self-consistent (ceil(best) + slippage >= best).
        # The fee budget needs no second pass either — a smaller n is strictly
        # cheaper (TS-08).
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
                # Nothing the cap can reach — the same verdict compute_trade
                # already returns when no count survives its gates.
                return None

    # Verify the win-scenario payoff is positive after exact fees. At very
    # small n the ceiling rounding can eat the entire profit margin. (For a
    # time-series pair this is the profit in either win cell, not a floor —
    # the in-between cell loses the whole stake.)
    min_payoff = n * (1.0 - price_a - price_b) - fee_a - fee_b
    if min_payoff <= 0:
        return None

    total_cost = n * (price_a + price_b)
    # Fees are cash out the door at execution — the portfolio budget must cover them
    total_cost_with_fees = total_cost + fee_a + fee_b

    # Per-MARKET cash requirements — same terms and fee calls as
    # total_cost_with_fees, just not summed together. cost_with_fees_a is
    # market_a's leg and cost_with_fees_b is market_b's, whatever side each
    # buys: trader.ensure_shard_collateral() pairs them with each market's own
    # exchange shard rather than funding the pair total on one shard.
    cost_with_fees_a = n * price_a + fee_a
    cost_with_fees_b = n * price_b + fee_b

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

    # Name the sides in the log so a time-series line (YES on A at pA, NO on B
    # at nB) is never misread as the same-title NO/YES layout. Also the mapping
    # the solved prices are written back through, below.
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
        # Hand the solved prices back on the pair itself, through the same
        # leg_sides mapping enrichment writes them with. leg_prices(spec.pair)
        # is the documented single source of truth for what a leg costs, so
        # doing it here means trader._ordered_legs, _v2_limit_price,
        # _buy_max_cost_cents, _rollback_floor_cents and the prod trade log all
        # pick up the marginal price with no changes of their own.
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
    a same-title pair is a near-arbitrage (identical questions must co-resolve, so
    one leg pays whenever they do) while a time-series pair is a directional bet
    on the market overstating the in-between probability — the same preference
    main._dedup_pairs applies when both scanners find the same ticker pair.

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
        # Fee-inclusive, matching the per-trade lines main._print_portfolio
        # emits and the figure this loop actually budgets against two lines
        # above. It summed total_cost, so the headline portfolio number was
        # the one cost on the page that was NOT the cash being committed —
        # a live prod dry run showed $60.47 here against $64.39 of per-trade
        # costs and a $52.08 collateral transfer (TS-12).
        "Portfolio: %d trades selected, total cost $%.2f incl. fees",
        len(selected),
        sum(s.total_cost_with_fees for s in selected),
    )
    return selected
