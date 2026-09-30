"""
File: strategy.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Decides how many contracts to buy for each candidate pair, and which
    trades to take. Each trade is sized with the Kelly rule (a formula that
    turns a trade's edge and its chance of paying into the share of the money
    to bet). That share is taken of the portfolio value, meaning cash plus
    what the open positions are worth, but only cash buys contracts, so no
    trade's budget is more than the cash on hand. select_portfolio then picks
    trades best-first and shrinks any trade that no longer fits the cash left.

Dependencies:
    Imports config (fees, budget and size-cap rules, the chance-of-profit
    model, LiveSettings, held_pair_fraction, which sizes a trade that adds to
    a pair the account holds, and count_text for the "adds to N held"
    marker) and scanner (CandidatePair, pair_held and order-book pricing
    helpers). main calls compute_trade and select_portfolio; trader and
    reporter read TradeSpec.

Notes:
    Prices here are leg prices (scanner.leg_prices): (nA, pB) same-title, (pA, nB) time-series.
    cost_with_fees_a/_b are per market, not per side; trader funds each market's shard from them.
    cash_need_cents is never below what shard funding asks for, so the cash set aside covers it in total.
    backtester and dashboard copy the Kelly formula, and backtester the ticker and ladder rules.
    Change every copy together.
    k (the time-series model's discount) and the size caps come from the LiveSettings
    each call is handed, or config.py's when none is.
    An add-on is a candidate that buys more of a pair the account already holds
    (CandidatePair.held, read through scanner.pair_held). Kelly sizes the whole
    position: the add-on takes only what the held pair's stake (its worth at
    today's prices plus the fees paid for it) is missing of its Kelly share of
    the portfolio value, and never more than a new pair would
    (config.held_pair_fraction).
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
    count_text,
    fee_leg_exact,
    fee_per_pair_approx,
    held_pair_fraction,
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
    pair_held,
    pair_ladder_keys,
    prefix_fill_prices,
    v2_effective_cap,
)


@dataclass
class TradeSpec:
    """
    One sized trade for a candidate pair, ready to send to the exchange.

    Attributes:
        pair (CandidatePair): The candidate, re-priced at the chosen size when it has a book.
        x (int): Contracts bought on market_a. Always equals y.
        y (int): Contracts bought on market_b.
        total_cost (float): What the contracts cost in dollars, fees left out.
        total_cost_with_fees (float): total_cost plus both legs' fees, at the fill prices.
        min_payoff (float): Profit if the trade wins, after fees; above 0 on every spec returned.
        profit_ratio (float): Win profit over contract cost; used to rank trades, not to size them.
        days_to_close (int): Days until the later market closes, at least 1.
        monthly_profit_ratio (float): profit_ratio scaled to 30 days; the ranking key.
        kelly_p (float): The model's chance the trade pays, in (0, 1].
        kelly_fraction (float): The share of the portfolio value Kelly calls for, after the
            size cap; for an add-on, reduced by config.held_pair_fraction to what the whole
            position is missing. The trade may spend less when cash is short.
        cost_with_fees_a (float): market_a's leg cost with its fee; trader funds that market's shard from it.
        cost_with_fees_b (float): market_b's leg cost with its fee. Both default to 0.0.
        cash_need_cents (int | None): Most cash the two orders can take, in whole cents; None if built by hand.
        interval_discount (float | None): The k (time-series discount) the spec was sized at; None for same-title.
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


def _kelly_p_at(pair: CandidatePair, yes_leg_price: float, k: float | None) -> float:
    """
    The pair's chance of profit, priced at a given YES-leg price.

    Used when sizing a given count, since the YES leg's price changes with the
    count. A same-title pair ignores the price and k and gets the fixed prior.

    Args:
        pair (CandidatePair): Supplies pair_type and pB.
        yes_leg_price (float): pA at the size being tested, in (0, 1).
        k (float | None): The run's time-series discount; None reads config's value, so only same-title may pass None.

    Returns:
        float: The chance of profit, in (0, 1].
    """
    if pair.pair_type == "time_series":
        # Single shared definition of the time-series model — backtester and
        # dashboard call the same helper so the three sizers cannot drift
        return time_series_profit_prob(yes_leg_price, pair.pB, k=k)
    return SAME_TITLE_CO_RESOLVE_PROB


class _Sizing(NamedTuple):
    """
    One contract count, priced and checked at its own fill price.

    Attributes:
        n (int): The count this was checked at; 0 when the pair has no book.
        target (int): How many the budget buys at that price, capped at the book's depth.
        price_a (float): market_a's leg price at n, in dollars.
        price_b (float): market_b's leg price at n, in dollars.
        p (float): The chance of profit at that price.
        profit_ratio (float): Win profit over contract cost; for ranking, not sizing.
        kelly_fraction (float): Kelly fraction after the size cap; for an add-on, after
            config.held_pair_fraction too.
        budget_dollars (float): What the trade may spend: its share of the portfolio value, at most the cash.
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
    refusal_out: dict | None = None,
) -> _Sizing | None:
    """
    Price n contract pairs off the book and return how many the budget then buys.

    Every sizing check runs here, at n's own price. With no book, the pair's
    stored leg prices are used and n is ignored. For an add-on
    (scanner.pair_held), Kelly sizes the whole position: the capped fraction
    becomes config.held_pair_fraction's, what the held pair's stake (its
    worth at today's prices plus the fees paid for it, HeldPair.stake_dollars)
    is missing of that share of the portfolio value.

    Args:
        pair (CandidatePair): The pair being sized.
        levels (tuple): The pair's book levels; () means use the stored prices.
        n (int): Contract pairs to price; ignored when levels is empty.
        portfolio_value_cents (int): The value the Kelly share is taken of, in cents.
        settings (LiveSettings): The run's settings (k and the size caps).
        cash_cents (int | None): Keyword-only. The cash on hand in cents; None means it is all cash.
        refusal_out (dict | None): Keyword-only. When given, its "holds_kelly_share" key is
            set True if an add-on's held pair already holds its Kelly share at this price
            (the refusal compute_trade's log line names); nothing else sets it.

    Returns:
        _Sizing | None: The result, or None if any check fails (such as no edge after fees,
            too little budget or depth, or an add-on whose held pair already holds its
            Kelly share).
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
    held = pair_held(pair)
    if held is not None:
        # An add-on: Kelly sizes the whole position, so this trade takes only
        # what the held pair's stake (its worth at today's prices plus the
        # fees paid for it) is missing of that share of the portfolio value,
        # and never more than a new pair would (the one definition). The
        # budget below then also keeps it within the cash.
        kelly_fraction_capped = held_pair_fraction(
            kelly_fraction_capped, held.stake_dollars, portfolio_value_cents / 100.0)
        if kelly_fraction_capped <= 0:
            # The held pair already holds its Kelly share at this price
            if refusal_out is not None:
                refusal_out["holds_kelly_share"] = True
            return None

    cash = None if cash_cents is None else cash_cents / 100.0
    # The trade's budget: its share of the portfolio value, at most the cash
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
    Find the largest contract count whose own fill price still justifies buying it.

    Searches 1 to pair.max_contracts by repeatedly halving the range. It can
    land low, but every count it returns has been checked by _evaluate_size.

    Args:
        pair (CandidatePair): The pair being sized; max_contracts bounds the search.
        levels (tuple): The pair's book levels (not empty).
        portfolio_value_cents (int): The value the Kelly share is taken of, in cents.
        settings (LiveSettings): The run's settings.
        cash_cents (int | None): Keyword-only. The cash on hand in cents; None means it is all cash.

    Returns:
        _Sizing | None: The largest checked count, or None if no count works.
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


def _holds_kelly_share(
    pair: CandidatePair, levels: tuple, portfolio_value_cents: int, settings: LiveSettings,
    *, cash_cents: int | None,
) -> bool:
    """
    Say whether an add-on is refused because its held pair already holds its Kelly share.

    Prices the smallest size (one contract pair off the book, or the pair's
    stored prices with no book) through _evaluate_size, the one set of
    checks, and reports whether config.held_pair_fraction is what refused it
    there. A book search that refuses every size ends on that smallest size,
    so this names the reason it met last. An ordinary pair answers False
    without pricing anything.

    Args:
        pair (CandidatePair): The pair compute_trade could not size.
        levels (tuple): The pair's book levels; () means use the stored prices.
        portfolio_value_cents (int): The value the Kelly share is taken of, in cents.
        settings (LiveSettings): The run's settings (k and the size caps).
        cash_cents (int | None): Keyword-only. The cash on hand in cents, as compute_trade
            got it; None means it is all cash.

    Returns:
        bool: True only for an add-on whose held pair's stake (its worth at today's prices
            plus the fees paid for it) is at least its Kelly share of the portfolio value
            at the smallest size's price.
    """
    if pair_held(pair) is None:
        return False
    refusal: dict = {}
    _evaluate_size(pair, levels, 1 if levels else 0, portfolio_value_cents, settings,
                   cash_cents=cash_cents, refusal_out=refusal)
    return refusal.get("holds_kelly_share", False)


def _log_no_add_on(pair: CandidatePair, holds_kelly_share: bool, *,
                   portfolio_value_cents: int) -> None:
    """
    Log why compute_trade adds nothing to a held pair; an ordinary pair logs nothing.

    An add-on to a held pair (scanner.pair_held) that sizes to nothing gets
    one INFO line, so a run that found a held pair and added nothing says
    why. compute_trade calls this once, at the return that refuses the pair;
    the sizes its search tries on the way are never logged. The Kelly-share
    line names the two parts of the pair's stake, its worth at today's
    prices and the fees paid for it, beside the portfolio value.

    Args:
        pair (CandidatePair): The pair compute_trade was asked to size.
        holds_kelly_share (bool): True when the held pair already holds its
            Kelly share (_holds_kelly_share); any other refusal (not
            tradeable, a book no order can reach, fees that eat the budget, no
            profit after exact fees, no cash) is "no size fits this run".
        portfolio_value_cents (int): Keyword-only. The value compute_trade sized on, in
            cents; the Kelly-share line names it.
    """
    held = pair_held(pair)
    if held is None:
        return
    if holds_kelly_share:
        logging.info(
            "Not adding to held pair '%s': it already holds its Kelly share "
            "(%g contracts each, worth $%.2f at today's prices, fees paid $%.2f, "
            "portfolio value $%.2f)",
            pair.canonical_title, held.count, held.value_dollars, held.fees_dollars,
            portfolio_value_cents / 100,
        )
    else:
        logging.info("Not adding to held pair '%s': no size fits this run",
                     pair.canonical_title)


def _order_cash_cents(pair: CandidatePair, n: int, price_a: float, price_b: float) -> int:
    """
    Return the most cash, in whole cents, a pair's two orders can take.

    Each leg's order may pay up to its limit price (the highest price the order
    will pay) on every contract, plus the exchange fee. Per leg this takes n
    times the higher of the limit and the fill price, plus the largest fee at
    any price between the two, rounded up to the cent as shard funding rounds
    it. So it is never less than what trader asks the shards to hold for these legs.
    The reserve is checked against the account's total cash, not each shard's.

    Args:
        pair (CandidatePair): Supplies which side each leg buys and each market's price grid.
        n (int): Contract pairs; each leg buys n. At least 1.
        price_a (float): market_a's leg price, in dollars.
        price_b (float): market_b's leg price, in dollars.

    Returns:
        int: Whole cents both orders can take together.
    """
    total = 0
    # Each leg: the side it buys, its fill price and its market
    for side, price, market in zip(leg_sides(pair.pair_type), (price_a, price_b),
                                   (pair.market_a, pair.market_b), strict=True):
        # The leg's limit price, from the same function the order itself uses
        cap = float(v2_effective_cap(f"buy_{side}", price, market))
        low, high = min(cap, price), max(cap, price)
        # The fee at the price between the two nearest 50c, where the fee is largest
        fee = fee_leg_exact(n, min(max(0.5, low), high))
        # Rounded up to the cent, as shard funding rounds it
        total += leg_cash_cents(n * high + fee)
    return total


def _priced_pair(pair: CandidatePair, n: int, price_a: float, price_b: float) -> CandidatePair:
    """
    Return a copy of a pair with its leg prices set for n contract pairs.

    Anything that later reads the pair's leg prices (the trader's order
    prices, the trade log) gets the price of the n actually sent.
    max_contracts becomes n; the book and the other quotes are kept.

    Args:
        pair (CandidatePair): A pair with a book.
        n (int): The contract pairs the prices are for.
        price_a (float): market_a's leg price at n, in dollars.
        price_b (float): market_b's leg price at n, in dollars.

    Returns:
        CandidatePair: The re-priced copy; the input is unchanged.
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
    Size a candidate pair into a TradeSpec, or return None if it is not worth trading.

    With a book, the count and its price are found together (_solve_marginal_size);
    without one, the pair's stored prices are used. The count is then lowered
    until the cost plus fees fits the trade's budget: its Kelly share of the
    portfolio value, never more than the cash on hand. With a book, a lowered
    count is priced again at its own fill and lowered again until it fits at
    those prices. Fees count as money at risk when sizing. Pass the same value, cash and settings enrichment got, or
    enrichment's depth limit no longer bounds the size.

    An add-on to a held pair (pair.held, read through scanner.pair_held) is
    sized on its whole position: the capped Kelly fraction becomes
    config.held_pair_fraction's, what the held pair's stake (its worth at
    today's prices plus the fees paid for it) is missing of that share of the
    portfolio value, never more than a new pair would get. When compute_trade
    returns None for an add-on it logs one INFO line saying why.

    Args:
        pair (CandidatePair): Must be tradeable; max_contracts 0 means no depth limit.
        portfolio_value_cents (int): Cash plus open positions' value, in cents (the cash alone if nothing is held).
        settings (LiveSettings | None): Keyword-only. The run's settings; None reads config.py's (tests only).
        cash_cents (int | None): Keyword-only. The cash on hand in cents; None means it is all cash.

    Returns:
        TradeSpec | None: The sized trade, or None if the pair is not tradeable, no size is worth
            buying, or it is an add-on whose held pair already holds its Kelly share.

    Raises:
        AttributeError/TypeError: If a market's close_time is None (only a hand-built pair can have one).
        ValueError: If settings is None and a config.py setting is invalid.
    """
    # Resolved once, so every size this call evaluates reads one k and cap
    settings = live_settings() if settings is None else settings
    if not pair.tradeable:
        _log_no_add_on(pair, holds_kelly_share=False,
                       portfolio_value_cents=portfolio_value_cents)
        return None

    # Book levels from enrichment; () means size on the scalar leg prices
    levels = _depth_levels(pair)

    if levels:
        sized = _solve_marginal_size(pair, levels, portfolio_value_cents, settings,
                                     cash_cents=cash_cents)
        if sized is None:
            # For an add-on, say whether its Kelly share or something else refused it
            _log_no_add_on(pair, _holds_kelly_share(pair, levels, portfolio_value_cents,
                                                    settings, cash_cents=cash_cents),
                           portfolio_value_cents=portfolio_value_cents)
            return None
        # The size whose OWN fill price justifies it, and that price
        n = sized.n
    else:
        sized = _evaluate_size(pair, levels, 0, portfolio_value_cents, settings,
                               cash_cents=cash_cents)
        if sized is None:
            # For an add-on, say whether its Kelly share or something else refused it
            _log_no_add_on(pair, _holds_kelly_share(pair, levels, portfolio_value_cents,
                                                    settings, cash_cents=cash_cents),
                           portfolio_value_cents=portfolio_value_cents)
            return None
        # No book (e.g. the bare pair the backtester's Kelly-parity test builds):
        # the single-shot sizing, which must stay unchanged
        n = sized.target

    price_a, price_b = sized.price_a, sized.price_b
    p                = sized.p
    profit_ratio     = sized.profit_ratio
    kelly_fraction_capped = sized.kelly_fraction
    budget_dollars   = sized.budget_dollars

    # The count the current leg prices are the fill for (unused with no book)
    priced_n = n
    while True:
        # budget_dollars covers contracts only; shrink n until the fees fit too
        fee_a = fee_leg_exact(n, price_a)
        fee_b = fee_leg_exact(n, price_b)
        while n > 0 and n * (price_a + price_b) + fee_a + fee_b > budget_dollars:
            n -= 1
            fee_a = fee_leg_exact(n, price_a)
            fee_b = fee_leg_exact(n, price_b)
        if n < 1:
            # Fees ate the entire Kelly budget — no contract count fits
            _log_no_add_on(pair, holds_kelly_share=False,
                           portfolio_value_cents=portfolio_value_cents)
            return None
        if not levels or n == priced_n:
            # The cost at the prices n is actually filled at fits the budget
            break

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
            _log_no_add_on(pair, holds_kelly_share=False,
                           portfolio_value_cents=portfolio_value_cents)
            return None
        # A cheaper leg can cost a cent more in exact fee (p(1 - p) grows
        # toward 0.5), so the re-priced cost can overrun the budget again:
        # shrink once more at these prices. n only falls, so this ends.
        priced_n = n

    # Exact-fee win payoff; ceiling rounding can erase it at small n
    min_payoff = n * (1.0 - price_a - price_b) - fee_a - fee_b
    if min_payoff <= 0:
        _log_no_add_on(pair, holds_kelly_share=False,
                       portfolio_value_cents=portfolio_value_cents)
        return None

    total_cost = n * (price_a + price_b)
    # Fees are cash out the door at execution — the portfolio budget must cover them
    total_cost_with_fees = total_cost + fee_a + fee_b

    # Per-MARKET costs, for trader.ensure_shard_collateral's per-shard funding
    cost_with_fees_a = n * price_a + fee_a
    cost_with_fees_b = n * price_b + fee_b
    # The most cash the two orders can take; select_portfolio sets this aside
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
    # The held pair this trade adds to, if any; the _priced_pair copy below
    # keeps it, so it reaches spec.pair
    held = pair_held(pair)
    logging.info(
        "Trade computed: %s [%s] | %s(A)@%.2f + %s(B)@%.2f | p=%.2f kelly=%.1f%% n=%d "
        "cost=$%.2f profit_ratio=%.2f%% monthly=%.2f%%%s",
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
        # Only an add-on's line gains this, so every other line reads as before
        f" | adds to {count_text(held.count)} held" if held is not None else "",
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
        # Only time-series pricing uses k
        interval_discount=(settings.interval_discount
                           if pair.pair_type == "time_series" else None),
    )


def _cash_need(spec: TradeSpec) -> int:
    """
    Return the whole cents select_portfolio takes from the cash for one spec.

    Uses the spec's cash_need_cents. A spec built by hand without it falls
    back to its cost with fees, rounded up to the cent both as a whole and
    leg by leg, whichever is more.

    Args:
        spec (TradeSpec): The spec to price.

    Returns:
        int: Whole cents, 0 or more.
    """
    need = spec.cash_need_cents
    # A real int only (Python counts True and False as ints)
    if isinstance(need, int) and not isinstance(need, bool):
        return need
    # Round up as shard funding does, so this never takes less than it asks for
    return max(leg_cash_cents(spec.total_cost_with_fees),
               leg_cash_cents(spec.cost_with_fees_a) + leg_cash_cents(spec.cost_with_fees_b))


def _spec_at_count(spec: TradeSpec, n: int) -> TradeSpec | None:
    """
    Return the same trade at n contract pairs, priced at n's own fill price.

    Costs, fees and the win payoff are worked out again for n; the ranking
    figures (kelly_p, kelly_fraction, the profit ratios, days_to_close) keep
    the spec's values. The Kelly, spread and edge checks are not repeated,
    since buying fewer contracts never raises the price per contract. What can
    fail at a smaller n is checked: the book must hold n, each order must reach
    n at its own limit price, a win must still pay after fees, and the expected
    profit must stay positive with the real, rounded-up fees.

    Args:
        spec (TradeSpec): A spec from compute_trade.
        n (int): The contract pairs wanted. At least 1.

    Returns:
        TradeSpec | None: A new spec at n, or None if a check fails or a time-series spec has no interval_discount.
    """
    pair = spec.pair
    k = spec.interval_discount
    if pair.pair_type == "time_series" and (
            not isinstance(k, (int, float)) or isinstance(k, bool)):
        # Without its k, a time-series spec cannot be re-priced
        return None
    levels = _depth_levels(pair)
    if levels:
        # Each leg's average price over the first n contracts of the book
        fills = prefix_fill_prices(levels, n)
        if fills is None:
            # The book holds fewer than n contracts
            return None
        price_a, price_b = fills
        # Each order only buys contracts priced at or under its limit
        if _reachable_contracts(pair, levels, price_a, price_b) < n:
            return None
    else:
        # No book: use the pair's stored leg prices
        price_a, price_b = leg_prices(pair)

    # Each leg's fee, rounded up to the cent
    fee_a = fee_leg_exact(n, price_a)
    fee_b = fee_leg_exact(n, price_b)
    # Win profit after fees; the tiny tolerance treats float noise above 0 as 0
    min_payoff = n * (1.0 - price_a - price_b) - fee_a - fee_b
    if min_payoff <= PRICE_EPSILON:
        return None
    total_cost = n * (price_a + price_b)
    total_cost_with_fees = total_cost + fee_a + fee_b
    # The chance of profit at n's own YES price
    p = _kelly_p_at(pair, price_a, k)
    if p * min_payoff <= (1.0 - p) * total_cost_with_fees:
        # Expected profit is not positive with the real, rounded-up fees
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
    Return the largest smaller version of a spec whose orders fit the cash left.

    Tries each count downward from the most the cash could buy at the book's
    cheapest price, and returns the first one that _spec_at_count accepts and
    whose cash need fits. Each count is checked on its own, since a smaller
    count can fail where a larger one passed.

    Args:
        spec (TradeSpec): A spec that needs more cash than is left.
        available_cents (int): The cash left, in whole cents.

    Returns:
        TradeSpec | None: The spec at the largest count below spec.x that fits, or None if none does.
    """
    if spec.x <= 1:
        # Nothing smaller to try
        return None
    levels = _depth_levels(spec.pair)
    # Cheapest price of one contract pair: the book's best level, or the stored prices
    cheapest = (levels[0][0] + levels[0][1]) if levels else sum(leg_prices(spec.pair))
    if cheapest <= 0:
        return None
    # More than this would cost more than the cash even at the cheapest price
    upper = min(spec.x - 1, int((available_cents / 100.0) / cheapest))
    for n in range(upper, 0, -1):
        shrunk = _spec_at_count(spec, n)
        if shrunk is not None and shrunk.cash_need_cents <= available_cents:
            return shrunk
    return None


def select_portfolio(specs: list, cash_cents: int, *,
                     held_ladders: frozenset = frozenset()) -> list:
    """
    Pick trades best-first without spending more than the cash.

    Specs are taken in order of monthly return (same-title first on ties). A
    spec is skipped if it reuses a ticker already picked, or, for a
    time-series spec, if it is on a ladder (one question asked at several
    deadlines) the account holds or this run already picked. Each spec takes
    its cash_need_cents from the cash left; one that no longer fits is shrunk
    to the largest size that does, or skipped if not even one contract pair
    fits, and the walk goes on. The backtester copies the ticker and ladder
    rules; change both together.

    Markets the account holds are kept out upstream (main._run_prod),
    except the two markets of a held pair the run adds to. Such an add-on
    (its pair.held, read through scanner.pair_held, names the spec's own two
    tickers and the side bought on each: HeldPair.matches) is blocked only by
    the ladders of specs picked earlier in this run: the held ladders it
    meets are its own, since scanner.held_pairs lets a run add only to a pair
    no other held market shares a ladder with. A shrunk add-on is still an
    add-on: it keeps its held pair and claims its tickers and ladders like
    any pick.

    Args:
        specs (list): TradeSpecs from compute_trade.
        cash_cents (int): The cash to spend, in whole cents, all shards together.
        held_ladders (frozenset): Keyword-only. Ladder labels of the markets we hold; empty for none.

    Returns:
        list: The picked specs in ranking order, a shrunk spec in place of the original; may be empty.
    """
    # Cash left to spend, in whole cents
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
    # Ladders we already hold, plus those of the specs picked earlier in this
    # run; an add-on to a held pair is blocked by the picks alone (the held
    # ladders it meets are the ones it adds to)
    used_ladders: set = set(held_ladders)
    picked_ladders: set = set()
    ladder_skips = 0
    shrinks = 0
    for spec in specs_sorted:
        ta = spec.pair.market_a.ticker
        tb = spec.pair.market_b.ticker
        # Skip trades that would re-use a ticker already committed to a higher-priority pair
        if ta in used_tickers or tb in used_tickers:
            continue
        keys = pair_ladder_keys(spec.pair)
        # At most one open time-series trade per ladder; an add-on meets only
        # this run's earlier picks, and only when the held pair it carries is
        # the spec's own two markets, each bought on its held side
        held = pair_held(spec.pair)
        is_add_on = held is not None and held.matches(
            spec.pair.market_a, spec.pair.market_b, spec.pair.pair_type)
        blocking = picked_ladders if is_add_on else used_ladders
        if spec.pair.pair_type == "time_series" and keys & blocking:
            ladder_skips += 1
            continue
        need = _cash_need(spec)
        if need > available:
            # Too big for the cash left: shrink it, or skip it if nothing fits
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
        picked_ladders |= keys
    if ladder_skips:
        logging.info(
            "Time-series trades skipped because the account already holds, or this "
            "run already picked, a trade on the same ladder: %d",
            ladder_skips,
        )
    if shrinks:
        logging.info("Trades shrunk to fit the cash left: %d", shrinks)
    logging.info(
        # Cost with fees at the fill prices, and the cash set aside at the limit prices
        "Portfolio: %d trades selected, total cost $%.2f incl. fees "
        "(up to $%.2f of cash at the orders' limit prices)",
        len(selected),
        sum(s.total_cost_with_fees for s in selected),
        (cash_cents - available) / 100,
    )
    return selected
