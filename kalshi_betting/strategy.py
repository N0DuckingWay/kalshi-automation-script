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
from datetime import UTC, datetime

from .config import (
    BUDGET_FRACTION,
    SAME_TITLE_CO_RESOLVE_PROB,
    fee_leg_exact,
    fee_per_pair_approx,
    time_series_profit_prob,
)
from .scanner import CandidatePair, leg_prices, leg_sides


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
        profit_ratio (float): Return on cost in a win scenario, net of the
            continuous fee approximation:
            ((1 - price_a - price_b) - fee_per_pair_approx(price_a, price_b)) /
            (price_a + price_b). This is "b" in the Kelly formula below (see
            compute_trade()'s net_spread/profit_ratio computation).
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

    Inputs after enrichment: pair.pA is the depth-weighted YES fill on A
    (enrich_with_orderbook_prices writes the time-series leg prices back to
    pA/nB) while pair.pB is still the scan-time best YES ask on B. A worse YES
    fill therefore raises pA, which shrinks the modelled gap (raising p) and
    shrinks the edge (raising the cost) together — a second-order effect that
    moves in the conservative direction on the sizing that matters. The gap is
    kept on YES asks (pB - pA) rather than on the executable spread because
    the YES-ask gap is the smaller, more conservative estimate of the mass the
    market assigns to the loss cell.

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
    if pair.pair_type == "time_series":
        # Single shared definition of the time-series model — backtester and
        # dashboard call the same helper so the three sizers cannot drift
        return time_series_profit_prob(pair.pA, pair.pB)
    return SAME_TITLE_CO_RESOLVE_PROB


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
        b = net_spread / (price_a + price_b)   [net profit per dollar risked]
        f* = p - (1-p)/b                        [optimal Kelly fraction]
        f_capped = min(BUDGET_FRACTION, f*)

    Where net_spread = (1 − price_a − price_b) − fee_per_pair_approx(price_a, price_b)
    and p comes from _kelly_p (the co-resolution prior for same_title; the
    discounted in-between model for time_series). For a time-series pair the
    result is a directional bet: f* is only positive when the modelled loss
    probability k * (pB - pA) is small enough relative to b, and a wide book
    (large price_a + price_b for the same YES-ask gap) drives it negative.

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

    # The two prices the legs actually cost — which of the pair's four quotes
    # they are depends on the pair type; leg_prices is the single source of truth
    price_a, price_b = leg_prices(pair)

    # Validate that both leg prices are in the open interval (0, 1).
    # Edge cases at 0 or 1 indicate a settled market and would break the fee formula.
    if price_b <= 0.0 or price_b >= 1.0 or price_a <= 0.0 or price_a >= 1.0:
        return None

    # Subtract the continuous fee approximation from the gross spread to get the
    # net edge. A zero or negative net_spread means the trade costs more than it pays.
    net_spread = (1.0 - price_a - price_b) - fee_per_pair_approx(price_a, price_b)
    if net_spread <= 0:
        return None

    # profit_ratio is the net return per dollar invested — this is "b" in the Kelly formula
    profit_ratio = net_spread / (price_a + price_b)

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
    n = int(budget_dollars / (price_a + price_b))
    if n < 1:
        # The Kelly budget can't afford even one contract pair — forcing n=1
        # would silently exceed both the Kelly fraction and BUDGET_FRACTION
        return None

    # Respect the order book depth limit set by scanner.enrich_with_orderbook_prices()
    if pair.max_contracts > 0:
        n = min(n, pair.max_contracts)

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
    # at nB) is never misread as the same-title NO/YES layout
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
        "Portfolio: %d trades selected, total cost $%.2f",
        len(selected),
        sum(s.total_cost for s in selected),
    )
    return selected
