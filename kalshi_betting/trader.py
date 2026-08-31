"""
File: trader.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Converts TradeSpec objects (produced by strategy.py) into Kalshi REST API
    order requests and submits them. Each pair's two legs are submitted sequentially
    with fill_or_kill semantics: leg A (NO on market A) first, then leg B (YES on
    market B) only if leg A filled. Both buy legs are price-protected against a
    book that moved since the pre-execution check, so such an order is killed
    instead of filling at a loss. If leg B fails, a rollback order is immediately
    submitted to unwind leg A, and the rollback's own fill status is verified —
    an unfilled rollback is reported as status="rollback_failed" (orphaned
    position, manual review). Multiple pairs are executed concurrently via
    ThreadPoolExecutor so no pair waits for another to complete.

    Two order paths exist, selected by config.ORDER_API_VERSION:
      "v2" (default) — POST config.V2_ORDER_PATH (/portfolio/events/orders) with
        a JSON body: bid/ask side on the single YES book, a dollar-string
        fill_or_kill LIMIT price, a fixed-point count, and an explicit
        exchange_index taken from THAT LEG'S OWN market, so each order routes to
        the shard its market actually lives on. There is no "market" order type
        in V2, so a taker order IS a marketable FoK limit and the LIMIT PRICE IS
        THE PRICE PROTECTION: scanned price ceiled to the market's own tick grid
        plus config.BUY_SLIPPAGE_TICKS ticks (see _v2_limit_price).
      "legacy" — the original /portfolio/orders create-order call
        (CreateOrderRequest, type="market", integer-cents buy_max_cost). Kept
        fully intact and unmodified so flipping ORDER_API_VERSION back to
        "legacy" is an instant, code-free rollback if the V2 mapping misbehaves
        on its first real submission. That endpoint has NO shard-routing
        parameter, so while it is selected a pair with a leg off
        config.DEFAULT_EXCHANGE_INDEX is refused outright (see _legacy_routable).

    An exception from either submission path does NOT prove the order was
    rejected (a timeout can land after the fill), so exception paths consult the
    actual account position for the ticker before classifying the outcome.

    pre_execution_check() re-fetches order books for each spec in the portfolio
    concurrently and drops any whose prices have moved since the scan, reducing
    the chance of submitting orders against a stale price.

Dependencies:
    Imports TradeResult from reporter.py and TradeSpec from strategy.py. Imports
    CreateOrderRequest from the kalshi_python_sync SDK, and fetch_json_page plus
    signed_request_json from _http.py. Imports validate_pair_price and
    tick_size_for_price from scanner.py, and ORDER_API_VERSION,
    BUY_SLIPPAGE_TICKS, BUY_MAX_COST_SLIPPAGE_CENTS, DEFAULT_EXCHANGE_INDEX,
    V2_ORDER_PATH and V2_ROLLBACK_BID_PRICE_DOLLARS from config.py. Called by
    main.py after select_portfolio() selects the final trade list. Depends on
    the KalshiClient produced by auth.py.

Notes:
    Do NOT add retry logic to order submission, on EITHER path. A failed leg
    indicates the market moved between scan time and execution time — retrying
    risks buying one leg at a worse price and creating an unhedged directional
    position. _submit_order calls fetch_json_page directly and _submit_order_v2
    calls signed_request_json directly; neither may ever be wrapped in
    api_call_with_retry.

    Shard routing is per LEG, not per bot. Kalshi partitioned the exchange into
    shards and every market carries its own exchange_index; each V2 order body
    therefore takes that field from its own leg's market rather than assuming a
    single shard, and never sends the -1 auto-route sentinel. The legacy
    endpoint cannot express any of that, so while ORDER_API_VERSION is not "v2"
    a spec with a leg off config.DEFAULT_EXCHANGE_INDEX is refused before
    anything is submitted (_legacy_routable, consulted at the top of
    _execute_one) rather than misrouted.

    Order submission and position lookups use the SDK's raw-response variants
    (via _submit_order / _position_count) because 2026-07 API drift broke the
    pinned SDK's Order and MarketPosition response models — the modeled calls
    raise ValidationError AFTER submission, misclassifying real fills. The
    legacy request side still uses the modeled CreateOrderRequest and serializes
    through the same SDK code path, so its wire format is unchanged. The V2 path
    has no SDK method at all, so it hand-builds the JSON body and signs the
    request through _http.signed_request_json.

    All V2 price math is done in Decimal, never float: the endpoint takes
    dollar-string prices, and binary float noise would produce a string the
    exchange rejects as off-grid.
"""
import logging
import math
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Any

from kalshi_python_sync.models import CreateOrderRequest

from ._http import fetch_json_page, signed_request_json
from .config import (
    BUY_MAX_COST_SLIPPAGE_CENTS,
    BUY_SLIPPAGE_TICKS,
    DEFAULT_EXCHANGE_INDEX,
    ORDER_API_VERSION,
    V2_ORDER_PATH,
    V2_ROLLBACK_BID_PRICE_DOLLARS,
)
from .reporter import TradeResult
from .scanner import tick_size_for_price, validate_pair_price
from .strategy import TradeSpec

# Side each order leg takes on Kalshi's SINGLE YES order book, where every
# order is quoted in YES terms:
#   buying YES        = bidding for YES at the YES price;
#   buying NO         = ASKING (selling YES you do not hold) at 1 - the NO
#                       price — a short YES position IS a long NO position;
#   closing a held NO = buying that YES short back, i.e. a reduce-only BID.
# Kept as one table so the mapping is a SINGLE point of correction: if the
# first live submission shows the exchange interprets a leg the other way
# round, only these three values change — no builder logic moves.
_V2_LEG_SIDE: dict[str, str] = {
    "buy_yes":  "bid",  # buy YES n @ pB  -> bid at capped pB
    "buy_no":   "ask",  # buy NO n @ nA   -> ask at 1 - capped nA
    "close_no": "bid",  # unwind held NO  -> reduce-only bid at an aggressive price
}

# Hard bounds for a V2 limit price, in dollars. Kalshi prices live in the open
# unit interval — 0 and 1 are settlement values, not tradeable levels — and the
# finest grid in any regime is $0.0001, so these are the extreme valid ticks.
_V2_MIN_PRICE = Decimal("0.0001")
_V2_MAX_PRICE = Decimal("0.9999")

# Number of decimal places in a V2 dollar-string price. Four places exactly
# represents every grid point of every known regime ($0.01 / $0.001 / $0.0001),
# so quantizing here can never move a price off-grid.
_V2_PRICE_QUANTUM = Decimal("0.0001")


def _buy_max_cost_cents(count: int, price_dollars: float) -> int:
    """
    Compute the buy_max_cost cap in cents for one buy leg.

    The cap is the scanned depth-weighted price for the full count, rounded up
    to a whole cent, plus BUY_MAX_COST_SLIPPAGE_CENTS per contract of tolerance.
    Kalshi rejects (FoK) any fill that would cost more, so a book that moved
    against us since the pre-execution check cannot fill at a guaranteed loss.

    Args:
        count (int): Number of contracts on this leg. >= 1.
        price_dollars (float): Scanned per-contract price in dollars (0, 1).

    Returns:
        int: Maximum total cost in cents for the order.
    """
    return math.ceil(count * price_dollars * 100) + count * BUY_MAX_COST_SLIPPAGE_CENTS


def _legacy_routable(spec: TradeSpec) -> bool:
    """
    Return True only when BOTH of a pair's legs are on DEFAULT_EXCHANGE_INDEX.

    The legacy /portfolio/orders endpoint has no shard-routing parameter, so
    the only shard it can reach is config.DEFAULT_EXCHANGE_INDEX — an order for
    a market on any other shard would fail or, worse, misroute. Markets carry
    their own exchange_index from ingest, so this is the single place where a
    pair the legacy endpoint cannot reach is turned away, after sizing and
    immediately before anything is submitted.

    Consulted ONLY while config.ORDER_API_VERSION is not "v2". On the V2 path it
    does not apply, because every V2 body routes itself via its own market's
    exchange_index. The legacy path is retained indefinitely as the instant
    rollback, so this guard outlives the migration with it.

    Args:
        spec (TradeSpec): The trade specification about to be executed. Both
            spec.pair.market_a and spec.pair.market_b must carry an int
            exchange_index (tagged at ingest).

    Returns:
        bool: True if both legs can be reached by the legacy order endpoint,
            False if either leg sits on any other shard.
    """
    return (
        spec.pair.market_a.exchange_index == DEFAULT_EXCHANGE_INDEX
        and spec.pair.market_b.exchange_index == DEFAULT_EXCHANGE_INDEX
    )


def _build_no_order(spec: TradeSpec) -> CreateOrderRequest:
    """
    Build a market (taker) order to buy NO contracts on market A.

    The NO leg is the more expensive side of the arbitrage pair — buying NO at
    price nA means we profit when market A resolves NO (or when market B resolves
    YES first). fill_or_kill is used so the order succeeds atomically or fails
    entirely, preventing partial fills at the wrong price. buy_max_cost caps the
    total spend at the scanned price plus a small slippage allowance.

    Args:
        spec (TradeSpec): The computed trade specification. Uses spec.pair.market_a
            for the ticker, spec.pair.nA for the price cap, and spec.x for the
            contract count.

    Returns:
        CreateOrderRequest: Kalshi SDK order request object for the NO leg.
    """
    return CreateOrderRequest(
        ticker=spec.pair.market_a.ticker,
        side="no",
        action="buy",
        # "market" type means we accept the current best ask — we are the taker
        type="market",
        count=spec.x,
        # fill_or_kill: execute the full count immediately or cancel with no fill
        time_in_force="fill_or_kill",
        # Price protection: never pay more than scanned price + slippage allowance
        buy_max_cost=_buy_max_cost_cents(spec.x, spec.pair.nA),
    )


def _build_yes_order(spec: TradeSpec) -> CreateOrderRequest:
    """
    Build a market (taker) order to buy YES contracts on market B.

    The YES leg is the cheaper side of the arbitrage pair — buying YES at price
    pB means we profit when market B resolves YES. Combined with the NO leg on
    market A, all three resolution scenarios (A=YES before B, both YES, both NO)
    yield a positive return. buy_max_cost caps the total spend at the scanned
    price plus a small slippage allowance.

    Args:
        spec (TradeSpec): The computed trade specification. Uses spec.pair.market_b
            for the ticker, spec.pair.pB for the price cap, and spec.y for the
            contract count.

    Returns:
        CreateOrderRequest: Kalshi SDK order request object for the YES leg.
    """
    return CreateOrderRequest(
        ticker=spec.pair.market_b.ticker,
        side="yes",
        action="buy",
        # "market" type means we accept the current best ask — we are the taker
        type="market",
        count=spec.y,
        # fill_or_kill: execute the full count immediately or cancel with no fill
        time_in_force="fill_or_kill",
        # Price protection: never pay more than scanned price + slippage allowance
        buy_max_cost=_buy_max_cost_cents(spec.y, spec.pair.pB),
    )


def _ceil_to_tick(price: Decimal, tick: Decimal) -> Decimal:
    """
    Round a price UP to the next point of a tick grid.

    Ceiling, never nearest or floor: this is the first half of a buy leg's price
    cap, and it mirrors the legacy _buy_max_cost_cents' math.ceil for exactly
    the same reason — a cap rounded BELOW the scanned depth-weighted price could
    never fill at the price we actually scanned, so a fill-or-kill order carrying
    it would be structurally killed every time rather than protected.

    Args:
        price (Decimal): Price in dollars to round. Range: [0, 1].
        tick (Decimal): Tick size in dollars for the grid to land on. Must be
            > 0 (tick_size_for_price guarantees this).

    Returns:
        Decimal: The smallest grid point >= price. Returns price unchanged when
            it already sits exactly on the grid.
    """
    return (price / tick).to_integral_value(rounding=ROUND_CEILING) * tick


def _v2_limit_price(leg: str, scanned_price_dollars: float, market: Any) -> Decimal:
    """
    Compute the fill-or-kill LIMIT price for one V2 buy leg, in dollars.

    V2 has no "market" order type, so a taker order is a marketable FoK limit
    and this price IS the price protection that buy_max_cost provided on the
    legacy path: the order fills at or better than the cap, or not at all.
    The cap is the scanned price ceiled onto the market's own tick grid plus
    BUY_SLIPPAGE_TICKS ticks of tolerance for a book that moved since the
    pre-execution check.

    Because the cap (or, for the NO leg, its complement 1 - cap) can land in a
    DIFFERENT band of the market's grid than the scanned price — stepping up
    across a band edge, or being mirrored to the other end of the book — the
    final price is re-quantized onto the grid of the band that actually
    contains it, rounding UP. Ceiling is the protective direction here too: it
    can only tighten-or-keep the cap by under one destination-band tick, so the
    worst outcome of the re-quantization is a killed FoK (trade skipped), never
    a worse-than-intended fill; a floor could instead round a YES cap below the
    scanned price and make the order structurally unfillable. Kalshi's nested
    grids ($0.01 subset of $0.001 subset of $0.0001) guarantee a price landing
    in a FINER band than it was computed on is already on that band's grid, so
    the snap is then a no-op.

    The clamp bounds are grid-aware for the same reason: the extreme tradeable
    levels are one tick inside 0 and 1 ON THIS MARKET'S GRID (e.g. 0.99, not
    0.9999, on a linear-cent market), so the bounds are derived from the tick
    size at each end of the book rather than the global finest-grid constants.

    This mapping (which side, and the complement for the NO leg) is the single
    assumption most in need of verification at the first live submission; see
    _V2_LEG_SIDE, which is where a correction would be made.

    Args:
        leg (str): Which leg is being priced — "buy_yes" or "buy_no". Any other
            value is treated as a YES-style leg (the price is used as-is).
        scanned_price_dollars (float): The scanned depth-weighted per-contract
            price for this leg, in dollars. Range: (0, 1). For "buy_no" this is
            the NO price (spec.pair.nA), which is complemented into a YES-book
            ask price.
        market (Any): The market object the leg trades, used only to look up its
            tick grid. Any object exposing price_level_structure / price_ranges.

    Returns:
        Decimal: The limit price in dollars — a valid grid point of the band
            containing it, clamped one tick inside the open unit interval on
            this market's grid.
    """
    scanned = Decimal(str(scanned_price_dollars))
    # Cross-module: the market's own tick grid is the only authority on what
    # price levels the exchange will accept for this leg
    tick = tick_size_for_price(market, scanned_price_dollars)
    cap = _ceil_to_tick(scanned, tick) + BUY_SLIPPAGE_TICKS * tick
    # Buying NO is selling YES on the single YES book, so the YES-side price is
    # the complement of the capped NO price
    price = Decimal("1") - cap if leg == "buy_no" else cap
    # Re-quantize onto the grid of the band the FINAL price sits in (see
    # docstring: ceiling is protective — worst case is a killed FoK)
    final_tick = tick_size_for_price(market, float(price))
    price = _ceil_to_tick(price, final_tick)
    # Grid-aware clamp: the extreme valid levels are one tick inside 0 and 1
    # on this market's own grid at each end of the book
    bottom_tick = tick_size_for_price(market, float(_V2_MIN_PRICE))
    top_tick = tick_size_for_price(market, float(Decimal("1") - _V2_MIN_PRICE))
    return min(max(price, bottom_tick), Decimal("1") - top_tick)


def _v2_rollback_price(market: Any) -> Decimal:
    """
    Compute the aggressive limit price for a V2 reduce-only unwind bid.

    The legacy path unwound with a type="market" order carrying no price cap;
    V2 has no market type, so the emulation is a fill-or-kill bid at the
    HIGHEST TRADEABLE LEVEL of this market's own grid — it crosses any resting
    ask the way the market order did. That level depends on the tick regime:
    0.99 on a linear-cent grid, 0.999 on deci-cent, 0.9999 on a
    centi-cent edge band. A flat 0.99 (valid only on the coarsest grid) would
    fail to cross asks resting in (0.99, 1) on sub-cent regimes, turning an
    unwind the legacy market order always filled into a structurally killed
    FoK and an orphaned position.

    config.V2_ROLLBACK_BID_PRICE_DOLLARS is the finest-grid target; it is
    FLOORED onto the grid of the band containing it — floor, not ceiling,
    because rounding up would leave the open unit interval (1 is a settlement
    value, not a tradeable level).

    Args:
        market (Any): The market whose position is being unwound. Any object
            exposing price_level_structure / price_ranges.

    Returns:
        Decimal: The highest valid bid level on this market's grid at the top
            of the book.
    """
    target = Decimal(V2_ROLLBACK_BID_PRICE_DOLLARS)
    tick = tick_size_for_price(market, float(target))
    return (target / tick).to_integral_value(rounding=ROUND_FLOOR) * tick


def _format_price(p: Decimal) -> str:
    """
    Serialize a dollar price as the fixed-point string the V2 endpoint expects.

    Always four decimal places (e.g. "0.5600"), which exactly represents every
    grid point of every known tick regime, so the quantization here can never
    move an on-grid price off-grid. Decimal in, string out — a float never
    touches the wire format.

    Args:
        p (Decimal): Price in dollars. Range: [0, 1].

    Returns:
        str: The price as a 4-decimal fixed-point string.
    """
    return str(p.quantize(_V2_PRICE_QUANTUM))


def _format_count(n: int) -> str:
    """
    Serialize a contract count as the fixed-point string the V2 endpoint expects.

    V2 counts are fixed-point strings (e.g. "10.00") because the endpoint also
    supports fractional contracts. The bot still sizes in whole contracts —
    strategy.compute_trade() returns integer counts — so this always emits a
    ".00" fraction. Fractional sizing is a deliberately deferred follow-up.

    Args:
        n (int): Whole contract count for one leg. >= 1.

    Returns:
        str: The count as a fixed-point string with two decimal places.
    """
    return f"{n}.00"


def _build_no_order_v2(spec: TradeSpec) -> dict:
    """
    Build the V2 request body for the leg-A order that buys NO on market A.

    On the single YES book, buying NO is expressed as an ASK (selling YES we do
    not hold — a short YES position is a long NO position), priced at
    1 - (capped NO price). The order is fill_or_kill so it executes in full or
    not at all, and the limit price is the price protection (see
    _v2_limit_price). This side/price conversion is the assumption most needing
    verification at the first live submission — correct it in _V2_LEG_SIDE.

    Args:
        spec (TradeSpec): The computed trade specification. Uses
            spec.pair.market_a for the ticker, tick grid and exchange shard,
            spec.pair.nA for the price cap, and spec.x for the contract count.

    Returns:
        dict: JSON body for POST config.V2_ORDER_PATH.
    """
    return {
        "ticker": spec.pair.market_a.ticker,
        # Client-generated idempotency key — lets a human match a log line to
        # an order in the account when a submission outcome is ambiguous
        "client_order_id": str(uuid.uuid4()),
        "side": _V2_LEG_SIDE["buy_no"],
        "price": _format_price(_v2_limit_price("buy_no", spec.pair.nA, spec.pair.market_a)),
        "count": _format_count(spec.x),
        # fill_or_kill: execute the full count immediately or cancel with no fill
        "time_in_force": "fill_or_kill",
        # This leg's OWN market's shard, read from the market itself — legs of
        # one pair can live on different shards. Explicit, never the -1
        # auto-route sentinel: if our notion of a market's shard is ever wrong
        # we want the exchange to reject the order loudly rather than silently
        # settle it against a shard we never modelled. An explicit-shard write
        # also bills only that shard's rate-limit bucket, where auto-route bills
        # every nonzero shard's.
        "exchange_index": spec.pair.market_a.exchange_index,
        # Opening exposure, not closing it
        "reduce_only": False,
        # We are deliberately takers — a post-only order would be rejected
        # rather than crossing the book
        "post_only": False,
    }


def _build_yes_order_v2(spec: TradeSpec) -> dict:
    """
    Build the V2 request body for the leg-B order that buys YES on market B.

    Buying YES is a BID on the YES book at the capped YES price — no complement
    is involved, unlike the NO leg. fill_or_kill with the limit price acting as
    the price protection (see _v2_limit_price). The side/price mapping is the
    assumption most needing verification at the first live submission — correct
    it in _V2_LEG_SIDE.

    Args:
        spec (TradeSpec): The computed trade specification. Uses
            spec.pair.market_b for the ticker, tick grid and exchange shard,
            spec.pair.pB for the price cap, and spec.y for the contract count.

    Returns:
        dict: JSON body for POST config.V2_ORDER_PATH.
    """
    return {
        "ticker": spec.pair.market_b.ticker,
        # Client-generated idempotency key — see _build_no_order_v2
        "client_order_id": str(uuid.uuid4()),
        "side": _V2_LEG_SIDE["buy_yes"],
        "price": _format_price(_v2_limit_price("buy_yes", spec.pair.pB, spec.pair.market_b)),
        "count": _format_count(spec.y),
        # fill_or_kill: execute the full count immediately or cancel with no fill
        "time_in_force": "fill_or_kill",
        # Leg B's own market's shard — market_b may sit on a different shard
        # than market_a. Explicit, never -1 auto-route — see _build_no_order_v2
        "exchange_index": spec.pair.market_b.exchange_index,
        "reduce_only": False,
        "post_only": False,
    }


def _build_rollback_order_v2(spec: TradeSpec) -> dict:
    """
    Build the V2 request body that unwinds a filled leg-A NO position.

    A held NO position is a short YES, so closing it is a YES BUY — a BID, not
    a sell. reduce_only guarantees the order can only close existing exposure,
    so it is safe to submit even when leg A's fill state is ambiguous (it cannot
    open a new position). V2 has no "market" type, so a bid at the highest
    tradeable level of this market's own grid (_v2_rollback_price) is what
    emulates the legacy market unwind: it crosses any resting ask, giving the
    fill_or_kill order the same chance of filling the old market order had. The
    bid/reduce-only semantics here are part of the mapping to verify at the
    first live unwind.

    Args:
        spec (TradeSpec): The trade whose leg A must be unwound. Uses
            spec.pair.market_a for the ticker, tick grid and exchange shard,
            and spec.x for the count.

    Returns:
        dict: JSON body for POST config.V2_ORDER_PATH.
    """
    return {
        "ticker": spec.pair.market_a.ticker,
        # Client-generated idempotency key — see _build_no_order_v2
        "client_order_id": str(uuid.uuid4()),
        "side": _V2_LEG_SIDE["close_no"],
        # Highest tradeable level on THIS market's grid — emulates the legacy
        # capless market-order unwind (see _v2_rollback_price)
        "price": _format_price(_v2_rollback_price(spec.pair.market_a)),
        "count": _format_count(spec.x),
        "time_in_force": "fill_or_kill",
        # Market A's own shard — the unwind must route to the same shard the
        # leg-A order opened the position on. Explicit, never -1 auto-route —
        # see _build_no_order_v2
        "exchange_index": spec.pair.market_a.exchange_index,
        # Can only reduce an existing position — never opens exposure even if
        # leg A turns out not to have filled after all
        "reduce_only": True,
        "post_only": False,
    }


def _parse_fixed_point(payload: dict, key: str) -> Decimal | None:
    """
    Read a count field that may arrive as a fixed-point string or a raw number.

    The V2 responses observed in the docs carry counts both ways — `fill_count`
    as an integer and `fill_count_fp` as a fixed-point string (Get Orders shows
    the _fp form) — and which one a given deployment sends is not yet verified
    live. The _fp variant is preferred when present because it is the newer,
    non-truncating representation. Decimal(str(v)) keeps float noise out.

    Args:
        payload (dict): The order object from a V2 response.
        key (str): Base field name, e.g. "fill_count". The f"{key}_fp" variant
            is tried first.

    Returns:
        Decimal | None: The parsed value, or None when neither key is present or
            neither value can be parsed as a number.
    """
    for candidate in (f"{key}_fp", key):
        value = payload.get(candidate)
        if value is None:
            continue
        try:
            return Decimal(str(value))
        except (ArithmeticError, TypeError, ValueError):
            continue
    return None


def _v2_fill_status(data: dict, requested_count: int) -> str:
    """
    Classify a V2 create-order response into the legacy fill-status vocabulary.

    Returns the same strings the legacy path's order.status field carried, so
    _execute_one's branching is untouched: "executed" for a full fill,
    "canceled" for a fill-or-kill that killed with no fill.

    Anything else — a partial fill (which violates the fill-or-kill invariant)
    or a response with no readable fill count at all — is NOT guessed at. It is
    logged at CRITICAL and raised, because raising routes the caller into
    _execute_one's EXISTING ambiguous-exception path, which consults the
    account's actual position (the ground truth) before classifying the trade.
    That means a mispredicted V2 response shape degrades safely into
    failed / rolled_back / manual_review instead of being misread as a clean
    fill or a clean kill. Silently returning a status here is what would be
    dangerous.

    Args:
        data (dict): The parsed V2 response body. The order object may be
            wrapped under an "order" key or sent flat; both are accepted.
        requested_count (int): Contract count the order asked for. >= 1.

    Returns:
        str: "executed" when the full count filled, "canceled" when nothing did.

    Raises:
        ValueError: When the fill count is missing, unparseable, or is a
            partial fill — deliberately routing the caller into the
            position-lookup disambiguation path.
    """
    inner = data.get("order")
    order = inner if isinstance(inner, dict) else data
    fill = _parse_fixed_point(order, "fill_count")
    if fill is not None:
        if fill == requested_count:
            return "executed"
        if fill == 0:
            return "canceled"
    logging.critical(
        "V2 order response could not be classified (fill_count=%s, requested=%d,"
        " response keys=%s, order keys=%s) — treating as an AMBIGUOUS submission"
        " so the account position decides the outcome.",
        fill, requested_count, sorted(data.keys()), sorted(order.keys()),
    )
    raise ValueError(
        f"Unclassifiable V2 order response: fill_count={fill}, requested={requested_count}"
    )


def _submit_order_v2(client: Any, body: dict) -> str:
    """
    Submit one V2 order and return its fill status in the legacy vocabulary.

    The pinned SDK has no method for the V2 create-order route at all, so the
    request is signed and executed through _http.signed_request_json, which
    raises ApiException on non-2xx exactly as the legacy path's fetch_json_page
    does — callers' exception handling is therefore unchanged.

    Deliberately NOT wrapped in api_call_with_retry, for the same reason as the
    legacy _submit_order: retrying a fill-or-kill leg could double-submit it at
    a different price and leave an unhedged position (see module docstring).

    Args:
        client (Any): Authenticated KalshiClient from auth.build_client().
        body (dict): Request body from one of the _build_*_order_v2 builders.

    Returns:
        str: "executed" (full fill) or "canceled" (killed with no fill).

    Raises:
        ApiException: On non-2xx HTTP status.
        ValueError: When the response's fill count is missing or partial —
            callers treat any exception as an ambiguous submission and consult
            the account position.
    """
    requested = int(Decimal(body["count"]))
    # Log before submitting: the client_order_id is the only handle a human has
    # for finding this order in the account if the outcome turns out ambiguous
    logging.info(
        "Submitting V2 order: ticker=%s side=%s price=%s count=%s client_order_id=%s",
        body["ticker"], body["side"], body["price"], body["count"], body["client_order_id"],
    )
    # Retry-free by design (see docstring); signed_request_json contains no
    # retry logic of its own precisely so this call site stays single-shot
    data = signed_request_json(client, "POST", V2_ORDER_PATH, body=body)
    return _v2_fill_status(data, requested)


def _build_no_order_any(spec: TradeSpec) -> Any:
    """
    Build the leg-A (buy NO on market A) order for the configured API version.

    Args:
        spec (TradeSpec): The computed trade specification.

    Returns:
        Any: A V2 request-body dict when config.ORDER_API_VERSION is "v2",
            otherwise a legacy CreateOrderRequest.
    """
    if ORDER_API_VERSION == "v2":
        return _build_no_order_v2(spec)
    return _build_no_order(spec)


def _build_yes_order_any(spec: TradeSpec) -> Any:
    """
    Build the leg-B (buy YES on market B) order for the configured API version.

    Args:
        spec (TradeSpec): The computed trade specification.

    Returns:
        Any: A V2 request-body dict when config.ORDER_API_VERSION is "v2",
            otherwise a legacy CreateOrderRequest.
    """
    if ORDER_API_VERSION == "v2":
        return _build_yes_order_v2(spec)
    return _build_yes_order(spec)


def _build_rollback_order_any(spec: TradeSpec) -> Any:
    """
    Build the leg-A unwind order for the configured API version.

    Args:
        spec (TradeSpec): The trade whose leg A must be unwound.

    Returns:
        Any: A V2 reduce-only bid body when config.ORDER_API_VERSION is "v2",
            otherwise a legacy reduce-only market sell CreateOrderRequest.
    """
    if ORDER_API_VERSION == "v2":
        return _build_rollback_order_v2(spec)
    return CreateOrderRequest(
        ticker=spec.pair.market_a.ticker,
        side="no",
        action="sell",
        type="market",
        count=spec.x,
        time_in_force="fill_or_kill",
        # Can only reduce an existing position — never opens a short even if
        # leg A turns out not to have filled after all
        reduce_only=True,
    )


def _submit_any(client: Any, order: Any) -> str:
    """
    Submit an order built by one of the _*_any builders and return its status.

    Dispatches on the ORDER'S OWN TYPE rather than re-reading
    config.ORDER_API_VERSION, so a build/submit pair can never split across
    versions mid-flight (e.g. if the constant were patched between the two
    calls, a V2 body would still go to the V2 endpoint). Neither branch retries.

    Args:
        client (Any): Authenticated KalshiClient from auth.build_client().
        order (Any): A V2 request-body dict, or a legacy CreateOrderRequest.

    Returns:
        str: The order's fill status — "executed", "canceled", or any other
            status string the legacy endpoint reports.

    Raises:
        ApiException: On non-2xx HTTP status.
        ValueError/KeyError/TypeError: When the response cannot be classified;
            callers treat any exception as an ambiguous submission.
    """
    if isinstance(order, dict):
        return _submit_order_v2(client, order)
    return _submit_order(client, order)


def _position_count(client: Any, ticker: str) -> float | None:
    """
    Fetch the signed contract position for one ticker, or None if the lookup fails.

    Used to disambiguate order-submission exceptions: an exception does not
    prove the order was rejected (a timeout can arrive after the fill), so the
    account's actual position is the ground truth. Kalshi convention: negative
    counts are NO contracts, positive counts are YES contracts.

    Uses the raw-response variant + JSON parsing because the pinned SDK's
    MarketPosition model requires legacy integer fields the API stopped
    sending in 2026-07 (the count now arrives as the `position_fp` string) —
    the modeled get_positions call raises ValidationError on any non-empty
    page, which would turn every ambiguous order into manual_review.

    Args:
        client (Any): Authenticated KalshiClient from auth.build_client().
        ticker (str): Market ticker to look up.

    Returns:
        float | None: Signed contract count (0 = confirmed no position; may be
            fractional — callers only test zero/non-zero), or None when the
            lookup itself failed and the state remains unknown.
    """
    try:
        # Filter server-side by ticker so a single page is guaranteed to contain it
        data = fetch_json_page(client.get_positions_without_preload_content, ticker=ticker)
        for pos in data.get("market_positions") or []:
            if pos.get("ticker") == ticker:
                # position_fp is the signed contract count the API now sends;
                # fall back to the legacy integer field if it ever reappears
                raw = pos.get("position_fp")
                if raw is None:
                    raw = pos.get("position")
                return float(raw)
        return 0
    except Exception as exc:
        logging.warning("Position lookup failed for %s: %s", ticker, exc)
        return None


def _submit_order(client: Any, order: CreateOrderRequest) -> str:
    """
    Submit one order via the raw-response endpoint and return its fill status.

    Uses create_order_without_preload_content because the pinned SDK's Order
    response model requires legacy integer fields the API stopped sending in
    2026-07 — the modeled create_order call raises ValidationError AFTER the
    order has been submitted, which would shove every real fill into the
    ambiguous-exception path (and unwind successfully filled legs). The raw
    variant serializes the request through the exact same code path as the
    modeled call, so the HTTP request on the wire is identical.

    Deliberately NOT wrapped in api_call_with_retry: retrying a FoK order
    could double-submit a leg at a different price (see module docstring).
    fetch_json_page raises ApiException on non-2xx just like the modeled call
    did, so callers' exception handling is unchanged.

    Args:
        client (Any): Authenticated KalshiClient from auth.build_client().
        order (CreateOrderRequest): The order to submit.

    Returns:
        str: The submitted order's status (e.g. "executed", "canceled").

    Raises:
        ApiException: On non-2xx HTTP status.
        KeyError/TypeError: If the response lacks order.status — callers treat
            any exception as an ambiguous submission and consult the position.
    """
    data = fetch_json_page(client.create_order_without_preload_content, create_order_request=order)
    return data["order"]["status"]


def _rollback_leg_a(client: Any, spec: TradeSpec, reason: str) -> TradeResult:
    """
    Close the leg-A NO position to unwind a half-filled pair, verifying the fill.

    The unwind is a reduce-only market sell on the legacy path and a reduce-only
    aggressive bid on V2 (closing a NO position is buying back the YES short);
    reduce_only guarantees either can only close an existing position, so it is
    safe to submit even when leg A's fill state is ambiguous (it cannot open new
    exposure). The rollback's own FoK status IS checked: an unfilled rollback
    means the leg-A position is still open, which is reported as
    status="rollback_failed" for manual review — never silently as "rolled_back".

    Args:
        client (Any): Authenticated KalshiClient from auth.build_client().
        spec (TradeSpec): The trade whose leg A must be unwound.
        reason (str): The upstream failure that triggered the rollback; recorded
            in the TradeResult error field.

    Returns:
        TradeResult: status="rolled_back" when the unwind filled,
            status="rollback_failed" when it did not fill or raised.
    """
    # Version-dispatched build: a reduce-only bid on V2, a reduce-only market
    # sell on the legacy path — both close the leg-A NO position
    rollback = _build_rollback_order_any(spec)
    try:
        # Raw-response / signed submission — see _submit_order and
        # _submit_order_v2 for why the modeled create_order call cannot be used
        rb_status = _submit_any(client, rollback)
    except Exception as rb_err:
        logging.critical(
            "ROLLBACK FAILED for '%s' — ORPHANED POSITION: %d NO contracts on %s."
            " Manual review required. Error: %s",
            spec.pair.canonical_title, spec.x, spec.pair.market_a.ticker, rb_err,
        )
        return TradeResult(
            spec=spec, status="rollback_failed",
            error=f"{reason}; rollback error: {rb_err}",
        )
    if rb_status != "executed":
        logging.critical(
            "ROLLBACK NOT FILLED (status=%s) for '%s' — ORPHANED POSITION: %d NO"
            " contracts on %s. Manual review required.",
            rb_status, spec.pair.canonical_title, spec.x, spec.pair.market_a.ticker,
        )
        return TradeResult(
            spec=spec, status="rollback_failed",
            error=f"{reason}; rollback FoK not filled: status={rb_status}",
        )
    logging.warning(
        "Rollback executed for '%s' — sold %d NO contracts on %s",
        spec.pair.canonical_title, spec.x, spec.pair.market_a.ticker,
    )
    return TradeResult(spec=spec, status="rolled_back", error=reason)


def pre_execution_check(client: Any, portfolio: list) -> list:
    """
    Re-validate order book prices for all specs concurrently before execution.

    Fetches both order books for each spec in parallel and drops any whose gap
    threshold is no longer met or whose available depth is less than the intended
    contract count. This reduces the window between price observation and order
    submission, lowering the chance of submitting against a stale price.

    Args:
        client (Any): Authenticated KalshiClient from auth.build_client().
        portfolio (list): List of TradeSpec objects selected by select_portfolio().

    Returns:
        list: Filtered list of TradeSpec objects that still pass the price check.
            May be empty if all pairs' prices moved since the scan.
    """
    if not portfolio:
        return []

    valid = []
    with ThreadPoolExecutor(max_workers=min(8, len(portfolio))) as pool:
        future_to_spec = {
            pool.submit(validate_pair_price, client, spec): spec for spec in portfolio
        }
        # Iterate as futures complete so one raising thread does not swallow the others;
        # a raised exception is caught per-spec and the spec is dropped like a False check.
        for future in as_completed(future_to_spec):
            spec = future_to_spec[future]
            try:
                ok = future.result()
            except Exception as exc:
                logging.warning(
                    "Pre-execution check raised for '%s' — dropping: %s",
                    spec.pair.canonical_title, exc,
                )
                continue
            if ok:
                valid.append(spec)
            else:
                logging.warning(
                    "Pre-execution price check failed for '%s' — dropping from portfolio",
                    spec.pair.canonical_title,
                )
    return valid


def _execute_one(client: Any, spec: TradeSpec) -> TradeResult:
    """
    Execute one arbitrage pair with sequential leg submission and rollback.

    Submits leg A (NO on market A) first via fill_or_kill. If it fills, submits
    leg B (YES on market B) via fill_or_kill. If leg B fails, immediately submits
    a reduce-only order closing the leg A contracts to unwind the position and
    verifies that the rollback itself filled. Which endpoint each order goes to
    is decided by config.ORDER_API_VERSION inside the _*_any dispatchers; every
    status and safety rule below is identical on both paths.

    A rejected FoK (status != "executed") is a confirmed non-fill. An exception,
    however, is ambiguous — the order may have filled before a timeout — so
    exception paths check the account's actual position for the ticker:
    leg A ambiguous with a position (or unknown state) is unwound; leg B
    ambiguous with a confirmed position means the pair actually completed.
    Leg B ambiguous with an UNKNOWN position (the lookup itself failed) is
    never auto-rolled-back — an automated unwind could reverse a real fill
    that we simply couldn't confirm — and is instead surfaced as
    status="manual_review" for a human to check the account.

    While the LEGACY order path is selected, a pair with a leg on a shard that
    endpoint cannot route to is refused before anything is submitted (see
    _legacy_routable) — status "failed", because nothing was sent and there is
    nothing to unwind. On the V2 path the guard does not apply: every order
    body routes itself via its own market's exchange_index.

    Args:
        client (Any): Authenticated KalshiClient from auth.build_client().
        spec (TradeSpec): The trade specification to execute.

    Returns:
        TradeResult: With status "executed", "failed", "rolled_back",
            "rollback_failed" (unwind did not fill — orphaned position needing
            manual review), or "manual_review" (leg B's fill state could not
            be determined and no automated action was taken).
    """
    # Legacy-mode shard guard — must run before ANY order is built or sent.
    # Markets are tagged with their shard at ingest, but the legacy order
    # endpoint has no shard-routing parameter and can only reach
    # DEFAULT_EXCHANGE_INDEX. Read the module-level ORDER_API_VERSION exactly
    # as the _*_any dispatchers do, so the guard and the dispatch can never
    # disagree about which path is active.
    if ORDER_API_VERSION != "v2" and not _legacy_routable(spec):
        shards = (
            f"{spec.pair.market_a.exchange_index}/{spec.pair.market_b.exchange_index}"
        )
        logging.warning(
            "Skipping '%s' — legs on exchange shards %s; the legacy order"
            " endpoint in use cannot route off shard %d (set"
            " ORDER_API_VERSION='v2')",
            spec.pair.canonical_title, shards, DEFAULT_EXCHANGE_INDEX,
        )
        return TradeResult(
            spec=spec, status="failed",
            error=(
                f"leg on exchange shard {shards} — legacy order endpoint "
                f"cannot route; set ORDER_API_VERSION='v2'"
            ),
        )

    order_a  = _build_no_order_any(spec)
    order_b  = _build_yes_order_any(spec)
    mA_title = spec.pair.market_a.title or spec.pair.market_a.ticker
    mB_title = spec.pair.market_b.title or spec.pair.market_b.ticker

    # Submit leg A — NO on market A (version-dispatched; see _submit_any)
    try:
        status_a = _submit_any(client, order_a)
        if status_a != "executed":
            # FoK rejection is a confirmed non-fill — safe to walk away
            logging.info(
                "Leg A (NO on '%s') not filled (status=%s) — aborting pair",
                mA_title[:60], status_a,
            )
            return TradeResult(
                spec=spec, status="failed",
                error=f"Leg A FoK not filled: status={status_a}",
            )
    except Exception as e:
        # Ambiguous: the order may have filled before the exception (e.g. a
        # timeout after the fill). Consult the actual position before concluding.
        held_a = _position_count(client, spec.pair.market_a.ticker)
        if held_a == 0:
            logging.error(
                "Leg A submission failed for '%s' (no position confirmed): %s",
                mA_title[:60], e,
            )
            return TradeResult(spec=spec, status="failed", error=f"Leg A error: {e}")
        # Position exists — or lookup failed and the state is unknown. Unwind:
        # the rollback is reduce_only, so it is harmless if nothing was filled.
        logging.error(
            "Leg A raised for '%s' but position=%s — unwinding: %s",
            mA_title[:60], held_a, e,
        )
        return _rollback_leg_a(client, spec, f"Leg A ambiguous error: {e}")

    # Leg A filled — submit leg B — YES on market B (version-dispatched)
    leg_b_error: str | None = None
    leg_b_ambiguous = False
    try:
        status_b = _submit_any(client, order_b)
        if status_b != "executed":
            leg_b_error = f"Leg B FoK not filled: status={status_b}"
    except Exception as e:
        leg_b_error = f"Leg B error: {e}"
        leg_b_ambiguous = True

    if leg_b_error:
        if leg_b_ambiguous:
            # The exception may have arrived after the fill — check the position
            # before rolling back leg A, or we'd reverse the unhedged exposure.
            held_b = _position_count(client, spec.pair.market_b.ticker)
            if held_b:  # non-zero and lookup succeeded → leg B actually filled
                logging.warning(
                    "Leg B raised for '%s' but position=%s — pair is complete: %s",
                    mB_title[:60], held_b, leg_b_error,
                )
                return TradeResult(
                    spec=spec, status="executed",
                    error=f"Leg B ambiguous but position confirmed: {leg_b_error}",
                )
            if held_b is None:
                # The position lookup itself failed, so leg B's fill state is
                # genuinely unknown — it may have filled. Rolling back leg A
                # here would be wrong if leg B actually did fill (we'd sell the
                # hedge and be left with a naked YES position on B while the
                # log says "rolled_back", implying flat). Do NOT auto-rollback;
                # surface for manual review instead.
                logging.critical(
                    "Leg B raised for '%s' and position lookup FAILED — fill "
                    "state unknown, NOT auto-rolling-back leg A to avoid "
                    "reversing a possible real fill. Manual review required: %s",
                    mB_title[:60], leg_b_error,
                )
                return TradeResult(
                    spec=spec, status="manual_review",
                    error=f"Leg B ambiguous and position lookup failed: {leg_b_error}",
                )
        logging.error(
            "Leg B (YES on '%s') failed after Leg A filled — attempting rollback: %s",
            mB_title[:60], leg_b_error,
        )
        return _rollback_leg_a(client, spec, leg_b_error)

    logging.info(
        "Both legs filled: '%s'  x=%d NO(A) y=%d YES(B)",
        spec.pair.canonical_title, spec.x, spec.y,
    )
    return TradeResult(spec=spec, status="executed")


def execute_trades(client: Any, specs: list, dry_run: bool = False) -> list:
    """
    Execute each TradeSpec as a sequential two-leg trade, with pairs running
    concurrently across specs.

    In live mode, each spec is handled by _execute_one(): leg A submitted first,
    then leg B only if leg A filled, with rollback if leg B fails. All specs are
    submitted concurrently via ThreadPoolExecutor so no pair waits on another.

    In dry_run mode, no orders are submitted. The function logs the intended trade
    and returns TradeResult objects with status="simulated", which are still written
    to the dev simulation Excel file by reporter.py.

    Args:
        client (Any): An authenticated KalshiClient produced by auth.build_client().
            Must be pointed at the correct endpoint (prod vs. sandbox).
        specs (list): List of TradeSpec objects from strategy.select_portfolio().
            Each spec encodes one arbitrage pair with a final integer contract count.
        dry_run (bool): If True, skip actual order submission and return simulated
            results. Defaults to False. Always True in dev/sandbox mode.

    Returns:
        list: List of TradeResult objects (from reporter.py), one per spec. Each
            result has status="executed" (both legs filled), "simulated" (dry run),
            "failed" (leg A failed), "rolled_back" (leg B failed, leg A unwound),
            "rollback_failed" (leg A unwind did not fill — orphaned position),
            or "manual_review" (leg B's fill state could not be determined —
            no automated rollback was attempted).
    """
    # ThreadPoolExecutor(max_workers=0) raises ValueError, so short-circuit empty input
    if not specs:
        return []

    if dry_run:
        results = []
        for spec in specs:
            mA_title = spec.pair.market_a.title or spec.pair.market_a.ticker
            mB_title = spec.pair.market_b.title or spec.pair.market_b.ticker
            logging.info(
                "[DRY RUN] Batch order: Buy %dx NO on '%s' @ %.2f%% | Buy %dx YES on '%s' @ %.2f%% | "
                "Total cost: $%.2f | Min profit: $%.2f",
                spec.x, mA_title[:60], spec.pair.nA * 100,
                spec.y, mB_title[:60], spec.pair.pB * 100,
                spec.total_cost, spec.min_payoff,
            )
            results.append(TradeResult(spec=spec, status="simulated"))
        return results

    with ThreadPoolExecutor(max_workers=min(8, len(specs))) as pool:
        futures = [pool.submit(_execute_one, client, spec) for spec in specs]
        results = [f.result() for f in futures]

    return results
