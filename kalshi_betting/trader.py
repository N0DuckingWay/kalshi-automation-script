"""
File: trader.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Converts TradeSpec objects (produced by strategy.py) into Kalshi REST API
    order requests and submits them. Each pair's two legs are submitted sequentially
    with fill_or_kill semantics: leg A (NO on market A) first, then leg B (YES on
    market B) only if leg A filled. Both buy legs carry a price cap derived
    from the scanned price plus a small slippage allowance, so a book that moved
    since the pre-execution check kills the order instead of filling it at a loss.
    If leg B fails, a rollback sell order is immediately submitted to unwind leg A,
    and the rollback's own fill status is verified — an unfilled rollback is
    reported as status="rollback_failed" (orphaned position, manual review).
    Multiple pairs are executed concurrently via ThreadPoolExecutor so no pair
    waits for another to complete.

    Every leg submission goes through the ONE seam _submit_leg(), which dispatches
    on config.V2_ORDERS_ENABLED between two wire formats: the legacy
    /portfolio/orders create-order request (integer-cent buy_max_cost, side
    yes/no + action buy/sell, no shard routing) and the V2
    /portfolio/events/orders body (dollar-string limit price quantized onto the
    market's own tick grid, fixed-point count strings, bid/ask sides, explicit
    per-order exchange_index). The flag is False today, so the legacy path is
    what actually runs; the surrounding rollback / manual_review state machine
    is identical on both branches.

    An exception from create_order does NOT prove the order was rejected (a
    timeout can land after the fill), so exception paths consult the actual
    account position for the ticker before classifying the outcome.

    pre_execution_check() re-fetches order books for each spec in the portfolio
    concurrently and drops any whose prices have moved since the scan, reducing
    the chance of submitting orders against a stale price.

    ensure_shard_collateral() runs immediately after that check and before any
    order: Kelly sizing is portfolio-wide (against the summed cross-shard
    balance) but each order settles against its OWN market's exchange shard, so
    a selected trade can need cash on a shard that doesn't hold enough. It
    computes each shard's cash requirement from the legs' cost_with_fees_*,
    plans greedy transfers out of surplus shards, executes them against the
    intra-exchange transfer endpoint, and waits for the (asynchronous) funds to
    land before returning the still-fundable subset of the portfolio.

Dependencies:
    Imports TradeResult from reporter.py and TradeSpec from strategy.py. Imports
    CreateOrderRequest from the kalshi_python_sync SDK and fetch_json_page /
    signed_raw_request from _http.py. Imports validate_pair_price from
    scanner.py, verify_auth from auth.py (used to confirm transfers settled;
    auth.py imports only config/_http, so this is a downward edge and creates
    no cycle), and BUY_MAX_COST_SLIPPAGE_CENTS / BUY_PRICE_SLIPPAGE_DOLLARS /
    DEFAULT_EXCHANGE_INDEX / TRANSFER_PATH / TRANSFER_SETTLE_TIMEOUT_SECONDS /
    TRANSFER_POLL_INTERVAL_SECONDS / V2_CREATE_ORDER_PATH /
    V2_SELF_TRADE_PREVENTION_TYPE from config.py. It ALSO imports the config
    MODULE itself, purely so the V2_ORDERS_ENABLED flag is read as
    config.V2_ORDERS_ENABLED at call time — a `from .config import` of a bool
    freezes its value at import and could not be flipped (by tests or by a
    future runtime switch) without reloading this module. Reads
    ApiMarket.price_level_structure / price_ranges (scanner.py) via the market
    objects hanging off each TradeSpec — no scanner import is needed for that.
    Called by main.py after select_portfolio() selects the final trade list.
    Depends on the KalshiClient produced by auth.py.

Notes:
    Do NOT add retry logic to order submission — on EITHER path. A failed leg
    indicates the market moved between scan time and execution time — retrying
    risks buying one leg at a worse price and creating an unhedged directional
    position.

    THE V2 NO-LEG MAPPING IS AN UNVERIFIED HYPOTHESIS. V2 has no side="no" /
    action="buy" vocabulary at all: an order is a `bid` or an `ask` on the YES
    book. The working hypothesis — isolated in _no_buy_as_v2() and
    _no_close_as_v2(), and nowhere else — is that selling YES you do not hold
    OPENS a NO position (Kalshi's unified ledger: negative position_fp = NO),
    so "buy NO at <= p" maps to "ask at >= 1 - p", and closing that NO position
    maps to a `bid` with reduce_only=True. This cannot be proven offline, which
    is exactly why config.V2_ORDERS_ENABLED stays False until the live probe
    (v2_probe) confirms both halves against the real API.

    Order submission and position lookups use the SDK's raw-response variants
    (via _submit_order / _position_count) because 2026-07 API drift broke the
    pinned SDK's Order and MarketPosition response models — the modeled calls
    raise ValidationError AFTER submission, misclassifying real fills. The
    request side still uses the modeled CreateOrderRequest and serializes
    through the same SDK code path, so the wire format is unchanged.

    _legacy_routable() is a TEMPORARY shard guard. The scanner ingests markets
    from every exchange shard (market data is cross-shard) and tags each with
    its exchange_index, but the legacy /portfolio/orders endpoint this module
    submits through has no shard-routing parameter — so a pair with a leg off
    DEFAULT_EXCHANGE_INDEX is refused at the top of _execute_one rather than
    misrouted. It goes away when order submission migrates to the V2 endpoint
    (/portfolio/events/orders), which routes per shard. While that guard is in
    place the collateral planner below is a pure no-op in practice (all funds
    and all tradeable markets are on shard 0 today); it lands early so the
    funding path is already proven when the V2 flip removes the guard.

    CENTICENTS TRAP. The transfer endpoint's `amount` is int64 CENTICENTS —
    1/100 of a cent, i.e. cents × 100. That is the THIRD money unit in this
    codebase, after integer cents (balances, buy_max_cost) and fixed-point
    dollar strings (auth._dollar_str_to_cents, the *_dollars API fields). The
    conversion lives in exactly one place, _cents_to_centicents(); an inlined
    ×100 anywhere else is a 100× money bug waiting to happen — and in the
    wrong direction it silently moves a hundredth of the intended collateral,
    which then shows up as an unexplained insufficient-funds order rejection.

    TRANSFERS ARE ASYNCHRONOUS. A 2xx from the transfer endpoint means the
    request was ACCEPTED, not that the funds have moved. No order may rely on
    transferred collateral until a fresh balance read shows it has landed, so
    ensure_shard_collateral() polls verify_auth() (bounded by
    TRANSFER_SETTLE_TIMEOUT_SECONDS) before returning; on timeout it logs
    CRITICAL with the in-flight transfer ids and drops the trades that were
    depending on the money.

    NO RETRY ON THE TRANSFER POST — same rule as order submission, different
    reason: the transfer endpoint is NOT idempotent, so a retry after an
    ambiguous failure (e.g. a timeout that landed) moves the money twice. A
    failed transfer is treated as an unfunded shard and its trades are dropped;
    the funds stay put and the next run re-plans from the real balances.
"""
import logging
import math
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal
from functools import partial
from typing import Any

from kalshi_python_sync.models import CreateOrderRequest

from . import config
from ._http import fetch_json_page, signed_raw_request
from .auth import verify_auth
from .config import (
    BUY_MAX_COST_SLIPPAGE_CENTS,
    BUY_PRICE_SLIPPAGE_DOLLARS,
    DEFAULT_EXCHANGE_INDEX,
    TRANSFER_PATH,
    TRANSFER_POLL_INTERVAL_SECONDS,
    TRANSFER_SETTLE_TIMEOUT_SECONDS,
    V2_CREATE_ORDER_PATH,
    V2_SELF_TRADE_PREVENTION_TYPE,
)
from .reporter import TradeResult
from .scanner import validate_pair_price
from .strategy import TradeSpec


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

    TEMPORARY until the V2 order migration: the legacy /portfolio/orders
    endpoint has no shard routing, so an order for a non-shard-0 market would
    fail or misroute. Remove when order submission moves to the V2 endpoint
    (/portfolio/events/orders), which routes per shard.

    The scanner deliberately ingests and tags markets from every shard — the
    market-data endpoints are cross-shard — so this is the single point where
    a shard we cannot reach is turned away, after sizing and immediately
    before anything is submitted.

    Args:
        spec (TradeSpec): The trade specification about to be executed.

    Returns:
        bool: True if both legs can be reached by the legacy order endpoint.
    """
    return (
        spec.pair.market_a.exchange_index == DEFAULT_EXCHANGE_INDEX
        and spec.pair.market_b.exchange_index == DEFAULT_EXCHANGE_INDEX
    )


# Tick grid assumed when a market carries no usable price_ranges: whole cents
# from 1c to 99c, i.e. the legacy grid every Kalshi market used before the
# sub-cent tick structures appeared. Fail-SAFE, not fail-open: a coarser grid
# than the market's real one can only make a quantized cap land on a price the
# exchange definitely accepts (every whole cent is a tick on every finer grid
# observed), and the FoK either fills within it or is killed.
_FALLBACK_TICK_BANDS: list[tuple[Decimal, Decimal, Decimal]] = [
    (Decimal("0.01"), Decimal("0.99"), Decimal("0.01"))
]

# The three leg kinds _submit_leg() knows how to submit. "no_buy" = leg A (open
# a NO position on market A), "yes_buy" = leg B (open a YES position on market
# B), "no_close" = the rollback that unwinds a filled leg A.
_LEG_KINDS = ("no_buy", "yes_buy", "no_close")


def _tick_bands(market: Any) -> list[tuple[Decimal, Decimal, Decimal]]:
    """
    Extract a market's tick-size bands as ascending (start, end, step) Decimals.

    This is the first CONSUMER of the price_level_structure / price_ranges
    fields ingested as groundwork in scanner.ApiMarket — see _quantize_to_tick
    for what they are used for. Only price_ranges is read; the
    price_level_structure string is a human-readable name for the same
    information ("linear_cent", "deci_cent", "tapered_deci_cent") and carries
    no band boundaries, so it is not consulted here.

    Fail-soft by design (see _FALLBACK_TICK_BANDS): the bands are only used to
    tighten a price cap, so anything unusable — a missing attribute, a
    non-list, an empty list, a band with a non-numeric or non-positive step —
    degrades to the whole-cent grid rather than raising. Bands are sorted by
    `start` so band selection is deterministic even if the API ever sends them
    out of order; scanner._parse_price_ranges preserves the API's order.

    Args:
        market (Any): A market object (scanner.ApiMarket, or anything exposing
            a `price_ranges` attribute). Bands may be PriceRange dataclasses
            or plain {"start", "end", "step"} dicts.

    Returns:
        list[tuple[Decimal, Decimal, Decimal]]: Non-empty list of
            (start, end, step) bands in ascending start order. Equal to
            _FALLBACK_TICK_BANDS when nothing usable could be parsed.
    """
    raw = getattr(market, "price_ranges", None)
    if not isinstance(raw, list) or not raw:
        return _FALLBACK_TICK_BANDS
    bands: list[tuple[Decimal, Decimal, Decimal]] = []
    for band in raw:
        try:
            # Accept both the parsed PriceRange dataclass (live path) and the
            # raw dict shape the backtest cache stores, so neither producer
            # silently degrades to the fallback grid.
            values = [
                band.get(name) if isinstance(band, dict) else getattr(band, name)
                for name in ("start", "end", "step")
            ]
            # str() first: PriceRange holds floats, and Decimal(float) would
            # inherit binary noise (Decimal(0.001) is 0.001000000000000000020...)
            # which then leaks into every quantized price.
            start, end, step = (Decimal(str(v)) for v in values)
        except (AttributeError, KeyError, TypeError, ValueError, ArithmeticError):
            continue
        if step <= 0 or end <= start:
            continue
        bands.append((start, end, step))
    if not bands:
        return _FALLBACK_TICK_BANDS
    return sorted(bands)


def _quantize_to_tick(price: Decimal, market: Any, rounding: str) -> Decimal:
    """
    Snap a price onto the market's own tick grid, rounding in a given direction.

    V2 limit prices are dollar strings that must land EXACTLY on a tick of the
    market's price grid or the exchange rejects the order — and that grid is no
    longer universally one cent (MVE/combo markets moved to $0.001 and, on
    2026-08-17, $0.0001 ticks). This is the first reader of the tick-structure
    fields ingested as groundwork (scanner.ApiMarket.price_level_structure /
    price_ranges); before it, the only price protection available was the
    legacy integer-cent buy_max_cost, on which one cent of tolerance meant ~100
    ticks of permitted drift on a $0.0001 grid.

    The caller chooses the rounding direction so quantization can only ever
    TIGHTEN a cap: ROUND_FLOOR for a bid (a lower max-price is stricter),
    ROUND_CEILING for an ask (a higher min-price is stricter).

    Out-of-range prices are clamped into [first band start, last band end]
    before a band is chosen, so an absurd input can't fall outside the grid.
    The first band whose `end` reaches the price wins, which makes a price
    sitting exactly on a shared band boundary resolve to the LOWER band
    (deterministic; the boundary price itself is a tick of both bands anyway).

    Args:
        price (Decimal): Desired price in dollars, before quantization. May sit
            outside [0, 1]; it is clamped.
        market (Any): Market object whose price_ranges define the grid (see
            _tick_bands — a market without usable bands gets the whole-cent
            fallback grid).
        rounding (str): A decimal module rounding mode, e.g. ROUND_FLOOR or
            ROUND_CEILING.

    Returns:
        Decimal: A price on the grid, never 0 or 1 — the result is clamped into
            [step, 1 - step] of the selected band, since neither bound is a
            priceable limit on a binary contract.
    """
    bands = _tick_bands(market)
    lowest, highest = bands[0][0], bands[-1][1]
    clamped = min(max(price, lowest), highest)
    # First band that reaches the price; the fallback to bands[-1] only fires
    # if the bands leave a gap at the top, which the clamp above already rules
    # out for a well-formed grid.
    start, _end, step = next((b for b in bands if clamped <= b[1]), bands[-1])
    # Measure the price in whole ticks FROM THE BAND START rather than from
    # zero: a band is only guaranteed to be aligned to its own start, so
    # quantizing against 0 would silently produce off-grid prices if Kalshi
    # ever ships a band whose start is not a multiple of its step.
    ticks = ((clamped - start) / step).to_integral_value(rounding=rounding)
    quantized = start + ticks * step
    # 0 and 1 are not priceable (a contract cannot trade at certainty), so the
    # extreme ticks of the selected band's step are the real bounds.
    return min(max(quantized, step), Decimal(1) - step)


def _fp(value: Decimal, decimals_max: int = 4) -> str:
    """
    Render a Decimal as a V2 fixed-point string with 2 to `decimals_max` places.

    V2 replaced the legacy integer-cent fields with fixed-point strings: prices
    carry 2-4 decimals and counts 0-2, and both are validated on arrival — so
    scientific notation ("5E+1"), a bare integer where two decimals are
    expected, or more decimals than the field allows are all rejections. This
    formats the minimum number of decimals that satisfies the field: trailing
    zeros beyond two are dropped, and at least two are always present.

    Args:
        value (Decimal): The value to render. Prices should already sit on the
            market's tick grid (see _quantize_to_tick); this only formats.
        decimals_max (int): Maximum decimal places the target field accepts —
            4 for a V2 price, 2 for a V2 count. Defaults to 4. A value with
            more precision than this is rounded half-up, which for a
            grid-quantized price is a no-op.

    Returns:
        str: Fixed-point decimal string, e.g. "0.56", "0.5605", "5.00". Never
            uses an exponent.
    """
    quantum = Decimal(1).scaleb(-decimals_max)
    # normalize() strips trailing zeros (and can hand back an exponent form,
    # e.g. Decimal("100") -> 1E+2), which the re-quantize below undoes.
    trimmed = value.quantize(quantum, rounding=ROUND_HALF_UP).normalize()
    if -trimmed.as_tuple().exponent < 2:
        trimmed = trimmed.quantize(Decimal("0.01"))
    # "f" formatting is what guarantees no exponent survives into the body.
    return f"{trimmed:f}"


def _yes_buy_as_v2(yes_price: float, market: Any) -> tuple[str, str]:
    """
    Map "buy YES at <= yes_price + slippage" onto a V2 (side, limit price) pair.

    A YES buy is the one leg whose V2 form is unambiguous: a `bid` is an order
    to buy the YES side of the book, and its limit price is the MAXIMUM we are
    willing to pay per contract. Flooring the cap onto the market's tick grid
    can therefore only TIGHTEN it (we may end up permitting slightly less
    slippage than BUY_PRICE_SLIPPAGE_DOLLARS, never more).

    Args:
        yes_price (float): Scanned depth-weighted YES price in dollars, (0, 1).
        market (Any): Market object supplying the tick grid (see _tick_bands).

    Returns:
        tuple[str, str]: ("bid", limit price as a fixed-point dollar string).
    """
    # Decimal(str(...)) throughout: the inputs are floats and Decimal(float)
    # would carry binary noise onto the wire.
    cap = Decimal(str(yes_price)) + Decimal(str(BUY_PRICE_SLIPPAGE_DOLLARS))
    return "bid", _fp(_quantize_to_tick(cap, market, ROUND_FLOOR))


def _no_buy_as_v2(no_price: float, market: Any) -> tuple[str, str]:
    """
    Map "buy NO at <= no_price + slippage" onto a V2 (side, limit price) pair.

    *** THIS IS THE UNVERIFIED MAPPING — the reason config.V2_ORDERS_ENABLED
    is still False. *** V2 has no side="no"/action="buy" vocabulary: every
    order is a bid or an ask on the YES book. The hypothesis, drawn from
    Kalshi's unified-ledger convention (a negative position_fp IS a NO
    position), is that selling YES you do not hold OPENS a NO position, so:

        buy NO at <= p   ==   ask (sell YES) at >= 1 - p

    Under that mapping the ask's limit price is the MINIMUM proceeds we accept
    per contract, so CEILING onto the tick grid raises the minimum, which
    tightens the implied NO cost cap — the same direction of conservatism the
    bid leg gets from flooring. If the hypothesis is wrong the order does not
    merely mis-price, it takes the wrong side of the market entirely, which is
    why the live probe (v2_probe) must confirm "ask opens NO" before the flag
    is flipped. Nothing else in this module encodes the assumption.

    Args:
        no_price (float): Scanned depth-weighted NO price in dollars, (0, 1).
        market (Any): Market object supplying the tick grid (see _tick_bands).

    Returns:
        tuple[str, str]: ("ask", limit price as a fixed-point dollar string).
    """
    cap = Decimal(str(no_price)) + Decimal(str(BUY_PRICE_SLIPPAGE_DOLLARS))
    return "ask", _fp(_quantize_to_tick(Decimal(1) - cap, market, ROUND_CEILING))


def _no_close_as_v2(market: Any) -> tuple[str, str]:
    """
    Map "close the held NO position" onto a V2 (side, limit price) pair.

    *** ALSO PART OF THE UNVERIFIED MAPPING (see _no_buy_as_v2) *** — the
    hypothesis is that a `bid` (buy YES) carrying reduce_only=True nets an
    existing NO position toward flat rather than opening a YES one, since the
    two sides share one signed ledger. reduce_only itself is set by
    _build_v2_order_body, not here.

    The limit price is the top of the grid (one tick below 1), which reproduces
    the legacy rollback's deliberate "unwind at any price" urgency: the legacy
    order was an uncapped market sell, because an orphaned leg-A position is an
    unhedged directional bet and getting flat matters more than the price of
    getting there. Under FoK a near-1 limit fills against anything resting.

    Args:
        market (Any): Market object supplying the tick grid (see _tick_bands).

    Returns:
        tuple[str, str]: ("bid", top-of-grid price as a fixed-point dollar
            string, e.g. "0.99" on a whole-cent market).
    """
    # _quantize_to_tick clamps its result to one step below 1, so asking for 1
    # yields exactly the highest priceable tick.
    return "bid", _fp(_quantize_to_tick(Decimal(1), market, ROUND_FLOOR))


def _build_v2_order_body(
    ticker: str,
    side: str,
    price_str: str,
    count: int,
    exchange_index: int,
    *,
    reduce_only: bool = False,
) -> dict:
    """
    Build the JSON body for one V2 POST /portfolio/events/orders request.

    Deliberately a PLAIN DICT, never the SDK's CreateOrderRequest: that pydantic
    model has no slot for exchange_index (or for the V2 field names at all) and
    would silently DROP it, sending an unrouted order. _http.signed_raw_request
    passes a dict through to the wire verbatim, which is the whole reason it
    exists.

    Args:
        ticker (str): Market ticker the order is for.
        side (str): "bid" (buy the YES side) or "ask" (sell it) — produced by
            one of the _*_as_v2 mapping functions, never assembled ad hoc.
        price_str (str): Limit price as a fixed-point dollar string, already
            quantized onto the market's tick grid.
        count (int): Whole contracts to trade. >= 1.
        exchange_index (int): The shard the market lives on, taken from that
            market's own ApiMarket.exchange_index.
        reduce_only (bool): True only for a position-closing order (the leg-A
            rollback), where it caps the order at the position we actually
            hold so it can never open the opposite exposure. Defaults to False.

    Returns:
        dict: The request body, ready for _submit_order_v2.
    """
    return {
        "ticker": ticker,
        "side": side,
        # Counts are fixed-point strings on V2 (0-2 decimals); the bot still
        # sizes in whole contracts, so this is always "<n>.00".
        "count": _fp(Decimal(count), 2),
        "price": price_str,
        # Same atomicity guarantee as the legacy path: the full count fills
        # immediately or nothing does. Never relax this to IOC — a partial
        # fill is an unhedged position by another name.
        "time_in_force": "fill_or_kill",
        # Required field on V2, no default.
        "self_trade_prevention_type": V2_SELF_TRADE_PREVENTION_TYPE,
        # Fresh per order: it is the idempotency handle the exchange dedupes
        # on, so two legs (or two runs) must never share one.
        "client_order_id": str(uuid.uuid4()),
        # We are always the taker — a maker-only order would simply rest, and
        # a resting order is the opposite of the atomic fill this bot needs.
        "post_only": False,
        # If the market pauses mid-flight, cancel rather than leave an order
        # queued to fill on prices we scanned before the pause.
        "cancel_order_on_pause": True,
        "reduce_only": reduce_only,
        # EXPLICIT shard, deliberately NOT the -1 auto-route sentinel: if our
        # notion of a market's shard is ever wrong we want the exchange to
        # reject the order loudly, not to silently paper over the disagreement
        # and settle the trade against a shard the collateral planner never
        # funded.
        "exchange_index": exchange_index,
    }


def _build_legacy_order(leg: str, ticker: str, count: int, price: float) -> CreateOrderRequest:
    """
    Build the legacy /portfolio/orders request for one leg kind.

    Single source of truth for the legacy wire format, shared by _submit_leg()
    and by the _build_no_order / _build_yes_order helpers. All three leg kinds
    are market (taker) fill_or_kill orders; the two buy legs carry the
    integer-cent buy_max_cost price cap, and the close leg is reduce_only.

    Args:
        leg (str): One of _LEG_KINDS — "no_buy", "yes_buy", or "no_close".
        ticker (str): Market ticker the order is for.
        count (int): Whole contracts to trade. >= 1.
        price (float): Scanned per-contract price in dollars for the cap.
            IGNORED for "no_close", which is an uncapped market sell (getting
            flat matters more than the price of getting there).

    Returns:
        CreateOrderRequest: The SDK request object for this leg.

    Raises:
        ValueError: If `leg` is not one of _LEG_KINDS.
    """
    if leg == "no_close":
        return CreateOrderRequest(
            ticker=ticker,
            side="no",
            action="sell",
            type="market",
            count=count,
            time_in_force="fill_or_kill",
            # Can only reduce an existing position — never opens a short even if
            # leg A turns out not to have filled after all
            reduce_only=True,
        )
    if leg not in ("no_buy", "yes_buy"):
        raise ValueError(f"unknown leg kind: {leg!r}")
    return CreateOrderRequest(
        ticker=ticker,
        side="yes" if leg == "yes_buy" else "no",
        action="buy",
        # "market" type means we accept the current best ask — we are the taker
        type="market",
        count=count,
        # fill_or_kill: execute the full count immediately or cancel with no fill
        time_in_force="fill_or_kill",
        # Price protection: never pay more than scanned price + slippage allowance
        buy_max_cost=_buy_max_cost_cents(count, price),
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
    # Thin spec-shaped wrapper over the shared legacy builder — _submit_leg
    # calls _build_legacy_order directly with the same three values, so the
    # request is identical whichever way it is reached.
    return _build_legacy_order(
        "no_buy", spec.pair.market_a.ticker, spec.x, spec.pair.nA
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
    # Thin spec-shaped wrapper — see _build_no_order.
    return _build_legacy_order(
        "yes_buy", spec.pair.market_b.ticker, spec.y, spec.pair.pB
    )


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


def _submit_order_v2(client: Any, body: dict) -> dict:
    """
    Submit one V2 order and return the parsed response body.

    The pinned SDK has no generated method for POST /portfolio/events/orders
    (and its CreateOrderRequest model could not express the body anyway — see
    _build_v2_order_body), so the request goes through _http.signed_raw_request
    composed with fetch_json_page for the shared non-2xx -> ApiException +
    JSON-parse contract. That keeps the exception types callers already handle
    unchanged between the two paths.

    Deliberately NOT wrapped in api_call_with_retry, for exactly the reason
    given in _submit_order: retrying a fill-or-kill order can double-submit a
    leg at a different price and leave an unhedged position. One attempt.

    Args:
        client (Any): Authenticated KalshiClient from auth.build_client().
        body (dict): Request body from _build_v2_order_body().

    Returns:
        dict: The parsed response body — order_id, client_order_id, fill_count,
            remaining_count, and (only when something filled) average_fill_price
            and average_fee_paid.

    Raises:
        ApiException: On non-2xx HTTP status.
        Exception: Any transport-level error — the order's fate is then unknown
            and callers must treat it as ambiguous, never resubmit it.
    """
    # partial() binds the signed-request builder so fetch_json_page can invoke
    # it with the body kwarg and apply its status-check + JSON-parse contract.
    return fetch_json_page(
        partial(signed_raw_request, client, "POST", V2_CREATE_ORDER_PATH), body=body
    )


def _v2_filled(data: dict, count: int) -> bool:
    """
    Decide whether a V2 fill-or-kill order filled completely.

    V2 responses carry NO `status` field — unlike the legacy Order object, which
    reported "executed"/"canceled". For a FoK (or IOC) order the response's
    remaining_count is the FINAL state, so a complete fill is exactly
    "nothing left over, and everything we asked for was filled". Both
    conditions are checked rather than just one, so a response that somehow
    reported neither honestly can't read as a fill.

    Missing or unparseable fields deliberately RAISE (KeyError from the lookup,
    TypeError/InvalidOperation from Decimal) instead of returning False. That
    routes the caller into its existing ambiguous-exception path — position
    lookup, then rollback or manual_review — which is the correct handling for
    "we cannot tell whether this filled", and is exactly what a legacy response
    missing order.status already did.

    Args:
        data (dict): Parsed response body from _submit_order_v2().
        count (int): Whole contracts the order asked for.

    Returns:
        bool: True only when remaining_count is 0 and fill_count equals count.

    Raises:
        KeyError: If remaining_count or fill_count is absent.
        TypeError / decimal.InvalidOperation: If either is unparseable.
    """
    # Fixed-point strings ("0.00", "5.00"), so compare as Decimals — float()
    # would be lossy for the fractional counts V2 permits.
    return Decimal(data["remaining_count"]) == 0 and Decimal(data["fill_count"]) == count


def _submit_leg(
    client: Any,
    *,
    ticker: str,
    leg: str,
    count: int,
    price: float,
    market: Any,
    reduce_only: bool = False,
) -> bool:
    """
    Submit ONE order leg on whichever order path is enabled, and report the fill.

    This is the single seam between the trading state machine and the wire
    format. _execute_one() and _rollback_leg_a() call nothing else to submit an
    order, so the legacy and V2 paths cannot drift in when they submit, only in
    what they put on the wire:

      * legacy (config.V2_ORDERS_ENABLED False — today): the same
        CreateOrderRequest this module has always built, filled = status is
        "executed".
      * V2: leg kind mapped to a (side, tick-quantized limit price) pair, a
        plain-dict body carrying the market's own exchange_index, filled =
        remaining_count 0 and fill_count == count.

    Neither branch retries, and both surface transport failures as exceptions
    for the caller's ambiguity handling — a returned False means CONFIRMED not
    filled, never "unknown".

    Args:
        client (Any): Authenticated KalshiClient from auth.build_client().
        ticker (str): Market ticker for this leg.
        leg (str): One of _LEG_KINDS — "no_buy" (leg A), "yes_buy" (leg B), or
            "no_close" (the leg-A rollback).
        count (int): Whole contracts. >= 1.
        price (float): Scanned per-contract price in dollars for this leg's
            cap. Ignored by BOTH paths for "no_close", which prices itself at
            the top of the grid / submits as an uncapped market sell.
        market (Any): The market object this leg trades, read for its tick grid
            (price_ranges) and its exchange_index. Unused on the legacy path,
            which has no shard routing and no sub-cent prices.
        reduce_only (bool): Forwarded to the V2 body. Forced True for
            "no_close" regardless of what the caller passed — a closing order
            that could open the opposite exposure is the one thing this seam
            must never emit.

    Returns:
        bool: True if the full count filled, False if the FoK was killed.

    Raises:
        ValueError: If `leg` is not one of _LEG_KINDS.
        ApiException / Exception: Propagated from submission; the fill state is
            then ambiguous and the caller must consult the account position.
    """
    if leg not in _LEG_KINDS:
        raise ValueError(f"unknown leg kind: {leg!r}")
    # A close must never be able to open a position, on either path.
    reduce_only = reduce_only or leg == "no_close"

    # Attribute access on the config MODULE, not a from-import: the flag has to
    # be read at call time so it can be flipped (tests monkeypatch it) without
    # re-importing this module. Every other config value here is a from-import
    # because none of them is a switch.
    if not config.V2_ORDERS_ENABLED:
        status = _submit_order(client, _build_legacy_order(leg, ticker, count, price))
        if status != "executed":
            logging.info("FoK not filled: leg=%s ticker=%s status=%s", leg, ticker, status)
        return status == "executed"

    if leg == "no_buy":
        # UNVERIFIED MAPPING — see _no_buy_as_v2.
        side, price_str = _no_buy_as_v2(price, market)
    elif leg == "yes_buy":
        side, price_str = _yes_buy_as_v2(price, market)
    else:
        # UNVERIFIED MAPPING — see _no_close_as_v2.
        side, price_str = _no_close_as_v2(market)

    body = _build_v2_order_body(
        ticker, side, price_str, count, market.exchange_index, reduce_only=reduce_only
    )
    data = _submit_order_v2(client, body)
    filled = _v2_filled(data, count)
    if not filled:
        logging.info(
            "FoK not filled: leg=%s ticker=%s side=%s price=%s remaining_count=%s",
            leg, ticker, side, price_str, data.get("remaining_count"),
        )
    return filled


def _rollback_leg_a(client: Any, spec: TradeSpec, reason: str) -> TradeResult:
    """
    Sell the leg-A NO position to unwind a half-filled pair, verifying the fill.

    reduce_only guarantees the sell can only close an existing position, so it
    is safe to submit even when leg A's fill state is ambiguous (it cannot open
    a short). The rollback's own FoK status IS checked: an unfilled rollback
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
    try:
        # Single submission seam — reduce_only on whichever order path is
        # enabled; see _submit_leg (and _no_close_as_v2 for the V2 mapping).
        rb_filled = _submit_leg(
            client,
            ticker=spec.pair.market_a.ticker,
            leg="no_close",
            count=spec.x,
            # Unused for a close on both paths — the unwind is deliberately
            # uncapped, since being flat matters more than the exit price.
            price=spec.pair.nA,
            market=spec.pair.market_a,
            reduce_only=True,
        )
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
    if not rb_filled:
        logging.critical(
            "ROLLBACK NOT FILLED for '%s' — ORPHANED POSITION: %d NO"
            " contracts on %s. Manual review required.",
            spec.pair.canonical_title, spec.x, spec.pair.market_a.ticker,
        )
        return TradeResult(
            spec=spec, status="rollback_failed",
            error=f"{reason}; rollback FoK not filled",
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


def _cents_to_centicents(cents: int) -> int:
    """
    Convert whole cents to CENTICENTS, the transfer endpoint's money unit.

    A centicent is 1/100 of a cent (so $1.00 = 100 cents = 10,000 centicents).
    This is the THIRD money unit in the codebase and it exists in exactly one
    place — the `amount` field of POST TRANSFER_PATH:

      * integer CENTS        — balances, buy_max_cost, MIN_BALANCE_CENTS
      * fixed-point DOLLARS  — the *_dollars API strings (auth._dollar_str_to_cents)
      * integer CENTICENTS   — intra-exchange transfer amounts (here, only here)

    The arithmetic is trivial on purpose; the function exists so the unit
    conversion is NAMED at the one call site that needs it. Never inline the
    factor: a missing ×100 moves 1% of the intended collateral (an unexplained
    insufficient-funds rejection later), and a doubled one moves 100× the
    intended amount out of a shard.

    Args:
        cents (int): Amount in whole cents. >= 0.

    Returns:
        int: The same amount expressed in centicents (cents × 100).
    """
    return cents * 100


def _required_cents_by_shard(portfolio: list) -> dict[int, int]:
    """
    Sum each exchange shard's cash requirement across a selected portfolio.

    Every leg draws its cash from the shard its own market lives on, so leg A
    charges spec.pair.market_a.exchange_index and leg B charges
    spec.pair.market_b.exchange_index. A pair whose legs share a shard simply
    adds both costs to that one shard.

    Args:
        portfolio (list): TradeSpec objects selected for execution. Each must
            carry per-leg cash requirements in cost_with_fees_a / cost_with_fees_b
            (dollars, fee-inclusive — see strategy.TradeSpec).

    Returns:
        dict[int, int]: exchange_index -> required cash in whole cents. Shards
            no leg touches are absent (not zero-filled).
    """
    required: dict[int, int] = {}
    for spec in portfolio:
        for market, cost_dollars in (
            (spec.pair.market_a, spec.cost_with_fees_a),
            (spec.pair.market_b, spec.cost_with_fees_b),
        ):
            # Ceiling, never floor: under-funding a shard by even a fraction of
            # a cent gets the order rejected for insufficient collateral, while
            # over-funding it by one cent costs nothing. The round() first
            # mirrors config.fee_leg_exact — it stops binary float noise (e.g.
            # 7.000000000000001) from claiming a whole extra cent.
            cents = math.ceil(round(cost_dollars * 100, 6))
            required[market.exchange_index] = required.get(market.exchange_index, 0) + cents
    return required


def _unfunded_shards(required: dict[int, int], available: dict[int, int]) -> set[int]:
    """
    Return the shards whose available balance does not cover their requirement.

    Pure comparison — a shard missing from `available` counts as holding zero,
    which is the safe reading (we never assume money we could not observe).

    Args:
        required (dict[int, int]): exchange_index -> required cents.
        available (dict[int, int]): exchange_index -> observed balance in cents.

    Returns:
        set[int]: exchange_index values still short of cash. Empty when every
            requirement is covered.
    """
    return {shard for shard, need in required.items() if available.get(shard, 0) < need}


def _plan_transfers(
    required_by_shard: dict[int, int], available_by_shard: dict[int, int]
) -> list[tuple[int, int, int]]:
    """
    Plan the shard-to-shard transfers that would fund every shard's requirement.

    PURE function — no I/O, no logging, no clock. Deficit per shard is
    max(0, required - available); surplus is max(0, available - required). Each
    deficit is filled greedily from the largest REMAINING surplus first, which
    minimizes the number of transfers (every transfer is a non-idempotent POST,
    so fewer is strictly better).

    Ordering is fully deterministic so the same balances always produce the same
    plan (and so tests can assert on it): deficits are processed in ascending
    shard-index order, and candidate sources are ranked by remaining surplus
    descending, ties broken by ascending shard index.

    When total surplus is less than total deficit, this plans what IS coverable
    and leaves the rest short rather than failing outright — the caller detects
    the still-unfunded shard(s) from the post-transfer balances and drops only
    the trades that depend on them.

    Args:
        required_by_shard (dict[int, int]): exchange_index -> required cents.
        available_by_shard (dict[int, int]): exchange_index -> available cents.

    Returns:
        list[tuple[int, int, int]]: (source_shard, destination_shard, cents)
            transfers to perform, in execution order. Empty when no shard is
            short, and also when nothing can be moved (no surplus anywhere) —
            so an empty plan alone must NOT be read as "everything is funded".
    """
    deficits = sorted(
        (shard, need - available_by_shard.get(shard, 0))
        for shard, need in required_by_shard.items()
        if need - available_by_shard.get(shard, 0) > 0
    )
    remaining_surplus = {
        shard: avail - required_by_shard.get(shard, 0)
        for shard, avail in available_by_shard.items()
        if avail - required_by_shard.get(shard, 0) > 0
    }

    plan: list[tuple[int, int, int]] = []
    for dest, shortfall in deficits:
        # Re-rank per deficit so "largest remaining surplus first" stays true
        # after earlier deficits have drawn a source down.
        for source in sorted(remaining_surplus, key=lambda s: (-remaining_surplus[s], s)):
            if shortfall <= 0:
                break
            take = min(remaining_surplus[source], shortfall)
            if take <= 0:
                continue
            plan.append((source, dest, take))
            remaining_surplus[source] -= take
            shortfall -= take
    return plan


def _transfers_active(shard_statuses: dict | None, shard: int) -> bool:
    """
    Report whether the exchange says intra-shard transfers are usable on a shard.

    shard_statuses=None means the per-shard breakdown was unavailable (sandbox
    or pre-sharding shape, see scanner.fetch_shard_statuses) — there is nothing
    to gate on, so transfers are attempted and the POST itself is allowed to
    fail loudly if the endpoint is unsupported.

    When a breakdown IS available, a shard missing from it is treated as
    inactive: refusing to move money to or from a shard the exchange did not
    advertise costs us at most a dropped trade, while attempting it risks a
    transfer into a shard whose state we cannot reason about.

    Args:
        shard_statuses (dict | None): scanner.fetch_shard_statuses() output.
        shard (int): exchange_index to check.

    Returns:
        bool: True if a transfer touching this shard may be attempted.
    """
    if shard_statuses is None:
        return True
    return bool((shard_statuses.get(shard) or {}).get("intra_exchange_transfers_active"))


def _spec_shards(spec: TradeSpec) -> set[int]:
    """
    Return the set of exchange shards a single trade's two legs settle against.

    Args:
        spec (TradeSpec): The trade specification.

    Returns:
        set[int]: One element when both legs share a shard, two otherwise.
    """
    return {spec.pair.market_a.exchange_index, spec.pair.market_b.exchange_index}


def _partition_by_funding(portfolio: list, unfunded: set[int]) -> tuple[list, list]:
    """
    Split a portfolio into the trades that are fully funded and those that aren't.

    PURE function. A trade is droppable iff ANY of its legs sits on an unfunded
    shard — both legs must be payable, since a funded leg A with an unpayable
    leg B is precisely the unhedged half-fill the whole rollback machinery
    exists to avoid.

    Args:
        portfolio (list): TradeSpec objects selected for execution.
        unfunded (set[int]): exchange_index values still short of cash.

    Returns:
        tuple[list, list]: (kept, dropped), each preserving the input order.
    """
    kept: list = []
    dropped: list = []
    for spec in portfolio:
        (dropped if _spec_shards(spec) & unfunded else kept).append(spec)
    return kept, dropped


def _execute_transfer(client: Any, source: int, dest: int, cents: int) -> str | None:
    """
    Submit one intra-exchange collateral transfer and return its transfer id.

    Deliberately NOT wrapped in api_call_with_retry: the transfer endpoint is
    not idempotent, so retrying an ambiguous failure (a timeout that actually
    landed) would move the money a second time. One attempt, and a failure is
    reported to the caller as "this shard did not get funded".

    The SDK has no generated method for this route, so the request goes through
    _http.signed_raw_request (signed POST with a verbatim JSON body) composed
    with fetch_json_page for the shared non-2xx -> ApiException + parse
    contract.

    Args:
        client (Any): Authenticated KalshiClient from auth.build_client().
        source (int): exchange_index the funds leave.
        dest (int): exchange_index the funds arrive on.
        cents (int): Amount to move, in whole CENTS (converted to the
            endpoint's centicents here — see _cents_to_centicents).

    Returns:
        str | None: The accepted transfer's id, or None when the response
            carried none (the transfer may still have been accepted, so the
            caller must treat this as "in flight", not "failed").

    Raises:
        ApiException: On a non-2xx status from the transfer endpoint.
        Exception: Any transport-level error — the transfer's fate is then
            unknown and it is NEVER re-sent.
    """
    body = {
        # Both endpoints of an intra-exchange transfer are the trading balance;
        # "event_contract" is Kalshi's name for that collateral pool.
        "source": "event_contract",
        "destination": "event_contract",
        # CENTICENTS (1/100 cent) — NOT cents. See _cents_to_centicents.
        "amount": _cents_to_centicents(cents),
        "source_exchange_shard": source,
        "destination_exchange_shard": dest,
    }
    # partial() binds the signed-request builder so fetch_json_page can invoke
    # it with the body kwarg and apply its status-check + JSON-parse contract.
    data = fetch_json_page(
        partial(signed_raw_request, client, "POST", TRANSFER_PATH), body=body
    )
    return data.get("transfer_id")


def _await_transfer_settlement(client: Any, required: dict[int, int]) -> dict[int, int]:
    """
    Re-read the per-shard balance until every requirement is covered, or time out.

    Transfers are asynchronous — acceptance is not settlement — so this is the
    only thing that proves the collateral actually arrived. Polls every
    TRANSFER_POLL_INTERVAL_SECONDS and gives up after
    TRANSFER_SETTLE_TIMEOUT_SECONDS, returning whatever the last read showed so
    the caller can decide which trades are still fundable.

    A balance read that raises is treated as "nothing observed" (empty dict) and
    retried until the deadline: it is never treated as success, because
    submitting orders against an unverifiable balance is exactly what this
    function exists to prevent.

    Args:
        client (Any): Authenticated KalshiClient from auth.build_client().
        required (dict[int, int]): exchange_index -> required cents; the loop
            ends as soon as every one of these is covered.

    Returns:
        dict[int, int]: The most recent per-shard balance in cents (possibly
            empty if every read failed).
    """
    deadline = time.monotonic() + TRANSFER_SETTLE_TIMEOUT_SECONDS
    balances: dict[int, int] = {}
    while True:
        try:
            # Same shard-aware balance read the run started with, so "landed"
            # is judged against exactly the numbers sizing was based on.
            balances = verify_auth(client)
        except Exception as exc:
            logging.warning("Balance re-read failed while awaiting transfers: %s", exc)
            balances = {}
        if not _unfunded_shards(required, balances):
            return balances
        if time.monotonic() >= deadline:
            return balances
        time.sleep(TRANSFER_POLL_INTERVAL_SECONDS)


def ensure_shard_collateral(
    client: Any,
    portfolio: list,
    shard_balances: dict[int, int],
    shard_statuses: dict | None,
    dry_run: bool = False,
) -> list:
    """
    Move collateral onto the exchange shards the selected portfolio draws from.

    Kelly sizing is portfolio-wide — it runs against the SUM of every shard's
    balance — but an order settles against its own market's shard only. This
    function closes that gap: it totals each shard's cash requirement from the
    legs' cost_with_fees_*, plans greedy transfers out of surplus shards
    (_plan_transfers), executes them, and confirms the asynchronous funds have
    actually landed before letting execution proceed.

    Failure is always degradation, never an abort: a blocked, failed, or
    unsettled transfer results in the affected trades being dropped from the
    returned portfolio while the rest execute normally. Specifically:

      * No shard is short  -> returns the portfolio unchanged, no API calls.
      * dry_run            -> logs the planned transfers and returns the
                              portfolio unchanged. NEVER POSTs.
      * Transfers inactive on either endpoint shard (per shard_statuses) ->
        that transfer is not attempted; a warning tells the operator to move
        the funds manually in the Kalshi UI.
      * A transfer POST raises -> logged as an error and NOT retried (the
        endpoint is not idempotent); its shard simply stays unfunded.
      * Transfers accepted but not settled within
        TRANSFER_SETTLE_TIMEOUT_SECONDS -> logged CRITICAL with the in-flight
        transfer ids ("money is in flight"), and only the trades needing a
        still-unfunded shard are dropped.

    Today this is a pure no-op in practice: all funds and all tradeable markets
    are on DEFAULT_EXCHANGE_INDEX, so no shard is ever short. It exists so the
    funding path is already in place — and already exercised by the zero-deficit
    fast path — when the V2 order migration lets trades route off shard 0.

    Args:
        client (Any): Authenticated KalshiClient from auth.build_client().
        portfolio (list): TradeSpec objects selected for execution, each
            carrying cost_with_fees_a / cost_with_fees_b.
        shard_balances (dict[int, int]): exchange_index -> cents, as returned
            by auth.verify_auth() before this run's orders.
        shard_statuses (dict | None): scanner.fetch_shard_statuses() output,
            used to skip shards whose intra_exchange_transfers_active is false.
            None (breakdown unavailable) means transfers are attempted anyway
            and the POST is allowed to fail loudly.
        dry_run (bool): When True, plan and log but never POST. Defaults to False.

    Returns:
        list: The subset of `portfolio` whose every leg sits on a shard
            confirmed to hold its required cash, in the input order. Equal to
            `portfolio` whenever nothing needed funding or every transfer
            settled; possibly empty when nothing could be funded.
    """
    if not portfolio:
        return []

    required = _required_cents_by_shard(portfolio)
    # An empty plan is ambiguous (nothing needed vs. nothing movable), so the
    # fast path keys off the deficit set itself, not off the plan being empty.
    if not _unfunded_shards(required, shard_balances):
        logging.info(
            "All shards sufficiently funded for %d selected trade(s) — no collateral "
            "transfers needed (required by shard: %s)", len(portfolio), required,
        )
        return portfolio

    plan = _plan_transfers(required, shard_balances)

    if dry_run:
        for source, dest, cents in plan:
            logging.info(
                "DRY RUN: would transfer $%.2f shard %d→%d", cents / 100, source, dest,
            )
        if not plan:
            logging.info("DRY RUN: shard(s) %s under-funded and no surplus to draw on",
                         sorted(_unfunded_shards(required, shard_balances)))
        return portfolio

    accepted: list[str] = []
    attempted = False
    for source, dest, cents in plan:
        if not _transfers_active(shard_statuses, source) or not _transfers_active(
            shard_statuses, dest
        ):
            logging.warning(
                "Intra-exchange transfers are not active on shard %d and/or %d — NOT "
                "moving $%.2f; trades needing shard %d will be dropped. Move the funds "
                "manually in the Kalshi UI to enable them.",
                source, dest, cents / 100, dest,
            )
            continue
        try:
            # Single attempt by design — see _execute_transfer (non-idempotent).
            transfer_id = _execute_transfer(client, source, dest, cents)
        except Exception as exc:
            logging.error(
                "Collateral transfer of $%.2f from shard %d to shard %d FAILED (not "
                "retried — the endpoint is not idempotent): %s",
                cents / 100, source, dest, exc,
            )
            continue
        attempted = True
        accepted.append(str(transfer_id))
        logging.info(
            "Collateral transfer accepted: $%.2f shard %d→%d (transfer_id=%s) — "
            "asynchronous, awaiting settlement",
            cents / 100, source, dest, transfer_id,
        )

    if attempted:
        # Acceptance is not settlement: block (bounded) until a fresh balance
        # read proves the money landed before any order relies on it.
        confirmed = _await_transfer_settlement(client, required)
    else:
        # Nothing moved, so the opening balances are still the truth — don't
        # burn the settle timeout waiting for transfers that were never sent.
        confirmed = shard_balances

    unfunded = _unfunded_shards(required, confirmed)
    if not unfunded:
        logging.info("All shard collateral requirements confirmed funded.")
        return portfolio

    if attempted:
        logging.critical(
            "Collateral transfer(s) did not settle within %ds — MONEY IS IN FLIGHT, "
            "CHECK THE ACCOUNT. transfer_ids=%s; shard(s) still under-funded: %s "
            "(required %s, confirmed %s)",
            TRANSFER_SETTLE_TIMEOUT_SECONDS, accepted, sorted(unfunded), required, confirmed,
        )

    kept, dropped = _partition_by_funding(portfolio, unfunded)
    for spec in dropped:
        logging.warning(
            "Dropping '%s' — leg shard(s) %s include an under-funded shard %s",
            spec.pair.canonical_title, sorted(_spec_shards(spec)), sorted(unfunded),
        )
    logging.warning(
        "Collateral funding incomplete: %d of %d selected trade(s) dropped.",
        len(dropped), len(portfolio),
    )
    return kept


def _execute_one(client: Any, spec: TradeSpec) -> TradeResult:
    """
    Execute one arbitrage pair with sequential leg submission and rollback.

    Submits leg A (NO on market A) first via fill_or_kill. If it fills, submits
    leg B (YES on market B) via fill_or_kill. If leg B fails, immediately submits
    a reduce-only market sell of the leg A contracts to unwind the position and
    verifies that the rollback itself filled.

    A rejected FoK (status != "executed") is a confirmed non-fill. An exception,
    however, is ambiguous — the order may have filled before a timeout — so
    exception paths check the account's actual position for the ticker:
    leg A ambiguous with a position (or unknown state) is unwound; leg B
    ambiguous with a confirmed position means the pair actually completed.
    Leg B ambiguous with an UNKNOWN position (the lookup itself failed) is
    never auto-rolled-back — an automated unwind could reverse a real fill
    that we simply couldn't confirm — and is instead surfaced as
    status="manual_review" for a human to check the account.

    A pair with a leg on a shard the legacy order endpoint cannot route to is
    refused before anything is submitted (see _legacy_routable) — status
    "failed", since nothing was sent and there is nothing to unwind.

    Args:
        client (Any): Authenticated KalshiClient from auth.build_client().
        spec (TradeSpec): The trade specification to execute.

    Returns:
        TradeResult: With status "executed", "failed", "rolled_back",
            "rollback_failed" (unwind did not fill — orphaned position needing
            manual review), or "manual_review" (leg B's fill state could not
            be determined and no automated action was taken).
    """
    # TEMPORARY shard guard — must run before ANY submission. The scanner now
    # ingests markets from every shard, but the legacy order endpoint can only
    # reach DEFAULT_EXCHANGE_INDEX; dies with the V2 order migration.
    if not _legacy_routable(spec):
        shards = (
            f"{spec.pair.market_a.exchange_index}/{spec.pair.market_b.exchange_index}"
        )
        logging.warning(
            "Skipping '%s' — legs on exchange shards %s; the legacy order "
            "endpoint cannot route off shard %d (V2 migration pending)",
            spec.pair.canonical_title, shards, DEFAULT_EXCHANGE_INDEX,
        )
        return TradeResult(
            spec=spec, status="failed",
            error=(
                f"leg on exchange shard {shards} — legacy order endpoint "
                f"cannot route; V2 migration pending"
            ),
        )

    mA_title = spec.pair.market_a.title or spec.pair.market_a.ticker
    mB_title = spec.pair.market_b.title or spec.pair.market_b.ticker

    # Submit leg A — NO on market A (single submission seam; see _submit_leg)
    try:
        if not _submit_leg(
            client,
            ticker=spec.pair.market_a.ticker,
            leg="no_buy",
            count=spec.x,
            price=spec.pair.nA,
            market=spec.pair.market_a,
        ):
            # FoK rejection is a confirmed non-fill — safe to walk away
            logging.info(
                "Leg A (NO on '%s') not filled — aborting pair", mA_title[:60],
            )
            return TradeResult(
                spec=spec, status="failed", error="Leg A FoK not filled",
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

    # Leg A filled — submit leg B — YES on market B (same submission seam)
    leg_b_error: str | None = None
    leg_b_ambiguous = False
    try:
        if not _submit_leg(
            client,
            ticker=spec.pair.market_b.ticker,
            leg="yes_buy",
            count=spec.y,
            price=spec.pair.pB,
            market=spec.pair.market_b,
        ):
            leg_b_error = "Leg B FoK not filled"
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
