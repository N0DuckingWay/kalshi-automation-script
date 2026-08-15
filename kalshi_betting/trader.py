"""
File: trader.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Converts TradeSpec objects (produced by strategy.py) into Kalshi REST API
    order requests and submits them. Each pair's two legs are submitted sequentially
    with fill_or_kill semantics: leg A (NO on market A) first, then leg B (YES on
    market B) only if leg A filled. Both buy legs carry a buy_max_cost cap derived
    from the scanned price plus a small slippage allowance, so a book that moved
    since the pre-execution check kills the order instead of filling it at a loss.
    If leg B fails, a rollback sell order is immediately submitted to unwind leg A,
    and the rollback's own fill status is verified — an unfilled rollback is
    reported as status="rollback_failed" (orphaned position, manual review).
    Multiple pairs are executed concurrently via ThreadPoolExecutor so no pair
    waits for another to complete.

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
    no cycle), and BUY_MAX_COST_SLIPPAGE_CENTS / DEFAULT_EXCHANGE_INDEX /
    TRANSFER_PATH / TRANSFER_SETTLE_TIMEOUT_SECONDS /
    TRANSFER_POLL_INTERVAL_SECONDS from config.py. Called by main.py after
    select_portfolio() selects the final trade list. Depends on the
    KalshiClient produced by auth.py.

Notes:
    Do NOT add retry logic to order submission. A failed leg indicates the market
    moved between scan time and execution time — retrying risks buying one leg at
    a worse price and creating an unhedged directional position.

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Any

from kalshi_python_sync.models import CreateOrderRequest

from ._http import fetch_json_page, signed_raw_request
from .auth import verify_auth
from .config import (
    BUY_MAX_COST_SLIPPAGE_CENTS,
    DEFAULT_EXCHANGE_INDEX,
    TRANSFER_PATH,
    TRANSFER_POLL_INTERVAL_SECONDS,
    TRANSFER_SETTLE_TIMEOUT_SECONDS,
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
    rollback = CreateOrderRequest(
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
    try:
        # Raw-response submission — see _submit_order for why the modeled
        # create_order call can no longer be used
        rb_status = _submit_order(client, rollback)
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

    order_a  = _build_no_order(spec)
    order_b  = _build_yes_order(spec)
    mA_title = spec.pair.market_a.title or spec.pair.market_a.ticker
    mB_title = spec.pair.market_b.title or spec.pair.market_b.ticker

    # Submit leg A — NO on market A (raw-response endpoint; see _submit_order)
    try:
        status_a = _submit_order(client, order_a)
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

    # Leg A filled — submit leg B — YES on market B (raw-response endpoint)
    leg_b_error: str | None = None
    leg_b_ambiguous = False
    try:
        status_b = _submit_order(client, order_b)
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
