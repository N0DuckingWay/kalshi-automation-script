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

Dependencies:
    Imports TradeResult from reporter.py and TradeSpec from strategy.py. Imports
    CreateOrderRequest from the kalshi_python_sync SDK. Imports validate_pair_price
    from scanner.py and BUY_MAX_COST_SLIPPAGE_CENTS from config.py. Called by
    main.py after select_portfolio() selects the final trade list. Depends on
    the KalshiClient produced by auth.py.

Notes:
    Do NOT add retry logic to order submission. A failed leg indicates the market
    moved between scan time and execution time — retrying risks buying one leg at
    a worse price and creating an unhedged directional position.
"""
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from kalshi_python_sync.models import CreateOrderRequest

from .config import BUY_MAX_COST_SLIPPAGE_CENTS
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


def _position_count(client: Any, ticker: str) -> int | None:
    """
    Fetch the signed contract position for one ticker, or None if the lookup fails.

    Used to disambiguate order-submission exceptions: an exception does not
    prove the order was rejected (a timeout can arrive after the fill), so the
    account's actual position is the ground truth. Kalshi convention: negative
    counts are NO contracts, positive counts are YES contracts.

    Args:
        client (Any): Authenticated KalshiClient from auth.build_client().
        ticker (str): Market ticker to look up.

    Returns:
        int | None: Signed contract count (0 = confirmed no position), or None
            when the lookup itself failed and the state remains unknown.
    """
    try:
        # Filter server-side by ticker so a single page is guaranteed to contain it
        resp = client.get_positions(ticker=ticker)
        for pos in resp.market_positions or []:
            if pos.ticker == ticker:
                return int(pos.position)
        return 0
    except Exception as exc:
        logging.warning("Position lookup failed for %s: %s", ticker, exc)
        return None


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
        rb_resp = client.create_order(**rollback.model_dump(exclude_none=True))
        rb_status = rb_resp.order.status
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

    Args:
        client (Any): Authenticated KalshiClient from auth.build_client().
        spec (TradeSpec): The trade specification to execute.

    Returns:
        TradeResult: With status "executed", "failed", "rolled_back",
            "rollback_failed" (unwind did not fill — orphaned position needing
            manual review), or "manual_review" (leg B's fill state could not
            be determined and no automated action was taken).
    """
    order_a  = _build_no_order(spec)
    order_b  = _build_yes_order(spec)
    mA_title = spec.pair.market_a.title or spec.pair.market_a.ticker
    mB_title = spec.pair.market_b.title or spec.pair.market_b.ticker

    # Submit leg A — NO on market A
    try:
        resp_a = client.create_order(**order_a.model_dump(exclude_none=True))
        if resp_a.order.status != "executed":
            # FoK rejection is a confirmed non-fill — safe to walk away
            logging.info(
                "Leg A (NO on '%s') not filled (status=%s) — aborting pair",
                mA_title[:60], resp_a.order.status,
            )
            return TradeResult(
                spec=spec, status="failed",
                error=f"Leg A FoK not filled: status={resp_a.order.status}",
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

    # Leg A filled — submit leg B — YES on market B
    leg_b_error: str | None = None
    leg_b_ambiguous = False
    try:
        resp_b = client.create_order(**order_b.model_dump(exclude_none=True))
        if resp_b.order.status != "executed":
            leg_b_error = f"Leg B FoK not filled: status={resp_b.order.status}"
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
