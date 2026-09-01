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
    If leg B fails, a rollback sell order is immediately submitted to unwind leg A.
    That unwind is itself a floored fill-or-kill LIMIT sell (leg A's scanned entry
    less ROLLBACK_MAX_LOSS_CENTS_PER_CONTRACT), so a collapsed book kills it rather
    than realizing an unbounded loss, and the rollback's own fill status is
    verified — an unfilled rollback is reported as status="rollback_failed"
    (orphaned position, manual review).
    Multiple pairs are executed concurrently via ThreadPoolExecutor so no pair
    waits for another to complete.

    An exception from create_order does NOT prove the order was rejected (a
    timeout can land after the fill), so exception paths consult the actual
    account position for the ticker before classifying the outcome. The check
    is a DELTA: baseline snapshots for both tickers are taken up front, before
    either order is submitted (so no blocking call sits in the unhedged window
    between leg A's fill and leg B's submission), and each is compared against
    a reading taken after the exception, so the decision reflects
    what this order did rather than what the account happens to hold (an
    unrelated pre-existing holding in the same ticker used to read as "our leg
    filled", and an unrelated absence used to read as "our leg didn't").

    pre_execution_check() re-fetches order books for each spec in the portfolio
    concurrently and drops any whose prices have moved since the scan, reducing
    the chance of submitting orders against a stale price.

Dependencies:
    Imports TradeResult from reporter.py and TradeSpec from strategy.py. Imports
    CreateOrderRequest from the kalshi_python_sync SDK and both fetch_json_page
    and api_call_with_retry from _http.py (the retry wrapper is used ONLY for
    the read-only position lookups, never for order submission). Imports
    validate_pair_price from scanner.py and BUY_MAX_COST_SLIPPAGE_CENTS,
    ROLLBACK_MAX_LOSS_CENTS_PER_CONTRACT and TRADER_MAX_WORKERS from
    config.py. Called by main.py after select_portfolio() selects the final
    trade list. Depends on the KalshiClient produced by auth.py.

Notes:
    Do NOT add retry logic to order submission. A failed leg indicates the market
    moved between scan time and execution time — retrying risks buying one leg at
    a worse price and creating an unhedged directional position. The position
    lookups in _position_count ARE retried: they are read-only GETs, so a
    transient 429 there cannot duplicate a trade — but it CAN escalate an
    otherwise-resolvable ambiguity into a rollback or manual_review.

    Order submission and position lookups use the SDK's raw-response variants
    (via _submit_order / _position_count) because 2026-07 API drift broke the
    pinned SDK's Order and MarketPosition response models — the modeled calls
    raise ValidationError AFTER submission, misclassifying real fills. The
    request side still uses the modeled CreateOrderRequest and serializes
    through the same SDK code path, so the wire format is unchanged.
"""
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from kalshi_python_sync.models import CreateOrderRequest

from ._http import api_call_with_retry, fetch_json_page
from .config import (
    BUY_MAX_COST_SLIPPAGE_CENTS,
    ROLLBACK_MAX_LOSS_CENTS_PER_CONTRACT,
    TRADER_MAX_WORKERS,
)
from .reporter import TradeResult
from .scanner import validate_pair_price
from .strategy import TradeSpec

# Tolerance for comparing position deltas against whole-contract expectations.
# Contract counts are always whole numbers on the wire, but the API now sends
# them as the `position_fp` STRING, which _position_count parses with float() —
# so an exact `delta == -spec.x` comparison would be at the mercy of decimal
# round-tripping. Any real mismatch is at least one whole contract, many orders
# of magnitude above this epsilon.
_DELTA_EPS = 1e-6


def _buy_max_cost_cents(count: int, price_dollars: float) -> int:
    """
    Compute the buy_max_cost cap in cents for one buy leg.

    The cap is the scanned depth-weighted price for the full count, rounded up
    to a whole cent, plus BUY_MAX_COST_SLIPPAGE_CENTS per contract of tolerance.
    Kalshi rejects (FoK) any fill that would cost more, so a book that moved
    against us since the pre-execution check cannot fill at a guaranteed loss.

    The product is rounded to 6 decimals before the ceiling, the same idiom as
    config.fee_leg_exact(): binary float noise (7 * 0.07 * 100 =
    49.000000000000006) would otherwise ceil to an extra cent and LOOSEN the
    cap. Removing that cent tightens price protection — the cap can only ever
    become stricter than before, never more permissive.

    Args:
        count (int): Number of contracts on this leg. >= 1.
        price_dollars (float): Scanned per-contract price in dollars (0, 1).

    Returns:
        int: Maximum total cost in cents for the order.
    """
    # Round before the ceiling so float noise can't buy the order an extra cent
    # of headroom; this TIGHTENS the cap (stricter price protection).
    return (
        math.ceil(round(count * price_dollars * 100, 6))
        + count * BUY_MAX_COST_SLIPPAGE_CENTS
    )


def _rollback_floor_cents(spec: TradeSpec) -> int:
    """
    Minimum acceptable per-contract NO sale price (cents) for a leg-A unwind.

    The unwind is a fill-or-kill LIMIT sell rather than a market sell, so a book
    that has collapsed since leg A filled kills the order instead of realizing
    an unbounded loss. The floor is leg A's scanned NO entry price less
    ROLLBACK_MAX_LOSS_CENTS_PER_CONTRACT, clamped into the API's valid limit
    price range (1..99 cents inclusive) — an entry near either extreme would
    otherwise produce a price the exchange rejects outright.

    A killed unwind leaves the leg-A position open, which _rollback_leg_a
    reports as status="rollback_failed" for manual review: the same outcome an
    unfilled market unwind already produced, now with a bounded loss instead of
    whatever the book happened to offer.

    Args:
        spec (TradeSpec): The trade being unwound. Uses spec.pair.nA, leg A's
            scanned NO entry price in dollars (0, 1).

    Returns:
        int: Limit price in cents, always within [1, 99].
    """
    # round() BEFORE int: nA is a cent-quantized book price carried as a float,
    # so int() alone would truncate 0.57 stored as 0.5699999999999998 to 56.
    entry_cents = int(round(spec.pair.nA * 100))
    return max(1, min(99, entry_cents - ROLLBACK_MAX_LOSS_CENTS_PER_CONTRACT))


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

    The returned number is the account's ABSOLUTE holding, which may include
    contracts this bot never bought (an earlier run, or a manual trade).
    Callers therefore never interpret a single reading: _execute_one snapshots
    before and after each submission and attributes only the DELTA (see
    _fill_delta). A single reading is meaningful only as a baseline.

    Uses the raw-response variant + JSON parsing because the pinned SDK's
    MarketPosition model requires legacy integer fields the API stopped
    sending in 2026-07 (the count now arrives as the `position_fp` string) —
    the modeled get_positions call raises ValidationError on any non-empty
    page, which would turn every ambiguous order into manual_review.

    The read goes through api_call_with_retry because it is a read-only GET:
    the project's no-retry rule covers order submission only, and retrying a
    GET cannot duplicate a trade. Without the retry a single transient 429 here
    reads as "state unknown" and escalates a recoverable ambiguity into a
    rollback or manual_review.

    Args:
        client (Any): Authenticated KalshiClient from auth.build_client().
        ticker (str): Market ticker to look up.

    Returns:
        float | None: Signed contract count (0 = confirmed no position; may be
            fractional because position_fp is float-parsed), or None when the
            lookup itself failed on every attempt and the state is unknown.
    """
    try:
        # Filter server-side by ticker so a single page is guaranteed to contain
        # it. Retried: read-only GET, so the no-retry rule (order submission
        # only) does not apply, and a transient 429 must not be mistaken for
        # "position unknown" — see _execute_one's ambiguity handling.
        data = api_call_with_retry(
            fetch_json_page, client.get_positions_without_preload_content, ticker=ticker
        )
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


def _fill_delta(before: float | None, after: float | None) -> float | None:
    """
    Signed position change attributable to one order, or None if unknown.

    The account's absolute position is not evidence about a specific order —
    it may already hold contracts in the same ticker from an earlier run or a
    manual trade. The change across the submission is, provided both readings
    succeeded. Either reading being None (the lookup failed after retries)
    makes the delta unknowable, and the caller must treat that as unknown
    state rather than as zero.

    Args:
        before (float | None): Signed position immediately before submission,
            or None if that lookup failed.
        after (float | None): Signed position after the ambiguous submission,
            or None if that lookup failed.

    Returns:
        float | None: after - before, or None when either snapshot is missing.
            Deltas produced by real fills are whole contracts, but callers
            compare with _DELTA_EPS because position_fp is float-parsed.
    """
    if before is None or after is None:
        return None
    return after - before


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
    _position_count in this same module IS wrapped — do not "unify" the two:
    that asymmetry is the rule, not an oversight. A GET can be repeated
    harmlessly; a state-changing POST cannot.
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

    The unwind is a FLOORED fill-or-kill LIMIT sell, not a market sell: a market
    sell has no proceeds floor (the SDK's CreateOrderRequest exposes no such
    knob), so a book that collapsed between leg A's fill and the unwind would
    realize an arbitrarily large loss. The limit price is _rollback_floor_cents()
    — leg A's scanned entry less ROLLBACK_MAX_LOSS_CENTS_PER_CONTRACT — so the
    unwind either recovers at least that much per contract or does not happen.

    reduce_only guarantees the sell can only close an existing position, so it
    is safe to submit even when leg A's fill state is ambiguous (it cannot open
    a short). The rollback's own FoK status IS checked: an unfilled rollback —
    including one killed by the price floor — means the leg-A position is still
    open, which is reported as status="rollback_failed" for manual review, never
    silently as "rolled_back". That is the same path an unfilled market unwind
    already took, so the caller's contract is unchanged; only the loss is now
    bounded.

    Args:
        client (Any): Authenticated KalshiClient from auth.build_client().
        spec (TradeSpec): The trade whose leg A must be unwound.
        reason (str): The upstream failure that triggered the rollback; recorded
            in the TradeResult error field.

    Returns:
        TradeResult: status="rolled_back" when the unwind filled,
            status="rollback_failed" when it did not fill (rejected, or killed
            by the price floor) or raised.
    """
    rollback = CreateOrderRequest(
        ticker=spec.pair.market_a.ticker,
        side="no",
        action="sell",
        # Limit, not market: only a limit order can carry a proceeds floor
        type="limit",
        # Floor the unwind: fill at >= (entry − max accepted loss) or not at
        # all. See ROLLBACK_MAX_LOSS_CENTS_PER_CONTRACT in config.py.
        no_price=_rollback_floor_cents(spec),
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
    with ThreadPoolExecutor(max_workers=min(TRADER_MAX_WORKERS, len(portfolio))) as pool:
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
    a reduce-only, price-floored fill-or-kill LIMIT sell of the leg A contracts to
    unwind the position (see _rollback_leg_a) and verifies that the rollback itself
    filled.

    A rejected FoK (status != "executed") is a confirmed non-fill. An exception,
    however, is ambiguous — the order may have filled before a timeout — so
    exception paths attribute the outcome by POSITION DELTA: baseline positions
    for BOTH tickers are read up front, before any order is submitted, and are
    compared with a reading taken after the exception. Only the change is
    evidence about this order; the absolute holding is not, because the account
    may already hold contracts in the same ticker from an earlier run or a
    manual trade. Both baselines are taken before leg A precisely so that no
    blocking network call sits between leg A's fill and leg B's submission —
    that gap is the unhedged window.

    Leg A ambiguous resolves as: delta 0 → confirmed non-fill, status="failed";
    delta of exactly -spec.x (our NO buy) → unwind via _rollback_leg_a. Anything
    else — the lookup failed, or the position moved by an amount this order
    cannot explain — is status="manual_review" with NO automated unwind: a
    reduce_only sell against a position this order may not own would liquidate
    an unrelated holding.

    Leg B ambiguous resolves as: delta of exactly +spec.y → the pair actually
    completed, status="executed"; delta 0 → confirmed non-fill, roll leg A
    back. Anything else — including an UNKNOWN state because the lookup itself
    failed — is never auto-rolled-back (an automated unwind could reverse a
    real fill we simply couldn't confirm) and is surfaced as
    status="manual_review" for a human to check the account.

    Args:
        client (Any): Authenticated KalshiClient from auth.build_client().
        spec (TradeSpec): The trade specification to execute.

    Returns:
        TradeResult: With status "executed", "failed", "rolled_back",
            "rollback_failed" (unwind did not fill — orphaned position needing
            manual review), or "manual_review" (a leg's fill state could not be
            attributed to this order and no automated action was taken).
    """
    order_a  = _build_no_order(spec)
    order_b  = _build_yes_order(spec)
    mA_title = spec.pair.market_a.title or spec.pair.market_a.ticker
    mB_title = spec.pair.market_b.title or spec.pair.market_b.ticker

    # Baselines for BOTH legs are read up front, before either order is
    # submitted, so an ambiguous outcome is judged by how the position MOVED
    # rather than by what the account happens to hold (which may predate this
    # bot entirely). Leg B's baseline is equally valid here — it reads a
    # different ticker, and no fill on market B can have happened yet — and
    # taking it now means ZERO blocking network calls sit between leg A's fill
    # and leg B's submission. Reading it after leg A filled put a retryable
    # lookup (up to ~62s of backoff) inside the window where the account holds
    # an unhedged NO position on market A.
    before_a = _position_count(client, spec.pair.market_a.ticker)
    before_b = _position_count(client, spec.pair.market_b.ticker)

    # Submit leg A — NO on market A (raw-response endpoint; see _submit_order)
    leg_a_error: str | None = None
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
        leg_a_error = str(e)

    # Disambiguation runs OUTSIDE the except block (mirroring leg B below) so
    # the position lookup is not executed while leg A's exception is still the
    # active one: anything raised in there would inherit it as __context__, and
    # api_call_with_retry walks that chain — a fatal lookup error would be
    # misread as transient and retried through the full backoff schedule
    # (~62s) before this already-urgent decision could be made.
    if leg_a_error is not None:
        # Ambiguous: the order may have filled before the exception (e.g. a
        # timeout after the fill). Attribute by delta against the baseline.
        after_a = _position_count(client, spec.pair.market_a.ticker)
        delta = _fill_delta(before_a, after_a)
        if delta is not None and abs(delta) < _DELTA_EPS:
            # Confirmed non-fill: the position did not move at all
            logging.error(
                "Leg A submission failed for '%s' (position unchanged — no fill): %s",
                mA_title[:60], leg_a_error,
            )
            return TradeResult(
                spec=spec, status="failed", error=f"Leg A error: {leg_a_error}",
            )
        if delta is not None and abs(delta + spec.x) < _DELTA_EPS:
            # Moved by exactly -spec.x: our NO buy filled (NO contracts are
            # negative by Kalshi convention). Unwind the now-unhedged leg.
            logging.error(
                "Leg A raised for '%s' but position moved by %s (our %d NO buy) —"
                " unwinding: %s",
                mA_title[:60], delta, spec.x, leg_a_error,
            )
            return _rollback_leg_a(
                client, spec, f"Leg A ambiguous error: {leg_a_error}",
            )
        # Unknown (a snapshot failed) or unattributable (the position moved by
        # an amount this order cannot explain — e.g. an unrelated trade landed
        # in the snapshot window). Do NOT auto-trade against it: a reduce_only
        # sell would liquidate a holding this order may not own.
        logging.critical(
            "Leg A raised for '%s' and the fill could NOT be attributed"
            " (position delta=%s, expected 0 or %d) — NOT unwinding, since a"
            " reduce-only sell could liquidate an unrelated position. Manual"
            " review required: %s",
            mA_title[:60], delta, -spec.x, leg_a_error,
        )
        return TradeResult(
            spec=spec, status="manual_review",
            error=f"Leg A ambiguous, delta={delta}: {leg_a_error}",
        )

    # Leg A filled — submit leg B immediately against the baseline already
    # taken above (see the up-front baseline comment): nothing blocking runs
    # between the fill and this submission.
    # Submit leg B — YES on market B (raw-response endpoint)
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
            # The exception may have arrived after the fill — attribute by delta
            # before rolling back leg A, or we'd reverse a completed hedge.
            after_b = _position_count(client, spec.pair.market_b.ticker)
            delta = _fill_delta(before_b, after_b)
            if delta is not None and abs(delta - spec.y) < _DELTA_EPS:
                # Moved by exactly +spec.y: our YES buy filled, pair complete
                logging.warning(
                    "Leg B raised for '%s' but position moved by %s (our %d YES buy)"
                    " — pair is complete: %s",
                    mB_title[:60], delta, spec.y, leg_b_error,
                )
                return TradeResult(
                    spec=spec, status="executed",
                    error=f"Leg B ambiguous but fill confirmed by position delta: {leg_b_error}",
                )
            if delta is None or abs(delta) >= _DELTA_EPS:
                # Either the lookup failed (state genuinely unknown) or the
                # position moved by an amount this order cannot explain. Rolling
                # back leg A here would be wrong if leg B actually did fill (we'd
                # sell the hedge and be left with a naked YES position on B while
                # the log says "rolled_back", implying flat). Do NOT auto-rollback;
                # surface for manual review instead.
                logging.critical(
                    "Leg B raised for '%s' and the fill could NOT be attributed"
                    " (position delta=%s, expected 0 or %d) — NOT auto-rolling-back"
                    " leg A to avoid reversing a possible real fill. Manual review"
                    " required: %s",
                    mB_title[:60], delta, spec.y, leg_b_error,
                )
                return TradeResult(
                    spec=spec, status="manual_review",
                    error=f"Leg B ambiguous, delta={delta}: {leg_b_error}",
                )
            # delta == 0 → confirmed non-fill; fall through to the rollback below
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
    then leg B only if leg A filled, with a floored-limit rollback if leg B
    fails. All specs are
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
            "failed" (leg A confirmed unfilled), "rolled_back" (leg B confirmed
            unfilled, leg A unwound), "rollback_failed" (leg A unwind did not
            fill — orphaned position), or "manual_review" (a leg's fill state
            could not be attributed to this order — no automated order was
            submitted in response).
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

    with ThreadPoolExecutor(max_workers=min(TRADER_MAX_WORKERS, len(specs))) as pool:
        futures = [pool.submit(_execute_one, client, spec) for spec in specs]
        results = [f.result() for f in futures]

    return results
