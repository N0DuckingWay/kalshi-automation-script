"""
File: trader.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Converts TradeSpec objects (produced by strategy.py) into Kalshi REST API
    order requests and submits them as atomic batch orders. Both legs of each
    arbitrage trade — buying NO on market A and YES on market B — are submitted
    in a single batch_create_orders call with fill_or_kill semantics, ensuring
    that a market price move never leaves an unhedged single-leg position.
    Results are returned as TradeResult objects consumed by reporter.py.

Dependencies:
    Imports TradeResult from reporter.py and TradeSpec from strategy.py. Imports
    CreateOrderRequest from the kalshi_python_sync SDK. Called by main.py after
    select_portfolio() selects the final trade list. Depends on the KalshiClient
    produced by auth.py.

Notes:
    Do NOT add retry logic to batch order submission. A failed batch indicates the
    market moved between scan time and execution time — retrying risks buying
    one leg at a worse price and creating an unhedged directional position.
"""
import logging
from typing import Any

from kalshi_python_sync.models import CreateOrderRequest

from .reporter import TradeResult
from .strategy import TradeSpec


def _build_no_order(spec: TradeSpec) -> CreateOrderRequest:
    """
    Build a market (taker) order to buy NO contracts on market A.

    The NO leg is the more expensive side of the arbitrage pair — buying NO at
    price nA means we profit when market A resolves NO (or when market B resolves
    YES first). fill_or_kill is used so the order succeeds atomically or fails
    entirely, preventing partial fills at the wrong price.

    Args:
        spec (TradeSpec): The computed trade specification. Uses spec.pair.market_a
            for the ticker and spec.x for the contract count.

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
    )


def _build_yes_order(spec: TradeSpec) -> CreateOrderRequest:
    """
    Build a market (taker) order to buy YES contracts on market B.

    The YES leg is the cheaper side of the arbitrage pair — buying YES at price
    pB means we profit when market B resolves YES. Combined with the NO leg on
    market A, all three resolution scenarios (A=YES before B, both YES, both NO)
    yield a positive return.

    Args:
        spec (TradeSpec): The computed trade specification. Uses spec.pair.market_b
            for the ticker and spec.y for the contract count.

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
    )


def execute_trades(client: Any, specs: list, dry_run: bool = False) -> list:
    """
    Execute each TradeSpec as an atomic batch order containing both trade legs.

    Both legs (NO on market A and YES on market B) are submitted together in a
    single batch_create_orders() call with fill_or_kill semantics. This is critical
    for safety: if the market price moved between scan and execution, the batch
    fails entirely rather than filling one leg and leaving an unhedged directional
    exposure. Do NOT retry or fall back to individual orders on batch failure.

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
            result has status="executed" (real order filled), "simulated" (dry run),
            or "failed" (batch submission raised an exception). Failed results
            include the error string in TradeResult.error.
    """
    results: list[TradeResult] = []

    for spec in specs:
        # Build the two order requests — NO on market A, YES on market B
        order_a = _build_no_order(spec)
        order_b = _build_yes_order(spec)

        mA_title = spec.pair.market_a.title or spec.pair.market_a.ticker
        mB_title = spec.pair.market_b.title or spec.pair.market_b.ticker

        if dry_run:
            logging.info(
                "[DRY RUN] Batch order: Buy %dx NO on '%s' @ %.2f%% | Buy %dx YES on '%s' @ %.2f%% | "
                "Total cost: $%.2f | Min profit: $%.2f",
                spec.x, mA_title[:60], spec.pair.nA * 100,
                spec.y, mB_title[:60], spec.pair.pB * 100,
                spec.total_cost, spec.min_payoff,
            )
            # Record the trade as simulated; reporter.py will include it in the dev Excel file
            results.append(TradeResult(spec=spec, status="simulated"))
            continue

        try:
            # Submit both legs atomically — batch_create_orders applies fill_or_kill
            # to the entire batch, not per-order
            client.batch_create_orders(orders=[order_a, order_b])
            logging.info(
                "Batch order submitted: '%s'  x=%d NO(A) y=%d YES(B)",
                spec.pair.canonical_title,
                spec.x,
                spec.y,
            )
            # Record the successful execution; reporter.py writes this to trade_log.xlsx
            results.append(TradeResult(spec=spec, status="executed"))
        except Exception as e:
            logging.error(
                "Batch order FAILED for '%s' (NOT retrying — partial fill risk): %s",
                spec.pair.canonical_title,
                e,
            )
            # Record the failure with the error message; reporter.py marks the row red
            results.append(TradeResult(spec=spec, status="failed", error=str(e)))

    return results
