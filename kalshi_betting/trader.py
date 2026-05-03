"""Order execution: build and submit batch orders for each TradeSpec."""
import logging
from typing import Any

from kalshi_python_sync.models import CreateOrderRequest

from .reporter import TradeResult
from .strategy import TradeSpec


def _build_no_order(spec: TradeSpec) -> CreateOrderRequest:
    """Buy x units of NO on market A as a market (taker) order."""
    return CreateOrderRequest(
        ticker=spec.pair.market_a.ticker,
        side="no",
        action="buy",
        type="market",
        count=spec.x,
        time_in_force="fill_or_kill",
    )


def _build_yes_order(spec: TradeSpec) -> CreateOrderRequest:
    """Buy y units of YES on market B as a market (taker) order."""
    return CreateOrderRequest(
        ticker=spec.pair.market_b.ticker,
        side="yes",
        action="buy",
        type="market",
        count=spec.y,
        time_in_force="fill_or_kill",
    )


def execute_trades(client: Any, specs: list, dry_run: bool = False) -> list:
    """
    Execute each TradeSpec as a single batch order (both legs together).

    fill_or_kill ensures atomicity: if the market moved, the whole batch
    fails rather than leaving an unhedged partial position. Do NOT fall back
    to individual orders on batch failure.

    Returns a list of TradeResult (status = "executed" | "failed").
    In dry_run mode, prints intent and returns status = "simulated".
    """
    results: list[TradeResult] = []

    for spec in specs:
        order_a = _build_no_order(spec)
        order_b = _build_yes_order(spec)

        mA_title = spec.pair.market_a.title or spec.pair.market_a.ticker
        mB_title = spec.pair.market_b.title or spec.pair.market_b.ticker

        if dry_run:
            print(
                f"[DRY RUN] Batch order:\n"
                f"  Buy {spec.x}x NO  on '{mA_title[:60]}' @ {spec.pair.nA:.2%}\n"
                f"  Buy {spec.y}x YES on '{mB_title[:60]}' @ {spec.pair.pB:.2%}\n"
                f"  Total cost: ${spec.total_cost:.2f} | Min profit: ${spec.min_payoff:.2f}"
            )
            results.append(TradeResult(spec=spec, status="simulated"))
            continue

        try:
            client.batch_create_orders(orders=[order_a, order_b])
            logging.info(
                "Batch order submitted: '%s'  x=%d NO(A) y=%d YES(B)",
                spec.pair.canonical_title,
                spec.x,
                spec.y,
            )
            results.append(TradeResult(spec=spec, status="executed"))
        except Exception as e:
            logging.error(
                "Batch order FAILED for '%s' (NOT retrying — partial fill risk): %s",
                spec.pair.canonical_title,
                e,
            )
            results.append(TradeResult(spec=spec, status="failed", error=str(e)))

    return results
