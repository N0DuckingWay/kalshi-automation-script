"""Kalshi Arbitrage Bot — main orchestration script."""
import argparse
import logging
import sys

from tabulate import tabulate

from .auth import build_client, verify_auth
from .config import MIN_BALANCE_CENTS
from .reporter import append_to_prod_log, write_dev_simulation
from .scanner import get_held_tickers, fetch_open_markets, find_candidate_pairs, find_same_title_pairs
from .strategy import compute_trade, select_portfolio
from .trader import execute_trades


def _truncate(text: str, n: int = 40) -> str:
    """
    Truncate text to at most n characters, appending an ellipsis if truncated.

    Args:
        text (str): Input string to truncate.
        n (int): Maximum number of characters to keep. Defaults to 40.

    Returns:
        str: The original string if len(text) <= n, otherwise text[:n] + "…".
    """
    return text[:n] + "…" if len(text) > n else text


def _market_display_title(market) -> str:
    """
    Return the best available display title for a market object.

    Prefers `.title`, falls back to `.subtitle`, then `.ticker` as a last resort.

    Args:
        market: A Kalshi market API object with `.title`, `.subtitle`, and `.ticker` attributes.

    Returns:
        str: The first non-falsy value among title, subtitle, and ticker.
    """
    return market.title or market.subtitle or market.ticker


def _format_deadline(dt) -> str:
    """
    Format a market close datetime as a "YYYY-MM-DD" string, or "?" if None.

    Args:
        dt: A datetime object representing the market deadline, or None.

    Returns:
        str: ISO date string "YYYY-MM-DD" if dt is not None, otherwise "?".
    """
    return dt.strftime("%Y-%m-%d") if dt else "?"


def _dedup_pairs(primary: list, secondary: list) -> list:
    """
    Combine two pair lists, dropping duplicates from secondary.
    A duplicate is any pair whose {ticker_a, ticker_b} frozenset already appears
    in primary (time_series is preferred over same_title when both detect the same pair).
    """
    seen: set = set()
    result = []
    for pair in primary:
        key = frozenset([pair.market_a.ticker, pair.market_b.ticker])
        seen.add(key)
        result.append(pair)
    for pair in secondary:
        key = frozenset([pair.market_a.ticker, pair.market_b.ticker])
        if key not in seen:
            seen.add(key)
            result.append(pair)
    return result


def print_pairs_table(candidate_pairs: list, display_specs: dict) -> None:
    """Print all qualifying pairs. display_specs maps id(pair) -> TradeSpec for selected trades."""
    rows = []
    for pair in candidate_pairs:
        spec = display_specs.get(id(pair))
        if spec:
            trade_str   = f"{spec.x}× NO(A) + {spec.y}× YES(B)"
            profit_str  = f"${spec.min_payoff:.2f}"
            monthly_str = f"{spec.monthly_profit_ratio:.2%}/mo"
            kelly_str   = f"{spec.kelly_fraction:.1%} (p={spec.kelly_p:.2f})"
        else:
            trade_str   = "—"
            profit_str  = "—"
            monthly_str = "—"
            kelly_str   = "—"

        rows.append([
            pair.pair_type,
            _truncate(_market_display_title(pair.market_a)),
            _truncate(_market_display_title(pair.market_b)),
            _format_deadline(pair.market_a.close_time),
            _format_deadline(pair.market_b.close_time),
            f"{pair.pA:.2%}",
            f"{pair.pB:.2%}",
            "YES ✓" if pair.tradeable else "no",
            trade_str,
            profit_str,
            monthly_str,
            kelly_str,
        ])

    headers = [
        "Type",
        "Market A", "Market B",
        "A Deadline", "B Deadline",
        "pA (YES)", "pB (YES)",
        "Tradeable?", "Recommended Trade", "Min Profit", "Monthly Return", "Kelly",
    ]
    print("\n" + tabulate(rows, headers=headers, tablefmt="rounded_outline"))


def _run_dev(client, args) -> None:
    """
    Dev/sandbox mode:
      - Fetches real market data from the Kalshi sandbox (public endpoint, no auth needed).
      - Skips held-positions check and balance lookup (sandbox requires separate credentials).
      - Uses --sandbox-balance as the virtual account size for trade sizing.
      - Never submits real orders; always writes a simulation Excel file.
    """
    sandbox_balance_cents = int(args.sandbox_balance * 100)
    logging.info(
        "DEV mode: using real sandbox market data | virtual balance $%.2f",
        args.sandbox_balance,
    )

    # Fetch real sandbox markets (public endpoint — no auth required)
    markets = fetch_open_markets(client)
    logging.info("Sandbox markets fetched: %d", len(markets))

    # Skip held-positions filter — sandbox auth not available with production key
    time_series_pairs = find_candidate_pairs(client, held_tickers=set(), markets=markets)
    same_title_pairs  = find_same_title_pairs(markets, held_tickers=set())
    candidate_pairs   = _dedup_pairs(time_series_pairs, same_title_pairs)

    if not candidate_pairs:
        print("No qualifying pairs found in sandbox (≥15% time-series or ≥5% same-title price diff).")
        out = write_dev_simulation([], [], sandbox_balance_cents)
        print(f"Dev simulation written (empty): {out}")
        return

    trade_specs: dict = {}
    for pair in candidate_pairs:
        spec = compute_trade(pair, sandbox_balance_cents)
        if spec is not None:
            trade_specs[id(pair)] = spec

    portfolio    = select_portfolio(list(trade_specs.values()), sandbox_balance_cents)
    display_specs = {id(s.pair): s for s in portfolio}

    print(f"\nKalshi Sandbox Scan — Virtual Balance: ${args.sandbox_balance:.2f} | Mode: DEV")
    print_pairs_table(candidate_pairs, display_specs)

    if not portfolio:
        print("\nNo executable arbitrage trades found.")
        out = write_dev_simulation([], candidate_pairs, sandbox_balance_cents)
        print(f"\nDev simulation written (candidates only): {out}")
        return

    print(f"\nSimulated {len(portfolio)} trade(s):")
    for spec in portfolio:
        print(
            f"  [{spec.pair.pair_type}] {spec.pair.canonical_title[:55]} — "
            f"{spec.x}× NO(A) + {spec.y}× YES(B) — "
            f"cost ${spec.total_cost:.2f}, min profit ${spec.min_payoff:.2f} "
            f"({spec.profit_ratio:.1%} return)"
        )

    # Simulate orders (dry_run=True always in dev)
    results = execute_trades(client, portfolio, dry_run=True)

    out = write_dev_simulation(results, candidate_pairs, sandbox_balance_cents)
    print(f"\nDev simulation written: {out}")


def _run_prod(client, args) -> None:
    """
    Production mode: real auth, real balance, real orders, appends to trade_log.xlsx.
    """
    logging.warning("Running in PRODUCTION mode — real money will be used!")

    balance_cents = verify_auth(client)
    if balance_cents < MIN_BALANCE_CENTS:
        logging.warning(
            "Balance $%.2f is below minimum $%.2f — skipping run.",
            balance_cents / 100,
            MIN_BALANCE_CENTS / 100,
        )
        return

    held_tickers      = get_held_tickers(client)
    markets           = fetch_open_markets(client)
    markets           = [m for m in markets if m.ticker not in held_tickers]

    time_series_pairs = find_candidate_pairs(client, held_tickers, markets)
    same_title_pairs  = find_same_title_pairs(markets, held_tickers)
    candidate_pairs   = _dedup_pairs(time_series_pairs, same_title_pairs)

    if not candidate_pairs:
        print("No qualifying pairs found (≥15% time-series or ≥5% same-title price diff).")
        return

    trade_specs: dict = {}
    for pair in candidate_pairs:
        spec = compute_trade(pair, balance_cents)
        if spec is not None:
            trade_specs[id(pair)] = spec

    portfolio    = select_portfolio(list(trade_specs.values()), balance_cents)
    display_specs = {id(s.pair): s for s in portfolio}

    print(f"\nKalshi Arbitrage Scan — Balance: ${balance_cents / 100:.2f} | Mode: PROD")
    print_pairs_table(candidate_pairs, display_specs)

    if not portfolio:
        print("\nNo executable arbitrage trades found.")
        return

    print(f"\nSelected {len(portfolio)} trade(s):")
    for spec in portfolio:
        print(
            f"  [{spec.pair.pair_type}] {spec.pair.canonical_title[:55]} — "
            f"{spec.x}× NO(A) + {spec.y}× YES(B) — "
            f"cost ${spec.total_cost:.2f}, min profit ${spec.min_payoff:.2f} "
            f"({spec.profit_ratio:.1%} return)"
        )

    results = execute_trades(client, portfolio, dry_run=args.dry_run)

    balance_after = verify_auth(client) / 100
    out = append_to_prod_log(results, balance_cents / 100, balance_after)
    print(f"\nTrade log updated: {out}")

    if args.dry_run:
        print("[DRY RUN] No orders were actually submitted.")
    else:
        n_ok = sum(1 for r in results if r.status == "executed")
        print(f"Submitted {n_ok} of {len(results)} batch order(s) successfully.")


def main() -> None:
    """
    CLI entry point for the Kalshi arbitrage bot.

    Parses command-line arguments (--mode, --dry-run, --sandbox-balance),
    configures logging, builds the appropriate Kalshi client, and dispatches
    to _run_dev (sandbox simulation) or _run_prod (real account trading).
    """
    parser = argparse.ArgumentParser(description="Kalshi Arbitrage Bot")
    parser.add_argument(
        "--mode", choices=["dev", "prod"], default="dev",
        help="'dev' scans real sandbox markets and simulates; 'prod' uses real account",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="(prod only) Discover and size trades but do not submit orders",
    )
    parser.add_argument(
        "--sandbox-balance", type=float, default=1000.0, metavar="DOLLARS",
        help="Virtual balance in dollars used for trade sizing in dev mode (default: 1000)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    client = build_client(args.mode)

    if args.mode == "dev":
        _run_dev(client, args)
    else:
        _run_prod(client, args)


if __name__ == "__main__":
    main()
