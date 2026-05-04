"""Kalshi Arbitrage Bot — main orchestration script."""
import argparse
import logging
import sys

from tabulate import tabulate

from .auth import build_client, verify_auth
from .config import MIN_BALANCE_CENTS, PROJECT_ROOT
from .reporter import append_to_prod_log, write_dev_simulation
from .scanner import (
    get_held_tickers, fetch_open_markets,
    find_candidate_pairs, find_same_title_pairs,
    enrich_with_orderbook_prices, market_title,
)
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



def _format_deadline(dt) -> str:
    """
    Format a market close datetime as a "YYYY-MM-DD" string, or "?" if None.

    Args:
        dt: A datetime object representing the market deadline, or None.

    Returns:
        str: ISO date string "YYYY-MM-DD" if dt is not None, otherwise "?".
    """
    return dt.strftime("%Y-%m-%d") if dt else "?"


def _compute_trade_specs(candidate_pairs: list, balance_cents: int) -> dict:
    """
    Compute trade specifications for all qualifying candidate pairs.

    Args:
        candidate_pairs (list): List of CandidatePair objects to evaluate.
        balance_cents (int): Current account balance in cents, used to size
            each trade via Kelly criterion in compute_trade().

    Returns:
        dict: Mapping of id(pair) -> TradeSpec for each pair that produced
            a valid trade specification. Pairs that do not meet profitability
            or size thresholds are excluded.
    """
    specs: dict = {}
    for pair in candidate_pairs:
        spec = compute_trade(pair, balance_cents)  # returns TradeSpec (Kelly-sized trade with cost/payoff/fractions) or None if pair is unprofitable
        if spec is not None:
            specs[id(pair)] = spec
    return specs


def _print_portfolio(portfolio: list, label: str) -> None:
    """
    Log a summary of selected portfolio trades to the log file.

    Args:
        portfolio (list): List of TradeSpec objects representing the trades
            selected for execution.
        label (str): Header label printed before the trade list (e.g.
            "Executing" or "Dry-run:").

    Returns:
        None
    """
    logging.info("%s %d trade(s):", label, len(portfolio))
    for spec in portfolio:
        logging.info(
            "  [%s] %s — %d× NO(A) + %d× YES(B) — "
            "cost $%.2f, min profit $%.2f (%.1f%% return)",
            spec.pair.pair_type,
            spec.pair.canonical_title[:55],
            spec.x, spec.y,
            spec.total_cost, spec.min_payoff,
            spec.profit_ratio * 100,
        )


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
    """
    Log a formatted table of all qualifying candidate pairs to the log file.

    Displays market titles, deadlines, prices, tradeability, and — for pairs
    selected in the portfolio — the computed trade size, minimum profit, monthly
    return, and Kelly fraction.

    Args:
        candidate_pairs (list): All CandidatePair objects returned by the
            scanner, regardless of whether they were selected for trading.
        display_specs (dict): Mapping of id(pair) -> TradeSpec for pairs
            selected by select_portfolio(). Pairs absent from this dict are
            shown with "—" in trade columns.

    Returns:
        None
    """
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
            _truncate(market_title(pair.market_a)),  # returns str — best available display title (.title → .subtitle → .ticker)
            _truncate(market_title(pair.market_b)),  # same
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
    table = tabulate(rows, headers=headers, tablefmt="rounded_outline")
    for line in table.splitlines():
        logging.info(line)


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
    markets = fetch_open_markets(client)  # returns list of Kalshi market API objects (all open, non-multivariate markets)
    logging.info("Sandbox markets fetched: %d", len(markets))

    # Skip held-positions filter — sandbox auth not available with production key
    time_series_pairs = find_candidate_pairs(client, held_tickers=set(), markets=markets)  # returns list[CandidatePair] — time-series pairs grouped by normalized (date-stripped) title
    same_title_pairs  = find_same_title_pairs(markets, held_tickers=set())  # returns list[CandidatePair] — pairs with identical title but different event tickers
    candidate_pairs   = _dedup_pairs(time_series_pairs, same_title_pairs)
    candidate_pairs   = enrich_with_orderbook_prices(client, candidate_pairs)  # returns list[CandidatePair] with nA/pB replaced by depth-weighted average fill prices

    if not candidate_pairs:
        logging.info("No qualifying pairs found in sandbox (≥15% time-series or ≥5% same-title price diff).")
        out = write_dev_simulation([], [], sandbox_balance_cents)  # returns Path to the newly written xlsx file
        logging.info("Dev simulation written (empty): %s", out)
        return

    trade_specs   = _compute_trade_specs(candidate_pairs, sandbox_balance_cents)
    portfolio     = select_portfolio(list(trade_specs.values()), sandbox_balance_cents)  # returns list[TradeSpec] ranked by monthly_profit_ratio desc, capped to available balance
    display_specs = {id(s.pair): s for s in portfolio}

    logging.info("Kalshi Sandbox Scan — Virtual Balance: $%.2f | Mode: DEV", args.sandbox_balance)
    print_pairs_table(candidate_pairs, display_specs)

    if not portfolio:
        logging.info("No executable arbitrage trades found.")
        out = write_dev_simulation([], candidate_pairs, sandbox_balance_cents)  # returns Path to the newly written xlsx file
        logging.info("Dev simulation written (candidates only): %s", out)
        return

    _print_portfolio(portfolio, "Simulated")

    # Simulate orders (dry_run=True always in dev)
    results = execute_trades(client, portfolio, dry_run=True)  # returns list[TradeResult], each with status "simulated"|"executed"|"failed"

    out = write_dev_simulation(results, candidate_pairs, sandbox_balance_cents)  # returns Path to the newly written xlsx file
    logging.info("Dev simulation written: %s", out)


def _run_prod(client, args) -> None:
    """
    Production mode: real auth, real balance, real orders, appends to trade_log.xlsx.
    """
    logging.warning("Running in PRODUCTION mode — real money will be used!")

    balance_cents = verify_auth(client)  # returns int — account balance in cents
    if balance_cents < MIN_BALANCE_CENTS:
        logging.warning(
            "Balance $%.2f is below minimum $%.2f — skipping run.",
            balance_cents / 100,
            MIN_BALANCE_CENTS / 100,
        )
        return

    held_tickers      = get_held_tickers(client)  # returns set[str] of tickers where the user currently holds a non-zero position
    markets           = fetch_open_markets(client)  # returns list of Kalshi market API objects (all open, non-multivariate markets)
    markets           = [m for m in markets if m.ticker not in held_tickers]

    time_series_pairs = find_candidate_pairs(client, held_tickers, markets)  # returns list[CandidatePair] — time-series pairs grouped by normalized (date-stripped) title
    same_title_pairs  = find_same_title_pairs(markets, held_tickers)  # returns list[CandidatePair] — pairs with identical title but different event tickers
    candidate_pairs   = _dedup_pairs(time_series_pairs, same_title_pairs)
    candidate_pairs   = enrich_with_orderbook_prices(client, candidate_pairs)  # returns list[CandidatePair] with nA/pB replaced by depth-weighted average fill prices

    if not candidate_pairs:
        logging.info("No qualifying pairs found (≥15% time-series or ≥5% same-title price diff).")
        return

    trade_specs   = _compute_trade_specs(candidate_pairs, balance_cents)
    portfolio     = select_portfolio(list(trade_specs.values()), balance_cents)  # returns list[TradeSpec] ranked by monthly_profit_ratio desc, capped to available balance
    display_specs = {id(s.pair): s for s in portfolio}

    logging.info("Kalshi Arbitrage Scan — Balance: $%.2f | Mode: PROD", balance_cents / 100)
    print_pairs_table(candidate_pairs, display_specs)

    if not portfolio:
        logging.info("No executable arbitrage trades found.")
        return

    _print_portfolio(portfolio, "Selected")

    results = execute_trades(client, portfolio, dry_run=args.dry_run)  # returns list[TradeResult], each with status "simulated"|"executed"|"failed"

    balance_after = verify_auth(client) / 100  # verify_auth returns int (cents); dividing by 100 converts to float dollars
    out = append_to_prod_log(results, balance_cents / 100, balance_after)  # returns Path to trade_log.xlsx (appended or created)
    logging.info("Trade log updated: %s", out)

    if args.dry_run:
        logging.info("[DRY RUN] No orders were actually submitted.")
    else:
        n_ok = sum(1 for r in results if r.status == "executed")
        logging.info("Submitted %d of %d batch order(s) successfully.", n_ok, len(results))


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
        handlers=[
            logging.FileHandler(PROJECT_ROOT / "kalshi_arb.log"),
        ],
    )

    client = build_client(args.mode)  # returns KalshiClient authenticated via RSA key from secrets.json

    if args.mode == "dev":
        _run_dev(client, args)
    else:
        _run_prod(client, args)


if __name__ == "__main__":
    main()
