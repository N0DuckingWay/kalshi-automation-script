"""
File: main.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Top-level orchestration for the Kalshi arbitrage bot's live trading pipeline.
    Parses command-line arguments to select dev (sandbox simulation) or prod
    (real-money trading) mode, then coordinates the full scan-size-execute-log
    cycle: building an authenticated API client, fetching open markets, finding
    arbitrage candidate pairs, sizing trades via Kelly criterion, submitting batch
    orders to the Kalshi REST API, and writing results to Excel. This is the only
    module that ties all other modules together in the live trading path. The
    process exit code communicates the run's outcome to the scheduler (a
    separate subprocess) — see the EXIT_* constants in config.py (BS-14): an
    unhandled exception still propagates to exit 1, same as always.

Dependencies:
    Imports from auth.py (client construction and auth verification), config.py
    (balance threshold, exit-code contract, and file paths), reporter.py (Excel
    output), scanner.py (market fetching and pair detection), strategy.py (trade
    sizing and portfolio selection), and trader.py (order execution). Entry
    point for `python3 -m kalshi_betting.main`.
"""
import argparse
import logging
import sys

from tabulate import tabulate

from .auth import build_client, verify_auth
from .config import (
    EXIT_OK,
    EXIT_SKIPPED_LOW_BALANCE,
    EXIT_TRADES_NEED_ATTENTION,
    MIN_BALANCE_CENTS,
    PROJECT_ROOT,
)
from .reporter import append_to_prod_log, write_dev_simulation
from .scanner import (
    display_title,
    enrich_with_orderbook_prices,
    fetch_open_events_with_markets,
    filter_markets_within_horizon,
    find_same_title_pairs,
    find_time_series_pairs,
    get_held_tickers,
)
from .strategy import compute_trade, select_portfolio
from .trader import execute_trades, pre_execution_check


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
    Merge two pair lists, excluding any pair from secondary that already appears in primary.

    A duplicate is defined as any pair whose frozenset of {ticker_a, ticker_b} already
    exists in primary. This can occur when both the same-title scanner and the time-series
    scanner detect the same two markets — in that case the same-title pair is preferred
    because the co-resolution guarantee is simpler (identical questions must co-resolve)
    and does not depend on an independence-model probability estimate. This matches the
    same_title > time_series tie-break already used by strategy.select_portfolio().

    Args:
        primary (list): List of CandidatePair objects from find_same_title_pairs().
            These are always kept.
        secondary (list): List of CandidatePair objects from find_time_series_pairs().
            Entries whose ticker pair already appears in primary are dropped.

    Returns:
        list: Combined list with primary entries first, then non-duplicate secondary
            entries appended in their original order.
    """
    seen: set = set()
    result = []
    for pair in primary:
        key = frozenset([pair.market_a.ticker, pair.market_b.ticker])
        seen.add(key)
        result.append(pair)
    for pair in secondary:
        key = frozenset([pair.market_a.ticker, pair.market_b.ticker])
        # Only add if this exact ticker pair was not already found by the
        # same-title scanner (the primary list, which is kept on conflict)
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
            # display_title prefixes the event title for MVE markets so multi-choice
            # option labels (e.g. "Trump", "Above $80k") carry their event context
            _truncate(display_title(pair.market_a)),
            _truncate(display_title(pair.market_b)),
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


def _run_dev(client, args) -> int:
    """
    Execute a full dev/sandbox mode scan and simulation.

    Dev mode uses real sandbox market data from demo-api.kalshi.co but never
    submits real orders. The held-positions check and real balance lookup are
    skipped because the production API key is not accepted by the sandbox
    endpoint. Instead, a virtual balance is supplied via --sandbox-balance.
    All results are written to a timestamped dev simulation Excel file.

    Args:
        client: KalshiClient pointed at the sandbox endpoint, produced by
            auth.build_client("dev").
        args: Parsed argparse Namespace with sandbox_balance and
            max_horizon_days attributes.

    Returns:
        int: Always EXIT_OK — dev mode never submits real orders, so there is
            no low-balance skip or manual-review outcome to distinguish.
            Returned as an int (rather than None) for symmetry with _run_prod,
            since main() dispatches to either and passes the result to
            sys.exit().
    """
    sandbox_balance_cents = int(args.sandbox_balance * 100)
    logging.info(
        "DEV mode: using real sandbox market data | virtual balance $%.2f",
        args.sandbox_balance,
    )

    # Fetch all open sandbox markets — the sandbox public endpoint does not require
    # valid authentication for read operations, so this works with the prod key too
    markets = fetch_open_events_with_markets(client)
    logging.info("Sandbox markets fetched: %d", len(markets))

    # Optional opt-in cap so both bet types only see markets closing within
    # the requested window — a no-op (returns markets unchanged) when unset
    markets = filter_markets_within_horizon(markets, args.max_horizon_days)

    # Skip held-positions filter — sandbox requires a separate account and credentials.
    # Pass an empty set so _filter_active_markets does not exclude any tickers.
    time_series_pairs = find_time_series_pairs(client, held_tickers=set(), markets=markets)
    # Detect same-title pairs separately — uses a different grouping key (exact title match)
    same_title_pairs  = find_same_title_pairs(markets, held_tickers=set())
    # Merge both lists, preferring same_title when both scanners found the same pair
    candidate_pairs   = _dedup_pairs(same_title_pairs, time_series_pairs)
    # Replace best-ask prices with depth-weighted order book averages to validate liquidity
    candidate_pairs   = enrich_with_orderbook_prices(client, candidate_pairs)

    if not candidate_pairs:
        logging.info("No qualifying pairs found in sandbox (≥15%/30% deadline-gap-tiered time-series or ≥5% same-title price diff).")
        # Write an empty simulation file so the run is still recorded
        out = write_dev_simulation([], [], sandbox_balance_cents)
        logging.info("Dev simulation written (empty): %s", out)
        return EXIT_OK

    # Apply Kelly sizing to each candidate pair using the virtual balance
    trade_specs   = _compute_trade_specs(candidate_pairs, sandbox_balance_cents)
    # Greedy portfolio selection ranked by monthly_profit_ratio descending
    portfolio     = select_portfolio(list(trade_specs.values()), sandbox_balance_cents)
    # Map pair id → TradeSpec for fast lookup in the pairs table display
    display_specs = {id(s.pair): s for s in portfolio}

    logging.info("Kalshi Sandbox Scan — Virtual Balance: $%.2f | Mode: DEV", args.sandbox_balance)
    print_pairs_table(candidate_pairs, display_specs)

    if not portfolio:
        logging.info("No executable arbitrage trades found.")
        # Write a simulation file showing candidates even though no trades were sized
        out = write_dev_simulation([], candidate_pairs, sandbox_balance_cents)
        logging.info("Dev simulation written (candidates only): %s", out)
        return EXIT_OK

    _print_portfolio(portfolio, "Simulated")

    # Simulate order execution (dry_run=True is always enforced in dev mode)
    # Returns TradeResult objects with status="simulated" — no API orders are placed
    results = execute_trades(client, portfolio, dry_run=True)

    # Write the simulation Excel file: Sheet 1 = simulated trades, Sheet 2 = all candidates
    out = write_dev_simulation(results, candidate_pairs, sandbox_balance_cents)
    logging.info("Dev simulation written: %s", out)
    return EXIT_OK


def _run_prod(client, args) -> int:
    """
    Execute a full production run using the real Kalshi account.

    Verifies authentication, reads the live account balance, fetches currently
    held positions to exclude them from scanning, discovers arbitrage candidates,
    sizes trades using Kelly criterion, and submits batch orders. Results are
    appended to the persistent trade_log.xlsx file. In dry_run mode (--dry-run
    flag), all steps run normally except order submission — the log still records
    rows with status="simulated".

    Args:
        client: KalshiClient pointed at the production endpoint, produced by
            auth.build_client("prod").
        args: Parsed argparse Namespace with dry_run and max_horizon_days
            attributes.

    Returns:
        int: EXIT_SKIPPED_LOW_BALANCE if the run was skipped because the
            account balance is below MIN_BALANCE_CENTS (no scan attempted).
            EXIT_TRADES_NEED_ATTENTION if any TradeResult in this run's
            results has status "rollback_failed" or "manual_review" — either
            means a human must check the account/trade log. EXIT_OK for every
            other path, including dry-run, no candidate pairs, no executable
            trades, and all-pairs-failed-pre-execution-check.
    """
    logging.warning("Running in PRODUCTION mode — real money will be used!")

    # Confirm auth works and read the pre-trade balance in cents
    balance_cents = verify_auth(client)
    if balance_cents < MIN_BALANCE_CENTS:
        # Don't waste API calls scanning when there's insufficient capital to trade
        logging.warning(
            "Balance $%.2f is below minimum $%.2f — skipping run.",
            balance_cents / 100,
            MIN_BALANCE_CENTS / 100,
        )
        return EXIT_SKIPPED_LOW_BALANCE

    # Get current open positions so we don't re-enter markets we already hold
    held_tickers      = get_held_tickers(client)
    # Fetch all open markets and pre-filter held tickers for downstream efficiency
    markets           = fetch_open_events_with_markets(client)
    markets           = [m for m in markets if m.ticker not in held_tickers]

    # Optional opt-in cap so both bet types only see markets closing within
    # the requested window — a no-op (returns markets unchanged) when unset
    markets           = filter_markets_within_horizon(markets, args.max_horizon_days)

    # Run both pair detection paths: time-series (deadline-gap) and same-title
    time_series_pairs = find_time_series_pairs(client, held_tickers, markets)
    same_title_pairs  = find_same_title_pairs(markets, held_tickers)
    # Merge both lists, preferring same_title when both scanners found the same pair
    candidate_pairs   = _dedup_pairs(same_title_pairs, time_series_pairs)
    # Replace best-ask prices with depth-weighted order book averages to validate liquidity
    candidate_pairs   = enrich_with_orderbook_prices(client, candidate_pairs)

    if not candidate_pairs:
        logging.info("No qualifying pairs found (≥15%/30% deadline-gap-tiered time-series or ≥5% same-title price diff).")
        return EXIT_OK

    # Apply Kelly sizing to each candidate pair using the real account balance
    trade_specs   = _compute_trade_specs(candidate_pairs, balance_cents)
    # Greedy portfolio selection ranked by monthly_profit_ratio descending
    portfolio     = select_portfolio(list(trade_specs.values()), balance_cents)
    # Map pair id → TradeSpec for fast lookup in the pairs table display
    display_specs = {id(s.pair): s for s in portfolio}

    logging.info("Kalshi Arbitrage Scan — Balance: $%.2f | Mode: PROD", balance_cents / 100)
    print_pairs_table(candidate_pairs, display_specs)

    if not portfolio:
        logging.info("No executable arbitrage trades found.")
        return EXIT_OK

    _print_portfolio(portfolio, "Selected")

    # Re-fetch order books for each pair concurrently and drop any whose prices moved
    portfolio = pre_execution_check(client, portfolio)
    if not portfolio:
        logging.info("All selected pairs failed pre-execution price check — no trades submitted.")
        return EXIT_OK

    # Submit orders sequentially per leg, concurrently across pairs
    results = execute_trades(client, portfolio, dry_run=args.dry_run)

    # Read the post-trade balance for the Excel log separator row. Real orders
    # may already have filled at this point, so a failure here must not lose the
    # trade records — fall back to the pre-trade balance and keep going.
    try:
        balance_after = verify_auth(client) / 100
    except Exception as exc:
        logging.error(
            "Post-trade balance fetch failed: %s — logging with pre-trade balance", exc,
        )
        balance_after = balance_cents / 100

    # Append this run's results to the cumulative trade_log.xlsx file. If the
    # write fails (e.g. the file is open in Excel), dump every result to the log
    # so the record of real fills is never lost, then re-raise.
    try:
        out = append_to_prod_log(results, balance_cents / 100, balance_after)
        logging.info("Trade log updated: %s", out)
    except Exception as exc:
        logging.critical("Failed to write trade log: %s — rescue dump follows", exc)
        for r in results:
            logging.critical(
                "  RESCUE | %s | %s | A=%s B=%s | x=%d cost=$%.2f | %s",
                r.status,
                r.spec.pair.canonical_title,
                r.spec.pair.market_a.ticker,
                r.spec.pair.market_b.ticker,
                r.spec.x,
                r.spec.total_cost,
                r.error or "",
            )
        raise

    if args.dry_run:
        logging.info("[DRY RUN] No orders were actually submitted.")
        return EXIT_OK

    n_ok       = sum(1 for r in results if r.status == "executed")
    n_rolled   = sum(1 for r in results if r.status == "rolled_back")
    n_orphaned = sum(1 for r in results if r.status == "rollback_failed")
    # "manual_review" means leg B's fill state was undetermined and no
    # automated rollback was attempted — just as urgent as an orphaned
    # rollback failure, so it's counted in the same manual-review alert.
    n_unknown  = sum(1 for r in results if r.status == "manual_review")
    logging.info(
        "Submitted %d of %d order pair(s) successfully. %d rolled back, "
        "%d rollback failure(s), %d unknown fill state(s).",
        n_ok, len(results), n_rolled, n_orphaned, n_unknown,
    )
    if n_orphaned or n_unknown:
        logging.critical(
            "%d pair(s) may have ORPHANED positions and %d pair(s) have an "
            "UNDETERMINED fill state — manual review required (see trade log).",
            n_orphaned, n_unknown,
        )
        # Surface via the process exit code too (BS-14) — the scheduler reads
        # this, not the log file, and needs to distinguish "trades happened
        # but need a human to check" from a clean run.
        return EXIT_TRADES_NEED_ATTENTION

    return EXIT_OK


def main() -> None:
    """
    CLI entry point for the Kalshi arbitrage bot.

    Parses command-line arguments (--mode, --dry-run, --sandbox-balance,
    --max-horizon-days), configures logging, builds the appropriate Kalshi
    client, and dispatches to _run_dev (sandbox simulation) or _run_prod
    (real account trading). Exits the process via sys.exit() with the
    dispatched run's return code (see the EXIT_* constants in config.py,
    BS-14) so a caller that only sees the process exit status — the
    scheduler, which runs this as a subprocess — can distinguish a clean run
    from a low-balance skip or a run with trades needing manual review. An
    unhandled exception is not caught here and propagates to the normal
    interpreter exit code 1.

    Returns:
        None: This function never returns to its caller — it always ends by
            calling sys.exit(code), which raises SystemExit.
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
    parser.add_argument(
        "--max-horizon-days", type=int, default=None, metavar="DAYS",
        help="Only consider markets closing within DAYS from now (both modes; default: no limit)",
    )
    args = parser.parse_args()
    if args.max_horizon_days is not None and args.max_horizon_days < 1:
        parser.error("--max-horizon-days must be a positive integer")

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
        code = _run_dev(client, args)
    else:
        code = _run_prod(client, args)

    # Only sys.exit() communicates the outcome to a subprocess caller (the
    # scheduler) — a bare return here would always look like exit 0.
    sys.exit(code)


if __name__ == "__main__":
    main()
