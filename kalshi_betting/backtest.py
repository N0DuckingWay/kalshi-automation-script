"""
File: backtest.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Command-line entry point for the Kalshi arbitrage backtester. Parses CLI
    arguments (start date, initial balance, cache behavior), configures logging
    to kalshi_backtest.log, constructs the necessary API clients, delegates the
    full backtest simulation to backtester.run_backtest(), and then calls
    dashboard.generate_dashboard() to produce the interactive HTML report.
    Prints a summary of key metrics (trade count, win rate, total return) to
    the log on completion.

Dependencies:
    Imports run_backtest from backtester.py, generate_dashboard from dashboard.py,
    and build_historical_client / build_prod_live_client from historical.py.
    Imports PROJECT_ROOT from config.py. Entry point for
    `python3 -m kalshi_betting.backtest`.

Notes:
    Historical data only exists on the production Kalshi API, so both API clients
    always use prod credentials regardless of what mode the live bot was run in.
    The backtest reads market data but never submits any orders.
"""
import argparse
import logging
from datetime import date

from .backtester import run_backtest
from .config import PROJECT_ROOT
from .dashboard import generate_dashboard
from .historical import build_historical_client, build_prod_live_client


def main() -> None:
    """
    CLI entry point for the Kalshi arbitrage backtester.

    Parses command-line arguments (--start-date, --balance, --no-cache,
    --max-horizon-days), configures logging, constructs historical and live
    Kalshi API clients, runs the full backtest simulation via run_backtest(),
    and generates an interactive HTML dashboard via generate_dashboard().
    Prints a summary table of key metrics to stdout on completion.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Kalshi Arbitrage Backtester — replays the strategy on all settled "
            "Kalshi markets and writes an HTML dashboard."
        )
    )
    parser.add_argument(
        "--start-date", default="2024-01-01", metavar="YYYY-MM-DD",
        help="Earliest settlement date to include (default: 2024-01-01)",
    )
    parser.add_argument(
        "--balance", type=float, default=10_000.0, metavar="DOLLARS",
        help="Simulated starting balance in dollars (default: 10000)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Re-fetch all data from the API instead of reading the disk cache",
    )
    parser.add_argument(
        "--max-horizon-days", type=int, default=None, metavar="DAYS",
        help="Only enter trades where the later-closing leg closes within DAYS "
             "of the simulated entry checkpoint (default: no limit)",
    )
    args = parser.parse_args()
    if args.max_horizon_days is not None and args.max_horizon_days < 1:
        parser.error("--max-horizon-days must be a positive integer")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(PROJECT_ROOT / "kalshi_backtest.log"),
        ],
    )

    try:
        start_date = date.fromisoformat(args.start_date)
    except ValueError:
        parser.error(f"Invalid --start-date: {args.start_date!r}. Use YYYY-MM-DD format.")

    use_cache = not args.no_cache

    logging.info(
        "Backtest config: start=%s | balance=$%.2f | cache=%s",
        start_date, args.balance, "on" if use_cache else "off",
    )

    # Always uses prod API — historical data only exists there.
    hist_client = build_historical_client()  # authenticated prod KalshiClient for /historical raw GETs
    live_client = build_prod_live_client()  # returns KalshiClient pointed at prod for recently-settled market fetching

    trades, equity_df = run_backtest(
        hist_client=hist_client,
        live_client=live_client,
        start_date=start_date,
        initial_balance=args.balance,
        use_cache=use_cache,
        max_horizon_days=args.max_horizon_days,
    )  # returns tuple[list[BacktestTrade], pd.DataFrame] — trades and daily equity curve

    if not trades:
        logging.info("No backtest trades found. Dashboard will show empty charts.")
    else:
        final_value  = float(equity_df["portfolio_value"].iloc[-1])
        total_return = (final_value - args.balance) / args.balance
        n_win        = sum(1 for t in trades if t.profit > 0)
        logging.info("Backtest Summary")
        logging.info("  Period:        %s → %s", start_date, date.today())
        logging.info("  Total trades:  %d", len(trades))
        logging.info("  Win rate:      %.1f%%", n_win / len(trades) * 100)
        logging.info("  Total return:  %+.1f%%", total_return * 100)
        # %-style logging has no thousands-separator flag — pre-format the value
        logging.info("  Final balance: $%s", f"{final_value:,.2f}")

    out = generate_dashboard(trades, equity_df, start_date, args.balance)  # returns Path to the self-contained HTML dashboard file
    logging.info("Dashboard written: %s", out)
    logging.info("Open the HTML file in a browser to view the interactive charts.")


if __name__ == "__main__":
    main()
