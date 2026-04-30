"""Backtesting entry point: python3 -m kalshi_betting.backtest"""
import argparse
import logging
from datetime import date

from .backtester import run_backtest
from .dashboard import generate_dashboard
from .historical import build_historical_client, build_prod_live_client


def main() -> None:
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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
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
    hist_client = build_historical_client()
    live_client = build_prod_live_client()

    trades, equity_df = run_backtest(
        hist_client=hist_client,
        live_client=live_client,
        start_date=start_date,
        initial_balance=args.balance,
        use_cache=use_cache,
    )

    if not trades:
        print("No backtest trades found. Dashboard will show empty charts.")
    else:
        final_value  = float(equity_df["portfolio_value"].iloc[-1])
        total_return = (final_value - args.balance) / args.balance
        n_win        = sum(1 for t in trades if t.profit > 0)
        print(f"\nBacktest Summary")
        print(f"  Period:        {start_date} → {date.today()}")
        print(f"  Total trades:  {len(trades)}")
        print(f"  Win rate:      {n_win / len(trades):.1%}")
        print(f"  Total return:  {total_return:+.1%}")
        print(f"  Final balance: ${final_value:,.2f}")

    out = generate_dashboard(trades, equity_df, start_date, args.balance)
    print(f"\nDashboard written: {out}")
    print("Open the HTML file in a browser to view the interactive charts.")


if __name__ == "__main__":
    main()
