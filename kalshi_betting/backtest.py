"""
File: backtest.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Command-line entry point for the Kalshi backtester. Parses CLI
    arguments (--start-date, --balance, --no-cache, --max-horizon-days,
    --interval-discount, --no-sweep, --same-event-ladders /
    --no-same-event-ladders), configures logging to
    kalshi_backtest.log, constructs the necessary API clients, delegates the
    full backtest simulation to backtester.run_backtest_sweep(), and then calls
    dashboard.generate_dashboard() to produce the interactive HTML report.
    Prints a summary of key metrics (trade count, win rate, total return) to
    the log on completion.

Dependencies:
    Imports run_backtest_sweep from backtester.py, generate_dashboard from
    dashboard.py, and build_historical_client / build_prod_live_client from
    historical.py. Imports PROJECT_ROOT from config.py. Entry point for
    `python3 -m kalshi_betting.backtest`.

Notes:
    Historical data only exists on the production Kalshi API, so both API clients
    always use prod credentials regardless of what mode the live bot was run in.
    The backtest reads market data but never submits any orders.

    --interval-discount overrides the time-series interval discount k for this
    run ONLY. It never reaches live sizing: strategy._kelly_p calls
    config.time_series_profit_prob with no override, so live trades always price
    on config.TIME_SERIES_INTERVAL_PROB_DISCOUNT. Nothing here writes config.py
    — the calibration the run reports is a recommendation for a human to act on.

    --same-event-ladders / --no-same-event-ladders (DR-73) is the same kind of
    one-run override for config.TIME_SERIES_SAME_EVENT_LADDERS, and also never
    reaches the live finder — scanner.py binds that constant at import. Unlike
    k, it changes WHICH PAIRS EXIST rather than how they are priced, so it
    applies identically to every swept discount and a run with it on is not
    comparable to a baseline taken without it. It is the way to measure the
    ladder strategy the switch gates before flipping the switch.

    That setting reaches kalshi_backtest.log ONLY — the pre-fetch echo below
    and run_backtest_sweep's resolved line. The HTML dashboard does not render
    it, so a ladder-enabled run's dashboard is indistinguishable from a
    switch-off one and must be labelled by hand. Recorded rather than fixed:
    dashboard.py is outside DR-73's blast radius, and unlike DR-66b's
    subtitle-coverage caveat this setting is chosen by the operator on the
    command line rather than discovered by the run.
"""
import argparse
import logging
import logging.handlers
from datetime import UTC, date, datetime

from .backtester import run_backtest_sweep
from .config import (
    PROJECT_ROOT,
    TIME_SERIES_INTERVAL_PROB_DISCOUNT,
    TIME_SERIES_SAME_EVENT_LADDERS,
)
from .dashboard import generate_dashboard
from .historical import build_historical_client, build_prod_live_client


def main() -> None:
    """
    CLI entry point for the Kalshi backtester.

    Parses command-line arguments (--start-date, --balance, --no-cache,
    --max-horizon-days, --interval-discount, --no-sweep), configures logging,
    constructs historical and live Kalshi API clients, runs the full backtest
    simulation via run_backtest_sweep(), and generates an interactive HTML
    dashboard via generate_dashboard(). Logs a summary table of key metrics to
    kalshi_backtest.log on completion (this module installs only a
    RotatingFileHandler, no console handler, so nothing reaches stdout).

    The summary block reports the PRIMARY point of the sweep — the run at the
    effective interval discount — so a default run's log output is identical to
    what the plain run_backtest() path produced. The other swept discounts exist
    only for the dashboard's k selector and the calibration report.
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
    parser.add_argument(
        "--interval-discount", type=float, default=None, metavar="K",
        help="Override the time-series interval discount k for this backtest "
             "(0-1; default: config.TIME_SERIES_INTERVAL_PROB_DISCOUNT). Affects "
             "the backtest only — the live sizer always reads config.py.",
    )
    parser.add_argument(
        "--no-sweep", action="store_true",
        help="Skip the k sweep; the dashboard's k selector will offer one value only",
    )
    # BooleanOptionalAction gives --same-event-ladders and
    # --no-same-event-ladders from one declaration; default=None is the "no
    # override" sentinel run_backtest_sweep resolves at call time, exactly as
    # --interval-discount's None resolves k. There is nothing to validate — the
    # action can only ever yield True, False or None.
    parser.add_argument(
        "--same-event-ladders", action=argparse.BooleanOptionalAction, default=None,
        help="Pair two dated cumulative rungs of ONE event as a time-series "
             "ladder for this backtest (default: config."
             "TIME_SERIES_SAME_EVENT_LADDERS). Affects the backtest only — the "
             "live finder binds that constant at import.",
    )
    args = parser.parse_args()
    if args.max_horizon_days is not None and args.max_horizon_days < 1:
        parser.error("--max-horizon-days must be a positive integer")
    if args.interval_discount is not None and not (0.0 <= args.interval_discount <= 1.0):
        parser.error("--interval-discount must be between 0 and 1")

    # Validated BEFORE logging is configured, alongside the other two argument
    # checks, so a rejected argument leaves no trace: parser.error exits, and
    # doing this after basicConfig created kalshi_backtest.log for a run that
    # never happened — or, worse, ROTATED it, evicting real history to record
    # nothing (TS-20). delay=True on the handler below is the belt to this
    # brace, matching main.py and scheduler.py.
    try:
        start_date = date.fromisoformat(args.start_date)
    except ValueError:
        parser.error(f"Invalid --start-date: {args.start_date!r}. Use YYYY-MM-DD format.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            # BS-25: a single unrotated FileHandler grew to 398MB during one backtest
            # sweep. 20MB/3 backups is logging infra sized for this CLI's own verbosity,
            # not a strategy constant, so it stays inline rather than in config.py.
            # TS-02: that budget only holds many runs because the per-ticker
            # candlestick failure is now ONE line (no HTTP header dump) and the
            # misses are summarized once per run by _fetch_candles_parallel.
            # Before that, 99.5% of this file was that single warning and a
            # sweep evicted ~419 MB of pre-sweep history through this rotation.
            # delay=True: the file is opened on the first emit, not at
            # construction, so a run that exits before logging anything leaves
            # no file behind. Matches main.py and scheduler.py (TS-20).
            logging.handlers.RotatingFileHandler(
                PROJECT_ROOT / "kalshi_backtest.log", maxBytes=20 * 1024 * 1024,
                backupCount=3, delay=True,
            ),
        ],
    )

    use_cache = not args.no_cache

    # Resolve --interval-discount's "no override" sentinel the same way
    # run_backtest_sweep does, purely so the echo lands BEFORE a multi-hour
    # fetch. result.primary.k below is the authoritative resolved copy and is
    # what the dashboard is handed.
    effective_k = (TIME_SERIES_INTERVAL_PROB_DISCOUNT if args.interval_discount is None
                   else args.interval_discount)
    # Same pre-fetch echo, same reason, for DR-73's switch: run_backtest_sweep
    # logs the authoritative resolved value with its source, but this line
    # lands BEFORE a multi-hour fetch so an operator can abort a run configured
    # the wrong way round.
    effective_ladders = (TIME_SERIES_SAME_EVENT_LADDERS if args.same_event_ladders is None
                         else args.same_event_ladders)

    logging.info(
        "Backtest config: start=%s | balance=$%.2f | cache=%s | k=%.3f | ladders=%s",
        start_date, args.balance, "on" if use_cache else "off", effective_k,
        "on" if effective_ladders else "off",
    )

    # Always uses prod API — historical data only exists there.
    hist_client = build_historical_client()  # authenticated prod KalshiClient for /historical raw GETs
    live_client = build_prod_live_client()  # returns KalshiClient pointed at prod for recently-settled market fetching

    result = run_backtest_sweep(
        hist_client=hist_client,
        live_client=live_client,
        start_date=start_date,
        initial_balance=args.balance,
        use_cache=use_cache,
        max_horizon_days=args.max_horizon_days,
        interval_discount=args.interval_discount,
        sweep=not args.no_sweep,
        same_event_ladders=args.same_event_ladders,
    )  # returns BacktestSweep — primary point, one point per swept k, and the calibration
    # Everything below reports the PRIMARY point, so the summary block and the
    # dashboard's other six sections read exactly as they did before the sweep
    # existed. The remaining points are consumed only by the k selector.
    trades, equity_df = result.primary.trades, result.primary.equity_df

    if not trades:
        logging.info("No backtest trades found. Dashboard will show empty charts.")
    else:
        final_value  = float(equity_df["portfolio_value"].iloc[-1])
        total_return = (final_value - args.balance) / args.balance
        n_win        = sum(1 for t in trades if t.profit > 0)
        logging.info("Backtest Summary")
        # UTC, matching the window the backtester actually simulates
        # (_prepare_candidates' feasibility end and _build_equity_curve's last
        # row are both UTC dates) — a local date here would print a period
        # the run did not cover (TS-13).
        logging.info("  Period:        %s → %s", start_date, datetime.now(UTC).date())
        logging.info("  Total trades:  %d", len(trades))
        logging.info("  Win rate:      %.1f%%", n_win / len(trades) * 100)
        logging.info("  Total return:  %+.1f%%", total_return * 100)
        # %-style logging has no thousands-separator flag — pre-format the value
        logging.info("  Final balance: $%s", f"{final_value:,.2f}")

    # generate_dashboard() already logs "Dashboard written: %s" itself (BS-26) —
    # don't duplicate that line here, just point the user at the file.
    #
    # sweep carries the calibration and every swept point for the k selector;
    # interval_discount is the resolved k these trades were sized at, which the
    # Risk section's Kelly scatter must price on
    generate_dashboard(trades, equity_df, start_date, args.balance,
                       sweep=result, interval_discount=result.primary.k)
    logging.info("Open the HTML file in a browser to view the interactive charts.")


if __name__ == "__main__":
    main()
