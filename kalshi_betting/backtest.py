"""
File: backtest.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    The command line for the backtester (`python3 -m kalshi_betting.backtest`).
    It reads the options, works out the starting balance, replays the
    strategy on past Kalshi markets (backtester.run_backtest_sweep), writes
    the HTML dashboard (dashboard.generate_dashboard) and logs a short
    summary to kalshi_backtest.log.

Dependencies:
    backtester (runs the backtest), dashboard (writes the page), historical
    (the API clients and Kalshi's list of series and their categories),
    auth (reads the account's balance), depth_model (the model of how deep
    order books are), treasury (T-bill yields for the Sharpe and Sortino
    ratios), _http (one-line error text) and config (settings and checks).
    Nothing in the package imports this module.

Notes:
    - It reads production data only, since past markets exist only there.
      It never places an order.
    - Starting balance: --balance, or else what the account is worth now
      (its cash plus Kalshi's value of its open positions). If that cannot
      be read, the run stops (exit 1) rather than guess an amount.
    - --sell-workers starts worker processes that re-import this module, so
      main() must only run under the `if __name__ == "__main__"` guard.
    - With no saved order-book snapshots, every trade fills at the best
      price in any size; the "Backtest config" line says which.
    - --interval-discount, --same-event-ladders and --spread-min/--spread-max
      change this backtest only. Nothing here changes live trading: the
      live settings change only when someone saves a scenario from the
      dashboard (run ./start_dashboard.sh, then "Save as live defaults…").
    - The dashboard's extra views are all on by default. The spread-band
      grid (--no-band-sweep) is simulated during the run; the size caps,
      adding to held pairs, selling, trimming and its minimum price rises
      (--no-cap-sweep, --no-add-on-sweep, --no-sell-sweep, --no-trim-sweep,
      --no-trim-rise-sweep) are simulated only while the dashboard is built.
      Each --no-… option skips its view, making the page smaller and faster.
"""
import argparse
import logging
import logging.handlers
import math
import os
from dataclasses import dataclass
from datetime import UTC, date, datetime

from ._http import api_error_summary
from .auth import read_account_balance
from .backtester import BacktestSweep, run_backtest_sweep
from .config import (
    DASHBOARD_SELL_MAX_WORKERS,
    MAX_DEADLINE_GAP_DAYS,
    MIN_BALANCE_CENTS,
    MIN_PRICE_DIFF_LONG_GAP,
    MIN_PRICE_DIFF_SHORT_GAP,
    PRICE_EPSILON,
    PROJECT_ROOT,
    SHORT_DEADLINE_GAP_DAYS,
    TIME_SERIES_INTERVAL_PROB_DISCOUNT,
    TIME_SERIES_SAME_EVENT_LADDERS,
    LiveDefaultsError,
    LiveDefaultsMissing,
    describe_time_series_rule,
    describe_trade_filter,
    live_defaults,
    time_series_spread_band,
    time_series_spread_too_wide,
)
from .dashboard import generate_dashboard
from .depth_model import DepthModel, load_depth_model
from .historical import build_historical_client, build_prod_live_client, load_series_categories
from .treasury import load_risk_free_rates


@dataclass(frozen=True)
class StartingBalance:
    """
    The amount a backtest starts with, all of it cash, and where it came from.

    Trade sizes and every return on the dashboard are measured from this
    amount. The source is shown beside it in the log and the dashboard.

    Attributes:
        dollars (float): The starting balance in dollars; a finite amount above 0.
        source (str): A short note saying where the amount came from.

    Raises:
        ValueError: If dollars is not a finite amount above 0.
    """
    dollars: float
    source: str

    def __post_init__(self) -> None:
        # Returns are measured against this amount, so it must be a real amount above 0
        if not (math.isfinite(self.dollars) and self.dollars > 0):
            raise ValueError(
                f"a starting balance must be a finite amount above 0, not {self.dollars!r}")


class StartingBalanceError(Exception):
    """
    The account gave no amount a backtest can start from; the message says why.

    main() stops the run when this is raised (exit 1) and suggests --balance.
    """


def _account_starting_balance(client) -> StartingBalance:
    """
    Read what the Kalshi account is worth now, the starting balance when --balance is not given.

    The amount is the account's cash plus what Kalshi says its open positions
    are worth, the same figure a live run sizes its trades on. If Kalshi does
    not report the positions' value, the cash alone is used, with a WARNING.
    Makes one read-only request, retried on temporary errors.

    Args:
        client: A production KalshiClient, as historical.build_prod_live_client() makes.

    Returns:
        StartingBalance: The amount in dollars, with a note saying when it was
            read and whether it includes the open positions.

    Raises:
        StartingBalanceError: If the balance cannot be read, or it comes to
            nothing (an account worth $0, or no cash and an unknown
            positions value).
    """
    try:
        # One retried read: the cash on each shard and Kalshi's value of the open positions
        account = read_account_balance(client)
    except Exception as e:
        # Describe the failure in one short line rather than the SDK's long error text
        raise StartingBalanceError(
            f"could not read the account balance ({api_error_summary(e)})") from e
    cash_cents = sum(account.shard_cash_cents.values())
    positions_cents = account.positions_value_cents
    read_at = f"{datetime.now(UTC):%Y-%m-%d %H:%M} UTC"
    if positions_cents is None:
        if cash_cents <= 0:
            # No cash, and positions of unknown worth: nothing safe to start from
            raise StartingBalanceError(
                f"the account's cash is ${cash_cents / 100:,.2f} and its open positions' "
                "value could not be read, so there is nothing to start from")
        logging.warning(
            "Kalshi's balance reply carried no readable portfolio_value — the backtest "
            "starts from the account's cash alone ($%s), as if no position were held",
            f"{cash_cents / 100:,.2f}",
        )
        return StartingBalance(
            cash_cents / 100,
            f"the account's cash at {read_at}; its open positions' value could not be read",
        )
    if cash_cents + positions_cents <= 0:
        raise StartingBalanceError(
            f"the account is worth ${(cash_cents + positions_cents) / 100:,.2f}, so there "
            "is nothing to start from")
    return StartingBalance(
        (cash_cents + positions_cents) / 100,
        f"the account's value at {read_at}: cash ${cash_cents / 100:,.2f} + open "
        f"positions ${positions_cents / 100:,.2f}",
    )


def _fills_echo(depth_model: DepthModel | None) -> str:
    """
    Say how the run's trades will fill, for the pre-fetch config line.

    Args:
        depth_model (DepthModel | None): The depth table load_depth_model
            fitted, or None when there is none.

    Returns:
        str: "walked book (S snapshots, L ladders)" with a model, otherwise
            "top of book (no usable depth snapshot)". The log line from
            load_depth_model says why there is none.
    """
    if depth_model is None:
        return "top of book (no usable depth snapshot)"
    snapshots, ladders = depth_model.snapshots, depth_model.ladders
    return (f"walked book ({snapshots} snapshot{'' if snapshots == 1 else 's'}, "
            f"{ladders} ladder{'' if ladders == 1 else 's'})")


def _log_corpus_provenance(sweep: BacktestSweep) -> None:
    """
    Log, as the run's last report line, which past-market data it used.

    The data holds no market settled after it was put together, so this says
    when that was and whether this run built it, extended it or reused a
    saved copy. It is logged on every run, even when nothing is known, so a
    missing line is never mistaken for good news.

    Args:
        sweep (BacktestSweep): The run's result. Its corpus_provenance is None
            when not recorded (no corpus was fetched, or it did not come from
            an assembled cache).
    """
    provenance = sweep.corpus_provenance
    if provenance is None:
        logging.info(
            "Settled-market corpus: assembly time and archive cutoff not recorded "
            "(no corpus was fetched, or it did not come from an assembled cache)"
        )
        return
    if provenance.assembled_at is None:
        assembled = "assembly time not recorded"
    else:
        assembled = (f"assembled {provenance.assembled_at:%Y-%m-%d %H:%M} UTC, "
                     "holding no market settled after that")
    if provenance.full_assembly_at is not None:
        assembled += (f" (extended day by day since a full assembly of "
                      f"{provenance.full_assembly_at:%Y-%m-%d %H:%M} UTC)")
    if provenance.stale:
        source = ("served as an earlier day's cache: it could not be brought up to "
                  "date this run (see the WARNING above); --no-cache re-assembles it "
                  "in full")
    elif provenance.from_cache:
        source = ("served from an earlier run's cache, assembled earlier today; "
                  "--no-cache re-assembles it in full")
    elif provenance.full_assembly_at is not None:
        source = "extended through today by this run"
    else:
        source = "assembled by this run"
    cutoff = ("not recorded" if provenance.archive_cutoff is None
              else f"{provenance.archive_cutoff:%Y-%m-%d}")
    logging.info(
        "Settled-market corpus: %s (%s); archive cutoff at assembly: %s",
        assembled, source, cutoff,
    )


def main() -> None:
    """
    Run the backtester from the command line.

    In order: read and check the options (a bad one stops the run before the
    log file is touched), set up logging to kalshi_backtest.log (there is no
    console output), work out the starting balance, run the backtest, log a
    summary and write the HTML dashboard.

    The summary reports the run's main scenario: the chosen k, spread band and
    size cap. The other scenarios the run works out are shown on the dashboard.
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
    # None means "read the account's value", which is done later, once a client exists
    parser.add_argument(
        "--balance", type=float, default=None, metavar="DOLLARS",
        help="Simulated starting balance in dollars (default: what the Kalshi "
             "account is worth when the run starts — its cash plus Kalshi's value "
             "of its open positions, read from the production account; the run "
             "stops if that cannot be read or comes to nothing)",
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
             "(above 0, at most 1; default: config.TIME_SERIES_INTERVAL_PROB_DISCOUNT). Affects "
             "this backtest only — the live sizer reads the saved live defaults' k "
             "unless main.py's own --interval-discount overrides it for one live run.",
    )
    parser.add_argument(
        "--no-sweep", action="store_true",
        help="Skip the k sweep; with the band sweep on, each band is "
             "simulated at the primary k only",
    )
    # Gives both --same-event-ladders and --no-same-event-ladders. None (neither
    # given) means "use the config setting"
    parser.add_argument(
        "--same-event-ladders", action=argparse.BooleanOptionalAction, default=None,
        help="Pair two dated cumulative rungs of ONE event as a time-series "
             "ladder for this backtest (default: config."
             "TIME_SERIES_SAME_EVENT_LADDERS, currently "
             f"{'on' if TIME_SERIES_SAME_EVENT_LADDERS else 'off'}). Affects the "
             "backtest only — the live finder binds that constant at import.",
    )
    # Either side may be given alone; the other comes from config's default band
    parser.add_argument(
        "--spread-min", type=float, default=None, metavar="X",
        help="Time-series spread-band FLOOR (0-1) for the primary scenario; "
             "default: config.BACKTEST_DEFAULT_SPREAD_BAND's floor (the "
             "deadline-gap tier alone while that floor is 0). Given alone, "
             "the ceiling comes from that default. Backtest only — live "
             "trading reads the saved live defaults' band, or main.py's own "
             "--spread-min / --spread-max for one live run.",
    )
    parser.add_argument(
        "--spread-max", type=float, default=None, metavar="Y",
        help="Time-series spread-band CEILING (0-1) for the primary "
             "scenario; default: config.BACKTEST_DEFAULT_SPREAD_BAND's "
             "ceiling (no ceiling while it is 1). Given alone, the floor "
             "comes from that default. Backtest only — live trading reads "
             "the saved live defaults' band, or main.py's own --spread-min / "
             "--spread-max for one live run.",
    )
    parser.add_argument(
        "--no-band-sweep", action="store_true",
        help="Skip the spread-band grid (and its tier-floors-off runs); the "
             "dashboard's scenario explorer is not computed, and its filter "
             "bar offers the primary band only, with the Tier floors select "
             "disabled",
    )
    parser.add_argument(
        "--no-cap-sweep", action="store_true",
        help="Skip the per-trade size-cap sweep: the dashboard offers the "
             "run's own cap (config.BUDGET_FRACTION) only — a far smaller, "
             "faster page. Backtest only — live sizing reads its caps from "
             "the saved live defaults, or main.py's own --size-cap / "
             "--same-title-size-cap for one live run",
    )
    parser.add_argument(
        "--no-add-on-sweep", action="store_true",
        help="Skip the dashboard's Add to held pairs choice: its simulations run "
             "only while the dashboard is built, so skipping them makes that step "
             "faster and leaves the choice disabled. Backtest only — live runs add "
             "to held pairs when the saved live defaults say so, or main.py's own "
             "--add-to-held-pairs / --no-add-to-held-pairs for one live run",
    )
    parser.add_argument(
        "--no-sell-sweep", action="store_true",
        help="Skip the dashboard's Sell and Min. days to maturity selects (sell a "
             "whole position once it has held a chosen share of the profit it could "
             "make for config.TAKE_PROFIT_HOLD_DAYS days in a row, while at least a "
             "chosen number of days remain before its last market stops trading): "
             "their simulations run only while the dashboard is built, so skipping "
             "them makes that step much faster and leaves both selects disabled. "
             "Backtest only — a live run sells only at its saved sell_at level, "
             "which the dashboard's Save as live defaults… can set to a level "
             "shown here",
    )
    parser.add_argument(
        "--no-trim-sweep", action="store_true",
        help="Skip the dashboard's Trim to Kelly select (sell part of a held pair, "
             "at each weekly checkpoint, once it is worth more than its Kelly share "
             "of the portfolio): its simulations run only while the dashboard is "
             "built, so skipping them makes that step faster and leaves the select "
             "disabled. Backtest only — a live run trims by its own saved "
             "trim_to_kelly (main.py --trim-to-kelly for one run), which the "
             "dashboard's Save as live defaults… can set to the choice shown here",
    )
    parser.add_argument(
        "--no-trim-rise-sweep", action="store_true",
        help="Skip the dashboard's Min. price rise select (with Trim to Kelly on, trim a "
             "held pair only once its total price has risen a chosen number of cents since "
             "the last trade on it): its simulations run only while the dashboard is built. "
             "Trim to Kelly itself is still built. Backtest only — a live run waits for a "
             "price rise by its own saved trim_min_rise (main.py --trim-min-rise for one "
             "run)",
    )
    parser.add_argument(
        "--sell-workers", type=int, default=None, metavar="N",
        help="Worker processes the dashboard simulates its Sell select in "
             "(default: one less than this machine's CPU count, at most "
             f"{DASHBOARD_SELL_MAX_WORKERS} — config.DASHBOARD_SELL_MAX_WORKERS); "
             "1 runs them in the main process. Each worker holds one spread band's "
             "entries, so memory grows with the count",
    )
    args = parser.parse_args()
    # Every option is checked here, before logging starts, so a bad one leaves
    # no log file behind. The balance must be a real amount above 0 (float()
    # also accepts "nan" and "inf")
    if args.balance is not None and not (math.isfinite(args.balance) and args.balance > 0):
        parser.error("--balance must be a positive number of dollars")
    if args.max_horizon_days is not None and args.max_horizon_days < 1:
        parser.error("--max-horizon-days must be a positive integer")
    if args.sell_workers is not None and args.sell_workers < 1:
        parser.error("--sell-workers must be a positive integer")
    # k = 0 would treat every time-series pair as risk-free
    if args.interval_discount is not None and not (0.0 < args.interval_discount <= 1.0):
        parser.error("--interval-discount must be above 0 and at most 1")
    if args.spread_min is not None and not (0.0 <= args.spread_min <= 1.0):
        parser.error("--spread-min must be between 0 and 1")
    if args.spread_max is not None and not (0.0 <= args.spread_max <= 1.0):
        parser.error("--spread-max must be between 0 and 1")

    # The spread band. Reading config's default band also checks it, so a bad
    # default stops the run here rather than hours in. With neither flag given
    # the backtest uses the default itself (None); with one or both, the floor
    # must be below the ceiling. Values print exactly, so the error never shows
    # two numbers that look equal
    default_floor, default_ceiling = time_series_spread_band(None)
    if args.spread_min is None and args.spread_max is None:
        spread_band = None
    else:
        resolved_floor = default_floor if args.spread_min is None else args.spread_min
        resolved_ceiling = default_ceiling if args.spread_max is None else args.spread_max
        if not (resolved_floor < resolved_ceiling):
            floor_source = "" if args.spread_min is not None else " (config default)"
            ceiling_source = "" if args.spread_max is not None else " (config default)"
            parser.error(
                f"--spread-min/--spread-max: the band floor {resolved_floor!r}"
                f"{floor_source} must be strictly less than the ceiling "
                f"{resolved_ceiling!r}{ceiling_source}"
            )
        spread_band = (resolved_floor, resolved_ceiling)

    # Also checked before logging starts: otherwise a bad date could create, or
    # rotate away, the log of earlier runs for a run that never happened
    try:
        start_date = date.fromisoformat(args.start_date)
    except ValueError:
        parser.error(f"Invalid --start-date: {args.start_date!r}. Use YYYY-MM-DD format.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            # The log starts a new file at 20 MB and keeps the 3 before it, so it
            # cannot grow without limit. delay=True opens the file only when
            # something is first written to it
            logging.handlers.RotatingFileHandler(
                PROJECT_ROOT / "kalshi_backtest.log", maxBytes=20 * 1024 * 1024,
                backupCount=3, delay=True,
            ),
        ],
    )

    use_cache = not args.no_cache

    # The values below are worked out now only so the "Backtest config" line,
    # logged before the long data download, can show them: a run set up wrong
    # can then be stopped early. The backtest works them out again itself.
    # k, from the flag or else config
    effective_k = (TIME_SERIES_INTERVAL_PROB_DISCOUNT if args.interval_discount is None
                   else args.interval_discount)
    # Same-event ladders, from the flag or else config
    effective_ladders = (TIME_SERIES_SAME_EVENT_LADDERS if args.same_event_ladders is None
                         else args.same_event_ladders)
    # Say so when the run differs from config, since it then does not replay the live rule
    ladders_echo = "on" if effective_ladders else "off"
    if bool(effective_ladders) != bool(TIME_SERIES_SAME_EVENT_LADDERS):
        ladders_echo += f" (config: {'on' if TIME_SERIES_SAME_EVENT_LADDERS else 'off'})"
    # The spread band, read the same way the backtest reads it (already checked above)
    echo_floor, echo_ceiling = time_series_spread_band(spread_band)
    # The dashboard's extra views, each on unless its --no-… option was given.
    # Only the band grid is simulated during the run; the others are simulated
    # while the dashboard is built. The minimum price rises are part of the
    # trim view, so they are off whenever it is
    band_sweep = not args.no_band_sweep
    cap_sweep = not args.no_cap_sweep
    add_on_sweep = not args.no_add_on_sweep
    sell_sweep = not args.no_sell_sweep
    trim_sweep = not args.no_trim_sweep
    trim_rise_sweep = trim_sweep and not args.no_trim_rise_sweep
    # Worker processes for the dashboard's Sell view: one fewer than the CPUs
    # (the main process waits on them), up to the config limit
    sell_workers = (args.sell_workers if args.sell_workers is not None
                    else max(1, min((os.cpu_count() or 1) - 1, DASHBOARD_SELL_MAX_WORKERS)))
    # The saved live settings, shown on the config line for comparison only.
    # A problem reading them is shown there, never allowed to stop the run
    try:
        _live = live_defaults()
        live_rule_echo = describe_time_series_rule(_live.tier_floors, _live.spread_band)
        # Add the category/tag filter only when one is set
        if _live.categories is not None or _live.tags is not None:
            live_rule_echo += f"; category/tag filter ({describe_trade_filter(_live)})"
        # Which saved file, and when it was saved
        live_rule_echo += f" ({_live.origin})"
    except LiveDefaultsMissing:
        live_rule_echo = "none saved (live runs refuse to start)"
    except LiveDefaultsError as e:
        live_rule_echo = f"not recorded — the saved live defaults are refused ({e})"

    # Past market data exists only on the production API
    hist_client = build_historical_client()  # client for the archive of past markets
    live_client = build_prod_live_client()  # client for recently settled markets and the account

    # The starting balance: --balance if given, otherwise the account's value now
    if args.balance is not None:
        start = StartingBalance(args.balance, "set by --balance")
    else:
        try:
            # One read-only request for the production account's balance
            start = _account_starting_balance(live_client)
        except StartingBalanceError as e:
            # Stop rather than guess an amount. The log file is not shown on
            # screen, so the reason is also printed there (exit 1)
            message = f"{e}; pass --balance DOLLARS to choose a starting balance"
            logging.error("Backtest not run: %s", message)
            raise SystemExit(f"backtest: {message}") from None
    logging.info("Starting balance: $%s (%s)", f"{start.dollars:,.2f}", start.source)
    if start.dollars < MIN_BALANCE_CENTS / 100:
        # A live run would not trade from this little; the backtest still does
        logging.warning(
            "Starting balance $%s is below the $%s minimum (config.MIN_BALANCE_CENTS) "
            "below which a live run does not trade; the backtest trades from it anyway",
            f"{start.dollars:,.2f}", f"{MIN_BALANCE_CENTS / 100:,.2f}",
        )

    # Build the order-book depth model from saved snapshots. Never fails: with
    # none usable it returns None and logs why
    depth_model = load_depth_model()

    logging.info(
        "Backtest config: start=%s | balance=$%.2f | cache=%s | k=%.3f | ladders=%s "
        "| spread band=%g-%g | band sweep=%s | cap sweep=%s | add-on sweep=%s "
        "| sell sweep=%s | trim sweep=%s | live rule=%s | fills=%s",
        start_date, start.dollars, "on" if use_cache else "off", effective_k,
        ladders_echo, echo_floor, echo_ceiling,
        "on" if band_sweep else "off", "on" if cap_sweep else "off",
        "on" if add_on_sweep else "off",
        f"on ({sell_workers} worker process{'' if sell_workers == 1 else 'es'})"
        if sell_sweep else "off",
        # The trim view, saying when it runs without its minimum price rises
        "off" if not trim_sweep else "on" if trim_rise_sweep
        else "on (no minimum price rises)",
        live_rule_echo, _fills_echo(depth_model),
    )
    # A time-series pair needs a minimum price gap, set in config by how far
    # apart its deadlines are. If the spread band's upper limit is at or below
    # one of those minimums, no pair (or almost none) of that kind can enter in
    # the main scenario, so warn that its result for them means nothing. It uses
    # the backtest's own upper-limit test, so it cannot disagree with the run
    tier_notes = []
    for tier_days, tier in (
        (f"0-{SHORT_DEADLINE_GAP_DAYS}-day", MIN_PRICE_DIFF_SHORT_GAP),
        (f"{SHORT_DEADLINE_GAP_DAYS + 1}-{MAX_DEADLINE_GAP_DAYS}-day", MIN_PRICE_DIFF_LONG_GAP),
    ):
        # Is even a gap exactly at this minimum over the upper limit?
        if time_series_spread_too_wide(tier, echo_ceiling):
            tier_notes.append(
                f"no {tier_days}-gap time-series candidate can enter "
                f"(deadline-gap tier {tier:.2f})"
            )
        elif echo_ceiling <= tier + PRICE_EPSILON:
            tier_notes.append(
                f"only {tier_days}-gap spreads sitting exactly on the "
                f"{tier:.2f} deadline-gap tier can enter"
            )
    if tier_notes:
        logging.warning(
            "Spread-band ceiling %r sits at or below a deadline-gap tier, so "
            "in the PRIMARY scenario %s — do not read its time-series result "
            "as a strategy result for the tier(s) named%s",
            echo_ceiling, "; ".join(tier_notes),
            " (the band sweep's grid bands are unaffected)" if band_sweep else "",
        )

    # Kalshi's list of series and their categories, read once (a saved copy is
    # reused for a week). The backtest uses it to tell two sports apart when
    # pairing same-title markets, and the dashboard to group trades by category.
    # Never fails: it falls back to an older copy, or to an empty list
    series_categories = load_series_categories(live_client)
    # Run the backtest. It returns the main scenario's result plus everything
    # the dashboard needs for its other views
    result = run_backtest_sweep(
        hist_client=hist_client,
        live_client=live_client,
        start_date=start_date,
        initial_balance=start.dollars,
        use_cache=use_cache,
        max_horizon_days=args.max_horizon_days,
        interval_discount=args.interval_discount,
        sweep=not args.no_sweep,
        same_event_ladders=args.same_event_ladders,
        spread_band=spread_band,
        band_sweep=band_sweep,
        # With the band grid, also run its bands with the price-gap minimums
        # switched off (the dashboard's Tier floors view)
        tier_off_sweep=band_sweep,
        cap_sweep=cap_sweep,
        add_on_sweep=add_on_sweep,
        sell_sweep=sell_sweep,
        trim_sweep=trim_sweep,
        trim_rise_sweep=trim_rise_sweep,
        # Each trade is sized on an order book built from this model (None:
        # trades fill at the best price, in any size)
        depth_model=depth_model,
        series_categories=series_categories,
    )
    # The summary reports the main scenario; the rest is for the dashboard
    trades, equity_df = result.primary.trades, result.primary.equity_df

    if not trades:
        logging.info("No backtest trades found. Dashboard will show empty charts.")
    else:
        final_value  = float(equity_df["portfolio_value"].iloc[-1])
        total_return = (final_value - start.dollars) / start.dollars
        n_win        = sum(1 for t in trades if t.profit > 0)
        logging.info("Backtest Summary")
        # Today's date in UTC, the time zone the simulation's dates use
        logging.info("  Period:        %s → %s", start_date, datetime.now(UTC).date())
        logging.info("  Total trades:  %d", len(trades))
        logging.info("  Win rate:      %.1f%%", n_win / len(trades) * 100)
        logging.info("  Total return:  %+.1f%%", total_return * 100)
        # Formatted first, to get the thousands separators
        logging.info("  Final balance: $%s", f"{final_value:,.2f}")

    # Say when the past-market data was put together: the period above runs to
    # today, but the data holds nothing that settled after that
    _log_corpus_provenance(result)

    # T-bill yields, which the dashboard's Sharpe and Sortino ratios subtract.
    # Never fails (it falls back to a saved copy), but a download that hangs
    # can take a few minutes first
    risk_free = load_risk_free_rates()
    # Write the dashboard (it logs where). It is given the full result for its
    # views, the k the trades were sized at, and where the balance came from
    generate_dashboard(trades, equity_df, start_date, start.dollars,
                       sweep=result, interval_discount=result.primary.k,
                       series_categories=series_categories, risk_free=risk_free,
                       sell_workers=sell_workers, balance_source=start.source)
    logging.info("Open the HTML file in a browser to view the interactive charts.")
    # The page cannot save settings or start runs itself; its buttons open the
    # pages of the server ./start_dashboard.sh starts
    logging.info("To save the filter bar's scenario as the live defaults, or to trade: run "
                 "./start_dashboard.sh (it starts the defaults server and the live dashboard, "
                 "whose Backtest tab is this page), then use the filter bar's Save as live "
                 "defaults… or Trade using defaults… button.")


if __name__ == "__main__":
    main()
