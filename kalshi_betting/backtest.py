"""
File: backtest.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Command-line entry point for the Kalshi backtester. Parses CLI
    arguments (--start-date, --balance, --no-cache, --max-horizon-days,
    --interval-discount, --no-sweep, --same-event-ladders /
    --no-same-event-ladders, --spread-min, --spread-max, --no-band-sweep,
    --no-cap-sweep, --no-add-on-sweep, --no-sell-sweep, --sell-workers),
    configures logging to kalshi_backtest.log, constructs the necessary API
    clients, works out the starting balance (--balance, or else what the Kalshi
    account is worth when the run starts), fits the depth model from the saved
    order-book snapshots (depth_model.load_depth_model) so each trade is sized
    over a modeled book, delegates the full backtest
    simulation to backtester.run_backtest_sweep(), and then calls
    dashboard.generate_dashboard() to produce the interactive HTML report and,
    when the sell family is on, the sidecar chunk files its Sell select loads
    (a folder beside the page, config.DASHBOARD_FILES_DIRNAME, which must be
    kept with it).
    Prints a summary of key metrics (trade count, win rate, total return) to
    the log on completion, closed on every run by what settled-market corpus
    the run read (its assembly time, whether it was cached, and the archive
    cutoff as of assembly, as information).

Dependencies:
    Imports run_backtest_sweep and BacktestSweep from backtester.py,
    generate_dashboard from dashboard.py, build_historical_client /
    build_prod_live_client / load_series_categories (the dashboard's
    returns-by-category labels) from historical.py, load_risk_free_rates
    (the T-bill yields the page's Sharpe and Sortino subtract) from
    treasury.py, load_depth_model (the table every trade's modeled order book
    is built from; the backtest's fills line names it) from depth_model.py,
    read_account_balance (the account's cash and open positions,
    the starting balance when --balance is not given) from auth.py, and
    api_error_summary (the one-line reason a failed balance read is reported
    with) from _http.py.
    Imports from config.py:
    PROJECT_ROOT, DASHBOARD_SELL_MAX_WORKERS (the --sell-workers default's
    bound), MIN_BALANCE_CENTS (the low starting-balance WARNING),
    TIME_SERIES_INTERVAL_PROB_DISCOUNT and TIME_SERIES_SAME_EVENT_LADDERS
    (the pre-fetch echo and the flag's help text), the deadline-gap tier constants
    MIN_PRICE_DIFF_SHORT_GAP, MIN_PRICE_DIFF_LONG_GAP, SHORT_DEADLINE_GAP_DAYS,
    MAX_DEADLINE_GAP_DAYS and PRICE_EPSILON (the spread-band tier WARNING),
    the band helpers time_series_spread_band and
    time_series_spread_too_wide — backtest is one of the four band readers
    tests/test_strategy.py::TestTimeSeriesKellyParity::
    test_ast_live_path_reads_toggles_only_through_live_settings allows — and
    live_defaults with its LiveDefaultsError / LiveDefaultsMissing refusals,
    describe_time_series_rule and describe_trade_filter (the pre-fetch echo's
    "live rule=" clause, which names the saved live defaults). Entry point for
    `python3 -m kalshi_betting.backtest`.

Notes:
    Historical data only exists on the production Kalshi API, so both API clients
    always use prod credentials regardless of what mode the live bot was run in.
    The backtest reads market data and, unless --balance is given, the
    account balance, but never submits any orders.

    The starting balance is the amount the simulation starts with, all of it
    cash: its trade sizes follow from it, and the dashboard's total return and
    cumulative-return charts are measured from it. --balance DOLLARS sets it.
    Without the flag the run reads, once and before the fetch, what the
    production account is worth now: the cash on every shard plus what Kalshi
    says the open positions are worth — the portfolio value a live run sizes
    its trades on — so the backtest's trades are sized for the account as it
    stands. When Kalshi's reply has no readable positions value the cash
    alone is used, with a WARNING. A balance that cannot be read, or a read
    that comes to nothing, stops the run before anything is fetched (exit 1,
    the reason on stderr and in the log), since a starting balance nobody
    chose would size every simulated trade for some other account; --balance
    then runs without the read. A starting balance below
    config.MIN_BALANCE_CENTS, the value below which a live run does not
    trade, draws a WARNING, since the backtest trades from it anyway. The
    log's "Starting balance" line and the dashboard's header say where the
    amount came from.

    --sell-workers N sets how many worker processes the dashboard simulates
    its Sell select in (default: one less than the CPU count, at most
    config.DASHBOARD_SELL_MAX_WORKERS; 1 runs it in the main process). The
    workers are spawned, so they re-import this module: main() must only ever
    run under the `if __name__ == "__main__"` guard at the bottom of the file.

    The depth model is fitted before the fetch from the order-book snapshots
    saved by `python3 -m kalshi_betting.depth_model snapshot`, and handed to
    run_backtest_sweep, which sizes each trade over a modeled book built from
    it. With no usable snapshot every trade fills at the top of the book, in
    any size. The config echo's fills= clause says which, and the log names
    the reason.

    --interval-discount overrides the time-series interval discount k for this
    run ONLY: no live module imports this one (main.py's --interval-discount is
    a separate flag), and nothing here writes config.py or the saved live
    defaults — the calibration the run reports is a recommendation for a human
    to act on.

    --same-event-ladders / --no-same-event-ladders (DR-73) is the same kind of
    one-run override for config.TIME_SERIES_SAME_EVENT_LADDERS, and also never
    reaches the live finder — scanner.py binds that constant at import. Unlike
    k, it changes WHICH PAIRS EXIST rather than how they are priced, so it
    applies identically to every swept discount and a run with it on is not
    comparable to a baseline taken without it. The switch is on by the
    operator's 2026-09-26 decision, so a default run pairs ladders and
    --no-same-event-ladders replays the rule as it stood before it.

    That setting reaches kalshi_backtest.log (the pre-fetch echo below and
    run_backtest_sweep's resolved line) AND the HTML dashboard's own page
    header: dashboard._run_settings_html(sweep) prints "same-event ladders:
    on / off / not recorded" (BacktestSweep.same_event_ladders), a recorded
    on/off followed by its reading against the configured switch, beside the
    primary spread band, under the "Period:" line, above every section — so a
    ladder-enabled run's dashboard is no longer indistinguishable from a
    switch-off one and needs no hand labelling. Unlike DR-66b's
    subtitle-coverage caveat, this setting is chosen by the operator on the
    command line rather than discovered by the run, which is why it is named
    rather than banner-flagged. When a run's resolved setting departs from
    the switch this checkout's config sets, all three places say so: the
    echo below appends " (config: on|off)" from this module's binding,
    run_backtest_sweep's own ladder line names the departure in its source
    clause from backtester's, and the header reads backtester's binding
    through BacktestSweep.config_same_event_ladders beside
    same_event_ladders.

    --spread-min/--spread-max set the PRIMARY scenario's backtest-only
    time-series spread band (floor, ceiling) on pB - pA. Once
    config.time_series_spread_band() has resolved and validated it,
    backtester._find_entry layers the floor on the deadline-gap tier through
    config.min_price_diff_for_gap(spread_min=) and refuses a spread above the
    ceiling through config.time_series_spread_too_wide(). They never reach live
    trading (main.py's same-named flags are separate). Either flag may be given
    alone; the omitted side comes from config.BACKTEST_DEFAULT_SPREAD_BAND,
    read through config.time_series_spread_band(None) at call time. Both
    omitted passes spread_band=None, the no-override sentinel, which
    run_backtest_sweep resolves to config.BACKTEST_DEFAULT_SPREAD_BAND and
    logs with that source. The flags (each in [0, 1], and the resolved floor
    strictly less than the resolved ceiling) and the configured default
    band are all validated BEFORE logging is configured (TS-20), so a
    rejected value leaves kalshi_backtest.log untouched. A ceiling at or
    below a deadline-gap tier empties that tier in the primary scenario (or,
    sitting on it, keeps only spreads exactly on it), whatever the floor;
    that is an advisory rather than a rejected value, so it is a WARNING
    logged after logging is configured. The band sweep is ON by default, so
    a default run also simulates every band of the
    config.SPREAD_BAND_SWEEP_FLOORS x SPREAD_BAND_SWEEP_CEILINGS grid (no
    grid ceiling sits at or below either tier) — and, with it, every band
    whose floor sits below a deadline-gap tier a second time with the tier
    floors off (tier_off_sweep=band_sweep: the band's floor alone gates the
    spread and sets the leg-price-sum ceiling, 1 − floor, backtest only);
    --no-band-sweep skips that grid and those
    tier-floors-off runs (band_sweep=False, tier_off_sweep=False): the
    primary scenario still runs, but the dashboard's scenario explorer has no
    scenarios to show, and its filter bar offers the primary band only, with
    the Tier floors select disabled. There is no separate flag for the
    tier-off runs.

    The per-trade size-cap sweep is ON by default too (cap_sweep=True): the
    result carries a lazy backtester.CapSweep over backtester.SIZE_CAP_SWEEP
    (5%..95% and no cap), whose cells are simulated only when a report reads
    them — the run itself simulates nothing extra, and every point it returns,
    the summary block included, is sized at the run's own cap,
    config.BUDGET_FRACTION. The dashboard reads every cell as it is built,
    for its filter bar's Size cap select, the Interval Discount section and
    the scenario explorer's cap axis. With the band sweep's tier-floors-off
    runs on too (the default), a second lazy CapSweep
    (BacktestSweep.tier_off_cap_sweep) covers them at every cap as well, so
    every tier x band x k x cap scenario the page offers is a real
    simulation.
    --no-cap-sweep returns cap_sweep=None, and the dashboard then offers the
    run's own cap only — a far smaller page, built far faster. Like the band
    and k sweeps it is backtest-only: live sizing reads
    its caps from its run's config.LiveSettings, built from the saved live
    defaults.

    The "Add to held pairs" family is ON by default too (add_on_sweep=True):
    the result carries two more lazy CapSweeps (BacktestSweep.add_on_cap_sweep,
    and add_on_tier_off_cap_sweep over the tier-floors-off runs) whose every
    simulation adds to pairs it still holds, for the "all" population only.
    Like the cap sweeps they simulate nothing during the run — every figure and
    point the run returns is unchanged — and each cell is simulated only when
    the dashboard is built, so --no-add-on-sweep makes that step faster.

    The "Sell" family is ON by default too (sell_sweep=True): the result
    carries a lazy SellSweep (BacktestSweep.sell_sweep) that simulates, when
    the dashboard is built, every scenario selling a position early at each
    level of config.TAKE_PROFIT_LEVELS with each minimum of days before
    maturity of config.TAKE_PROFIT_MIN_DAYS (the dashboard's Sell and Min.
    days to maturity selects). It simulates nothing during the run;
    --no-sell-sweep skips it, leaving both selects disabled. Nothing here
    reaches live trading by itself: a live run sells only at its saved
    sell_at level (seller.plan_sales), which the dashboard's Save as live
    defaults… can set to a level shown there.

    The pre-fetch echo's "live rule=" clause names the saved live defaults'
    time-series rule and, when one is set, their category/tag filter (never
    main.py's per-run overrides), from a read of its own, failing soft;
    run_backtest_sweep reads them again and logs where THIS run's grid holds
    that rule. Only the dashboard's "Save as live defaults…" button (or
    ./start_dashboard.sh --seed), confirmed through defaults_server, sets the
    live defaults; nothing else chosen here reaches live trading. The run's
    last line says how: run ./start_dashboard.sh (it starts the defaults
    server and the live dashboard, whose Backtest tab is the page), then use
    the filter bar's Save as live defaults… or Trade using defaults… button.
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
    The amount a backtest starts with, and where that amount came from.

    main() works it out before the fetch: from --balance when the flag is
    given, otherwise from the Kalshi account (_account_starting_balance). The
    simulation starts with the amount, all of it cash, so its trade sizes
    follow from it, and the dashboard's total return and cumulative-return
    charts are measured from it. The source is printed beside it in the log
    and in the dashboard's header.

    Attributes:
        dollars (float): The starting balance in dollars; a finite amount above 0.
        source (str): A short note saying where the amount came from.

    Raises:
        ValueError: If dollars is not a finite amount above 0.
    """
    dollars: float
    source: str

    def __post_init__(self) -> None:
        # A run cannot start from nothing, and its returns are measured from this amount
        if not (math.isfinite(self.dollars) and self.dollars > 0):
            raise ValueError(
                f"a starting balance must be a finite amount above 0, not {self.dollars!r}")


class StartingBalanceError(Exception):
    """
    The account gave no amount a backtest can start from; the message says why.

    Raised by _account_starting_balance. main() turns it into a stop before
    the fetch (exit 1), adding that --balance chooses an amount instead.
    """


def _account_starting_balance(client) -> StartingBalance:
    """
    Read what the Kalshi account is worth now, the starting balance of a run given no --balance.

    The amount is the account's portfolio value: the cash on every shard plus
    what Kalshi says the open positions are worth. That is the figure a live
    run sizes its trades on, so the backtest's trades are sized for the
    account as it stands. When Kalshi's reply has no readable positions value,
    the cash alone is used and a WARNING says so, as a live run does. The live
    run's check of that value against the contracts held is not made here:
    the backtest risks no money, and the source note names the cash and the
    positions value apart, so a wrong part can be seen.

    Makes one read-only GET of the account balance, retried on temporary
    errors (auth.read_account_balance, which also logs its "Auth OK" line).

    Args:
        client: A production KalshiClient, as historical.build_prod_live_client() makes.

    Returns:
        StartingBalance: The account's value in dollars, and a note naming
            when it was read and what it is made of: "the account's value at
            <time> UTC: cash $X + open positions $Y", or, when the positions
            value could not be read, "the account's cash at <time> UTC; its
            open positions' value could not be read".

    Raises:
        StartingBalanceError: If the balance cannot be read (an error status
            such as 401 for bad credentials, a reply with no readable cash,
            or a network failure that outlasts the retries), or what was read
            comes to nothing: an account worth $0, or no cash beside a
            positions value that could not be read.
    """
    try:
        # One retried read: the cash on each shard and Kalshi's value of the open positions
        account = read_account_balance(client)
    except Exception as e:
        # The failure in one line (the HTTP status and Kalshi's error code, or
        # the error's type and first line), never the SDK's multi-line text
        raise StartingBalanceError(
            f"could not read the account balance ({api_error_summary(e)})") from e
    cash_cents = sum(account.shard_cash_cents.values())
    positions_cents = account.positions_value_cents
    read_at = f"{datetime.now(UTC):%Y-%m-%d %H:%M} UTC"
    if positions_cents is None:
        if cash_cents <= 0:
            # Positions of unknown worth may be held, so never say "worth $0"
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
    Close the run's report with what settled-market corpus it read.

    The "Period:" line prints start_date → today (the simulated window), and
    the corpus holds no market settled after its assembly. Every run brings
    its corpus up to date — a cache from an earlier UTC day is extended
    through today, so a cached corpus is at most a few hours old (assembled
    earlier the same UTC day) — and this line says which it was (DR-13).
    Logged on every run, "not recorded" included — absence must never be the
    only signal (DR-66). A cache from an earlier day that could not be
    brought up to date (CorpusProvenance.stale: the cutoff read or the
    extension failed, each with its WARNING) is named as that, never as
    today's. The archive cutoff is information only: a market
    settled after it is priced from Kalshi's live candlestick endpoint.

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
    CLI entry point for the Kalshi backtester.

    Parses command-line arguments (--start-date, --balance, --no-cache,
    --max-horizon-days, --interval-discount, --no-sweep,
    --same-event-ladders / --no-same-event-ladders, --spread-min,
    --spread-max, --no-band-sweep, --no-cap-sweep, --no-add-on-sweep,
    --no-sell-sweep), configures logging, constructs historical and live
    Kalshi API clients, works out the starting balance (--balance, or else the
    account's value now, read through _account_starting_balance; a read that
    fails, or comes to nothing, stops the run with exit 1 before the
    fetch), runs the full backtest simulation via run_backtest_sweep(), and generates
    an interactive HTML dashboard via generate_dashboard(). Logs a summary
    table of key metrics to
    kalshi_backtest.log on completion (this module installs only a
    RotatingFileHandler, no console handler, so nothing reaches stdout).

    The summary block reports the PRIMARY point of the sweep — the run at the
    effective interval discount, the primary spread band and the run's own
    per-trade size cap — so a default run's summary block reads exactly as
    the plain run_backtest() path's did. The other swept discounts and bands
    exist for the dashboard (its page-wide filter bar, whose k select the
    Interval Discount section follows, its scenario explorer and k-hat
    breakdown) and the calibration report; the tier-floors-off family is read
    by the dashboard's filter bar, k-hat breakdown and scenario explorer
    (their Tier floors choice), carried to generate_dashboard on the sweep;
    and the lazily simulated size caps (result.cap_sweep, and the
    tier-floors-off family's at every cap, result.tier_off_cap_sweep) only
    by the dashboard, which reads every cell as the page is built (its filter
    bar, Interval Discount section and scenario explorer); the lazily
    simulated add-on sweeps (result.add_on_cap_sweep, and
    result.add_on_tier_off_cap_sweep) and sell family (result.sell_sweep) are
    read the same way.
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
    # None means "read the account": its value is only known once a client exists
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
    # BooleanOptionalAction gives --same-event-ladders and
    # --no-same-event-ladders from one declaration; default=None is the "no
    # override" sentinel run_backtest_sweep resolves at call time, exactly as
    # --interval-discount's None resolves k. There is nothing to validate — the
    # action can only ever yield True, False or None.
    parser.add_argument(
        "--same-event-ladders", action=argparse.BooleanOptionalAction, default=None,
        help="Pair two dated cumulative rungs of ONE event as a time-series "
             "ladder for this backtest (default: config."
             "TIME_SERIES_SAME_EVENT_LADDERS, currently "
             f"{'on' if TIME_SERIES_SAME_EVENT_LADDERS else 'off'}). Affects the "
             "backtest only — the live finder binds that constant at import.",
    )
    # Each side is independently optional; the omitted one resolves from
    # config's own default band (see the validation block below), not from a
    # literal (0.0, 1.0) here — which is why the help names the constant
    # rather than a number.
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
        "--sell-workers", type=int, default=None, metavar="N",
        help="Worker processes the dashboard simulates its Sell select in "
             "(default: one less than this machine's CPU count, at most "
             f"{DASHBOARD_SELL_MAX_WORKERS} — config.DASHBOARD_SELL_MAX_WORKERS); "
             "1 runs them in the main process. Each worker holds one spread band's "
             "entries, so memory grows with the count",
    )
    args = parser.parse_args()
    # Every return is measured against the starting balance, so it must be a
    # real amount above 0 (float() also reads "nan" and "inf")
    if args.balance is not None and not (math.isfinite(args.balance) and args.balance > 0):
        parser.error("--balance must be a positive number of dollars")
    if args.max_horizon_days is not None and args.max_horizon_days < 1:
        parser.error("--max-horizon-days must be a positive integer")
    if args.sell_workers is not None and args.sell_workers < 1:
        parser.error("--sell-workers must be a positive integer")
    # Above 0, as the live sizer the backtest sizes through requires (k = 0
    # would price every time-series pair as riskless)
    if args.interval_discount is not None and not (0.0 < args.interval_discount <= 1.0):
        parser.error("--interval-discount must be above 0 and at most 1")
    if args.spread_min is not None and not (0.0 <= args.spread_min <= 1.0):
        parser.error("--spread-min must be between 0 and 1")
    if args.spread_max is not None and not (0.0 <= args.spread_max <= 1.0):
        parser.error("--spread-max must be between 0 and 1")

    # Resolved here, BEFORE logging is configured (TS-20 — see the comment
    # below). The configured default is read and validated on EVERY run,
    # flags or not, through config's own function at call time (never a
    # by-value copy of BACKTEST_DEFAULT_SPREAD_BAND): an invalid default then
    # raises here, before any log file exists, rather than after basicConfig
    # in the echo below; and it supplies the side a lone flag omits. Both
    # flags omitted still hands run_backtest_sweep None, the "no override"
    # sentinel it resolves to the configured default itself and logs with
    # that source; either flag given hands it a (floor, ceiling) validated
    # here against the same rule config.time_series_spread_band enforces
    # (0 <= floor < ceiling <= 1, NaN refused by every comparison), so that
    # function's own ValueError can never fire out of a multi-hour run. The
    # bounds print with repr, which is exact, so a rejection can never read
    # "floor 0.3 must be strictly less than the ceiling 0.3".
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

    # Validated BEFORE logging is configured, alongside the other argument
    # checks above, so a rejected argument leaves no trace: parser.error
    # exits, and doing this after basicConfig created kalshi_backtest.log for
    # a run that never happened — or, worse, ROTATED it, evicting real
    # history to record nothing (TS-20). delay=True on the handler below is
    # the belt to this brace, matching main.py and scheduler.py.
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
    # Named against the config when the run departs from it, so the line read
    # before a multi-hour fetch says the run will not replay the live rule.
    ladders_echo = "on" if effective_ladders else "off"
    if bool(effective_ladders) != bool(TIME_SERIES_SAME_EVENT_LADDERS):
        ladders_echo += f" (config: {'on' if TIME_SERIES_SAME_EVENT_LADDERS else 'off'})"
    # Same pre-fetch echo, same reason, for the spread band: the VALUE is
    # resolved through the same config function run_backtest_sweep calls
    # (never a by-value copy of BACKTEST_DEFAULT_SPREAD_BAND), so it cannot
    # resolve a different band. Both the configured default and any flag
    # were validated above, so this call only reads them back and cannot
    # raise here. It is RENDERED with %g, like k's %.3f beside it, so two
    # bands differing below %g's six significant digits print alike;
    # run_backtest_sweep's own "Time-series spread band" line, also logged
    # before any fetch, renders the resolved band exactly.
    echo_floor, echo_ceiling = time_series_spread_band(spread_band)
    band_sweep = not args.no_band_sweep
    # ON by default, like the band sweep: it simulates nothing during the run
    # (each cell is simulated lazily, only when a report reads it), so the
    # opt-out saves nothing in the run itself — only the time and size of
    # whatever report reads the cells, and the entries kept alive for it
    cap_sweep = not args.no_cap_sweep
    # ON by default, like the cap sweep, and for the same reason: nothing is
    # simulated during the run, so the opt-out saves only the dashboard step
    add_on_sweep = not args.no_add_on_sweep
    # ON by default too, and lazy the same way: the dashboard simulates it
    sell_sweep = not args.no_sell_sweep
    # The dashboard simulates the Sell select in worker processes: one less
    # than the CPUs (the main process waits on them), within the config bound
    sell_workers = (args.sell_workers if args.sell_workers is not None
                    else max(1, min((os.cpu_count() or 1) - 1, DASHBOARD_SELL_MAX_WORKERS)))
    # The saved live defaults' rule and filter, for the echo only: a read of its
    # own, separate from run_backtest_sweep's (which logs the INFO or WARNING).
    # Fails soft, unlike the checks above (one echo clause must not abort a run)
    try:
        _live = live_defaults()
        live_rule_echo = describe_time_series_rule(_live.tier_floors, _live.spread_band)
        # The category/tag filter only when one is set, in the live run's words
        if _live.categories is not None or _live.tags is not None:
            live_rule_echo += f"; category/tag filter ({describe_trade_filter(_live)})"
        # Which saved file, when and from what
        live_rule_echo += f" ({_live.origin})"
    except LiveDefaultsMissing:
        live_rule_echo = "none saved (live runs refuse to start)"
    except LiveDefaultsError as e:
        live_rule_echo = f"not recorded — the saved live defaults are refused ({e})"

    # Always uses prod API — historical data only exists there.
    hist_client = build_historical_client()  # authenticated prod KalshiClient for /historical raw GETs
    live_client = build_prod_live_client()  # returns KalshiClient pointed at prod for recently-settled market fetching

    # The starting balance, before the fetch so the echo below can name it:
    # --balance when given, otherwise what the account is worth now
    if args.balance is not None:
        start = StartingBalance(args.balance, "set by --balance")
    else:
        try:
            # One read-only GET of the production account's balance
            start = _account_starting_balance(live_client)
        except StartingBalanceError as e:
            # Stop rather than guess an amount: the run's trade sizes and
            # returns follow from it. The log has no console handler, so the
            # reason is printed to stderr too (SystemExit with a message, exit 1)
            message = f"{e}; pass --balance DOLLARS to choose a starting balance"
            logging.error("Backtest not run: %s", message)
            raise SystemExit(f"backtest: {message}") from None
    logging.info("Starting balance: $%s (%s)", f"{start.dollars:,.2f}", start.source)
    if start.dollars < MIN_BALANCE_CENTS / 100:
        # A live run does not trade below this value; the backtest has no such stop
        logging.warning(
            "Starting balance $%s is below the $%s minimum (config.MIN_BALANCE_CENTS) "
            "below which a live run does not trade; the backtest trades from it anyway",
            f"{start.dollars:,.2f}", f"{MIN_BALANCE_CENTS / 100:,.2f}",
        )

    # Fit the depth table from the saved order-book snapshots before the fetch,
    # so the echo below can say how trades will fill; never raises (with no
    # usable snapshot it returns None and logs why)
    depth_model = load_depth_model()

    logging.info(
        "Backtest config: start=%s | balance=$%.2f | cache=%s | k=%.3f | ladders=%s "
        "| spread band=%g-%g | band sweep=%s | cap sweep=%s | add-on sweep=%s "
        "| sell sweep=%s | live rule=%s | fills=%s",
        start_date, start.dollars, "on" if use_cache else "off", effective_k,
        ladders_echo, echo_floor, echo_ceiling,
        "on" if band_sweep else "off", "on" if cap_sweep else "off",
        "on" if add_on_sweep else "off",
        f"on ({sell_workers} worker process{'' if sell_workers == 1 else 'es'})"
        if sell_sweep else "off", live_rule_echo, _fills_echo(depth_model),
    )
    # Warn on a ceiling that empties a tier. config.time_series_spread_band's
    # docstring asks a caller taking an operator-typed ceiling to warn when it
    # sits at or below a deadline-gap tier, so an emptied tier is not read as
    # a strategy result. TS-20 does not apply — this is an advisory, not a
    # rejected argument — so the WARNING is deliberately AFTER logging is
    # configured, unlike the parser.error() checks above. The floor is
    # irrelevant: the effective floor is max(tier, floor) and the floor sits
    # below the ceiling. Each tier is judged on a spread sitting exactly on
    # it, the smallest real spread that tier's floor keeps. Two outcomes are
    # warned:
    #   - EMPTIED: config.time_series_spread_too_wide refuses even that
    #     spread. That function is the one place the ceiling's PRICE_EPSILON
    #     lives, and _find_entry refuses on the same predicate, so this
    #     cannot drift from what the backtest does.
    #   - ON THE TIER: the ceiling is within PRICE_EPSILON of the tier. The
    #     tier then keeps only spreads sitting exactly on it, since no price
    #     grid is fine enough to put a real spread within 2 x PRICE_EPSILON
    #     above it.
    # Only the PRIMARY scenario is affected: every band of the sweep grid
    # sits above both tiers (config.SPREAD_BAND_SWEEP_CEILINGS), so the
    # message says so when that grid runs. Tier labels come from the config
    # day counts, never a literal.
    tier_notes = []
    for tier_days, tier in (
        (f"0-{SHORT_DEADLINE_GAP_DAYS}-day", MIN_PRICE_DIFF_SHORT_GAP),
        (f"{SHORT_DEADLINE_GAP_DAYS + 1}-{MAX_DEADLINE_GAP_DAYS}-day", MIN_PRICE_DIFF_LONG_GAP),
    ):
        # The backtest's own ceiling test, applied to an on-tier spread
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
        # The dashboard's "Tier floors: off" view: the band sweep's
        # tier-bound bands simulated again with the tiers off (needs the grid)
        tier_off_sweep=band_sweep,
        cap_sweep=cap_sweep,
        # The dashboard's "Add to held pairs: on" views, simulated lazily when
        # the page is built (no simulation during the run)
        add_on_sweep=add_on_sweep,
        # The dashboard's Sell select, lazy the same way
        sell_sweep=sell_sweep,
        # Every trade's order book is built from this table (None: top of book)
        depth_model=depth_model,
    )  # returns BacktestSweep — primary point, one point per swept k and the calibration, plus the band-sweep payload (scenarios, same_title_point, calibrations_by_band) and the tier-floors-off family (tier_off_scenarios, tier_off_calibrations_by_band) unless --no-band-sweep, the lazy size-cap sweeps (cap_sweep, and tier_off_cap_sweep with the band sweep) unless --no-cap-sweep, and the lazy add-on sweeps (add_on_cap_sweep, and add_on_tier_off_cap_sweep with the band sweep) unless --no-add-on-sweep
    # Everything below reports the PRIMARY point, so the summary block and the
    # dashboard's other six sections read exactly as they did before the sweep
    # existed. The remaining k points are read by the filter bar's k select
    # (which the Interval Discount section follows), the band-sweep payload by
    # the dashboard's scenario explorer, filter bar and k-hat breakdown, the
    # tier-floors-off family by that filter bar, k-hat breakdown and scenario
    # explorer too (their Tier floors choice), and the lazy size-cap sweeps
    # (result.cap_sweep and, over the
    # tier-floors-off family, result.tier_off_cap_sweep, simulated only when
    # a cell is read) only by the dashboard, whose one grid walk reads
    # every cell as the page is built (filter bar, Interval Discount section
    # and scenario explorer). The add-on sweeps (result.add_on_cap_sweep and
    # result.add_on_tier_off_cap_sweep) are lazy the same way.
    trades, equity_df = result.primary.trades, result.primary.equity_df

    if not trades:
        logging.info("No backtest trades found. Dashboard will show empty charts.")
    else:
        final_value  = float(equity_df["portfolio_value"].iloc[-1])
        total_return = (final_value - start.dollars) / start.dollars
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

    # After either branch: the window (the Period line, when printed) runs to
    # today but the corpus only to its assembly — say so beside the result
    # rather than only at the top of a long log (DR-13)
    _log_corpus_provenance(result)

    # generate_dashboard() already logs "Dashboard written: %s" itself (BS-26) —
    # don't duplicate that line here, just point the user at the file.
    #
    # sweep carries the calibration and every swept point for the k
    # selectors, the scenario explorer's band x k payload, the filter bar,
    # the k-hat breakdown, both Tier floors views (the bar's and the
    # explorer's, from its tier-floors-off family), the lazy size-cap sweeps
    # (cap_sweep and tier_off_cap_sweep, None under --no-cap-sweep — every
    # cell of them is simulated here, in the page's one grid walk), the lazy
    # add-on sweeps (add_on_cap_sweep and add_on_tier_off_cap_sweep, None
    # under --no-add-on-sweep) and the header's run-settings line;
    # interval_discount is the resolved k these trades were sized at, which
    # the Risk section's Kelly scatter must price on
    #
    # series_categories files each trade under Kalshi's own series category and
    # tags for the Returns Decomposition breakdown: one cached read-only GET of
    # /series, which never raises (it falls back to a stale copy or to {})
    series_categories = load_series_categories(live_client)
    # The T-bill yields every Sharpe and Sortino on the page subtracts; never
    # raises (falls back to the saved copy, then "unavailable"), but an
    # unresponsive single-address host can cost about 4 minutes first
    risk_free = load_risk_free_rates()
    # balance_source: the header says where the starting balance came from
    generate_dashboard(trades, equity_df, start_date, start.dollars,
                       sweep=result, interval_discount=result.primary.k,
                       series_categories=series_categories, risk_free=risk_free,
                       sell_workers=sell_workers, balance_source=start.source)
    logging.info("Open the HTML file in a browser to view the interactive charts.")
    # How a scenario on the page becomes the live defaults, or is traded: the
    # page cannot write files or start runs, so its buttons open the defaults
    # server's pages (started by ./start_dashboard.sh, beside the live
    # dashboard, whose Backtest tab shows this page)
    logging.info("To save the filter bar's scenario as the live defaults, or to trade: run "
                 "./start_dashboard.sh (it starts the defaults server and the live dashboard, "
                 "whose Backtest tab is this page), then use the filter bar's Save as live "
                 "defaults… or Trade using defaults… button.")


if __name__ == "__main__":
    main()
