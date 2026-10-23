"""
File: backtest.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Command-line entry point for the Kalshi backtester. Parses CLI
    arguments (--start-date, --balance, --no-cache, --max-horizon-days,
    --interval-discount, --no-sweep, --same-event-ladders /
    --no-same-event-ladders, --scheduled-close / --no-scheduled-close,
    --stale-quote-hours / --no-stale-quote-check,
    --max-volume-share / --no-volume-cap,
    --spread-min, --spread-max, --no-band-sweep,
    --no-cap-sweep, --no-sell-sweep, --sell-workers, --no-rebalance-sweep,
    --no-move-sweep, --no-eda, --dashboard-only, --accept-changed-simulator),
    configures logging to kalshi_backtest.log, constructs the necessary API
    clients, works out the starting balance (--balance, or else what the Kalshi
    account is worth when the run starts), fits the depth model from the saved
    order-book snapshots (depth_model.load_depth_model) so each trade is sized
    over a modeled book, delegates the full backtest
    simulation to backtester.run_backtest_sweep(), saves what the dashboard is
    handed (config.BACKTEST_SWEEP_FILENAME), and then calls
    dashboard.generate_dashboard() to produce the interactive HTML report and,
    when the sell family is on, the sidecar chunk files its Sell select loads
    (a folder beside the page, config.DASHBOARD_FILES_DIRNAME, which must be
    kept with it). With --dashboard-only it runs no backtest: it rebuilds the
    page from the inputs the last full run saved, and then the Data page from
    the saved sweep and listing too, unless --no-eda. A full run likewise
    rebuilds the Data page (eda.build_eda_page, the live dashboard's Data tab)
    after its backtest page, with its own sweep and /series listing, unless
    --no-eda; that page is reporting only, and a failure there is one WARNING
    that changes neither the exit code nor the backtest page.
    Prints a summary of key metrics (trade count, win rate, total return) to
    the log on completion, closed on every run by what settled-market corpus
    the run read (its assembly time, whether it was cached, and the archive
    cutoff as of assembly, as information).

Dependencies:
    Imports run_backtest_sweep and BacktestSweep from backtester.py,
    generate_dashboard from dashboard.py, build_historical_client /
    build_prod_live_client / load_series_categories (Kalshi's /series
    listing, read once and handed to both the sweep, whose same-title sport
    check files each series by it, and the dashboard's returns-by-category
    labels) / load_series_fee_multipliers (each series' fee multiplier from
    the same listing, handed to the sweep and carried, not yet used) from
    historical.py, load_risk_free_rates
    (the T-bill yields the page's Sharpe and Sortino subtract) from
    treasury.py, load_depth_model (the table every trade's modeled order book
    is built from; the backtest's fills line names it) from depth_model.py,
    read_account_balance (the account's cash and open positions,
    the starting balance when --balance is not given) from auth.py,
    build_eda_page (the Data page, built after the backtest page; this is the
    one module of the package that imports eda) from eda.py, and
    api_error_summary (the one-line reason a failed balance read, or a failed
    save or load of the saved sweep file, is reported with) from _http.py.
    It reads, but does not import, the source files of the simulator modules
    (backtester, strategy, scanner, depth_model, historical) and config.py,
    to fingerprint the code a saved sweep was simulated with.
    Imports from config.py:
    PROJECT_ROOT, DASHBOARD_SELL_MAX_WORKERS (the --sell-workers default's
    bound), EDA_MAX_WORKERS (the most worker processes the Data page counts
    day files in), BACKTEST_SWEEP_FILENAME, BACKTEST_SWEEP_FORMAT and
    BACKTEST_SWEEP_HEADER_MAX_BYTES (the saved sweep file: where it goes, the
    format tag it is written in, the most of its first line a rebuild reads),
    MIN_BALANCE_CENTS (the low starting-balance WARNING),
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

    --scheduled-close / --no-scheduled-close is the one-run override for
    config.BACKTEST_SCHEDULED_CLOSE (on by default): whether the backtest reads
    each settled market's scheduled close, from a rule learned per series,
    where a live run reads its close. It never reaches a live run.

    --stale-quote-hours N / --no-stale-quote-check is the one-run override for
    config.BACKTEST_STALE_QUOTE_MAX_AGE_SECONDS (24 hours by default): how
    recently each market of an entry, a sale or a Kelly trim must have traded,
    or moved its quote, before the checkpoint. It never reaches a live run.

    --max-volume-share PCT / --no-volume-cap is the one-run override for
    config.BACKTEST_MAX_VOLUME_SHARE (10% by default): the most of a market's
    next-day traded contracts a backtest buy or sale may take. It reads hours
    a live run cannot see, only ever makes a trade smaller, and never reaches
    a live run.

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

    The "Sell" family is ON by default too (sell_sweep=True): the result
    carries a lazy SellSweep (BacktestSweep.sell_sweep) that simulates, when
    the dashboard is built, every scenario selling a position early at each
    level of config.TAKE_PROFIT_LEVELS with each minimum of days before
    maturity of config.TAKE_PROFIT_MIN_DAYS (the dashboard's Sell and Min.
    days to maturity selects). None of those scenarios rebalances held
    pairs: the dashboard sets Sell to no selling and shuts it while
    Rebalance is yes. Like the cap sweeps it simulates nothing during the
    run — every figure and point the run returns is unchanged — and
    --no-sell-sweep skips it, leaving both selects disabled. Nothing here
    reaches live trading by itself: a live run sells at its saved sell_at
    level (main.py --sell-at overrides it for one run), which the dashboard's
    Save as live defaults… can set.

    The "Rebalance" family is ON by default too (rebalance_sweep=True): the
    result carries a lazy RebalanceSweep (BacktestSweep.rebalance_sweep)
    that simulates, when the dashboard is built, every scenario again with
    held pairs rebalanced at each weekly checkpoint — topped up to their
    Kelly size when they have room and sold down to it when they have
    grown past it, the two together (the dashboard's Rebalance select). It
    simulates nothing during the run; --no-rebalance-sweep skips it,
    leaving the select disabled. A live run adds to and trims held pairs
    only by its own saved settings (main.py's flags for one run), which the
    dashboard's Save as live defaults… can set.

    The family's minimum price movements are ON by default with it
    (move_sweep=True): the RebalanceSweep also carries every minimum of
    config.MIN_PRICE_MOVES (RebalanceSweep.min_moves), where a held pair is
    added to or trimmed only once its total price has moved that many
    cents, up or down, since the last trade on it (the dashboard's Min.
    price movement select). They too are simulated only when the dashboard
    is built. --no-move-sweep skips them and still builds Rebalance itself;
    --no-rebalance-sweep skips both.

    Every full run saves what it hands generate_dashboard, right after the
    sweep and before the page is built, in config.BACKTEST_SWEEP_FILENAME in
    PROJECT_ROOT (gitignored): the BacktestSweep with every lazy family, the
    start date, the starting balance and where it came from, the /series
    listing, the primary k and the window's last day (the UTC day the sweep
    finished, which the page's Period line names). The file is a gzip stream:
    one line of JSON (the format tag, the simulator fingerprint with a digest
    per module, when it was saved, the Python version and the start date),
    then the inputs pickled; it is written to a ".tmp" file renamed over the
    old one. A save that fails (an object that cannot be pickled, a full disk)
    is one WARNING: the page is still built, and any earlier run's file is
    deleted, since it no longer matches the page.

    --dashboard-only rebuilds the page from that file, with no fetch and no
    Kalshi request: it reads the T-bill yields again and calls
    generate_dashboard with the run's own arguments and its own
    --sell-workers. The flags that shape a run (--start-date, --balance,
    --no-cache, --max-horizon-days, --interval-discount, --no-sweep, the
    ladder flags, --spread-min/--spread-max, --no-band-sweep) are refused with
    it; the other --no-*-sweep flags drop their lazy family from the loaded
    sweep (_without_families), giving the page a run with those flags would
    have built, and without the size-cap sweep the families that stay are
    kept at the run's own cap, as such a run builds them. The fingerprint
    (_simulator_fingerprint_parts) is a SHA-256 per simulator module over its
    parsed code with docstrings left out, plus one over the config.py
    statements those modules read and everything those statements use, so
    comments, docstrings, layout and dashboard-only constants never count.
    The lazy families simulate their cells while the page is built, with the
    code as it is then, so a rebuild whose fingerprint differs from the
    file's is refused (exit 2) unless --accept-changed-simulator is given:
    the run's own figures and those cells would come from two different
    versions of the code. A missing file exits 2; a file that cannot be read
    or loaded exits 1. Loading a pickle runs the code its objects name, so
    only a file this program wrote should be loaded.

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
import ast
import functools
import gzip
import hashlib
import json
import logging
import logging.handlers
import math
import os
import pickle
import platform
import time
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from ._http import api_error_summary
from .auth import read_account_balance
from .backtester import BacktestSweep, run_backtest_sweep
from .config import (
    BACKTEST_MAX_VOLUME_SHARE,
    BACKTEST_SCHEDULED_CLOSE,
    BACKTEST_STALE_QUOTE_MAX_AGE_SECONDS,
    BACKTEST_SWEEP_FILENAME,
    BACKTEST_SWEEP_FORMAT,
    BACKTEST_SWEEP_HEADER_MAX_BYTES,
    DASHBOARD_SELL_MAX_WORKERS,
    EDA_MAX_WORKERS,
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
from .eda import build_eda_page
from .historical import (
    build_historical_client,
    build_prod_live_client,
    load_series_categories,
    load_series_fee_multipliers,
)
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


def _stale_quote_text(limit: float | None) -> str:
    """
    A stale-quote limit as the pre-fetch echo and --help name it: "24 h", or "off".

    Args:
        limit (float | None): The limit in seconds; None for the check off.

    Returns:
        str: The hours with %g and " h", or "off".
    """
    return "off" if limit is None else f"{limit / 3600:g} h"


def _volume_cap_text(share: float | None) -> str:
    """
    A volume cap as the pre-fetch echo and --help name it: "10%", or "off".

    Args:
        share (float | None): The share in (0, 1]; None for the cap off.

    Returns:
        str: The percent with %g, or "off".
    """
    return "off" if share is None else f"{share * 100:g}%"


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


# The modules whose code decides what a backtest simulates: during the run, and
# again while its dashboard is built, when the lazy families (the size caps,
# Sell and Rebalance) simulate their cells. With the
# parts of config.py they read, they make up the simulator fingerprint a saved
# sweep is stamped with. dashboard.py is not among them: changing it is what a
# rebuild from the saved sweep is for.
_SIMULATOR_MODULES = ("backtester", "strategy", "scanner", "depth_model", "historical")

# What a full run saves for the dashboard, by key: exactly what main() hands
# generate_dashboard but the T-bill yields (read again by a rebuild) and the
# worker count (a rebuild's own --sell-workers). The trades and the equity
# curve are the sweep's primary point's, so they are not saved twice.
_DASHBOARD_INPUT_KEYS = frozenset({
    "sweep", "start_date", "initial_balance", "balance_source", "series_categories",
    "interval_discount", "period_end",
})

# The flags that only shape a run (what it fetches, simulates and sizes).
# --dashboard-only refuses each one left away from its default, since a page
# rebuilt from a saved run shows that run's own settings; every other option
# is one a rebuild reads (tests/test_backtest.py checks that the two lists
# together cover the parser).
_RUN_ONLY_FLAGS = (
    ("start_date", "--start-date"), ("balance", "--balance"), ("no_cache", "--no-cache"),
    ("max_horizon_days", "--max-horizon-days"), ("interval_discount", "--interval-discount"),
    ("no_sweep", "--no-sweep"),
    ("same_event_ladders", "--same-event-ladders / --no-same-event-ladders"),
    ("scheduled_close", "--scheduled-close / --no-scheduled-close"),
    ("stale_quote_hours", "--stale-quote-hours"),
    ("no_stale_quote_check", "--no-stale-quote-check"),
    ("max_volume_share", "--max-volume-share"),
    ("no_volume_cap", "--no-volume-cap"),
    ("spread_min", "--spread-min"), ("spread_max", "--spread-max"),
    ("no_band_sweep", "--no-band-sweep"),
)


class SavedSweepError(Exception):
    """
    The saved sweep file cannot be used for a rebuild; the message says why.

    Raised by _read_saved_header and _load_saved_inputs. main() turns it into
    a stop (exit 1) that says to run the backtest again.
    """


def _without_docstrings(tree: ast.Module) -> ast.Module:
    """
    Remove every docstring from a parsed module, so a fingerprint reads its code alone.

    Why: a rebuild is refused whenever the fingerprint differs, and the
    simulator modules' docstrings are reworded often. Without this, a
    docstring edit, which changes no simulation, would refuse every rebuild
    until --accept-changed-simulator is passed, and a flag that has to be
    passed for harmless edits soon gets passed for real ones too.

    A docstring is the string a module, class or function body opens with.
    Comments are never in the parsed tree, and ast.dump ignores line numbers
    and layout, so with docstrings gone an edit to wording alone leaves the
    fingerprint as it was.

    Args:
        tree (ast.Module): A freshly parsed module; changed in place.

    Returns:
        ast.Module: The same tree, without docstrings.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                # A body cannot be empty, so a docstring alone becomes "pass"
                node.body = body[1:] or [ast.Pass()]
    return tree


def _config_names_read(tree: ast.Module) -> set[str]:
    """
    The names a simulator module reads from config.py.

    Why: the fingerprint covers only the parts of config.py the simulator
    uses (with _config_closure), not the whole file. config.py also holds the
    dashboard's own constants, and a change to one of those is exactly what a
    rebuild is for; fingerprinting the whole file would refuse that rebuild.
    These names are where that part starts.

    A module reads them as `from .config import NAME` (anywhere in it) or as
    `config.NAME` after `from . import config`, the two ways the package
    reaches config (tests/test_backtest.py checks that the simulator modules
    use no other). A star import reads every name: "*" stands for that.

    Args:
        tree (ast.Module): The module, parsed.

    Returns:
        set[str]: The config names it reads.
    """
    names: set[str] = set()
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level != 1:
            continue
        if node.module == "config":
            names.update(alias.name for alias in node.names)
        elif node.module is None:
            # `from . import config [as name]`: config.NAME is read through the alias
            aliases.update(alias.asname or alias.name for alias in node.names
                           if alias.name == "config")
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in aliases):
            names.add(node.attr)
    return names


def _config_closure(tree: ast.Module, names: set[str]) -> list[tuple[str, str]]:
    """
    The config.py statements that define the names read, and every config name they use.

    Why: a name the simulator reads can depend on others it never names. The
    backtester calls config.rebalance_min_days, which reads
    REBALANCE_MIN_DAYS_SINCE_TRADE_MAX; a change to that constant changes
    what a simulation does, so it must change the fingerprint, or a rebuild
    would mix two versions of the code without a word.

    A constant defined from other constants, a function calling other
    functions, a class whose fields default to constants: each brings in the
    statements its own names come from, until nothing new is reached. A name
    config.py does not define at its top level (an import such as UTC) is
    left out, since its definition is not config's code.

    Args:
        tree (ast.Module): config.py, parsed, its docstrings removed.
        names (set[str]): The names read ("*" for every one).

    Returns:
        list[tuple[str, str]]: Each name reached, in name order, with the
            ast.dump of the statement(s) defining it.
    """
    defined: dict[str, list[ast.stmt]] = {}
    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound = {stmt.name}
        else:
            # An assignment, or any other top-level statement: the names it stores
            bound = {node.id for node in ast.walk(stmt)
                     if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
        for name in bound:
            defined.setdefault(name, []).append(stmt)
    todo = sorted(defined) if "*" in names else sorted(names)
    reached: dict[str, str] = {}
    while todo:
        name = todo.pop()
        if name in reached or name not in defined:
            continue
        reached[name] = "\n".join(ast.dump(stmt) for stmt in defined[name])
        for stmt in defined[name]:
            todo.extend(node.id for node in ast.walk(stmt)
                        if isinstance(node, ast.Name) and node.id in defined
                        and node.id not in reached)
    return sorted(reached.items())


def _simulator_fingerprint_parts(sources: Mapping[str, str]) -> dict[str, str]:
    """
    Fingerprint the simulator's code, module by module.

    Each module of _SIMULATOR_MODULES is fingerprinted as the SHA-256 of its
    parsed code (ast.dump, docstrings removed, so comments, docstrings and
    layout never count), and config.py as that of the statements those
    modules read, with everything they use in turn (_config_closure). An edit
    to config.py that no simulator module reads, a dashboard constant say,
    leaves it unchanged. ast.dump's form can change between Python versions,
    so the same code under another Python can fingerprint differently.

    Args:
        sources (Mapping[str, str]): The source text of each of
            _SIMULATOR_MODULES and of "config".

    Returns:
        dict[str, str]: Module name ("config" included) -> hex digest.

    Raises:
        SyntaxError: If a source cannot be parsed.
        KeyError: If a module's source is missing.
    """
    parts: dict[str, str] = {}
    read: set[str] = set()
    for name in _SIMULATOR_MODULES:
        tree = _without_docstrings(ast.parse(sources[name]))
        read |= _config_names_read(tree)
        parts[name] = hashlib.sha256(ast.dump(tree).encode("utf-8")).hexdigest()
    digest = hashlib.sha256()
    for name, dump in _config_closure(_without_docstrings(ast.parse(sources["config"])), read):
        digest.update(f"{name}\0{dump}\0".encode())
    parts["config"] = digest.hexdigest()
    return parts


def _fingerprint_of(parts: Mapping[str, str]) -> str:
    """
    Combine the per-module digests into one fingerprint.

    Args:
        parts (Mapping[str, str]): _simulator_fingerprint_parts' result.

    Returns:
        str: The SHA-256 hex digest over every module's name and digest.
    """
    return hashlib.sha256("".join(
        f"{name}={value}\n" for name, value in sorted(parts.items())).encode()).hexdigest()


@functools.cache
def _loaded_simulator_fingerprint() -> tuple[tuple[str, str], ...]:
    """
    Fingerprint the simulator's code as it is on disk now, once per process.

    main() asks for it as soon as a run starts, so it describes the code the
    run imported moments before; cached, since the code a process runs does
    not change while it runs.

    Returns:
        tuple[tuple[str, str], ...]: (module name, hex digest) pairs, in name order.

    Raises:
        OSError: If a module's file cannot be read.
        SyntaxError: If a module cannot be parsed.
    """
    here = Path(__file__).parent
    sources = {name: (here / f"{name}.py").read_text(encoding="utf-8")
               for name in (*_SIMULATOR_MODULES, "config")}
    return tuple(sorted(_simulator_fingerprint_parts(sources).items()))


def _changed_modules(saved: Mapping[str, Any], now: Mapping[str, str]) -> list[str]:
    """
    Name the simulator files whose fingerprint differs between a saved run and this code.

    Args:
        saved (Mapping[str, Any]): The saved file's per-module digests.
        now (Mapping[str, str]): This code's.

    Returns:
        list[str]: "<module>.py" for each one that differs, in a fixed order.
    """
    return [f"{name}.py" for name in (*_SIMULATOR_MODULES, "config")
            if saved.get(name) != now.get(name)]


def _dashboard_inputs(sweep: BacktestSweep, start_date: date, start: StartingBalance,
                      series_categories: dict, period_end: date) -> dict[str, Any]:
    """
    Gather what main() hands generate_dashboard, as a full run saves it.

    Args:
        sweep (BacktestSweep): run_backtest_sweep's result.
        start_date (date): The run's first day.
        start (StartingBalance): Its starting balance and where it came from.
        series_categories (dict): Kalshi's /series listing, as the run read it.
        period_end (date): The window's last day (the UTC day the sweep finished).

    Returns:
        dict[str, Any]: One value per key of _DASHBOARD_INPUT_KEYS.
    """
    return {
        "sweep": sweep, "start_date": start_date, "initial_balance": start.dollars,
        "balance_source": start.source, "series_categories": series_categories,
        # The k the plotted trades were sized at, read back off the point
        "interval_discount": sweep.primary.k, "period_end": period_end,
    }


def _build_dashboard(inputs: Mapping[str, Any], *, risk_free, sell_workers: int) -> Path:
    """
    Write the dashboard from a run's inputs: the one call a full run and a rebuild share.

    Args:
        inputs (Mapping[str, Any]): _dashboard_inputs' result, or a saved copy of it.
        risk_free: treasury.load_risk_free_rates' bill yields.
        sell_workers (int): The most worker processes the Sell select is built in.

    Returns:
        Path: The page written.
    """
    sweep = inputs["sweep"]
    # balance_source: the header says where the starting balance came from;
    # period_end: the window's last day, the run's own even on a later rebuild
    return generate_dashboard(sweep.primary.trades, sweep.primary.equity_df,
                              inputs["start_date"], inputs["initial_balance"],
                              sweep=sweep, interval_discount=inputs["interval_discount"],
                              series_categories=inputs["series_categories"],
                              risk_free=risk_free, sell_workers=sell_workers,
                              balance_source=inputs["balance_source"],
                              period_end=inputs["period_end"])


def _build_data_page(inputs: Mapping[str, Any], *, sell_workers: int) -> None:
    """
    Rebuild the Data page (the live dashboard's Data tab) from a run's inputs, never stopping the run.

    eda.build_eda_page counts the cached day files (reusing every saved
    summary that still matches) and takes the k view from the run's sweep.
    It never raises; it is called inside a try anyway, so that whatever goes
    wrong is one WARNING and the run's exit code and backtest page are
    untouched. Reporting only: nothing the backtest computed is read back.

    Args:
        inputs (Mapping[str, Any]): _dashboard_inputs' result, or a saved copy
            of it (its sweep and its /series listing are read).
        sell_workers (int): The Sell select's worker count; the Data page
            uses at most that many, and never more than config.EDA_MAX_WORKERS.
    """
    # Before the call, so a long first build is not taken for a hang
    logging.info("Building the Data page from backtest_cache/: the first build (or one after "
                 "many new day files) reads each cached day file once and can take several "
                 "minutes (about 10 on a full cache); later builds take seconds. --no-eda "
                 "skips it")
    try:
        page = build_eda_page(sweep=inputs["sweep"],
                              series_categories=inputs["series_categories"],
                              workers=min(sell_workers, EDA_MAX_WORKERS))
    except Exception as e:
        logging.warning("Data page not built (%s); the backtest page above is unchanged",
                        api_error_summary(e), exc_info=True)
        return
    if page is None:
        logging.warning("Data page: no page could be written (the WARNING above says why); "
                        "the backtest page above is unchanged")
    else:
        logging.info("Data page written: %s (the live dashboard's Data tab)", page)


def _build_pages(inputs: Mapping[str, Any], *, risk_free, sell_workers: int,
                 data_page: bool) -> None:
    """
    Write the backtest page and then, unless told not to, the Data page: what a full run and a rebuild both end on.

    Args:
        inputs (Mapping[str, Any]): _dashboard_inputs' result, or a saved copy of it.
        risk_free: treasury.load_risk_free_rates' bill yields.
        sell_workers (int): The most worker processes the Sell select is built in.
        data_page (bool): False (--no-eda) skips the Data page.
    """
    _build_dashboard(inputs, risk_free=risk_free, sell_workers=sell_workers)
    if data_page:
        # After the backtest page, so a Data page that fails costs nothing above
        _build_data_page(inputs, sell_workers=sell_workers)


def _save_dashboard_inputs(path: Path, inputs: Mapping[str, Any],
                           parts: Mapping[str, str] | None) -> None:
    """
    Save a run's dashboard inputs for --dashboard-only, never stopping the run.

    The file is a gzip stream: one line of JSON (the format tag, the simulator
    fingerprint with its per-module digests, when it was saved, the Python
    version and the run's start date), then the inputs pickled. It is written
    to a sibling ".tmp" file renamed over the old one, so a file at the path
    is always a whole one. Saving can fail (an object that cannot be pickled,
    a full disk): one WARNING then says so, the leftover file and any earlier
    run's file are deleted, since that one no longer matches the page this run
    builds, and the run goes on to build its page.

    Args:
        path (Path): Where the file goes (PROJECT_ROOT / BACKTEST_SWEEP_FILENAME).
        inputs (Mapping[str, Any]): _dashboard_inputs' result.
        parts (Mapping[str, str] | None): The simulator's per-module digests
            when the run started; None when they could not be computed, which
            saves nothing.
    """
    tmp = path.with_name(path.name + ".tmp")
    logging.info("Saving this run's dashboard inputs to %s, for --dashboard-only", path)
    started = time.perf_counter()
    try:
        if parts is None:
            raise ValueError("the simulator's code could not be fingerprinted when the run "
                             "started (see the WARNING above)")
        header = {
            "format": BACKTEST_SWEEP_FORMAT,
            "fingerprint": _fingerprint_of(parts),
            "fingerprint_parts": dict(parts),
            "saved_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "start_date": inputs["start_date"].isoformat(),
        }
        # compresslevel 1, like the day slices: the file is large and written once
        with gzip.open(tmp, "wb", compresslevel=1) as handle:
            handle.write(json.dumps(header, sort_keys=True).encode("utf-8") + b"\n")
            pickle.dump(dict(inputs), handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
    except Exception as e:
        for leftover in (tmp, path):
            try:
                leftover.unlink(missing_ok=True)
            except OSError:
                pass
        logging.warning(
            "Could not save this run's dashboard inputs (%s): the page is still built, but "
            "--dashboard-only cannot rebuild it from this run, and any earlier run's saved "
            "file was removed since it no longer matches the page", api_error_summary(e))
        return
    logging.info(
        "Saved this run's dashboard inputs (%.1f MB in %.1f s): python3 -m "
        "kalshi_betting.backtest --dashboard-only rebuilds the page from them",
        path.stat().st_size / 1e6, time.perf_counter() - started)


def _read_header_line(handle) -> dict[str, Any]:
    """
    Read and check a saved sweep file's first line, from its open gzip stream.

    Args:
        handle: The file, opened with gzip.open(path, "rb"), at its start.

    Returns:
        dict[str, Any]: The header: its format is BACKTEST_SWEEP_FORMAT, its
            fingerprint a string and its fingerprint_parts a dict of strings.

    Raises:
        SavedSweepError: If the line is not a saved sweep's header in this format.
    """
    line = handle.readline(BACKTEST_SWEEP_HEADER_MAX_BYTES + 1)
    if not line.endswith(b"\n"):
        raise SavedSweepError("it does not start with a saved backtest's header line")
    try:
        header = json.loads(line)
    except ValueError:
        raise SavedSweepError("its header line is not JSON") from None
    if not isinstance(header, dict):
        raise SavedSweepError("its header line is not a JSON object")
    if header.get("format") != BACKTEST_SWEEP_FORMAT:
        raise SavedSweepError(f"it is in format {header.get('format')!r}, and this code "
                              f"reads {BACKTEST_SWEEP_FORMAT!r}")
    parts = header.get("fingerprint_parts")
    if (not isinstance(header.get("fingerprint"), str) or not isinstance(parts, dict)
            or not all(isinstance(k, str) and isinstance(v, str) for k, v in parts.items())):
        raise SavedSweepError("its header line names no simulator fingerprint")
    return header


def _read_saved_header(path: Path) -> dict[str, Any]:
    """
    Read a saved sweep file's header line alone, without loading the run.

    Args:
        path (Path): The saved file.

    Returns:
        dict[str, Any]: The checked header (_read_header_line).

    Raises:
        SavedSweepError: If the file cannot be read or its header is not one
            in this format.
    """
    try:
        with gzip.open(path, "rb") as handle:
            return _read_header_line(handle)
    except (OSError, EOFError, zlib.error) as e:
        # gzip.BadGzipFile is an OSError; zlib.error a damaged stream
        raise SavedSweepError(f"it cannot be read ({api_error_summary(e)})") from e


def _load_saved_inputs(path: Path) -> dict[str, Any]:
    """
    Load the dashboard inputs a full run saved, checking their shape.

    Loading a pickle runs the code its objects name, so only a file this
    program wrote should ever be loaded: the file lives in the checkout's own
    folder and is gitignored.

    Args:
        path (Path): The saved file.

    Returns:
        dict[str, Any]: One value per key of _DASHBOARD_INPUT_KEYS, the sweep
            a BacktestSweep and both dates real dates.

    Raises:
        SavedSweepError: If the file cannot be read or unpickled with this
            code, or what it holds is not a run's dashboard inputs.
    """
    try:
        with gzip.open(path, "rb") as handle:
            _read_header_line(handle)
            inputs = pickle.load(handle)
    except SavedSweepError:
        raise
    except Exception as e:
        # A truncated or damaged stream, or a class this code no longer has
        raise SavedSweepError(f"its saved run could not be loaded ({api_error_summary(e)})") from e
    if not isinstance(inputs, dict) or set(inputs) != _DASHBOARD_INPUT_KEYS:
        raise SavedSweepError("it does not hold a run's dashboard inputs")
    if not (isinstance(inputs["sweep"], BacktestSweep)
            and isinstance(inputs["start_date"], date)
            and isinstance(inputs["period_end"], date)):
        raise SavedSweepError("its sweep or its dates are not what a run saves")
    return inputs


def _without_families(sweep: BacktestSweep, *, cap_sweep: bool, sell_sweep: bool,
                      rebalance_sweep: bool,
                      move_sweep: bool) -> tuple[BacktestSweep, list[str]]:
    """
    Drop lazy families from a loaded sweep, as a run given the matching --no-*-sweep flags builds it.

    Each flag that is False drops its family, so the rebuilt page leaves its
    select disabled and skips its simulations: a quick page for checking a
    change. Without the size-cap sweep, a run builds the Sell and Rebalance
    families at its own cap alone, so those that stay are narrowed to it
    too. Without the minimum price movements, the Rebalance family stays and
    carries none.

    Args:
        sweep (BacktestSweep): The loaded sweep; left as it is.
        cap_sweep, sell_sweep, rebalance_sweep, move_sweep (bool): Whether
            each family stays.

    Returns:
        tuple[BacktestSweep, list[str]]: The sweep the page is built from
            (the same object when nothing is dropped), and a short note per
            family dropped, for the log.
    """
    changes: dict[str, Any] = {}
    dropped: list[str] = []
    if not cap_sweep:
        if sweep.cap_sweep is not None or sweep.tier_off_cap_sweep is not None:
            changes.update(cap_sweep=None, tier_off_cap_sweep=None)
            dropped.append("the size caps (--no-cap-sweep)")
        # The other lazy families at the run's own cap, as a run without the
        # size-cap sweep builds them
        for name in ("sell_sweep", "rebalance_sweep"):
            family = getattr(sweep, name)
            if family is not None and family.caps != (family.primary_cap,):
                changes[name] = replace(family, caps=(family.primary_cap,))
    if not sell_sweep and sweep.sell_sweep is not None:
        changes["sell_sweep"] = None
        dropped.append("Sell (--no-sell-sweep)")
    if not rebalance_sweep and sweep.rebalance_sweep is not None:
        changes["rebalance_sweep"] = None
        dropped.append("Rebalance (--no-rebalance-sweep)")
    elif (not move_sweep and sweep.rebalance_sweep is not None
          and sweep.rebalance_sweep.min_moves):
        family = changes.get("rebalance_sweep", sweep.rebalance_sweep)
        changes["rebalance_sweep"] = replace(family, min_moves=())
        dropped.append("Min. price movement (--no-move-sweep)")
    return (replace(sweep, **changes) if changes else sweep), dropped


def _configure_logging() -> None:
    """
    Send the log to kalshi_backtest.log in PROJECT_ROOT, as every run of this CLI does.

    No console handler: nothing reaches stdout.
    """
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


def _sell_workers(requested: int | None) -> int:
    """
    The worker processes the dashboard builds its Sell select in.

    Args:
        requested (int | None): --sell-workers, or None when not given.

    Returns:
        int: The flag's value, or one less than the CPU count (the main
            process waits on them), at least 1 and at most
            config.DASHBOARD_SELL_MAX_WORKERS.
    """
    if requested is not None:
        return requested
    return max(1, min((os.cpu_count() or 1) - 1, DASHBOARD_SELL_MAX_WORKERS))


def _log_how_to_open() -> None:
    """Close a run, or a rebuild, by saying how to view the page and use its buttons."""
    logging.info("Open the HTML file in a browser to view the interactive charts.")
    # How a scenario on the page becomes the live defaults, or is traded: the
    # page cannot write files or start runs, so its buttons open the defaults
    # server's pages (started by ./start_dashboard.sh, beside the live
    # dashboard, whose Backtest tab shows this page)
    logging.info("To save the filter bar's scenario as the live defaults, or to trade: run "
                 "./start_dashboard.sh (it starts the defaults server and the live dashboard, "
                 "whose Backtest tab is this page), then use the filter bar's Save as live "
                 "defaults… or Trade using defaults… button.")


def _rebuild_dashboard(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """
    Rebuild the dashboard from the inputs the last full run saved (--dashboard-only).

    Fetches nothing and makes no Kalshi request: it loads the saved file,
    reads the T-bill yields again (treasury.load_risk_free_rates) and calls
    generate_dashboard as the run did, with this command's --sell-workers,
    then rebuilds the Data page from the saved sweep and listing unless
    --no-eda (_build_pages, as a full run ends).
    Each --no-*-sweep flag drops its lazy family first (_without_families).

    The lazy families simulate their cells while the page is built, with the
    code as it is then, so after a change to the simulator (the modules of
    _SIMULATOR_MODULES or the config they read) the run's own figures and
    those cells would come from two versions of the code. A file whose
    simulator fingerprint differs from this code's is therefore refused
    (exit 2) unless --accept-changed-simulator is given, and then a WARNING
    names the files that changed.

    Exits 2, before logging is configured, for a run-only flag, a bad
    --sell-workers, a missing file or a refused fingerprint; exits 1 for a
    file that cannot be read or loaded (its header before logging is
    configured, the run itself after, with an ERROR in the log).

    Args:
        parser (argparse.ArgumentParser): The CLI's parser, for its errors.
        args (argparse.Namespace): The parsed arguments.
    """
    given = [flag for dest, flag in _RUN_ONLY_FLAGS
             if getattr(args, dest) != parser.get_default(dest)]
    if given:
        parser.error(f"{', '.join(given)} cannot be used with --dashboard-only: the page is "
                     "rebuilt from the saved run, with that run's settings")
    if args.sell_workers is not None and args.sell_workers < 1:
        parser.error("--sell-workers must be a positive integer")
    path = PROJECT_ROOT / BACKTEST_SWEEP_FILENAME
    if not path.exists():
        parser.exit(2, f"backtest: no saved backtest at {path}: a full run "
                       "(python3 -m kalshi_betting.backtest) saves one\n")
    try:
        header = _read_saved_header(path)
    except SavedSweepError as e:
        raise SystemExit(f"backtest: {path} cannot be used: {e}; run the backtest again "
                         "to save a new one") from None
    try:
        now_parts = dict(_loaded_simulator_fingerprint())
    except (OSError, SyntaxError) as e:
        raise SystemExit("backtest: the simulator's code could not be fingerprinted "
                         f"({api_error_summary(e)})") from None
    changed = _changed_modules(header["fingerprint_parts"], now_parts)
    python_note = ("" if header.get("python") == platform.python_version()
                   else f" (under Python {header.get('python')}; this is Python "
                        f"{platform.python_version()}, and the fingerprint reads code "
                        "through Python's own parser, so that alone can change it)")
    if changed and not args.accept_changed_simulator:
        parser.exit(2, (
            f"backtest: the run saved at {path} ({header.get('saved_at')}) was simulated by "
            f"other code: {', '.join(changed)} changed since it was saved{python_note}. "
            "A rebuild simulates the size caps, Sell and Rebalance "
            "cells while the page is built, with the code as it is now, so the run's own "
            "figures and those cells would come from two different versions of the code. "
            "Run the backtest again, or pass --accept-changed-simulator to build the page "
            "anyway.\n"))

    _configure_logging()
    logging.info("Dashboard rebuild from %s: saved %s by a run that started %s (%.1f MB)",
                 path, header.get("saved_at"), header.get("start_date"),
                 path.stat().st_size / 1e6)
    if changed:
        logging.warning(
            "The simulator's code changed since this run was saved (%s)%s: the page's "
            "lazily simulated cells (size caps, Sell, Rebalance) "
            "come from this code and its other figures from the run's "
            "(--accept-changed-simulator)", ", ".join(changed), python_note)
    started = time.perf_counter()
    try:
        inputs = _load_saved_inputs(path)
    except SavedSweepError as e:
        message = f"{path} cannot be used: {e}; run the backtest again to save a new one"
        logging.error("Dashboard not rebuilt: %s", message)
        raise SystemExit(f"backtest: {message}") from None
    logging.info("Loaded the saved run in %.1f s", time.perf_counter() - started)
    sweep, dropped = _without_families(
        inputs["sweep"], cap_sweep=not args.no_cap_sweep, sell_sweep=not args.no_sell_sweep,
        rebalance_sweep=not args.no_rebalance_sweep, move_sweep=not args.no_move_sweep)
    if dropped:
        logging.info("Dashboard rebuilt without: %s", "; ".join(dropped))
    # The T-bill yields, read again: never raises (see the full run's read)
    risk_free = load_risk_free_rates()
    # The backtest page, then the Data page (unless --no-eda), as a full run ends
    _build_pages({**inputs, "sweep": sweep}, risk_free=risk_free,
                 sell_workers=_sell_workers(args.sell_workers), data_page=not args.no_eda)
    _log_how_to_open()


def _build_parser() -> argparse.ArgumentParser:
    """
    Build the CLI's argument parser (main() parses with it; tests read its options).

    Returns:
        argparse.ArgumentParser: Every option of `python3 -m kalshi_betting.backtest`.
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
    # The same idiom for the scheduled close: None is the "no override"
    # sentinel _prepare_candidates resolves at call time
    parser.add_argument(
        "--scheduled-close", action=argparse.BooleanOptionalAction, default=None,
        help="Read each settled market's scheduled close (the close a live run "
             "saw, from a rule learned per series from the captures under "
             "backtest_cache/scheduled_closes/) for days to maturity, the "
             "ranking, a cross-event pair's order, gap and horizon and the "
             "same-title close gate; a series with no rule keeps the realized "
             "close (default: config.BACKTEST_SCHEDULED_CLOSE, currently "
             f"{'on' if BACKTEST_SCHEDULED_CLOSE else 'off'}). Backtest only.",
    )
    # The stale-quote check: a limit in hours for this run, or off; neither
    # given reads config.BACKTEST_STALE_QUOTE_MAX_AGE_SECONDS at call time
    stale = parser.add_mutually_exclusive_group()
    stale.add_argument(
        "--stale-quote-hours", type=float, default=None, metavar="N",
        help="An entry, a sale or a Kelly trim at a checkpoint needs each of its "
             "markets to have traded, or moved its quote, within N hours before "
             "it (default: config.BACKTEST_STALE_QUOTE_MAX_AGE_SECONDS, currently "
             f"{_stale_quote_text(BACKTEST_STALE_QUOTE_MAX_AGE_SECONDS)}). Backtest only.",
    )
    stale.add_argument(
        "--no-stale-quote-check", action="store_true",
        help="Trade on a quote however long it has been unchanged, for this run. "
             "Backtest only.",
    )
    # The volume cap: a percent for this run, or off; neither given reads
    # config.BACKTEST_MAX_VOLUME_SHARE at call time
    volume = parser.add_mutually_exclusive_group()
    volume.add_argument(
        "--max-volume-share", type=float, default=None, metavar="PCT",
        help="A buy or a sale at a checkpoint may take at most PCT percent (above 0, "
             "at most 100) of the contracts that traded in each of its markets over the "
             "next day (default: config.BACKTEST_MAX_VOLUME_SHARE, currently "
             # argparse formats help with %, so the percent sign is doubled
             f"{_volume_cap_text(BACKTEST_MAX_VOLUME_SHARE).replace('%', '%%')}). It reads "
             "hours a live run "
             "cannot see and only ever makes a trade smaller. Backtest only.",
    )
    volume.add_argument(
        "--no-volume-cap", action="store_true",
        help="Hold a trade only to the modeled book, for this run. Backtest only.",
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
        "--no-rebalance-sweep", action="store_true",
        help="Skip the dashboard's Rebalance select and its Min. price movement select "
             "(at each weekly checkpoint, top a held pair up to its Kelly share of the "
             "portfolio when it has room and sell it down to it when it has grown past "
             "it): their simulations run only while the dashboard is built, so skipping "
             "them makes that step faster and leaves both selects disabled. Backtest "
             "only — a live run adds to and trims held pairs by its own saved settings, "
             "which the dashboard's Save as live defaults… can set to the choice shown "
             "here",
    )
    parser.add_argument(
        "--no-move-sweep", action="store_true",
        help="Skip the dashboard's Min. price movement select (with Rebalance yes, add "
             "to or trim a held pair only once its total price has moved a chosen "
             "number of cents, up or down, since the last trade on it); Rebalance "
             "itself is still built. Its simulations run only while the dashboard is "
             "built. Backtest only — a live run's rebalancing (add-ons and trims) "
             "waits for a price move by its own saved min_price_move (main.py "
             "--min-price-move for one run)",
    )
    parser.add_argument(
        "--sell-workers", type=int, default=None, metavar="N",
        help="Worker processes the dashboard simulates its Sell select in "
             "(default: one less than this machine's CPU count, at most "
             f"{DASHBOARD_SELL_MAX_WORKERS} — config.DASHBOARD_SELL_MAX_WORKERS); "
             "1 runs them in the main process. Each worker holds one spread band's "
             "entries, so memory grows with the count",
    )
    parser.add_argument(
        "--no-eda", action="store_true",
        help="Skip the Data page (eda_dashboard.html, the live dashboard's Data tab). By "
             "default it is rebuilt after the backtest page, also with --dashboard-only: the "
             "first build (or one after many new day files) reads each cached day file once "
             "and can take several minutes (about 10 on a full cache); later builds take "
             "seconds. A failure there is one WARNING and changes neither the exit code nor "
             "the backtest page",
    )
    parser.add_argument(
        "--dashboard-only", action="store_true",
        help="Rebuild the dashboard from the inputs the last full run saved "
             f"({BACKTEST_SWEEP_FILENAME} in the repo root) instead of running the "
             "backtest: no fetch and no Kalshi request. --sell-workers and --no-eda apply, "
             "and each --no-*-sweep flag but --no-band-sweep drops its family from the "
             "saved run for a quicker page; the flags that shape a run are refused",
    )
    parser.add_argument(
        "--accept-changed-simulator", action="store_true",
        help="With --dashboard-only: build the page even though the simulator's code "
             "(backtester, strategy, scanner, depth_model, historical, or the config they "
             "read) changed since the run was saved. The size caps, Sell and Rebalance "
             "cells are simulated while the page is built, so they then come from this "
             "code and the run's other figures from the old code",
    )
    return parser


def main() -> None:
    """
    CLI entry point for the Kalshi backtester.

    Parses command-line arguments (--start-date, --balance, --no-cache,
    --max-horizon-days, --interval-discount, --no-sweep,
    --same-event-ladders / --no-same-event-ladders, --scheduled-close /
    --no-scheduled-close, --stale-quote-hours / --no-stale-quote-check,
    --max-volume-share / --no-volume-cap, --spread-min,
    --spread-max, --no-band-sweep, --no-cap-sweep, --no-sell-sweep,
    --no-rebalance-sweep, --no-move-sweep, --sell-workers, --no-eda,
    --dashboard-only, --accept-changed-simulator); with --dashboard-only it
    hands over to _rebuild_dashboard, which rebuilds the page from the saved
    inputs and runs nothing below. Otherwise it configures
    logging, constructs historical and live
    Kalshi API clients, works out the starting balance (--balance, or else the
    account's value now, read through _account_starting_balance; a read that
    fails, or comes to nothing, stops the run with exit 1 before the
    fetch), runs the full backtest simulation via run_backtest_sweep(), saves
    what the dashboard is handed (_save_dashboard_inputs, for a later
    --dashboard-only), and generates
    an interactive HTML dashboard via generate_dashboard(), then the Data
    page (eda.build_eda_page) unless --no-eda. Logs a summary
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
    simulated sell family (result.sell_sweep) and Rebalance family
    (result.rebalance_sweep, with its minimum price movements) are read the
    same way.
    """
    parser = _build_parser()
    args = parser.parse_args()
    if args.dashboard_only:
        # Rebuild the page from the saved run and stop: nothing below runs
        _rebuild_dashboard(parser, args)
        return
    if args.accept_changed_simulator:
        parser.error("--accept-changed-simulator applies only with --dashboard-only")
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
    if args.stale_quote_hours is not None and not (
            math.isfinite(args.stale_quote_hours) and args.stale_quote_hours > 0):
        parser.error("--stale-quote-hours must be a positive number of hours")
    if args.max_volume_share is not None and not (
            math.isfinite(args.max_volume_share) and 0 < args.max_volume_share <= 100):
        parser.error("--max-volume-share must be a percent above 0 and at most 100")

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

    _configure_logging()
    # The simulator's code as this run imported it, for the saved sweep's
    # stamp; a failure saves no sweep file, never stops the run
    try:
        simulator_parts: dict[str, str] | None = dict(_loaded_simulator_fingerprint())
    except (OSError, SyntaxError) as e:
        simulator_parts = None
        logging.warning("The simulator's code could not be fingerprinted (%s): this run "
                        "saves no inputs for --dashboard-only", api_error_summary(e))

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
    # ... and for the scheduled close, named against the config when the run
    # departs from it
    effective_scheduled = (BACKTEST_SCHEDULED_CLOSE if args.scheduled_close is None
                           else args.scheduled_close)
    scheduled_echo = "on" if effective_scheduled else "off"
    if bool(effective_scheduled) != bool(BACKTEST_SCHEDULED_CLOSE):
        scheduled_echo += f" (config: {'on' if BACKTEST_SCHEDULED_CLOSE else 'off'})"
    # ... and for the stale-quote check: the limit this run hands the sweep
    # (None: the config's; False: off), echoed against the config's
    stale_quote_max_age = (False if args.no_stale_quote_check
                           else None if args.stale_quote_hours is None
                           else args.stale_quote_hours * 3600.0)
    effective_stale = (BACKTEST_STALE_QUOTE_MAX_AGE_SECONDS if stale_quote_max_age is None
                       else None if stale_quote_max_age is False else stale_quote_max_age)
    stale_echo = _stale_quote_text(effective_stale)
    if stale_echo != _stale_quote_text(BACKTEST_STALE_QUOTE_MAX_AGE_SECONDS):
        stale_echo += f" (config: {_stale_quote_text(BACKTEST_STALE_QUOTE_MAX_AGE_SECONDS)})"
    # ... and for the volume cap (a share for the sweep; None: config; False: off)
    max_volume_share = (False if args.no_volume_cap
                        else None if args.max_volume_share is None
                        else args.max_volume_share / 100.0)
    effective_volume = (BACKTEST_MAX_VOLUME_SHARE if max_volume_share is None
                        else None if max_volume_share is False else max_volume_share)
    volume_echo = _volume_cap_text(effective_volume)
    if volume_echo != _volume_cap_text(BACKTEST_MAX_VOLUME_SHARE):
        volume_echo += f" (config: {_volume_cap_text(BACKTEST_MAX_VOLUME_SHARE)})"
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
    # ON by default too, and lazy the same way: the dashboard simulates it,
    # so the opt-out saves only the dashboard step
    sell_sweep = not args.no_sell_sweep
    # ... and so is the Rebalance select's family
    rebalance_sweep = not args.no_rebalance_sweep
    # ... and, as part of that family, the minimum price movements its
    # add-ons and trims can be made to wait for (the Min. price movement
    # select): on with it unless opted out, and off whenever the family is
    move_sweep = rebalance_sweep and not args.no_move_sweep
    # The dashboard simulates the Sell select in worker processes: one less
    # than the CPUs (the main process waits on them), within the config bound
    sell_workers = _sell_workers(args.sell_workers)
    # The saved live defaults' rule and filter, for the echo only: a read of its
    # own, separate from run_backtest_sweep's (which logs the INFO or WARNING).
    # Fails soft, unlike the checks above (one echo clause must not abort a run)
    try:
        _live = live_defaults()
        live_rule_echo = describe_time_series_rule(_live.tier_floors, _live.spread_band)
        # The category/tag filter only when one is set, in the live run's words
        if _live.exclude_categories is not None or _live.tags is not None:
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
        "Backtest config: start=%s | balance=$%.2f | cache=%s | scheduled close=%s "
        "| k=%.3f | stale quotes=%s | volume cap=%s | ladders=%s | spread band=%g-%g "
        "| band sweep=%s | cap sweep=%s "
        "| sell sweep=%s | rebalance sweep=%s | live rule=%s | fills=%s",
        start_date, start.dollars, "on" if use_cache else "off", scheduled_echo,
        effective_k, stale_echo, volume_echo, ladders_echo, echo_floor, echo_ceiling,
        "on" if band_sweep else "off", "on" if cap_sweep else "off",
        f"on ({sell_workers} worker process{'' if sell_workers == 1 else 'es'})"
        if sell_sweep else "off",
        # A default run reads "on"; the family without its minimum price
        # movements says so
        "off" if not rebalance_sweep else "on" if move_sweep
        else "on (no minimum price movements)",
        live_rule_echo, _fills_echo(depth_model),
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

    # Kalshi's /series listing, read once: the same-title sport check files
    # each series by it (DR-79), and the dashboard files each trade under its
    # series' category and tags. One cached read-only GET of /series, which
    # never raises (it falls back to a stale copy or to {})
    series_categories = load_series_categories(live_client)
    # ... and each series' fee multiplier from the same cache file (the line
    # above's refresh writes both; upgrade=True makes the one fetch that adds
    # the fees to a current file from before they were kept). Carried by the
    # sweep; no fee reads it yet
    fee_multipliers = load_series_fee_multipliers(live_client, upgrade=True)
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
        # Whether a market's close is read as a live run read it (None: config)
        scheduled_close=args.scheduled_close,
        # How recently each market's quote must have moved (None: config; False: off)
        stale_quote_max_age=stale_quote_max_age,
        # The share of the next day's volume a buy or sale may take (None: config; False: off)
        max_volume_share=max_volume_share,
        spread_band=spread_band,
        band_sweep=band_sweep,
        # The dashboard's "Tier floors: off" view: the band sweep's
        # tier-bound bands simulated again with the tiers off (needs the grid)
        tier_off_sweep=band_sweep,
        cap_sweep=cap_sweep,
        # The dashboard's Sell select, simulated lazily when the page is built
        # (no simulation during the run)
        sell_sweep=sell_sweep,
        # ... and its Rebalance select, lazy the same way
        rebalance_sweep=rebalance_sweep,
        # ... with the Min. price movement select's minimums, part of that family
        move_sweep=move_sweep,
        # Every trade's order book is built from this table (None: top of book)
        depth_model=depth_model,
        # The listing the same-title label check files each series by
        series_categories=series_categories,
        # Each series' fee multiplier (carried only)
        fee_multipliers=fee_multipliers,
    )  # returns BacktestSweep — primary point, one point per swept k and the calibration, plus the band-sweep payload (scenarios, same_title_point, calibrations_by_band) and the tier-floors-off family (tier_off_scenarios, tier_off_calibrations_by_band) unless --no-band-sweep, the lazy size-cap sweeps (cap_sweep, and tier_off_cap_sweep with the band sweep) unless --no-cap-sweep, and the lazy sell and Rebalance families (sell_sweep, rebalance_sweep) unless --no-sell-sweep / --no-rebalance-sweep
    # What the dashboard is handed, saved before the page is built, so
    # --dashboard-only can rebuild the page (or finish a build that stopped)
    # without running the backtest again. period_end is the window's last day,
    # read now so the page and any later rebuild name the same day
    inputs = _dashboard_inputs(result, start_date, start, series_categories,
                               datetime.now(UTC).date())
    _save_dashboard_inputs(PROJECT_ROOT / BACKTEST_SWEEP_FILENAME, inputs, simulator_parts)

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
    # and scenario explorer). The sell and Rebalance families
    # (result.sell_sweep and result.rebalance_sweep) are lazy the same way.
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
    # sell and Rebalance families (sell_sweep and rebalance_sweep, None under
    # --no-sell-sweep and --no-rebalance-sweep) and the header's run-settings
    # line;
    # interval_discount is the resolved k these trades were sized at, which
    # the Risk section's Kelly scatter must price on
    #
    # series_categories (read above, the same copy the sweep's pairs were
    # filed by) files each trade under Kalshi's own series category and tags
    # for the Returns Decomposition breakdown.

    # The T-bill yields every Sharpe and Sortino on the page subtracts; never
    # raises (falls back to the saved copy, then "unavailable"), but an
    # unresponsive single-address host can cost about 4 minutes first
    risk_free = load_risk_free_rates()
    # The one call a rebuild from the saved inputs makes too: the backtest
    # page, then the Data page with this run's sweep and listing (--no-eda skips it)
    _build_pages(inputs, risk_free=risk_free, sell_workers=sell_workers,
                 data_page=not args.no_eda)
    _log_how_to_open()


if __name__ == "__main__":
    main()
