"""
File: backtest.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Command-line entry point for the Kalshi backtester. Parses CLI
    arguments (--start-date, --balance, --no-cache, --max-horizon-days,
    --interval-discount, --no-sweep, --same-event-ladders /
    --no-same-event-ladders, --spread-min, --spread-max, --no-band-sweep,
    --no-cap-sweep), configures logging to kalshi_backtest.log, constructs the necessary API
    clients, delegates the full backtest simulation to
    backtester.run_backtest_sweep(), and then calls
    dashboard.generate_dashboard() to produce the interactive HTML report.
    Prints a summary of key metrics (trade count, win rate, total return) to
    the log on completion, closed on every run by what settled-market corpus
    the run read (its assembly time, whether it was cached, and the archive
    cutoff as of assembly — a WARNING when the window starts at or after it).

Dependencies:
    Imports run_backtest_sweep, BacktestSweep and max_trades_simulated (the
    closing corpus line tests a stamped post-cutoff verdict against the run's
    own trades with it — over the eager points; the dashboard header adds the
    size-cap points its filter walk simulates) from backtester.py,
    generate_dashboard from dashboard.py, and build_historical_client /
    build_prod_live_client / load_series_categories (the dashboard's
    returns-by-category labels) from historical.py. Imports from config.py:
    PROJECT_ROOT,
    TIME_SERIES_INTERVAL_PROB_DISCOUNT and TIME_SERIES_SAME_EVENT_LADDERS
    (the pre-fetch echo), the deadline-gap tier constants
    MIN_PRICE_DIFF_SHORT_GAP, MIN_PRICE_DIFF_LONG_GAP, SHORT_DEADLINE_GAP_DAYS,
    MAX_DEADLINE_GAP_DAYS and PRICE_EPSILON (the spread-band tier WARNING),
    and the backtest band helpers time_series_spread_band and
    time_series_spread_too_wide — backtest is one of the four band readers
    tests/test_strategy.py::TestTimeSeriesKellyParity::
    test_ast_live_path_reads_no_band allows. Entry point for
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

    That setting reaches kalshi_backtest.log (the pre-fetch echo below and
    run_backtest_sweep's resolved line) AND the HTML dashboard's own page
    header: dashboard._run_settings_html(sweep) prints "same-event ladders:
    on / off / not recorded" (BacktestSweep.same_event_ladders) beside the
    primary spread band, under the "Period:" line, above every section — so a
    ladder-enabled run's dashboard is no longer indistinguishable from a
    switch-off one and needs no hand labelling. Unlike DR-66b's
    subtitle-coverage caveat, this setting is chosen by the operator on the
    command line rather than discovered by the run, which is why it is named
    rather than banner-flagged.

    --spread-min/--spread-max set the PRIMARY scenario's backtest-only
    time-series spread band (floor, ceiling) on pB - pA. Once
    config.time_series_spread_band() has resolved and validated it,
    backtester._find_entry layers the floor on the deadline-gap tier through
    config.min_price_diff_for_gap(spread_min=) and refuses a spread above the
    ceiling through config.time_series_spread_too_wide(). Live trading never
    reads a band — it filters on the tier alone. Either flag may be given
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
    grid ceiling sits at or below either tier); --no-band-sweep skips that
    grid (band_sweep=False): the primary scenario still runs, but the
    dashboard's scenario explorer has no scenarios to show.

    The per-trade size-cap sweep is ON by default too (cap_sweep=True): the
    result carries a lazy backtester.CapSweep over backtester.SIZE_CAP_SWEEP
    (5%..95% and no cap), whose cells are simulated only when a report reads
    them — the run itself simulates nothing extra, and every point it returns,
    the summary block included, is sized at the run's own cap,
    config.BUDGET_FRACTION. The dashboard reads every cell as it is built,
    for its filter bar's Size cap select, the Interval Discount section and
    the scenario explorer's cap axis. --no-cap-sweep returns
    cap_sweep=None, and the dashboard then offers the run's own cap only — a
    far smaller page, built far faster. Like the band and k sweeps it is
    backtest-only: live sizing reads config.BUDGET_FRACTION and nothing here
    writes config.py.
"""
import argparse
import logging
import logging.handlers
from datetime import UTC, date, datetime

from .backtester import BacktestSweep, max_trades_simulated, run_backtest_sweep
from .config import (
    MAX_DEADLINE_GAP_DAYS,
    MIN_PRICE_DIFF_LONG_GAP,
    MIN_PRICE_DIFF_SHORT_GAP,
    PRICE_EPSILON,
    PROJECT_ROOT,
    SHORT_DEADLINE_GAP_DAYS,
    TIME_SERIES_INTERVAL_PROB_DISCOUNT,
    TIME_SERIES_SAME_EVENT_LADDERS,
    time_series_spread_band,
    time_series_spread_too_wide,
)
from .dashboard import generate_dashboard
from .historical import build_historical_client, build_prod_live_client, load_series_categories


def _log_corpus_provenance(sweep: BacktestSweep) -> None:
    """
    Close the run's report with what settled-market corpus it read.

    The "Period:" line prints start_date → today (the simulated window), but
    the corpus holds no market settled after its assembly, and a cached re-run
    reads a corpus an earlier run assembled (DR-13). A window at or after the
    archive cutoff can enter no trade at all (M2): historical.py logs that as
    a WARNING at fetch time (and, "as of assembly", on a cache hit), which on
    a long run sits far above the result it explains, so it is repeated here,
    beside it. Logged on every run, "not recorded" included — absence must
    never be the only signal (DR-66). Worded as a bound, not a cause: such a
    window may also have formed no pairs at all. And a stamped verdict can go
    stale once the cutoff moves past start_date, so it is read beside
    backtester.max_trades_simulated: if any simulated point traded, the
    verdict is reported as stale instead of repeated. That counts the EAGER
    points only; the dashboard header, built after this line, tests the
    larger of it and the size-cap points its filter walk simulates, so the
    two agree unless only a size-cap scenario traded — then the page calls
    the verdict stale while this line still repeats it.

    Takes the sweep WHOLE, like dashboard._section_interval_discount, so the
    provenance and the trade counts it is judged against cannot drift apart.

    Args:
        sweep (BacktestSweep): The run's result. Its corpus_provenance is None
            when not recorded (no corpus was fetched, or it did not come from
            an assembled cache); a legacy settled_markets_*.json hit carries
            its file time (legacy=True) and no cutoff.
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
    elif provenance.legacy:
        assembled = (f"last written {provenance.assembled_at:%Y-%m-%d %H:%M} UTC "
                     "(a legacy cache's file time), holding no market settled "
                     "after that")
    else:
        assembled = (f"assembled {provenance.assembled_at:%Y-%m-%d %H:%M} UTC, "
                     "holding no market settled after that")
    if provenance.archive_cutoff is not None:
        cutoff = f"{provenance.archive_cutoff:%Y-%m-%d}"
    elif provenance.legacy:
        cutoff = "not recorded (the legacy format records none)"
    else:
        cutoff = "not recorded"
    logging.info(
        "Settled-market corpus: %s (%s); archive cutoff at assembly: %s",
        assembled,
        "served from an earlier run's cache; --no-cache extends it"
        if provenance.from_cache else "assembled by this run",
        cutoff,
    )
    if not provenance.post_cutoff:
        return
    # A trade at any simulated point disproves "no trade could be entered".
    # The eager points only: the dashboard header also counts the size-cap
    # points its walk simulates, so the two agree unless only one of those traded
    traded = max_trades_simulated(sweep)
    if traded:
        logging.warning(
            "The archive cutoff recorded %s is at or after this window's start "
            "date, which would mean no trade could be entered — but this run "
            "entered trades (up to %d in one simulated scenario), so that "
            "verdict is stale: the cutoff has since moved past the start date. "
            "--no-cache re-reads the cutoff and re-stamps the cache.",
            "at this corpus's assembly" if provenance.from_cache else "by this run",
            traded,
        )
        return
    logging.warning(
        "This window starts at or after the archive cutoff as of its "
        "corpus's assembly — post-cutoff markets have no historical "
        "candlesticks, so no trade could be entered whatever pairs formed; "
        "a zero-trade result here is structural, not a strategy result%s",
        " (a cached run does not re-read the cutoff; --no-cache re-checks it)"
        if provenance.from_cache else "",
    )


def main() -> None:
    """
    CLI entry point for the Kalshi backtester.

    Parses command-line arguments (--start-date, --balance, --no-cache,
    --max-horizon-days, --interval-discount, --no-sweep,
    --same-event-ladders / --no-same-event-ladders, --spread-min,
    --spread-max, --no-band-sweep, --no-cap-sweep), configures logging, constructs
    historical and live Kalshi API clients, runs the full backtest
    simulation via run_backtest_sweep(), and generates an interactive HTML
    dashboard via generate_dashboard(). Logs a summary table of key metrics to
    kalshi_backtest.log on completion (this module installs only a
    RotatingFileHandler, no console handler, so nothing reaches stdout).

    The summary block reports the PRIMARY point of the sweep — the run at the
    effective interval discount, the primary spread band and the run's own
    per-trade size cap — so a default run's summary block reads exactly as
    the plain run_backtest() path's did. The other swept discounts and bands
    exist only for the dashboard's page-wide filter bar (whose k select the
    Interval Discount section follows), its scenario explorer and the
    calibration report, and the lazily simulated size caps
    (result.cap_sweep) only for the dashboard, which reads every cell as the
    page is built (its filter bar, Interval Discount section and scenario
    explorer).
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
             "TIME_SERIES_SAME_EVENT_LADDERS). Affects the backtest only — the "
             "live finder binds that constant at import.",
    )
    # Each side is independently optional; the omitted one resolves from
    # config's own default band (see the validation block below), not from a
    # literal (0.0, 1.0) here — which is why the help names the constant
    # rather than a number. Backtest-only — nothing on the live path reads a
    # spread band at all.
    parser.add_argument(
        "--spread-min", type=float, default=None, metavar="X",
        help="Time-series spread-band FLOOR (0-1) for the primary scenario; "
             "default: config.BACKTEST_DEFAULT_SPREAD_BAND's floor (the "
             "deadline-gap tier alone while that floor is 0). Given alone, "
             "the ceiling comes from that default. Backtest only — live "
             "trading never reads it.",
    )
    parser.add_argument(
        "--spread-max", type=float, default=None, metavar="Y",
        help="Time-series spread-band CEILING (0-1) for the primary "
             "scenario; default: config.BACKTEST_DEFAULT_SPREAD_BAND's "
             "ceiling (no ceiling while it is 1). Given alone, the floor "
             "comes from that default. Backtest only.",
    )
    parser.add_argument(
        "--no-band-sweep", action="store_true",
        help="Skip the spread-band grid; the dashboard's scenario explorer "
             "is not computed",
    )
    parser.add_argument(
        "--no-cap-sweep", action="store_true",
        help="Skip the per-trade size-cap sweep: the dashboard offers the "
             "run's own cap (config.BUDGET_FRACTION) only — a far smaller, "
             "faster page. Backtest only — live sizing always reads "
             "config.BUDGET_FRACTION",
    )
    args = parser.parse_args()
    if args.max_horizon_days is not None and args.max_horizon_days < 1:
        parser.error("--max-horizon-days must be a positive integer")
    if args.interval_discount is not None and not (0.0 <= args.interval_discount <= 1.0):
        parser.error("--interval-discount must be between 0 and 1")
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

    logging.info(
        "Backtest config: start=%s | balance=$%.2f | cache=%s | k=%.3f | ladders=%s "
        "| spread band=%g-%g | band sweep=%s | cap sweep=%s",
        start_date, args.balance, "on" if use_cache else "off", effective_k,
        "on" if effective_ladders else "off", echo_floor, echo_ceiling,
        "on" if band_sweep else "off", "on" if cap_sweep else "off",
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
        spread_band=spread_band,
        band_sweep=band_sweep,
        cap_sweep=cap_sweep,
    )  # returns BacktestSweep — primary point, one point per swept k and the calibration, plus the band-sweep payload (scenarios, same_title_point, calibrations_by_band) unless --no-band-sweep, and the lazy size-cap sweep (cap_sweep) unless --no-cap-sweep
    # Everything below reports the PRIMARY point, so the summary block and the
    # dashboard's other six sections read exactly as they did before the sweep
    # existed. The remaining k points and the band-sweep payload are consumed
    # only by the dashboard's filter bar (whose k select the Interval Discount
    # section follows) and its scenario explorer, and the lazy size-cap sweep
    # (result.cap_sweep, simulated only when a cell is read) only by the
    # dashboard, whose one grid walk reads every cell as the page is built
    # (filter bar, Interval Discount section and scenario explorer).
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

    # After either branch: the window (the Period line, when printed) runs to
    # today but the corpus only to its assembly, and a post-cutoff window's
    # zero is structural — say so beside the result rather than only at the
    # top of a long log (DR-13, M2)
    _log_corpus_provenance(result)

    # generate_dashboard() already logs "Dashboard written: %s" itself (BS-26) —
    # don't duplicate that line here, just point the user at the file.
    #
    # sweep carries the calibration and every swept point for the k
    # selectors, the scenario explorer's band x k payload, the lazy size-cap
    # sweep (cap_sweep, None under --no-cap-sweep — every cell of it is
    # simulated here, in the page's one grid walk) and the
    # header's run-settings line; interval_discount is the resolved k these
    # trades were sized at, which the Risk section's Kelly scatter must price on
    #
    # series_categories files each trade under Kalshi's own series category and
    # tags for the Returns Decomposition breakdown: one cached read-only GET of
    # /series, which never raises (it falls back to a stale copy or to {})
    series_categories = load_series_categories(live_client)
    generate_dashboard(trades, equity_df, start_date, args.balance,
                       sweep=result, interval_discount=result.primary.k,
                       series_categories=series_categories)
    logging.info("Open the HTML file in a browser to view the interactive charts.")


if __name__ == "__main__":
    main()
