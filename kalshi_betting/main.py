"""
File: main.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Top-level orchestration for the Kalshi Arbitrage Bot's live trading
    pipeline. Parses command-line arguments to select dev (sandbox simulation)
    or prod (real-money trading) mode, then coordinates the full
    scan-size-execute-log cycle: building an authenticated API client,
    fetching open markets, finding candidate pairs under the bot's two pair
    strategies — same-title pairs (one question listed twice with divergent
    prices; a near-arbitrage on co-resolution) and time-series pairs (one
    question at two deadlines, a directional bet that the market overstates
    the chance the event first happens between them) — sizing trades via the
    Kelly criterion, submitting fill-or-kill orders leg-by-leg to the Kalshi
    REST API, and writing results to Excel. This is the only module that ties
    all other modules together in the live trading path. The process exit
    code communicates the run's outcome to the scheduler (a separate
    subprocess) — see the EXIT_* constants in config.py (BS-14): an unhandled
    exception still propagates to exit 1, same as always.

    Right after parsing its arguments, main() exits 2 if
    config.ORDER_API_VERSION is not "v2" (config.order_api_version_error),
    before logging is configured.

    A production run that sends orders (--mode prod without --dry-run) holds
    the machine-wide live-run lock (run_lock) from before it builds a client
    until main() ends, so two such runs never trade the account at once. When
    another run holds it, this one exits config.EXIT_RUN_IN_PROGRESS (50)
    without making any request. Dry runs and dev runs send no orders, so they
    neither take the lock nor wait for it.

    A production run started with --result-file PATH (the flag is refused in
    dev, right after the order-path check) first removes any file already at
    PATH, then writes what it did there as JSON in a finally once logging is
    set up, so however the run ends from that point: the exit code, the line
    it logged when it stopped or finished, its balances, whether it began
    sending orders, each pair's outcome, its WARNING-or-worse log lines and
    any error (reporter.RunReport, written by reporter.write_run_report).
    Two endings leave no file: a usage error that exits 2 before logging is
    set up (a bad flag, no or refused saved live defaults, a non-v2 order
    path), whose reason is on stderr, and a kill signal. A caller therefore
    reads the exit code first. The file is written before the lock is
    released, and neither filling nor writing it changes the exit code.

    The live toggles are the saved live defaults (config.LIVE_DEFAULTS_FILE,
    live_defaults.json, saved through python3 -m kalshi_betting.defaults_server),
    each overridable for one run by a flag of the "live trading toggles"
    group. _resolve_live_settings reads them once and builds the run's one
    config.LiveSettings before logging is configured: with no file saved, a
    refused file or a bad flag, the run exits 2 before anything is logged or
    requested, and it never falls back to config.py's toggle constants. Each
    run mode logs where the defaults came from and the run's settings
    (_log_live_settings) and hands them to every live site. The scheduler
    passes no toggle flag. Two pair-list filters run between the finders and
    enrichment: _dedup_pairs and _filter_by_category.

    A production run keeps every market the account holds out of new trades,
    except, when the run's add_to_held_pairs setting is on, the two markets
    of an exact held pair it may add to (scanner.held_pairs), which each
    finder lets through only as that same pair (add_on_pairs).

Dependencies:
    Imports from auth.py (client construction and auth verification), config.py
    (balance threshold, exit-code contract, the order-path check
    order_api_version_error, the same-title threshold and
    close-gap bound, file paths, and the live toggles: LiveSettings,
    live_defaults with its LiveDefaultsError / LiveDefaultsMissing refusals
    and LIVE_DEFAULTS_FROM_CONFIG, live_settings (the no-settings fallback),
    the describe_* helpers, live_rule_warnings, SIZE_CAP_STEP, and
    held_pair_fraction, the one definition of an add-on's size, which
    _run_prod reads to leave out a held pair with no room left, and
    count_text, which writes an add-on's held count exactly),
    historical.py (load_series_categories, series_labels, infer_category —
    the dashboard's filing rule, which _filter_by_category shares),
    reporter.py (Excel output, and the run result: RunReport,
    RunReportHandler, report_trades, write_run_report), _http.py
    (api_error_summary, the one-line description of the error that stopped
    a run, for its result), scanner.py (market fetching, pair detection,
    get_held_positions, resolve_held_ladders and held_pairs — the positions,
    their ladders and the exact held pairs a run may add to — leg_sides,
    the only source of truth for which side each leg buys, and pair_held,
    which names the held pair a trade adds to in the pairs table, the
    portfolio lines and the trade log's rescue dump, and close_gap_bound_text,
    which renders that close-gap bound in the same words the finders'
    refusal lines use),
    strategy.py (trade sizing and portfolio selection), trader.py (order
    execution), and run_lock.py (the lock that lets one real-money run trade
    at a time, which main() takes before building a client). Entry point for
    `python3 -m kalshi_betting.main`.

Notes:
    Label rule for everything this module logs: "A"/"B" always mean
    market_a/market_b, and the pairs table, _print_portfolio and the rescue
    dump render legs in MARKET order (A then B) with the bought side next to
    each count. The trader's own execution logs list legs in SUBMISSION order
    (the NO leg first) — the two views describe the same trade.

    Both run modes read the exchange's per-shard status breakdown
    (scanner.fetch_shard_statuses) before fetching markets and pass the set of
    trading-inactive shards into the fetch. Markets on every other shard are
    ingested and tagged with their exchange_index — market data is cross-shard.
    The full parsed status dict is kept in a local (`shard_statuses`) because
    prod reads more than trading_active from it: it is handed to
    trader.ensure_shard_collateral(), which refuses to move collateral to or
    from a shard whose intra_exchange_transfers_active is false. Immediately
    after the fetch, `_log_shard_coverage` (wrapping
    scanner.check_shard_coverage) compares that advertised breakdown against
    the shards actually observed in ingested markets (and, in prod, the
    balance breakdown) and logs any gap at CRITICAL or WARNING — see the
    function docstring for the empty-vs-funded severity split. A run only
    claims "Full shard coverage" when the breakdown exists and nothing was
    wrong; the run always continues regardless of what the check finds.
"""
import argparse
import contextlib
import logging
import logging.handlers
import os
import pathlib
import sys
from collections import Counter
from dataclasses import replace as dc_replace
from datetime import UTC, datetime

from tabulate import tabulate

from . import run_lock
from ._http import api_error_summary
from .auth import build_client, verify_auth
from .config import (
    EXIT_NO_TRADEABLE_SHARDS,
    EXIT_OK,
    EXIT_RUN_IN_PROGRESS,
    EXIT_SKIPPED_LOW_BALANCE,
    EXIT_TIME_SERIES_SKIPPED,
    EXIT_TRADES_NEED_ATTENTION,
    LIVE_DEFAULTS_FROM_CONFIG,
    MIN_BALANCE_CENTS,
    PROJECT_ROOT,
    SAME_TITLE_MAX_CLOSE_GAP_SECONDS,
    SAME_TITLE_MIN_PRICE_DIFF,
    SIZE_CAP_STEP,
    LiveDefaultsError,
    LiveDefaultsMissing,
    LiveSettings,
    count_text,
    describe_live_settings,
    describe_time_series_rule,
    describe_trade_filter,
    held_pair_fraction,
    live_defaults,
    live_rule_warnings,
    live_settings,
    order_api_version_error,
)
from .historical import infer_category, load_series_categories, series_labels
from .reporter import (
    RunReport,
    RunReportHandler,
    append_to_prod_log,
    report_trades,
    write_dev_simulation,
    write_run_report,
)
from .scanner import (
    check_shard_coverage,
    close_gap_bound_text,
    display_title,
    enrich_with_orderbook_prices,
    fetch_open_events_with_markets,
    fetch_shard_statuses,
    filter_markets_within_horizon,
    find_same_title_pairs,
    find_time_series_pairs,
    get_held_positions,
    held_pairs,
    inactive_shard_indexes,
    leg_sides,
    pair_held,
    resolve_held_ladders,
)
from .strategy import compute_trade, select_portfolio
from .trader import (
    ensure_shard_collateral,
    execute_trades,
    pre_execution_check,
)


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


def _compute_trade_specs(
    candidate_pairs: list, balance_cents: int, settings: LiveSettings,
) -> dict:
    """
    Compute trade specifications for all qualifying candidate pairs.

    Args:
        candidate_pairs (list): List of CandidatePair objects to evaluate.
        balance_cents (int): Current account balance in cents, used to size
            each trade via Kelly criterion in compute_trade().
        settings (LiveSettings): The run's toggles — the object enrichment
            bounded the depth with (strategy.compute_trade).

    Returns:
        dict: Mapping of id(pair) -> TradeSpec for each pair that produced
            a valid trade specification. Pairs that do not meet profitability
            or size thresholds are excluded.
    """
    specs: dict = {}
    for pair in candidate_pairs:
        # Kelly-size under the run's k and caps; None when the pair does not qualify
        spec = compute_trade(pair, balance_cents, settings=settings)
        if spec is not None:
            specs[id(pair)] = spec
    return specs


def _print_portfolio(portfolio: list, label: str) -> None:
    """
    Log a summary of selected portfolio trades to the log file.

    Legs are rendered in MARKET order (A then B) with the side bought on each
    market next to its count — e.g. "5× YES(A) + 5× NO(B)" for a time-series
    pair, "5× NO(A) + 5× YES(B)" for a same-title pair. The sides come from
    scanner.leg_sides, never from the pair type by hand. Each line also names
    both tickers: a time-series pair's canonical title now ends with the
    outcome label (scanner.time_series_group_key) and the 55-character cut
    applied below drops it, so two strikes of one daily family would otherwise
    log an identical line (DR-17). "profit if won" is
    spec.min_payoff: the guaranteed floor for a same-title pair, and the
    profit in either winning settlement of a time-series pair (event by A, or
    never by B) — the in-between settlement loses the whole stake. A trade
    that adds to a pair the account already holds (scanner.pair_held) ends
    " — adds to N held", N being the contracts held on each market; every
    other line is unchanged.

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
        # Which side each market's leg buys — the only source of truth for sides
        side_a, side_b = leg_sides(spec.pair.pair_type)
        # Cross-module: the held pair this trade adds to, read by type
        held = pair_held(spec.pair)
        logging.info(
            "  [%s] %s (%s / %s) — %d× %s(A) + %d× %s(B) — "
            "cost $%.2f incl. fees, profit if won $%.2f (%.1f%% return)%s",
            spec.pair.pair_type,
            spec.pair.canonical_title[:55],
            # DR-17: the tickers identify the trade when the title cannot —
            # the 55-character cut above drops the outcome label a time-series
            # canonical title ends with, so two strikes of one daily family
            # produced the same line in the review of a dry run
            spec.pair.market_a.ticker, spec.pair.market_b.ticker,
            spec.x, side_a.upper(), spec.y, side_b.upper(),
            spec.total_cost_with_fees, spec.min_payoff,
            spec.profit_ratio * 100,
            f" — adds to {count_text(held.count)} held" if held is not None else "",
        )


def _no_pairs_msg(sandbox: bool = False, settings: LiveSettings | None = None, *,
                  time_series_searched: bool = True) -> str:
    """
    Build the "no qualifying pairs found" log message with the run's live rule.

    Words the time-series rule as the finder's "Time-series entry rule" line
    does, and names the cumulative-deadline requirement so a reader does not
    blame the price rule for what the wording decided. Names the
    same-title pairing rules for the same reason: a same-title pair also needs
    two different event series whose markets close within
    SAME_TITLE_MAX_CLOSE_GAP_SECONDS of each other (DR-02/DR-54, DR-74), with
    the bound read from that constant and rendered by
    scanner.close_gap_bound_text. A set category/tag filter is named too,
    since it can empty the list on its own.

    Args:
        sandbox (bool): True to phrase the message for a dev/sandbox run
            ("... found in sandbox ..."), False for a production run.
            Defaults to False.
        settings (LiveSettings | None): The run's toggles; None reads config.py's.
        time_series_searched (bool): Keyword-only. False says the run did not
            look for time-series pairs, instead of naming their rule.

    Returns:
        str: The fully formatted log message, ready to pass to logging.info().

    Raises:
        ValueError: When settings is None and a config.py toggle is invalid.
    """
    # The run's toggles, or config.py's for a caller that hands none
    settings = live_settings() if settings is None else settings
    if time_series_searched:
        time_series = (
            "time-series: both legs worded as cumulative deadlines "
            "(“by <date>”, two different ones) with the later leg's YES ask above "
            "the earlier's by the run's entry rule — "
            # In the finder's own rule-line words, so the two cannot disagree
            f"{describe_time_series_rule(settings.tier_floors, settings.spread_band)}"
        )
    else:
        time_series = ("time-series: not searched this run, because a held market "
                       "could not be identified (see the ERROR above)")
    thresholds = (
        f"{time_series}"
        " — or same-title: "
        f"≥{SAME_TITLE_MIN_PRICE_DIFF:.0%} price diff on two different series "
        # This module's own binding, like the same-title threshold above; the scanner
        # helper only renders it, in the words the refusal lines use.
        f"closing within {close_gap_bound_text(SAME_TITLE_MAX_CLOSE_GAP_SECONDS)}"
    )
    if settings.categories is not None or settings.tags is not None:
        # The run's category/tag filter, in the "Live settings:" line's words
        thresholds += (f" — among pairs filed under the run's category/tag filter "
                       f"({describe_trade_filter(settings)})")
    if sandbox:
        return f"No qualifying pairs found in sandbox ({thresholds})."
    return f"No qualifying pairs found ({thresholds})."


def _dedup_pairs(primary: list, secondary: list) -> list:
    """
    Merge two pair lists, excluding any pair from secondary that already appears in primary.

    A duplicate is defined as any pair whose frozenset of {ticker_a, ticker_b} already
    exists in primary. Since DR-67 the two finders cannot produce a cross-type
    collision on one ticker pair: a same-title pair requires identical wording,
    a time-series pair requires two DIFFERENT stated deadlines, and identical
    wording states identical deadline spans, or none — never two different
    ones. This function stays as a guard against a collision arising another
    way (both finders have their own defences too — see the one-series and
    cumulative-deadline gotchas in CLAUDE.md), and its preference is the same
    as before: were a collision to occur, the same-title pair would be kept
    because it is the near-arbitrage — identical questions must co-resolve, so
    its payoff is a floor — whereas the time-series pair is a directional bet
    whose sizing rests on the discounted-gap estimate of the in-between
    probability (config.time_series_profit_prob). This matches the
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


def _filter_by_category(pairs: list, settings: LiveSettings, listing_client) -> list:
    """
    Keep only the pairs filed under the run's categories and tags.

    Files each pair by MARKET A's series through historical.series_labels,
    the dashboard's filing rule. Matching is case-insensitive, the two axes
    combine by AND (None = any), and a tag matches under EVERY category, unlike
    the dashboard's category-scoped Tag options. Fails CLOSED: with no
    listing nothing is kept, rather than filing nearly every KX ticker as "Other",
    and its WARNING is the one line saying why, since the run exits EXIT_OK.
    A name matching no listed label and no pair draws a typo WARNING.

    Args:
        pairs (list): CandidatePairs after dedup, before enrichment.
        settings (LiveSettings): The run's toggles.
        listing_client: A production KalshiClient, or None to read the cached
            /series listing only (dev: the sandbox key never signs a prod request).

    Returns:
        list: The kept pairs, in their original order — the input list itself,
            with no request, when no filter is set; [] when there is no listing.
    """
    if settings.categories is None and settings.tags is None:
        return pairs
    # Kalshi's category and tags per series, the map the dashboard files by
    series_categories = load_series_categories(listing_client)
    # The filter in the "Live settings:" line's own words
    wanted = describe_trade_filter(settings)
    if not series_categories:
        where = ("no cached copy exists (a dev run reads the cached copy only — a "
                 "production run or a backtest fetches it)" if listing_client is None
                 else "Kalshi's /series listing could not be read and no cached copy exists")
        logging.warning(
            "Category/tag filter (%s) is set but %s — no pair can be filed, so this "
            "run trades none", wanted, where)
        return []
    cats = None if settings.categories is None else {c.casefold() for c in settings.categories}
    tags = None if settings.tags is None else {t.casefold() for t in settings.tags}
    kept: list = []
    dropped: Counter = Counter()
    filed_cats: set = set()
    filed_tags: set = set()
    for pair in pairs:
        event = pair.market_a.event_ticker
        # Filed exactly as the dashboard files a trade of this event
        category, tag = series_labels(event, infer_category(event), series_categories)
        filed_cats.add(category.casefold())
        filed_tags.add(tag.casefold())
        if ((cats is None or category.casefold() in cats)
                and (tags is None or tag.casefold() in tags)):
            kept.append(pair)
        else:
            dropped[f"{category} · {tag}"] += 1
    logging.info(
        "Category/tag filter (%s): kept %d of %d candidate pairs%s", wanted,
        len(kept), len(pairs),
        "" if not dropped else " — dropped " + ", ".join(
            f"{name} {n}" for name, n in dropped.most_common()))
    # A name matching no listed label and no pair is almost certainly a typo
    known_cats = filed_cats | {(c or "Uncategorised").casefold()
                               for c, _ in series_categories.values()}
    known_tags = filed_tags | {(t[0] if t else "General").casefold()
                               for _, t in series_categories.values()}
    for name in settings.categories or ():
        if name.casefold() not in known_cats:
            logging.warning(
                "Category %r names no category in Kalshi's listing of %d series, nor "
                "any this run's pairs were filed under — check the spelling; it "
                "matches nothing", name, len(series_categories))
    for name in settings.tags or ():
        if name.casefold() not in known_tags:
            hint = ""
            if " · " in name:
                # The dashboard's Tag select names "category · tag"
                category, _, tag = name.partition(" · ")
                hint = (f" (the dashboard's Tag option {name!r} is --category "
                        f"{category!r} --tag {tag!r})")
            logging.warning(
                "Tag %r is no series' first tag in Kalshi's listing of %d series, nor "
                "any this run's pairs were filed under — check the spelling; it "
                "matches nothing%s", name, len(series_categories), hint)
    return kept


def print_pairs_table(candidate_pairs: list, display_specs: dict) -> None:
    """
    Log a formatted table of all qualifying candidate pairs to the log file.

    Displays market titles, each leg's outcome label (the subtitle — the
    discriminator that separates two strikes of one daily family; it gets its
    own column because display_title appends it at the END and _truncate cuts
    the title cells at 40 characters, so on any title that runs past 40 it is
    the first thing dropped), each leg's exchange shard, deadlines, prices —
    both YES asks
    plus the NO ask of market B, which is the traded price of a time-series
    pair's NO leg and reporting-only for a same-title pair —
    tradeability, and, for pairs selected in the portfolio, the computed trade
    (counts in MARKET order with the side bought on each market, from
    scanner.leg_sides, followed by "(adds to N held)" for a trade that adds
    to a pair the account already holds), the profit if won (spec.min_payoff:
    a guaranteed floor for same-title, the profit in either winning
    settlement for time-series), monthly return, and Kelly fraction.

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
        # Price columns come from the spec's own pair when there is one: that is
        # the marginal fill price for the recommended size, so the prices and
        # the trade on a row always describe the same thing. Candidates with no
        # spec keep the enrichment-stage quote.
        priced = spec.pair if spec else pair
        if spec:
            # Sides rendered next to each count, in market order (A then B)
            side_a, side_b = leg_sides(pair.pair_type)
            trade_str   = f"{spec.x}× {side_a.upper()}(A) + {spec.y}× {side_b.upper()}(B)"
            # Cross-module: a trade adding to a held pair says how much is held
            held = pair_held(spec.pair)
            if held is not None:
                trade_str += f" (adds to {count_text(held.count)} held)"
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
            # DR-17: the outcome label in its own cells — display_title
            # appends it at the END, so _truncate's 40-character cut on the
            # title cells above drops it first on any title that runs past 40
            # characters, which the daily families this exists for all do
            _truncate(getattr(pair.market_a, "subtitle", "") or "—", 24),
            _truncate(getattr(pair.market_b, "subtitle", "") or "—", 24),
            # Which exchange shard each leg's market is on ("a/b"); each
            # order goes to its own leg's shard
            f"{pair.market_a.exchange_index}/{pair.market_b.exchange_index}",
            _format_deadline(pair.market_a.close_time),
            _format_deadline(pair.market_b.close_time),
            f"{priced.pA:.2%}",
            f"{priced.pB:.2%}",
            # The NO ask of market B — the traded NO-leg price of a time-series
            # pair (depth-weighted after enrichment); reporting-only for same-title
            f"{priced.nB:.2%}",
            "YES ✓" if pair.tradeable else "no",
            trade_str,
            profit_str,
            monthly_str,
            kelly_str,
        ])

    headers = [
        "Type",
        "Market A", "Market B", "Outcome A", "Outcome B",
        "Shards",
        "A Deadline", "B Deadline",
        "pA (YES)", "pB (YES)", "nB (NO)",
        "Tradeable?", "Recommended Trade", "Profit (win)", "Monthly Return", "Kelly",
    ]
    table = tabulate(rows, headers=headers, tablefmt="rounded_outline")
    for line in table.splitlines():
        logging.info(line)


def _log_shard_coverage(shard_statuses, market_shards: set, balance_shards: set) -> None:
    """
    Run scanner.check_shard_coverage() and emit its findings at the right
    severity, shared by both _run_dev and _run_prod so the logging split
    (critical vs warning vs "full coverage" vs "unassessable") lives in one
    place.

    A run must only ever CLAIM full coverage when every shard the exchange
    advertises was actually scanned; a missing shard is reported loudly but
    never aborts the run — trading continues on whatever was covered. When NO
    advertised shard was scannable (every one explicitly trading-inactive —
    an exchange-wide halt), check_shard_coverage's two loops are empty and it
    returns ([], []), which used to print the same "Full shard coverage" line
    a healthy run prints (TS-01); that case now warns instead. A shard whose
    trading_active is None (flag absent or unreadable, TS-04) counts as
    scannable, matching scanner.inactive_shard_indexes; a re-typed "false"
    does NOT, because scanner.fetch_shard_statuses normalises it to a real
    False first (TS-04b).

    Args:
        shard_statuses (dict | None): Return value of scanner.fetch_shard_statuses().
        market_shards (set): exchange_index values seen among ingested markets.
        balance_shards (set): exchange_index values holding a nonzero balance.

    Returns:
        None
    """
    if shard_statuses is None:
        logging.info("Per-shard exchange status unavailable — coverage not assessable.")
        return
    # Pure comparison of advertised vs. observed shards; this function only
    # decides how loudly to log what check_shard_coverage found.
    critical, warnings = check_shard_coverage(shard_statuses, market_shards, balance_shards)
    # Coverage may only be CLAIMED over shards that were scannable.
    # check_shard_coverage deliberately skips trading-inactive shards (their
    # ingest drop already warned), so an all-inactive exchange yields ([], [])
    # — which read as success (TS-01). Only a False is inactive (a real bool
    # by now: scanner.fetch_shard_statuses recovers a re-typed "false" into
    # one); None (flag unknown) is scannable, matching inactive_shard_indexes.
    scannable = sorted(
        idx for idx, st in shard_statuses.items() if st.get("trading_active") is not False
    )
    for problem in critical:
        # Same severity channel as orphaned positions — this must never be missable.
        logging.critical("SHARD COVERAGE FAILURE: %s", problem)
    for problem in warnings:
        logging.warning("Shard coverage: %s", problem)
    if not scannable:
        logging.warning(
            "Shard coverage NOT claimable: all %d advertised shards are "
            "trading-inactive — markets were ingested from %d shard(s)",
            len(shard_statuses), len(market_shards),
        )
    elif not critical and not warnings:
        # Lists the shards actually scanned, not every advertised one — a
        # funded-but-halted shard is no longer folded into "full".
        logging.info("Full shard coverage: shards %s scanned", scannable)


def _blind_run_reason(markets: list, shard_statuses, inactive_shards: set) -> str | None:
    """
    Decide whether this run scanned NOTHING at all, and name the reason.

    The single definition of "blind run" for BOTH run modes, so dev and prod
    can never silently disagree about what one is. A blind run is any run whose
    market ingest yielded nothing to pair, and it must be reported with
    EXIT_NO_TRADEABLE_SHARDS rather than EXIT_OK: EXIT_OK claims "scanned
    everything, found no edge", which would let scheduler.run_job record the
    weekly slot as satisfied by a run that never looked at a single book
    (TS-01).

    Two independent causes, checked in that order because the first is the
    more specific diagnosis:

    1. Every advertised exchange shard is trading-inactive (the exchange-wide
       halt observed live 2026-09-03) — ingest dropped every market by design.
       Kept as its OWN disjunct rather than folded into the census below,
       because the two are not equivalent: a halt that still leaves one stray
       market ingested (e.g. one tagged with a shard /exchange/status does not
       advertise) would pass a census test while every advertised book is shut.
    2. Ingest produced zero markets for any other reason — most importantly
       when scanner.fetch_shard_statuses() returned None, which it does on ANY
       internal failure (it is fail-soft by design, and correctly so). That
       makes `shard_statuses` falsy, so cause 1 cannot fire, and an ingest that
       came back empty used to exit EXIT_OK claiming a clean scan (VI-02).

    This must be evaluated on the RAW ingest, before held-ticker or horizon
    filtering: those filters legitimately empty the list on a healthy exchange
    (`--max-horizon-days 1` on a quiet week), which is "no edge", not blind.

    Args:
        markets (list): The ApiMarket objects ingest produced, BEFORE any
            held-ticker or horizon filtering.
        shard_statuses (dict | None): Return value of
            scanner.fetch_shard_statuses(). None (breakdown unavailable) makes
            cause 1 unknowable, which is why cause 2 exists.
        inactive_shards (set): Return value of
            scanner.inactive_shard_indexes(shard_statuses) — only ever a subset
            of the advertised shards, so equality with the advertised set means
            "every one of them".

    Returns:
        str | None: A ready-to-log sentence naming which cause fired, or None
            when the run actually scanned something and is not blind.

    Raises:
        Nothing. This is a pure comparison over already-fetched values; it
        performs no I/O and swallows nothing.
    """
    if shard_statuses and inactive_shards == set(shard_statuses):
        return (
            f"Every advertised exchange shard is trading-inactive "
            f"({sorted(inactive_shards)}) — nothing can be scanned this run"
        )
    if not markets:
        return "Ingest produced zero markets — nothing can be scanned this run"
    return None


# argparse destination -> the flag an operator types, for the usage error
_LIVE_FLAGS = (
    ("tier_floors", "--tier-floors/--no-tier-floors"),
    ("spread_min", "--spread-min"),
    ("spread_max", "--spread-max"),
    ("interval_discount", "--interval-discount"),
    ("size_cap", "--size-cap"),
    ("same_title_size_cap", "--same-title-size-cap"),
    ("add_to_held_pairs", "--add-to-held-pairs/--no-add-to-held-pairs"),
    ("category", "--category"),
    ("any_category", "--any-category"),
    ("tag", "--tag"),
    ("any_tag", "--any-tag"),
)

# The cap flags, which take a whole percent where LiveSettings holds a fraction
_LIVE_PERCENT_FLAGS = frozenset({"size_cap", "same_title_size_cap"})


def _resolve_live_settings(args, parser) -> tuple[LiveSettings, LiveSettings]:
    """
    Resolve this run's LiveSettings: the saved live defaults, with each given flag laid over.

    The run's one read of the defaults (config.live_defaults, the one caller
    the AST pin test_ast_live_path_reads_toggles_only_through_live_settings
    allows); exits 2 when none are saved or the file is refused. There is no
    fallback to config.py's toggle constants. LiveSettings validates the
    result (dataclasses.replace re-runs __post_init__), so a flag meets the
    same rule a saved value does; the result keeps the defaults' origin.
    Called before logging is configured, so a refusal logs nothing and makes
    no request.

    Args:
        args (argparse.Namespace): The parsed flags; a missing attribute reads
            as not given.
        parser (argparse.ArgumentParser): Used to report a refusal.

    Returns:
        tuple[LiveSettings, LiveSettings]: (this run's settings, the saved
            defaults they were built from), equal when no toggle flag was given.

    Raises:
        SystemExit: Status 2 (parser.error) when no live defaults are saved,
            the saved file is refused, or a flag's value is invalid.
    """
    try:
        # The saved live defaults: the reference, and the base the flags lay over.
        # There is no fallback: with none saved, the run does not start
        reference = live_defaults()
    except LiveDefaultsMissing as exc:
        parser.error(str(exc))
    except LiveDefaultsError as exc:
        # The defaults server will not save over a file it refuses, so the
        # remedy is to fix the file, or to delete it before saving new ones
        parser.error(f"the saved live defaults are refused ({exc}): fix the file, or delete "
                     f"it and then save new ones through the defaults server, "
                     f"./start_dashboard.sh (or python3 -m kalshi_betting.defaults_server): "
                     f"its --seed, or the backtest dashboard's \"Save as live defaults…\" "
                     f"button — the server will not save over a file it refuses")
    overrides: dict = {}
    if getattr(args, "tier_floors", None) is not None:
        overrides["tier_floors"] = args.tier_floors
    lo, hi = getattr(args, "spread_min", None), getattr(args, "spread_max", None)
    if lo is not None or hi is not None:
        overrides["spread_band"] = (reference.spread_band[0] if lo is None else lo,
                                    reference.spread_band[1] if hi is None else hi)
    if getattr(args, "interval_discount", None) is not None:
        overrides["interval_discount"] = args.interval_discount
    if getattr(args, "size_cap", None) is not None:
        overrides["size_cap"] = args.size_cap / 100
    if getattr(args, "same_title_size_cap", None) is not None:
        overrides["same_title_size_cap"] = args.same_title_size_cap / 100
    if getattr(args, "add_to_held_pairs", None) is not None:
        overrides["add_to_held_pairs"] = args.add_to_held_pairs
    # --category / --tag (repeatable) set the filter; --any-category / --any-tag
    # clear the saved one (argparse keeps each pair mutually exclusive)
    if getattr(args, "category", None) is not None:
        overrides["categories"] = tuple(args.category)
    elif getattr(args, "any_category", None):
        overrides["categories"] = None
    if getattr(args, "tag", None) is not None:
        overrides["tags"] = tuple(args.tag)
    elif getattr(args, "any_tag", None):
        overrides["tags"] = None
    try:
        # replace re-validates every field (LiveSettings.__post_init__)
        return dc_replace(reference, **overrides), reference
    except ValueError as exc:
        # Given means not None, never truthy: 0 is a given --spread-min and
        # False a given --no-tier-floors (the --any-* switches default to None)
        given = [(dest, flag) for dest, flag in _LIVE_FLAGS
                 if getattr(args, dest, None) is not None]
        message = (f"invalid live setting for this run "
                   f"({', '.join(flag for _, flag in given)}): {exc}")
        percent = [flag for dest, flag in given if dest in _LIVE_PERCENT_FLAGS]
        if percent:
            # The flag's own unit, from the one grid definition (SIZE_CAP_STEP)
            step = f"{SIZE_CAP_STEP * 100:g}"
            message += (f" ({' and '.join(percent)} take{'s' if len(percent) == 1 else ''} "
                        f"a whole percent, a multiple of {step} from {step} to 100, read "
                        "as that percent / 100)")
        parser.error(message)


def _log_live_settings(settings: LiveSettings, reference: LiveSettings, *,
                       real_money: bool) -> None:
    """
    Log the run's defaults' origin, its toggles, any departure, and every rule warning.

    One INFO line names the defaults' origin (the saved file, when and from
    what it was saved). One INFO line marks each field that departs from
    reference: "(default: X)" when reference is the saved live defaults, as a
    live run's always is, "(config: X)" when it was built from config.py's
    constants (a reference a test or direct call builds, e.g. with
    config.live_settings()). Only a real-money run WARNs on a departure.
    Resolves neither object itself (TestLiveSettingsReachEverySite).

    Args:
        settings (LiveSettings): The run's toggles.
        reference (LiveSettings): The defaults the run's toggles were built
            from (the run's own when a run mode was handed none, which marks
            nothing).
        real_money (bool): True for a prod run that submits orders.
    """
    # Where the defaults came from: the saved file, when and from what
    logging.info("Live defaults: %s", reference.origin)
    # Every field, with a mark on each one a flag moved: "(default: X)" (the
    # saved defaults), or "(config: X)" for a reference built from config.py
    logging.info("Live settings: %s", describe_live_settings(settings, reference))
    if real_money and settings != reference:
        if reference.origin == LIVE_DEFAULTS_FROM_CONFIG:
            logging.warning(
                "This PRODUCTION run overrides config.py's live settings (see the "
                "\"(config: …)\" marks on the line above): its trades follow the "
                "flags, not the committed configuration")
        else:
            logging.warning(
                "This PRODUCTION run overrides the saved live defaults (see the "
                "\"(default: …)\" marks on the line above): its trades follow the "
                "flags, not the saved defaults")
    # A setting that empties part of the strategy or lifts one pair's stake
    for text in live_rule_warnings(settings):
        logging.warning("Live settings: %s", text)


def _run_dev(client, args, settings: LiveSettings | None = None,
             reference: LiveSettings | None = None) -> int:
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
        settings (LiveSettings | None): The run's toggles; None resolves
            config.py's (tests and direct calls only: main() always hands the
            run's, built from the saved live defaults).
        reference (LiveSettings | None): The saved live defaults the run's
            toggles were built from; None means the run's own.

    Returns:
        int: EXIT_NO_TRADEABLE_SHARDS when the run was blind — every
            advertised exchange shard trading-inactive so ingest dropped every
            market (TS-01), or an ingest that produced zero markets for any
            other reason, including the one fetch_shard_statuses() cannot
            diagnose because it failed fail-soft to None (VI-02); see
            _blind_run_reason. Dev's code is not consumed by the scheduler,
            but the two modes must not disagree about what a blind run is, so
            a blind dev run short-circuits before write_dev_simulation exactly
            as it does today — an empty simulation file is written for a run
            that SCANNED and found no pairs, never for one that never looked.
            EXIT_OK otherwise: dev mode never submits real orders, so there is
            no low-balance skip or manual-review outcome to distinguish.
            Returned as an int (rather than None) for symmetry with _run_prod,
            since main() dispatches to either and passes the result to
            sys.exit().

    Raises:
        ValueError: When settings is None and a config.py toggle is invalid.
    """
    # Resolved ONCE and handed to every site below that reads a toggle
    settings = live_settings() if settings is None else settings
    reference = settings if reference is None else reference
    sandbox_balance_cents = int(args.sandbox_balance * 100)
    logging.info(
        "DEV mode: using real sandbox market data | virtual balance $%.2f",
        args.sandbox_balance,
    )
    # Log the run's toggles; dev never submits, so no departure WARNING
    _log_live_settings(settings, reference, real_money=False)

    # Read the exchange's per-shard status breakdown so ingest can drop shards
    # that aren't trading. Returns None on the sandbox / pre-sharding shape,
    # which degrades to single-shard semantics (keep everything).
    shard_statuses = fetch_shard_statuses(client)
    inactive_shards = inactive_shard_indexes(shard_statuses)

    # Fetch all open sandbox markets — the sandbox public endpoint does not require
    # valid authentication for read operations, so this works with the prod key too.
    # Markets from every shard are ingested and tagged; only trading-inactive
    # shards are dropped.
    markets = fetch_open_events_with_markets(client, inactive_shards=inactive_shards)
    logging.info("Sandbox markets fetched: %d", len(markets))

    # Dev mode has one virtual balance, not a real per-shard breakdown, so
    # there is no balance-shard set to compare against — pass empty and let
    # the market-coverage half of the check still catch a missing shard
    _log_shard_coverage(shard_statuses, {m.exchange_index for m in markets}, set())

    # A run that ingested nothing simulated nothing. Dev's exit code is not
    # consumed by the scheduler, but the two modes must not disagree about what
    # a blind run is, so both ask the same helper (TS-01, VI-02).
    blind_reason = _blind_run_reason(markets, shard_statuses, inactive_shards)
    if blind_reason:
        logging.warning("%s", blind_reason)
        return EXIT_NO_TRADEABLE_SHARDS

    # Optional opt-in cap so both bet types only see markets closing within
    # the requested window — a no-op (returns markets unchanged) when unset
    markets = filter_markets_within_horizon(markets, args.max_horizon_days)

    # Skip held-positions filter — sandbox requires a separate account and credentials.
    # Pass an empty set so _filter_active_markets does not exclude any tickers.
    # The run's settings decide the time-series entry rule (tier floors, band)
    time_series_pairs = find_time_series_pairs(
        client, held_tickers=set(), markets=markets, settings=settings,
    )
    # Detect same-title pairs separately — uses a different grouping key (exact title match)
    same_title_pairs  = find_same_title_pairs(markets, held_tickers=set())
    # Merge both lists, preferring same_title when both scanners found the same pair
    candidate_pairs   = _dedup_pairs(same_title_pairs, time_series_pairs)
    # Category/tag filter; no listing client, so the sandbox key never signs a prod request
    candidate_pairs   = _filter_by_category(candidate_pairs, settings, None)
    # Replace best-ask prices with order book averages over the depth this
    # balance could actually buy, and validate liquidity under the run's rule
    candidate_pairs   = enrich_with_orderbook_prices(
        client, candidate_pairs, sandbox_balance_cents, settings=settings,
    )

    if not candidate_pairs:
        # BS-26: write_dev_simulation() already logs "Dev simulation written: %s" —
        # this line carries the qualifier (why the file is empty) instead of
        # repeating the artifact path a second time. The thresholds come from
        # _no_pairs_msg, handed the run's settings, so they match the rule the run applied.
        logging.info(
            "%s Simulation file will contain headers only.",
            _no_pairs_msg(sandbox=True, settings=settings),
        )
        # Write an empty simulation file so the run is still recorded
        write_dev_simulation([], [], sandbox_balance_cents)
        return EXIT_OK

    # Apply Kelly sizing to each candidate pair using the virtual balance
    trade_specs   = _compute_trade_specs(candidate_pairs, sandbox_balance_cents, settings)
    # Greedy portfolio selection ranked by monthly_profit_ratio descending
    portfolio     = select_portfolio(list(trade_specs.values()), sandbox_balance_cents)
    # Map pair id → TradeSpec for fast lookup in the pairs table display.
    # Keyed off the CANDIDATE each spec was built from, not off spec.pair:
    # compute_trade returns a re-priced copy of the pair (the marginal fill
    # price for the size it settled on), so id(spec.pair) no longer matches any
    # entry in candidate_pairs and every selected row would render as "—".
    chosen = {id(s) for s in portfolio}
    display_specs = {pid: s for pid, s in trade_specs.items() if id(s) in chosen}

    logging.info("Kalshi Sandbox Scan — Virtual Balance: $%.2f | Mode: DEV", args.sandbox_balance)
    print_pairs_table(candidate_pairs, display_specs)

    if not portfolio:
        # BS-26: qualifier only — write_dev_simulation() logs the "written" line itself.
        logging.info(
            "No executable trades found — simulation file will "
            "contain candidates only."
        )
        # Write a simulation file showing candidates even though no trades were sized
        write_dev_simulation([], candidate_pairs, sandbox_balance_cents)
        return EXIT_OK

    _print_portfolio(portfolio, "Simulated")

    # Simulate order execution (dry_run=True is always enforced in dev mode)
    # Returns TradeResult objects with status="simulated" — no API orders are placed
    results = execute_trades(client, portfolio, dry_run=True)

    # Write the simulation Excel file: Sheet 1 = simulated trades, Sheet 2 = all
    # candidates. write_dev_simulation() logs "Dev simulation written: %s" itself
    # (BS-26) — don't duplicate that line here.
    write_dev_simulation(results, candidate_pairs, sandbox_balance_cents)
    return EXIT_OK


def _run_prod(client, args, settings: LiveSettings | None = None,
              reference: LiveSettings | None = None, *,
              report: RunReport | None = None) -> int:
    """
    Execute a full production run using the real Kalshi account.

    Verifies authentication, reads the live account balance, fetches currently
    held positions to keep them out of new trades, discovers candidate pairs
    under both strategies (same-title near-arbitrage pairs and directional
    time-series pairs), sizes trades using the Kelly criterion, and submits
    fill-or-kill orders leg-by-leg (the NO leg first, then the YES leg — see
    trader.execute_trades). Results are appended to the persistent
    trade_log.xlsx file. In dry_run mode (--dry-run flag), all steps run
    normally except order submission — the log still records rows with
    status="simulated". Along the way it fills in the run report it is
    handed (see report below); every return value is the same with or
    without one.

    It makes no new time-series trade on a ladder the account holds (a ladder
    is one question asked at several deadlines), and none at all if a held
    market cannot be identified; same-title trades still go ahead. With
    settings.add_to_held_pairs on, it may add to an exact held pair
    (scanner.held_pairs: the same two markets, the same side on each), sized
    on the whole position (config.held_pair_fraction); only when the
    positions listing was read to its end and every held market was
    identified, and never to a pair already holding its per-trade cap of the
    account value.

    Args:
        client: KalshiClient pointed at the production endpoint, produced by
            auth.build_client("prod").
        args: Parsed argparse Namespace with dry_run and max_horizon_days
            attributes.
        settings (LiveSettings | None): The run's toggles; None resolves
            config.py's (tests and direct calls only: main() always hands the
            run's, built from the saved live defaults).
        reference (LiveSettings | None): The saved live defaults the run's
            toggles were built from, which departures are marked against;
            None means the run's own.
        report (RunReport | None): Keyword-only. The run result main() writes
            for --result-file, filled in as the run goes: the balance before
            and after trading, that it began sending orders (set just before
            execute_trades on a run that is not a dry run), one record per
            pair (reporter.report_trades, which never raises, so the trade log
            is still written), and the message — the line the run logged when
            it stopped without trading, or its closing summary line, word for
            word. None (no --result-file) fills a report nobody reads, so the
            run is the same either way.

    Returns:
        int: EXIT_SKIPPED_LOW_BALANCE if the run was skipped because the
            account balance is below MIN_BALANCE_CENTS (no scan attempted).
            EXIT_NO_TRADEABLE_SHARDS if the run was blind — every advertised
            exchange shard trading-inactive so ingest dropped every market
            (TS-01), or an ingest that produced zero markets for any other
            reason, including the one fetch_shard_statuses() cannot diagnose
            because it failed fail-soft to None (VI-02); see
            _blind_run_reason. Deliberately distinct from EXIT_OK's "scanned
            everything, found no edge", because the scheduler must not count
            the weekly slot as satisfied by a run that never looked at a book.
            EXIT_TRADES_NEED_ATTENTION if any TradeResult in this run's
            results has status "rollback_failed" or "manual_review" — either
            means a human must check the account/trade log, and wins over the
            next code. EXIT_TIME_SERIES_SKIPPED if a held market could not be
            identified, so the run made no time-series trade. EXIT_OK for
            every other path, including dry-run, no candidate pairs, no
            executable trades, and all-pairs-failed-pre-execution-check.

    Raises:
        ValueError: When settings is None and a config.py toggle is invalid.
    """
    # What the run did, for main.py --result-file; without it nothing reads this one
    report = RunReport(dry_run=args.dry_run, started_at=datetime.now(UTC)) if report is None else report
    # Resolved ONCE, before any request, and handed to every site below that reads a toggle
    settings = live_settings() if settings is None else settings
    reference = settings if reference is None else reference
    logging.warning("Running in PRODUCTION mode — real money will be used!")
    # Log the run's toggles; a departure is a WARNING when orders go out
    _log_live_settings(settings, reference, real_money=not args.dry_run)

    # Confirm auth works and read the pre-trade balance broken out by shard
    shard_balances = verify_auth(client)
    # Sizing is portfolio-wide, not per-shard: collateral is made fungible
    # across shards by the pre-execution transfers in ensure_shard_collateral,
    # so Kelly sizing runs on the sum here, not any single shard's balance.
    # shard_balances itself stays a live local — the coverage check and the
    # transfer planner below both consume the full per-shard picture.
    balance_cents = sum(shard_balances.values())
    report.balance_before = balance_cents / 100
    if balance_cents < MIN_BALANCE_CENTS:
        # Don't waste API calls scanning when there's insufficient capital to trade
        message = (f"Balance ${balance_cents / 100:.2f} is below minimum "
                   f"${MIN_BALANCE_CENTS / 100:.2f} — skipping run.")
        logging.warning("%s", message)
        report.message = message
        return EXIT_SKIPPED_LOW_BALANCE

    # Current open positions, each with its side and cost. A held market is
    # never traded again, except an exact held pair this run's settings add to;
    # held_listing["complete"] says whether the listing was read to its end
    held_listing: dict = {}
    held_positions    = get_held_positions(client, complete_out=held_listing)
    held_tickers      = set(held_positions)

    # Read the exchange's per-shard status breakdown so ingest can drop shards
    # that aren't trading. Returns None on the pre-sharding shape, which
    # degrades to single-shard semantics (keep everything). The full dict is
    # kept — ensure_shard_collateral below also reads each shard's
    # intra_exchange_transfers_active off it.
    shard_statuses    = fetch_shard_statuses(client)
    inactive_shards   = inactive_shard_indexes(shard_statuses)

    # Fetch all open markets (every shard, tagged — only trading-inactive
    # shards are dropped) and pre-filter held tickers for downstream efficiency
    markets           = fetch_open_events_with_markets(client, inactive_shards=inactive_shards)

    # Verify the exchange's advertised shards were actually covered by this
    # fetch BEFORE any held-ticker/horizon filtering trims the market list —
    # a blind spot must be measured against what ingest actually saw, not a
    # subsequently-filtered view of it. Only shards holding money count as
    # funded, per check_shard_coverage's contract.
    _log_shard_coverage(
        shard_statuses,
        {m.exchange_index for m in markets},
        {s for s, c in shard_balances.items() if c > 0},
    )

    # An ingest that produced nothing scanned nothing — an exchange-wide halt
    # dropping every market, or an empty ingest whose cause the status
    # breakdown could not name. Either way that is not "no edge this week":
    # return the dedicated code so scheduler.run_job never records the Monday
    # slot as satisfied (TS-01, VI-02). Evaluated here, before the held-ticker
    # and horizon filters below, so a filter that legitimately empties the list
    # is never mistaken for a blind run.
    blind_reason = _blind_run_reason(markets, shard_statuses, inactive_shards)
    if blind_reason:
        logging.warning("%s", blind_reason)
        report.message = blind_reason
        return EXIT_NO_TRADEABLE_SHARDS

    # Our positions' ladders, read before held markets are dropped; None
    # means one could not be identified, so no time-series trade this run.
    # Each identified market's own labels land in held_labels, for held_pairs
    held_labels: dict = {}
    held_ladders      = resolve_held_ladders(client, markets, held_tickers,
                                             labels_out=held_labels)
    # The exit code of every clean return below
    clean_exit        = EXIT_OK if held_ladders is not None else EXIT_TIME_SERIES_SKIPPED
    # The exact held pairs this run may add to: only with the setting on, and
    # only when every held market was listed and identified, since an unknown
    # one could share a pair's ladder, and being alone on its ladder is what
    # makes adding to a pair safe
    add_on_pairs: dict = {}
    if settings.add_to_held_pairs:
        if held_ladders is None or held_listing.get("complete") is not True:
            logging.warning("Not adding to held pairs this run: %s",
                            "a market the account holds could not be looked up"
                            if held_ladders is None
                            else "the list of the account's positions was cut short, "
                                 "so a held market may be missing from it")
        else:
            # Cross-module: the one definition of an exact held pair (both
            # markets alone on one ladder, one YES and one NO of equal size)
            add_on_pairs = held_pairs(held_positions, held_labels, balance_cents)
            # A pair that already holds its per-trade cap of the account value
            # can add nothing whatever Kelly says: it would only take its
            # group's one slot and size to nothing, so its markets stay
            # blocked like any other held market's (the last argument is the
            # cash in dollars, as held_pair_fraction takes it)
            full = {key for key, pair in add_on_pairs.items()
                    if held_pair_fraction(settings.size_cap, pair.cost_dollars,
                                          pair.account_value_dollars,
                                          balance_cents / 100) <= 0}
            if full:
                logging.info("Held pairs already at their size cap, not added to this "
                             "run: %d", len(full))
                add_on_pairs = {key: pair for key, pair in add_on_pairs.items()
                                if key not in full}
    # The markets of those pairs stay in (each pairs only with its own
    # partner); every other held market is dropped
    add_on_tickers    = {ticker for key in add_on_pairs for ticker in key}
    blocked_tickers   = held_tickers - add_on_tickers
    markets           = [m for m in markets if m.ticker not in blocked_tickers]

    # Optional opt-in cap so both bet types only see markets closing within
    # the requested window — a no-op (returns markets unchanged) when unset
    markets           = filter_markets_within_horizon(markets, args.max_horizon_days)

    # Run both pair detection paths: time-series (the run's entry rule, and no
    # pair on a ladder we hold but an exact held pair it adds to) and
    # same-title (a held pair's markets pair only with each other)
    if held_ladders is None:
        time_series_pairs = []
    else:
        time_series_pairs = find_time_series_pairs(
            client, blocked_tickers, markets, settings=settings, held_ladders=held_ladders,
            add_on_pairs=add_on_pairs,
        )
    same_title_pairs  = find_same_title_pairs(markets, blocked_tickers,
                                              add_on_pairs=add_on_pairs)
    # Merge both lists, preferring same_title when both scanners found the same pair
    candidate_pairs   = _dedup_pairs(same_title_pairs, time_series_pairs)
    # Category/tag filter, before enrichment so a dropped pair costs no book request
    candidate_pairs   = _filter_by_category(candidate_pairs, settings, client)
    # Replace best-ask prices with order book averages over the depth this
    # balance could actually buy, and validate liquidity under the run's rule
    candidate_pairs   = enrich_with_orderbook_prices(
        client, candidate_pairs, balance_cents, settings=settings,
    )

    if not candidate_pairs:
        # Names the run's entry rule, or says time-series was not searched
        message = _no_pairs_msg(settings=settings, time_series_searched=held_ladders is not None)
        logging.info("%s", message)
        report.message = message
        return clean_exit

    # Apply Kelly sizing to each candidate pair using the real account balance
    trade_specs   = _compute_trade_specs(candidate_pairs, balance_cents, settings)
    # Greedy selection by monthly_profit_ratio, one time-series trade per ladder
    portfolio     = select_portfolio(list(trade_specs.values()), balance_cents,
                                     held_ladders=held_ladders or frozenset())
    # Map pair id → TradeSpec for fast lookup in the pairs table display.
    # Keyed off the CANDIDATE each spec was built from, not off spec.pair:
    # compute_trade returns a re-priced copy of the pair (the marginal fill
    # price for the size it settled on), so id(spec.pair) no longer matches any
    # entry in candidate_pairs and every selected row would render as "—".
    chosen = {id(s) for s in portfolio}
    display_specs = {pid: s for pid, s in trade_specs.items() if id(s) in chosen}

    logging.info("Kalshi Pair Scan — Balance: $%.2f | Mode: PROD", balance_cents / 100)
    print_pairs_table(candidate_pairs, display_specs)

    if not portfolio:
        message = "No executable trades found."
        logging.info("%s", message)
        report.message = message
        return clean_exit

    _print_portfolio(portfolio, "Selected")

    # Re-fetch each pair's books and drop any that moved, under the run's rule
    portfolio = pre_execution_check(client, portfolio, settings=settings)
    if not portfolio:
        message = "All selected pairs failed pre-execution price check — no trades submitted."
        logging.info("%s", message)
        report.message = message
        return clean_exit

    # Move collateral to the shards the selected trades draw from — sizing is
    # portfolio-wide, but each order settles against its own shard's balance.
    # Trades whose shard could not be funded (transfer blocked, failed, or not
    # settled in time) are dropped here rather than submitted underfunded.
    portfolio = ensure_shard_collateral(
        client, portfolio, shard_balances, shard_statuses, dry_run=args.dry_run,
    )
    if not portfolio:
        message = (
            "No selected pair could be funded on its exchange shard — no trades submitted."
        )
        logging.info("%s", message)
        report.message = message
        # Never a bare return: sys.exit(None) exits 0, which would hide
        # EXIT_TIME_SERIES_SKIPPED
        return clean_exit

    # Recorded before the first order is sent, so a result written after an
    # exception or Ctrl-C during execute_trades (whose trades are not yet
    # recorded) still says orders may have been placed
    if not args.dry_run:
        report.submission_started = True

    # Submit orders sequentially per leg, concurrently across pairs (on the V2
    # path, one pair at a time until a NO fill has confirmed the order-side
    # mapping, for at most config.V2_MAPPING_CHECK_SERIAL_BUDGET_SECONDS; a
    # disproof stops every pair that starts after it)
    results = execute_trades(client, portfolio, dry_run=args.dry_run)
    # Each pair's outcome for the run result, before the trade log is written;
    # it never raises, so the trade log and its rescue dump are always reached
    report.trades = report_trades(results)

    # Read the post-trade balance for the Excel log separator row. Real orders
    # may already have filled at this point, so a failure here must not lose the
    # trade records — fall back to the pre-trade balance and keep going.
    try:
        balance_after = sum(verify_auth(client).values()) / 100
        # Only a balance actually read goes into the run result
        report.balance_after = balance_after
    except Exception as exc:
        logging.error(
            "Post-trade balance fetch failed: %s — logging with pre-trade balance", exc,
        )
        balance_after = balance_cents / 100

    # Append this run's results to the cumulative trade_log.xlsx file. If the
    # write fails (e.g. the file is open in Excel), dump every result to the log
    # so the record of real fills is never lost, then re-raise.
    try:
        # append_to_prod_log() already logs "Trade log updated: %s (%d new row(s))"
        # itself (BS-26) — don't duplicate that line here. The note marks the
        # toggles a flag moved and names the saved defaults the run started
        # from, so the workbook tells rows traded under a flag from rows traded
        # under the defaults, and one set of saved defaults from the next
        append_to_prod_log(
            results, balance_cents / 100, balance_after,
            run_note=(f"settings: {describe_live_settings(settings, reference)}"
                      + ("" if reference.origin == LIVE_DEFAULTS_FROM_CONFIG
                         else f" | defaults: {reference.origin}")),
        )
    except Exception as exc:
        logging.critical("Failed to write trade log: %s — rescue dump follows", exc)
        for r in results:
            # Cross-module: an add-on's counts sit on top of what the pair
            # already held, so its line names that held count too
            held = pair_held(r.spec.pair)
            # Both counts are printed: x is market A's leg and y is market B's,
            # and for a time-series pair the NO leg (the one that gets unwound)
            # is market B's, so y is the count a human must reconcile first.
            logging.critical(
                "  RESCUE | %s | %s | A=%s B=%s | x=%d y=%d%s cost=$%.2f incl. fees | %s",
                r.status,
                r.spec.pair.canonical_title,
                r.spec.pair.market_a.ticker,
                r.spec.pair.market_b.ticker,
                r.spec.x,
                r.spec.y,
                f" (adds to {count_text(held.count)} held)" if held is not None else "",
                r.spec.total_cost_with_fees,
                r.error or "",
            )
        raise

    if args.dry_run:
        message = "[DRY RUN] No orders were actually submitted."
        logging.info("%s", message)
        report.message = message
        return clean_exit

    n_ok       = sum(1 for r in results if r.status == "executed")
    n_rolled   = sum(1 for r in results if r.status == "rolled_back")
    n_orphaned = sum(1 for r in results if r.status == "rollback_failed")
    # "manual_review": the trader could not tell what a leg did (see
    # reporter.TradeResult) and sent no follow-up order. It needs a person as
    # urgently as a failed rollback, so both are counted in one alert. A
    # pair stopped by a disproof of the V2 NO-leg mapping sent nothing and
    # comes back "failed", so it is not counted here; the disproving pair's
    # manual_review is what returns EXIT_TRADES_NEED_ATTENTION.
    n_unknown  = sum(1 for r in results if r.status == "manual_review")
    message = (f"Submitted {n_ok} of {len(results)} order pair(s) successfully. "
               f"{n_rolled} rolled back, {n_orphaned} rollback failure(s), "
               f"{n_unknown} unknown fill state(s).")
    logging.info("%s", message)
    report.message = message
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

    return clean_exit


def _build_parser() -> argparse.ArgumentParser:
    """
    Build main.py's command-line parser: every flag, with the "live trading toggles" group.

    main() parses sys.argv with it. A test builds it to parse a list of flags,
    such as config.live_settings_argv's, the way main() would.

    Returns:
        argparse.ArgumentParser: The parser, with nothing parsed yet.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Kalshi Arbitrage Bot — scans for two kinds of mispriced contract "
            "pairs (same-title near-arbitrage pairs, and directional "
            "time-series pairs betting against the market's implied chance "
            "that an event first happens between two deadlines), sizes them "
            "with the Kelly criterion, and submits fill-or-kill orders."
        ),
    )
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
    parser.add_argument(
        "--result-file", type=pathlib.Path, default=None, metavar="PATH",
        help="(prod only) When the run ends, write what it did (outcome, trades, "
             "warnings) to PATH as JSON, replacing any file there; a usage error "
             "(exit 2, reason on stderr) or a kill signal leaves no file",
    )
    # Every flag defaults to None (not given); LiveSettings alone validates the
    # values — no choices, range check or literal here
    live = parser.add_argument_group(
        "live trading toggles",
        "Override one live default for THIS run only, in either mode "
        "(adding to held pairs: production runs only). The live defaults "
        "are the ones saved through python3 -m kalshi_betting.defaults_server "
        "(live_defaults.json); a run refuses to start without them. The weekly scheduler "
        "passes none of these flags, so a scheduled run trades exactly the saved defaults.",
    )
    live.add_argument(
        "--tier-floors", action=argparse.BooleanOptionalAction, default=None,
        help="Apply (or, with --no-tier-floors, drop) the deadline-gap tier floors on "
             "time-series pairs (default: the saved live defaults)",
    )
    live.add_argument(
        "--spread-min", type=float, default=None, metavar="X",
        help="Time-series spread-band FLOOR on pB - pA, 0-1 "
             "(default: the saved live defaults)",
    )
    live.add_argument(
        "--spread-max", type=float, default=None, metavar="Y",
        help="Time-series spread-band CEILING on pB - pA, 0-1 "
             "(default: the saved live defaults)",
    )
    live.add_argument(
        "--interval-discount", type=float, default=None, metavar="K",
        help="Time-series interval discount k, in (0, 1] "
             "(default: the saved live defaults)",
    )
    # The caps' grid step in percent (SIZE_CAP_STEP); argparse %-formats help, hence "%%"
    cap_step = f"{SIZE_CAP_STEP * 100:g}"
    live.add_argument(
        "--size-cap", type=int, default=None, metavar="PCT",
        help=f"Per-trade Kelly cap for every pair, in whole percent, in {cap_step}%% "
             "steps; 100 = no cap (default: the saved live defaults)",
    )
    live.add_argument(
        "--same-title-size-cap", type=int, default=None, metavar="PCT",
        help=f"Extra per-trade cap on same-title pairs, in whole percent, in {cap_step}%% "
             "steps; 100 = no extra cap beyond --size-cap "
             "(default: the saved live defaults)",
    )
    live.add_argument(
        "--add-to-held-pairs", action=argparse.BooleanOptionalAction, default=None,
        help="Let this run add to a pair the account already holds (exactly the same "
             "two markets, the same side on each), sizing the old and new contracts "
             "together as a share of the account value. --no-add-to-held-pairs never "
             "trades a held market. Production runs only: a dev run holds nothing "
             "(default: the saved live defaults)",
    )
    # Filed as the backtest dashboard files a trade (_filter_by_category)
    categories = live.add_mutually_exclusive_group()
    categories.add_argument(
        "--category", action="append", default=None, metavar="NAME",
        help="Trade only pairs filed under this Kalshi category, as the backtest "
             "dashboard's Category select names it (repeatable; case-insensitive; "
             "default: the saved live defaults)",
    )
    categories.add_argument(
        "--any-category", action="store_true", default=None,
        help="Trade any category this run, whatever the saved live defaults say",
    )
    tags = live.add_mutually_exclusive_group()
    tags.add_argument(
        "--tag", action="append", default=None, metavar="NAME",
        help="Trade only pairs whose series' FIRST Kalshi tag is NAME, under ANY "
             "category unless --category narrows it: the backtest dashboard's Tag "
             "option \"C · T\" is --category C --tag T (repeatable; case-insensitive; "
             "combined with --category by AND; default: the saved live defaults)",
    )
    tags.add_argument(
        "--any-tag", action="store_true", default=None,
        help="Trade any tag this run, whatever the saved live defaults say",
    )
    return parser


def _setup_logging(log_path: pathlib.Path) -> None:
    """
    Configure root logging with both a console handler and a file handler.

    Uses logging.basicConfig() with two handlers so every log line reaches
    both the terminal (for interactive/foreground runs) and the persistent
    log file (for later inspection, e.g. by the scheduler daemon). The file
    handler is opened with delay=True so merely calling this function — e.g.
    from a test or an import that doesn't go on to log anything — does not
    touch disk; the file is only created on the first emitted record.

    The file handler ROTATES (BS-25): a scheduler daemon runs this process
    weekly forever, and a plain FileHandler would grow kalshi_arb.log without
    bound (see backtest.py's kalshi_backtest.log for the failure mode that
    motivated this). 5MB across 3 backups is logging infrastructure sized for
    this CLI's own verbosity, not a strategy constant, so it stays inline here
    rather than moving to config.py.

    Args:
        log_path (pathlib.Path): Path to the log file to append to.

    Returns:
        None
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.handlers.RotatingFileHandler(
                log_path, maxBytes=5 * 1024 * 1024, backupCount=3, delay=True,
            ),
        ],
    )


def main() -> None:
    """
    CLI entry point for the Kalshi Arbitrage Bot — the live pipeline that
    scans for same-title near-arbitrage pairs and directional time-series
    pairs, sizes them, and trades them.

    Parses command-line arguments (--mode, --dry-run, --sandbox-balance,
    --max-horizon-days, and the "live trading toggles" group), exits 2 if
    config.ORDER_API_VERSION is not "v2" (config.order_api_version_error),
    checks --max-horizon-days, resolves the run's settings and the saved live
    defaults they were built from (_resolve_live_settings: with no defaults
    saved, a refused file or a bad flag it exits 2) — all before logging is
    configured or any request is made — then configures logging, builds the
    Kalshi client and runs _run_dev (sandbox simulation) or _run_prod (real
    trading). Ends with sys.exit() and the run's return code (the EXIT_*
    constants in config.py), which the scheduler reads. An unhandled exception
    propagates and exits 1.

    A production run that sends orders (--mode prod without --dry-run) first
    takes the machine-wide live-run lock (run_lock.acquire), before it builds
    the client, and holds it until the run mode returns or raises; the lock is
    released in a finally, so a second call in the same process takes it
    afresh. When another run still holds the lock after
    config.LIVE_RUN_LOCK_WAIT_SECONDS, this run logs a WARNING naming the
    holder and exits EXIT_RUN_IN_PROGRESS (50) without building a client or
    making any request. An error making or opening the lock file propagates
    (exit 1). Dry runs and dev runs send no orders, so they neither take the
    lock nor wait for it.

    With --result-file PATH (--mode prod only: in dev it exits 2 right after
    the order-path check, before logging is configured, a client is built or
    any request is made), any file already at PATH is removed right after
    that check, and a reporter.RunReport is filled in as the run goes: the
    run's settings and where its defaults came from, every WARNING-or-worse
    line logged after logging is configured (reporter.RunReportHandler, on
    the root logger), and what _run_prod records. It is written to PATH as
    JSON (reporter.write_run_report) in a finally, so however the run ends
    once logging is configured — with the exit code, or with no exit code
    and a one-line error (_http.api_error_summary) when an exception stops
    it, which still propagates — before the handler is removed and the lock
    released. A usage error that exits 2 before logging is configured (a bad
    flag, no or refused saved live defaults, a non-v2 order path) and a kill
    signal leave no file, so a caller reads the exit code first. Neither
    filling nor writing it changes the exit code.

    Returns:
        None: This function never returns to its caller — it always ends by
            calling sys.exit(code), which raises SystemExit.
    """
    # Every flag, their help and the live-toggle group (_build_parser)
    parser = _build_parser()
    args = parser.parse_args()
    # Exit 2 unless ORDER_API_VERSION is "v2", before anything is logged, a
    # client is built or an order could be sent
    problem = order_api_version_error()
    if problem:
        parser.error(problem)
    # The run result describes a production run only; refused right after the
    # order-path check, before logging, a client or any request
    if args.result_file is not None and args.mode != "prod":
        parser.error("--result-file is for --mode prod only")
    if args.result_file is not None:
        # An older result at PATH must never pass for this run's: whatever is
        # there when main() ends was written by this run, or nothing is (a
        # usage error below, or a kill signal). A file that cannot be removed
        # is left to the final write, which logs why it cannot replace it.
        with contextlib.suppress(OSError):
            args.result_file.unlink(missing_ok=True)
    if args.max_horizon_days is not None and args.max_horizon_days < 1:
        parser.error("--max-horizon-days must be a positive integer")
    # Read and validated BEFORE logging is configured: no saved live defaults,
    # a refused file or a bad flag exits 2 with nothing logged or requested
    settings, reference = _resolve_live_settings(args, parser)

    # Echo to the console (foreground/interactive runs) as well as the
    # persistent log file (later inspection, scheduler-spawned runs)
    _setup_logging(PROJECT_ROOT / "kalshi_arb.log")

    # With --result-file (prod only): what the run did, filled in as it goes
    # and written in the finally below, however the run ends from here on; the
    # handler copies in every WARNING-or-worse line logged from here on
    report = handler = None
    if args.result_file is not None:
        report = RunReport(dry_run=args.dry_run, started_at=datetime.now(UTC),
                           settings=describe_live_settings(settings, reference),
                           defaults=reference.origin)
        handler = RunReportHandler(report)
        logging.getLogger().addHandler(handler)
    # One production run that sends orders at a time on this machine
    # (run_lock); dry runs and dev runs send none, so they neither take the
    # lock nor wait for it
    lock_fd = None
    code = None
    try:
        if args.mode == "dev" and args.dry_run:
            # _run_dev always calls execute_trades(dry_run=True) regardless of
            # args.dry_run (dev never submits real orders), so --dry-run has no
            # effect in dev mode. Logged (not parser.error'd) after basicConfig so
            # it lands in kalshi_arb.log for anyone diagnosing "why didn't
            # --dry-run change anything" after the fact.
            logging.warning("--dry-run is inert in dev mode — dev never submits orders")

        if args.mode == "prod" and args.sandbox_balance != parser.get_default("sandbox_balance"):
            # Mirror of the --dry-run-in-dev twin above. _run_prod sizes on the
            # REAL per-shard balance from verify_auth and never reads
            # sandbox_balance, so passing it in prod silently does nothing — an
            # operator who meant to cap their exposure would get full-size live
            # orders instead. Logged rather than parser.error'd, same as the twin,
            # so it lands in kalshi_arb.log for later diagnosis (TS-19).
            logging.warning(
                "--sandbox-balance is inert in prod mode — prod sizes on the real "
                "account balance; use --dry-run to avoid submitting orders",
            )

        if args.mode == "prod" and not args.dry_run:
            # The machine-wide lock every real-money run shares; None when
            # another run still holds it after LIVE_RUN_LOCK_WAIT_SECONDS
            lock_fd = run_lock.acquire()
            if lock_fd is None:
                # The record names the run in the way (run_lock.holder)
                message = (f"Another live trading run is in progress "
                           f"({run_lock.holder().describe()}) — this run stops "
                           f"without contacting Kalshi, so nothing is sent "
                           f"(exit {EXIT_RUN_IN_PROGRESS}).")
                logging.warning("%s", message)
                if report is not None:
                    report.message = message
                code = EXIT_RUN_IN_PROGRESS
        if code is None:
            client = build_client(args.mode)  # returns KalshiClient authenticated via RSA key from secrets.json

            # Both run modes get the run's settings and the saved defaults they were
            # built from; a production run also gets the report to fill (None
            # without --result-file)
            if args.mode == "dev":
                code = _run_dev(client, args, settings, reference)
            else:
                code = _run_prod(client, args, settings, reference, report=report)
    except BaseException as exc:
        if report is not None:
            # One line, as the trade log records a failed order, never a traceback
            report.error = api_error_summary(exc)
        raise
    finally:
        if report is not None:
            # Written before the lock is released, with the exit code, or None
            # when an exception is on its way out (report.error names it);
            # never raises
            write_run_report(args.result_file, report, code)
            logging.getLogger().removeHandler(handler)
        if lock_fd is not None:
            # Released only once the run mode has returned or raised and its
            # result is written; nothing trades after this
            os.close(lock_fd)

    # Only sys.exit() communicates the outcome to a subprocess caller (the
    # scheduler) — a bare return here would always look like exit 0.
    sys.exit(code)


if __name__ == "__main__":
    main()
