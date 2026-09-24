"""
File: backtester.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Replays both pair strategies — same-title co-resolution pairs and the
    directional time-series bet — on the full history of settled Kalshi markets.
    Groups settled markets into potential time-series and same-title pairs, fetches
    hourly candlestick price series for each involved market, then scans weekly
    Monday snapshots to find the first date each pair was tradeable at the required
    threshold. Applies Kelly sizing to compute trade size, records actual P&L from
    settlement outcomes, deduplicates overlapping pairs by priority, and builds
    a daily equity curve. Results feed into dashboard.py for visualization.

Dependencies:
    Imports time_series_group_key (the single definition of the time-series
    grouping key, shared with the live scanner), event_series (the single
    definition of an event's series IDENTITY — the literal prefix, with every
    combo (KXMVE*) prefix collapsed onto one family — so the live and backtest
    one-series rules can never disagree), leg_sides, deadline_profile and
    cumulative_deadline_pair (the single definition of whether two legs are a
    two-cumulative-deadline pair, over the DEADLINE_CUMULATIVE/
    DEADLINE_SNAPSHOT/DEADLINE_UNKNOWN verdict constants), deadline_pair_refusal
    (the single definition of WHY a candidate is not one — DR-72; built on the
    same three verdicts cumulative_deadline_pair reads, so the boolean and the
    reason can never disagree) and its
    REFUSED_SNAPSHOT/REFUSED_NO_STATED_DEADLINE/REFUSED_SAME_DEADLINE
    constants, and stated_deadline / same_event_ladder with its SAME_DAY
    sentinel (the single definition of which calendar day a rung's deadline
    names and of how two rungs of one event are ordered and gapped — DR-73,
    so the live and backtest ladder rules can never disagree), from
    scanner.py; fee/model helpers
    (fee_leg_exact, fee_per_pair_approx, min_price_diff_for_gap,
    time_series_profit_prob), the backtest-only spread-band helpers
    time_series_spread_band and time_series_spread_too_wide (which
    _find_entry applies to time-series candidates; the first also validates
    and resolves a band up front in _entries_for_band, run_backtest_sweep,
    _sweep_from_candidates and _simulate_at_discount's completion line; no
    live module reads either), plus BUDGET_FRACTION,
    CANDLESTICK_FETCH_MAX_WORKERS, CANDLESTICK_PERIOD_INTERVAL_MINUTES (the
    grid _candle_window_open floors a market's open onto),
    LARGE_GROUP_WARN_THRESHOLD,
    INTERVAL_DISCOUNT_SWEEP and the band grid SPREAD_BAND_SWEEP_FLOORS /
    SPREAD_BAND_SWEEP_CEILINGS (both read only by _sweep_from_candidates),
    MAX_DEADLINE_GAP_DAYS, SAME_TITLE_CO_RESOLVE_PROB, SAME_TITLE_MIN_PRICE_DIFF,
    SETTLED_PREFILTER_CACHE_TAG, SHORT_DEADLINE_GAP_DAYS,
    TIME_SERIES_INTERVAL_PROB_DISCOUNT and TIME_SERIES_SAME_EVENT_LADDERS from
    config.py; fetch_all_settled_markets(),
    fetch_candlesticks(), and infer_category() from historical.py. Also
    depends on pandas (external) for the equity-curve DataFrame and numpy
    (external, a declared dependency pandas already pulls in) for counting
    the grouping-key hashes behind the groupable subset. Does NOT
    import strategy.py — Kelly sizing and portfolio selection are
    re-implemented inline against the same config.py constants, so a change
    to either sizing formula must be made in both places to keep live/backtest
    parity. Exports BacktestTrade, HalfSplit, SweepPoint,
    IntervalCalibrationBucket, IntervalCalibration, OutcomeLabelCoverage and
    BacktestSweep (BacktestTrade, BacktestSweep, OutcomeLabelCoverage and
    SweepPoint are consumed by dashboard.py, which also imports the private
    label helper _exact_label) plus run_backtest() and run_backtest_sweep()
    (called by backtest.py).

Notes:
    The backtester uses a two-pass approach: Pass 1 collects all potential entries
    (prices, dates, Kelly fraction — no sizing), keeps only the best entry per
    title group (mirroring the live scanners' one-pair-per-group rule), and then
    drops any time-series candidate whose ticker pair was also found as a
    same-title candidate — the same preference main._dedup_pairs applies live,
    since the same-title co-resolution model (identical questions must
    co-resolve) is simpler than the directional time-series bet, whose edge
    rests on the operator-tuned interval discount behind
    config.time_series_profit_prob. Pass 2 walks
    entries in chronological order (priority-ordered within a date using the
    ENTRY-TIME expected return, never realized results), maintains a running cash
    balance — sizing every candidate of an entry date against that checkpoint's
    opening balance, admitting them greedily against the running cash (mirroring
    main._run_prod + strategy.select_portfolio) and releasing settlement receipts
    on exit dates — and applies a greedy ticker-conflict filter
    so each market ticker appears in at most one OPEN trade at a time (the ticker
    is released on its trade's exit date, alongside the cash). This mirrors the
    live bot's Kelly sizing against one per-run balance snapshot and its
    one-active-position-per-ticker rule: get_held_tickers() reads positions with
    count_filter="position", so a settled ticker leaves the blocked set live too.

    The work is split at two boundaries. _prepare_candidates() is the half
    that depends on neither the backtest's time-series spread band nor the
    interval discount k — fetch, prefilter, census, grouping, pair extraction
    and candlesticks — returned as a _Candidates. _entries_for_band() is the
    _find_entry sweep at ONE spread band (the band acts only there; it holds
    no probability model, so its entries are k-independent).
    _simulate_at_discount() is everything that reads k — the Kelly gate, the
    dedups, Pass 2 and the equity curve — and returns a SweepPoint stamped
    with the resolved discount, band and population. run_backtest() is a thin
    wrapper over _prepare_entries() (_prepare_candidates() plus one
    _entries_for_band() pass at the default band, which is no band at all)
    and one _simulate_at_discount() call with k=None, which
    config.time_series_profit_prob resolves to the live sizer's constant.

    _interval_calibration() measures the EMPIRICAL discount from one band's
    k-independent entries — the realised in-between rate divided by the mean
    market-implied gap, pooled and per deadline-gap band — and
    _log_interval_calibration() reports it. Because it reads the prepared
    entries rather than _simulate_at_discount(), it is never filtered by the
    Kelly gate, which is what stops the estimate confirming whatever k
    produced it. It is a RECOMMENDATION ONLY: nothing here writes config.py,
    and live sizing keeps reading config.TIME_SERIES_INTERVAL_PROB_DISCOUNT.

    run_backtest_sweep() is the entry point that exposes all of that:
    _prepare_candidates() once, then _sweep_from_candidates() — one
    _find_entry pass per band, one calibration per band, and one
    _simulate_at_discount() per discount on config.INTERVAL_DISCOUNT_SWEEP
    (unioned with the caller's own, so the primary is always an exact grid
    member), returned as a BacktestSweep. With band_sweep it crosses every
    band of config.SPREAD_BAND_SWEEP_FLOORS x SPREAD_BAND_SWEEP_CEILINGS with
    that k grid and adds standalone time-series (ladders + cross-event),
    ladder, cross-event and same-title populations, a split-half check and an
    excluding-top-event check — the backtest-only scenario explorer; no live
    module reads a band.
    run_backtest() is untouched by it — same signature, same two-tuple — so
    every existing caller keeps working.

    Before grouping, _prepare_candidates() filters markets through _can_ever_enter(),
    a necessary-condition prefilter: _find_entry() can only open a trade at a
    Monday-09:00-UTC checkpoint on/after start_date, and requires both legs to
    have an hourly candle at-or-before that Monday (i.e. opened by then). A
    market whose [open_time, close_time - 1 day] window contains no such
    Monday can never appear in any entered pair, as either leg, in either pair
    type — dropping it up front avoids materializing it into any group at all.
    This matters because normalized-title groups can have 10,000+ members at
    current Kalshi volumes (hourly/intraday crypto ladders collapsing into one
    group) — without the prefilter, and without the close-time-windowed
    enumeration in _extract_pairs() for time-series groups, pair extraction is
    O(n^2) per group and infeasible (500B+ iterations observed on a single
    53k-member group). Neither optimization changes results: both only skip
    work that provably cannot produce an entry.

    Only the GROUPABLE eligible records are then grouped (SS-1), and this
    module builds no list of every eligible record of its own: the corpus is
    walked twice. The first walk (_index_eligible_keys) hashes each eligible
    record's time-series and same-title grouping keys — through
    _ts_group_key/_st_group_key, the very helpers the two grouping functions
    group on — and the second (_materialize_groupable) keeps only the records
    whose key hash is shared with another eligible record. Both groupings
    drop every single-member group, so a record sharing neither key can never
    appear in any pair; on a measured 7-day window only 184,255 of 7,260,952
    eligible records share one. The subset holds every member of every group
    of two or more, in order, so the groups and pairs are exactly those of
    the whole eligible list; a hash collision can only keep an extra record,
    which the exact grouping then drops. The corpus itself is whatever
    historical.fetch_all_settled_markets returns — since SS-1 a disk-backed
    historical.SettledCorpus that streams the assembled
    settled_markets_*.jsonl.gz cache afresh on each walk, so the eligible set
    is never resident as a whole; only a hit on a LEGACY
    settled_markets_*.json cache still hands over one list (read whole, as
    before), resident through both walks until _prepare_candidates releases
    it. The corpus must re-iterate identically, and a second walk that
    disagrees with the first on anything the subset was chosen from (the
    eligible count, or an eligible record's ticker or grouping fields) raises
    rather than misaligning the subset — as does a SettledCorpus walk that
    cannot read its file or ends on a different record count
    (historical.SettledCorpusError, a RuntimeError).

    Time-series pairs buy YES on the earlier-closing contract (market A) and
    NO on the later one (market B) — scanner.leg_sides is the only source of
    truth for the sides, and _settlement_receipt pays by side. Their
    settlement table therefore has exactly three cells (the premise behind
    that table — both legs being cumulative "by <date>" markets — is screened
    in _extract_pairs through scanner.cumulative_deadline_pair, the same helper
    the live finder uses): event by A (A=YES,
    B=YES — YES-on-A pays n), never by B (A=NO, B=NO — NO-on-B pays n), and in
    between (A=NO, B=YES — both legs worthless, the full stake is lost). A=YES
    with B=NO is impossible for a cumulative-deadline pair: a candidate that
    settled that way is excluded from Pass 1 (never traded, never paid) and
    counted, and one summary WARNING reports the count. Kalshi does list
    snapshot-style markets ("on <date>"), which the normalized-title grouping
    puts in one group with cumulative ones; _extract_pairs now refuses such a
    pair up front on its WORDING (scanner.cumulative_deadline_pair, shared with
    the live finder), so this counter is no longer the only signal that one was
    admitted — it is DEFENCE IN DEPTH behind a text heuristic, and a non-zero
    count now means one of the WARNING's named causes fired, most likely a
    wording false negative — though legs genuinely nested but ordered on an
    early REALIZED close, or strike-blind grouping on a cache without
    subtitles, can also produce it (DR-72) — rather than that nothing was
    watching.
"""
import logging
import resource
import statistics
import sys
from array import array
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    BACKTEST_MARKETS_RAM_WARN,
    BACKTEST_OUTCOME_LABEL_WARN_FRACTION,
    BACKTEST_RECORD_BYTES_ESTIMATE,
    BUDGET_FRACTION,
    CANDLESTICK_FETCH_MAX_WORKERS,
    CANDLESTICK_PERIOD_INTERVAL_MINUTES,
    INTERVAL_DISCOUNT_SWEEP,
    LARGE_GROUP_WARN_THRESHOLD,
    MAX_DEADLINE_GAP_DAYS,
    PRICE_EPSILON,
    SAME_TITLE_CO_RESOLVE_PROB,
    SAME_TITLE_MIN_PRICE_DIFF,
    SETTLED_PREFILTER_CACHE_TAG,
    SHORT_DEADLINE_GAP_DAYS,
    SPREAD_BAND_SWEEP_CEILINGS,
    SPREAD_BAND_SWEEP_FLOORS,
    TIME_SERIES_INTERVAL_PROB_DISCOUNT,
    TIME_SERIES_SAME_EVENT_LADDERS,
    fee_leg_exact,
    fee_per_pair_approx,
    min_price_diff_for_gap,
    time_series_profit_prob,
    time_series_spread_band,
    time_series_spread_too_wide,
)
from .historical import (
    fetch_all_settled_markets,
    fetch_candlesticks,
    infer_category,
)
from .scanner import (
    DEADLINE_CUMULATIVE,
    DEADLINE_SNAPSHOT,
    DEADLINE_UNKNOWN,
    REFUSED_NO_STATED_DEADLINE,
    REFUSED_SAME_DEADLINE,
    REFUSED_SNAPSHOT,
    SAME_DAY,
    cumulative_deadline_pair,
    deadline_pair_refusal,
    deadline_profile,
    event_series,
    leg_sides,
    same_event_ladder,
    stated_deadline,
    time_series_group_key,
)

# Seconds in one UTC day. Same value as historical._DAY_SECONDS, kept local
# rather than importing a private name.
_DAY_SECONDS = 86_400

# Deadline-gap bands the interval-discount calibration report groups its
# candidates into, as inclusive (lo, hi) day counts; the row label is derived
# from the pair, so the numbers exist exactly once. Reporting only — nothing
# sizes, prices or filters on these bands.
#
# The edges come from config wherever config defines one, so a band can never
# straddle the price tier it is labelled with: the short tier ends at
# SHORT_DEADLINE_GAP_DAYS and _find_entry rejects any pair beyond
# MAX_DEADLINE_GAP_DAYS, so every band's tier is min_price_diff_for_gap of its
# own upper edge. The short tier is split at 7 days purely so the report can
# show whether the empirical discount drifts WITHIN a tier — one row per tier
# could not reveal that.
_CALIBRATION_GAP_BANDS: tuple[tuple[int, int], ...] = (
    (0, 7),
    (8, SHORT_DEADLINE_GAP_DAYS),
    (SHORT_DEADLINE_GAP_DAYS + 1, MAX_DEADLINE_GAP_DAYS),
)

# Label of the calibration's all-bands row. Not a gap band, so it carries no
# single price tier (see IntervalCalibrationBucket.tier).
_CALIBRATION_POOLED_LABEL = "POOLED"

# The populations a band sweep simulates as standalone scenarios (see
# SweepPoint.population), and the run labels its split-half and
# excluding-top-event simulations carry on their completion lines only — the
# points those runs return are reduced to a return and a trade count and never
# kept. Those two checks run on the two populations that carry them, "all" and
# "time_series", hence one label set per population. _simulate_at_discount
# refuses anything else, so a typo cannot mislabel a scenario.
_SCENARIO_POPULATIONS = ("all", "time_series", "ladder", "cross", "same_title")
_CHECKED_POPULATIONS = ("all", "time_series")
_SIMULATION_LABELS = _SCENARIO_POPULATIONS + tuple(
    f"{population}/{run}" for population in _CHECKED_POPULATIONS
    for run in ("H1", "H2", "ex-top"))

# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    """
    Complete record of a single simulated pair trade from the backtest.

    Captures both the trade setup (entry prices, sizing, pair metadata) and the
    final outcome (settlement results, P&L, slippage) for use in performance analysis
    and dashboard generation. Which side each leg bought depends on pair_type
    (scanner.leg_sides): same_title buys NO on market A and YES on market B;
    time_series buys YES on market A (the earlier-closing contract) and NO on
    market B (the later one).

    Attributes:
        pair_type (str): Strategy variant used: "time_series" or "same_title".
        ticker_a (str): Kalshi ticker of market A — the pricier side, NO bought
            (same_title), or the earlier-closing contract, YES bought (time_series).
        ticker_b (str): Kalshi ticker of market B — the cheaper side, YES bought
            (same_title), or the later-closing contract, NO bought (time_series).
        title_a (str): Display title of market A.
        title_b (str): Display title of market B.
        category (str): Human-readable market category inferred from event_ticker prefix
            (e.g. "Crypto", "Sports", "Politics").
        entry_date (date): The Monday on which the trade was first tradeable and sized.
        exit_date (date): The date the later-settling market resolved; marks when cash returned.
        entry_pA (float): YES ask price of market A at entry. Range: [0.01, 0.99].
            The traded price of the market-A leg for time_series; for same_title
            it is the quote that made A the pricier side (reporting only).
        entry_pB (float): YES ask price of market B at entry. Range: [0.01, 0.99].
            The traded price of the market-B leg for same_title; for time_series
            it feeds the profit model together with entry_pA but is not traded.
        entry_nA (float): NO ask price of market A at entry (≈ 1 − yes_bid_A).
            Range: [0.01, 0.99]. The traded price of the market-A leg for
            same_title; reporting only for time_series.
        entry_nB (float): NO ask price of market B at entry (≈ 1 − yes_bid_B).
            Range: [0.01, 0.99]. The traded price of the market-B leg for
            time_series; reporting only for same_title.
        n (int): Number of contracts bought on each leg (x = y = n). Always >= 1.
            Both legs are always the same size.
        total_cost (float): Dollar cost of the contracts: n * (price_a + price_b),
            where the leg prices are (entry_nA, entry_pB) for same_title and
            (entry_pA, entry_nB) for time_series. Excludes taker fees (see fees).
        fees (float): Exact ceiling-rounded taker fee for both legs, charged at entry.
        outcome_a (str): Settlement result of market A — "yes" or "no".
        outcome_b (str): Settlement result of market B — "yes" or "no".
        actual_payoff (float): Gross dollar value received at settlement: $1 per
            contract for each leg whose market resolved to the side it bought
            (see _settlement_receipt for the per-type tables). Fees and entry
            cost are NOT deducted here — they are accounted in profit.
        profit (float): actual_payoff − total_cost − fees. same_title: negative
            only when A=YES and B=NO. time_series: negative only in the
            in-between cell (A=NO, B=YES); both win cells (A=YES, B=YES and
            A=NO, B=NO) realize exactly expected_payoff.
        profit_ratio (float): profit / (total_cost + fees). Return on the cash
            actually invested.
        monthly_profit_ratio (float): Realized profit_ratio scaled to 30 days:
            profit_ratio * 30 / holding_days. Reporting only — trade selection
            uses the entry-time expected ratio to avoid look-ahead bias.
        kelly_fraction (float): Capped Kelly fraction used for sizing, <= BUDGET_FRACTION.
        expected_payoff (float): NET profit in a win scenario after exact fees:
            n * (1 − price_a − price_b) − fees on the leg prices above. For
            same_title this is the guaranteed floor (every co-resolution
            outcome pays at least n); for time_series it is the profit realized
            in either win cell, while the in-between cell loses total_cost +
            fees instead. Always > 0 for recorded trades.
        slippage (float): profit − expected_payoff. same_title: positive when
            both legs paid (A=NO, B=YES), zero in the co-resolution cells,
            negative only in the loss cell. time_series: zero in both win cells
            and negative in the loss cell — there is no positive-slippage cell.
        holding_days (int): Calendar days between entry_date and exit_date. Always >= 1.
        balance_at_entry (float): Simulated cash balance in dollars at the OPEN
            of this trade's entry-date checkpoint — the base the Kelly budget
            was sized against, shared by every trade entering that same date
            (mirroring the single balance read at the top of a live run).
        deadline_gap_days (int | None): Calendar days between the two legs'
            deadlines — their close_times for a cross-event pair, their two
            STATED deadlines for a same-event ladder (DR-73), which is also
            what ordered its legs. Carried out of _find_entry (which selected
            the price tier and applied the MAX_DEADLINE_GAP_DAYS cutoff on
            exactly this number) rather than recomputed, so the number
            reported is always the one the pair was tiered on — for a ladder
            the two quantities routinely disagree (measured on a 1,506-market
            archive corpus: 101 of 408 ladder pairs have a ZERO-day close gap,
            37 of them closing at the identical instant, while every stated
            gap is in [1, 30]). None for same_title, which has no
            deadline-gap concept, and for any trade constructed without it
            (test fixtures). Reporting only — nothing sizes, prices or settles
            on this field.
        event_ticker (str): Event ticker of market A as traded (after
            _find_entry's canonicalization, so for same_title it is the
            pricier side's event), "" when the record carried none. The key a
            report groups trades by to measure how much of a run's P&L one
            event contributed. Reporting only.
        same_event_ladder (bool): True for a time_series trade whose two legs
            share one NON-EMPTY event ticker — a same-event deadline ladder
            (DR-73), which _extract_pairs only ever proposes while the ladder
            switch is on. False for a cross-event pair, for every same_title
            trade (two events of different series by construction), and when
            either leg's event ticker is missing, since an unknown event
            cannot be shown to be one event. Reporting only — it lets a report
            separate the ladder and cross-event populations, which price and
            settle identically here.
    """
    pair_type: str       # "time_series" | "same_title"
    ticker_a: str
    ticker_b: str
    title_a: str
    title_b: str
    category: str
    entry_date: date
    exit_date: date      # date the last-settling market resolved
    entry_pA: float      # YES ask of A at entry — the market-A leg price for time_series
    entry_pB: float      # YES ask of B at entry — the market-B leg price for same_title
    entry_nA: float      # NO ask of A at entry (≈ 1 - yes_bid_A) — the market-A leg price for same_title
    entry_nB: float      # NO ask of B at entry (≈ 1 - yes_bid_B) — the market-B leg price for time_series
    n: int               # contracts bought on each leg (x = y = n)
    total_cost: float
    fees: float          # exact both-leg taker fees, charged at entry
    outcome_a: str       # "yes" | "no"
    outcome_b: str       # "yes" | "no"
    actual_payoff: float
    profit: float
    profit_ratio: float
    monthly_profit_ratio: float  # realized profit_ratio * 30 / holding_days (reporting only)
    kelly_fraction: float        # capped Kelly fraction used for sizing
    expected_payoff: float  # n * (1 - price_a - price_b) minus fees — same_title floor / time_series win-cell profit
    slippage: float         # profit - expected_payoff
    holding_days: int
    balance_at_entry: float  # checkpoint opening balance the Kelly budget used
    # Calendar days between the two legs' deadlines — their close_times for a
    # cross-event pair, their two STATED deadlines for a same-event ladder
    # (DR-73) — carried out of _find_entry rather than recomputed, so it is
    # always the gap that chose the tier. None for same_title (no deadline-gap
    # concept) and for any trade constructed without it (test fixtures).
    # Reporting only — nothing sizes, prices or settles on this field, but
    # _interval_calibration DOES band k-hat on the same quantity via
    # _TimeSeriesOutcome.gap_days, so a ladder's bands are stated-gap bands.
    deadline_gap_days: int | None = None
    # Market A's event ticker ("" when absent) and whether the pair is a
    # same-event ladder (time_series, both legs on one non-empty event
    # ticker). Set by _simulate_at_discount; defaulted so a trade constructed
    # without them (test fixtures) still builds. Reporting only — nothing
    # sizes, prices or settles on either field.
    event_ticker: str = ""
    same_event_ladder: bool = False


@dataclass(frozen=True)
class HalfSplit:
    """
    A split-half out-of-sample check on one band-sweep scenario.

    The scenario's entries are split at BacktestSweep.split_date and each half
    is simulated ALONE from the run's initial balance, at the scenario's own
    band and k — so a band x k cell that only looks good because of one
    stretch of history shows it as two very different numbers. Only the two
    final-balance returns and trade counts are kept; neither half's equity
    curve is (a band sweep would otherwise hold two extra frames per
    scenario).

    Declared BEFORE SweepPoint on purpose: SweepPoint annotates a field with
    this class, and the annotation is evaluated when the class body runs (this
    module has no `from __future__ import annotations`), so a later
    declaration would raise NameError at import on CI's Python 3.11.

    A half with NO entries is not a measurement: its simulation enters
    nothing, so its h*_return reads 0.0 — indistinguishable from a half that
    entered and broke even. The entry counts are carried so a reader can tell
    the two apart (the dashboard renders such a half's return as "—" and
    leaves the cell out of its split-half correlation). Two things empty a
    half. At the primary band, for the time-series entries the split date is
    computed from: H1 is empty whenever AT LEAST half of them share the
    earliest entry date (one of two is enough), since the split date is their
    median_low and H1 is strictly before it — _sweep_from_candidates warns
    when that happens. And everywhere else — every other band's cells, and
    any "all" point that also carries same-title entries — the split date is
    that ONE primary-band date, not the point's own median, so either half is
    empty whenever all of the point's own entries fall on one side of it; no
    WARNING names those, but the dashboard blanks every empty half alike.

    Attributes:
        h1_return (float): Total return of the entries entering STRICTLY
            BEFORE split_date, simulated alone: (final portfolio value −
            initial balance) / initial balance. 0.0 when that half is empty
            — read h1_entries before trusting it.
        h2_return (float): The same for the entries entering ON or after
            split_date.
        h1_trades (int): Trades the first half's simulation entered.
        h2_trades (int): Trades the second half's simulation entered.
        h1_entries (int | None): Entries the first half's simulation was
            handed — 0 means that half is EMPTY and h1_return is not a
            measurement. None means not recorded (a hand-built instance);
            every sweep records it. Appended after the four original fields,
            with a default, so a positional four-field construction still
            builds.
        h2_entries (int | None): The same for the second half.
    """
    h1_return: float
    h2_return: float
    h1_trades: int
    h2_trades: int
    h1_entries: int | None = None
    h2_entries: int | None = None


@dataclass
class SweepPoint:
    """
    One complete simulation of the prepared entries at one interval discount.

    Returned by _simulate_at_discount(). Deliberately carries NO derived
    presentation metrics (total return, max drawdown, Sharpe): dashboard.py
    imports FROM this module, so importing its _max_drawdown()/_sharpe()
    helpers back here would be a circular import and a layering violation.
    Every such metric is computable from equity_df by the dashboard, using the
    helpers it already owns. The two exceptions below — halves and
    ex_top_event — are returns of simulations whose equity curves are NOT
    kept, so there is nothing for the dashboard to derive them from.

    Attributes:
        k (float): The RESOLVED interval discount this point was simulated at
            — never None. When _simulate_at_discount() was called with k=None
            (the "no override" sentinel, which config.time_series_profit_prob
            resolves at call time), this is
            config.TIME_SERIES_INTERVAL_PROB_DISCOUNT, the value the live
            sizer reads.
        trades (list[BacktestTrade]): One record per entered pair, in
            entry-date order; empty if nothing was ever entered.
        equity_df (pd.DataFrame): Daily equity curve with columns
            [date, portfolio_value, daily_return], opening one row before the
            run's start_date at the initial balance and flat at it when trades
            is empty. portfolio_value is cash plus open positions carried at
            cost, so it moves only on realized costs and P&L, never on
            deployment (see _build_equity_curve).
        spread_band (tuple[float, float] | None): The RESOLVED time-series
            spread band (floor, ceiling) this point's entries were detected
            under — the same tuple that was handed to _entries_for_band, so a
            reader never has to re-derive it. None when the simulation was
            given no band (run_backtest(), and the band-independent
            same_title_point); a band sweep stamps every scenario with its
            tuple, the default band's (0.0, 1.0) included.
        population (str): Which entries were simulated: "all" (every entry at
            this band, same-title included — the run's actual result),
            "time_series" (every time-series entry at this band — ladders and
            cross-event together, same-title excluded: the population the
            dashboard's heatmap and fragility banner read, since the band and
            k act on time-series pairs alone), "ladder" (only the time-series
            entries whose two legs share one non-empty event ticker), "cross"
            (every other time-series entry) or "same_title". Each population
            is its OWN standalone simulation from the initial balance, never a
            slice of an "all" run, so its return, drawdown and Sharpe are
            defined. On every point a BacktestSweep holds it is one of those
            five; _simulate_at_discount also accepts the completion-line
            labels "<all|time_series>/H1", "/H2" and "/ex-top" for the
            split-half and excluding-top-event runs, whose transient points
            are reduced to the numbers below and never kept.
        halves (HalfSplit | None): The split-half check — each half of this
            point's entries, split at BacktestSweep.split_date, simulated
            alone from the initial balance. Set only on the "all" and
            "time_series" points of a band sweep; None everywhere else.
        ex_top_event (tuple[str, float] | None): The concentration check:
            (event ticker, total return) — the event whose trades made the
            largest summed profit on this point (by BacktestTrade.event_ticker,
            ignoring trades with no event ticker, ties to the alphabetically
            first) and the return of ONE re-simulation of this point's entries
            without those whose market-A event ticker is that event. A
            re-simulation, never a subtraction of that event's P&L from this
            point's return: the survivors are re-sized against the cash the
            removed trades no longer consume, and a subtraction is not bounded
            below by −100%. Set only on the "all" and "time_series" points of
            a band sweep, and None there too when no trade names an event
            (there is no event to drop).
    """
    k: float
    trades: list[BacktestTrade]
    equity_df: pd.DataFrame
    spread_band: tuple[float, float] | None = None
    population: str = "all"
    halves: HalfSplit | None = None
    ex_top_event: tuple[str, float] | None = None


@dataclass
class IntervalCalibrationBucket:
    """
    One row of the interval-discount calibration report.

    A bucket is either one deadline-gap band from _CALIBRATION_GAP_BANDS or
    the pooled all-bands row. Every field is a measurement over the candidates
    in that bucket; nothing here is written back to config.py.

    Attributes:
        label (str): Row label — "<lo>-<hi>d" for a gap band, or
            _CALIBRATION_POOLED_LABEL ("POOLED") for the all-bands row.
        tier (float): The minimum YES-gap floor config.min_price_diff_for_gap
            returns for this band — its deadline-gap tier (0.15 or 0.30), or
            the backtest spread band's floor where that is higher, since the
            report labels each band with the floor its entries were actually
            detected under (_interval_calibration's spread_min). The pooled
            row spans every band and therefore has no single tier: it carries
            0.0, which the report renders as "-". Read `tier <= 0` as "not a
            single band", never as a real threshold.
        n (int): Candidates in the bucket. Can be 0 on the pooled row when
            every time-series candidate was a premise violation (empty gap
            bands are omitted from IntervalCalibration.buckets entirely).
        realised_rate (float): Fraction of the bucket that actually settled
            in-between (earlier NO, later YES) — the event the time-series bet
            loses on. 0.0 when n is 0.
        mean_implied (float): Mean market-implied in-between mass (pB - pA) at
            entry across the bucket. 0.0 when n is 0.
        empirical_k (float | None): realised_rate / mean_implied — the
            fraction of the market-implied in-between mass that actually
            materialized, i.e. the empirical counterpart of
            config.TIME_SERIES_INTERVAL_PROB_DISCOUNT. None when mean_implied
            is not positive (including the n == 0 case), since the ratio is
            undefined rather than zero.
    """
    label: str
    tier: float
    n: int
    realised_rate: float
    mean_implied: float
    empirical_k: float | None


@dataclass
class IntervalCalibration:
    """
    The empirical interval-discount report for one backtest window.

    Produced by _interval_calibration() from one spread band's prepared
    entries (_prepare_entries()' output, or one band's _entries_for_band()
    output inside a sweep), so it is INDEPENDENT of the interval discount k:
    it is computed once per band and is valid for every k at that band,
    which is why it hangs off BacktestSweep (the primary band's as
    `calibration`, every band's in `calibrations_by_band`) rather than off
    any single SweepPoint. It is NOT band-independent: a band changes which
    pairs enter, and when.

    Attributes:
        pooled (IntervalCalibrationBucket): The all-bands row, labelled
            _CALIBRATION_POOLED_LABEL. Its empirical_k is the single number
            the report recommends comparing against
            config.TIME_SERIES_INTERVAL_PROB_DISCOUNT.
        buckets (list[IntervalCalibrationBucket]): One row per deadline-gap
            band that had at least one candidate, in ascending gap order.
            Empty bands are omitted rather than reported as zero-width rows.
        excluded_premise_violations (int): Time-series candidates dropped from
            the denominator because they settled earlier-YES/later-NO, which
            is impossible for a cumulative-deadline pair. This is NOT the same
            number as _simulate_at_discount()'s `premise_violations` counter:
            that one is k-dependent (it counts only candidates that already
            passed the Kelly gate) while this one counts over the whole
            k-independent population, so this is generally the LARGER of the
            two. Neither is a bug in the other — see _interval_calibration().
    """
    pooled: IntervalCalibrationBucket
    buckets: list[IntervalCalibrationBucket]
    excluded_premise_violations: int


@dataclass(frozen=True)
class OutcomeLabelCoverage:
    """
    The outcome-label census over one backtest window's eligible markets.

    Produced by _report_outcome_label_coverage() as it emits its log line —
    reached through _log_outcome_label_coverage() for a whole iterable, or
    directly by _prepare_candidates, which counts during its first walk over
    the corpus (SS-1) — so the page and the log can never report two
    different numbers or disagree about where the warning floor sits: there is
    exactly ONE measurement, one pass and one threshold comparison per run.

    Holds scalars only (counts, fractions and one verdict) and no reference to
    any market record: _prepare_candidates (the first half of
    _prepare_entries) releases the corpus right after its second pass and
    the groupable subset immediately after pair extraction, to lower
    residency across grouping and the candlestick fetch (TS-07, SS-1), and a
    carrier that kept examples (sample tickers, a per-category breakdown)
    would pin those dicts alive past both statements.

    The population is the ELIGIBLE-MARKET CORPUS — every record of
    _prepare_candidates' corpus that passes the _can_ever_enter prefilter,
    counted during its first pass. Since SS-1 that is a SUPERSET of what the
    two grouping calls receive (only the eligible records that share a
    grouping key are materialized for them), but it is the same population
    this census has always covered: every eligible record, whether or not it
    can group. It is NOT the population the empirical k-hat is
    computed over (_interval_calibration measures over entered, binarily
    settled, non-premise-violating time-series candidates, a far smaller and
    differently-selected subset). Any rendering of these numbers must be
    phrased over the corpus, never over "the pairs behind k̂".

    Attributes:
        total (int): Eligible markets censused. Zero means the corpus was
            empty, not that the census failed to run.
        with_subtitle (int): Records carrying a non-blank `subtitle` — the
            outcome discriminator in the time-series key and the third
            component of the same-title key.
        with_event_title (int): Records carrying a non-blank `event_title` —
            the first component of the same-title key.
        subtitle_fraction (float | None): with_subtitle / total, or None when
            total is 0. None rather than 0.0 because the fraction is UNDEFINED
            on an empty corpus, matching the helper's own empty-list branch,
            which reports no coverage rather than 0%.
        event_title_fraction (float | None): with_event_title / total, or None
            when total is 0, for the same reason.
        cumulative_markets (int): Records worded as a cumulative "by <date>"
            deadline (scanner.deadline_phrasing). Only a pair of these can be a
            time-series candidate, so a ZERO here on a non-empty corpus means
            this run can produce no time-series trade at all — the one reading
            of this census that is actionable on its own.
        snapshot_markets (int): Records worded as a snapshot ("price ON
            <date>", "in <Month>", "after <date>"). Refused as a time-series
            leg because a snapshot probability need not nest inside another's:
            "after <date>" nests the wrong way, and a shared-start "after X and
            before Y" window CAN nest — a known, accepted over-refusal.
        unknown_deadline_markets (int): Records that carry no deadline wording
            the classifier recognises — the deadline may live in the event
            ticker, but nothing the pair-finders read can prove it, so nesting
            cannot be shown and they are refused too. Expected to dominate a
            combo-heavy corpus.

            These three are DESCRIPTIVE and carry no warning floor, unlike
            subtitle coverage: the cumulative FRACTION has no healthy baseline
            (most Kalshi markets are not deadline markets at all), so any
            threshold would be arbitrary and would fire on every run.
        below_floor (bool): Whether subtitle_fraction fell below
            config.BACKTEST_OUTCOME_LABEL_WARN_FRACTION — the SAME comparison
            the WARNING branches on, evaluated once and carried, so a reader of
            the log and a reader of the dashboard can never be told different
            things. False when total is 0 (nothing to warn about) and False
            when only event_title coverage is low, which never escalates (see
            config.py beside that constant for why).
    """
    total: int
    with_subtitle: int
    with_event_title: int
    subtitle_fraction: float | None
    event_title_fraction: float | None
    below_floor: bool
    # Declared AFTER below_floor: this dataclass is frozen but not kw_only, so
    # appending is the only way to add a field without reordering every
    # positional construction.
    #
    # REQUIRED, with no default (DR-71). A default of 0 let a construction
    # that forgot them render the dashboard's strongest sentence ("could not
    # have produced a time-series trade at all") as if it had been measured. A
    # forgotten keyword is now a TypeError at construction — the same "cannot
    # be forgotten" reasoning DR-66b used for returning the carrier in the
    # first place.
    cumulative_markets: int
    snapshot_markets: int
    unknown_deadline_markets: int


@dataclass
class _Candidates:
    """
    The band- and k-independent half of a backtest: fetch -> pairs -> candles.

    Returned by _prepare_candidates() and consumed by _entries_for_band(),
    which runs the _find_entry sweep over it at one spread band —
    once, inside _prepare_entries(), or once per band, inside
    _sweep_from_candidates(). Everything up
    to and including the candlestick fetch is independent of the band (the
    band acts only inside _find_entry's per-Monday price tests) and of the
    interval discount k, so one of these can feed an entry pass per band.

    It also carries the three inputs the entry pass must share with pair
    extraction — start_date, max_horizon_days and the ladder flag — so no
    later pass can be handed a different start date than the candles were
    fetched from, a horizon other than the one the run asked for, or a
    different ladder-flag ARGUMENT than _extract_pairs was handed (the DR-73c
    agreement rule: a pair the ladder rule admitted must be ordered and
    tiered by that same rule). _entries_for_band reads them from here and
    takes none of them as arguments.

    The ladder flag is stored UNRESOLVED, which bounds that last guarantee:
    when it is None, _extract_pairs and every later _find_entry call each
    resolve this module's TIME_SERIES_SAME_EVENT_LADDERS at their OWN call
    time, so they agree only if that name is not rebound between
    _prepare_candidates and the last entry pass. Before the split both
    resolutions sat inside one _prepare_entries call; a _Candidates object
    now stretches that window for as long as it is kept. Production never
    rebinds the name; a test or harness that patches it between the halves
    re-opens the DR-73c inversion (a ladder admitted on stated deadlines,
    then entered on close_time), and a caller that needs the guarantee
    unconditionally passes an explicit bool instead of None.

    Not frozen, deliberately: a caller that has finished every entry pass
    may del its candles_by_ticker and all_pairs attributes to release the
    candle series and the pair list (and every market record only it still
    references) before a long simulation phase — _sweep_from_candidates()
    does exactly that, which is why one _Candidates feeds one sweep (and why
    such an instance can no longer be repr()'d).

    Attributes:
        all_pairs (list): [((mA, mB, canon, group_key), pair_type), ...] — every
            candidate pair _extract_pairs proposed, in scan order: every
            time-series pair, then every same-title pair.
        candles_by_ticker (dict): Ticker -> hourly candle list, from
            _fetch_candles_parallel; every ticker of every pair in all_pairs
            is a key.
        label_coverage (OutcomeLabelCoverage | None): The eligible-market
            census _prepare_candidates counted in its first walk and reported
            through _report_outcome_label_coverage. _prepare_candidates
            always carries it — on the feasibility short-circuit, where no
            census is taken, it returns no _Candidates at all — so None only
            ever appears on a hand-built instance (a test stub).
        start_date (date): The backtest start date the markets, pairs and
            candles were prepared for.
        max_horizon_days (int | None): The optional bet-horizon cap,
            forwarded to every _find_entry call. None applies no cap.
        same_event_ladders (bool | None): The ladder flag EXACTLY as
            _prepare_candidates received it — UNRESOLVED, None included — the
            same argument both _extract_pairs calls were handed and every
            entry pass hands _find_entry (see above for when None resolves
            the same way in both).
    """
    all_pairs: list
    candles_by_ticker: dict
    label_coverage: OutcomeLabelCoverage | None
    start_date: date
    max_horizon_days: int | None
    same_event_ladders: bool | None


@dataclass
class BacktestSweep:
    """
    Everything one backtest run produces across every interval discount — and,
    when the spread-band sweep is on, across every band of the grid.

    Returned by run_backtest_sweep(). One preparation pass (the expensive,
    network-bound half, _prepare_candidates) feeds every point here, so the
    whole aggregate costs one fetch, one _find_entry pass per band, and one
    sizing/selection pass per simulated scenario.

    Attributes:
        primary (SweepPoint): The point at the effective discount and the
            primary spread band — the run's actual result, and the one a
            caller that wants a single answer should read. It is the SAME
            object as the matching entry of points (and of scenarios, on a
            band sweep), never a copy. There is deliberately no separate
            primary_k field: primary.k already carries the resolved discount,
            and a second copy could disagree with it.
        points (list[SweepPoint]): The primary band's k sweep, population
            "all": one point per swept discount, ascending by k, always
            including primary. A single-element list when sweeping is off or
            the run was infeasible. Unchanged by the band sweep — every object
            here is also in scenarios, never simulated twice.
        calibration (IntervalCalibration | None): The empirical-discount
            measurement over the PRIMARY band's entries, or None when there
            was no time-series candidate to measure. It hangs off the sweep
            rather than off any point because it is k-independent — one
            measurement valid for every k at that band (see
            _interval_calibration). The same object as
            calibrations_by_band[primary.spread_band] on a feasible run.
        label_coverage (OutcomeLabelCoverage | None): The outcome-label census
            over this run's eligible-market corpus, or None when no census was
            taken (the Monday-feasibility short-circuit skips the fetch
            entirely, and a hand-built sweep never had a corpus). It hangs off
            the sweep for exactly the reason `calibration` does: one
            measurement over one corpus, k-independent, valid at every point.
            DEFAULTED so no existing construction breaks — but a caller that
            omits it renders "not measured" on the dashboard rather than the
            caveat, so the two production constructions — the infeasible
            branch of run_backtest_sweep() and _sweep_from_candidates() — must
            always pass it.
        scenarios (list[SweepPoint]): Every simulated (band, k) cell of a
            band sweep, band by band in ascending band order and ascending k
            within a band: the "all" point, then a "time_series", a "ladder"
            and a "cross" point for each of those populations that is
            non-empty at that band. Every "all" and "time_series" point
            carries halves, and ex_top_event whenever one of its trades names
            an event. Every object in points is in here (the primary band's
            "all" points).
            [] when the band sweep is off OR the window was infeasible —
            label_coverage (None only on an infeasible window) tells the two
            apart.
        same_title_point (SweepPoint | None): The same-title entries simulated
            alone, ONCE: a same-title pair prices on the fixed co-resolution
            prior and never reads the band, so this one point is valid at
            every band and every k (its k and spread_band stamps are nominal:
            the primary k, and None). None when the band sweep is off, the
            window was infeasible, or no same-title pair produced an entry.
        calibrations_by_band (dict): Band -> that band's own
            IntervalCalibration (or None), keyed by the RESOLVED band tuple,
            each labelling its tiers with the floor that band actually
            applied. Always holds the primary band on a feasible run; every
            grid band on a band sweep; {} on an infeasible window.
        same_event_ladders (bool | None): The RESOLVED ladder setting the
            run's pairs were extracted and entered under (DR-73) — the flag
            decides which pairs exist, so a report must say which it was.
            None means not recorded (a hand-built sweep).
        split_date (date | None): The date the split-half check (SweepPoint.
            halves) splits entries at — the median_low of the primary band's
            TIME-SERIES entry dates (same-title entries are left out, so they
            cannot move the split the time-series checks are read at), or the
            window's midpoint when that band has none. One date for every
            scenario and both checked populations, so every cell's halves
            cover the same two stretches of history. None when the band sweep
            is off or the window was infeasible.
    """
    primary: SweepPoint
    points: list[SweepPoint]
    calibration: IntervalCalibration | None
    label_coverage: OutcomeLabelCoverage | None = None
    scenarios: list[SweepPoint] = field(default_factory=list)
    same_title_point: SweepPoint | None = None
    calibrations_by_band: dict[tuple[float, float], IntervalCalibration | None] = field(
        default_factory=dict)
    same_event_ladders: bool | None = None
    split_date: date | None = None


@dataclass
class _TimeSeriesOutcome:
    """
    One time-series candidate reduced to the three numbers calibration needs.

    Internal to _interval_calibration()/_calibration_bucket(); never returned
    to a caller.

    Attributes:
        gap_days (int | None): Deadline gap the pair's price tier was selected
            from, carried out of _find_entry. None only if a time-series entry
            somehow reached here without one, in which case the observation
            still counts in the pooled row but lands in no gap band.
        implied (float): Market-implied in-between mass at entry, pB - pA.
        in_between (bool): Whether the pair actually settled in-between
            (earlier NO, later YES) — the time-series bet's only loss cell.
    """
    gap_days: int | None
    implied: float
    in_between: bool


def _settlement_receipt(n: int, outcome_a: str, outcome_b: str, pair_type: str) -> float:
    """
    Compute the gross dollar amount received at settlement for one pair trade.

    Each contract pays exactly $1 when its side wins and $0 otherwise, so the
    receipt is n per leg whose market resolved to the side that leg bought —
    scanner.leg_sides(pair_type) is the only source of the sides — and is
    independent of entry prices.

    same_title (n NO on market A + n YES on market B):

      A=YES, B=YES: n   [YES on B pays; NO on A worthless]
      A=NO,  B=YES: 2n  [both legs pay — best scenario]
      A=NO,  B=NO:  n   [NO on A pays; YES on B worthless]
      A=YES, B=NO:  0   [loss scenario — both legs worthless]

    time_series (n YES on the earlier market A + n NO on the later market B)
    has exactly three cells:

      A=YES, B=YES: n   [event by A — YES on A pays; NO on B worthless]
      A=NO,  B=NO:  n   [never by B — NO on B pays; YES on A worthless]
      A=NO,  B=YES: 0   [in between — the loss cell; both legs worthless]

    A=YES with B=NO is impossible for a cumulative-deadline pair (YES by the
    earlier deadline implies YES by the later one), so that combination is not
    a payout cell at all: it raises. run_backtest excludes such candidates in
    Pass 1 before ever sizing them, so the guard here is defensive and
    unreachable from the pipeline — it exists so no caller can ever price the
    fourth cell by accident.

    Entry cost and taker fees are deliberately NOT deducted here — callers
    subtract them exactly once when computing profit and the equity curve.

    Args:
        n (int): Number of contracts bought on each leg.
        outcome_a (str): Settlement result of market A — "yes" or "no".
        outcome_b (str): Settlement result of market B — "yes" or "no".
        pair_type (str): "time_series" or "same_title" — selects the sides via
            scanner.leg_sides (anything but "time_series" is same-title).

    Returns:
        float: Gross settlement receipt in dollars: n per leg that paid.

    Raises:
        ValueError: For a time_series pair whose earlier contract settled YES
            while the later settled NO — a premise violation, never a payout.
    """
    if pair_type == "time_series" and outcome_a == "yes" and outcome_b == "no":
        raise ValueError(
            "time-series premise violated: earlier contract settled YES but later settled NO"
        )
    # Sides bought on (market_a, market_b) — the pipeline's single side mapping
    side_a, side_b = leg_sides(pair_type)
    receipt = 0.0
    if outcome_a == side_a:
        # Market A resolved to the side bought there: $1 per contract
        receipt += n
    if outcome_b == side_b:
        # Market B resolved to the side bought there: $1 per contract
        receipt += n
    return receipt


def _leg_prices_for(
    pair_type: str, pA: float, nA: float, pB: float, nB: float,
) -> tuple[float, float]:
    """
    Return the per-contract cost of the side bought on each leg, from the four quotes.

    Dict-world mirror of scanner.leg_prices for the backtester, which carries a
    market's quotes as loose floats rather than a CandidatePair: (nA, pB) for a
    same-title pair (NO on market A, YES on market B) and (pA, nB) for a
    time-series pair (YES on the earlier market A, NO on the later market B).
    Every price the backtester sizes, fees or pays out on comes through here,
    so the leg mapping is written exactly once.

    Args:
        pair_type (str): "time_series" or "same_title" — anything but the exact
            string "time_series" is treated as same-title, matching
            scanner.leg_sides.
        pA (float): YES ask of market A, dollars in [0.01, 0.99].
        nA (float): NO ask of market A, dollars in [0.01, 0.99].
        pB (float): YES ask of market B, dollars in [0.01, 0.99].
        nB (float): NO ask of market B, dollars in [0.01, 0.99].

    Returns:
        tuple[float, float]: (price_a, price_b) — the cost of the side bought on
            market A and on market B respectively.
    """
    if pair_type == "time_series":
        return pA, nB
    return nA, pB


# ─── Eligibility prefilter ─────────────────────────────────────────────────────

def _parse_iso_date(value: str | None) -> date | None:
    """
    Parse an ISO 8601 timestamp string to a date, tolerating missing/bad input.

    Shared by _can_ever_enter() and _extract_pairs()'s close-time windowing so
    the "can't parse it → treat as unknown, not an error" behavior is written
    exactly once.

    Args:
        value (str | None): An ISO 8601 timestamp string (e.g. "close_time" or
            "open_time" from a market dict), or None/empty if absent.

    Returns:
        date | None: The parsed date, or None if value is falsy or fails to
            parse. Never raises.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).date()
    except (ValueError, TypeError):
        return None


def _parse_iso_datetime(value: str | None) -> datetime | None:
    """
    Parse an ISO 8601 timestamp string to a datetime, tolerating missing/bad input.

    Sibling of _parse_iso_date for the callers that need the time-of-day
    component, which the date-only helper throws away: the candlestick fetch
    window is expressed in unix seconds, so truncating a close_time to midnight
    would silently move the window. Same "can't parse it → treat as unknown,
    not an error" contract as _parse_iso_date.

    Args:
        value (str | None): An ISO 8601 timestamp string (e.g. "close_time"
            from a market dict), or None/empty if absent.

    Returns:
        datetime | None: The parsed datetime (naive or aware, exactly as the
            string expressed it), or None if value is falsy or fails to parse.
            Never raises.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _can_ever_enter(m: dict, start_date: date) -> bool:
    """
    Necessary-condition prefilter: could this market possibly appear in any
    entered pair, as either leg, of either pair type?

    _find_entry() only opens a trade at a Monday-09:00-UTC checkpoint inside
    [start_date, min(close_a, close_b) - 1 day], and requires an hourly candle
    at-or-before that Monday for BOTH legs — which requires each market to
    have opened on or before it. So a market whose own
    [open_time, close_time - 1 day] window contains no Monday on/after
    start_date can never satisfy that condition for any partner market,
    regardless of pair_type. Dropping such a market before grouping/pairing
    is therefore provably safe — it would have contributed entry=None to
    every possible pair anyway (see CLAUDE.md for the full invariant).

    Args:
        m (dict): Market dict as produced by historical._market_to_dict().
        start_date (date): Backtest start date — _find_entry() never scans a
            Monday before this.

    Returns:
        bool: True if the market MIGHT be enterable (keep it) — this includes
            the case where open_time or close_time is missing/unparseable,
            since then we can't prove ineligibility (also keeps older cache
            files, written before open_time was added, working correctly —
            just without the speedup). False only when we can prove no
            Monday checkpoint falls in the market's eligible window.
    """
    open_d = _parse_iso_date(m.get("open_time"))
    close_d = _parse_iso_date(m.get("close_time"))
    if open_d is None or close_d is None:
        # Can't prove ineligibility — keep it rather than risk dropping a
        # market that could actually enter a pair.
        return True

    lower = max(open_d, start_date)
    upper = close_d - timedelta(days=1)  # mirrors _find_entry's scan_end
    if lower > upper:
        return False

    # Advance to the first Monday on/after `lower` — identical convention to
    # _monday_timestamps' own advance-to-Monday step, so the two stay in sync.
    d = lower
    while d.weekday() != 0:
        d += timedelta(days=1)
    return d <= upper


# ─── Pair grouping (metadata only, no prices) ─────────────────────────────────

def _pair_key(m: dict) -> str:
    """
    Combined grouping key for a market dict — event title joined with market title.

    Mirrors scanner.pair_key() for the backtester's dict-based market representation.
    The event_title prefix prevents cross-event option-label collisions in MVE
    markets — e.g. two markets both titled "Trump" in unrelated events will have
    different event titles and therefore won't be grouped together.

    Falls back to the bare title when event_title is missing (older cache files
    or non-MVE markets).

    Args:
        m (dict): A market dict in the compact historical._market_to_dict form.

    Returns:
        str: "{event_title} | {title}" when event_title is present, otherwise
            just the title (falling back to subtitle, then ticker, if the
            market has no title).
    """
    event_title = m.get("event_title") or ""
    title = m.get("title") or m.get("subtitle") or m.get("ticker", "")
    if not event_title:
        return title
    return f"{event_title} | {title}"


def _identical_wording_dicts(mA: dict, mB: dict) -> bool:
    """
    Dict-world mirror of scanner._identical_wording over cached market records.

    Compares the RAW (title, subtitle, event_title) strings, so a pair whose
    two contracts are worded identically cannot be "the same question at two
    deadlines" — the deadline is not in the wording at all.

    Args:
        mA (dict): A market dict in the compact historical._market_to_dict form.
        mB (dict): A second market dict, same form.

    Returns:
        bool: True only when all three strings match exactly. Missing keys read
            as "" on both sides, exactly as the live helper reads a falsy
            attribute.
    """
    return (
        (mA.get("title") or "", mA.get("subtitle") or "", mA.get("event_title") or "")
        == (mB.get("title") or "", mB.get("subtitle") or "", mB.get("event_title") or "")
    )


def _same_series_dicts(mA: dict, mB: dict) -> bool:
    """
    Dict-world mirror of scanner._same_series over cached market records.

    Resolves the series through the same scanner.event_series the live path
    uses, so the collapse of every combo (KXMVE*) prefix onto one family
    applies here identically: two combo tickets share a series while sharing no
    literal prefix (DR-55).

    Fails CLOSED like the live helper: an absent or unreadable event_ticker on
    either side reads as the same series, so a pair whose fixture identity
    cannot be established is refused rather than replayed on the 95%
    co-resolution prior (DR-02, DR-54).

    Args:
        mA (dict): A market dict in the compact historical._market_to_dict form.
        mB (dict): A second market dict, same form.

    Returns:
        bool: True when both records resolve to the same series, or when either
            is unreadable.
    """
    sa, sb = event_series(mA.get("event_ticker")), event_series(mB.get("event_ticker"))
    return not sa or not sb or sa == sb


def _deadline_profile_dict(m: dict) -> tuple:
    """
    Dict-world mirror of scanner._market_deadline_profile over cached records.

    Only the FIELD EXTRACTION is mirrored — the classification itself is
    scanner.deadline_profile, called here, so the live finder and this one can
    never disagree about what counts as a cumulative-deadline market (pinned by
    AST in tests/test_strategy.py).

    Every key is read with `.get(...) or ""`, matching _identical_wording_dicts:
    a cached record legitimately carries subtitle=None (historical
    ._market_to_dict stores `subtitle or yes_sub_title`, which is None when the
    payload had neither), and old records predate `event_title` entirely.

    One divergence to know, which no AST pin can catch: `event_title` reaches
    essentially every LIVE market but only a small fraction of cached ones (see
    config.BACKTEST_OUTCOME_LABEL_WARN_FRACTION for the measured coverage), so
    a market whose DECIDING field is its event title (no marker in its subtitle
    or title) reads as its event title's verdict live (cumulative for a
    "by <date>" event title, snapshot for an "on <date>" one) and as unknown
    here. At the
    phrasing-rule level, where the cache carries subtitles, the backtest's
    profile equals the live one or is unknown: the spans come only from the
    field that decided the verdict (DR-69), so a blank event title cannot
    change a profile the subtitle or title decided. (Before DR-69 the live
    spans also folded in the event title's dates, so the two paths could
    differ in either direction.) That is not a pipeline guarantee: a blank
    cached event_title changes the time-series group key itself.

    Args:
        m (dict): A market dict in the compact historical._market_to_dict form.

    Returns:
        tuple[str, tuple[str, ...]]: The record's (verdict, deadline spans).
    """
    return deadline_profile(
        m.get("event_title") or "", m.get("title") or "", m.get("subtitle") or "",
    )


def _stated_deadline_dict(m: dict, profile: tuple) -> date | None:
    """
    Dict-world mirror of the field extraction scanner.stated_deadline() reads.

    The arithmetic itself is scanner.stated_deadline, called here, so the live
    finder and this path can never disagree about which calendar day a rung's
    deadline names (pinned by AST in tests/test_strategy.py) — only the field
    extraction differs, exactly as _deadline_profile_dict mirrors
    scanner._market_deadline_profile. Every key is read with `.get(...) or ""`
    for the same reason: a cached record legitimately carries subtitle=None,
    and old records predate `event_title` entirely.

    The profile is PASSED IN rather than recomputed, because _extract_pairs
    classifies each group member exactly once and compares the cheap results
    pairwise (DR-70). _find_entry, which holds no such memo, is a deliberate
    bounded exemption — see its own comment.

    Two cache-vs-live field divergences reach this reader, and NEITHER is the
    one-directional guarantee DR-69 gives deadline_profile — that rule holds
    because spans come only from the deciding field, while stated_deadline's
    CROSS-CHECK reads all three fields, so blanking one can DISARM a refusal
    the live path applies. Measured on the 2026-09-22 snapshot's 113,303
    actively-statused markets, and again as ladder PAIRS through
    _extract_pairs(..., same_event_ladders=True) on the same snapshot (378
    ladders live):

    * event_title, which reaches essentially every live market and only a
      small fraction of cached ones. Mostly the narrow direction: 71 markets
      read a date live and `unknown` from the cache shape, i.e. a missed
      ladder. But 1 goes the OTHER way — KXACAREPEAL-29-29JAN20, whose event
      title "before 2029" (2028-12-31) is disjoint from its own title and
      subtitle ("Before Jan 20, 2029"), so live REFUSES it for a field
      conflict and the blank-event_title cache dates it 2029-01-19. None reads
      a DIFFERENT day, and at pair level the two shapes agree exactly (378
      ladders either way, 0 extra, 0 missing).
    * subtitle, which a cached record legitimately carries as None, and which
      is the MATERIAL one because DR-69 makes it the deciding field: blanking
      it moves the decision to the TITLE, so the rung reads a different day
      rather than none. 649 markets live-only, 53 cache-only and 9 reading a
      DIFFERENT day (e.g. KXMEDIARELEASEPRISONBREAK-30JAN01-27JAN01, subtitle
      "Before Jan 1, 2027" live against the title's "before Jan 1, 2030" from
      the cache shape). At pair level the subtitle-blank shape forms 246
      ladders: 136 of the live 378 missed AND 4 admitted that the live finder
      REFUSES (KXALIENS, KXGROK-GROK5, KXLEAVEGROUPKATSEYE, KXNEWGLENN, each
      a Nov/Dec rung pair whose subtitle "Before November" states no year).

    So a ladder-enabled backtest's pair population is NOT a subset of the live
    one on a subtitle-blank corpus, and the k-hat the flip gate demands is
    measured on exactly such a corpus (see CLAUDE.md's DR-67 residual list and
    the plan's own "prove the corpus before spending on it" step).

    Args:
        m (dict): A market dict in the compact historical._market_to_dict form.
        profile (tuple): That record's (verdict, spans) from
            _deadline_profile_dict().

    Returns:
        date | None: The last calendar day this rung's deadline includes, or
            None when its wording states no single placeable day.
    """
    return stated_deadline(
        profile,
        m.get("event_title") or "", m.get("title") or "", m.get("subtitle") or "",
    )


def _ts_group_key(m: dict) -> str:
    """
    The time-series grouping key of one market dict — the single definition in
    this module.

    Read by BOTH _group_by_normalized_title, which groups on it, and
    _index_eligible_keys, which decides from it (and _st_group_key) which
    eligible records are worth materializing at all (SS-1). One definition is
    what makes that decision exact: a key the index computed differently from
    the grouping would drop a record the grouping needs.

    The key itself is scanner.time_series_group_key — the same helper the live
    finder calls, pinned by AST in tests/test_strategy.py — over _pair_key
    (event_title + market title, so an option label shared across unrelated
    events does not collide) and the subtitle, which keeps two different
    OUTCOMES (two strikes of one daily family) out of one group (DR-01).

    Args:
        m (dict): A market dict in the compact historical._market_to_dict form.
            A cached `subtitle` of None reads as absent, exactly as `or ""`
            reads it everywhere else.

    Returns:
        str: The normalized key. An empty string means the market is not
            grouped for time-series detection at all (its title normalizes
            away).
    """
    # The live finder's own key helper, never a local copy: the backtester
    # once keyed on its own normalize_title call and so reproduced DR-01
    # instead of detecting it.
    return time_series_group_key(_pair_key(m), m.get("subtitle") or "")


def _st_group_key(m: dict) -> tuple[str, str, str] | None:
    """
    The same-title grouping key of one market dict — the single definition in
    this module.

    Read by BOTH _group_by_exact_title and _index_eligible_keys, for the same
    reason _ts_group_key is shared: the groupable-subset decision (SS-1) is
    only exact if it is taken on the very key the grouping uses.

    Args:
        m (dict): A market dict in the compact historical._market_to_dict form.
            Every field reads through `or ""`, so a cached None, a blank string
            and an absent key are all the same "no value".

    Returns:
        tuple[str, str, str] | None: (event_title, title, subtitle), or None
            when title and subtitle are both empty — such a market carries no
            wording to match on and is not grouped for same-title detection.
    """
    event_title = m.get("event_title") or ""
    title = m.get("title") or ""
    subtitle = m.get("subtitle") or ""
    if not (title or subtitle):
        return None
    return (event_title, title, subtitle)


def _group_by_exact_title(markets: Iterable[dict]) -> dict[tuple, list[dict]]:
    """
    Group markets by exact (event_title, title, subtitle) tuple for same-title pair detection.

    Three-element key: the event_title component prevents cross-event option-label
    collisions in MVE markets; (title, subtitle) distinguishes markets within an event.
    The key is _st_group_key's, shared with _index_eligible_keys so the
    groupable-subset decision in _prepare_candidates is taken on this exact key.

    Grouping is deliberately unchanged by the one-series rule (DR-02, DR-54):
    two events of one recurring fixture still land in one group, and
    _extract_pairs is what refuses to pair them. Keeping the rule in one place
    mirrors the live scanner, where find_same_title_pairs groups first and
    filters inside its inner loop.

    Args:
        markets (Iterable[dict]): Market dicts in the compact
            historical._market_to_dict form, walked once.

    Returns:
        dict[tuple, list[dict]]: Mapping of (event_title, title, subtitle) ->
            member markets, for groups with >= 2 members and at least one of
            title/subtitle non-empty. Single-member groups are dropped.
    """
    groups: dict = defaultdict(list)
    for m in markets:
        key = _st_group_key(m)
        if key is not None:
            groups[key].append(m)
    return {k: v for k, v in groups.items() if len(v) >= 2}


def _group_by_normalized_title(markets: Iterable[dict]) -> dict[str, list[dict]]:
    """
    Group markets by date-stripped combined key (event_title + title) plus the
    outcome label (subtitle), for time-series pair detection.

    Mirrors the live scanner exactly by calling the same helper,
    scanner.time_series_group_key, through this module's one definition of the
    key, _ts_group_key — the two-link chain is pinned by AST in
    tests/test_strategy.py.

    Args:
        markets (Iterable[dict]): Market dicts in the compact
            historical._market_to_dict form, walked once.

    Returns:
        dict[str, list[dict]]: Mapping of normalized (event_title + title +
            subtitle) key -> member markets, for groups with >= 2 members. A
            market whose key normalizes to an empty string is dropped.

    Note:
        Day slices cached before the 2026-08 subtitle fix carry subtitle=None,
        which time_series_group_key reads as absent — such records still group
        by title alone, so an old cache reproduces the pre-DR-01 grouping. See
        CLAUDE.md's subtitle-drift gotcha for how to refresh them.
    """
    groups: dict = defaultdict(list)
    for m in markets:
        # Same key as the live scanner, through the same helper: _pair_key
        # combines event_title + market title (so an option label shared across
        # unrelated events does not collide) and the subtitle keeps two
        # different OUTCOMES — two strikes of one daily family — out of one
        # group. This mirror reproduced DR-01 and so could never detect it.
        norm = _ts_group_key(m)
        if norm:
            groups[norm].append(m)
    return {k: v for k, v in groups.items() if len(v) >= 2}


def _drop_cross_type_duplicates(candidates: list[dict]) -> list[dict]:
    """
    Drop time-series candidates whose ticker pair was also found as a same-title candidate.

    Mirrors main._dedup_pairs on the live path: when both scanners detect the
    same two markets, the same-title pair is kept because its co-resolution
    model is simpler (identical questions must co-resolve) than the directional
    time-series bet, whose edge depends on the interval discount behind
    config.time_series_profit_prob — see CLAUDE.md, "Pair dedup prefers
    same-title", and the same_title > time_series tie-break in
    strategy.select_portfolio(). Duplicate identity is the frozenset of the two
    tickers, so a pair discovered with its legs in the opposite order still
    matches.

    This has to happen in Pass 1, not Pass 2: Pass 2's ticker-conflict filter
    only prevents both copies being OPEN at the same time. Whenever the
    same-title copy is skipped there for sizing reasons (n < 1, or a
    non-positive expected payoff), it never claims the tickers, so the
    time-series duplicate is entered instead — a candidate the live pipeline
    would never have had at all.

    Args:
        candidates (list[dict]): Pass 1 candidate dicts, each carrying
            "pair_type" ("same_title" or "time_series") and the "mA"/"mB"
            market dicts (which carry "ticker").

    Returns:
        list[dict]: Every same-title candidate plus every time-series candidate
            whose ticker pair is not already covered by a same-title candidate,
            in the input's original order.
    """
    same_title_keys = {
        frozenset((c["mA"]["ticker"], c["mB"]["ticker"]))
        for c in candidates
        if c["pair_type"] == "same_title"
    }
    kept = [
        c for c in candidates
        if c["pair_type"] == "same_title"
        or frozenset((c["mA"]["ticker"], c["mB"]["ticker"])) not in same_title_keys
    ]
    dropped = len(candidates) - len(kept)
    if dropped:
        logging.info(
            "Dropped %d time-series candidate(s) already covered by a same-title candidate",
            dropped,
        )
    return kept


def _extract_pairs(
    groups: dict, *, same_event_ladders: bool | None = None,
) -> list[tuple[dict, dict, str, object]]:
    """
    Return list of (market_a, market_b, canonical_title, group_key) tuples where
    the two markets have different event_tickers — or, with same-event deadline
    ladders enabled, are two dated cumulative rungs of ONE event — AND are not
    two events of one series worded identically. No price filtering at this stage.

    The one-series rule (DR-02, DR-54) mirrors both live finders through
    _same_series_dicts / _identical_wording_dicts: two events resolving to one
    series are two instances of one recurring fixture, so identical wording
    across them is one question about two DIFFERENT events. Two combo tickets
    resolve to one series while sharing no literal prefix (DR-55). The 3-tuple
    (same-title) branch tests the series alone, because its group key already
    guarantees the wording is identical; the string (time-series) branch tests
    the conjunct, because there the wording is only date-stripped-equal and a
    genuine cumulative pair (deadline IN the wording) must survive.

    For string-keyed (time-series) groups only, both legs must also be
    CUMULATIVE-deadline markets ("will X happen BY <date>") stating two
    DIFFERENT deadlines — scanner.cumulative_deadline_pair over
    scanner.deadline_phrasing, the same helper find_time_series_pairs' item 4
    calls, applied here through _deadline_profile_dict (DR-67). A snapshot
    family ("price ON <date>") groups here exactly as it does live — the
    grouping key is untouched — and is refused here exactly as it is live,
    a heuristic over wording rather than a proof. 3-tuple-keyed (same-title)
    groups have no deadline concept and are untouched by this rule.

    The pair type is NOT a parameter: the shape of each group key (see below)
    decides which sweep applies, and run_backtest() attaches the pair_type
    label to each returned tuple itself.

    Group keys may be:
      - a string (normalized title+outcome group from
        _group_by_normalized_title), or
      - a 3-tuple (event_title, title, subtitle) from _group_by_exact_title.
    For the 3-tuple form, the display canonical is taken from title-or-subtitle,
    but the FULL 3-tuple (including event_title) is also returned as group_key —
    the one-pair-per-group dedup in run_backtest must key on the full group, not
    just the display title, or two unrelated events sharing an option label
    (e.g. "Trump" in two different events) would collide into a single group
    and silently drop one of the two legitimate pairs.

    For string-keyed (time-series) groups, members are sorted ascending by
    close_time and swept with a two-pointer window bounded by
    MAX_DEADLINE_GAP_DAYS + 1 day of margin: _find_entry() unconditionally
    rejects any time-series pair whose close dates differ by more than
    MAX_DEADLINE_GAP_DAYS, so pairs outside that window can never produce an
    entry and are skipped without ever being materialized as a candidate
    pair. The +1 day margin is slack only (it can never cause a pair within
    the true limit to be skipped) — _find_entry() still applies the exact
    `.days > MAX_DEADLINE_GAP_DAYS` cutoff itself. Members with a missing or
    unparseable close_time are dropped from this sweep (group-local only —
    _group_by_exact_title's same-title groups are untouched), because
    _find_entry() unconditionally requires close_time on both legs and
    returns None immediately without it, regardless of pair type.

    3-tuple-keyed (same-title) groups have no deadline-gap concept, so they
    stay naive — the eligibility prefilter (_can_ever_enter, applied in
    run_backtest before grouping) keeps these groups small in practice.

    SAME-EVENT DEADLINE LADDERS (DR-73), when same_event_ladders resolves
    True, are the mirror of scanner.find_time_series_pairs' ladder branch: two
    dated cumulative rungs of ONE event ("... before Sep 23, 2026?" and
    "... by Oct 16, 2026?", both KXSPACEXSTARSHIP-14) are a time-series pair,
    ordered by STATED deadline (scanner.stated_deadline / same_event_ladder)
    and capped on the STATED gap, never on close_time. They come from a
    SEPARATE per-event sub-pass over the same sorted `dated` list, carrying
    POSITIONAL indexes because group_profiles is positional; the windowed
    cross-event sweep above is untouched and still skips every same-event
    candidate, so the two populations cannot overlap.

    That sub-pass is deliberately UNWINDOWED. The close-date window exists
    because _find_entry rejects a CROSS-EVENT pair on its close gap, but a
    ladder is capped on its stated gap instead, and the two disagree
    routinely: in the archive 681 of 1,821 dated same-event pairs have a close
    gap of ZERO days (on the 101-slice sub-corpus, 101 of 408 ladder pairs, 37
    of which close at the identical INSTANT — the weaker reading is the one
    that matters, because _find_entry's cross-event branch orders on close
    DATETIMES and a zero-day gap at two times of day is still orderable),
    13 are ordered the wrong way round by realized close, and 463
    would be admitted on their close gap with a stated gap beyond the cap —
    against exactly 1 the other way, a pair whose stated gap is inside the cap
    while its realized closes sit further apart, which a windowed sub-pass
    would silently drop.

    Complexity, stated honestly: the sub-pass is O(B^2) per event bucket,
    NOT the O(N * window) the sweep pays. Measured bucket sizes (2026-09-22):
    max 26 on the live snapshot's 113,303 markets (KXNHLHART-27, 3,704
    same-event candidate pairs in all) — counted over every actively-STATUSED
    market, which is what this function sees; the live funnel in the
    TIME_SERIES_SAME_EVENT_LADDERS comment in config.py quotes 3,354 and max
    bucket 21 for the same snapshot AFTER the live price filter
    (scanner._filter_active_markets), so the two pairs of numbers describe two
    populations rather than disagreeing. Also max 3 on
    live_days/2026-09-08.json.gz (391,609 records, 6 candidate pairs); max 75
    on the strike-blind legacy slices, where a blank subtitle collapses a
    whole daily strike family onto one key — archive_days/2026-05-01.json.gz
    reaches 615,266 candidate pairs that way. That last figure is why the
    sub-pass does NO pairwise work at all while the switch is off (not even
    the bucketing).

    The LARGE_GROUP_WARN_THRESHOLD-style canary below covers ONE runaway event
    bucket, and nothing more: it fires on a single same-event bucket past that
    threshold, so it would stay silent through an exact repeat of the 615,266
    figure above, whose largest bucket is 75. That is deliberate rather than
    an oversight — the AGGREGATE cost is bounded by measurement instead, and
    the measurement is small: on that same 2026-05-01 slice the whole
    615,266-pair sub-pass costs +0.3 to +0.5 s against a ~19 s switch-off call
    (measured 2026-09-23, two runs each way), and it is paid only with the
    switch on. A per-call aggregate guard is the alternative and is
    deliberately not built; if the cost ever stops being negligible, sum
    len(idxs)*(len(idxs)-1)//2 across buckets here and warn past a second
    threshold.

    Args:
        groups (dict): Mapping of group key -> list of market dicts (the
            compact historical._market_to_dict form). Keys are either a
            normalized title+outcome string (time-series groups) or an
            (event_title, title, subtitle) 3-tuple (same-title groups).
        same_event_ladders (bool | None): Whether to run the same-event ladder
            sub-pass. None (the default) resolves THIS MODULE's
            TIME_SERIES_SAME_EVENT_LADDERS — the by-value binding of
            config.TIME_SERIES_SAME_EVENT_LADDERS taken at import — at CALL
            time, so a run-level override and a monkeypatch OF THAT NAME both
            take effect, which a def-time default would silently ignore.
            Patching config.TIME_SERIES_SAME_EVENT_LADDERS is a silent no-op
            here, exactly as it is for scanner: this is NOT the
            time_series_profit_prob(k=None) idiom, which works only because
            that helper lives in config.py and reads config's own global. Has
            no effect on 3-tuple-keyed (same-title) groups, which have no
            deadline concept.

    Returns:
        list[tuple[dict, dict, str, object]]: One (market_a, market_b,
            canonical_title, group_key) tuple per candidate pair, in group
            iteration order — the cross-event sweep's pairs for a group first,
            then that group's ladder pairs. Empty if no group has two members
            on different event_tickers of different event series whose
            wording, for a string-keyed group, states two different cumulative
            deadlines, and no group holds an admissible ladder.
    """
    pairs = []
    # Resolved at CALL time, never bound as a def-time default: the constant
    # is what a test monkeypatches and what --same-event-ladders overrides for
    # one run, and a `same_event_ladders: bool = TIME_SERIES_SAME_EVENT_LADDERS`
    # default would freeze the import-time value and ignore both. Same idiom
    # config.time_series_profit_prob uses for k (DR-73c).
    ladders_on = (TIME_SERIES_SAME_EVENT_LADDERS if same_event_ladders is None
                  else same_event_ladders)
    # Time-series candidates refused as not one question at two cumulative
    # deadlines, split by REASON (DR-72) — the mirror of the live scanner's
    # split, over deadline_pair_refusal, the one shared definition. Reported
    # once at the end of the call (silent at zero) — this function previously
    # reported nothing at all about refused pairs, so a rule that can empty
    # the strategy had no signal on this path.
    snapshot_skips = 0
    no_deadline_skips = 0
    same_deadline_skips = 0
    # DR-73's same-event ladder sub-pass keeps its OWN counters, for the same
    # reason the live branch does: they count candidates INSIDE one event, a
    # population the three above have never seen, and folding them in blurs
    # exactly the distinction DR-72 split apart. Each is silent at zero.
    #
    # Two of the live branch's counters have no mirror here, deliberately.
    # ladder_disabled_skips is live-only: counting the disabled population
    # here would mean enumerating every same-event pair just to feed a
    # counter, which is 615,266 pairs on one real strike-blind day slice — the
    # switch-off path must do no pairwise work at all. ladder_price_sum_skips
    # has nothing to count: this function applies no price filter of any kind.
    ladder_no_event_skips = 0
    ladder_identical_wording_skips = 0
    ladder_snapshot_skips = 0
    ladder_no_deadline_skips = 0
    ladder_same_deadline_skips = 0
    ladder_undated_skips = 0
    ladder_field_conflict_skips = 0
    ladder_same_day_skips = 0
    ladder_gap_cap_skips = 0
    ladder_pairs = 0
    # Whether this CALL saw any time-series group at all. _prepare_candidates
    # (the first half of _prepare_entries) calls this function twice — once
    # per grouping — and the ladder rule applies only to string keys, so
    # without this the same-title call would log a permanent "ladder
    # candidates: 0" that reads as the ladder pass having found nothing when
    # it never ran.
    saw_time_series_group = False
    for key, members in groups.items():
        if isinstance(key, str):
            canon = key
        else:
            # 3-tuple (event_title, title, subtitle) — use title-or-subtitle for display
            canon = key[1] or key[2]

        if len(members) > LARGE_GROUP_WARN_THRESHOLD:
            # Visibility only — not a cap. Confirms the prefilter/windowing
            # above are actually keeping group sizes tractable in practice.
            logging.warning(
                "Pair-extraction group %r has %d members after filtering — "
                "still large; verify the eligibility prefilter is firing as expected",
                canon, len(members),
            )

        seen: set[frozenset] = set()

        if isinstance(key, str):
            saw_time_series_group = True
            # Time-series: sort by close_time and sweep only the pairs within
            # the deadline-gap window (see docstring above) instead of the
            # naive O(n^2) double loop over the whole group.
            dated = [(_parse_iso_date(m.get("close_time")), m) for m in members]
            dated = [(d, m) for d, m in dated if d is not None]
            dated.sort(key=lambda pair: pair[0])
            margin = timedelta(days=MAX_DEADLINE_GAP_DAYS + 1)
            n = len(dated)
            # Classify each member's wording ONCE, positionally, then compare
            # the cheap results pairwise below. The sweep is O(n * window), so
            # classifying per CANDIDATE would re-run the regex tables about 60x
            # more often than per member (2 legs x ~31 in-window neighbours on
            # TestExtractPairsPerformanceSmoke's 50,000-member group). Scoped to
            # this group and dropped with it: a corpus-wide ticker->verdict map
            # would add residency in exactly the place TS-07 did work to
            # reduce it.
            group_profiles = [_deadline_profile_dict(m) for _d, m in dated]
            for i in range(n):
                close_a, mA = dated[i]
                for j in range(i + 1, n):
                    close_b, mB = dated[j]
                    if close_b - close_a > margin:
                        # Sorted ascending by close_time — every further j is
                        # at least this far from mA, so nothing later qualifies.
                        break
                    if mA["event_ticker"] == mB["event_ticker"]:
                        continue
                    # Mirror of the scanner's time-series conjunct: identical
                    # wording across two events of one series is two instances
                    # of one recurring fixture, not one question at two
                    # deadlines (DR-02, DR-54).
                    if _identical_wording_dicts(mA, mB) and _same_series_dicts(mA, mB):
                        continue
                    # Mirror of the scanner's cumulative-deadline rule, through
                    # the SAME scanner.cumulative_deadline_pair: the two legs
                    # must be one question at two different "by <date>"
                    # deadlines. A snapshot family ("price ON <date>") groups
                    # here exactly as it does live — the grouping key is
                    # untouched — and is refused here exactly as it is live.
                    # Deliberately a separate check from the one-series
                    # conjunct above, not fused into it: they refuse different
                    # shapes for different reasons.
                    if not cumulative_deadline_pair(
                        group_profiles[i], group_profiles[j]
                    ):
                        # Same decision as cumulative_deadline_pair, re-asked
                        # for its REASON (DR-72) through the one shared
                        # deadline_pair_refusal, so the verdict here and the
                        # boolean just tested can never disagree.
                        reason = deadline_pair_refusal(
                            group_profiles[i], group_profiles[j]
                        )
                        if reason == REFUSED_SNAPSHOT:
                            snapshot_skips += 1
                        elif reason == REFUSED_NO_STATED_DEADLINE:
                            no_deadline_skips += 1
                        elif reason == REFUSED_SAME_DEADLINE:
                            same_deadline_skips += 1
                        continue
                    pair_key = frozenset([mA["ticker"], mB["ticker"]])
                    if pair_key in seen:
                        continue
                    seen.add(pair_key)
                    pairs.append((mA, mB, canon, key))

            # ── DR-73: the same-event deadline ladder sub-pass ────────────
            # A SEPARATE pass, not a relaxation of the sweep above, for two
            # reasons. The sweep's `continue` on equal event tickers stays
            # exactly as it was, so every cross-event count and every
            # cross-event pair is untouched; and this pass is deliberately
            # UNWINDOWED, because a ladder is capped on its STATED gap while
            # the window bounds the CLOSE gap, and the two disagree
            # routinely (see the docstring's measured 463-against-1).
            #
            # Gated on the resolved flag, and the gate wraps the BUCKETING as
            # well as the pairwise loops: on a strike-blind legacy slice one
            # event bucket reaches 75 members and a whole slice reaches
            # 615,266 same-event candidate pairs, so a disabled census here
            # would cost real time on every switch-off run to report a number
            # nobody asked for. The live finder keeps that census instead,
            # where its memoized profiles make it free.
            if ladders_on:
                # Positional indexes, NOT members: group_profiles is indexed
                # by position in `dated`, and the stated-deadline memo below
                # keys the same way.
                event_buckets: dict[str, list[int]] = defaultdict(list)
                for idx, (_d, m) in enumerate(dated):
                    event_buckets[m.get("event_ticker") or ""].append(idx)

                for event_ticker, idxs in event_buckets.items():
                    if len(idxs) < 2:
                        continue
                    if not event_ticker:
                        # Two members sharing an EMPTY event ticker share no
                        # event at all, so nothing identifies the ladder they
                        # would belong to. Fails closed, exactly as it did
                        # before this pass existed; counted per CANDIDATE
                        # pair so the number means the same thing as the live
                        # finder's.
                        ladder_no_event_skips += len(idxs) * (len(idxs) - 1) // 2
                        continue
                    if len(idxs) > LARGE_GROUP_WARN_THRESHOLD:
                        # Visibility only, not a cap — the
                        # LARGE_GROUP_WARN_THRESHOLD idiom applied to the one
                        # quadratic loop in this function that has no window
                        # bounding it. Measured maxima are 26 live and 75 on
                        # a strike-blind legacy slice, so this cannot fire on
                        # any corpus on disk today; it exists so a future one
                        # cannot reintroduce the blow-up silently.
                        logging.warning(
                            "Same-event ladder bucket %r in group %r holds %d dated "
                            "rungs — the ladder sub-pass is O(B^2) and unwindowed; "
                            "verify this event is a deadline ladder and not a "
                            "strike family grouped strike-blind",
                            event_ticker, canon, len(idxs),
                        )
                    # Each rung's stated deadline, read ONCE per member and
                    # only for the members that actually land in a >= 2
                    # bucket (DR-70's rule: a group of N members yields
                    # O(N^2) candidates, and one event's ladder is the
                    # densest such group there is; an index appears in
                    # exactly one bucket, so this whole map is one read per
                    # member). Each value is a PAIR of readings — the
                    # cross-checked deadline, and the same read with the
                    # cross-check disarmed (stated_deadline consults only its
                    # three FIELD arguments, so "" for all three leaves the
                    # deciding field's own day) — which is what separates
                    # "this rung states no placeable day" from "this rung's
                    # own fields name irreconcilable days" when a refusal is
                    # counted. Same two-value shape, and the same second
                    # read, as the live finder's ladder_deadlines map.
                    rung_deadlines = {
                        idx: (
                            _stated_deadline_dict(dated[idx][1], group_profiles[idx]),
                            stated_deadline(group_profiles[idx], "", "", ""),
                        )
                        for idx in idxs
                    }
                    for x, i in enumerate(idxs):
                        for j in idxs[x + 1:]:
                            # Fresh per-candidate locals, never the bucket
                            # entries: a ladder may SWAP its legs below, and
                            # the same index reappears in every later
                            # candidate of this bucket.
                            mA = dated[i][1]
                            mB = dated[j][1]
                            if _identical_wording_dicts(mA, mB):
                                # Same event AND identical wording: the
                                # deadline is not in the wording, so there is
                                # nothing here to order two rungs by (DR-02's
                                # reasoning, one level in).
                                # cumulative_deadline_pair below would refuse
                                # it too, but counting it here keeps the
                                # ladder counters about ladders — it is the
                                # largest single ladder refusal live.
                                ladder_identical_wording_skips += 1
                                continue
                            if not cumulative_deadline_pair(
                                group_profiles[i], group_profiles[j]
                            ):
                                # The same three DR-72 reasons as the sweep
                                # above, through the same one shared
                                # deadline_pair_refusal, on their own
                                # counters.
                                reason = deadline_pair_refusal(
                                    group_profiles[i], group_profiles[j]
                                )
                                if reason == REFUSED_SNAPSHOT:
                                    ladder_snapshot_skips += 1
                                elif reason == REFUSED_NO_STATED_DEADLINE:
                                    ladder_no_deadline_skips += 1
                                elif reason == REFUSED_SAME_DEADLINE:
                                    ladder_same_deadline_skips += 1
                                continue
                            deadline_a, deciding_a = rung_deadlines[i]
                            deadline_b, deciding_b = rung_deadlines[j]
                            ladder = same_event_ladder(deadline_a, deadline_b)
                            if ladder is None:
                                # Two findings with different remedies,
                                # counted apart: a rung whose wording states
                                # no placeable day (a parser gap — year-less
                                # wording, mostly) against one whose own
                                # fields name irreconcilable days (stale
                                # wording in one of them).
                                if (deadline_a is None and deciding_a is not None) or (
                                    deadline_b is None and deciding_b is not None
                                ):
                                    ladder_field_conflict_skips += 1
                                else:
                                    ladder_undated_skips += 1
                                continue
                            if ladder is SAME_DAY:
                                # Both rungs name ONE calendar day — one
                                # deadline spelled two ways ("by Mar 31,
                                # 2027" beside "before Apr 1, 2027"), not a
                                # two-rung ladder. Compared with `is`, never
                                # truthiness: SAME_DAY is a non-empty string.
                                ladder_same_day_skips += 1
                                continue
                            swap, stated_gap = ladder
                            if stated_gap > MAX_DEADLINE_GAP_DAYS:
                                # The ladder's OWN cap, on the STATED gap.
                                # _find_entry re-applies it on the same
                                # arithmetic, so this is the sweep's
                                # window-plus-exact-cutoff contract mirrored
                                # for ladders.
                                ladder_gap_cap_skips += 1
                                continue
                            if swap:
                                # market_a must be the EARLIER contract,
                                # which for a ladder means the earlier STATED
                                # deadline: the close_time sort above cannot
                                # order rungs that close at one instant.
                                mA, mB = mB, mA
                            pair_key = frozenset([mA["ticker"], mB["ticker"]])
                            if pair_key in seen:
                                continue
                            seen.add(pair_key)
                            ladder_pairs += 1
                            pairs.append((mA, mB, canon, key))
        else:
            # Same-title: no deadline-gap constraint, stays naive.
            for i, mA in enumerate(members):
                for mB in members[i + 1:]:
                    if mA["event_ticker"] == mB["event_ticker"]:
                        continue
                    # Mirror of scanner.find_same_title_pairs' one-series rule:
                    # the group key already guarantees identical wording, so
                    # two events of one series are two fixtures (DR-02, DR-54).
                    if _same_series_dicts(mA, mB):
                        continue
                    pair_key = frozenset([mA["ticker"], mB["ticker"]])
                    if pair_key in seen:
                        continue
                    seen.add(pair_key)
                    pairs.append((mA, mB, canon, key))
    # Mirror of the live scanner's three-way split (DR-72), each silent at
    # zero: within the deadline-gap window this sweep already restricted
    # itself to, before any price filter runs (this function does no price
    # filtering at all — see the docstring).
    if snapshot_skips:
        logging.info(
            "Time-series candidates refused because a leg's deciding field "
            "is snapshot wording (within the deadline-gap window, before "
            "price filters): %d",
            snapshot_skips,
        )
    if no_deadline_skips:
        logging.info(
            "Time-series candidates refused because a leg's deciding field "
            "carries no recognised deadline wording or no comparable date "
            "(within the deadline-gap window, before price filters): %d",
            no_deadline_skips,
        )
    if same_deadline_skips:
        logging.info(
            "Time-series candidates refused because the two deciding fields "
            "state the same deadline, or truncate to one (within the "
            "deadline-gap window, before price filters): %d",
            same_deadline_skips,
        )
    # DR-73's own reporting, the mirror of the live finder's. Every line below
    # counts candidates INSIDE one event — a population none of the counters
    # above has ever seen — and each is silent at zero, so a run with the
    # switch off adds no line at all to this function's output. They say
    # "same-event sub-pass" rather than "within the deadline-gap window",
    # because that sub-pass is deliberately unwindowed (see the docstring).
    if ladder_no_event_skips:
        logging.info(
            "Same-event ladder candidates refused because the shared event "
            "ticker is empty (same-event sub-pass, before price filters): %d",
            ladder_no_event_skips,
        )
    if ladder_identical_wording_skips:
        logging.info(
            "Same-event ladder candidates refused because the two rungs' "
            "wording is identical (the deadline is not in the wording): %d",
            ladder_identical_wording_skips,
        )
    if ladder_snapshot_skips:
        logging.info(
            "Same-event ladder candidates refused because a rung's deciding "
            "field is snapshot wording: %d",
            ladder_snapshot_skips,
        )
    if ladder_no_deadline_skips:
        logging.info(
            "Same-event ladder candidates refused because a rung's deciding "
            "field carries no recognised deadline wording or no comparable "
            "date: %d",
            ladder_no_deadline_skips,
        )
    if ladder_same_deadline_skips:
        logging.info(
            "Same-event ladder candidates refused because the two rungs' "
            "deciding fields state the same deadline, or truncate to one: %d",
            ladder_same_deadline_skips,
        )
    if ladder_undated_skips:
        logging.info(
            "Same-event ladder candidates refused because a rung's deadline "
            "states no placeable calendar day (year-less wording, mostly — "
            "see scanner._span_deadline): %d",
            ladder_undated_skips,
        )
    if ladder_field_conflict_skips:
        logging.info(
            "Same-event ladder candidates refused because a rung's own "
            "wording fields name irreconcilable days (stated_deadline's "
            "cross-check): %d",
            ladder_field_conflict_skips,
        )
    if ladder_same_day_skips:
        logging.info(
            "Same-event ladder candidates refused because both rungs name "
            "one calendar day (one deadline spelled two ways): %d",
            ladder_same_day_skips,
        )
    if ladder_gap_cap_skips:
        logging.info(
            "Same-event ladder candidates worded as two different cumulative "
            "deadlines, refused at the %d-day STATED gap cap (price not "
            "evaluated): %d",
            MAX_DEADLINE_GAP_DAYS,
            ladder_gap_cap_skips,
        )
    if ladders_on and saw_time_series_group:
        # ALWAYS logged while the switch is on, zero included: a switch that
        # silently produces nothing must be distinguishable from one that is
        # working and finding nothing (DR-66). The live finder's counterpart
        # counts pairs that SURVIVED its one-best-per-group contest; this one
        # counts candidates, because that contest happens later here, in
        # _simulate_at_discount.
        logging.info(
            "Same-event ladder candidates among the time-series candidates: %d",
            ladder_pairs,
        )
    return pairs


# ─── Entry point detection from candlestick data ──────────────────────────────

def _monday_timestamps(start_date: date, end_date: date) -> list[int]:
    """
    Generate a list of Unix timestamps for every Monday in the given date range.

    Each timestamp corresponds to 09:00 UTC on the Monday. The backtest scans
    these weekly checkpoints to simulate the bot's Monday morning trading schedule.

    Args:
        start_date (date): First date of the scan range (inclusive). The function
            advances to the first Monday on or after this date.
        end_date (date): Last date of the scan range (inclusive).

    Returns:
        list[int]: Unix timestamps (09:00 UTC) for every Monday in [start_date, end_date].
            Returns an empty list if start_date > end_date or no Monday falls in range.
    """
    d = start_date
    # Advance to the first Monday (weekday() == 0) if start_date is not already one
    while d.weekday() != 0:
        d += timedelta(days=1)
    ts_list = []
    while d <= end_date:
        ts_list.append(int(datetime(d.year, d.month, d.day, 9, 0, tzinfo=UTC).timestamp()))
        d += timedelta(weeks=1)
    return ts_list


def _candle_at_or_before(candles: list[dict], ts: int) -> dict | None:
    """
    Find the most recent candlestick at or before a given Unix timestamp.

    Candles are assumed to be sorted ascending by their "ts" field. Returns the
    last candle whose "ts" is <= ts, or None if no such candle exists. ("ts" is
    derived from the API's end_period_ts when the candle is fetched — see
    historical.fetch_candlesticks — so this is equivalently "at or before the
    candle's period end", just read off the dict's own key rather than the
    wire field name.) This is used to read the closing price on or before a
    given Monday snapshot.

    Args:
        candles (list[dict]): List of candle dicts with a "ts" key (unix timestamp).
            Must be sorted ascending by "ts".
        ts (int): The target Unix timestamp to search at or before.

    Returns:
        Optional[dict]: The last candle dict with ts <= the target, or None if all
            candles are after the target timestamp or the list is empty.
    """
    result = None
    for c in candles:
        if c["ts"] <= ts:
            result = c
        else:
            # Candles are sorted ascending, so once we exceed ts we can stop
            break
    return result


def _find_entry(
    candles_a: list[dict],
    candles_b: list[dict],
    mA: dict,
    mB: dict,
    pair_type: str,
    start_date: date,
    max_horizon_days: int | None = None,
    same_event_ladders: bool | None = None,
    spread_band: tuple[float, float] | None = None,
) -> dict | None:
    """
    Find the first Monday where a potential pair was tradeable at the required threshold.

    Scans weekly Monday snapshots up to (not including) the day before the
    earlier of the two market close dates. The scan's start is the LATER of
    the backtest start date and one calendar year before that end point — the
    one-year figure bounds how far back the window can reach, it does not
    describe where the window ends. At each Monday, reads the candlestick
    prices, applies the price gap, price-sum, and fee filters from the live
    trading logic, and returns the entry data for the first qualifying week.

    Direction rules mirror the live scanner exactly:
      - time_series: market A is fixed as the EARLIER-closing contract —
        decided on the close DATETIMES, like scanner.find_time_series_pairs
        sorts its group members, so two contracts closing on the same UTC date
        at different times of day are ordered rather than tied. That parity is
        on the datetime-vs-date fix only, not on what close_time MEANS: live
        reads the SCHEDULED close of a still-open market, this function reads
        the REALIZED close Kalshi recorded for a settled one, which for an
        event that resolved early can sit before the original schedule — a
        later-deadline leg that resolved early is ordered first on the
        realized gap, a recorded pre-existing residual, not something this
        function's datetime ordering fixes (CLAUDE.md TS-06). A SAME-EVENT
        DEADLINE LADDER is the one exception and is the point of DR-73: with
        same_event_ladders resolved True, two markets sharing a non-empty
        event_ticker are ordered and gapped on their STATED deadlines
        (scanner.stated_deadline / same_event_ladder) instead, because a
        settled event closes every rung at one instant — the realized-close
        residual above is not a residual there but the normal case. Such a
        pair returns None when either deadline cannot be read, or when both
        rungs name one day. An entry
        additionally requires pB − pA >= the deadline-gap-tiered threshold from
        min_price_diff_for_gap (15% for gaps <= 15 days, 30% for 16-30 days) —
        the LATER contract priced higher by at least the tier is the anomaly
        the strategy disputes (the market implies an outsized probability that
        the event first happens between the two deadlines). A pricier earlier
        contract is never a candidate. The legs are YES on A at pA and NO on B
        at nB, so the price-sum ceiling (pA + nB <= 1 − threshold) and the fee
        check are applied to those two leg prices — never to (nA, pB), which
        are reporting-only quotes for this pair type. The deadline gap driving
        the tier (and the 30-day cutoff) is the ABSOLUTE timedelta.days on the
        two close_time datetimes, exactly as scanner.deadline_gap_days computes
        it (order-independent) — not calendar-date subtraction, which counts a
        day boundary the live path does not. For a same-event ladder it is the
        STATED deadline gap instead, exactly as scanner.same_event_ladder
        computes it and as scanner.pair_gap_days hands it to the live
        downstream sites; close_time is not consulted for the tier there at
        all.
      - same_title: market A is canonicalized per Monday as the more expensive
        side (the two contracts ask the identical question, so direction is
        price-only); the legs are NO on market A (nA) and YES on market B (pB),
        and the entry requires pA − pB >= SAME_TITLE_MIN_PRICE_DIFF with the
        ceiling and fee check on (nA, pB).
    In both cases the traded pair of prices comes from _leg_prices_for, and
    both of them must be live [0.01, 0.99] quotes.

    The BACKTEST-only spread band (spread_band, resolved through
    config.time_series_spread_band) narrows the time-series rule and nothing
    else. Its floor is layered on the deadline-gap tier —
    threshold = min_price_diff_for_gap(gap_days, spread_min=floor), i.e.
    max(tier, floor) — and that raised threshold drives BOTH the gap test and
    the leg-price-sum ceiling (price_a + price_b <= 1 − threshold), so the sum
    ceiling stays tied to the floor exactly as it is tied to the tier live.
    Its ceiling refuses a Monday whose pB − pA exceeds it
    (config.time_series_spread_too_wide, the one place the ceiling's
    PRICE_EPSILON lives, on the keep side like the floor's) — that Monday
    only: the scan moves on, because a LATER Monday whose spread has come
    back inside the band can still be the entry. The default band,
    config.BACKTEST_DEFAULT_SPREAD_BAND = (0.0, 1.0), is no band at all — a
    floor of 0 is inert under every tier and no spread exceeds 1 — so the
    default reproduces the live rule. same_title pairs never read the band.

    Scanning stops at the earlier close date (not the later one) because after
    the first market closes, the pair is no longer open for entry.

    Args:
        candles_a (list[dict]): Hourly candlestick dicts for market A (sorted by ts).
        candles_b (list[dict]): Hourly candlestick dicts for market B (sorted by ts).
        mA (dict): Market A metadata dict (with "close_time", "result", etc.).
        mB (dict): Market B metadata dict (with "close_time", "result", etc.).
        pair_type (str): "time_series" or "same_title" — controls price gap threshold
            and deadline gap check.
        start_date (date): Backtest start date; no entry is recorded before this date.
        max_horizon_days (int | None): Optional cap on how far a checkpoint's Monday
            may be from the later-closing leg's close date. A Monday where
            (later close date − Monday) exceeds this is skipped (not rejected
            outright — a later Monday closer to the close dates may still
            qualify). None means no cap (default), matching live-path semantics.
        same_event_ladders (bool | None): Whether two markets of ONE event may
            be replayed as a time-series ladder, ordered and gapped on their
            stated deadlines. None (the default) resolves THIS MODULE's
            TIME_SERIES_SAME_EVENT_LADDERS (the by-value binding of
            config.TIME_SERIES_SAME_EVENT_LADDERS taken at import) at CALL
            time, so a run-level override and a monkeypatch of that name both
            take effect; patching the config attribute itself is a silent
            no-op here. Deliberately NOT triggered
            by event_ticker equality alone: _extract_pairs only proposes a
            same-event pair while the switch is on, but any other caller —
            a test, a harness, a future entry point — must not get ladder
            semantics from a pair it built itself while the switch is off.
        spread_band (tuple[float, float] | None): BACKTEST-only (floor,
            ceiling) band on the time-series spread pB − pA, dollars. None
            (the default) resolves config.BACKTEST_DEFAULT_SPREAD_BAND at call
            time — (0.0, 1.0), no band. Resolved and validated once per call,
            before anything else, so an invalid band is refused whatever the
            pair's data. Ignored by same_title pairs.

    Returns:
        Optional[dict]: A dict with keys "entry_date" (date), "pA" (float), "pB"
            (float), "nA" (float), "nB" (float), "mA" (dict), "mB" (dict),
            "gap_days" (int | None) for the first qualifying Monday — all four
            quotes of the canonicalized A and B (YES ask and NO ask of each),
            of which _leg_prices_for picks the two that were actually traded.
            "gap_days" is the loop-invariant deadline gap the price tier and
            the MAX_DEADLINE_GAP_DAYS cutoff were applied on, carried out for
            reporting so a report cannot bucket a pair under a gap it was not
            filtered by; it is None for same_title, which has no deadline-gap
            concept.
            Returns None if no qualifying Monday was found in the scan window, or
            if either leg's close_time is missing or unparseable (no scan window
            can be derived, so the pair is simply not enterable).

    Raises:
        ValueError: From config.time_series_spread_band, when spread_band
            does not unpack to exactly two values or does not satisfy
            0 <= floor < ceiling <= 1 — a caller bug, not a data condition.
        TypeError: From config.time_series_spread_band, when spread_band is
            not iterable or an element cannot be compared with a float.
    """
    # Resolve the backtest spread band once, BEFORE any data-dependent early
    # return, so a caller bug surfaces on the first call rather than only on a
    # pair that happens to have a readable close_time. config owns the default
    # and the validation (time_series_spread_band); this module only applies
    # it. band_lo feeds the time-series threshold below and band_hi the
    # per-Monday ceiling; the same_title branch reads neither.
    band_lo, band_hi = time_series_spread_band(spread_band)

    # Both markets must have a PARSEABLE close_time; without one we can't
    # determine the scan window. A malformed timestamp is treated exactly like
    # a missing one (the file-wide "can't parse it = unknown, not an error"
    # convention) rather than raising out of the caller's candidate loop.
    close_a = _parse_iso_date(mA.get("close_time"))
    close_b = _parse_iso_date(mB.get("close_time"))
    if close_a is None or close_b is None:
        return None

    # Scan up to (but not including) the day the earlier market closes —
    # after that, the pair is no longer fully open for entry.
    #
    # close_time, NOT the stated deadline, even for a DR-73 ladder, and that
    # is deliberate rather than an oversight: this bounds the window in which
    # both rungs are still OPEN and have candles, which is an exchange fact
    # about the listing. The stated deadline bounds the ORDER and the TIER,
    # which is a fact about the question. min() is order-independent, so it is
    # unaffected by any swap below.
    scan_end   = min(close_a, close_b) - timedelta(days=1)
    # Look back at most 1 year from scan_end to keep the scan window manageable
    scan_start = max(start_date, scan_end - timedelta(days=365))

    # A single-day window (scan_start == scan_end) is still a valid scan window
    if scan_start > scan_end:
        return None

    if pair_type == "time_series":
        # Live-scanner invariant: market A is the EARLIER-closing contract.
        # Never swap by price — the trade only exists when the LATER contract
        # is priced higher (checked per Monday below).
        #
        # Decide on the close DATETIMES, like scanner.find_time_series_pairs
        # sorts on m.close_time. Two contracts closing on the same UTC date at
        # different times are a valid zero-day-gap pair (live data shows 9
        # distinct close dates across 43 distinct times of day), and deciding on
        # the parsed DATES made that a tie — neither `close_b < close_a` nor its
        # mirror was true — so "market A" was left as whichever the group list
        # happened to hold first. The pair was then either dropped (the
        # direction test goes negative) or replayed with the legs inverted,
        # whose genuine in-between settlement reads as the impossible
        # A=YES/B=NO cell: booked as a premise violation, excluded from P&L and
        # dropped from the interval-discount calibration's denominator (TS-06).
        # The date comparison survives only as the fallback for a naive/aware
        # mix (reachable from a hand-edited cache, since every live timestamp is
        # tz-aware), per this file's "can't parse it = unknown, not an error"
        # convention.
        #
        # That parity is on the datetime-vs-date fix only, not on what
        # close_time MEANS: live reads the SCHEDULED close of a still-open
        # market, this reads the REALIZED close Kalshi recorded for a settled
        # one, which for an event that resolved early can sit before the
        # original schedule — a later-deadline leg that resolved early is
        # ordered first on the realized gap. Recorded, pre-existing residual
        # (CLAUDE.md TS-06); not fixed by this datetime ordering.
        # DR-73: a same-event deadline LADDER is the one time-series shape
        # close_time cannot order or measure, so it is ordered and gapped on
        # the two rungs' STATED deadlines instead. Gated on the resolved flag,
        # never on event_ticker equality alone: _extract_pairs only proposes a
        # same-event pair while the switch is on, but a caller that builds one
        # itself must not silently get ladder semantics from a switched-off
        # tree.
        #
        # The classification here IS per pair, and that is a deliberate,
        # bounded exemption from DR-70's once-per-market rule: this function
        # receives two market dicts and holds no memo to hang a per-market
        # verdict on, and it runs once per CANDIDATE PAIR rather than once per
        # pair of members — 975 ladder pairs over the whole 284-slice archive
        # at the <= 30-day cap, at ~12.3 us a classification. _extract_pairs,
        # which does face the O(N^2) enumeration, memoizes per member.
        ladders_on = (TIME_SERIES_SAME_EVENT_LADDERS if same_event_ladders is None
                      else same_event_ladders)
        event_a = mA.get("event_ticker") or ""
        if ladders_on and event_a and event_a == (mB.get("event_ticker") or ""):
            ladder = same_event_ladder(
                _stated_deadline_dict(mA, _deadline_profile_dict(mA)),
                _stated_deadline_dict(mB, _deadline_profile_dict(mB)),
            )
            # Fails CLOSED, like every other step of the ladder rule: a rung
            # whose deadline cannot be read (None) gives no order and no gap,
            # and two rungs naming ONE day (SAME_DAY) are one deadline spelled
            # two ways rather than a ladder. Branch on `is`, never truthiness
            # — SAME_DAY is a non-empty string, so `if ladder:` is true for it
            # and unpacking it raises.
            if ladder is None or ladder is SAME_DAY:
                return None
            b_before_a, gap_days = ladder
        else:
            dt_a = _parse_iso_datetime(mA.get("close_time"))
            dt_b = _parse_iso_datetime(mB.get("close_time"))
            try:
                b_before_a = dt_b < dt_a
            except TypeError:
                b_before_a = close_b < close_a
            # Deadline gap is loop-invariant: a wider gap carries more genuine
            # in-between probability mass, so 16-30 day gaps demand the larger
            # tier and gaps beyond MAX_DEADLINE_GAP_DAYS are never disputed.
            # Measure it on the close_time DATETIMES parsed above, not on the
            # dates: timedelta.days floors, while calendar-date subtraction
            # counts day boundaries, so the two disagree by up to a day
            # whenever the closes straddle midnight (2026-02-01T23:00Z vs
            # 2026-02-17T01:00Z is gap 15 live but 16 by date). That one day
            # flips both the tier boundary and the 30-day cutoff, so a
            # backtest that is supposed to replay the live strategy must use
            # the live arithmetic. The gap is order-independent (abs), which
            # is why it is computed before the swap below rather than after —
            # the swap cannot change it, and dt_a/dt_b are read nowhere else.
            try:
                # Identical arithmetic to scanner.deadline_gap_days (used by
                # find_time_series_pairs / _pair_max_sum): absolute
                # timedelta.days on tz-aware datetimes, so the result is
                # order-independent
                gap_days = abs(dt_b - dt_a).days
            except TypeError:
                # A naive/aware mix (only reachable from a hand-edited cache)
                # can't be subtracted; fall back to the dates rather than
                # raising, per this file's "can't parse it = unknown, not an
                # error" convention
                gap_days = abs(close_b - close_a).days
        if b_before_a:
            mA, mB = mB, mA
            candles_a, candles_b = candles_b, candles_a
            close_a, close_b = close_b, close_a
        if gap_days > MAX_DEADLINE_GAP_DAYS:
            return None
        # Tier the required price gap by deadline distance (15% for gaps
        # <= 15 days, 30% for 16-30 days) — the same tiering, computed off the
        # same gap arithmetic, as scanner.find_time_series_pairs — and layer
        # the backtest band's floor on top of it: config returns
        # max(tier, band_lo), so the default floor of 0 leaves the live tier
        # untouched. This one threshold drives BOTH the gap test and the
        # leg-price-sum ceiling below, which is what keeps the sum ceiling at
        # 1 - floor when the band raises the floor (the live pairing of the
        # two, applied to the raised floor rather than to the tier alone).
        threshold = min_price_diff_for_gap(gap_days, spread_min=band_lo)
    else:
        # same_title pairs have no deadline-gap concept — flat 5% threshold
        threshold = SAME_TITLE_MIN_PRICE_DIFF
        # No deadline gap to report for same_title. The time_series branch
        # above assigns gap_days on both its try and except paths, so this is
        # the only branch where the name would otherwise be unbound below.
        gap_days = None

    for ts in _monday_timestamps(scan_start, scan_end):
        entry_date = datetime.fromtimestamp(ts, tz=UTC).date()

        # Optional opt-in bet-horizon cap: skip checkpoints where the
        # later-closing leg would close further out than max_horizon_days from
        # THIS simulated checkpoint — a cheap comparison done before touching
        # candle data. Deliberately max(close_a, close_b) rather than close_b:
        # same_title has no ordering at all, and since DR-73 a same-event
        # LADDER is ordered by STATED deadline, so close_b >= close_a no longer
        # holds for every time_series pair either (in the archive 13 dated
        # same-event pairs are ordered the wrong way round by realized close,
        # and 681 have a close gap of ZERO days). The max() keeps this correct
        # under all three; close_time is read here ON PURPOSE, because this
        # bounds when a market is still OPEN, not when its deadline falls.
        if max_horizon_days is not None and (max(close_a, close_b) - entry_date).days > max_horizon_days:
            continue

        # Read the closing prices at this Monday snapshot
        ca = _candle_at_or_before(candles_a, ts)
        cb = _candle_at_or_before(candles_b, ts)
        if ca is None or cb is None:
            continue

        try:
            p_a_raw = float(ca["yes_ask_close"])
            p_b_raw = float(cb["yes_ask_close"])
            n_a_raw = float(ca["no_ask_close"])
            n_b_raw = float(cb["no_ask_close"])
        except (ValueError, TypeError):
            continue

        # Skip settled or illiquid candles (prices at the extreme ends of the
        # range). Both YES asks are checked regardless of pair type: for
        # time_series pB is not a leg price, but it still drives the gap and
        # the profit model, so a dead quote there is just as disqualifying.
        if not (0.01 <= p_a_raw <= 0.99 and 0.01 <= p_b_raw <= 0.99):
            continue

        if pair_type == "time_series":
            # A stays the earlier-closing market — no price canonicalization.
            # The anomaly is the LATER contract priced higher: gap = pB - pA.
            mA_i, mB_i = mA, mB
            pA, pB, nA, nB = p_a_raw, p_b_raw, n_a_raw, n_b_raw
            gap = pB - pA
        else:
            # same_title: canonicalize per iteration so the swap never leaks to
            # the next Monday. Market A is the more expensive side this week,
            # and the anomaly is its YES ask exceeding B's: gap = pA - pB.
            if p_a_raw >= p_b_raw:
                mA_i, mB_i = mA, mB
                pA, pB, nA, nB = p_a_raw, p_b_raw, n_a_raw, n_b_raw
            else:
                mA_i, mB_i = mB, mA
                pA, pB, nA, nB = p_b_raw, p_a_raw, n_b_raw, n_a_raw
            gap = pA - pB

        # Enforce the minimum price gap for this pair type (directional in
        # both cases — the gap above is signed, never an absolute value).
        # PRICE_EPSILON mirrors scanner.find_time_series_pairs /
        # find_same_title_pairs: candle prices are floats too, so a gap sitting
        # exactly on the tier can evaluate a hair under it and be rejected for
        # representation noise. The AST parity pins do NOT check this constant,
        # so a missed mirror here is silent divergence (TS-09).
        if gap < threshold - PRICE_EPSILON:
            continue

        # The band's CEILING (time-series only). `continue`, never
        # `return None`: a LATER Monday whose spread has come back inside the
        # band can still be the entry, exactly as a Monday under the floor
        # does not end the scan. config.time_series_spread_too_wide is the
        # one place the ceiling's PRICE_EPSILON lives (on the keep side, so
        # 0.90 - 0.30 == 0.6000000000000001 is kept at a 0.60 ceiling) — no
        # tolerance is added here. The default ceiling of 1.0 can never fire,
        # since both YES asks are banded into [0.01, 0.99] above.
        if pair_type == "time_series" and time_series_spread_too_wide(gap, spread_max=band_hi):
            continue

        # The two prices actually paid — (nA, pB) for same_title, (pA, nB) for
        # time_series — via the module's single leg mapping
        price_a, price_b = _leg_prices_for(pair_type, pA, nA, pB, nB)

        # Both traded leg prices must be live (0.01–0.99) quotes — mirrors the
        # live pipeline's (0, 1) price validation in compute_trade. This adds
        # the NO-leg check (nA for same_title, nB for time_series) to the YES
        # asks banded above. Note the candle no_ask_close is already clamped
        # into [0.01, 0.99] by historical.fetch_candlesticks, so on candle
        # data this cannot fire for the NO leg — it is parity, not a filter.
        if not (0.01 <= price_a <= 0.99 and 0.01 <= price_b <= 0.99):
            continue

        # Live orderbook-depth parity: enrich_with_orderbook_prices only keeps
        # contracts whose combined LEG price leaves the required gap
        # (price_a + price_b <= 1 - threshold) — apply the same cut to candle
        # entries. For time_series `threshold` already carries the band's
        # floor, so under a raised floor this is 1 - max(tier, floor).
        if price_a + price_b > 1.0 - threshold + PRICE_EPSILON:
            continue

        # Check that the gross spread on the leg prices exceeds the continuous fee estimate
        if (1.0 - price_a - price_b) <= fee_per_pair_approx(price_a, price_b):
            continue

        return {
            "entry_date": entry_date,
            "pA": pA, "pB": pB, "nA": nA, "nB": nB,
            "mA": mA_i, "mB": mB_i,
            # The gap the tier above was selected from, carried out so the
            # calibration report buckets each pair under exactly the gap it
            # was filtered by. None for same_title. Reporting only — every
            # decision this value drives was already made above.
            "gap_days": gap_days,
        }

    return None


# ─── Candlestick fetching ─────────────────────────────────────────────────────

def _candle_window_open(market: dict, window_open_ts: int, close_ts: int) -> int:
    """
    Pick where one market's candlestick request starts.

    The later of the backtest window's start and the market's own open_time,
    the latter floored onto the candle grid (CANDLESTICK_PERIOD_INTERVAL_MINUTES,
    so the top of the hour at hourly candles). Opening every request at the
    window's start instead asked for months of candles a market that opened
    later never had, which cost a request per 5,000 empty candles — and before
    historical.fetch_candlesticks paged a long window, it cost the whole
    series (HTTP 400, read as "no candles").

    Result-neutral by construction and by measurement. A market has no candles
    before it opens: across the DR-73 calibration corpus's 573 cached series
    whose request opened before the market did, not one candle ends at or
    before the start of the hour its market opened in. And the open is FLOORED
    because the endpoint's reading of a start_ts that falls mid-period is
    undocumented: a request that starts on the boundary gets the candle
    covering the opening hour under either reading. (58 of 1,093 cached
    series requested from an unfloored off-the-hour open_time start later than
    that hour — possibly no quote in the first hour, possibly the endpoint;
    flooring removes the doubt at the cost of one candle period.) The rest of
    the backtester already treats open_time as when the market opened:
    _can_ever_enter drops a market whose open_time leaves no entry checkpoint.

    A missing or unparseable open_time — every cache record written before
    historical._market_to_dict carried the field reads back None — keeps the
    window's start, as does an open_time that does not precede the request's
    end (a data defect; the old request is the only safe one to send), so this
    only ever moves a request's start LATER, and only for a market that
    demonstrably opened after the window began.

    Args:
        market (dict): The market dict (historical._market_to_dict shape);
            only "open_time" is read.
        window_open_ts (int): Unix timestamp of the backtest window's start
            (start_date midnight UTC), where every request used to open.
        close_ts (int): Unix timestamp the market's request ends at.

    Returns:
        int: Unix timestamp the market's candlestick request should start at:
            window_open_ts, or a later candle-period boundary.
    """
    open_dt = _parse_iso_datetime(market.get("open_time"))
    if open_dt is None:
        return window_open_ts
    period_seconds = CANDLESTICK_PERIOD_INTERVAL_MINUTES * 60
    market_open_ts = int(open_dt.timestamp())
    market_open_ts -= market_open_ts % period_seconds
    if market_open_ts >= close_ts:
        return window_open_ts
    return max(window_open_ts, market_open_ts)


def _fetch_candles_parallel(
    hist_client: Any,
    needed_tickers: dict[str, dict],
    start_date: date,
    use_cache: bool,
) -> dict[str, list[dict]]:
    """
    Fetch the hourly candlestick series for every needed ticker, in parallel.

    One fetch per ticker (one HTTP request, or more for a window longer than
    one request serves — historical.fetch_candlesticks pages it), spread
    across CANDLESTICK_FETCH_MAX_WORKERS threads. Each ticker's request runs
    from _candle_window_open — the later of start_date and the market's own
    open_time, floored to a candle boundary — to one day past its close. Parallelism is result-neutral here for three reasons: the returned
    mapping is only ever read by key (never iterated), so completion order
    cannot matter; each ticker's disk cache path is derived from its ticker, so
    two workers can never write the same file (historical._save_json_cache is an
    atomic tmp+replace, but its tmp name is derived from the destination, so a
    shared path WOULD still collide — path uniqueness stays load-bearing); and
    each fetch is
    an independent read-only GET whose retry/backoff already lives per-call
    inside api_call_with_retry.

    Markets with no close_time get an empty series without any HTTP call, which
    is what the sequential version did — there is no window to request. A
    present-but-unparseable close_time is handled the same way (with a warning):
    it is a data defect in one market, not a reason to abort the whole run from
    the main thread before any worker starts.

    Worker exceptions are deliberately NOT caught: fetch_candlesticks already
    fail-softs network errors to an empty list internally, so anything that
    still escapes is a real defect (e.g. a market that should have been
    prefiltered out) and must surface rather than be silently degraded into
    "this ticker has no prices".

    Before returning, the tickers that resolved to an empty series are counted
    and reported in ONE summary WARNING (silent at zero) — the per-run signal
    that replaces reading hundreds of individual 404 lines (TS-02).

    Args:
        hist_client (Any): Historical KalshiClient, shared across worker
            threads (the same pattern historical.py's fetch pools use).
        needed_tickers (dict[str, dict]): Ticker -> market dict, for exactly
            the markets appearing in at least one candidate pair.
        start_date (date): Start of the backtest window; the earliest any
            ticker's request starts (a market that opened later starts at its
            own open — _candle_window_open).
        use_cache (bool): Passed through to fetch_candlesticks — whether the
            per-ticker disk cache may be reused.

    Returns:
        dict[str, list[dict]]: Ticker -> candle list (keys: ts, yes_ask_close,
        no_ask_close). Every key of needed_tickers is present; the value is an
        empty list for markets with no usable (missing or unparseable) close_time.

    Raises:
        Exception: Whatever a worker's fetch_candlesticks call raises, after
            the pool has been torn down without draining its queue.
    """
    candles_by_ticker: dict[str, list[dict]] = {}

    # Start of the backtest window: the earliest any request opens. Depends
    # only on start_date, so it is computed once.
    window_open_ts = int(datetime(start_date.year, start_date.month, start_date.day,
                                  tzinfo=UTC).timestamp())

    # Split the work first: no-close_time markets resolve without any HTTP, so
    # they never occupy a worker slot.
    work: list[tuple[str, int, int]] = []
    for ticker, m in needed_tickers.items():
        close_time = m.get("close_time")
        if not close_time:
            candles_by_ticker[ticker] = []
            continue
        close_dt = _parse_iso_datetime(close_time)
        if close_dt is None:
            # A malformed timestamp yields no requestable window, exactly like a
            # missing one — resolve the ticker to an empty series instead of
            # raising on the main thread and killing a multi-hour run. Logged
            # (not silent) because an empty series is otherwise indistinguishable
            # from "this market genuinely has no price history".
            logging.warning(
                "Unparseable close_time %r for %s — fetching no candles for it",
                close_time, ticker,
            )
            candles_by_ticker[ticker] = []
            continue
        # close_ts: one day past market close to include the final candle
        close_ts = int(close_dt.timestamp()) + _DAY_SECONDS
        # Open at the market's own open when it opened after the window began:
        # it has no candles before then, so asking for them only costs requests.
        open_ts = _candle_window_open(m, window_open_ts, close_ts)
        work.append((ticker, open_ts, close_ts))

    if work:
        with ThreadPoolExecutor(max_workers=CANDLESTICK_FETCH_MAX_WORKERS) as pool:
            # Returns list[dict] with keys: ts (unix int), yes_ask_close (float),
            # no_ask_close (float) — cached per ticker, so a second run is much faster
            futures = {
                pool.submit(fetch_candlesticks, hist_client, ticker,
                            open_ts, close_ts, use_cache): ticker
                for ticker, open_ts, close_ts in work
            }
            done = 0
            try:
                for future in as_completed(futures):
                    candles_by_ticker[futures[future]] = future.result()
                    done += 1
                    if done % 50 == 0:
                        # Denominator is len(work), not len(needed_tickers): tickers
                        # with a missing/unparseable close_time never enter `work`
                        # (they're resolved to [] above without a worker), so
                        # `done` can never reach len(needed_tickers) whenever any
                        # were skipped.
                        logging.info("  Candlestick progress: %d / %d",
                                     done, len(work))
            except BaseException:
                # Same tear-down as historical.py's fetch pools: without it the
                # executor's __exit__ drains every still-queued ticker (hours of
                # work) before the error ever reaches the caller.
                pool.shutdown(wait=False, cancel_futures=True)
                raise

    # Summarize the misses ONCE. On a post-cutoff window every ticker 404s
    # (documented, and deliberately never cached), so the count is the useful
    # signal — not one warning per ticker (TS-02). Counted off the RESULT dict
    # rather than off caught exceptions, because fetch_candlesticks already
    # fail-softs a failure to [] internally; deliberately outside the `if work`
    # block so the tickers resolved to [] above for a missing or unparseable
    # close_time are counted too.
    empty = sum(1 for series in candles_by_ticker.values() if not series)
    if empty:
        logging.warning(
            "Candlestick fetch: %d of %d tickers returned no candles "
            "(post-cutoff tickers 404 by design and are never cached)",
            empty, len(candles_by_ticker),
        )

    return candles_by_ticker


# ─── Main backtest loop ───────────────────────────────────────────────────────

def _log_rss(label: str) -> None:
    """
    Log this process's peak resident set size so far, in MiB.

    Diagnostics only — nothing branches on the value. Two calls bracket the
    grouping/pairing step of _prepare_candidates (the first half of
    _prepare_entries), the phase that follows a fetch
    already hardened to stream to disk and that was nonetheless the suspected
    home of a multi-GiB peak (TS-07). Without these lines that peak is
    invisible: it lives entirely between two existing INFO lines and falls
    back to 100-300 MB immediately after. The two runs actually measured, and
    which of them the cost belonged to, are recorded in config.py beside
    BACKTEST_RECORD_BYTES_ESTIMATE — deliberately in one place, so no figure
    from one run is ever restated somewhere it will go stale.

    getrusage reports ru_maxrss in BYTES on macOS and in KILOBYTES on Linux,
    so the two are normalized here — otherwise the same line means two things
    on the two platforms this runs on (dev is macOS, CI is Linux). It is a
    high-water mark for the whole process, so it never decreases.

    Args:
        label (str): Phase name for the log line (e.g. "before grouping").

    Returns:
        None
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mib = peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024
    logging.info("Peak RSS %s: %.0f MiB", label, mib)


@dataclass
class _OutcomeLabelTally:
    """
    The running counters of one outcome-label census, fed one record at a time.

    The COUNTING half of the census. _report_outcome_label_coverage is the
    logging half, and _log_outcome_label_coverage composes the two over any
    iterable. They are separable because _prepare_candidates must count inside
    its first pass over the corpus — the census may not cost a pass of its own,
    and since SS-1 _prepare_candidates walks the corpus rather than building
    an eligible list of its own to hand over —
    while the census's log lines must still appear where they always have:
    after the "Peak RSS before grouping" line and the RAM-budget warning, which
    can only be written once both of _prepare_candidates' passes are done.

    Holds counts only, never a record, so a tally kept across the two passes
    pins nothing alive (TS-07).

    Attributes:
        total (int): Records added.
        with_subtitle (int): Records carrying a non-blank `subtitle`.
        with_event_title (int): Records carrying a non-blank `event_title`.
        phrasing (defaultdict[str, int]): Records per deadline-phrasing
            verdict (DEADLINE_CUMULATIVE / DEADLINE_SNAPSHOT /
            DEADLINE_UNKNOWN), from scanner.deadline_profile.
    """
    total: int = 0
    with_subtitle: int = 0
    with_event_title: int = 0
    phrasing: defaultdict = field(default_factory=lambda: defaultdict(int))

    def add(self, m: dict) -> None:
        """
        Count one record.

        Blank/None/absent all read as "no label", the same falsiness the two
        grouping keys apply with `or ""`.

        Args:
            m (dict): An eligible market record in the compact
                historical._market_to_dict form. Read only; no reference to it
                is kept.

        Returns:
            None
        """
        self.total += 1
        if m.get("subtitle"):
            self.with_subtitle += 1
        if m.get("event_title"):
            self.with_event_title += 1
        # Folded into whatever pass feeds this tally rather than given its own.
        # The census's own cost is this classifier, not the walk (about 12 us
        # vs 0.3 us per record after DR-70, measured on 200,000 records of the
        # 2026-09-08 day slice, 2026-09-22), so a walk of its own would add
        # little; what bounds the census on a multi-million-record corpus is
        # the classifier's cost. Since SS-1 the pass that feeds it in a
        # backtest is _index_eligible_keys, which also computes both grouping
        # keys per eligible record — and on a combo-heavy corpus the
        # time-series key dominates that pass (~135 us/record, normalize_title
        # on long combo titles, measured on the 2026-09-17 7-day window), so
        # there the census is the smaller share.
        self.phrasing[_deadline_profile_dict(m)[0]] += 1


def _report_outcome_label_coverage(tally: _OutcomeLabelTally) -> OutcomeLabelCoverage:
    """
    Log a finished outcome-label census and return exactly what it logged.

    The LOGGING half of the census: see _log_outcome_label_coverage for what
    the census measures and why it exists, and _OutcomeLabelTally for why the
    counting and the logging are separable. Every line this emits, in order,
    and the dataclass it returns are byte-for-byte what the census emitted and
    returned before SS-1 split it in two.

    Args:
        tally (_OutcomeLabelTally): The completed counts.

    Returns:
        OutcomeLabelCoverage: Scalars only (counts, fractions and one
            verdict) — the numbers this census just logged, with below_floor
            carrying the very comparison the WARNING branches on. On an empty
            corpus, total 0 with both fractions None (undefined, not zero),
            below_floor False and all three phrasing counts 0. Holds no
            reference to any record, so it is safe to keep past
            _prepare_candidates' release of the corpus and the groupable
            subset.
    """
    total = tally.total
    with_subtitle = tally.with_subtitle
    with_event_title = tally.with_event_title
    phrasing = tally.phrasing

    # An empty corpus has no coverage to report: the fraction is undefined,
    # not zero, so warning here would manufacture a drift alarm out of a corpus
    # that simply has no records — a cause the surrounding "Total settled
    # markets" and "Eligibility prefilter" lines already name. The census still
    # emits one line, so its ABSENCE always means this helper did not run.
    if not total:
        logging.info("Outcome-label coverage: no eligible markets to census")
        # Fractions are None, not 0.0: undefined rather than zero, so a
        # renderer can say "nothing to census" instead of "0% coverage". The
        # three phrasing counts are passed explicitly (DR-71: they carry no
        # default) because an empty corpus is itself a measurement — zero
        # markets of every kind — not an omission.
        return OutcomeLabelCoverage(
            total=0, with_subtitle=0, with_event_title=0,
            subtitle_fraction=None, event_title_fraction=None,
            below_floor=False,
            cumulative_markets=0, snapshot_markets=0, unknown_deadline_markets=0,
        )

    subtitle_fraction = with_subtitle / total
    logging.info(
        "Outcome-label coverage over %d eligible markets: subtitle on %d "
        "(%.2f%%), event_title on %d (%.2f%%)",
        total, with_subtitle, subtitle_fraction * 100.0,
        with_event_title, with_event_title / total * 100.0,
    )
    # ALWAYS logged, never only on a shortfall: a rule that can empty the
    # time-series strategy must not be detectable solely by the absence of a
    # warning (DR-66's lesson). "cumulative 0" on a non-empty corpus is the
    # actionable reading — it says this run can produce no time-series trade,
    # whether because the corpus genuinely holds no deadline families or
    # because the phrasing tables stopped matching.
    logging.info(
        "Deadline phrasing over %d eligible markets: %d worded as a cumulative "
        "deadline, %d snapshot, %d with no deadline wording the classifier "
        "recognises",
        total,
        phrasing[DEADLINE_CUMULATIVE],
        phrasing[DEADLINE_SNAPSHOT],
        phrasing[DEADLINE_UNKNOWN],
    )

    # Evaluated ONCE and carried out on the dataclass. The dashboard branches
    # on this verdict rather than re-deriving it from the constant, so a
    # future `<` that becomes a `<=` cannot make the page and the log fire on
    # different conditions.
    below_floor = subtitle_fraction < BACKTEST_OUTCOME_LABEL_WARN_FRACTION

    if below_floor:
        # Known residual (SS-1 Commit C): the remedy's "equivalently" clause
        # names only the LEGACY settled_markets_*.json. Since that commit the
        # assembled cache is settled_markets_*.jsonl.gz, so deleting the .json
        # alone leaves the streamed cache to be served; the primary remedy
        # (--no-cache after deleting the day slices) is unaffected. The text is
        # kept byte-identical because SS-1 must not change any log line or
        # dashboard section (dashboard._label_coverage_html renders the same
        # sentence and tests/test_dashboard.py::TestGoldenSections pins it);
        # correcting both is a follow-up that re-captures that golden.
        logging.warning(
            "Outcome-label coverage is %.2f%%, below the %.2f%% floor: most "
            "eligible markets carry no subtitle, so the time-series key falls "
            "back to the bare normalized title — the strike-blind grouping the "
            "live scanner no longer uses — and the same-title key loses its "
            "outcome discriminator. Treat this run's potential-pair counts, "
            "trades, returns and empirical interval-discount recommendation as "
            "describing a different strategy from the shipped one. Remedy: "
            "delete backtest_cache/archive_days/ and backtest_cache/live_days/, "
            "then re-run with --no-cache (equivalently, also delete the "
            "assembled backtest_cache/settled_markets_*.json); --no-cache ALONE "
            "does not refresh the day slices, which are reused unconditionally",
            subtitle_fraction * 100.0,
            BACKTEST_OUTCOME_LABEL_WARN_FRACTION * 100.0,
        )

    return OutcomeLabelCoverage(
        total=total,
        with_subtitle=with_subtitle,
        with_event_title=with_event_title,
        subtitle_fraction=subtitle_fraction,
        event_title_fraction=with_event_title / total,
        below_floor=below_floor,
        cumulative_markets=phrasing[DEADLINE_CUMULATIVE],
        snapshot_markets=phrasing[DEADLINE_SNAPSHOT],
        unknown_deadline_markets=phrasing[DEADLINE_UNKNOWN],
    )


def _log_outcome_label_coverage(markets: Iterable[dict]) -> OutcomeLabelCoverage:
    """
    Census how many eligible markets carry an outcome label and how their
    deadline wording classifies, and warn when few carry a label.

    Both backtest grouping keys are built from fields a stale cache may simply
    not have. `subtitle` is the outcome discriminator in the time-series key
    (scanner.time_series_group_key) and the third component of the same-title
    key (event_title, title, subtitle); `event_title` is the first component of
    the latter. A record whose subtitle is blank keys by title alone — the
    pre-DR-01 strike-blind grouping the live scanner was fixed to stop using —
    and a blank event_title collapses the same-title key toward (title,
    subtitle), the direction that manufactures cross-event false positives
    under the 0.95 co-resolution prior (TS-11).

    The defect this closes is the SILENCE, not the grouping (DR-66). A backtest
    over a cache written before the 2026-08-14 yes_sub_title ingest fix reports
    potential-pair counts, trade counts, a return figure and an empirical k-hat
    recommendation for the real-money constant
    TIME_SERIES_INTERVAL_PROB_DISCOUNT, all describing a strategy the shipped
    code does not implement — and neither the backtest log nor the dashboard
    said so, making such a run indistinguishable in its own output from a run
    on a good cache. The live scanner's own "Distinct normalized title+outcome
    keys" counter cannot cover this: it lives in find_time_series_pairs, which
    the backtester never calls, and it moves the OTHER way here — it detects
    the one-leg-labelled case, where keys SPLIT, whereas a wholesale-blank
    cache makes keys MERGE.

    Only subtitle coverage escalates to WARNING. event_title coverage shares
    the INFO line but is never warned on, because it is legitimately near zero
    on a healthy cache; the reasoning and the measured coverages behind both
    decisions live in config.py beside BACKTEST_OUTCOME_LABEL_WARN_FRACTION, so
    that no figure from another run is baked into a string emitted on every run
    (TS-07).

    Advisory only: this reads the records and logs. No market, group, pair or
    entry is dropped, filtered or altered, and no count the run reports moves.

    It also RETURNS what it just measured, so the same figure can reach the
    dashboard (DR-66b): a run over a label-less cache used to log the warning
    and then render a bare "Pooled empirical k̂" card with no caveat anywhere on
    the page, while backtest.py's own closing line points the operator at that
    page. Measuring and reporting in one call is a deliberate departure from
    the check_shard_coverage / _log_shard_coverage pure-plus-loud split: the
    single pass is the memory-sensitive part and must not be duplicated, and
    the threshold must be evaluated exactly once so the log line and the page
    cannot disagree about where the floor sits.

    This function is the one-shot composition of the census's two halves —
    _OutcomeLabelTally counts, _report_outcome_label_coverage logs and returns
    — for a caller holding a whole iterable. _prepare_candidates uses the two
    halves directly instead (SS-1): it feeds the tally inside its FIRST pass
    over the corpus, so the census never costs a pass of its own, and reports
    at the position this census has always logged from. Both routes run the
    same counting and the same reporting code, so they cannot disagree.

    Args:
        markets (Iterable[dict]): The eligible market records, in the compact
            historical._market_to_dict form. Walked EXACTLY ONCE, counting
            `total` as it goes rather than asking for a length, so a one-shot
            iterator serves as well as a list and no second list is ever
            materialized, since this can be millions of records.

    Returns:
        OutcomeLabelCoverage: Scalars only (counts, fractions and one
            verdict) — the numbers this census just logged, with below_floor
            carrying the very comparison the WARNING branches on. On an empty
            corpus, total 0 with both fractions None (undefined, not zero),
            below_floor False and all three phrasing counts 0. Holds no
            reference to any record (see _report_outcome_label_coverage).
    """
    tally = _OutcomeLabelTally()
    for m in markets:
        tally.add(m)
    return _report_outcome_label_coverage(tally)


@dataclass(frozen=True)
class _EligibleKeyIndex:
    """
    What _prepare_candidates' first pass over the corpus learned, compactly.

    One entry per ELIGIBLE record (a record passing _can_ever_enter), in the
    order the corpus yielded them — the positional contract the second pass,
    _materialize_groupable, reads back. Holds no record, only fixed-width
    numbers per eligible record, which is the whole point on a corpus of
    millions (SS-1). While the first pass builds it, it holds about 27 bytes
    per eligible record (three int64 arrays and two flag bytes; 187 MiB for
    7,260,952 eligible records, measured on synthetic keys on 2026-09-24),
    and computing `keep` from those buffers adds a transient of about
    74 bytes per record (np.unique inside _shared_key_mask; 515 MiB at that
    count). Once built it keeps only `keep` (1 byte) and `identities`
    (8 bytes) per eligible record.

    Attributes:
        total (int): Every record the first pass walked, eligible or not — the
            figure "Total settled markets to analyze" reports.
        eligible (int): Records that passed _can_ever_enter.
        groupable (int): Eligible records whose time-series key OR same-title
            key hash is shared with another eligible record — how many the
            second pass will materialize.
        keep (bytes): One byte per eligible record, 1 when that record is
            groupable, else 0.
        identities (array): _corpus_identity() of each eligible record, int64
            — one hash over its ticker and every field either grouping key
            reads — which the second pass compares position by position, so a
            corpus that does not re-iterate identically in anything the keep
            flags were chosen from is refused rather than silently misaligned.
    """
    total: int
    eligible: int
    groupable: int
    keep: bytes
    identities: array


def _shared_key_mask(hashes: array, valid: bytearray) -> np.ndarray:
    """
    Flag every key hash that occurs at least twice among the VALID keys.

    A record can only ever land in a group of two or more — and both grouping
    functions drop every single-member group — if some OTHER eligible record
    carries the same key. So "this record's key is shared" is exactly "this
    record can appear in a group", and a record whose time-series key and
    same-title key are both unshared can never appear in any pair of either
    type.

    Hashes stand in for the keys to keep this compact (8 bytes per record
    instead of a multi-hundred-byte string; np.unique's sort, inverse and
    counts add a transient of about 74 bytes per position while this runs —
    see _EligibleKeyIndex for the measurement), which is safe in one direction
    only, and that is the direction that matters: equal keys ALWAYS hash equal,
    so a genuinely shared key is always flagged, and a 64-bit collision
    between two DIFFERENT keys can only flag extra records as shared. Those are
    then kept and grouped on their real keys, where the exact grouping drops
    them as the single-member groups they are — so a collision costs a few
    bytes of residency, never a changed result. hash() is salted per process
    (PYTHONHASHSEED), which is harmless: both passes and the grouping run in
    one process.

    Args:
        hashes (array): int64 ('q') key hashes, one per eligible record.
            Positions whose key is invalid hold a placeholder that is never
            read.
        valid (bytearray): One byte per position: 1 when that record has a
            key of this kind at all (a non-empty time-series key; a same-title
            key that is not None), else 0.

    Returns:
        np.ndarray: Boolean, one per position — True exactly when the position
            is valid and its hash occurs at least twice among valid positions.
    """
    h = np.frombuffer(hashes, dtype=np.int64)
    ok = np.frombuffer(valid, dtype=np.bool_)
    shared = np.zeros(len(h), dtype=np.bool_)
    if ok.any():
        # counts[inverse] is, for each valid position, how many valid
        # positions carry the same hash.
        _unique, inverse, counts = np.unique(
            h[ok], return_inverse=True, return_counts=True,
        )
        shared[ok] = counts[inverse] >= 2
    return shared


def _corpus_identity(m: dict) -> int:
    """
    One hash over a record's ticker and every field either grouping key reads.

    _materialize_groupable applies the first pass's keep flags BY POSITION, so
    it must be able to tell when the second walk is not the corpus the flags
    were chosen for. A ticker alone cannot: a corpus that repeats the same
    tickers in the same order but changes a grouping field between walks (a
    cache file re-read after another run replaced it with different event
    titles, say) would have the flags applied to records whose keys are not
    the ones the first pass hashed — dropping a record whose NEW key is shared
    — with nothing raised. So the identity covers exactly what the flags
    depend on: the ticker (position) plus event_title, title and subtitle,
    which are every field _ts_group_key (through _pair_key, whose last
    fallback is the ticker) and _st_group_key read. Eligibility, the only
    other input, is re-applied by the second pass itself, and a change in it
    shifts the eligible sequence, which this comparison or the count check
    catches. No other field is compared: none can change which records are
    grouped.

    Raw values, not the `or ""`-normalized ones: a field that changes from
    None to "" did not iterate identically either, and refusing it costs
    nothing on a corpus that genuinely re-iterates. Being a 64-bit hash, a
    change could in principle slip through on a collision; the guard is
    against a corpus that drifts, not an adversarial one.

    Args:
        m (dict): A market record in the compact historical._market_to_dict
            form.

    Returns:
        int: hash((ticker, event_title, title, subtitle)) of the raw values.
    """
    return hash((m.get("ticker"), m.get("event_title"), m.get("title"), m.get("subtitle")))


def _index_eligible_keys(
    markets: Iterable[dict], start_date: date, census: _OutcomeLabelTally,
) -> _EligibleKeyIndex:
    """
    First pass over the corpus: count it, prefilter it, census it, and hash
    every eligible record's two grouping keys.

    The keys are the same _ts_group_key / _st_group_key the two grouping
    functions group on, computed once per eligible record here — the SAME
    per-record cost the time-series grouping used to pay over the whole
    eligible list (the normalize_title inside the time-series key dominates
    it), so this pass adds a walk, not a second keying. Only hashes and flags
    are kept, never a record — about 27 bytes per eligible record while the
    pass runs, plus the transient of computing the keep flags (see
    _EligibleKeyIndex for both measurements).

    Args:
        markets (Iterable[dict]): The settled-market corpus, in the compact
            historical._market_to_dict form. Walked once here and once more by
            _materialize_groupable, so it must re-iterate IDENTICALLY — a list,
            or any re-iterable that yields the same records in the same order
            on every walk (fresh dict objects each walk are fine).
        start_date (date): The backtest start date _can_ever_enter tests
            against.
        census (_OutcomeLabelTally): Fed every eligible record, in order, so
            the outcome-label census costs no pass of its own.

    Returns:
        _EligibleKeyIndex: The counts, the per-eligible-record keep flags and
            identity hashes (_corpus_identity). Holds no record.
    """
    total = 0
    identities = array("q")
    ts_hashes = array("q")
    ts_valid = bytearray()
    st_hashes = array("q")
    st_valid = bytearray()
    for m in markets:
        total += 1
        if not _can_ever_enter(m, start_date):
            continue
        census.add(m)
        identities.append(_corpus_identity(m))
        ts_key = _ts_group_key(m)
        # An empty time-series key is never grouped, so it is never "shared"
        # either; its placeholder hash is masked out by the validity flag.
        ts_hashes.append(hash(ts_key) if ts_key else 0)
        ts_valid.append(1 if ts_key else 0)
        st_key = _st_group_key(m)
        st_hashes.append(0 if st_key is None else hash(st_key))
        st_valid.append(0 if st_key is None else 1)
    keep = _shared_key_mask(ts_hashes, ts_valid) | _shared_key_mask(st_hashes, st_valid)
    return _EligibleKeyIndex(
        total=total,
        eligible=len(identities),
        groupable=int(keep.sum()),
        keep=keep.tobytes(),
        identities=identities,
    )


def _materialize_groupable(
    markets: Iterable[dict], start_date: date, index: _EligibleKeyIndex,
) -> list[dict]:
    """
    Second pass over the corpus: keep exactly the eligible records the first
    pass flagged as groupable, in corpus order.

    The keep decision is read by POSITION among eligible records from the first
    pass's index — the expensive keys are never recomputed for the millions of
    records that are dropped. Because every member of every group of two or
    more is kept, and kept in its original relative order, the two grouping
    functions return the same keys, the same members in the same order and the
    same insertion order over this subset as they would over the whole
    eligible list, and so _extract_pairs returns the same pairs.

    Position is only meaningful if the corpus re-iterates identically, so this
    fails LOUDLY rather than misalign: each eligible record's _corpus_identity
    — its ticker and every field either grouping key reads — must match the
    one the first pass recorded at that position, and the eligible count must
    match in full. That covers everything the keep flags were chosen from;
    a field no key reads is not compared, because it cannot change which
    records are grouped.

    Args:
        markets (Iterable[dict]): The same corpus _index_eligible_keys walked.
        start_date (date): The same start date, re-applied through
            _can_ever_enter so the eligible positions line up.
        index (_EligibleKeyIndex): The first pass's result.

    Returns:
        list[dict]: The groupable records — every eligible record whose
            time-series or same-title key is shared with another eligible
            record — in corpus order.

    Raises:
        RuntimeError: When the corpus did not iterate identically twice — an
            eligible record's ticker or grouping fields (event_title, title,
            subtitle) differ from the first pass's at the same position, or the
            two passes found different eligible counts. Continuing would group
            a subset chosen for a different corpus.
    """
    groupable: list[dict] = []
    keep = index.keep
    identities = index.identities
    expected = index.eligible
    position = 0
    for m in markets:
        if not _can_ever_enter(m, start_date):
            continue
        if position < expected:
            if _corpus_identity(m) != identities[position]:
                raise RuntimeError(
                    f"The settled-market corpus did not iterate identically "
                    f"twice: eligible record {position} is {m.get('ticker')!r} "
                    f"on the second pass, but the first pass recorded a "
                    f"different ticker or different grouping fields "
                    f"(event_title, title, subtitle) at that position. The "
                    f"groupable subset is chosen by position, so it cannot be "
                    f"applied to this corpus; refusing to continue rather than "
                    f"group the wrong records"
                )
            if keep[position]:
                groupable.append(m)
        position += 1
    if position != expected:
        raise RuntimeError(
            f"The settled-market corpus did not iterate identically twice: the "
            f"first pass found {expected} eligible markets and the second "
            f"{position}. The groupable subset is chosen by position, so it "
            f"cannot be applied to this corpus; refusing to continue rather "
            f"than group the wrong records"
        )
    return groupable


def _prepare_candidates(
    hist_client: Any,
    live_client,
    start_date: date,
    use_cache: bool,
    max_horizon_days: int | None,
    same_event_ladders: bool | None = None,
) -> _Candidates | None:
    """
    Run the half of the backtest that depends on neither the band nor k.

    Everything here — the Monday-feasibility pre-check, the settled-market
    fetch, the eligibility prefilter, the outcome-label census, both
    groupings, pair extraction and the candlestick fetch — is driven purely by
    which markets exist and when they traded. Neither the backtest's spread
    band (which acts only inside _find_entry's per-Monday price tests) nor the
    interval discount k (which only _simulate_at_discount reads) touches any
    of it, so one call can feed an entry pass per band through
    _entries_for_band(). This is the whole of what _prepare_entries() did
    before its _find_entry sweep: same log lines in the same order, the same
    two _log_rss brackets, and the same release of the group maps and records
    before the candlestick pool spawns.

    Since SS-1 it builds no eligible list of its own and groups only the
    groupable subset. The fetched corpus is walked TWICE and treated as any
    re-iterable of market dicts: the first pass (_index_eligible_keys) counts
    it, prefilters it, feeds the census and hashes each eligible record's two
    grouping keys; the second (_materialize_groupable) keeps only the eligible
    records whose time-series or same-title key is shared with another
    eligible record — every other record would form a single-member group,
    which both grouping functions drop. Every member of every group of two or
    more is kept, in order, so the group maps, the pairs, the census and every
    number the run reports are exactly what grouping the whole eligible list
    produced. One INFO line ("Groupable subset: ...") is new, and the
    RAM-budget warning now counts the groupable records it describes rather
    than every eligible one.

    The corpus itself is not held either, since SS-1's Commit C:
    fetch_all_settled_markets returns a historical.SettledCorpus that streams
    the assembled settled_markets_*.jsonl.gz cache off disk on every walk
    (fresh dicts each time), so neither walk ever has more than a record in
    hand besides what it keeps. The one exception is a hit on a LEGACY
    settled_markets_*.json cache, which is still read whole and handed over as
    one list: because the prefilter was applied during its assembly that list
    IS the eligible set, resident through both walks — and counted by the
    "Peak RSS before grouping" line — until it is released right after the
    second pass. Both walks are written for any corpus that re-iterates
    identically, which is why neither form needed a change here.

    Args:
        hist_client (Any): Signed client for the historical archive/live endpoints.
        live_client: Client passed through to fetch_all_settled_markets.
        start_date (date): Earliest settlement date to include.
        use_cache (bool): Whether to reuse the disk-cached assembled market list.
        max_horizon_days (int | None): Optional opt-in bet-horizon cap. Not
            applied here — no pair is priced here — but carried on the result
            so every entry pass applies the cap the caller asked for. None
            applies no cap.
        same_event_ladders (bool | None): Whether two dated cumulative rungs
            of ONE event may pair (DR-73). None (the default) resolves this
            module's TIME_SERIES_SAME_EVENT_LADDERS (bound from config at
            import) at call time; patching config itself is a silent no-op —
            see _extract_pairs' own entry. Handed verbatim to BOTH
            _extract_pairs() calls and stored UNRESOLVED on the result, where
            _entries_for_band() reads it for every _find_entry() call. That is
            load-bearing: the two must agree, or a pair this function proposes
            is replayed under the other rule's ordering.

    Returns:
        _Candidates | None: The candidate pairs (in scan order: time-series,
            then same-title), their candle series, this run's
            OutcomeLabelCoverage, and the start date / horizon / ladder flag
            every entry pass must reuse. None — the codebase's
            return-None-on-validation-failure convention — when the Monday
            feasibility pre-check fails, a "no simulation is possible in this
            window at all" signal distinct from "no pair was ever tradeable";
            the fetch never ran on that path, so no census exists either.

    Raises:
        KeyError: Propagates out of the candlestick-fetch pool
            (_fetch_candles_parallel) if a ticker needed by a candidate pair
            was not properly excluded by the eligibility prefilter — this is
            treated as a real defect (a market that should never have reached
            this stage), not degraded into "no price history".
        RuntimeError: Propagates out of _materialize_groupable when the
            corpus did not iterate identically on its two walks (a different
            eligible count, or a different ticker at some eligible position):
            the groupable subset is chosen by position, so continuing would
            group records chosen for a different corpus. Also propagates as
            historical.SettledCorpusError (a RuntimeError subclass) out of
            either walk over a streamed corpus whose cache file cannot be
            read, or whose complete walk yields a different record count.
    """

    # Feasibility pre-check, BEFORE any network call: a trade can only ever be
    # entered at a Monday 09:00 UTC checkpoint. If [start_date, today] contains
    # no Monday at all, no trade can ever be entered by construction, no matter
    # what the fetch returns — this is exactly the class of run that burned ~59
    # minutes fetching 9.2M records into a 2-byte assembled cache (0 markets
    # survived the Monday-eligibility prefilter). Detecting it up front skips
    # the fetch entirely instead of discovering it only after paying for it.
    #
    # The window end is today, NOT yesterday. _find_entry()'s per-pair scan
    # ends at (min(close_a, close_b) - 1 day), but a market that settled early
    # can still carry a close_time in the future, which makes TODAY a
    # legitimate checkpoint for that pair. This guard exists only to catch the
    # structurally-impossible case, so it must be strictly conservative: an
    # over-tight end date would wrongly skip a real run (e.g. today is Monday
    # and start_date is within the last week).
    # UTC, not local. _monday_timestamps builds 09:00 UTC checkpoints, and
    # _build_equity_curve already uses datetime.now(UTC).date() with a comment
    # explaining why local is wrong there — this is the same reasoning, and it
    # was the only date.today() left in the fetch path. West of UTC the local
    # date lags for the first hours of each UTC day (7 of every 24 on a PDT
    # host), so a window whose ONLY Monday is the current UTC day short-
    # circuits to zero trades and reports the run as structurally impossible
    # when it is not (TS-13).
    feasibility_end = datetime.now(UTC).date()
    if not _monday_timestamps(start_date, feasibility_end):
        logging.warning(
            "No Monday 09:00 UTC entry checkpoint exists in [%s, %s] — no "
            "trade can ever be entered; skipping the fetch entirely",
            start_date, feasibility_end,
        )
        # None rather than an empty _Candidates so the caller can tell "no
        # simulation is possible in this window" apart from "nothing was ever
        # tradeable". run_backtest (through _prepare_entries) and
        # run_backtest_sweep each turn it into the same empty-result shape the
        # zero-trade path already produces, so backtest.py / generate_dashboard
        # need no changes to handle this early-exit.
        #
        # No _Candidates at all, so no census either: the fetch never ran, so
        # no corpus was ever censused. _prepare_entries turns this into its
        # (None, None) pair and run_backtest_sweep into label_coverage=None,
        # which reads on the page as "not measured" — the truth, and distinct
        # from a corpus that WAS censused and held zero records.
        return None

    # Fetch all settled markets from start_date onward (uses disk cache if
    # available). The eligibility predicate below is handed to the fetch so
    # ineligible markets are dropped during assembly rather than materialized
    # and cached first — result-neutral, since both passes below would
    # discard exactly those records anyway, but it keeps peak memory and the
    # assembled cache proportional to what the backtest can actually use.
    # SETTLED_PREFILTER_CACHE_TAG keys that cache to _can_ever_enter's current
    # semantics and MUST be bumped if this predicate changes.
    markets = fetch_all_settled_markets(
        hist_client, live_client, start_date, use_cache,
        prefilter=lambda m: _can_ever_enter(m, start_date),
        prefilter_tag=SETTLED_PREFILTER_CACHE_TAG,
    )
    # Two walks over the corpus, and no eligible list of this function's own
    # (SS-1). The corpus is treated as any RE-ITERABLE of market dicts: it is
    # walked here and once more by _materialize_groupable, and nothing else in
    # this function iterates it. The fetch returns a historical.SettledCorpus,
    # which streams the assembled cache file on each walk (so the corpus is
    # never resident), or — on a hit on a LEGACY .json cache only — one list,
    # which since the prefilter ran during its assembly is the eligible set,
    # resident until the `del markets` below.
    #
    # Pass 1 counts every record (the "Total settled markets" figure), applies
    # the eligibility prefilter below, feeds each eligible record to the
    # outcome-label census, and hashes that record's two grouping keys — about
    # 27 bytes per eligible record, never the record itself (plus a transient
    # while the keep flags are computed; _EligibleKeyIndex has both measured).
    # A 7-day window (--start-date 2026-09-17) measured 7,260,952 eligible
    # records of which only 184,255 (2.5%) share either key with another
    # eligible record; both grouping functions drop every single-member group,
    # so the rest can never appear in any pair of either type, yet holding
    # them all at the 3,926 B/record measured on that window is ~28 GB — past
    # a 16 GB host. Pass 2 then keeps exactly the records whose key is shared.
    #
    # Necessary-condition prefilter, applied in BOTH passes: drop markets
    # whose [open_time, close_time - 1 day] window contains no Monday
    # checkpoint on/after start_date, since _find_entry() can then never enter
    # them as either leg of either pair type. This is what makes
    # grouping/pairing tractable at current Kalshi volumes (hourly/intraday
    # ladders are the overwhelming majority of settled markets and almost
    # never span a scannable Monday).
    #
    # Retained even though the same predicate was passed into the fetch above:
    # it is idempotent, it costs nothing extra (both passes walk the corpus
    # anyway), and it keeps this guarantee local to the code that depends on
    # it (a cached unfiltered list, a caller that skips the prefilter argument,
    # or a future fetch path would otherwise reach the O(n^2) pairing
    # unfiltered).
    census = _OutcomeLabelTally()
    key_index = _index_eligible_keys(markets, start_date, census)
    logging.info("Total settled markets to analyze: %d", key_index.total)
    logging.info(
        "Eligibility prefilter: skipping %d/%d markets that cannot appear in any tradeable pair",
        key_index.total - key_index.eligible, key_index.total,
    )

    # Pass 2: materialize the groupable subset, in corpus order. Raises if the
    # corpus did not iterate identically twice, which would misalign the
    # positional keep flags.
    groupable = _materialize_groupable(markets, start_date, key_index)
    eligible_count = key_index.eligible
    logging.info(
        "Groupable subset: materializing %d of %d eligible markets — the rest "
        "share no grouping key with any other eligible market, so both "
        "groupings would drop them as single-member groups",
        len(groupable), eligible_count,
    )
    # Nothing below reads the corpus or the first pass's index: `groupable`
    # holds every record grouping needs, and the census has its counts. For a
    # corpus held as a list (a legacy .json cache hit, or a test stub) this is
    # what lets every eligible record that shares no key be collected BEFORE
    # the group maps are built, rather than after pair extraction; for a
    # streamed SettledCorpus it only drops a small handle.
    del markets, key_index

    # Logged BEFORE the RAM-budget warning below so the two read in causal
    # order: this line is what the fetch — or, on a cache hit, the cache load —
    # and the two passes have ALREADY cost, the groupable subset included, and
    # the warning that follows names that subset's share of it and what
    # grouping is about to add on top.
    _log_rss("before grouping")

    # Everything from here to the end of pair extraction is held live at once:
    # the groupable subset, two group maps over it, and two candidate-pair
    # lists referencing those same dicts. The subset is ALREADY resident when
    # this fires — the second pass has just materialized it — so this is a
    # budget line covering money already spent plus money about to be spent,
    # not a forecast issued ahead of the whole cost. It is keyed on the
    # GROUPABLE count, not the eligible one, because the subset is what stays
    # resident from here on: the corpus was released just above.
    #
    # Known residual, recorded rather than implied away: when the corpus is a
    # list — only on a hit on a LEGACY settled_markets_*.json cache since
    # SS-1's Commit C; a fetched or streamed-cache corpus is never resident —
    # every eligible record WAS resident up to that release, and the peak RSS
    # line above includes all of them, yet this warning does not count them.
    # So such a run whose eligible count is far above the threshold but whose
    # groupable count is not (the 7-day window above: 7,260,952 eligible,
    # 184,255 groupable) gets no warning for the list that set its peak; its
    # eligible count is still on the "Eligibility prefilter" and "Groupable
    # subset" lines, and the cost on the RSS line. It deliberately carries only THIS
    # run's numbers; the historical measurements live in config.py beside
    # BACKTEST_RECORD_BYTES_ESTIMATE, where a reader is prompted to keep them
    # current, rather than in a string emitted on every run (TS-07). Advisory
    # only: nothing is capped or dropped.
    if len(groupable) > BACKTEST_MARKETS_RAM_WARN:
        logging.warning(
            "%d groupable markets (of %d eligible) are materialized for "
            "grouping: their records alone are roughly %.1f GB and are already "
            "resident — the peak RSS line above covers them; grouping and pair "
            "extraction add the group maps and the pair lists on top of them",
            len(groupable), eligible_count,
            len(groupable) * BACKTEST_RECORD_BYTES_ESTIMATE / 1e9,
        )

    # Report the census of the two fields the grouping keys below are built
    # from. It was COUNTED over every eligible record during pass 1 and is
    # only logged here, at the position it has always logged from. A cache
    # predating the 2026-08-14 yes_sub_title ingest fix carries subtitle=None
    # on nearly every record, which makes the time-series key collapse to the
    # pre-DR-01 strike-blind title-only form — silently, with the run's pair
    # counts, trades, return and empirical k-hat all still reported as if it
    # had grouped correctly (DR-66). Advisory: it logs and changes nothing.
    #
    # The measurement is carried out of this function (DR-66b) so the dashboard
    # can render the same caveat beside the k-hat card it recommends a
    # real-money constant from. It is scalars only (counts, fractions and one
    # verdict) with no reference to any record here, so holding it costs
    # nothing and the release of the subset below is unaffected.
    label_coverage = _report_outcome_label_coverage(census)

    # Group the groupable subset into potential pairs using the same logic as
    # the live scanner. Every member of every group of two or more is in the
    # subset, in its original relative order, so these maps — keys, members,
    # member order and insertion order — are exactly the maps the whole
    # eligible list would have produced.
    ts_groups    = _group_by_normalized_title(groupable)
    same_groups  = _group_by_exact_title(groupable)
    # The ladder flag rides through unresolved (None included): the two calls
    # here and every _find_entry call of every later entry pass (it is carried
    # on the returned _Candidates) receive the same argument and resolve the
    # same module constant at their own call time — see _Candidates for why
    # that holds only while the name is not rebound between the halves. The
    # same-title call takes it too, for signature uniformity — 3-tuple-keyed
    # groups have no deadline concept and the flag is inert there.
    ts_pairs     = _extract_pairs(ts_groups, same_event_ladders=same_event_ladders)
    same_pairs   = _extract_pairs(same_groups, same_event_ladders=same_event_ladders)
    # Release the group maps AND the groupable subset together, before the
    # candlestick pool spawns CANDLESTICK_FETCH_MAX_WORKERS threads rather than
    # at function exit. Nothing below reads any of them — the pair lists carry
    # the market dicts they need. Dropping the group maps alone frees no record
    # dicts at all: `groupable` still references every one of them, so all
    # three names have to go for the groupable records that landed in no
    # candidate pair to become collectable. (The corpus itself was released
    # right after the second pass, above.)
    #
    # This lowers RESIDENCY across the candlestick fetch and every later
    # _find_entry sweep (_entries_for_band). It does NOT lower the run's peak
    # RSS, which is a high-water mark already reached by the time this
    # statement runs.
    del ts_groups, same_groups, groupable
    _log_rss("after pair extraction")

    logging.info("Potential pairs: %d time-series, %d same-title", len(ts_pairs), len(same_pairs))

    # Collect only the tickers that actually appear in a potential pair to avoid
    # fetching candlesticks for thousands of unrelated markets
    needed_tickers: dict[str, dict] = {}
    for mA, mB, _, _ in ts_pairs + same_pairs:
        needed_tickers[mA["ticker"]] = mA
        needed_tickers[mB["ticker"]] = mB

    logging.info("Fetching candlesticks for %d markets (cached per ticker)...", len(needed_tickers))

    # Fetch hourly candlestick price series for all needed tickers, in parallel
    # across tickers — one independent read-only GET each, cached per ticker so
    # a second run is much faster. Sequentially this loop dominated the whole
    # backtest (~4.3 tickers/sec live-measured).
    candles_by_ticker = _fetch_candles_parallel(
        hist_client, needed_tickers, start_date, use_cache
    )

    logging.info("Candlestick fetch complete.")

    # Combine both pair types in scan order — time-series first, then
    # same-title — which is the order every entry pass walks and therefore the
    # order of _prepare_entries' output.
    all_pairs = [(p, "time_series") for p in ts_pairs] + [(p, "same_title") for p in same_pairs]

    # The ladder flag is stored exactly as received (None included): every
    # entry pass hands _find_entry the same argument _extract_pairs was handed
    # above, which resolves the same module name at call time as long as it is
    # not rebound between the halves (DR-73c; see _Candidates).
    return _Candidates(
        all_pairs=all_pairs,
        candles_by_ticker=candles_by_ticker,
        label_coverage=label_coverage,
        start_date=start_date,
        max_horizon_days=max_horizon_days,
        same_event_ladders=same_event_ladders,
    )


# The two pair-type labels an entry pass can be restricted to. Their order is
# irrelevant to the result: _entries_for_band always walks all_pairs in scan
# order and only FILTERS on this set.
_PAIR_TYPES = ("time_series", "same_title")


def _exact_label(value: float, spec: str) -> str:
    """
    Format a float with a short spec, falling back to repr when that is lossy.

    The completion lines of one sweep must have unique prefixes (TS-21), and
    a prefix names the k and the band — so two DIFFERENT values must never
    print alike. A fixed spec alone cannot promise that: "%g" prints both 0.3
    and 0.3000001 as "0.3", and "%.3f" prints both 0.75 and 0.7500001 as
    "0.750", so an off-grid override within a rounding of a grid member (or
    one carrying float noise, 0.1 + 0.2) would repeat that member's prefix at
    every point it shares. The short form is kept whenever it reads back as
    exactly the value — every grid member does, so the familiar "0.3-0.6" and
    "k=0.750" are unchanged — and repr, the shortest string that round-trips,
    is used otherwise. The mapping is therefore injective: two strings are
    equal only if the floats they came from are.

    Args:
        value (float): The number to render.
        spec (str): The preferred format spec, e.g. "g" or ".3f".

    Returns:
        str: format(value, spec) when float() of it equals value, else
            repr(value).
    """
    short = format(value, spec)
    return short if float(short) == value else repr(value)


def _band_label(band: tuple[float, float]) -> str:
    """
    Render a resolved spread band as the "floor-ceiling" text every log line uses.

    One definition, so the Phase-1 announcement of a band sweep and the
    completion line of every simulation at that band spell it identically —
    a reader can match them by text. Each bound is formatted with :g, which
    drops trailing zeros, so the default band reads "0-1" and the plan's
    30-60% band "0.3-0.6" — unless :g would lose precision, when the bound is
    printed exactly (_exact_label), so a primary band of (0.3000001, 0.6) is
    never announced or labelled as the grid band "0.3-0.6".

    Args:
        band (tuple[float, float]): A (floor, ceiling) already resolved by
            config.time_series_spread_band.

    Returns:
        str: "<floor>-<ceiling>" — distinct for every distinct band.
    """
    lo, hi = band
    return f"{_exact_label(lo, 'g')}-{_exact_label(hi, 'g')}"


def _is_ladder_pair(pair_type: str, mA: dict, mB: dict) -> bool:
    """
    Report whether a pair is a same-event deadline ladder (DR-73).

    A same-event ladder is a time-series pair whose two legs share one
    NON-EMPTY event ticker — the only same-event pair _extract_pairs ever
    proposes, and only while the ladder switch is on. A missing ticker on
    both legs ("" == "") must not read as one event: an unknown event cannot
    be shown to be one event, so emptiness fails the test. The single
    definition behind both BacktestTrade.same_event_ladder and a band sweep's
    "ladder"/"cross" populations, so a trade's label and the population it was
    simulated in can never disagree. Reporting only — nothing prices, sizes or
    settles on it.

    Args:
        pair_type (str): "time_series" or "same_title".
        mA (dict): Market A's record (after _find_entry's canonicalization).
        mB (dict): Market B's record.

    Returns:
        bool: True for a time-series pair whose legs share one non-empty
            event ticker; False otherwise, including every same-title pair.
    """
    event_a = mA.get("event_ticker") or ""
    return (pair_type == "time_series" and bool(event_a)
            and event_a == (mB.get("event_ticker") or ""))


def _entries_for_band(
    candidates: _Candidates,
    spread_band: tuple[float, float] | None = None,
    *,
    pair_types: tuple[str, ...] = _PAIR_TYPES,
    _pairs: list | None = None,
) -> list[dict]:
    """
    Locate each candidate pair's first tradeable Monday under one spread band.

    The Pass-1a sweep: one _find_entry() call per pair in candidates.all_pairs
    (or in _pairs, when given) whose type is in pair_types, in scan order.
    _find_entry applies price,
    deadline and band thresholds only — it holds no probability model — so
    the result is identical at every interval discount; only the band can
    change it, and only for time-series pairs (a same-title pair never reads
    the band). A caller sweeping many bands can therefore compute the
    same-title entries once (pair_types=("same_title",)) and the time-series
    entries once per band (pair_types=("time_series",)); concatenating the two
    in that order reproduces the default call exactly, since the default
    walks every time-series pair before every same-title one.

    The start date, the bet-horizon cap and the ladder flag are read FROM
    candidates, never taken as arguments: they must be the values pair
    extraction and the candle fetch used, and a second copy passed here could
    disagree with them (for the ladder flag, that is the DR-73c inversion —
    a ladder admitted on stated deadlines and then entered on close_time).
    The flag is carried UNRESOLVED, so this pass hands _find_entry the same
    ARGUMENT _extract_pairs was handed; when that argument is None each of
    them resolves this module's TIME_SERIES_SAME_EVENT_LADDERS at its own
    call time, and the two agree only if that name is not rebound between
    _prepare_candidates and this pass (see _Candidates).

    Both arguments are validated up front, before any pair is scanned — an
    empty or unknown pair_types here, an invalid spread_band through
    config.time_series_spread_band — so a pass that happens to scan no pair
    still refuses an argument that could never apply rather than returning []
    for it.

    Args:
        candidates (_Candidates): _prepare_candidates() output.
        spread_band (tuple[float, float] | None): BACKTEST-only (floor,
            ceiling) band on the time-series spread pB − pA, handed verbatim
            to every _find_entry() call. None (the default) resolves
            config.BACKTEST_DEFAULT_SPREAD_BAND there — (0.0, 1.0), no band,
            i.e. the live rule.
        pair_types (tuple[str, ...]): Which pair types to scan — any subset
            of ("time_series", "same_title"). Keyword-only. Defaults to both.
        _pairs (list | None): PRIVATE, keyword-only. The
            [((mA, mB, canon, group_key), pair_type), ...] items to scan in
            place of candidates.all_pairs — in practice a subsequence of it,
            in its order, so the result keeps the full scan's order. None (the
            default, and what every caller but _sweep_from_candidates passes)
            scans candidates.all_pairs. It only ever NARROWS which pairs are
            looked at; the caller owns the proof that no pair it leaves out
            could have produced an entry at this band (see
            _sweep_from_candidates' no-band pre-pass). candidates.all_pairs
            itself is never mutated.

    Returns:
        list[dict]: One record per pair that produced an entry, in scan order,
            each shaped {"pair_type": str, "canon": str, "group_key": object,
            "entry": dict} where "entry" is _find_entry()'s return dict (which
            already carries the possibly-swapped mA/mB). Empty when no pair of
            the requested types was ever tradeable under this band.

    Raises:
        ValueError: If pair_types is empty or names anything other than
            "time_series" or "same_title" (including a bare string, whose
            characters are not pair types) — either would silently scan
            nothing and read as a band with no tradeable pair. Also, from
            config.time_series_spread_band, if spread_band does not unpack to
            exactly two values or does not satisfy 0 <= floor < ceiling <= 1.
        TypeError: From config.time_series_spread_band, if spread_band is not
            iterable or an element cannot be compared with a float.
    """
    if not pair_types or set(pair_types) - set(_PAIR_TYPES):
        raise ValueError(
            f"pair_types must be a non-empty selection from {_PAIR_TYPES}, "
            f"got {pair_types!r}"
        )
    # Validation only — the resolved value is discarded and spread_band is
    # handed to _find_entry verbatim, which resolves it again per call. config
    # owns both the default and the rule (time_series_spread_band).
    time_series_spread_band(spread_band)

    raw_entries: list[dict] = []
    scan = candidates.all_pairs if _pairs is None else _pairs
    for (mA_orig, mB_orig, canon, group_key), pair_type in scan:
        if pair_type not in pair_types:
            continue
        candles_a = candidates.candles_by_ticker.get(mA_orig["ticker"], [])
        candles_b = candidates.candles_by_ticker.get(mB_orig["ticker"], [])

        # Find the first Monday where this pair was tradeable at the threshold
        # prices — max_horizon_days (if set) restricts entries to checkpoints
        # close enough to the legs' close dates, and spread_band narrows the
        # time-series spread rule for this pass only
        entry = _find_entry(
            candles_a, candles_b, mA_orig, mB_orig, pair_type,
            candidates.start_date,
            max_horizon_days=candidates.max_horizon_days,
            # Same unresolved flag _extract_pairs was handed: a same-event
            # pair it proposed must be ordered and tiered by the same rule
            # that admitted it (DR-73).
            same_event_ladders=candidates.same_event_ladders,
            spread_band=spread_band,
        )
        if entry is None:
            continue

        # Carry the group identity alongside the entry: _simulate_at_discount
        # needs canon/group_key for the one-pair-per-group dedup, while mA/mB
        # already ride inside the entry dict.
        raw_entries.append({
            "pair_type": pair_type,
            "canon": canon,
            "group_key": group_key,
            "entry": entry,
        })
    return raw_entries


def _prepare_entries(
    hist_client: Any,
    live_client,
    start_date: date,
    use_cache: bool,
    max_horizon_days: int | None,
    same_event_ladders: bool | None = None,
) -> tuple[list[dict] | None, OutcomeLabelCoverage | None]:
    """
    Run the half of the backtest that does not depend on the interval discount.

    Everything here — the Monday-feasibility pre-check, the settled-market
    fetch, the eligibility prefilter, both groupings, pair extraction, the
    candlestick fetch and the _find_entry sweep — is driven purely by prices,
    dates and thresholds. _find_entry applies no probability model at all, so
    none of this changes when the time-series interval discount k changes.
    Separating it out lets _simulate_at_discount() be re-run at many discounts
    over one expensive, network-bound preparation pass. run_backtest() is its
    one production caller: run_backtest_sweep() composes _prepare_candidates()
    with _sweep_from_candidates(), which runs the _entries_for_band() passes
    itself, because a band sweep needs one entry pass per band where this
    runs exactly one.

    It is the composition of the two halves split at the spread band:
    _prepare_candidates() (everything through the candlestick fetch) and one
    _entries_for_band() pass at the DEFAULT band — config's
    BACKTEST_DEFAULT_SPREAD_BAND, (0.0, 1.0), which is no band at all — so it
    produces exactly the entries it produced before the band existed
    (pinned against values captured from the pre-split code by
    tests/test_backtester.py::TestPrepareEntriesGolden).

    Args:
        hist_client (Any): Signed client for the historical archive/live endpoints.
        live_client: Client passed through to fetch_all_settled_markets.
        start_date (date): Earliest settlement date to include.
        use_cache (bool): Whether to reuse the disk-cached assembled market list.
        max_horizon_days (int | None): Optional opt-in bet-horizon cap mirroring
            scanner.filter_markets_within_horizon on the live path, but relative
            to each simulated checkpoint rather than real-world now: at a given
            Monday checkpoint, a pair can only enter if the later-closing leg
            closes within max_horizon_days of THAT checkpoint. None applies no
            cap. Passed straight through to _find_entry() for each pair.
        same_event_ladders (bool | None): Whether two dated cumulative rungs
            of ONE event may pair (DR-73). None (the default) resolves this
            module's TIME_SERIES_SAME_EVENT_LADDERS (bound from config at
            import) at call time; patching config itself is a silent no-op —
            see _extract_pairs' own entry. Handed
            verbatim to _prepare_candidates(), which gives it to BOTH
            _extract_pairs() calls and carries it, unresolved, to every
            _find_entry() call of the entry pass — load-bearing: the two must
            agree, or a pair this function proposes is replayed under the
            other rule's ordering.

    Returns:
        tuple[list[dict] | None, OutcomeLabelCoverage | None]: The prepared
            entries and this run's outcome-label census.

            Element 0 is one record per pair that produced an entry, in scan
            order (time-series pairs first, then same-title), each shaped
            {"pair_type": str, "canon": str, "group_key": object, "entry": dict}
            where "entry" is _find_entry()'s return dict (which already carries
            the possibly-swapped mA/mB). An empty list means no pair was ever
            tradeable. It is None — the codebase's
            return-None-on-validation-failure convention — when the Monday
            feasibility pre-check fails, a "no simulation is possible in this
            window at all" signal distinct from "nothing entered". NOTE that
            the sentinel now lives on element 0: a caller that forgets to
            unpack holds a 2-tuple, which is never None, so its
            `if raw_entries is None` guard would silently go false.

            Element 1 is the OutcomeLabelCoverage the census measured over the
            eligible-market corpus — carried out so the dashboard can render
            the same caveat the log warns about (DR-66b) — and is None on
            exactly the feasibility-short-circuit path, where the fetch never
            ran and there was no corpus to census. That is distinct from a
            censused corpus of zero records, which carries total=0.

    Raises:
        KeyError: Propagates out of the candlestick-fetch pool
            (_fetch_candles_parallel) if a ticker needed by a candidate pair
            was not properly excluded by the eligibility prefilter — this is
            treated as a real defect (a market that should never have reached
            this stage), not degraded into "no price history".
    """
    # Everything through the candlestick fetch. None means the feasibility
    # pre-check failed and nothing was fetched — so no census either.
    candidates = _prepare_candidates(
        hist_client, live_client, start_date, use_cache, max_horizon_days,
        same_event_ladders=same_event_ladders,
    )
    if candidates is None:
        return None, None

    # ── Pass 1a: locate each pair's first tradeable Monday (k-independent) ──
    # _find_entry applies price and deadline thresholds only — it holds no
    # probability model — so this sweep yields identical entries at every
    # interval discount and is run exactly once, ahead of any sizing. At the
    # default spread band (spread_band=None, which _find_entry resolves to
    # config.BACKTEST_DEFAULT_SPREAD_BAND, i.e. no band) it is the live rule.
    raw_entries = _entries_for_band(candidates, spread_band=None)

    logging.info("Prepared %d candidate entries for sizing", len(raw_entries))
    return raw_entries, candidates.label_coverage


def _simulate_at_discount(
    raw_entries: list[dict],
    start_date: date,
    initial_balance: float,
    k: float | None = None,
    spread_band: tuple[float, float] | None = None,
    population: str = "all",
) -> SweepPoint:
    """
    Size, select and settle prepared entries at one interval discount.

    This is the half of the backtest that depends on k, the time-series
    interval discount: the Kelly gate, the settlement-outcome and
    cumulative-deadline-premise checks, the one-pair-per-group dedup, the
    cross-type dedup, and the chronological cash-constrained Pass 2 that
    produces the trades and the equity curve. Same-title candidates are
    unaffected by k — they price on the fixed co-resolution prior — but they
    still share the cash and the ticker-conflict filter with the time-series
    ones, so the whole selection has to be replayed per discount rather than
    merely re-scored.

    Statement order inside the candidate loop is load-bearing and is preserved
    exactly as it stood before this function was extracted: the Kelly gate runs
    BEFORE the outcome-validity and premise-violation checks, so
    premise_violations counts only candidates that already passed Kelly — which
    makes that count itself k-dependent. Likewise the one-pair-per-group dedup
    runs after the Kelly gate, so a different k can change which candidate wins
    its group. Both are intended; do not reorder or hoist them.

    spread_band and population change NOTHING about the simulation: the band
    has already acted by the time entries reach here (inside _find_entry,
    through _entries_for_band), and the population is whichever subset of
    entries the caller chose to hand over. Both only name the run on its
    completion line — so each of a band sweep's thousands of simulations is
    distinguishable in the log (TS-21) — and stamp the returned point.

    Args:
        raw_entries (list[dict]): Prepared entries — _prepare_entries()
            output, or (inside a sweep) one band's _entries_for_band() output
            or a subset of it — one record per pair that produced an entry.
        start_date (date): First trading date of the window; the equity curve
            _build_equity_curve returns opens one row earlier than this.
        initial_balance (float): Simulated starting cash balance in dollars.
        k (float | None): Interval-discount override in [0, 1], handed to
            config.time_series_profit_prob for every time-series candidate.
            None (default) means "no override", which that helper resolves at
            call time to config.TIME_SERIES_INTERVAL_PROB_DISCOUNT — the value
            the live sizer reads — so the default path prices exactly as it
            always has.
        spread_band (tuple[float, float] | None): The spread band the entries
            were detected under — a label, never applied here. None (default)
            renders on the completion line as the resolved default band
            (config.time_series_spread_band(None)) and is stamped as None.
        population (str): Which entries these are, for the completion line
            and the stamp: one of "all" (default), "time_series", "ladder",
            "cross", "same_title", or the run labels "all/H1", "all/H2",
            "all/ex-top", "time_series/H1", "time_series/H2" and
            "time_series/ex-top" a band sweep gives its split-half and
            excluding-top-event runs.

    Returns:
        SweepPoint: The trades (in entry-date order, empty if none entered) and
            the daily equity curve produced at this discount, stamped with the
            RESOLVED k — never None — the resolved spread_band (None when None
            was passed) and the population.

    Raises:
        ValueError: If population is not one of the labels above (a typo would
            otherwise mislabel a scenario silently), or, from
            config.time_series_spread_band, if spread_band is not a valid
            band. Both are caller bugs, checked before any entry is scored.
        TypeError: From config.time_series_spread_band, for a band that is not
            a pair of numbers.
    """
    if population not in _SIMULATION_LABELS:
        raise ValueError(
            f"population must be one of {_SIMULATION_LABELS}, got {population!r}"
        )
    # The band this run is LABELLED with. config.time_series_spread_band owns
    # the default and the validation; resolving here (once, before the loop)
    # also means a bad band fails before any work rather than after it.
    band_lo, band_hi = time_series_spread_band(spread_band)
    # The discount actually in force, recorded on the result so no caller has
    # to re-derive it from the None sentinel.
    effective_k = TIME_SERIES_INTERVAL_PROB_DISCOUNT if k is None else k

    # ── Pass 1b: score the prepared entries and keep the tradeable ones ──
    candidates = []
    # Time-series candidates that settled earlier-YES/later-NO — impossible
    # for a cumulative-deadline pair, so the grouping admitted a
    # non-cumulative one. Counted here, reported once after the loop.
    #
    # This counter is k-DEPENDENT by construction: the Kelly gate below runs
    # BEFORE the premise check, so only candidates that already passed Kelly
    # reach it. That statement order is preserved exactly as it stood before
    # this function was extracted (a test pins the resulting count in the
    # WARNING) — do not reorder the two.
    premise_violations = 0

    for rec in raw_entries:
        # Group identity and the _find_entry result, exactly as recorded by
        # _entries_for_band (inside _prepare_entries, or once per band in a
        # sweep) — mA/mB inside the entry may have been swapped there to
        # canonicalize which leg is A.
        pair_type = rec["pair_type"]
        canon     = rec["canon"]
        group_key = rec["group_key"]
        entry     = rec["entry"]

        # Unpack entry — mA/mB may have been swapped inside _find_entry to canonicalize
        mA = entry["mA"]
        mB = entry["mB"]
        pA, pB, nA, nB = entry["pA"], entry["pB"], entry["nA"], entry["nB"]
        entry_date = entry["entry_date"]

        # The two prices actually paid — (nA, pB) for same_title, (pA, nB) for
        # time_series — the backtester's mirror of scanner.leg_prices
        price_a, price_b = _leg_prices_for(pair_type, pA, nA, pB, nB)

        # ── Kelly fraction (sizing happens in Pass 2 against the checkpoint) ──
        # Compute the net spread on the leg prices after the continuous fee approximation
        fee_approx = fee_per_pair_approx(price_a, price_b)
        net_spread = (1.0 - price_a - price_b) - fee_approx
        # REPORTED/RANKED return on the contracts' cost (feeds
        # entry_monthly_ratio, Pass 2's look-ahead-free sort key) — the mirror
        # of strategy.TradeSpec.profit_ratio, fee-less denominator and all.
        profit_ratio_entry = net_spread / (price_a + price_b) if net_spread > 0 else 0.0
        # Kelly's "b": the SAME numerator over the dollars actually at risk,
        # which include the fee — a losing pair loses cost + fees, not cost
        # (DR-62). A DIFFERENT quantity from profit_ratio_entry above; mirrors
        # strategy._evaluate_size's kelly_b exactly, so live and backtest admit
        # the same pairs. Do not collapse the two back together.
        kelly_b_entry = (net_spread / (price_a + price_b + fee_approx)
                         if net_spread > 0 else 0.0)

        # Probability model. time_series: the discounted market-implied
        # in-between mass, 1 - k * (pB - pA), from config.time_series_profit_prob
        # — the single definition strategy._kelly_p and dashboard._kelly_fraction
        # also call, so the three can never drift (called directly by name here;
        # a test pins the two-link chain run_backtest -> _simulate_at_discount
        # -> the helper). k is this function's override, and None — what
        # run_backtest passes — is the sentinel the helper resolves to
        # config.TIME_SERIES_INTERVAL_PROB_DISCOUNT, so the default path prices
        # exactly as live sizing does. same_title: the fixed co-resolution prior.
        p = (time_series_profit_prob(pA, pB, k=k)
             if pair_type == "time_series" else SAME_TITLE_CO_RESOLVE_PROB)
        q = 1.0 - p

        # Kelly formula: f* = p - q/b; non-positive means no positive expected
        # value once the fee is counted on the losing side too (DR-62)
        kelly_f = (p - q / kelly_b_entry) if kelly_b_entry > 0 else -1.0
        if kelly_f <= 0:
            # Kelly fraction is non-positive — the pair has no positive expected value
            continue
        # Cap at BUDGET_FRACTION (20%) to avoid over-concentration
        kelly_f_capped = min(BUDGET_FRACTION, kelly_f)

        # Skip pairs where the settlement result is missing or non-binary
        outcome_a = mA.get("result", "")
        outcome_b = mB.get("result", "")
        if outcome_a not in ("yes", "no") or outcome_b not in ("yes", "no"):
            continue

        # Earlier YES with later NO cannot happen for a cumulative-deadline
        # pair (YES by the earlier deadline implies YES by the later one), so
        # this pair was not one: skip it — never traded, never paid — and
        # count it for the summary WARNING after the loop. _settlement_receipt
        # would raise on this cell; excluding it here keeps that guard
        # unreachable from the pipeline.
        if pair_type == "time_series" and outcome_a == "yes" and outcome_b == "no":
            premise_violations += 1
            continue

        # Determine the exit date as the later of the two settlement timestamps.
        # Belt-and-braces: fetch_all_settled_markets only keeps records that
        # carry a settlement_ts, so a miss here should be impossible — but an
        # unparseable value must not raise out of the candidate loop, and a
        # candidate with no exit date can't be cash-simulated at all (its
        # capital would be released on an invented day), so it is skipped.
        exit_date_a = _parse_iso_date(mA.get("settlement_ts"))
        exit_date_b = _parse_iso_date(mB.get("settlement_ts"))
        if exit_date_a is None or exit_date_b is None:
            logging.debug(
                "Skipping candidate %s/%s: missing or unparseable settlement_ts",
                mA.get("ticker"), mB.get("ticker"),
            )
            continue
        exit_date   = max(exit_date_a, exit_date_b)

        holding_days = max(1, (exit_date - entry_date).days)

        # Entry-time priority metric: the expected return normalized to 30 days
        # using only information available at entry (entry prices and the market
        # close dates). Sorting Pass 2 by REALIZED returns would leak settlement
        # outcomes into trade selection (look-ahead bias).
        # Belt-and-braces again: _find_entry already required a parseable
        # close_time on both legs to derive its scan window, so neither guard
        # can fire in practice — but a bare [...] index plus an unguarded
        # fromisoformat would turn any future data drift into a mid-run crash.
        close_a_d = _parse_iso_date(mA.get("close_time"))
        close_b_d = _parse_iso_date(mB.get("close_time"))
        if close_a_d is None or close_b_d is None:
            logging.debug(
                "Skipping candidate %s/%s: missing or unparseable close_time",
                mA.get("ticker"), mB.get("ticker"),
            )
            continue
        expected_days = max(1, (max(close_a_d, close_b_d) - entry_date).days)
        entry_monthly_ratio = profit_ratio_entry * 30.0 / expected_days

        # Use title > subtitle > ticker as the display label for each market
        title_a = mA.get("title") or mA.get("subtitle") or mA.get("ticker", "")
        title_b = mB.get("title") or mB.get("subtitle") or mB.get("ticker", "")

        candidates.append({
            "pair_type": pair_type,
            "canon": canon,
            "group_key": group_key,
            "mA": mA, "mB": mB,
            "pA": pA, "pB": pB, "nA": nA, "nB": nB,
            # Leg prices Pass 2 sizes, fees and pays out on
            "price_a": price_a, "price_b": price_b,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "outcome_a": outcome_a,
            "outcome_b": outcome_b,
            "kelly_f_capped": kelly_f_capped,
            "entry_monthly_ratio": entry_monthly_ratio,
            "holding_days": holding_days,
            "title_a": title_a,
            "title_b": title_b,
            # Carried straight from _find_entry (None for same_title) so the
            # recorded trade reports the same gap the tier was chosen from
            "gap_days": entry["gap_days"],
        })

    if premise_violations:
        # Summary-warning idiom (silent at zero). This used to be the ONLY
        # signal that the grouping had admitted a non-cumulative pair; since
        # _extract_pairs screens both legs' wording, it is defence in depth
        # behind that heuristic, and a non-zero count is now itself a finding
        # — it means a pair whose wording read as two cumulative deadlines
        # settled as an apparently non-nesting pair. That is most likely a
        # wording false negative, but it is not the only mechanism: it can
        # also be a CROSS-EVENT pair whose legs were ordered on an early
        # REALIZED close (ranking them by scheduled close would have kept the
        # nesting) — cross-event only since DR-73, because a same-event ladder
        # is ordered on its two STATED deadlines and never on close_time — or
        # strike-blind grouping on a cache without subtitles.
        # DR-72 widens the named CAUSES beyond the single "mixed snapshot
        # family" guess this line used to make — see the cause list below,
        # and CLAUDE.md's strategy-change gotcha for what each one means.
        logging.warning(
            "Excluded %d time-series candidate(s) whose settlement violated the "
            "cumulative-deadline premise (earlier YES, later NO) — the "
            "pair passed the wording screen in _extract_pairs but still settled "
            "as a non-nesting pair. Most likely a wording false negative (e.g. "
            "snapshot markets, or recurring windows worded 'before <date>', "
            "read as cumulative); legs ordered on an early REALIZED close — a "
            "later-deadline leg that resolved YES before the earlier leg's "
            "deadline, which is possible for CROSS-EVENT pairs only, since a "
            "same-event ladder is ordered on its stated deadlines; or "
            "strike-blind grouping on a cache without subtitles (see the "
            "outcome-label coverage line)",
            premise_violations,
        )

    # Keep only the single best candidate per title group — mirrors the live
    # scanners' CONCEPT (one pair per normalized-title / exact-title group so
    # the portfolio isn't flooded with near-identical correlated positions),
    # not their tie-break rule: the live scanners rank by tradeable-first then
    # largest price gap (scanner.py's group_pairs.sort), while this ranks by
    # largest entry_monthly_ratio — a different quantity, chosen here because
    # entry-time expected return is what Pass 2 below sizes and orders on.
    # Dedup on the FULL group_key (which includes event_title for same_title
    # groups), not the display-only canon — two unrelated events sharing an
    # option label (e.g. "Trump" in two different events) have the same canon
    # but distinct group_keys, and must remain two separate candidates.
    best_by_group: dict = {}
    for c in candidates:
        key = (c["pair_type"], c["group_key"])
        cur = best_by_group.get(key)
        if cur is None or c["entry_monthly_ratio"] > cur["entry_monthly_ratio"]:
            best_by_group[key] = c
    candidates = list(best_by_group.values())

    # Cross-type dedup, mirroring main._dedup_pairs on the live path: the same
    # two tickers can be discovered by BOTH groupings, and the live pipeline
    # keeps only the same-title copy (simpler co-resolution model). Must run
    # here rather than relying on Pass 2's ticker-conflict filter, which only
    # stops both copies being open at once — not the time-series copy being
    # entered whenever the same-title one is skipped for sizing reasons.
    candidates = _drop_cross_type_duplicates(candidates)

    # Chronological order for the cash simulation; within one entry date, take
    # the best ENTRY-TIME expected return first (same_title preferred at ties,
    # matching strategy.select_portfolio)
    candidates.sort(
        key=lambda c: (c["entry_date"], -c["entry_monthly_ratio"], c["pair_type"] != "same_title"),
    )

    # ── Pass 2: chronological cash-constrained greedy selection ───────────────
    # Walk entries in date order, maintaining a running cash balance. Sizing
    # mirrors main._run_prod exactly: every candidate on a given entry date
    # (one Monday checkpoint = one live run) is Kelly-sized against that
    # checkpoint's OPENING balance — the cash after that day's settlement
    # receipts have returned, i.e. what verify_auth would report at the top
    # of the run — and then admitted greedily, in the same order
    # select_portfolio uses, only while its fee-inclusive cost still fits the
    # RUNNING cash (select_portfolio's `total_cost_with_fees > available`).
    # Sizing later same-day trades off the running cash instead (the old
    # behaviour) made every trade after the first on a busy Monday smaller
    # than live would make it. Settlement receipts return to cash on their
    # exit dates. A ticker-conflict
    # filter mirrors the live
    # bot's rule precisely: at most one ACTIVE position per ticker. Live, that
    # rule comes from scanner.get_held_tickers(), which queries positions with
    # count_filter="position" — so a ticker leaves the held set once its market
    # settles and may be entered again. The backtest therefore holds a ticker
    # only until its trade's exit date, released below on the same schedule as
    # the settlement receipts.
    trades: list[BacktestTrade] = []
    active_tickers: set[str] = set()
    cash = initial_balance
    # Opening balance of the checkpoint currently being walked — the Kelly
    # base for every candidate entering on that date (see comment above).
    checkpoint_date: date | None = None
    checkpoint_cash = cash
    pending_exits: list[tuple[date, float]] = []  # (exit_date, settlement receipt)
    # (exit_date, ticker) for every leg of a still-open trade — the release
    # ledger for active_tickers, kept alongside pending_exits so cash and
    # ticker availability are always freed on exactly the same day.
    active_until: list[tuple[date, str]] = []

    for c in candidates:
        d = c["entry_date"]
        # Release settlement receipts from trades that exited on or before this entry
        cash += sum(amt for ed, amt in pending_exits if ed <= d)
        pending_exits = [(ed, amt) for ed, amt in pending_exits if ed > d]
        # Release the tickers of those same settled trades — the position is
        # closed, so (as live) the ticker is no longer blocked. Set difference
        # is safe because the conflict filter below guarantees a ticker is in
        # at most one open trade at a time.
        # NOTE the `<= d` (not `< d`): a trade exiting ON date d frees its
        # tickers for a later candidate entering that same date d. This is a
        # deliberate symmetry with the cash rule directly above, which likewise
        # returns that trade's settlement receipt on d — both resources are
        # freed on exactly the same day, so a same-day re-entry is funded and
        # unblocked together rather than one without the other.
        active_tickers.difference_update(tk for ed, tk in active_until if ed <= d)
        active_until = [(ed, tk) for ed, tk in active_until if ed > d]

        if d != checkpoint_date:
            # First candidate of a new checkpoint: receipts for this date have
            # just been returned above, so this is the balance a live run
            # starting today would read and size everything against.
            checkpoint_date = d
            checkpoint_cash = cash

        mA, mB = c["mA"], c["mB"]
        # Skip if either ticker is still committed to a trade that hasn't settled
        if mA["ticker"] in active_tickers or mB["ticker"] in active_tickers:
            continue

        # The traded leg prices (see _leg_prices_for) — every dollar figure
        # below is computed on these, never on the reporting-only quotes
        price_a, price_b = c["price_a"], c["price_b"]

        # Kelly sizing against the checkpoint's opening balance (live:
        # compute_trade(pair, balance_cents) with one balance for the run)
        budget = checkpoint_cash * c["kelly_f_capped"]
        n = int(budget / (price_a + price_b))
        if n < 1:
            # Kelly budget can't afford one contract — live compute_trade skips too
            continue

        # Mirror live compute_trade's shrink loop exactly (strategy.py): the
        # budget above covers the CONTRACTS only, while the exact ceiling-rounded
        # fees ride on top — so the raw n systematically overshoots the Kelly cap.
        # Shrink until the fee-inclusive cost actually fits the Kelly budget.
        # (No max_contracts analog here: the backtest has no orderbook depth to
        # cap against, only candle closes.)
        fee_a, fee_b = fee_leg_exact(n, price_a), fee_leg_exact(n, price_b)
        while n > 0 and n * (price_a + price_b) + fee_a + fee_b > budget:
            n -= 1
            fee_a, fee_b = fee_leg_exact(n, price_a), fee_leg_exact(n, price_b)
        if n < 1:
            # Fees ate the entire Kelly budget — no contract count fits
            continue

        total_cost = n * (price_a + price_b)
        # Exact ceiling-rounded taker fees for both legs, charged at entry
        fees = fee_a + fee_b
        # Win-scenario NET profit after exact fees (same_title: the floor every
        # co-resolution outcome clears; time_series: the profit of either win
        # cell) — reject if the ceiling rounding ate the margin (mirrors live
        # compute_trade's min_payoff gate)
        expected_payoff = n * (1.0 - price_a - price_b) - fees
        if expected_payoff <= 0:
            continue
        # Greedy fit against the RUNNING cash — select_portfolio's admission
        # rule; a spec that no longer fits is skipped, later cheaper ones may
        # still be admitted.
        if total_cost + fees > cash:
            continue

        # Realized P&L from the settlement outcomes — n per leg whose market
        # resolved to the side bought (sides depend on the pair type)
        receipt      = _settlement_receipt(n, c["outcome_a"], c["outcome_b"], c["pair_type"])
        profit       = receipt - total_cost - fees
        invested     = total_cost + fees
        profit_ratio = profit / invested if invested > 0 else 0.0
        # Normalize realized return to a 30-day equivalent (reporting only)
        monthly_profit_ratio = profit_ratio * 30.0 / c["holding_days"]
        # Slippage = realized profit vs. the win-scenario payoff (net vs. net)
        slippage = profit - expected_payoff

        # Reporting-only population labels: market A's event ticker, and
        # whether the pair is a same-event ladder — through _is_ladder_pair,
        # the one definition a band sweep's "ladder"/"cross" populations also
        # split on. Named is_ladder, not same_event_ladder: that name is
        # scanner's imported ladder helper, which a local would shadow for
        # this whole function.
        event_a = mA.get("event_ticker") or ""
        is_ladder = _is_ladder_pair(c["pair_type"], mA, mB)

        trades.append(BacktestTrade(
            pair_type=c["pair_type"],
            ticker_a=mA["ticker"],
            ticker_b=mB["ticker"],
            title_a=c["title_a"],
            title_b=c["title_b"],
            # infer_category maps the event_ticker prefix to a human-readable label (e.g. "Crypto")
            category=infer_category(mA.get("event_ticker", "")),
            entry_date=c["entry_date"],
            exit_date=c["exit_date"],
            entry_pA=c["pA"],
            entry_pB=c["pB"],
            entry_nA=c["nA"],
            entry_nB=c["nB"],
            n=n,
            total_cost=total_cost,
            fees=fees,
            outcome_a=c["outcome_a"],
            outcome_b=c["outcome_b"],
            actual_payoff=receipt,
            profit=profit,
            profit_ratio=profit_ratio,
            monthly_profit_ratio=monthly_profit_ratio,
            kelly_fraction=c["kelly_f_capped"],
            expected_payoff=expected_payoff,
            slippage=slippage,
            holding_days=c["holding_days"],
            balance_at_entry=checkpoint_cash,
            deadline_gap_days=c["gap_days"],
            event_ticker=event_a,
            same_event_ladder=is_ladder,
        ))

        # Cash out the door: contracts plus fees; the receipt comes back at exit
        cash -= invested
        pending_exits.append((c["exit_date"], receipt))

        # Mark both tickers as active so no OVERLAPPING pair is added later;
        # the release ledger frees them again on this trade's exit date.
        active_tickers.add(mA["ticker"])
        active_tickers.add(mB["ticker"])
        active_until.append((c["exit_date"], mA["ticker"]))
        active_until.append((c["exit_date"], mB["ticker"]))

    # Named with the RESOLVED discount, the resolved band and the population.
    # A default run emits this line once per swept k, and a band sweep once
    # per (band, k, population) plus its split-half and ex-top runs, so
    # without all three a reader could not tell which simulation a trade
    # count belonged to (TS-21); every prefix up to the ':' is unique within
    # one sweep — k and band are printed through _exact_label, so an off-grid
    # value that rounds onto a grid member still prints distinctly.
    # effective_k and the resolved band, never the arguments, so the None
    # sentinels are never printed.
    logging.info(
        "Backtest complete at k=%s, band %s, %s: %d trades, %d profitable",
        _exact_label(effective_k, ".3f"),
        _band_label((band_lo, band_hi)),
        population,
        len(trades),
        sum(1 for t in trades if t.profit > 0),
    )

    equity_df = _build_equity_curve(trades, start_date, initial_balance)
    return SweepPoint(
        k=effective_k, trades=trades, equity_df=equity_df,
        # The resolved tuple when a band was given, so (0, 1) and (0.0, 1.0)
        # stamp the same scenario; None stays None ("given no band").
        spread_band=None if spread_band is None else (band_lo, band_hi),
        population=population,
    )


# ─── Interval-discount calibration ────────────────────────────────────────────

def _calibration_bucket(
    label: str,
    tier: float,
    observations: list[_TimeSeriesOutcome],
) -> IntervalCalibrationBucket:
    """
    Reduce a set of time-series observations to one calibration report row.

    Computes the realised in-between rate, the mean market-implied in-between
    mass, and their ratio — the empirical interval discount k_hat, i.e. how
    much of the mass the market priced actually materialized.

    Args:
        label (str): Row label for the bucket ("0-7d", "POOLED", ...).
        tier (float): Price tier for the bucket, or 0.0 for the pooled row,
            which spans every tier (see IntervalCalibrationBucket.tier).
        observations (list[_TimeSeriesOutcome]): The bucket's candidates. May
            be empty, which yields a zeroed row rather than a division error.

    Returns:
        IntervalCalibrationBucket: The row. empirical_k is None whenever
            mean_implied is not strictly positive — with a zero (or, from
            reporting-only clamping, negative) implied mass the ratio is
            undefined, and reporting it as 0.0 would read as "the market
            overstated everything" rather than "not measurable".
    """
    n = len(observations)
    if n == 0:
        # Reachable only for the pooled row (empty gap bands are dropped by
        # the caller), when every time-series candidate was a premise
        # violation. Report the shape rather than dividing by zero.
        return IntervalCalibrationBucket(
            label=label, tier=tier, n=0,
            realised_rate=0.0, mean_implied=0.0, empirical_k=None,
        )

    realised_rate = sum(1 for o in observations if o.in_between) / n
    mean_implied  = sum(o.implied for o in observations) / n
    empirical_k   = realised_rate / mean_implied if mean_implied > 0 else None
    return IntervalCalibrationBucket(
        label=label, tier=tier, n=n,
        realised_rate=realised_rate, mean_implied=mean_implied,
        empirical_k=empirical_k,
    )


def _interval_calibration(
    raw_entries: list[dict],
    spread_min: float | None = None,
) -> IntervalCalibration | None:
    """
    Measure the empirical interval discount k over the prepared entries.

    The time-series bet loses exactly one settlement cell: the event first
    happens BETWEEN the two deadlines (earlier NO, later YES). The market
    prices that cell at pB - pA; config.time_series_profit_prob believes only
    TIME_SERIES_INTERVAL_PROB_DISCOUNT of it. This function measures the
    fraction that actually materialized:

        k_hat = P(earlier NO, later YES) / mean(pB - pA)

    pooled and per deadline-gap band, so an operator can compare the hand-set
    constant against what the history did.

    The population is deliberately k-INDEPENDENT: it reads the prepared
    entries directly (_prepare_entries()' output, or one spread band's
    _entries_for_band() output inside a sweep), NOT _simulate_at_discount()'s
    surviving candidates, so it is NOT filtered by the Kelly gate. Filtering
    by Kelly would make the estimate circular — the in-between rate would be
    measured only among the pairs the CURRENT k already liked, so a wrong k
    would confirm itself. Being k-independent also means one computation is
    valid for every k at one band, which is why BacktestSweep holds one of
    these per band rather than each SweepPoint holding its own. It is not
    band-independent: a band decides which pairs enter at all.

    Two properties of the population to keep in mind when reading the number:

      - Premise violations (earlier YES, later NO) are excluded from the
        denominator entirely. Such a pair is most likely not a cumulative-
        deadline pair at all (a wording false negative), though a genuinely
        nested pair ordered on an early REALIZED close, or strike-blind
        grouping on a cache without subtitles, can also land here (DR-72).
        Either way it is neither a clean in-between event nor a valid
        non-event, and leaving it in would bias the rate in an arbitrary
        direction, so it is excluded regardless of which cause produced it.
        They are counted separately on the result. That count is NOT the
        same quantity
        as _simulate_at_discount()'s `premise_violations`, whose WARNING is
        emitted per simulated discount: that counter sits AFTER the Kelly
        gate, so it sees only Kelly-passing candidates and is generally
        SMALLER than this one. Two different populations, two different names,
        neither a bug in the other.
      - It is conditional on the strategy's own entry filters — only pairs
        whose gap already cleared its tier ever produced an entry. That is a
        feature, not a sampling flaw: it is exactly the conditional
        distribution the live sizer faces, so k_hat is the right number to
        compare TIME_SERIES_INTERVAL_PROB_DISCOUNT against.

    Args:
        raw_entries (list[dict]): Prepared entries (_prepare_entries() output,
            or one band's _entries_for_band() output) — one record per pair
            that produced an entry. Same-title records are ignored: they
            price on the fixed co-resolution prior and have no in-between cell
            or deadline gap at all.
        spread_min (float | None): The backtest spread band's FLOOR the
            entries were detected under, handed to
            config.min_price_diff_for_gap so each gap band's `tier` is the
            floor its entries actually cleared — max(tier, spread_min). None
            (default) labels the deadline-gap tiers alone, which is also what
            the default band's floor of 0.0 labels. Labelling only: it filters
            nothing, since the band already acted inside _find_entry.

    Returns:
        IntervalCalibration | None: The report, or None when there is nothing
            to report — no time-series candidate produced a usable
            observation AND none was excluded as a premise violation (the
            codebase's return-None-on-nothing-to-say convention). Unlike most
            such conventions here, this None is NOT silent at the caller
            (DR-72): _log_interval_calibration logs one explanatory line for
            it rather than nothing at all.
    """
    observations: list[_TimeSeriesOutcome] = []
    excluded = 0

    for rec in raw_entries:
        # Same-title pairs have no in-between cell and no deadline gap — the
        # discount being calibrated does not appear in their model at all.
        if rec["pair_type"] != "time_series":
            continue

        entry = rec["entry"]
        mA, mB = entry["mA"], entry["mB"]
        outcome_a = mA.get("result", "")
        outcome_b = mB.get("result", "")

        # A missing or non-binary settlement cannot be classified as
        # in-between or not, so it can be neither numerator nor denominator.
        if outcome_a not in ("yes", "no") or outcome_b not in ("yes", "no"):
            continue

        # Earlier YES with later NO is impossible for a cumulative-deadline
        # pair: most likely the grouping admitted a non-cumulative one (a
        # wording false negative), though a genuinely nested pair inverted by
        # early-REALIZED-close leg ordering can land here too (DR-72).
        # Excluded from the denominator and counted for the report either way.
        if outcome_a == "yes" and outcome_b == "no":
            excluded += 1
            continue

        observations.append(_TimeSeriesOutcome(
            # Carried out of _find_entry rather than recomputed, so a
            # candidate is always bucketed under the very gap its price tier
            # and the MAX_DEADLINE_GAP_DAYS cutoff were applied on.
            gap_days=entry["gap_days"],
            implied=entry["pB"] - entry["pA"],
            in_between=(outcome_a == "no" and outcome_b == "yes"),
        ))

    if not observations and not excluded:
        # Nothing measurable and nothing excluded — a same-title-only (or
        # empty) run. None keeps the caller from printing an all-zero table
        # (the caller itself is no longer silent on None — DR-72 — it logs
        # one explanatory line instead).
        return None

    buckets: list[IntervalCalibrationBucket] = []
    for lo, hi in _CALIBRATION_GAP_BANDS:
        band = [o for o in observations
                if o.gap_days is not None and lo <= o.gap_days <= hi]
        if not band:
            # Omit empty bands rather than emitting an all-zero row.
            continue
        buckets.append(_calibration_bucket(
            f"{lo}-{hi}d",
            # Never hardcode the 0.15/0.30 tiers: read them from the same
            # helper _find_entry and the live scanner select with. A band
            # never straddles the tier boundary (see _CALIBRATION_GAP_BANDS),
            # so its upper edge names the whole band's tier. spread_min is the
            # same band floor _find_entry layered on that tier, so the label
            # is the floor these entries were actually detected under.
            min_price_diff_for_gap(hi, spread_min=spread_min),
            band,
        ))

    # 0.0 tier: the pooled row spans every band, so it has no single tier.
    pooled = _calibration_bucket(_CALIBRATION_POOLED_LABEL, 0.0, observations)
    return IntervalCalibration(
        pooled=pooled,
        buckets=buckets,
        excluded_premise_violations=excluded,
    )


def _log_interval_calibration(calibration: IntervalCalibration | None) -> None:
    """
    Log the interval-discount calibration report.

    Split from _interval_calibration() the same way scanner.check_shard_coverage
    (pure comparison) is split from main._log_shard_coverage (decides how
    loudly to report): the measurement stays testable and reusable without log
    noise, and this decides the presentation. Follows the file's summary-line
    idiom for its SUB-counts — the premise-violation line is silent at zero —
    but the top-level None case is no longer silent (DR-72): absence of a
    warning must never be the only signal (the DR-66 lesson), and a truly
    empty log line here read exactly like "nothing was logged because this
    run has not gotten here yet", indistinguishable from a hang or a crash
    upstream. calibration is None precisely when no time-series candidate
    produced a usable observation and none was excluded as a premise
    violation — see _interval_calibration's Returns — and that fact is now
    stated explicitly rather than implied by silence.

    The report is a RECOMMENDATION ONLY. Nothing in the backtester writes
    config.py, and the live sizer keeps reading
    config.TIME_SERIES_INTERVAL_PROB_DISCOUNT regardless of what this prints;
    acting on it is a deliberate human edit.

    Args:
        calibration (IntervalCalibration | None): _interval_calibration()'s
            result. None logs one explanatory line and returns.

    Returns:
        None
    """
    if calibration is None:
        logging.info(
            "Interval-discount calibration: no time-series candidate entry "
            "with a readable settlement in this window — empirical k_hat is "
            "not measurable"
        )
        return

    logging.info(
        "Interval-discount calibration (k_hat = realised in-between rate / "
        "market-implied gap)"
    )
    logging.info("  %-14s%4s%9s%12s%11s%10s",
                 "bucket", "tier", "n", "realised", "implied", "k_hat")

    for b in [*calibration.buckets, calibration.pooled]:
        # tier <= 0 marks the pooled row, which spans every tier and so has no
        # single one to print (IntervalCalibrationBucket.tier).
        tier_txt = "-" if b.tier <= 0 else f"{b.tier:.2f}"
        k_txt = "-" if b.empirical_k is None else f"{b.empirical_k:.3f}"
        logging.info("  %-14s%4s%9d%12.4f%11.4f%10s",
                     b.label, tier_txt, b.n, b.realised_rate, b.mean_implied, k_txt)

    pooled_k = calibration.pooled.empirical_k
    logging.info(
        "  Configured k = %.3f (config.TIME_SERIES_INTERVAL_PROB_DISCOUNT) | "
        "pooled empirical k_hat = %s",
        # The CONFIG constant, not any sweep point's override: this line
        # compares the measurement against what live sizing actually reads.
        TIME_SERIES_INTERVAL_PROB_DISCOUNT,
        "-" if pooled_k is None else f"{pooled_k:.3f}",
    )
    logging.info("  Recommendation only — config.py is never written by the backtester.")

    if calibration.excluded_premise_violations:
        # Summary-line idiom: silent at zero.
        logging.info(
            "  Excluded %d premise violation(s) (earlier YES / later NO) from "
            "the denominator.",
            calibration.excluded_premise_violations,
        )


def run_backtest(
    hist_client: Any,
    live_client,
    start_date: date = date(2024, 1, 1),
    initial_balance: float = 10_000.0,
    use_cache: bool = True,
    max_horizon_days: int | None = None,
) -> tuple[list[BacktestTrade], pd.DataFrame]:
    """
    Replay both pair strategies on all settled Kalshi markets from start_date.

    Thin composition of the backtest's two halves, at the interval discount
    config.TIME_SERIES_INTERVAL_PROB_DISCOUNT (i.e. exactly what the live sizer
    uses): _prepare_entries() does the k-independent work and
    _simulate_at_discount(..., k=None) does the k-dependent work.

    Algorithm (unchanged; the step split between the two helpers is noted):
      1. Fetch all settled markets since start_date.                [prepare]
      2. Drop markets that provably can never enter (_can_ever_enter). [prepare]
      3. Group into potential time-series and same-title pairs.     [prepare]
      4. Fetch hourly candlesticks for every ticker appearing in a
         potential pair, in parallel across CANDLESTICK_FETCH_MAX_WORKERS
         threads.                                                   [prepare]
      5. Find the first Monday where the pair was tradeable at the
         threshold.                                                 [prepare]
      6. Kelly-gate each entry; exclude (and count, with one summary
         WARNING) any time-series candidate whose settlement was
         earlier-YES/later-NO — impossible for a cumulative-deadline
         pair, so a premise violation rather than a payout; keep only
         the best entry per title group (live one-pair-per-group rule),
         then drop any time-series candidate whose ticker pair was also
         found as a same-title candidate (live main._dedup_pairs rule).
                                                                   [simulate]
      7. Walk entries chronologically with a running cash balance: Kelly-size
         every candidate of an entry date against that checkpoint's opening
         balance, admit it only while its fee-inclusive cost still fits the
         running cash (mirroring main._run_prod + strategy.select_portfolio),
         and record actual P&L from settlement outcomes.            [simulate]
      8. Build an equity curve from the trade timeline.             [simulate]

    Args:
        hist_client (Any): Signed client for the historical archive/live endpoints.
        live_client: Client passed through to fetch_all_settled_markets.
        start_date (date): Earliest settlement date to include.
        initial_balance (float): Simulated starting cash balance in dollars.
        use_cache (bool): Whether to reuse the disk-cached assembled market list.
        max_horizon_days (int | None): Optional opt-in bet-horizon cap mirroring
            scanner.filter_markets_within_horizon on the live path, but relative
            to each simulated checkpoint rather than real-world now: at a given
            Monday checkpoint, a pair can only enter if the later-closing leg
            closes within max_horizon_days of THAT checkpoint. None (default)
            applies no cap, matching current behavior. Passed straight through
            to _find_entry() for each candidate pair.

    Returns:
        tuple[list[BacktestTrade], pd.DataFrame]: (trades, equity_df).
            trades is one BacktestTrade per entered pair, in entry-date order
            (empty if none were ever entered). equity_df has columns
            [date, portfolio_value, daily_return], one row per day from
            start_date - 1 day (the untouched initial balance) through today,
            flat at initial_balance if trades is empty. portfolio_value is cash
            plus open positions carried at cost, so deploying capital does not
            move it (see _build_equity_curve).

    Raises:
        KeyError: Propagates out of the candlestick-fetch pool
            (_fetch_candles_parallel) if a ticker needed by a candidate pair
            was not properly excluded by the eligibility prefilter — this is
            treated as a real defect (a market that should never have reached
            this stage), not degraded into "no price history".

    Note:
        Before any network call, _prepare_entries() checks whether [start_date,
        today] contains at least one Monday 09:00 UTC checkpoint (the only kind
        _find_entry() can ever act on). If not, no trade can ever be entered
        regardless of what the fetch would return, so the fetch is skipped
        entirely, that helper returns None, and this returns the same
        empty-result shape as the zero-trade path ([], an equity curve flat at
        initial_balance — for a future start_date, its leading row plus
        start_date's own) with a WARNING logged.
    """
    logging.info("Starting backtest from %s with $%.2f", start_date, initial_balance)

    # The k-independent half: fetch, group, pair and locate each pair's first
    # tradeable Monday. None means the feasibility pre-check failed.
    #
    # The outcome-label census rides out alongside the entries (DR-66b), but
    # this entry point returns the historical two-tuple and feeds no dashboard,
    # so it is discarded here — the census has already logged itself.
    #
    # No same_event_ladders argument, deliberately: this function keeps its
    # exact pre-DR-73 signature for every existing caller, and omitting the
    # keyword leaves _prepare_entries' None sentinel to resolve this module's
    # TIME_SERIES_SAME_EVENT_LADDERS (bound from config at import) at call
    # time — the value live sizing uses. A harness that wants ladders here must
    # patch backtester.TIME_SERIES_SAME_EVENT_LADDERS, not the config
    # attribute, which this module never reads; the supported lever is
    # run_backtest_sweep(same_event_ladders=...), which needs no patching.
    raw_entries, _ = _prepare_entries(
        hist_client, live_client, start_date, use_cache, max_horizon_days
    )
    if raw_entries is None:
        # Same empty-result shape the zero-trade path produces, so backtest.py
        # and generate_dashboard need no special case for this early-exit.
        return [], _build_equity_curve([], start_date, initial_balance)

    # The k-dependent half, at the config discount: k=None is the "no override"
    # sentinel config.time_series_profit_prob resolves to
    # TIME_SERIES_INTERVAL_PROB_DISCOUNT, so this prices exactly as the live
    # sizer does.
    point = _simulate_at_discount(raw_entries, start_date, initial_balance, k=None)
    return point.trades, point.equity_df


def _total_return(point: SweepPoint, initial_balance: float) -> float:
    """
    Return a simulated point's total return off its equity curve's last row.

    The same (final − initial) / initial the dashboard's performance card
    computes from equity_df, so a split-half or excluding-top-event figure is
    on exactly the footing of the scenario's own return beside it.

    Args:
        point (SweepPoint): A simulation's result.
        initial_balance (float): The balance that simulation started from, in
            dollars.

    Returns:
        float: The fractional total return — 0.0 for a point with no trades,
            and 0.0 when initial_balance is 0 (nothing can be sized from it),
            the same guard the dashboard's per-k table applies to its opening
            balance, so a zero-balance band sweep cannot raise
            ZeroDivisionError after its fetch.
    """
    if not initial_balance:
        return 0.0
    final_value = float(point.equity_df["portfolio_value"].iloc[-1])
    return (final_value - initial_balance) / initial_balance


def _split_date(entries: list[dict], start_date: date) -> date:
    """
    Choose the date a band sweep's split-half check splits entries at.

    statistics.median_low of the entry dates, never statistics.median: median
    AVERAGES the two middle values of an even-length list, which raises
    TypeError on dates, while median_low returns one of them. The fallback,
    for a band with no entry at all, is the midpoint of the backtest window
    [start_date, today UTC] — a date that splits nothing, since there is
    nothing to split, but one that keeps BacktestSweep.split_date a date.

    median_low is always one of the dates, so H2 (on or after it) is never
    empty when there is an entry — but H1 (strictly before it) IS empty
    whenever at least half of the entries share the earliest entry date (one
    of two is enough); _sweep_from_candidates warns when that happens at the
    primary band.

    Args:
        entries (list[dict]): The entries to split — _sweep_from_candidates
            hands it the primary band's TIME-SERIES entries only.
        start_date (date): The backtest's start date.

    Returns:
        date: The split date. H1 is every entry strictly before it, H2 every
            entry on or after it.
    """
    dates = [rec["entry"]["entry_date"] for rec in entries]
    if dates:
        return statistics.median_low(dates)
    # UTC for the same reason _build_equity_curve and the feasibility check
    # use it (TS-13): the window the backtest simulates ends on today's UTC date.
    today = datetime.now(UTC).date()
    return start_date + timedelta(days=max((today - start_date).days, 0) // 2)


def _split_halves(entries: list[dict], split_date: date) -> tuple[list[dict], list[dict]]:
    """
    Split entries at a band sweep's one split date, keeping their order.

    Args:
        entries (list[dict]): Prepared entry records (each with
            ["entry"]["entry_date"]).
        split_date (date): BacktestSweep.split_date.

    Returns:
        tuple[list[dict], list[dict]]: (every entry strictly before
            split_date, every entry on or after it) — either may be empty.
    """
    return ([rec for rec in entries if rec["entry"]["entry_date"] < split_date],
            [rec for rec in entries if rec["entry"]["entry_date"] >= split_date])


def _half_split(
    halves: tuple[list[dict], list[dict]],
    start_date: date,
    initial_balance: float,
    k: float,
    band: tuple[float, float],
    population: str = "all",
) -> HalfSplit:
    """
    Simulate each half of one scenario's entries alone and keep three numbers each.

    Args:
        halves (tuple[list[dict], list[dict]]): The scenario's entries split
            at BacktestSweep.split_date — (before it, on or after it).
        start_date (date): The backtest's start date.
        initial_balance (float): The balance EACH half starts from, in dollars
            — the halves are two independent runs, never one run's two parts.
        k (float): The scenario's resolved interval discount.
        band (tuple[float, float]): The scenario's resolved band (a label for
            the completion lines; the entries already reflect it).
        population (str): The checked population the halves belong to —
            "all" (default) or "time_series" — which names the two runs
            "<population>/H1" and "<population>/H2" on their completion lines,
            so the two populations' split-half runs never share a prefix.

    Returns:
        HalfSplit: Each half's total return, trade count and entry count. The
            halves' equity curves are dropped; an entry count of 0 marks a
            half whose 0.0 return is not a measurement.
    """
    first, second = halves
    h1 = _simulate_at_discount(first, start_date, initial_balance, k=k,
                               spread_band=band, population=f"{population}/H1")
    h2 = _simulate_at_discount(second, start_date, initial_balance, k=k,
                               spread_band=band, population=f"{population}/H2")
    return HalfSplit(
        h1_return=_total_return(h1, initial_balance),
        h2_return=_total_return(h2, initial_balance),
        h1_trades=len(h1.trades),
        h2_trades=len(h2.trades),
        h1_entries=len(first),
        h2_entries=len(second),
    )


def _ex_top_event(
    point: SweepPoint,
    entries: list[dict],
    start_date: date,
    initial_balance: float,
    band: tuple[float, float],
    population: str = "all",
) -> tuple[str, float] | None:
    """
    Measure how much of one scenario's result a single event carried.

    Finds the event whose trades made the largest summed profit on the point
    (by BacktestTrade.event_ticker — market A's event as traded — ignoring
    trades with no event ticker, since an unknown event cannot be shown to be
    one event; ties go to the alphabetically first ticker so the choice is
    deterministic), then RE-SIMULATES the point's entries without every entry
    whose market-A event ticker is that event, from the same initial balance
    at the same k and band. A re-simulation rather than a subtraction of that
    event's P&L: the remaining trades are re-sized against the cash the
    removed ones no longer tie up, and a subtraction is not even bounded below
    by −100%. Measured 2026-09-23 on the DR-73 calibration corpus's same-event
    ladders alone (no band, k = 0.65, $10,000 from 2020-01-01, 71 trades,
    +157.3%): without its top event the subtraction reads +2.5% and the
    re-simulation +14.3%.

    Args:
        point (SweepPoint): The scenario's "all" or "time_series" point.
        entries (list[dict]): The entries that point was simulated from.
        start_date (date): The backtest's start date.
        initial_balance (float): The balance the re-simulation starts from.
        band (tuple[float, float]): The scenario's resolved band (a label).
        population (str): The point's population — "all" (default) or
            "time_series" — naming the re-simulation "<population>/ex-top" on
            its completion line.

    Returns:
        tuple[str, float] | None: (event ticker, total return without it).
            None when no trade on the point names an event — there is no
            event to drop, so no re-simulation runs.
    """
    pnl_by_event: dict[str, float] = defaultdict(float)
    for t in point.trades:
        if t.event_ticker:
            pnl_by_event[t.event_ticker] += t.profit
    if not pnl_by_event:
        return None
    # Largest summed profit first; the ticker breaks exact ties.
    top = min(pnl_by_event, key=lambda ev: (-pnl_by_event[ev], ev))
    # The same market-A event ticker BacktestTrade.event_ticker records —
    # entry["mA"] is _find_entry's canonicalized leg, which is the mA
    # _simulate_at_discount reads — so the trades removed are exactly the
    # ones counted above (plus any same-event entry the point did not trade).
    rest = [rec for rec in entries
            if (rec["entry"]["mA"].get("event_ticker") or "") != top]
    without = _simulate_at_discount(rest, start_date, initial_balance, k=point.k,
                                    spread_band=band, population=f"{population}/ex-top")
    return top, _total_return(without, initial_balance)


def _sweep_from_candidates(
    candidates: _Candidates,
    initial_balance: float,
    *,
    interval_discount: float | None,
    sweep: bool,
    spread_band: tuple[float, float] | None,
    band_sweep: bool,
) -> BacktestSweep:
    """
    Run every entry pass and every simulation of one backtest over one fetch.

    The second half of run_backtest_sweep(): everything after
    _prepare_candidates(). It runs in two phases so the candle series are
    held no longer than today's single-band run holds them.

    Phase 1 — entries. The same-title entries are computed ONCE (a same-title
    pair never reads the band), then, for every band, the time-series
    entries at that band, announced as "Spread band i/N: <floor>-<ceiling>"
    with " (primary)" on the primary band. Each band's entries are its
    time-series entries followed by the shared same-title ones, which is
    exactly the default _entries_for_band() call's scan order. On a band
    sweep the no-band band (0.0, 1.0) is scanned FIRST, over every
    time-series pair, and every other band rescans only the pairs that
    produced an entry there — each band's accepted Mondays are a subset of
    the no-band band's (see the comment at the pre-pass), so this changes no
    band's entries and turns every other band's full scan into a rescan of
    the few pairs that can enter at all. Then the candles and the pair list
    are released (candidates.candles_by_ticker and candidates.all_pairs are
    deleted) before any simulation runs: nothing after Phase 1 reads either,
    and the simulations are where a band sweep spends its time.

    Phase 2 — simulations. The primary band's calibration is measured and
    logged exactly as a single-band run always logged it, and the primary
    (band, k) point is simulated first, so its resolved k can be read back
    off the point and unioned into the k grid (one resolution — the grid can
    never disagree with the point it contains). Every band then gets its own
    calibration (labelled with its own floor) and one "all" simulation per k
    on the SAME k grid, so a band x k table is rectangular. On a band sweep
    each (band, k) also gets a standalone "time_series" (ladders and
    cross-event together, same-title excluded), "ladder" and "cross"
    simulation for each of those populations that is non-empty at that band
    (standalone, never sliced out of the "all" run, so return, drawdown and
    Sharpe are defined for each), and both the "all" and the "time_series"
    point get the split-half check (SweepPoint.halves, split at ONE date,
    the median_low of the primary band's time-series entry dates — a WARNING
    names it when it leaves a half of those entries empty, since the check
    is then not measurable) and the excluding-top-event check
    (SweepPoint.ex_top_event). The same-title entries are then simulated
    alone once (same_title_point).

    The primary point is reused, never re-simulated: it is the same object in
    points and scenarios. With band_sweep False this is exactly the pre-band
    single-band sweep — one calibration, one simulation per k, scenarios [],
    no same_title_point and no split_date — and it logs what that sweep
    logged, plus the "Spread band 1/1" announcement (run_backtest_sweep adds
    the band-source line), with the band and population on each completion
    line.

    Args:
        candidates (_Candidates): _prepare_candidates() output. CONSUMED: its
            candles_by_ticker and all_pairs attributes are deleted after
            Phase 1, so one _Candidates feeds one sweep.
        initial_balance (float): Simulated starting cash balance in dollars;
            every scenario, half and re-simulation starts from it.
        interval_discount (float | None): The primary k, in [0, 1]. None means
            no override, which config.time_series_profit_prob resolves to
            TIME_SERIES_INTERVAL_PROB_DISCOUNT.
        sweep (bool): When True, every band is simulated at every k of
            config.INTERVAL_DISCOUNT_SWEEP unioned with the primary k; when
            False, at the primary k alone.
        spread_band (tuple[float, float] | None): The primary band.
            run_backtest_sweep() has already resolved and validated it before
            the fetch; it is resolved again here only so a direct caller (a
            harness handing in its own _Candidates) gets the same default and
            validation — a no-op on an already-resolved tuple.
        band_sweep (bool): When True, sweep every band of
            config.SPREAD_BAND_SWEEP_FLOORS x SPREAD_BAND_SWEEP_CEILINGS
            (unioned with the primary band) and compute the population,
            split-half and concentration scenarios; when False, the primary
            band alone and none of those.

    Returns:
        BacktestSweep: primary, points (the primary band's k sweep),
            calibration (the primary band's), label_coverage (carried from
            candidates), scenarios, same_title_point, calibrations_by_band,
            same_event_ladders (resolved) and split_date — see BacktestSweep.

    Raises:
        ValueError: From config.time_series_spread_band, if spread_band is not
            a valid band.
        AttributeError: If candidates has already fed a sweep (its
            all_pairs and candles_by_ticker were deleted) — loud rather than
            a silent sweep with no pairs and therefore no entries.
    """
    start_date = candidates.start_date
    primary_band = time_series_spread_band(spread_band)
    # The ladder setting this sweep's pairs were extracted under, resolved the
    # way _extract_pairs and _find_entry resolve the unresolved flag carried on
    # candidates (see _Candidates for when those can disagree), for the report.
    ladders = bool(TIME_SERIES_SAME_EVENT_LADDERS if candidates.same_event_ladders is None
                   else candidates.same_event_ladders)
    if band_sweep:
        # Each grid band through config.time_series_spread_band, so it is
        # validated and normalised exactly like the primary and a grid band
        # equal to the primary is the SAME tuple — the union adds no duplicate.
        grid_bands = {time_series_spread_band((lo, hi))
                      for lo in SPREAD_BAND_SWEEP_FLOORS
                      for hi in SPREAD_BAND_SWEEP_CEILINGS}
        bands = sorted(grid_bands | {primary_band})
    else:
        bands = [primary_band]

    # ── Phase 1: every band's entries, then release the candles ─────────────
    # Same-title entries once: _find_entry never reads the band for them
    # (pinned by TestFindEntrySpreadBand::test_same_title_is_untouched_at_any_band).
    st_entries = _entries_for_band(candidates, primary_band, pair_types=("same_title",))
    if band_sweep:
        logging.info("Same-title candidate entries (band-independent, computed once): %d",
                     len(st_entries))
    # ── The no-band pre-pass (band sweep only) ──────────────────────────────
    # Every band's accepted Mondays are a SUBSET of the no-band band's, pair by
    # pair, because a band only ever tightens _find_entry's per-Monday tests:
    #   * the floor only rises — threshold = min_price_diff_for_gap(gap,
    #     spread_min=floor) = max(tier, floor) >= tier, the no-band threshold
    #     (floor 0.0 is inert under every tier);
    #   * the leg-price-sum ceiling, 1 - threshold, therefore only falls;
    #   * the spread ceiling only drops (1.0, the no-band ceiling, never fires:
    #     both YES asks are banded into [0.01, 0.99]);
    #   * everything else — the leg order, the deadline gap and its cap, the
    #     horizon, the candle lookups, the live-quote checks and the fee check
    #     — never reads the band.
    # Float arithmetic keeps each comparison monotone in the threshold, so no
    # float edge can admit at a band what no band refused. A pair that
    # produced NO entry at (0.0, 1.0) therefore produces none at any band, and
    # rescanning only the pairs that did enter there gives every band exactly
    # the entries a full scan gives it — in the same order, since the subset
    # keeps all_pairs' order (pinned against a full scan per band by
    # TestBandSweepPhaseOneSubset). A single-band run keeps its one full scan.
    no_band = time_series_spread_band((0.0, 1.0))
    no_band_entries: list[dict] | None = None
    rescan: list | None = None
    if band_sweep:
        n_ts_pairs = sum(1 for _, pair_type in candidates.all_pairs
                         if pair_type == "time_series")
        logging.info("No-band pre-pass: scanning all %d time-series pairs at %s",
                     n_ts_pairs, _band_label(no_band))
        no_band_entries = _entries_for_band(candidates, no_band, pair_types=("time_series",))
        # Matched on the legs' TICKERS, not on object identity: whatever
        # _find_entry hands back (the legs it was given, possibly swapped),
        # the two tickers name the pair, and a ticker pair can only ever
        # OVER-include a pair here (a duplicate is rescanned, never lost).
        entered = {frozenset((rec["entry"]["mA"]["ticker"], rec["entry"]["mB"]["ticker"]))
                   for rec in no_band_entries}
        # A new list — candidates.all_pairs itself is never mutated.
        rescan = [item for item in candidates.all_pairs
                  if item[1] == "time_series"
                  and frozenset((item[0][0]["ticker"], item[0][1]["ticker"])) in entered]
        logging.info(
            "No-band pre-pass: %d of %d time-series pairs produced an entry; every "
            "other band rescans only those", len(rescan), n_ts_pairs)

    entries_by_band: dict[tuple[float, float], list[dict]] = {}
    for i, band in enumerate(bands, start=1):
        # Announced BEFORE the pass, so a slow band is attributable while it runs
        logging.info("Spread band %d/%d: %s%s", i, len(bands), _band_label(band),
                     " (primary)" if band == primary_band else "")
        if no_band_entries is not None and band == no_band:
            # This band's full scan IS the pre-pass — never run it twice.
            ts_entries = no_band_entries
        elif rescan is None:
            # A single-band run: the one full time-series _find_entry pass,
            # exactly as before the pre-pass existed.
            ts_entries = _entries_for_band(candidates, band, pair_types=("time_series",))
        else:
            # The time-series _find_entry pass at this band — the only
            # per-band cost of Phase 1 — narrowed to the pairs the pre-pass
            # proved can enter at all. ts + st is the default call's scan
            # order.
            ts_entries = _entries_for_band(candidates, band, pair_types=("time_series",),
                                           _pairs=rescan)
        entries_by_band[band] = ts_entries + st_entries
        if band_sweep:
            logging.info(
                "Prepared %d candidate entries for sizing (%d time-series, %d same-title)",
                len(entries_by_band[band]), len(ts_entries), len(st_entries))
        else:
            # A single-band run keeps the pre-band wording byte-for-byte.
            logging.info("Prepared %d candidate entries for sizing", len(entries_by_band[band]))
    # Nothing below reads a candle or the pair list. Releasing both here keeps
    # the peak at a single-band run's entry-pass peak and, like the old
    # single-band path (whose pair list died with _prepare_entries' locals),
    # holds no pair tuple through the simulations: the entry dicts carry the
    # market records they need (mA/mB) and never a candle. Only the scalar
    # fields — label_coverage, start_date, same_event_ladders — are read after
    # this point. The pre-pass's rescan list is a list of pair tuples too, so
    # it goes with them (its entries already live on in entries_by_band).
    del candidates.candles_by_ticker, candidates.all_pairs
    del rescan, no_band_entries

    # ── Phase 2: simulations ───────────────────────────────────────────────
    primary_entries = entries_by_band[primary_band]
    # Measured from the k-independent entries of the primary band, so it is
    # valid for every k there and is never filtered by any point's Kelly gate.
    # Its floor labels the tiers it applied (a no-op at the default floor 0.0).
    calibration = _interval_calibration(primary_entries, spread_min=primary_band[0])
    # Reported here rather than inside the measurement, mirroring the
    # check_shard_coverage / _log_shard_coverage split. Sub-counts inside the
    # report stay silent at zero, but calibration is None is itself reported
    # with one explanatory line rather than nothing at all (DR-72). Only the
    # primary band's is logged: 36 tables would bury it.
    _log_interval_calibration(calibration)

    # The run's actual result. interval_discount is handed over verbatim —
    # including the None sentinel — so a no-override run prices identically to
    # run_backtest(). Announced before it runs, like every swept point below —
    # its slot used to be unnumbered, so "Sweeping 2/13" was the FIRST counter
    # a reader saw and slot 1 appeared to be missing (TS-21).
    logging.info("Simulating the primary interval discount: k = %s",
                 "config default" if interval_discount is None
                 else f"{interval_discount:.2f}")
    primary = _simulate_at_discount(
        primary_entries, start_date, initial_balance, k=interval_discount,
        spread_band=primary_band, population="all",
    )
    # Read the RESOLVED discount back off the point rather than re-deriving it
    # from the sentinel: one resolution, so the grid membership below cannot
    # disagree with the point it is supposed to contain.
    effective_k = primary.k

    # Union rather than "nearest point": the primary must be an exact member,
    # so an override that is not on the standard grid still gets its own
    # entry. sorted() gives the ascending order BacktestSweep.points promises.
    # The SAME grid for every band, so the band x k table is rectangular.
    grid = sorted(set(INTERVAL_DISCOUNT_SWEEP) | {effective_k}) if sweep else [effective_k]
    if band_sweep:
        logging.info(
            "Re-simulating prepared entries at %d interval discount(s) across %d "
            "spread band(s) (primary k = %.3f, primary band %s)",
            len(grid), len(bands), effective_k, _band_label(primary_band),
        )
    elif len(grid) > 1:
        # A single-band run keeps the pre-band wording byte-for-byte.
        logging.info(
            "Re-simulating %d prepared entries at %d interval discounts (primary k = %.3f)",
            len(primary_entries), len(grid), effective_k,
        )

    # One split date for every scenario, from the PRIMARY band's entries, so
    # every cell's halves cover the same two stretches of history — its
    # TIME-SERIES entries only: the dashboard's banner and heatmap read the
    # time-series population's halves, and a same-title entry (which neither
    # the band nor k ever moves) must not be able to move where they split.
    split_date = None
    if band_sweep:
        primary_ts = [rec for rec in primary_entries if rec["pair_type"] == "time_series"]
        split_date = _split_date(primary_ts, start_date)
        logging.info("Split-half check: entries before %s vs on or after it", split_date)
        # median_low is one of the dates, so H1 (strictly before it) is empty
        # whenever at least half of the entries share the earliest date — and
        # an empty half's 0.0 return is not a measurement. Said here, once,
        # for the band every other band is compared against (another band's,
        # or an "all" point's, halves can also come out empty at this one
        # date, unwarned); the dashboard blanks each such half and leaves it
        # out of its correlation.
        n_h1, n_h2 = (len(half) for half in _split_halves(primary_ts, split_date))
        empty = " and ".join(name for name, n in (("H1", n_h1), ("H2", n_h2)) if n == 0)
        if empty:
            logging.warning(
                "Split-half check: split date %s leaves %s empty; the split-half check "
                "is not measurable for this window (primary band %s: %d time-series "
                "entries before it, %d on or after it)",
                split_date, empty, _band_label(primary_band), n_h1, n_h2)

    points: list[SweepPoint] = []
    scenarios: list[SweepPoint] = []
    calibrations_by_band: dict[tuple[float, float], IntervalCalibration | None] = {}
    for bi, band in enumerate(bands, start=1):
        entries = entries_by_band[band]
        # The primary's is the object already measured and logged above.
        calibrations_by_band[band] = (
            calibration if band == primary_band
            else _interval_calibration(entries, spread_min=band[0])
        )
        if band_sweep:
            # Standalone populations, split once per band (they are
            # k-independent subsets) on _is_ladder_pair — the same rule that
            # labels each trade's same_event_ladder, so a trade and the
            # population it was simulated in always agree. "time_series" is
            # ladders + cross-event together, same-title excluded: the
            # population the band and k actually act on, and the one the
            # dashboard's heatmap and fragility banner read, so a same-title
            # result (band- and k-independent) can never dilute them. A
            # population with no entry at this band is skipped rather than
            # simulated as an empty scenario.
            ladder_flags = [_is_ladder_pair(rec["pair_type"], rec["entry"]["mA"],
                                            rec["entry"]["mB"]) for rec in entries]
            ts_only = [rec for rec in entries if rec["pair_type"] == "time_series"]
            populations = [
                ("time_series", ts_only),
                ("ladder", [rec for rec, is_ladder in zip(entries, ladder_flags, strict=True)
                            if is_ladder]),
                ("cross", [rec for rec, is_ladder in zip(entries, ladder_flags, strict=True)
                           if rec["pair_type"] == "time_series" and not is_ladder]),
            ]

            # Both checked populations' halves, at the ONE split date.
            halves_by_population = {"all": _split_halves(entries, split_date),
                                    "time_series": _split_halves(ts_only, split_date)}
            if len(bands) > 1:
                logging.info("Simulating spread band %d/%d: %s%s (%d entries)",
                             bi, len(bands), _band_label(band),
                             " (primary)" if band == primary_band else "", len(entries))

        for ki, point_k in enumerate(grid, start=1):
            if band == primary_band and point_k == effective_k:
                # Already simulated; reuse the object so BacktestSweep.primary
                # and its entries in points and scenarios are one point.
                point = primary
            else:
                # Each non-primary "all" point announces itself here. Its own
                # completion line — and, on a band sweep, those of its
                # split-half, ex-top and population runs, which follow it
                # unannounced — name the k, band and population, so every
                # completion line is self-describing (TS-21).
                logging.info("Sweeping interval discount %d/%d: k = %.2f", ki, len(grid), point_k)
                point = _simulate_at_discount(
                    entries, start_date, initial_balance, k=point_k,
                    spread_band=band, population="all",
                )
            if band == primary_band:
                points.append(point)
            if not band_sweep:
                continue

            scenarios.append(point)
            # The two robustness checks, set on the "all" point itself (the
            # primary included — same object everywhere it is held) and, below,
            # on the "time_series" point: the dashboard reads the latter's,
            # and keeps the former's for its own "All" row.
            point.halves = _half_split(halves_by_population["all"], start_date,
                                       initial_balance, point_k, band, population="all")
            point.ex_top_event = _ex_top_event(point, entries, start_date,
                                               initial_balance, band, population="all")
            for label, subset in populations:
                if not subset:
                    continue
                pop_point = _simulate_at_discount(
                    subset, start_date, initial_balance, k=point_k,
                    spread_band=band, population=label,
                )
                if label in _CHECKED_POPULATIONS:
                    pop_point.halves = _half_split(
                        halves_by_population[label], start_date, initial_balance,
                        point_k, band, population=label)
                    pop_point.ex_top_event = _ex_top_event(
                        pop_point, subset, start_date, initial_balance, band,
                        population=label)
                scenarios.append(pop_point)

    same_title_point = None
    if band_sweep and st_entries:
        # Once, not per (band, k): same-title entries never read the band and
        # price on the fixed co-resolution prior, never on k. Its k is the
        # primary's and its band None — both nominal.
        logging.info("Simulating the same-title population once (band- and k-independent)")
        same_title_point = _simulate_at_discount(
            st_entries, start_date, initial_balance, k=effective_k,
            spread_band=None, population="same_title",
        )

    return BacktestSweep(
        primary=primary, points=points, calibration=calibration,
        label_coverage=candidates.label_coverage,
        scenarios=scenarios,
        same_title_point=same_title_point,
        calibrations_by_band=calibrations_by_band,
        same_event_ladders=ladders,
        split_date=split_date,
    )


def run_backtest_sweep(
    hist_client: Any,
    live_client,
    start_date: date = date(2024, 1, 1),
    initial_balance: float = 10_000.0,
    use_cache: bool = True,
    max_horizon_days: int | None = None,
    interval_discount: float | None = None,
    sweep: bool = True,
    same_event_ladders: bool | None = None,
    spread_band: tuple[float, float] | None = None,
    band_sweep: bool = False,
) -> BacktestSweep:
    """
    Replay both pair strategies at one interval discount, or at a grid of them —
    and, optionally, across a grid of time-series spread bands.

    The richer sibling of run_backtest(): same simulation, but it also returns
    the empirical-discount calibration and, by default, one full re-simulation
    per discount on config.INTERVAL_DISCOUNT_SWEEP so a report can offer a k
    selector without a re-run. With band_sweep it additionally crosses every
    band of config.SPREAD_BAND_SWEEP_FLOORS x SPREAD_BAND_SWEEP_CEILINGS with
    that k grid and simulates the time-series (ladders + cross-event),
    ladder, cross-event and same-title populations, a split-half check and an
    excluding-top-event check per cell (on the "all" and "time_series"
    points) — the backtest-only scenario explorer. run_backtest() is unchanged and
    remains the two-tuple entry point for every existing caller; this is what
    backtest.py calls when it needs the sweep payload.

    It is the composition _prepare_candidates() + _sweep_from_candidates().
    The expensive half runs ONCE: _prepare_candidates() (fetch, prefilter,
    grouping, pair extraction, candlesticks) depends on neither the band nor
    k. Only the _find_entry pass is repeated per band (the band acts there and
    nowhere else), and only _simulate_at_discount() — Kelly gate, dedups,
    Pass 2, equity curve — per simulated scenario; it must be a full
    re-simulation rather than a re-score: the Kelly gate precedes the
    one-pair-per-group dedup, so a different k changes which candidate wins
    its group, and every surviving candidate then competes for the same
    simulated cash. _interval_calibration() is computed once per band.

    The primary point is simulated at the primary band with the caller's
    interval_discount passed through verbatim, sentinel included, so with no
    override (and the default band) it prices exactly as run_backtest() does
    (k=None is resolved inside config.time_series_profit_prob at call time).
    Its resolved k is then read back off the point and unioned into the sweep
    grid, so the primary is always an EXACT grid member — an
    --interval-discount 0.62 run gets a grid entry at exactly 0.62 rather than
    the nearest standard point — and it is the same object in points (and
    scenarios), never a re-simulated copy. The primary band is likewise
    unioned into the band grid.

    This function never writes config.py. The calibration it reports is a
    recommendation for a human to act on, live sizing keeps reading
    config.TIME_SERIES_INTERVAL_PROB_DISCOUNT no matter what is passed here,
    and no live module reads a spread band at all.

    Args:
        hist_client (Any): Signed client for the historical archive/live endpoints.
        live_client: Client passed through to fetch_all_settled_markets.
        start_date (date): Earliest settlement date to include.
        initial_balance (float): Simulated starting cash balance in dollars.
        use_cache (bool): Whether to reuse the disk-cached assembled market list.
        max_horizon_days (int | None): Optional opt-in bet-horizon cap, passed
            straight through to _prepare_candidates(), which carries it to
            every entry pass. None applies no cap.
        interval_discount (float | None): Interval discount for the primary
            point, in [0, 1]. None (default) means "no override", which
            resolves to config.TIME_SERIES_INTERVAL_PROB_DISCOUNT — the value
            live sizing reads.
        sweep (bool): When True (default), also simulate every discount in
            config.INTERVAL_DISCOUNT_SWEEP. When False, points holds the
            primary alone (and a band sweep simulates each band at the
            primary k only) — the escape hatch for a full-history run where
            the extra passes are not worth their time.
        same_event_ladders (bool | None): Whether two dated cumulative rungs
            of ONE event may pair for this run (DR-73). None (the default)
            resolves this module's TIME_SERIES_SAME_EVENT_LADDERS (bound from
            config at import) at call time — the value the live finder uses.
            Passing it here is the SUPPORTED way to flip ladders for one
            backtest and needs no monkeypatching at all; patching
            config.TIME_SERIES_SAME_EVENT_LADDERS would be a silent no-op.
            Passed straight through to _prepare_candidates(), which hands it
            to pair extraction and carries it, unresolved, to every entry
            pass, so it is band- and k-INDEPENDENT like everything else
            there: it changes which pairs exist, not how any of them is
            priced, and therefore applies identically to every scenario.
            Like --interval-discount, it never reaches live sizing: nothing
            here writes config.py. Its resolved value is recorded on
            BacktestSweep.same_event_ladders.
        spread_band (tuple[float, float] | None): The primary scenario's
            BACKTEST-only time-series spread band (floor, ceiling) on pB − pA.
            None (default) resolves config.BACKTEST_DEFAULT_SPREAD_BAND —
            (0.0, 1.0), no band, the live rule. Resolved and validated at the
            TOP of this function, before anything is logged or fetched, so a
            bad band fails in milliseconds rather than after the fetch.
        band_sweep (bool): When True, also sweep every band of the config
            grid and compute BacktestSweep.scenarios, same_title_point,
            split_date and every band's calibration. False (default) keeps
            this the single-band k sweep it always was.

    Returns:
        BacktestSweep: primary (the effective-discount, primary-band result),
            points (the primary band's k sweep, ascending, always containing
            primary), calibration (the primary band's; None when no
            time-series candidate was measurable), label_coverage (the run's
            outcome-label census, None when the feasibility short-circuit
            skipped the fetch), and the band-sweep payload — scenarios,
            same_title_point, calibrations_by_band, split_date — plus the
            resolved same_event_ladders (see BacktestSweep).

    Raises:
        ValueError: From config.time_series_spread_band, before any fetch, if
            spread_band is not a valid band (0 <= floor < ceiling <= 1).
        TypeError: From config.time_series_spread_band, before any fetch, if
            spread_band is not a pair of numbers.
        KeyError: Propagates out of the candlestick-fetch pool
            (_fetch_candles_parallel) if a ticker needed by a candidate pair
            was not properly excluded by the eligibility prefilter — a real
            defect rather than a ticker with no price history.

    Note:
        When _prepare_candidates()'s Monday feasibility pre-check fails, no
        simulation is possible at any discount or band: the result is a sweep
        holding one empty point (built by the same _simulate_at_discount()
        call every other point comes from, over an empty entry list, so its
        shape, its resolved k and its band stamp cannot drift from a real
        one), calibration=None, label_coverage=None, scenarios=[],
        calibrations_by_band={} and the resolved same_event_ladders. Callers
        therefore need no special case for that path.
    """
    # Resolved and validated FIRST — before anything is logged or fetched: an
    # invalid band is a caller bug, and it must surface in milliseconds, not
    # after a multi-hour fetch. config owns the default and the rule.
    primary_band = time_series_spread_band(spread_band)

    logging.info("Starting backtest from %s with $%.2f", start_date, initial_balance)

    # Logged with its SOURCE, not just its value: a run that reads the config
    # and one that was handed the same value on the command line are different
    # facts about what produced the numbers below, and DR-73's switch is the
    # one input here that changes which PAIRS exist rather than how they are
    # priced. Resolved for the log (and the infeasible return) only —
    # _prepare_candidates takes the sentinel verbatim and resolves it itself,
    # so there is exactly one resolution that the run actually depends on.
    ladders = (TIME_SERIES_SAME_EVENT_LADDERS if same_event_ladders is None
               else same_event_ladders)
    logging.info(
        "Same-event deadline ladders (DR-73): %s (%s)",
        "on" if ladders else "off",
        "config.TIME_SERIES_SAME_EVENT_LADDERS" if same_event_ladders is None
        else "run-level override",
    )
    # The same idiom for the band: the resolved value and where it came from,
    # so a report can never be read as the default band when it was not.
    logging.info(
        "Time-series spread band (backtest only): %s (%s); band sweep %s",
        _band_label(primary_band),
        "config.BACKTEST_DEFAULT_SPREAD_BAND" if spread_band is None
        else "run-level override",
        "on" if band_sweep else "off",
    )

    # The band- and k-independent half — one fetch, one pairing, one candle
    # fetch, reused by every band and every point below. None means the
    # feasibility pre-check failed. The ladder flag rides through unresolved
    # and is carried on the result to every entry pass (DR-73c).
    candidates = _prepare_candidates(
        hist_client, live_client, start_date, use_cache, max_horizon_days,
        same_event_ladders=same_event_ladders,
    )
    if candidates is None:
        # Nothing can be simulated at any discount or band. Build the empty
        # point through the normal path (an empty entry list yields no trades
        # and a flat curve) so it resolves the discount sentinel, stamps the
        # band and shapes its equity curve exactly as every other point does.
        empty = _simulate_at_discount(
            [], start_date, initial_balance, k=interval_discount,
            spread_band=primary_band,
        )
        # label_coverage is None on this path by construction — the fetch was
        # skipped, so nothing was censused. The dashboard renders that as "not
        # measured" rather than as healthy coverage, and it is what tells an
        # empty scenarios list here apart from a band sweep that was off.
        return BacktestSweep(primary=empty, points=[empty], calibration=None,
                             label_coverage=None, scenarios=[],
                             calibrations_by_band={},
                             same_event_ladders=bool(ladders))

    # Every entry pass and every simulation. It deletes the candle series and
    # the pair list itself once the last entry pass is done (before any
    # simulation), so holding `candidates` here pins neither.
    return _sweep_from_candidates(
        candidates, initial_balance,
        interval_discount=interval_discount, sweep=sweep,
        spread_band=primary_band, band_sweep=band_sweep,
    )


# ─── Equity curve construction ────────────────────────────────────────────────

def _build_equity_curve(
    trades: list[BacktestTrade],
    start_date: date,
    initial_balance: float,
) -> pd.DataFrame:
    """
    Construct a daily equity curve DataFrame from the list of backtest trades.

    "portfolio_value" is a PORTFOLIO VALUE, not a cash balance: it is cash plus
    the carrying value of every position still open on that date, where an open
    position is carried at its COST BASIS (total_cost) for its whole holding
    period. So committing capital does not move the curve, and the only two
    moves a trade can make are real economic ones:
      * entry_date: -fees. Cash falls by total_cost + fees while the contracts
        bought with it enter the portfolio at total_cost, so the net step is the
        taker fee alone. Fees are deliberately NOT capitalised into the carrying
        value — they buy nothing that can be sold on, they are realized the
        moment the order fills, and capitalising them would make the trade look
        free on the day it was actually charged.
      * exit_date: +(actual_payoff - total_cost). The position is written off at
        cost and the gross settlement receipt credited, so the step is exactly
        the realized P&L before fees. Summed over both dates a trade moves the
        curve by actual_payoff - total_cost - fees, i.e. its own `profit`.

    Carrying at cost rather than marking to market daily is a deliberate choice
    (DR-61), and NOT because the prices are missing. A true daily mark-to-market
    off each leg's candles would need a per-day quote for every open position on
    every calendar day of the window, and those quotes are already fetched:
    _fetch_candles_parallel requests each leg's WHOLE hourly series (from the
    later of start_date midnight UTC and the market's own open, through one day
    past its close — everything a position in it could need a quote for) and
    disk-caches it per ticker. What is missing is PLUMBING plus a policy — _find_entry returns
    entry-checkpoint prices only, so candles_by_ticker lives on the _Candidates
    object only until the last entry pass (it dies with _prepare_entries on
    run_backtest's path, and _sweep_from_candidates deletes it before its first
    simulation), and mark-to-market means threading a per-day series
    through raw_entries and _simulate_at_discount into this function and
    deciding what to carry on a day a leg has no candle at all. Cost-basis carry
    is the minimal change that makes the derived metrics mean what their labels
    say; do not price the rejected alternative as a new multi-hour fetch. The
    cost of the choice is that an unrealized swing inside the holding period is
    invisible, so drawdown here is REALIZED drawdown and is a lower bound on the
    intraperiod one.

    This matters because the curve is the sole input to every risk figure on the
    dashboard: the "Max Drawdown" KPI, the "Drawdown (%)" chart, _sharpe and
    _sortino (which read the derived "daily_return" column), the per-k sweep
    table's drawdown and Sharpe columns, and the benchmark row that sits in the
    same column as ^GSPC's genuine mark-to-market drawdown. While the curve was
    cash-only an open position was carried at ZERO, so every one of those read
    capital DEPLOYMENT as loss: a real 2026-05-01 run with three trades, all
    three profitable and a +4.8% return, reported a max drawdown of -60.0%, and
    a k=0.40 point reported -100.0% (total ruin) against a final balance of
    $4,655.87. Do not reintroduce cash-only accounting here.

    The curve opens one day before start_date at the untouched initial balance,
    so a trade entering on start_date itself shows its day-0 cost as a real
    pct_change and a real decline from the cummax peak. Without that leading row
    the day-0 step was invisible to both (DR-03), and the per-k sweep table's
    iloc[0] base was the post-outflow balance while the performance card's base
    was initial_balance — one run reported two ways on one page. That guarantee
    is independent of what the day-0 step contains: under DR-61 it is the fees
    rather than the whole stake, and it is still the leading row that keeps the
    cummax peak at initial_balance instead of at the already-charged value.

    Args:
        trades (list[BacktestTrade]): Completed backtest trades with entry_date,
            exit_date, total_cost, fees and actual_payoff populated.
        start_date (date): The first TRADING date of the window; the curve opens
            one row earlier, on start_date - 1 day, at the untouched initial
            balance.
        initial_balance (float): Starting portfolio value in dollars.

    Returns:
        pd.DataFrame: DataFrame with one leading row for start_date - 1 day at
            the initial balance, followed by one row per calendar day from
            start_date to today (UTC) — and, when start_date is itself in the
            future, exactly those two rows — with columns:
            - "date" (date): Calendar date.
            - "portfolio_value" (float): Cash plus open positions at cost, in
              dollars (see above).
            - "daily_return" (float): Fractional daily return (pct_change of portfolio_value).
            Never zero rows: a column-less DataFrame would violate this contract
            and crash the "daily_return" assignment below, as well as every
            .iloc[0]/.iloc[-1] read in dashboard.py.
    """
    # entry_date and exit_date come from UTC-derived timestamps, so use UTC today
    # here as well — otherwise `date.today()` in a non-UTC timezone can drop or add
    # a day around the boundary and misalign the equity curve.
    today = datetime.now(UTC).date()
    # Floored at 1: a start_date after today (reachable through run_backtest /
    # run_backtest_sweep, whose Monday-feasibility short-circuit builds an empty
    # curve for whatever window it was handed) makes the raw span zero or
    # negative. The leading row below already keeps the frame from being the
    # column-less pd.DataFrame([]), so the floor is what guarantees start_date
    # itself is on the axis — the documented future-window shape is exactly the
    # leading row plus start_date.
    span_days = max((today - start_date).days + 1, 1)
    # The curve opens one day BEFORE start_date at the untouched initial
    # balance. start_date itself can carry a Monday-09:00 entry (the default
    # 2024-01-01 is a Monday), and applying that day's charges to the FIRST
    # row hides them from pct_change and cummax entirely: the pre-DR-03 curve
    # reported max drawdown 0.0% and Sortino 0.00 on a run that ended the day
    # with $1.84 of CASH out of its $10,000, and the per-k table's "opening"
    # was the already-charged balance. No trade can enter before start_date, so the
    # leading row is always flat.
    dates = [start_date - timedelta(days=1)] + [
        start_date + timedelta(days=i) for i in range(span_days)
    ]

    # Two accumulators, because a portfolio is cash PLUS whatever is still
    # open. Tracking cash alone carried every open position at zero, which made
    # the curve dive on entry and recover at settlement whether the trade won
    # or lost — deployment reported as drawdown (DR-61, see the docstring).
    cash_changes: dict[date, float] = defaultdict(float)
    position_changes: dict[date, float] = defaultdict(float)
    for t in trades:
        # Cash leaves the portfolio on entry day (contract cost + taker fees)
        cash_changes[t.entry_date] -= t.total_cost + t.fees
        # ...but the contracts it bought are an ASSET held until settlement, so
        # they re-enter the portfolio at cost and only the fees are a realized
        # day-one charge. Fees are deliberately not capitalised: they are gone
        # the moment the order fills and nothing can be sold on for them.
        position_changes[t.entry_date] += t.total_cost
        # At settlement the position is written off at cost and the gross
        # receipt credited, so the step is exactly the realized pre-fee P&L.
        cash_changes[t.exit_date]      += t.actual_payoff
        position_changes[t.exit_date]  -= t.total_cost

    rows = []
    cash = initial_balance
    open_positions = 0.0
    for d in dates:
        # Apply any net change for this day (may be zero if nothing entered/exited)
        cash           += cash_changes.get(d, 0.0)
        open_positions += position_changes.get(d, 0.0)
        rows.append({"date": d, "portfolio_value": cash + open_positions})

    df = pd.DataFrame(rows)
    # Compute fractional daily returns; the leading initial-balance row has no
    # prior day so it gets 0.0, and start_date's own row is the first one that
    # can show a day-0 charge as a real return.
    df["daily_return"] = df["portfolio_value"].pct_change().fillna(0.0)
    return df
