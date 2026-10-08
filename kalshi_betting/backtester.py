"""
File: backtester.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Replays both pair strategies — same-title co-resolution pairs and the
    directional time-series bet — on the full history of settled Kalshi markets.
    Groups settled markets into potential time-series and same-title pairs, fetches
    hourly candlestick price series for each involved market, then scans weekly
    Monday snapshots to find every date each pair was tradeable at the required
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
    (the single definition of WHY a candidate is not one; built on the
    same three verdicts cumulative_deadline_pair reads, so the boolean and the
    reason can never disagree) and its
    REFUSED_SNAPSHOT/REFUSED_NO_STATED_DEADLINE/REFUSED_SAME_DEADLINE
    constants, and stated_deadline / same_event_ladder with its SAME_DAY
    sentinel (the single definition of which calendar day a rung's deadline
    names and of how two rungs of one event are ordered and gapped, so the
    live and backtest ladder rules can never disagree), and
    closes_apart (the single definition of the same-title close gate,
    so the two paths can never disagree about which cross-series pair closes
    at one moment) with close_gap_bound_text (the bound its refusal line
    prints, read from the same scanner binding the gate reads, so the line
    stays the live one's verbatim twin), and ladder_keys (which ladders a
    market is on, shared with the live rule), from
    scanner.py; fee/model helpers
    (fee_leg_exact, fee_per_pair_approx, min_price_diff_for_gap — whose
    spread_min and tier_floors keywords apply this module's bands and
    tier-floors-off family — time_series_profit_prob, and
    time_series_mid_spread, the mid spread the Kelly gate prices each
    time-series Monday at, _candidate_pair gives each time-series
    candidate at its Monday's quotes and _interval_calibration measures
    k-hat against at each pair's first qualifying Monday), the
    spread-band helpers
    time_series_spread_band and time_series_spread_too_wide (which
    _find_entry applies to time-series candidates; the first also validates
    and resolves a band up front in _entries_for_band, run_backtest_sweep,
    _sweep_from_candidates and _simulate_at_discount's completion line; no
    live module calls either except through config's own live helpers, and
    TestLiveBacktestSpreadParity pins _find_entry to the live spread rule),
    plus BUDGET_FRACTION and SAME_TITLE_SIZE_CAP (bound by value),
    pair_size_cap (one definition, shared with live sizing),
    held_pair_fraction (the one definition of an add-on's size, shared with
    live sizing), LiveSettings (one per simulation, handed to the live
    sizer), _step_cap (the size-cap grid LiveSettings accepts),
    live_defaults with its LiveDefaultsError / LiveDefaultsMissing refusals,
    describe_time_series_rule, _names_text, _exact_number and _cap_text (the
    reporting-only "Live time-series rule" line, which names the saved live
    defaults),
    CANDLESTICK_FETCH_MAX_WORKERS, CANDLESTICK_PERIOD_INTERVAL_MINUTES (the
    candle length, the grid _candle_window_open and _checkpoint_floor round
    down to),
    LARGE_GROUP_WARN_THRESHOLD,
    INTERVAL_DISCOUNT_SWEEP and the band grid SPREAD_BAND_SWEEP_FLOORS /
    SPREAD_BAND_SWEEP_CEILINGS (both read only by _sweep_from_candidates;
    the size-cap grid SIZE_CAP_SWEEP is defined HERE instead, see Notes),
    MAX_DEADLINE_GAP_DAYS, SAME_TITLE_CO_RESOLVE_PROB, SAME_TITLE_MIN_PRICE_DIFF,
    SCHEDULED_RUN and ScheduledRun (the weekly live run, whose UTC moments
    are the entry checkpoints; tests patch backtester.SCHEDULED_RUN, never
    config.*), SETTLED_PREFILTER_CACHE_TAG (the prefilter's version name),
    SHORT_DEADLINE_GAP_DAYS, TIME_SERIES_INTERVAL_PROB_DISCOUNT and
    TIME_SERIES_SAME_EVENT_LADDERS from config.py; fetch_all_settled_markets(),
    fetch_candlesticks(), and infer_category() from historical.py, plus its
    SettledCorpus (read by TYPE, to take the corpus's provenance) and
    CorpusProvenance (carried out on
    BacktestSweep.corpus_provenance and re-exported to dashboard.py, which
    imports only from here). Also
    depends on pandas (external) for the equity-curve DataFrame and numpy
    (external, a declared dependency pandas already pulls in) for counting
    the grouping-key hashes behind the groupable subset. Sizes every trade
    with the live bot's own code: scanner._enrich_pair walks a synthetic
    order book (depth_model.book, from the DepthModel the caller hands in)
    and strategy.compute_trade picks the count, through scanner's
    CandidatePair, HeldPair, _market_from_dict and leg_prices. The Kelly
    gate in Pass 1 still copies the formula, at the top of the book, with
    the forecast read from that Monday's mid spread as live sizing reads it.
    Exports BacktestTrade, HalfSplit, SaleCheck, SweepPoint,
    CalibrationObservation, IntervalCalibrationBucket, IntervalCalibration,
    OutcomeLabelCoverage, CapSweep, SIZE_CAP_SWEEP and BacktestSweep
    (BacktestTrade, BacktestSweep,
    IntervalCalibration, OutcomeLabelCoverage and SweepPoint are consumed by
    dashboard.py, which also imports the private helpers _exact_label,
    _paid_prices (what a trade paid, which its trade rows and Kelly
    scatter read), _build_equity_curve — the one definition of an equity
    curve, which its page-wide filter runs over a category's or tag's trades
    for that slice's curve — _calibration_bucket, _band_label (the bare
    "floor-ceiling" its filter bar and scenario explorer name a band's
    tier-floors-off run with, so the page and the log spell that run alike),
    _tier_floors_bind (the one test of whether a deadline-gap tier binds
    at a band, which the page's Tier floors views — the filter bar's and the
    scenario explorer's — read to decide whether a band absent from the
    tier-off family may show its tier-on run, or no off view is shown),
    _cap_percent (the size-cap option formatter), and the pieces of the live
    rule's report that dashboard._live_rule_html shares with this module's
    log line — _live_rule_view with its _LIVE_RULE_PRIMARY and
    _LIVE_RULE_NOT_SIMULATED verdicts, _live_rule_ladder_note,
    _live_filter_text, _live_sizing_note, _live_add_on_note,
    _LIVE_RULE_LABEL and _LIVE_RULE_NONE; an
    IntervalCalibration carries the CalibrationObservations its pooled row
    was reduced from, so a report can regroup that population through
    _calibration_bucket, the one definition of the k-hat arithmetic, as
    dashboard.py's k-hat breakdown does by category, tag and spread band)
    plus run_backtest() and run_backtest_sweep() (called by backtest.py).

Notes:
    The backtester works in two passes. A pair can pass the entry checks on
    several Mondays. Pass 1 scores each of them with the Kelly rule (the
    formula that sets how much to bet) at the simulated interval discount k:
    a time-series pair keeps every Monday with a positive Kelly fraction, a
    same-title pair only its first (with add_to_held, also its later Mondays
    with the same legs, as add-ons), and only the best same-title pair per
    title group survives. Pass 1 also drops a time-series candidate whose
    tickers were found as a same-title pair, as the live run does.

    Pass 2 walks the candidates in date order, best expected return first
    within a date, with a running cash balance. Each trade is sized by the
    live sizer, strategy.compute_trade, on that Monday's opening portfolio
    value (the cash after that day's pay-outs plus every open trade at
    market: each leg at the latest usable ask of the side it holds, at its
    payout once paid out; a trade with no quotes at its cost) and the cash
    left, so it bets a share of the value but never spends more than the
    cash. With a depth model it first walks a synthetic order book
    (scanner._enrich_pair), as a live run walks the real one, so a larger
    trade pays a worse average price; without one, or with no volume data
    for a leg that Monday, it fills at the candle prices (the top of the
    book). A candidate is skipped if not one contract pair fits or a win no
    longer pays after fees. A market is in the open trades of at
    most one pair at a time (freed when that pair pays out; an add-on to a
    held pair is another trade of that pair, with the same pay-out day), and
    at most one time-series pair is open per ladder (one question at several
    deadlines; a market's ladders are freed on the day it pays out). Each
    pair opens at most once; a time-series pair that cannot be taken one
    Monday is tried again on its next passing Monday. The live run values
    open positions at Kalshi's own price, the backtest at the held side's
    latest usable ask on its candles (_open_value); the two can differ either
    way.

    The prices that value an open trade come from the candles the backtest
    already fetched: _attach_leg_quotes (the one writer) samples each traded
    market into a LegQuotes on the entry records before the candles are
    released, Pass 1b (the one reader) hands them to each trade as
    BacktestTrade.marks, Pass 2 values open trades at each checkpoint through
    _open_value, and the equity curve values them at each day's end through
    _open_value_path (read by _carry_steps and _value_steps, which the
    dashboard reads too). A record without quotes (every hand-built one)
    values its trades at cost: each is carried at what it cost from its
    entry day to its pay-out day.

    A simulation run with add_to_held (off unless a caller asks) may also add
    to a pair it still holds, as the live sizer does for a held pair: on a
    later passing Monday, while both of its markets are unpaid, the pair
    trades again as a new trade of its own, sized through
    config.held_pair_fraction (the live sizer's rule: Kelly sizes the whole
    position as a share of the portfolio value, counting the pair's open
    trades at market plus the fees paid for them, and an add-on never stakes
    more than a new pair would). It is refused while any other
    open trade shares one of its ladders, as live adds only to a held pair no
    other held market shares a ladder with.

    The work is split at two boundaries. _prepare_candidates() is the half
    that depends on neither the backtest's time-series spread band nor the
    interval discount k — fetch, prefilter, census, grouping, pair extraction
    and candlesticks — returned as a _Candidates. _entries_for_band() is the
    _find_entry sweep at ONE spread band (the band acts only there; it holds
    no probability model, so its entries are k-independent).
    _simulate_at_discount() is everything that reads k — the Kelly gate, the
    dedups, Pass 2 and the equity curve — and returns a SweepPoint stamped
    with the resolved discount, band, population and tier-floor setting.
    run_backtest() is a thin
    wrapper over _prepare_entries() (_prepare_candidates() plus one
    _entries_for_band() pass at the default band, which is no band at all)
    and one _simulate_at_discount() call with k=None, which
    config.time_series_profit_prob resolves to
    config.TIME_SERIES_INTERVAL_PROB_DISCOUNT.

    _interval_calibration() measures the EMPIRICAL discount from one band's
    k-independent entries — the realised in-between rate divided by the mean
    market-implied gap at the midpoints (the mid spread, which the forecast's
    k multiplies too), pooled and per deadline-gap band — and
    _log_interval_calibration() reports it. Because it reads the prepared
    entries rather than _simulate_at_discount(), it is never filtered by the
    Kelly gate, which is what stops the estimate confirming whatever k
    produced it. It is a RECOMMENDATION ONLY: nothing here writes config.py
    or the saved live defaults, and live sizing reads its k from its run's
    config.LiveSettings, built from the saved live defaults.

    run_backtest_sweep() is the entry point that exposes all of that:
    _prepare_candidates() once, then _sweep_from_candidates() — one
    _find_entry pass per band, one calibration per band, and one
    _simulate_at_discount() per discount on config.INTERVAL_DISCOUNT_SWEEP
    (unioned with the caller's own, so the primary is always an exact grid
    member), returned as a BacktestSweep. With band_sweep it crosses every
    band of config.SPREAD_BAND_SWEEP_FLOORS x SPREAD_BAND_SWEEP_CEILINGS with
    that k grid and adds standalone time-series (ladders + cross-event),
    ladder, cross-event and same-title populations, a split-half check and an
    excluding-top-event check — the backtest-only scenario explorer.
    run_backtest() keeps its own signature and two-tuple return.

    With tier_off_sweep as well (backtest.py turns it on with the band
    sweep), every band where a deadline-gap tier floor binds
    (_tier_floors_bind: a floor below a tier — on the shipped grid floors 0,
    0.20 and 0.25, 18 of the 36 bands) is entered and simulated a second
    time with the tiers not applied — config.min_price_diff_for_gap's
    tier_floors=False, so that band's floor alone gates pB − pA
    (which must still be strictly positive) and sets the leg-price-sum
    ceiling — through the same scenario block, and returned beside the
    tier-on grid as BacktestSweep.tier_off_scenarios /
    tier_off_calibrations_by_band. Every tier-on figure is exactly what the
    run returns without it; a band at or above both tiers is not re-run,
    since it enters the same pairs either way.

    The per-trade Kelly size cap is a simulation parameter too:
    _simulate_at_discount(..., size_cap=None) sizes every candidate at
    min(config.pair_size_cap(pair type, size_cap, SAME_TITLE_SIZE_CAP), f*),
    None meaning this module's BUDGET_FRACTION.

    With cap_sweep, run_backtest_sweep also returns a CapSweep over every other
    cap of SIZE_CAP_SWEEP (defined here, not in config.py), seeded from the
    eager points (caps at or above a point's peak share one simulation; with
    a walked book, at or above its cap_free_from too) and
    simulated lazily, one (band, k) cell at a time. With tier_off_sweep too, a SECOND CapSweep
    (BacktestSweep.tier_off_cap_sweep, tier_floors False) does the same over
    the tier-floors-off family's binding bands, seeded from that family's own
    points — so every tier x band x k x cap scenario is a real simulation.
    The two never share a seed: the tier-on CapSweep is seeded from tier-on
    points only and the tier-off one from tier-off points only. Live sizing
    never reads any of it.

    With add_on_sweep, run_backtest_sweep also returns the dashboard's "Add to
    held pairs" family: two more lazy CapSweeps (BacktestSweep.add_on_cap_sweep
    over the tier-on entries and add_on_tier_off_cap_sweep over the
    tier-floors-off family's binding bands) whose every simulation adds to
    held pairs, for the "all" population only and at every cap of the grid.
    No eager run of this module adds to held pairs, so they have no eager
    point to start from: every cell is simulated when a report reads it,
    ending its curves on the day its eager twin's curve ended. The run itself
    simulates nothing extra, and every figure it reports is unchanged by the
    flag.

    With sell_sweep, it also returns the dashboard's "Sell" family
    (BacktestSweep.sell_sweep, a SellSweep): every level of
    config.TAKE_PROFIT_LEVELS — sell a whole position once its realized
    profit has stayed at or above that share of its potential profit for
    config.TAKE_PROFIT_HOLD_DAYS days in a row (_simulate_at_discount's
    sell_at) — each with every minimum of config.TAKE_PROFIT_MIN_DAYS — sell
    only while at least that many days remain before the position's last
    market stops trading (sell_min_days) — over every scenario the filter bar
    shows, tier floors on and off, adding to held pairs or not. Like the
    add-on family it simulates nothing during the run; each cell is simulated
    when the dashboard reads it. Every selling run records each sale it makes
    as a SaleCheck on SweepPoint.sales, which is how SellSweep.sold_grid
    simulates only the (level, minimum) settings that differ: one no position
    of the no-selling run would sell at is that run (_sale_reach), and one at
    least as strict as a run it already simulated, when every sale of that run
    also meets it, is that run (_sale_cover, yielded as a SameSale). The
    sell rule's arithmetic lives where live code may import it, and the
    backtest reads it there through its own one-line wrappers: the test at
    each check (config.take_profit_reached, through _sells_at), the days to
    maturity (config.days_to_maturity, through _days_left), the walk down a
    bid ladder (scanner.walk_bids, through _ladder_average) and the candle
    bid rule (historical.usable_candle_ask and candle_sale_bids, through
    _usable_ask and _leg_quotes).

    An ENTRY CHECKPOINT is a moment at which the backtest may open a
    simulated trade: the live bot's weekly run time (config.SCHEDULED_RUN,
    Monday 09:00 America/Los_Angeles) on each run weekday. The backtest keeps
    UTC dates, so _prepare_candidates() refuses, before fetching, a schedule
    whose run does not fall once on its own UTC date (ScheduledRun.date_problems).

    Before grouping, _prepare_candidates() filters markets through
    _can_ever_enter(), a cheap per-market PREFILTER that drops only markets no
    entry checkpoint can reach, given Kalshi's observed candle behaviour (that
    function gives the argument): they can never appear in any entered pair,
    as either leg, in either pair type, so dropping them up front avoids
    materializing them into any group at all. A normalized-title group can
    hold tens of thousands of markets (intraday crypto ladders), and without
    the prefilter and _extract_pairs' close-time window, pair extraction is
    O(n^2) per group and infeasible. Neither optimization changes results:
    both only skip work that cannot produce an entry.

    Only the GROUPABLE eligible records are then grouped, and this
    module builds no list of every eligible record of its own: the corpus is
    walked twice. The first walk (_index_eligible_keys) hashes each eligible
    record's time-series and same-title grouping keys — through
    _ts_group_key/_st_group_key, the very helpers the two grouping functions
    group on — and the second (_materialize_groupable) keeps only the records
    whose key hash is shared with another eligible record. Both groupings
    drop every single-member group, so a record sharing neither key can never
    appear in any pair. The subset holds every member of every group
    of two or more, in order, so the groups and pairs are exactly those of
    the whole eligible list; a hash collision can only keep an extra record,
    which the exact grouping then drops. The corpus itself is whatever
    historical.fetch_all_settled_markets returns — a disk-backed
    historical.SettledCorpus that streams the assembled
    settled_markets_*.jsonl.gz cache afresh on each walk, so the eligible set
    is never resident as a whole (a legacy settled_markets_*.json cache, the
    one format that was handed over as a list, is no longer served). The
    corpus must re-iterate identically, and a second walk that
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
    counted, and one summary WARNING reports the count. Kalshi also lists
    snapshot-style markets ("on <date>"), which the normalized-title grouping
    can put in one group with cumulative ones; _extract_pairs refuses such a
    pair on its WORDING (scanner.cumulative_deadline_pair, shared with the
    live finder), so this counter is a backstop behind a text heuristic. A
    non-zero count most likely means a wording false negative, though legs
    that nest but were ordered on an early REALIZED close, or strike-blind
    grouping on a cache without subtitles, can also produce it.
"""
import functools
import hashlib
import logging
import math
import numbers
import resource
import statistics
import sys
from array import array
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfoNotFoundError

import numpy as np
import pandas as pd

from .config import (
    BACKTEST_MARKETS_RAM_WARN,
    BACKTEST_OUTCOME_LABEL_WARN_FRACTION,
    BACKTEST_RECORD_BYTES_ESTIMATE,
    BUDGET_FRACTION,
    CANDLE_NO_ASK_CEILING,
    CANDLESTICK_FETCH_MAX_WORKERS,
    CANDLESTICK_PERIOD_INTERVAL_MINUTES,
    CONTRACT_PAYOUT_DOLLARS,
    INTERVAL_DISCOUNT_SWEEP,
    LARGE_GROUP_WARN_THRESHOLD,
    MAX_DEADLINE_GAP_DAYS,
    PRICE_EPSILON,
    SAME_TITLE_CO_RESOLVE_PROB,
    SAME_TITLE_MIN_PRICE_DIFF,
    SAME_TITLE_SIZE_CAP,
    SCHEDULED_RUN,
    SETTLED_PREFILTER_CACHE_TAG,
    SHORT_DEADLINE_GAP_DAYS,
    SPREAD_BAND_SWEEP_CEILINGS,
    SPREAD_BAND_SWEEP_FLOORS,
    TAKE_PROFIT_HOLD_DAYS,
    TAKE_PROFIT_LEVELS,
    TAKE_PROFIT_MIN_DAYS,
    TIME_SERIES_INTERVAL_PROB_DISCOUNT,
    TIME_SERIES_SAME_EVENT_LADDERS,
    LiveDefaultsError,
    LiveDefaultsMissing,
    LiveSettings,
    ScheduledRun,
    _cap_text,
    _exact_number,
    _names_text,
    _step_cap,
    days_to_maturity,
    describe_time_series_rule,
    fee_leg_exact,
    fee_per_pair_approx,
    held_pair_fraction,
    live_defaults,
    max_kelly_fraction,
    min_price_diff_for_gap,
    pair_size_cap,
    take_profit_reached,
    time_series_mid_spread,
    time_series_profit_prob,
    time_series_spread_band,
    time_series_spread_too_wide,
)
from .depth_model import DepthModel, bid_ladder, can_start_at, volume_24h
from .depth_model import book as _synthetic_book
from .historical import (
    CorpusProvenance,
    SettledCorpus,
    candle_sale_bids,
    fetch_all_settled_markets,
    fetch_candlesticks,
    infer_category,
    series_ticker,
    usable_candle_ask,
)
from .scanner import (
    DEADLINE_CUMULATIVE,
    DEADLINE_SNAPSHOT,
    DEADLINE_UNKNOWN,
    ENRICH_UNAFFORDABLE,
    ENRICH_UNPROFITABLE,
    REFUSED_NO_STATED_DEADLINE,
    REFUSED_SAME_DEADLINE,
    REFUSED_SNAPSHOT,
    SAME_DAY,
    CandidatePair,
    HeldPair,
    _cash_binds,
    _enrich_pair,
    _market_from_dict,
    close_gap_bound_text,
    closes_apart,
    cumulative_deadline_pair,
    deadline_pair_refusal,
    deadline_profile,
    event_series,
    ladder_keys,
    leg_prices,
    leg_sides,
    same_event_ladder,
    stated_deadline,
    time_series_group_key,
    walk_bids,
)
from .strategy import TradeSpec, compute_trade

# Seconds in one UTC day. Same value as historical._DAY_SECONDS, kept local
# rather than importing a private name.
_DAY_SECONDS = 86_400

# How far past today _prepare_candidates checks SCHEDULED_RUN's run dates
# (ScheduledRun.date_problems): ten years, since a market's close_time, and so
# the checkpoints the prefilter compares it with, can lie years ahead.
_SCHEDULE_CHECK_DAYS_AHEAD = 3_653

# The top of the range historical.fetch_candlesticks clamps a candle's NO ask
# into (config.CANDLE_NO_ASK_CEILING, the one definition): a NO ask at this
# value is no usable quote (_usable_ask), so the NO leg keeps its last usable
# NO ask instead.
_CANDLE_NO_ASK_CEILING = CANDLE_NO_ASK_CEILING

# How old, in days, the candle behind a leg's day-end value may be before
# _attach_leg_quotes counts the leg on its DEBUG line of legs valued on old
# quotes. Reporting only: no quote is ever refused for its age.
_STALE_QUOTE_DAYS = 7

# The most days the sell rule may span (TAKE_PROFIT_HOLD_DAYS): one week. A
# sale at an entry checkpoint then looks back at most six days, all after the
# previous checkpoint, so every trade the position holds at the sale was
# already held at each daily check (trades are bought only at checkpoints);
# the rule values those trades at every check.
_HOLD_DAYS_MAX = 7

# date.toordinal() of 1970-01-01, so a UTC midnight's Unix time is
# (ordinal - _EPOCH_ORDINAL) * _DAY_SECONDS.
_EPOCH_ORDINAL = date(1970, 1, 1).toordinal()

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

# The per-trade size caps a size-cap sweep offers the dashboard: 5% steps to
# 95%, then 1.0 — NO cap. Kelly's f* = p - q/b <= p <= 1 (time_series_profit_prob
# is 1 - k*max(0, mid spread) with k in [0, 1]; same-title prices at the fixed
# SAME_TITLE_CO_RESOLVE_PROB), so min(1.0, f*) is f* itself and "100%" and
# "off" are one run. A same-title candidate also stays under
# SAME_TITLE_SIZE_CAP at every cap (config.pair_size_cap). BACKTEST-ONLY:
# strategy.py never imports this module. Rounded to two decimals, so 0.2
# is a member by value; the run's own cap is unioned in anyway (CapSweep), as
# the k grid unions its primary. Lives here rather than in config.py beside
# INTERVAL_DISCOUNT_SWEEP by the operator's instruction for this change (only
# the dashboard and backtest code may move) — the one named exception in
# CLAUDE.md's constants rule. Read only by _sweep_from_candidates.
SIZE_CAP_SWEEP: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 20)) + (1.0,)


def _validated_cap(value: Any, name: str) -> float:
    """
    Check one per-trade size cap and return it as a builtin float.

    The cap must be a real number in (0, 1] (numpy numbers included). NaN, a
    bool and anything that is not a real-number type (e.g. a string or a
    Decimal) are refused; a wrong type gets a message naming the type.

    Args:
        value (Any): The cap, the largest share of the portfolio value one trade may bet.
        name (str): The cap's name, for the error message.

    Returns:
        float: The cap, as a builtin float.

    Raises:
        ValueError: If value is a bool, not a numbers.Real, or not in (0, 1].
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(
            f"{name} must be a real number in (0, 1], got {value!r} "
            f"({type(value).__name__})")
    resolved = float(value)
    if not 0.0 < resolved <= 1.0:
        raise ValueError(f"{name} must be in (0, 1], got {value!r}")
    return resolved


def _resolve_size_cap(size_cap: float | None) -> float:
    """
    Return the checked per-trade size cap one simulation sizes under.

    None reads this module's BUDGET_FRACTION when called, not when defined,
    so a test that patches backtester.BUDGET_FRACTION takes effect. The cap
    must sit on config.SIZE_CAP_STEP's grid, as LiveSettings requires: each
    simulation hands it to the live sizer.

    Args:
        size_cap (float | None): The cap, a share in (0, 1]; None for BUDGET_FRACTION.

    Returns:
        float: The resolved cap, as a builtin float on the grid.

    Raises:
        ValueError: If the resolved cap is a bool, not a numbers.Real, NaN, outside
            (0, 1], or off the grid.
    """
    cap = _validated_cap(BUDGET_FRACTION if size_cap is None else size_cap, "size_cap")
    # config's own grid rule, which names the value it refuses
    return _step_cap(cap, "size_cap")


def _resolve_same_title_size_cap() -> float:
    """
    Resolve and validate the extra per-trade cap on same-title pairs.

    Reads this module's by-value SAME_TITLE_SIZE_CAP at CALL time (patch
    backtester's). Checked because a bad value would not fail otherwise:
    pair_size_cap's min() reads NaN as no cap and 0 or below as no same-title
    trade. It must sit on config.SIZE_CAP_STEP's grid too, as LiveSettings
    requires.

    Returns:
        float: SAME_TITLE_SIZE_CAP, as a builtin float in (0, 1] on the grid.

    Raises:
        ValueError: As _validated_cap, or for a cap off the grid.
    """
    cap = _validated_cap(SAME_TITLE_SIZE_CAP, "SAME_TITLE_SIZE_CAP")
    return _step_cap(cap, "SAME_TITLE_SIZE_CAP")


def _cap_percent(cap: float) -> str:
    """
    Render a size cap as a percentage, injectively.

    The cap's shortest round-trip decimal (repr) shifted two places, in plain
    notation with trailing zeros dropped: every SIZE_CAP_SWEEP member reads as
    it is written ("5", "20", "55", "100"), while an off-grid cap keeps every
    digit that tells it apart (0.19999999999999998 -> "19.999999999999998",
    never "20"). The shift is exact decimal arithmetic, so two different caps
    can never print alike — the property a completion-line prefix needs
    (TS-21). A float product such as repr(cap * 100) is NOT injective: cap *
    100 rounds, and neighbouring doubles (0.003 and 0.0029999999999999996)
    land on one product.

    Args:
        cap (float): A cap in (0, 1] — a builtin float or any float
            subclass (numpy's float64 included), rendered by the shortest
            round-trip digits of its builtin float value.

    Returns:
        str: The percentage, without the "%" sign.
    """
    # repr of the BUILTIN float: a float subclass such as numpy's float64
    # reprs as "np.float64(0.35)", which Decimal cannot parse
    return format((Decimal(repr(float(cap))) * 100).normalize(), "f")


def _cap_label(cap: float) -> str:
    """
    Name a resolved size cap for a completion line or a log summary.

    Args:
        cap (float): A resolved cap in (0, 1].

    Returns:
        str: "no cap" for 1.0 (a same-title pair still stays under
            SAME_TITLE_SIZE_CAP), else "cap <percent>%" through _cap_percent.
    """
    return "no cap" if cap >= 1.0 else f"cap {_cap_percent(cap)}%"


def _sim_options(size_cap: float | None, quiet: bool, *,
                 end_date: date | None = None,
                 tier_floors: bool = True,
                 add_to_held: bool = False,
                 sell_at: float | None = None,
                 sell_min_days: int | None = None) -> dict:
    """
    Build the keyword arguments a sweep helper forwards to _simulate_at_discount.

    ONLY the options that differ from _simulate_at_discount's defaults are
    included, so every default call is made with exactly the keywords it
    carried before the size cap and the tier-floors-off family existed — the
    test stand-ins with fixed six-parameter signatures (tests/test_backtester.py's
    golden_band_sweep and cap_sweep_run simulate_spy, and the fake_simulate of
    TestBandSweepSplitAndPopulationWiring, TestTimeSeriesPopulation and
    TestBandSweepPhaseOneSubset's tier-on sweep) and the verbatim kwargs
    assertions in
    TestSweepHelpers::test_ex_top_event_picks_the_largest_summed_profit keep
    working. The other stand-ins accept only the options their runs forward:
    the tier-off sweeps' take tier_floors, while TestCapSweepSeeding's and
    TestCapSweepEndDate's take only size_cap/quiet[/end_date], so a stray
    tier_floors raises there too. A cap equal to this module's BUDGET_FRACTION
    (read at call time) means the same as None and is dropped too, and
    tier_floors is forwarded only when it is exactly False — the one value
    that marks a tier-off simulation — so a tier-on call never carries it.
    _half_split, _ex_top_event and _band_sweep_cell build their keywords
    here, and so do CapSweep's split-half and excluding-top-event checks
    (through those two helpers) and CapSweep._by_cap's own cap simulation —
    which builds quiet, end_date and tier_floors here (as
    _sim_options(None, True, end_date=..., tier_floors=...)) but passes
    size_cap EXPLICITLY beside them: dropping a cap equal to BUDGET_FRACTION
    would hand the simulation None, which resolves BUDGET_FRACTION again at
    call time — the same value unless that binding moved after the run, when
    the cell would silently size at a cap its caps tuple does not name — and
    would change the keywords TestCapSweepSeeding's stand-ins record. So a
    tier-on CapSweep calls the simulator with exactly the keywords it always
    did (size_cap, quiet, and end_date when pinned), and a tier-off one adds
    tier_floors=False. add_to_held is forwarded only when it is exactly True
    (a CapSweep that adds to held pairs), and sell_at and sell_min_days each
    only when it is set (a CapSweep that sells early, with or without a
    minimum of days before maturity), so every other call is unchanged.

    Args:
        size_cap (float | None): The cap the caller simulates under; None or
            BUDGET_FRACTION for the default.
        quiet (bool): Whether the simulation's completion line and
            premise-violation WARNING go to DEBUG.
        end_date (date | None): Keyword-only. The day the simulation's
            equity curve must end on (a lazy size-cap run pins its eager
            point's); None (default) leaves _build_equity_curve reading
            today (UTC) and is not forwarded.
        tier_floors (bool): Keyword-only. Whether the entries were detected
            with the deadline-gap tier floors applied; forwarded (as False)
            only when it is exactly False. Default True, not forwarded.
        add_to_held (bool): Keyword-only. Whether the simulation may add to a
            pair it still holds (see _simulate_at_discount); forwarded (as
            True) only when it is exactly True. Default False, not forwarded.
        sell_at (float | None): Keyword-only. The share of potential profit
            at which the simulation sells a position early (see
            _simulate_at_discount); forwarded only when it is not None.
            Default None, not forwarded.
        sell_min_days (int | None): Keyword-only. The fewest days a position
            must have left before its last market stops trading for the
            simulation to sell it (see _simulate_at_discount); forwarded only
            when it is not None. Default None, not forwarded.

    Returns:
        dict: {} on a default call; otherwise "size_cap", "quiet",
            "end_date", "tier_floors" (always False when present),
            "add_to_held" (always True when present), "sell_at" and/or
            "sell_min_days".
    """
    out: dict = {}
    if size_cap is not None and size_cap != BUDGET_FRACTION:
        out["size_cap"] = size_cap
    if quiet:
        out["quiet"] = True
    if end_date is not None:
        out["end_date"] = end_date
    if tier_floors is False:
        out["tier_floors"] = False
    # Only when on, so an ordinary call carries no extra keyword
    if add_to_held is True:
        out["add_to_held"] = True
    if sell_at is not None:
        out["sell_at"] = sell_at
    if sell_min_days is not None:
        out["sell_min_days"] = sell_min_days
    return out

# ─── Data structures ──────────────────────────────────────────────────────────

def _carried_forward(values: Any) -> np.ndarray:
    """
    A float array with every NaN after a number replaced by the last number before it.

    NaNs before the first number stay NaN. LegQuotes stores its samples this
    way, so a leg with no usable quote on some day counts at its last usable
    one.

    Args:
        values: A sequence of floats (NaN for no usable ask).

    Returns:
        np.ndarray: A new float array of the same length.
    """
    array = np.array(values, dtype=float)
    if array.size:
        known = ~np.isnan(array)
        # The index of the last number at or before each position
        last = np.maximum.accumulate(np.where(known, np.arange(array.size), 0))
        array = np.where(np.logical_or.accumulate(known), array[last], np.nan)
    return array


class LegQuotes:
    """
    One market's asks over the days a backtest trade in it can be open.

    The backtest values each open leg at the ask of the side it holds. This
    holds one market's asks, from candles already fetched (_leg_quotes
    builds it), sampled two ways:
      * day-end, one per UTC day: the equity curve reads these
        (_open_value_path);
      * checkpoint, one per weekly entry checkpoint: Pass 2 sizes on these
        (_leg_mark).
    Each sample is the latest usable ask (_usable_ask) at or before that
    moment, kept with no age limit; from settlement on it is the payout (1.0
    or 0.0), and before the first usable ask it is NaN, read as the price
    the leg paid. Past the arrays' end it reads the payout (or the last
    day-end sample when the payout is unknown).

    For selling a position early (_simulate_at_discount's sell_at), it also
    holds each side's BID at every checkpoint and whether the market had paid
    out by then. A bid is what selling one contract of that side would
    fetch: the YES bid is 1 − the NO ask and the NO bid is 1 − the YES ask,
    read from the latest candle at or before the checkpoint only when that
    candle ended at most one candle period before it and the opposite ask is
    usable. Unlike the asks, bids are NOT carried forward: an old or missing
    quote means no bid (NaN), and a leg with no bid cannot be sold.

    The sell rule also checks a position on the days before a sale
    (TAKE_PROFIT_HOLD_DAYS), so it holds, for every day from first_day, each
    side's bid at that day's check and whether the market had paid out by
    then. A day's check is at the moment of the next checkpoint (on or after
    that day) less whole days — 24 hours apart, the check on a checkpoint's
    date being the checkpoint itself — and reads the latest candle that ended
    in the 24 hours before it, that day's last quote. These bids are not
    carried forward either.

    For trading through a modeled order book it also holds the market's
    24-hour traded volume at every checkpoint and at every day's check, and
    the depth model (one object shared by every market). From them book_at
    builds a checkpoint's synthetic book for a buy, and sale_ladder the bids
    a sale walks at a checkpoint or at one of the daily checks before it.

    A plain __slots__ class, not a dataclass, so astuple does not walk into
    it and a copy is the object itself. Two built from the same candles
    compare equal and share a `fingerprint` (read by dashboard._list_key).
    The arrays are read-only.

    Attributes:
        ticker (str): The market's ticker.
        first_day (date): The first day with a day-end sample.
        yes_days (np.ndarray): Day-end YES asks, one per day from first_day.
        no_days (np.ndarray): Day-end NO asks, the same days.
        first_checkpoint (date): The first entry checkpoint date on or after
            first_day; every later one is a whole number of weeks after it.
        yes_checkpoints (np.ndarray): YES asks at each checkpoint.
        no_checkpoints (np.ndarray): NO asks at each checkpoint.
        paid_yes (float): What a YES contract pays: 1.0 or 0.0, NaN when the
            result or the settlement time is unknown.
        paid_no (float): What a NO contract pays (1 - paid_yes, NaN alike).
        yes_bid_checkpoints (np.ndarray): YES bids at each checkpoint (NaN:
            no fresh bid); never carried forward.
        no_bid_checkpoints (np.ndarray): NO bids at each checkpoint, alike.
        paid_checkpoints (np.ndarray): Whether the market had paid out (a
            known payout, at or before the checkpoint) at each checkpoint.
        yes_bid_daily (np.ndarray): YES bids at each day's check, one per day
            from first_day (NaN: no quote in the 24 hours before it); never
            carried forward.
        no_bid_daily (np.ndarray): NO bids at each day's check, alike.
        paid_daily (np.ndarray): Whether the market had paid out by each
            day's check.
        volume_checkpoints (np.ndarray): Contracts traded in the 24 hours up
            to each checkpoint (depth_model.volume_24h; NaN: no volume data).
        volume_daily (np.ndarray): Contracts traded in the 24 hours up to
            each day's check, one per day from first_day, alike.
        depth (DepthModel | None): The depth model the synthetic books come
            from; None means every trade fills at the top of the book.
        fingerprint (str): A hex digest of everything above; two LegQuotes
            that compare equal share it.
    """

    __slots__ = ("ticker", "first_day", "yes_days", "no_days", "first_checkpoint",
                 "yes_checkpoints", "no_checkpoints", "paid_yes", "paid_no",
                 "yes_bid_checkpoints", "no_bid_checkpoints", "paid_checkpoints",
                 "yes_bid_daily", "no_bid_daily", "paid_daily",
                 "volume_checkpoints", "volume_daily", "depth", "fingerprint")

    def __init__(self, ticker: str, first_day: date, yes_days: np.ndarray, no_days: np.ndarray,
                 first_checkpoint: date, yes_checkpoints: np.ndarray,
                 no_checkpoints: np.ndarray, paid_yes: float, paid_no: float,
                 yes_bid_checkpoints: np.ndarray | None = None,
                 no_bid_checkpoints: np.ndarray | None = None,
                 paid_checkpoints: np.ndarray | None = None,
                 yes_bid_daily: np.ndarray | None = None,
                 no_bid_daily: np.ndarray | None = None,
                 paid_daily: np.ndarray | None = None,
                 volume_checkpoints: np.ndarray | None = None,
                 volume_daily: np.ndarray | None = None,
                 depth: DepthModel | None = None) -> None:
        """
        Store one market's samples, made read-only, and their fingerprint.

        Each ask array is carried forward first: a NaN after a usable sample
        takes that sample's value, so a side reads NaN only before its first
        usable ask, whoever built the arrays (_leg_quotes already builds them
        so). The bid arrays, checkpoint and daily, are stored as given — a bid
        is never carried forward.

        Args:
            ticker (str): The market's ticker.
            first_day (date): The day of yes_days[0] and no_days[0].
            yes_days (np.ndarray): Day-end YES asks (NaN: no usable ask).
            no_days (np.ndarray): Day-end NO asks, the same days.
            first_checkpoint (date): The date of yes_checkpoints[0].
            yes_checkpoints (np.ndarray): YES asks at weekly checkpoints.
            no_checkpoints (np.ndarray): NO asks at the same checkpoints.
            paid_yes (float): A YES contract's payout, NaN when unknown.
            paid_no (float): A NO contract's payout, NaN when unknown.
            yes_bid_checkpoints (np.ndarray | None): YES bids at the same
                checkpoints (NaN: none). None (a hand-built quote) reads as
                no bid at any checkpoint.
            no_bid_checkpoints (np.ndarray | None): NO bids, alike.
            paid_checkpoints (np.ndarray | None): Whether the market had paid
                out by each checkpoint. None reads as never.
            yes_bid_daily (np.ndarray | None): YES bids at each day's check,
                one per day from first_day (NaN: none). None (a hand-built
                quote) reads as no bid on any day.
            no_bid_daily (np.ndarray | None): NO bids at each day's check, alike.
            paid_daily (np.ndarray | None): Whether the market had paid out by
                each day's check. None reads as never.
            volume_checkpoints (np.ndarray | None): 24-hour volume at each
                checkpoint (NaN: none). None reads as no volume data.
            volume_daily (np.ndarray | None): 24-hour volume at each day's
                check, one per day from first_day (NaN: none). None reads as
                no volume data.
            depth (DepthModel | None): The depth model; None for none.
        """
        self.ticker = ticker
        self.first_day = first_day
        self.first_checkpoint = first_checkpoint
        # Adding 0.0 turns -0.0 into 0.0, so equal payouts give one fingerprint
        self.paid_yes = float(paid_yes) + 0.0
        self.paid_no = float(paid_no) + 0.0
        digest = hashlib.blake2b(digest_size=16)
        digest.update(repr((ticker, first_day.isoformat(), first_checkpoint.isoformat(),
                            self.paid_yes, self.paid_no)).encode("utf-8"))
        for name, values in (("yes_days", yes_days), ("no_days", no_days),
                             ("yes_checkpoints", yes_checkpoints),
                             ("no_checkpoints", no_checkpoints)):
            array = _carried_forward(values)
            array.setflags(write=False)
            setattr(self, name, array)
            # One spelling of NaN and of zero, so arrays that compare equal
            # (NaN equal to NaN) hash alike; the length keeps two arrays apart
            canonical = np.where(np.isnan(array), np.nan, array + 0.0)
            digest.update(len(array).to_bytes(8, "little"))
            digest.update(canonical.tobytes())
        # The bids, as given (never carried forward), and the paid-out
        # markers; missing ones read as no bid and never paid
        count = len(self.yes_checkpoints)
        for name, values in (("yes_bid_checkpoints", yes_bid_checkpoints),
                             ("no_bid_checkpoints", no_bid_checkpoints)):
            array = (np.full(count, np.nan) if values is None
                     else np.array(values, dtype=float))
            array.setflags(write=False)
            setattr(self, name, array)
            canonical = np.where(np.isnan(array), np.nan, array + 0.0)
            digest.update(len(array).to_bytes(8, "little"))
            digest.update(canonical.tobytes())
        paid = (np.zeros(count, dtype=bool) if paid_checkpoints is None
                else np.array(paid_checkpoints, dtype=bool))
        paid.setflags(write=False)
        self.paid_checkpoints = paid
        digest.update(len(paid).to_bytes(8, "little"))
        digest.update(paid.tobytes())
        # The sell rule's daily checks (TAKE_PROFIT_HOLD_DAYS): one per day
        # from first_day, each side's bid at that day's check and whether the
        # market had paid out by then. The old checkpoint arrays above only
        # cover Mondays; a sale needs a price for each of the days before
        # its Monday too (a leg already paid out counts at its payout). A
        # hand-built quote has none: no bid, never paid. Like every array
        # here they go into the fingerprint, so two quotes that differ in any
        # stored number never share one (dashboard._list_key shares a page
        # chunk between trade lists whose quotes' fingerprints match)
        days = len(self.yes_days)
        for name, values in (("yes_bid_daily", yes_bid_daily),
                             ("no_bid_daily", no_bid_daily)):
            array = (np.full(days, np.nan) if values is None
                     else np.array(values, dtype=float))
            array.setflags(write=False)
            setattr(self, name, array)
            canonical = np.where(np.isnan(array), np.nan, array + 0.0)
            digest.update(len(array).to_bytes(8, "little"))
            digest.update(canonical.tobytes())
        paid = (np.zeros(days, dtype=bool) if paid_daily is None
                else np.array(paid_daily, dtype=bool))
        paid.setflags(write=False)
        self.paid_daily = paid
        digest.update(len(paid).to_bytes(8, "little"))
        digest.update(paid.tobytes())
        # The volume, at each checkpoint and at each day's check, as given;
        # missing reads as no volume data
        for name, values, size in (("volume_checkpoints", volume_checkpoints, count),
                                   ("volume_daily", volume_daily, days)):
            volume = (np.full(size, np.nan) if values is None
                      else np.array(values, dtype=float))
            volume.setflags(write=False)
            setattr(self, name, volume)
            canonical = np.where(np.isnan(volume), np.nan, volume + 0.0)
            digest.update(len(volume).to_bytes(8, "little"))
            digest.update(canonical.tobytes())
        # The model's digest names its table, so equal models fingerprint alike
        self.depth = depth
        digest.update((depth.digest if depth is not None else "none").encode("utf-8"))
        self.fingerprint = digest.hexdigest()

    def __reduce__(self) -> tuple:
        """
        Rebuild from the samples when pickled (a dashboard worker process receives it so).

        Re-running __init__ gives the same read-only arrays and fingerprint:
        carrying the asks forward again changes nothing, and the bids and
        volumes are stored as given. The depth model travels with it, so a
        worker buys and sells through the same books.

        Returns:
            tuple: (LegQuotes, its constructor arguments).
        """
        return (LegQuotes, (self.ticker, self.first_day, self.yes_days, self.no_days,
                            self.first_checkpoint, self.yes_checkpoints, self.no_checkpoints,
                            self.paid_yes, self.paid_no, self.yes_bid_checkpoints,
                            self.no_bid_checkpoints, self.paid_checkpoints,
                            self.yes_bid_daily, self.no_bid_daily, self.paid_daily,
                            self.volume_checkpoints, self.volume_daily, self.depth))

    def __copy__(self) -> "LegQuotes":
        """Return this object: it is read-only, and trades share one per market."""
        return self

    def __deepcopy__(self, memo: dict) -> "LegQuotes":
        """
        Return this object (dataclasses.astuple deep-copies every field).

        Args:
            memo (dict): copy.deepcopy's memo, unused.

        Returns:
            LegQuotes: self.
        """
        return self

    def __eq__(self, other: object) -> bool:
        """
        Compare two markets' samples by value, NaN equal to NaN.

        Args:
            other (object): Anything.

        Returns:
            bool: True when every attribute matches; NotImplemented for a
                non-LegQuotes.
        """
        if not isinstance(other, LegQuotes):
            return NotImplemented

        def same(a: float, b: float) -> bool:
            """Whether two payouts are equal, NaN counting as equal to NaN."""
            return a == b or (a != a and b != b)

        def model(quotes: "LegQuotes") -> str | None:
            """The depth model's digest, or None for no model."""
            return None if quotes.depth is None else quotes.depth.digest

        return (self.ticker == other.ticker and self.first_day == other.first_day
                and self.first_checkpoint == other.first_checkpoint
                and same(self.paid_yes, other.paid_yes) and same(self.paid_no, other.paid_no)
                and all(np.array_equal(getattr(self, name), getattr(other, name), equal_nan=True)
                        for name in ("yes_days", "no_days", "yes_checkpoints",
                                     "no_checkpoints", "yes_bid_checkpoints",
                                     "no_bid_checkpoints", "yes_bid_daily",
                                     "no_bid_daily", "volume_checkpoints",
                                     "volume_daily"))
                and np.array_equal(self.paid_checkpoints, other.paid_checkpoints)
                and np.array_equal(self.paid_daily, other.paid_daily)
                and model(self) == model(other))

    def __hash__(self) -> int:
        """Hash on the ticker and first day, which equal objects share."""
        return hash((self.ticker, self.first_day))

    def __repr__(self) -> str:
        """Name the market and its first day, not the arrays."""
        return f"LegQuotes({self.ticker!r}, from {self.first_day.isoformat()})"

    def _side(self, side: str) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Pick one side's day-end samples, checkpoint samples and final value.

        The final value is what a lookup past the end of the arrays reads:
        the side's payout, or, when the payout is unknown, its last day-end
        sample (the last usable ask it had; NaN when it never had one).

        Args:
            side (str): "yes" or "no" (scanner.leg_sides).

        Returns:
            tuple: (day-end samples, checkpoint samples, final value).

        Raises:
            ValueError: For any other side.
        """
        if side == "yes":
            days, checkpoints, paid = self.yes_days, self.yes_checkpoints, self.paid_yes
        elif side == "no":
            days, checkpoints, paid = self.no_days, self.no_checkpoints, self.paid_no
        else:
            raise ValueError(f"side must be 'yes' or 'no', got {side!r}")
        if paid != paid and len(days):
            # An unknown payout: past the samples the last usable ask stands
            paid = float(days[-1])
        return days, checkpoints, paid

    def at_checkpoint(self, day: date, side: str, entry_price: float) -> float:
        """
        One contract's value at the entry checkpoint on `day`.

        Args:
            day (date): A checkpoint date on this market's weekly grid.
            side (str): The side the leg holds, "yes" or "no".
            entry_price (float): The price the leg paid, returned when the
                side has had no usable ask yet.

        Returns:
            float: The side's latest usable ask at the checkpoint, the payout
                once the market has paid out, or entry_price.

        Raises:
            ValueError: For a day before first_checkpoint or not a whole
                number of weeks after it — a schedule other than the one the
                samples were taken on — or for an unknown side.
        """
        _days, checkpoints, final = self._side(side)
        offset = (day - self.first_checkpoint).days
        if offset < 0 or offset % 7:
            raise ValueError(
                f"{day} is not an entry checkpoint of {self.ticker}'s quotes "
                f"(weekly from {self.first_checkpoint})")
        index = offset // 7
        value = checkpoints[index] if index < len(checkpoints) else final
        # NaN (no usable ask yet, and no payout) values the leg at entry
        return entry_price if value != value else float(value)

    def _checkpoint_index(self, day: date) -> int:
        """
        The index of the checkpoint on `day` in the weekly checkpoint arrays.

        Args:
            day (date): A checkpoint date on this market's weekly grid.

        Returns:
            int: Weeks since first_checkpoint (may be past the arrays' end).

        Raises:
            ValueError: For a day before first_checkpoint or not a whole
                number of weeks after it.
        """
        offset = (day - self.first_checkpoint).days
        if offset < 0 or offset % 7:
            raise ValueError(
                f"{day} is not an entry checkpoint of {self.ticker}'s quotes "
                f"(weekly from {self.first_checkpoint})")
        return offset // 7

    @staticmethod
    def _days_back(days_back: Any) -> int:
        """
        Check how many days before a checkpoint a lookup asks about.

        Args:
            days_back (Any): 0 for the checkpoint itself, or 1 to
                _HOLD_DAYS_MAX - 1 for a daily check before it.

        Returns:
            int: days_back, as a builtin int.

        Raises:
            ValueError: For a bool, a number that is not whole, or one
                outside 0 to _HOLD_DAYS_MAX - 1.
        """
        if (isinstance(days_back, (bool, np.bool_)) or not isinstance(days_back, numbers.Integral)
                or not 0 <= days_back < _HOLD_DAYS_MAX):
            raise ValueError(f"days_back must be a whole number from 0 to "
                             f"{_HOLD_DAYS_MAX - 1}, got {days_back!r}")
        return int(days_back)

    def _day_index(self, day: date, back: int) -> int:
        """
        The index, in the daily bid arrays, of the check `back` days before the checkpoint on `day`.

        Args:
            day (date): A checkpoint date on this market's weekly grid.
            back (int): A checked number of days back (_days_back), 1 or more.

        Returns:
            int: Days from first_day to that check (negative before
                first_day; may be past the arrays' end).

        Raises:
            ValueError: For a day off the weekly checkpoint grid.
        """
        self._checkpoint_index(day)
        return (day - self.first_day).days - back

    def paid_at_checkpoint(self, day: date, days_back: int = 0) -> bool:
        """
        Whether the market had paid out, with a known payout, by the checkpoint on `day` (or a check before it).

        Args:
            day (date): A checkpoint date on this market's weekly grid.
            days_back (int): 0 (the default) for the checkpoint itself; 1 to
                _HOLD_DAYS_MAX - 1 for the daily check that many days before
                it.

        Returns:
            bool: The marker; past the arrays' end (after the market's
                settlement date), True exactly when the payout is known; for
                a check before first_day, False.

        Raises:
            ValueError: For a day off the weekly checkpoint grid, or a bad
                days_back.
        """
        back = self._days_back(days_back)
        if back:
            index, paid = self._day_index(day, back), self.paid_daily
        else:
            index, paid = self._checkpoint_index(day), self.paid_checkpoints
        if index < 0:
            return False
        if index < len(paid):
            return bool(paid[index])
        return self.paid_yes == self.paid_yes

    def bid_at_checkpoint(self, day: date, side: str, days_back: int = 0) -> float:
        """
        What selling one contract of `side` would fetch at the checkpoint on `day` (or a check before it).

        Args:
            day (date): A checkpoint date on this market's weekly grid.
            side (str): The side held, "yes" or "no".
            days_back (int): 0 (the default) for the checkpoint itself; 1 to
                _HOLD_DAYS_MAX - 1 for the daily check that many days before
                it.

        Returns:
            float: At the checkpoint, the side's fresh bid (from a candle
                within one candle period of it); at an earlier check, the bid
                from the last quote in the 24 hours before that check. NaN
                when there is none (no such candle, an unusable opposite ask,
                before first_day, or past the arrays' end).

        Raises:
            ValueError: For a day off the weekly checkpoint grid, an unknown
                side or a bad days_back.
        """
        if side == "yes":
            bids, daily = self.yes_bid_checkpoints, self.yes_bid_daily
        elif side == "no":
            bids, daily = self.no_bid_checkpoints, self.no_bid_daily
        else:
            raise ValueError(f"side must be 'yes' or 'no', got {side!r}")
        back = self._days_back(days_back)
        if back:
            index, bids = self._day_index(day, back), daily
        else:
            index = self._checkpoint_index(day)
        return float(bids[index]) if 0 <= index < len(bids) else float("nan")

    def _volume_at(self, day: date, days_back: int = 0) -> float | None:
        """
        The 24-hour volume behind the checkpoint on `day` (or a check before it), when a book can be built there.

        Args:
            day (date): A checkpoint date.
            days_back (int): 0 (the default) for the checkpoint itself; 1 to
                _HOLD_DAYS_MAX - 1 for the daily check that many days before
                it.

        Returns:
            float | None: The volume; None when there is no depth model, the
                day is not one of this market's checkpoints, the check falls
                outside the samples, or the volume is unknown there.

        Raises:
            ValueError: For a bad days_back.
        """
        back = self._days_back(days_back)
        if self.depth is None:
            return None
        offset = (day - self.first_checkpoint).days
        if offset < 0 or offset % 7:
            return None
        if back:
            # The daily check `back` days before the checkpoint
            index, volumes = (day - self.first_day).days - back, self.volume_daily
        else:
            index, volumes = offset // 7, self.volume_checkpoints
        if not 0 <= index < len(volumes):
            return None
        volume = float(volumes[index])
        return None if volume != volume else volume

    def has_book_at(self, day: date) -> bool:
        """
        Whether book_at can build this market's book at the checkpoint on `day`.

        Args:
            day (date): A checkpoint date.

        Returns:
            bool: True when there is a depth model and a volume there.
        """
        return self._volume_at(day) is not None

    def book_at(self, day: date, yes_ask: float, yes_bid: float) -> dict | None:
        """
        This market's synthetic order book at the checkpoint on `day`, anchored at these prices.

        Built by depth_model.book from the depth model and the market's 24-hour
        volume there, in scanner._fetch_orderbook's shape, so the live
        enrichment can walk it.

        Args:
            day (date): A checkpoint date.
            yes_ask (float): The market's YES ask then, where the NO bids start.
            yes_bid (float): Its YES bid then (1 - its NO ask), where the YES bids start.

        Returns:
            dict | None: {"yes": YES bids, "no": NO bids}; None when there is no
                model, the day is not one of its checkpoints, or the volume is
                unknown there.
        """
        volume = self._volume_at(day)
        if volume is None:
            return None
        return _synthetic_book(self.depth, yes_ask, yes_bid, volume)

    def sale_ladder(self, day: date, best_bid: float,
                    days_back: int = 0) -> list[list[float]] | None:
        """
        The bids a sale into this market would walk at the checkpoint on `day` (or a check before it).

        Built by depth_model.bid_ladder from the depth model and the market's
        24-hour volume there, starting at best_bid: the bid of the side sold
        at that check, which the caller reads. The sell rule's daily checks
        (TAKE_PROFIT_HOLD_DAYS) read the volume of the 24 hours before each
        check (volume_daily); the checkpoint reads its own
        (volume_checkpoints).

        Args:
            day (date): A checkpoint date.
            best_bid (float): The side's bid at that check.
            days_back (int): 0 (the default) for the checkpoint itself; 1 to
                _HOLD_DAYS_MAX - 1 for the daily check that many days before
                it.

        Returns:
            list[list[float]] | None: [[price, contracts], ...], best first;
                empty when the model puts no contracts near that bid. None
                when the model cannot say: no model, no volume data there,
                or a bid the model has no ladders for (below 1c or above
                99c, depth_model.can_start_at). The sale then takes that one
                bid, in any size.

        Raises:
            ValueError: For a bad days_back.
        """
        volume = self._volume_at(day, days_back)
        if volume is None or not can_start_at(best_bid):
            return None
        return bid_ladder(self.depth, best_bid, volume)

    def day_values(self, first: date, last: date, side: str, entry_price: float) -> np.ndarray:
        """
        One contract's value at the end of each day from `first` to `last`.

        Args:
            first (date): The first day (inclusive).
            last (date): The last day (inclusive); before `first` gives [].
            side (str): The side the leg holds, "yes" or "no".
            entry_price (float): The price the leg paid, used for a day before
                the side's first usable ask (before first_day too).

        Returns:
            np.ndarray: One float per day: that side's latest usable ask at the
                day's end, the payout once the market has paid out, or
                entry_price.

        Raises:
            ValueError: For an unknown side.
        """
        days, _checkpoints, final = self._side(side)
        count = (last - first).days + 1
        if count <= 0:
            return np.empty(0)
        start = (first - self.first_day).days
        # Past the end of the samples: the payout (or the last usable ask)
        out = np.full(count, final)
        lo, hi = max(start, 0), min(start + count, len(days))
        if hi > lo:
            out[lo - start:hi - start] = days[lo:hi]
        if start < 0:
            # Before the first sample there is no quote
            out[:min(-start, count)] = np.nan
        return np.where(np.isnan(out), entry_price, out)


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
        entry_date (date): The Monday the trade was entered. The entry
            prices (entry_pA..entry_nB), the tickers and event_ticker all
            come from that Monday.
        exit_date (date): The date the later-settling market resolved; marks when cash returned.
        entry_pA (float): YES ask price of market A at entry (the candle
            quote, the top of the book). Range: [0.01, 0.99]. The quote of the
            market-A leg for time_series; for same_title it is the quote that
            made A the pricier side (reporting only).
        entry_pB (float): YES ask price of market B at entry. Range: [0.01, 0.99].
            The quote of the market-B leg for same_title; for time_series it
            is not traded, and with the other three entry quotes it gives the
            mid spread the profit model reads (config.time_series_mid_spread).
        entry_nA (float): NO ask price of market A at entry (≈ 1 − yes_bid_A).
            Range: [0.01, 0.99]. The quote of the market-A leg for
            same_title; for time_series it is not traded, and gives market
            A's YES bid in the mid spread.
        entry_nB (float): NO ask price of market B at entry (≈ 1 − yes_bid_B).
            Range: [0.01, 0.99]. The quote of the market-B leg for
            time_series; reporting only for same_title.
        n (int): Number of contracts bought on each leg (x = y = n). Always >= 1.
            Both legs are always the same size.
        total_cost (float): Dollar cost of the contracts: n times the two
            prices paid (_paid_prices: fill_price_a and fill_price_b, or the
            leg quotes for a trade built without them). Excludes taker fees
            (see fees).
        fees (float): Exact ceiling-rounded taker fee for both legs at the
            prices paid, charged at entry.
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
            profit_ratio * 30 / holding_days. Reporting only — trades are
            ranked on an expected ratio computed at entry, which never reads
            how the markets settled (though its horizon ends at whichever
            leg actually closed last).
        kelly_fraction (float): Capped Kelly fraction used for sizing (on an
            add-on, the share of the portfolio value held_pair_fraction
            allowed); <= the size cap it was simulated under —
            config.BUDGET_FRACTION unless a size-cap sweep (CapSweep)
            simulated another.
        expected_payoff (float): NET profit in a win scenario after exact fees:
            n * (1 − price_a − price_b) − fees on the prices paid. For
            same_title this is the guaranteed floor (every co-resolution
            outcome pays at least n); for time_series it is the profit realized
            in either win cell, while the in-between cell loses total_cost +
            fees instead. Always > 0 for recorded trades.
        slippage (float): profit − expected_payoff. same_title: positive when
            both legs paid (A=NO, B=YES), zero in the co-resolution cells,
            negative only in the loss cell. time_series: zero in both win cells
            and negative in the loss cell — there is no positive-slippage cell.
        holding_days (int): Calendar days between entry_date and exit_date. Always >= 1.
        balance_at_entry (float): Portfolio value in dollars at the entry
            checkpoint, after that day's pay-outs and before its trades: the
            cash plus every open trade at market (_open_value: each leg at the
            ask of the side it holds, at its payout once paid out; a trade
            with no quotes at its cost). The Kelly share is taken of it; the
            trade can cost less when the cash left was smaller. It is taken at
            the checkpoint, so it equals no day-end row of the equity curve
            exactly.
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
        subtitle_a (str): Market A's outcome label ("" when absent).
        subtitle_b (str): Market B's outcome label ("" when absent).
        close_date_a (date | None): The date market A closed for trading.
        close_date_b (date | None): The date market B closed for trading.
        settled_date_a (date | None): The date market A settled.
        settled_date_b (date | None): The date market B settled. All six are
            read by the dashboard's best/worst trade tables, and the two
            close dates also by the sell rule's minimum of days before
            maturity (_days_left: a position matures when its last market
            closes); None/"" on a trade constructed without them.
        add_on (bool): True when this trade added to a pair the simulation
            still held — the same two markets, bought the same way round, while
            an earlier trade of the pair was open (only a simulation run with
            add_to_held makes one). It is a trade of its own, with its own
            count, cost, fees and payout; the earlier trade is never changed.
            Reporting only. False by default, so every other construction
            still builds.
        sold (bool): True when the trade's position was sold before it paid
            out (only a simulation run with sell_at sells): exit_date is then
            the sale day, actual_payoff what the sale returned after its fees
            (each leg at its sale price less config.fee_leg_exact on the sale,
            a leg whose market had already paid out at its payout), and
            profit, profit_ratio, monthly_profit_ratio, holding_days and
            slippage are the sale's. outcome_a/_b and settled_date_a/_b keep
            how the markets really settled. False by default.
        sale_price_a (float | None): The price market A's leg sold at: the
            average down the modeled bid ladder, or the bid itself with no
            ladder (_position_sale_value). None when the trade was not sold,
            or market A had paid out by the sale.
        sale_price_b (float | None): The same for market B's leg.
        sale_fees (float): The taker fees the sale paid, both legs; 0.0 when
            the trade was not sold. Already deducted from actual_payoff.
        marks (tuple[LegQuotes, LegQuotes] | None): Market A's and market B's
            prices over time (LegQuotes), which value the open trade at
            market: Pass 2 at each checkpoint (_open_value) and the equity
            curve at each day's end (_open_value_path). None (the default, and
            every hand-built trade) values it at its cost. Left out of
            equality and repr, so two trades compare on what was traded;
            references only, shared by every trade in the same markets.
        fill_price_a (float | None): The average price paid per contract on
            market A's leg: the candle quote at the top of the book, or the
            average over the synthetic book's levels the trade walked. None on
            a trade built without it, which reads its leg quote instead
            (_paid_prices).
        fill_price_b (float | None): The same for market B's leg.
        book_walked (bool): True when the trade was sized by walking a
            synthetic order book; False when it filled at the top of the book
            (no depth model, or no volume data that Monday).
    """
    pair_type: str       # "time_series" | "same_title"
    ticker_a: str
    ticker_b: str
    title_a: str
    title_b: str
    category: str
    entry_date: date
    exit_date: date      # date the last-settling market resolved
    entry_pA: float      # YES ask of A at entry — the market-A leg quote for time_series
    entry_pB: float      # YES ask of B at entry — the market-B leg quote for same_title
    entry_nA: float      # NO ask of A at entry (≈ 1 - yes_bid_A) — the market-A leg quote for same_title
    entry_nB: float      # NO ask of B at entry (≈ 1 - yes_bid_B) — the market-B leg quote for time_series
    n: int               # contracts bought on each leg (x = y = n)
    total_cost: float
    fees: float          # exact both-leg taker fees, charged at entry
    outcome_a: str       # "yes" | "no"
    outcome_b: str       # "yes" | "no"
    actual_payoff: float
    profit: float
    profit_ratio: float
    monthly_profit_ratio: float  # realized profit_ratio * 30 / holding_days (reporting only)
    # Capped Kelly fraction used for sizing (on an add-on, the share of the
    # portfolio value held_pair_fraction allowed); <= the run's size cap
    kelly_fraction: float
    expected_payoff: float  # n * (1 - price_a - price_b) minus fees — same_title floor / time_series win-cell profit
    slippage: float         # profit - expected_payoff
    holding_days: int
    balance_at_entry: float  # portfolio value at the entry checkpoint (cash + open trades at market)
    # Calendar days between the two legs' deadlines — their close_times for a
    # cross-event pair, their two STATED deadlines for a same-event ladder
    # (DR-73) — carried out of _find_entry rather than recomputed, so it is
    # always the gap that chose the tier. None for same_title (no deadline-gap
    # concept) and for any trade constructed without it (test fixtures).
    # Reporting only — nothing sizes, prices or settles on this field, but
    # _interval_calibration DOES band k-hat on the same quantity via
    # CalibrationObservation.gap_days, so a ladder's bands are stated-gap bands.
    deadline_gap_days: int | None = None
    # Market A's event ticker ("" when absent) and whether the pair is a
    # same-event ladder (time_series, both legs on one non-empty event
    # ticker). Set by _simulate_at_discount; defaulted so a trade constructed
    # without them (test fixtures) still builds. Reporting only — nothing
    # sizes, prices or settles on either field.
    event_ticker: str = ""
    same_event_ladder: bool = False
    # Each leg's own market: its outcome label (the subtitle — for a game
    # "Western Illinois", for a strike "$80,000 or above"), the date it closed
    # for trading and the date it settled. Set by _simulate_at_discount for
    # the dashboard's best/worst trade tables, which spell out what each leg
    # bought and how it resolved; defaulted so a trade constructed without
    # them (test fixtures) still builds. Reporting only — nothing sizes,
    # prices or settles on any of them (exit_date is what settlement reads).
    subtitle_a: str = ""
    subtitle_b: str = ""
    close_date_a: date | None = None
    close_date_b: date | None = None
    settled_date_a: date | None = None
    settled_date_b: date | None = None
    # Whether this trade added to a pair the simulation still held (only with
    # _simulate_at_discount(add_to_held=True)); reporting only
    add_on: bool = False
    # Whether its position was sold early (only with _simulate_at_discount's
    # sell_at), at what price each leg sold (None: not sold, or that market had
    # already paid out) and the fees the sale paid; defaulted so every other
    # construction still builds
    sold: bool = False
    sale_price_a: float | None = None
    sale_price_b: float | None = None
    sale_fees: float = 0.0
    # Each leg's prices over time, for valuing the open trade at market; None
    # values it at cost. Not compared or printed: it is how the trade is
    # valued while open, not what was traded
    marks: tuple[LegQuotes, LegQuotes] | None = field(default=None, compare=False, repr=False)
    # The average price paid per contract on each leg (None: read the leg
    # quotes, _paid_prices), and whether a synthetic book was walked
    fill_price_a: float | None = None
    fill_price_b: float | None = None
    book_walked: bool = False


@dataclass(frozen=True)
class HalfSplit:
    """
    A split-half out-of-sample check on one band-sweep scenario.

    The scenario's entries are split at BacktestSweep.split_date and each half
    is simulated ALONE from the run's initial balance, at the scenario's own
    band and k — so a band x k cell that only looks good because of one
    stretch of history shows it as two very different numbers. A pair is in
    the half its first qualifying Monday falls in, and a first-half pair is
    only ever entered before the split (see _split_halves). Only the two
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
        h1_return (float): Total return of the pairs that first qualified
            STRICTLY BEFORE split_date, simulated alone and only on their
            Mondays before it: (final portfolio value − initial balance) /
            initial balance. 0.0 when that half is empty — read h1_entries
            before trusting it.
        h2_return (float): The same for the pairs that first qualified ON or
            after split_date.
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


@dataclass(frozen=True)
class SaleCheck:
    """
    One position a selling simulation sold: how many days it had left, and its profit at each check.

    _simulate_at_discount records one for every position it sells, in the
    order it sells them, on SweepPoint.sales. Together they say, without
    simulating again, whether a stricter sell setting would have made every
    one of the same sales: a higher share of potential profit (every check's
    profits must still reach it, as _reached_every_day reads them) or more
    days required before maturity (days_left must still be at least that
    many, as _far_enough reads it).

    Declared before SweepPoint for the same reason as HalfSplit: SweepPoint
    annotates a field with this class.

    Attributes:
        days_left (int | None): Calendar days from the sale's checkpoint date
            to the latest close date among the position's markets
            (_days_left); None when one of its trades has no close date.
        profits (tuple[tuple[float, float], ...]): (realized profit,
            potential profit), in dollars, at each daily check the sale
            read, the checkpoint's own first (_hold_readings).
    """
    days_left: int | None
    profits: tuple[tuple[float, float], ...]


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
            config.TIME_SERIES_INTERVAL_PROB_DISCOUNT, config.py's k, which the
            backtest defaults to (a live run prices at the saved live
            defaults' k instead).
        trades (list[BacktestTrade]): One record per entered pair, in
            entry-date order; empty if nothing was ever entered.
        equity_df (pd.DataFrame): Daily equity curve with columns
            [date, portfolio_value, daily_return], opening one row before the
            run's start_date at the initial balance and flat at it when trades
            is empty. portfolio_value is cash plus open positions at market
            (at cost for a trade with no quotes), so deploying capital never
            moves it: it moves on fees, on open trades' changes in value and
            on pay-outs (see _build_equity_curve).
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
        ex_top_event (tuple[str, float] | None): (event ticker, total return):
            the event whose trades made the most profit here, and the return
            of a re-run of this point's entries without that event's pairs
            (see _ex_top_event). Set only on a band sweep's "all" and
            "time_series" points; None when no trade names an event.
        tier_floors (bool): False only for a tier-off sweep's points
            (BacktestSweep.tier_off_scenarios), whose entries were detected at
            the band floor alone with the deadline-gap tier floors not
            applied. A label; the entries already reflect it. Appended with a
            default of True, so every existing construction still builds as a
            tier-on point.
        size_cap (float | None): The RESOLVED per-trade Kelly size cap this
            point was sized under — config.BUDGET_FRACTION unless a size-cap
            sweep simulated another; 1.0 means no per-trade cap (full Kelly),
            same-title pairs still under SAME_TITLE_SIZE_CAP. None only on a
            hand-built point, which a report labels "not recorded".
            Appended with a default, like the fields below it, so no
            existing construction moves.
        peak_kelly_fraction (float | None): The largest uncapped Kelly
            fraction f* over every Monday a pair may be traded on (every
            passing Monday of a time-series pair; a same-title pair's first,
            and with add_to_held also its later passing Mondays that keep the
            first one's legs the same way round); 0.0 when none passed, None
            only on a hand-built point. It does not depend on the cap, and
            with no walked book every cap at or above it sizes the point the
            same way — an add-on's size included, since the cap reaches it
            only through min(pair cap, f*) — so CapSweep reuses one
            simulation for all of them (with a walked book, from
            cap_free_from).
        add_to_held (bool): True when the simulation could add to a pair it
            still held (_simulate_at_discount(add_to_held=True)). Not only a
            label: _ex_top_event re-simulates the point at this setting, so its
            excluding-top-event run adds to held pairs exactly when the point
            did. Appended with a default of False, so every existing
            construction still builds as a point that never adds.
        sell_at (float | None): The share of potential profit at which the
            simulation sold a position early (_simulate_at_discount's sell_at),
            or None when it never sold. Like add_to_held, _half_split and
            _ex_top_event re-simulate the point at this setting. Appended with
            a default of None, so every existing construction still builds as
            a point that never sells.
        cap_free_from (float | None): The size cap at and above which no
            trade of this simulation depends on the cap: the larger of
            peak_kelly_fraction and, for each pair type with a candidate that
            could walk a synthetic book, the cap above which the live search
            range stops moving (round(1 - k, 12) for time-series,
            min(same-title cap, SAME_TITLE_CO_RESOLVE_PROB) for same-title).
            CapSweep shares one simulation only at caps at or above it. None
            on a hand-built point, which shares from peak_kelly_fraction.
        sell_min_days (int | None): The fewest days a position had to have
            left before its last market stopped trading for the simulation
            to sell it (_simulate_at_discount's sell_min_days), or None when
            there was no such minimum. Like sell_at, _half_split and
            _ex_top_event re-simulate the point at this setting. Defaulted,
            so every existing construction still builds as a point with no
            minimum.
        sales (tuple[SaleCheck, ...] | None): One SaleCheck per position the
            simulation sold, in the order it sold them: () when it could sell
            and sold nothing, None when it never sells (sell_at None) or the
            point was built by hand. A copy CapSweep shares at another cap
            keeps it, since its walk, and so its sales, are the same. Not in
            repr and not compared: it describes how the trades came about,
            which the trades themselves already record.
    """
    k: float
    trades: list[BacktestTrade]
    equity_df: pd.DataFrame
    spread_band: tuple[float, float] | None = None
    population: str = "all"
    halves: HalfSplit | None = None
    ex_top_event: tuple[str, float] | None = None
    tier_floors: bool = True
    size_cap: float | None = None
    peak_kelly_fraction: float | None = None
    add_to_held: bool = False
    sell_at: float | None = None
    cap_free_from: float | None = None
    sell_min_days: int | None = None
    sales: tuple[SaleCheck, ...] | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class CalibrationObservation:
    """
    One time-series candidate entry reduced to what calibration needs.

    Built by _interval_calibration() — one per candidate it measures, so a
    band's observations ARE the population its pooled row is computed over
    (premise violations, same-title entries and non-binary settlements never
    become one) — and carried out on IntervalCalibration.observations, so a
    report can regroup that population (by category, by tag) without a second
    definition of which entries count. Reduce any subset of them with
    _calibration_bucket(), the one definition of the k-hat arithmetic.

    Holds scalars and strings only, never a market record: a band sweep keeps
    one calibration per band for the whole simulation phase, and a reference
    to a market dict here would pin it alive long after the entry pass
    released the candles and the pair list (TS-07's residency rule).

    Attributes:
        gap_days (int | None): Deadline gap the pair's price tier was selected
            from, carried out of _find_entry. None only if a time-series entry
            somehow reached here without one, in which case the observation
            still counts in the pooled row but lands in no gap band.
        implied (float): The market-implied chance, at entry, that the event
            first happens between the two deadlines: the pair's mid spread
            at its first qualifying Monday's four quotes
            (config.time_series_mid_spread — the later market's midpoint
            minus the earlier one's), the quantity the forecast's k
            multiplies. Zero or negative only when the earlier market's
            quote is crossed (its YES ask below its own YES bid): on any
            other entry, _find_entry's fee check keeps the later book's
            width (its YES ask plus its NO ask, minus 1) under the YES-ask
            gap, which keeps the mid spread above half that gap.
        in_between (bool): Whether the pair actually settled in-between
            (earlier NO, later YES) — the time-series bet's only loss cell.
        event_ticker (str): Market A's event ticker as the entry carries it
            (_find_entry's canonicalized leg — the same one
            BacktestTrade.event_ticker records for a traded pair), "" when
            the record carried none or carried a non-string. The key a report
            files the observation under: its series names Kalshi's category
            and tags.
        category (str): infer_category(event_ticker) — the ticker-prefix label
            BacktestTrade.category carries, which a report falls back to when
            the series is missing from Kalshi's category listing.

    event_ticker and category are REQUIRED, with no default, for the reason
    DR-71 made OutcomeLabelCoverage's phrasing fields required: a construction
    that forgot them would silently file its observation under "" — not even
    the "Other" infer_category("") gives a missing ticker — and a report would
    render that as a real category.
    """
    gap_days: int | None
    implied: float
    in_between: bool
    event_ticker: str
    category: str


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
            detected under (_interval_calibration's spread_min) — or, in a
            tier-off sweep's calibration (_interval_calibration's
            tier_floors=False), the band's floor alone. The pooled row spans
            every band and therefore has no single tier: it carries 0.0,
            which the report renders as "-". Read `tier <= 0` as "no floor to
            print" — the pooled row, or a tier-off bucket at a band floor of
            0, whose entries cleared no floor at all — never as a real
            threshold.
        n (int): Candidates in the bucket. Can be 0 on the pooled row when
            every time-series candidate was a premise violation (empty gap
            bands are omitted from IntervalCalibration.buckets entirely).
        realised_rate (float): Fraction of the bucket that actually settled
            in-between (earlier NO, later YES) — the event the time-series bet
            loses on. 0.0 when n is 0.
        mean_implied (float): Mean market-implied in-between mass at entry
            across the bucket: the mean of its observations' mid spreads
            (CalibrationObservation.implied). 0.0 when n is 0.
        empirical_k (float | None): realised_rate / mean_implied — the
            fraction of the market-implied in-between mass that actually
            materialized, i.e. the empirical counterpart of
            config.TIME_SERIES_INTERVAL_PROB_DISCOUNT, a fraction of the same
            mid spread. None when mean_implied is not positive (including the
            n == 0 case), since the ratio is undefined rather than zero.
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
    `calibration`, every band's in `calibrations_by_band`, and each binding
    band's tier-floors-off entries' in `tier_off_calibrations_by_band`)
    rather than off any single SweepPoint. It is NOT band-independent: a
    band changes which pairs enter, and when — and so do the tier floors.

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
        observations (tuple[CalibrationObservation, ...]): Every candidate the
            pooled row was computed over, in the order they were measured —
            _calibration_bucket(label, 0.0, observations) on this very tuple
            reproduces `pooled` exactly (a regrouped, reordered or reassembled
            copy agrees only to float rounding, since its sums run in another
            order). Carried so a report can regroup the same population — by
            category, tag or spread band — through the same arithmetic instead
            of re-deriving which entries count. Appended with a default of ()
            so a hand-built instance (a test fixture, the golden fixture) still
            builds, which leaves "not carried" and "no candidate" looking
            alike: a consumer must compare len(observations) with pooled.n,
            never test observations == ().
    """
    pooled: IntervalCalibrationBucket
    buckets: list[IntervalCalibrationBucket]
    excluded_premise_violations: int
    observations: tuple[CalibrationObservation, ...] = ()


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
        corpus_provenance (CorpusProvenance | None): What the fetched corpus
            says about itself — when it was assembled or last extended,
            whether it was served from an earlier run's cache, and the
            archive cutoff as of that assembly — taken off the corpus BY TYPE
            (a historical.SettledCorpus) before it is released. None when the
            corpus was a plain list (a test stub).
            Carried to BacktestSweep.corpus_provenance for the dashboard
            header.
    """
    all_pairs: list
    candles_by_ticker: dict
    label_coverage: OutcomeLabelCoverage | None
    start_date: date
    max_horizon_days: int | None
    same_event_ladders: bool | None
    corpus_provenance: CorpusProvenance | None = None


def _curve_end_date(point: SweepPoint | None) -> date | None:
    """
    Read the last day of a point's equity curve, when it can be read.

    CapSweep pins every lazy size-cap simulation of a population to this day,
    so a cell read after UTC midnight still ends every cap's curve where the
    eager point's ended (see CapSweep). Read BY TYPE, never by truthiness:
    a hand-built point whose frame has no "date" column, an empty frame or a
    non-date value yields None, and the caller then leaves the simulation
    reading today (UTC) as before.

    Args:
        point (SweepPoint | None): The eager point, or None when there is none.

    Returns:
        date | None: The curve's last "date" (a datetime reduced to its date),
            or None when point is None or its curve carries no readable date.
    """
    if point is None:
        return None
    df = point.equity_df
    if not isinstance(df, pd.DataFrame) or "date" not in df.columns or df.empty:
        return None
    last = df["date"].iloc[-1]
    if isinstance(last, datetime):
        return last.date()
    return last if isinstance(last, date) else None


def _sharing_floor(point: SweepPoint | None) -> float | None:
    """
    The size cap at and above which a point's simulation is the same at every cap.

    Args:
        point (SweepPoint | None): A simulated point, or None.

    Returns:
        float | None: The larger of its peak_kelly_fraction and its
            cap_free_from (the peak alone when cap_free_from is None, as on a
            hand-built point); None when there is no point or no peak.
    """
    if point is None or point.peak_kelly_fraction is None:
        return None
    if point.cap_free_from is None:
        return point.peak_kelly_fraction
    return max(point.peak_kelly_fraction, point.cap_free_from)


@dataclass
class CapSweep:
    """
    Every per-trade size cap of one sweep, simulated ON DEMAND, one (band, k) cell at a time.

    run_backtest_sweep simulates every band x k scenario eagerly at the run's
    own cap (config.BUDGET_FRACTION), exactly as it did before the size cap
    existed. The other caps of SIZE_CAP_SWEEP are simulated only here, when a
    reader asks for a cell: keeping every band x k x cap x population point
    would hold gigabytes of trades and curves (an estimated 2.5-5 GB; one
    BacktestTrade retains ~2.3 KB and one equity curve ~27 KB at 370 rows,
    ~114 KB at 2,463 rows, measured 2026-09-26 with tracemalloc on synthetic
    records). The reader — the dashboard — is meant
    to summarise each cell and drop it; nothing here is memoised, so asking
    for a cell twice simulates it twice.

    Seeded from the eager point. A point's peak_kelly_fraction (its "peak"
    below; with a walked book, the larger cap_free_from — _sharing_floor
    reads whichever applies) does not depend on the cap, and every cap at or
    above it sizes the point's entries identically — provided
    SAME_TITLE_SIZE_CAP, read at call time, is not rebound between the eager
    point and the cell. So:
      * the primary cap always returns the eager object ITSELF;
      * when the primary cap is itself at or above the eager point's peak,
        every other cap at or above that peak returns a
        dataclasses.replace(eager, size_cap=cap) copy sharing its trades,
        curve, halves and top-event check;
      * when the primary cap is BELOW the peak (the eager point was capped),
        the eager point does NOT size as the larger caps do — the first cap
        at or above the peak is simulated, and every larger cap shares THAT
        point's objects instead;
      * every other cap below the peak is simulated.
    Sharing is exact. The peak covers every Monday a pair may be traded on,
    so at any cap at or above it no size depends on the cap, and the whole
    walk — cash, open markets, ladders, the Monday each pair trades on — is
    the same. A simulation that walks synthetic books shares only from its
    cap_free_from (_sharing_floor), since there the cap also bounds how deep
    the live enrichment averages a book. The halves and the
    excluding-top-event run use subsets of the point's candidates, whose
    floors are no higher.
    A simulated cap runs quiet — its completion line and its
    premise-violation WARNING go to DEBUG, since the primary-cap run already
    reported the same count and ~10 repeats per cell would flood the log
    (TS-02) — and it is pinned to the eager point's end date
    (_curve_end_date): _build_equity_curve otherwise runs every curve to
    today (UTC) as read at simulation time, so a cell read after UTC
    midnight (a dashboard walks every cell long after the run) would give
    its simulated caps one more row than the eager point and the copies
    sharing its curve, and neighbouring caps' Sharpe/Sortino would cover
    different spans. The eager points themselves still read the clock as
    they are simulated, exactly as before the cap sweep, so an eager phase
    that straddles UTC midnight can end two cells' (or two populations')
    curves on different days; every cap of ONE population ends on the same
    day. A population with no eager point (a hand-built CapSweep) reads the
    clock at simulation time.

    Retention (TS-07's residency rule, declared): the sweep keeps
    entries_by_band alive for the reader: one small entry dict per band per
    entry, and every entered pair's two market dicts. Every band's entered
    PAIRS are a subset of the no-band band's (see the pre-pass comment in
    _sweep_from_candidates), so the market dicts are bounded by the no-band
    band's entries: 405 (399 time-series + 6 same-title; 60-405 across the
    36 bands) on the 2026-09-26 01:22 365-day run's log, hence at most ~810
    distinct market dicts — a few MB. Nothing else from the corpus is kept;
    the eager points it references are the ones BacktestSweep already
    holds. Each entry also carries one small dict per later qualifying
    Monday; those point at the pair's own two market dicts, so they add no
    market dicts. Each record's "leg_quotes" (the prices that value its
    trades at market) holds one LegQuotes per market, shared by every record
    and band: about 47 bytes per market per day of its life in the window
    (two asks, two bids, a 24-hour volume and a paid-out marker a day, and
    the same at each checkpoint; the depth model is one object shared by
    every market), and no candle list. The quotes depend on no cap, k, band,
    tier setting or population, so the reuse rule below stays exact.

    One Tier floors setting per CapSweep (tier_floors). The tier-on one
    (BacktestSweep.cap_sweep, tier_floors True) holds the tier-on
    entries_by_band and is seeded from tier-on points only; the tier-off one
    (BacktestSweep.tier_off_cap_sweep, tier_floors False, built only when the
    tier-floors-off family ran) holds the family's entries at the bands a
    deadline-gap tier binds at, and is seeded from the family's own points
    (BacktestSweep.tier_off_scenarios) only — _sweep_from_candidates keeps
    the two seed maps apart, so a cell can never hand one setting's
    simulation back under the other's (band, k, population) key. Every
    simulation it runs, and its split-half and excluding-top-event checks,
    carry its tier_floors (forwarded through _sim_options, so only a False is
    ever forwarded), and every point it returns is stamped with it: a
    simulated one by _simulate_at_discount, a shared copy by the point it
    copies. The tier-off one has no same-title population of its own
    (st_entries is [] — the same-title entries never read the tiers, so the
    tier-on sweep's same_title() serves both) and runs the band sweep's
    checks (its family exists only on a band sweep).

    A CapSweep with add_to_held runs every simulation, and its checks, with
    _simulate_at_discount(add_to_held=True). No eager run adds to held pairs,
    so such a sweep has no eager point to seed from: it refuses to be built
    with any (eager must be {} and same_title_eager None), since an eager
    point would be handed back as one of its own. Each cap is simulated in
    turn until one reaches the peak of the point it simulated, and every
    larger cap shares that point; each cell's curves end on the day end_dates
    gives it, and a cell end_dates has no day for is refused rather than ended
    on today. Sharing stays exact, because the cap reaches an add-on's size
    only through min(pair cap, f*), as it reaches any trade's, and the peak
    covers every Monday an add-on may be made on.

    A CapSweep with sell_at sells a position early at that share of its
    potential profit in every simulation it runs, and in its checks
    (_simulate_at_discount's sell_at); it may also add to held pairs. No
    eager run sells either, so it takes no eager point and needs end_dates
    exactly as an add-to-held sweep does. Sharing stays exact: selling
    changes neither which Mondays a pair may trade on nor its Kelly fraction
    (peak_kelly_fraction is the no-selling run's), and at caps at or above
    the peak every size, and so every position and sale, is the same. A
    selling sweep may also hold back every sale that comes fewer than
    sell_min_days days before the position's last market stops trading
    (_simulate_at_discount's sell_min_days), in every simulation and check
    it runs; the days rule reads only the positions' close dates and the
    checkpoint date, which no cap moves, so sharing stays exact with it too.

    Retention of the tier-off one, declared the same way: it keeps the
    family's entries_by_band — one entry dict per binding band per tier-off
    entry, each band's tier-off time-series entries followed by the shared
    same-title ones (the SAME dict objects the tier-on sweep holds). Its
    market dicts are the corpus records the entries point at, shared with
    the tier-on entries of any pair that also enters with the tiers on; with
    the tiers off a binding band's floor alone gates the spread, so its
    entered pairs are a superset of the same band's tier-on ones, all bounded
    by the tier-off no-band band's. Measured 2026-09-26 on the golden fixture
    (tests/test_backtester.py's TestPrepareEntriesGolden, 36 bands, 18
    binding, ladders on): the tier-on sweep holds 108 entries over its 36
    bands (73 distinct entry dicts, 10 distinct market dicts); the tier-off
    one holds 63 over its 18 (46 distinct entry dicts, 45 of them new — the
    one same-title entry is shared) and ZERO new market dicts, since every
    pair the family enters also enters at some tier-on band there. On any
    corpus the new market dicts are only those of pairs that enter with the
    tiers off and at no tier-on band, so they are bounded by that corpus's
    own tier-off-only pairs at the no-band band — a bound to re-measure per
    corpus. On the DR-73 calibration corpus (start 2020-01-01, ladders on;
    see CLAUDE.md's tier-off paragraph) that is 278 pairs, i.e. at most 556
    new dicts (derived, not measured) beside at most 660 for its 330 tier-on
    no-band entries, a few MB. The ~810 tier-on figure above is a different
    corpus's (the 365-day run's), whose tier-off-only count was not measured.
    The later-Monday dicts of its entries add no market dicts either.

    Attributes:
        caps (tuple[float, ...]): Ascending: SIZE_CAP_SWEEP with the primary
            cap unioned in (1.0 = no cap).
        primary_cap (float): The run's own resolved cap — every eager point's
            size_cap.
        bands (tuple[tuple[float, float], ...]): The resolved bands the sweep
            simulated, ascending.
        ks (tuple[float, ...]): The k grid every band was simulated on,
            ascending, the primary k included.
        primary_k (float): The resolved primary k — the nominal k of the
            same-title population, as on BacktestSweep.same_title_point.
        start_date (date): The backtest's start date.
        initial_balance (float): The balance every simulation starts from, in
            dollars.
        split_date (date | None): BacktestSweep.split_date — the date the
            split-half checks split at; None without a band sweep.
        checks (bool): Whether the band sweep ran: cells then carry the
            "time_series"/"ladder"/"cross" populations and the "all" and
            "time_series" points their halves and top-event checks, and
            same_title() has a population to return.
        entries_by_band (dict): Band -> that band's entries (time-series
            then same-title), exactly as the eager loop simulated them. Not in
            repr.
        st_entries (list): The same-title entries (band-independent). Not in
            repr.
        eager (dict): (band, k, population) -> the eager primary-cap point
            of that cell, recorded by _sweep_from_candidates — always a point
            of this sweep's own Tier floors setting (tier_floors) and add-on
            setting, so {} when add_to_held. Not in repr.
        same_title_eager (SweepPoint | None): BacktestSweep.same_title_point
            (None on the tier-off sweep and on an add-to-held sweep). Not in
            repr.
        tier_floors (bool): Whether this sweep's entries were detected with
            the deadline-gap tier floors applied (True, default) or not
            (False: the tier-floors-off family's). Forwarded to every
            simulation and check it runs, only when False.
        add_to_held (bool): Whether every simulation it runs, and its checks,
            may add to a pair still held (_simulate_at_discount's add_to_held).
            Forwarded only when True. Default False.
        sell_at (float | None): The share of potential profit at which every
            simulation it runs, and its checks, sells a position early
            (_simulate_at_discount's sell_at). Forwarded only when set.
            Default None (never sells).
        sell_min_days (int | None): The fewest days a position must have
            left before its last market stops trading for every simulation
            it runs, and its checks, to sell it (_simulate_at_discount's
            sell_min_days). Needs sell_at. Forwarded only when set. Default
            None (no minimum).
        end_dates (dict): (band, k, population) -> the day that population's
            curves end on, for a cell with no eager point (the eager map is
            empty for a sweep whose setting no eager run shares: one that
            adds to held pairs or sells early); a cell with an eager point
            ends where that point's curve ended. A cell in neither map ends
            on today (UTC) on a sweep that neither adds to held pairs nor
            sells, and is refused (ValueError) on one that does. The keys
            must be the exact band and k objects the cells are read with.
            Not in repr.
        simulated (int): Simulations this object has run (each cap point
            counts once; its halves and top-event re-simulations are not
            counted separately).
        reused (int): Cap points returned as a copy of another point rather
            than simulated. The primary-cap identity is counted in neither.
    """
    caps: tuple[float, ...]
    primary_cap: float
    bands: tuple[tuple[float, float], ...]
    ks: tuple[float, ...]
    primary_k: float
    start_date: date
    initial_balance: float
    split_date: date | None
    checks: bool
    entries_by_band: dict = field(repr=False)
    st_entries: list = field(repr=False)
    eager: dict = field(repr=False)               # (band, k, population) -> primary-cap point
    same_title_eager: SweepPoint | None = field(default=None, repr=False)
    tier_floors: bool = True
    # Whether every simulation adds to held pairs, forwarded like tier_floors
    add_to_held: bool = False
    # (band, k, population) -> the day a cell's curves end when it has no
    # eager point to take it from
    end_dates: dict = field(default_factory=dict, repr=False)
    simulated: int = 0
    reused: int = 0
    # The share of potential profit that sells a position early (None: never)
    sell_at: float | None = None
    # The fewest days before maturity a sale needs (None: no minimum)
    sell_min_days: int | None = None

    def __post_init__(self) -> None:
        """
        Refuse an add-to-held or selling sweep that was handed eager points, or a bad sell setting.

        Every eager point was simulated without adding to held pairs and
        without selling, and _by_cap returns the eager point itself at the
        primary cap (and copies of it above its peak), so such a sweep holding
        one would hand back points that never added or sold, stamped as if
        they had.

        Raises:
            ValueError: If add_to_held or sell_at is set and eager is not
                empty or same_title_eager is not None, if sell_at is not a
                share in (0, 1], or if sell_min_days is set without sell_at
                or is not a whole number of at least 1.
        """
        if self.sell_at is not None:
            # Checked as the simulation checks it, before any cell is read
            _resolve_sell_at(self.sell_at)
        # The days rule, checked the same way (it needs a sell level)
        _resolve_sell_min_days(self.sell_min_days, self.sell_at)
        if ((self.add_to_held or self.sell_at is not None)
                and (self.eager or self.same_title_eager is not None)):
            raise ValueError(
                "a CapSweep that adds to held pairs or sells early simulates every cap "
                "itself: it takes no eager points (eager must be {} and same_title_eager "
                "None)")

    def _by_cap(
        self,
        subset: list[dict],
        band: tuple[float, float] | None,
        k: float,
        population: str,
        eager_point: SweepPoint | None,
    ) -> dict[float, SweepPoint]:
        """
        Simulate (or share) one population of one cell at every cap.

        The eager branch comes FIRST, so the primary cap always returns the
        eager object itself; then the eager seed (only when the primary cap
        is at or above the cap the eager point shares from, _sharing_floor —
        see the class docstring); then a point this call simulated at a cap
        at or above the cap it shares from;
        else a quiet simulation, pinned to the eager point's end date (with
        no eager point, to end_dates' day for this cell) and run at this
        sweep's Tier floors, add-on and sell settings (tier_floors,
        add_to_held, sell_at, sell_min_days), with the split-half and
        top-event checks — at those settings too — when the band sweep ran
        and the population carries them.

        Args:
            subset (list[dict]): The population's entries.
            band (tuple[float, float] | None): The resolved band (None for
                the same-title population, as on its eager point).
            k (float): The resolved interval discount.
            population (str): The population label.
            eager_point (SweepPoint | None): The eager primary-cap point of
                this (band, k, population), or None when the sweep recorded
                none (a hand-built CapSweep, or one that adds to held pairs,
                which holds no eager points).

        Returns:
            dict[float, SweepPoint]: cap -> point, one per self.caps entry,
                each stamped with its own size_cap (and, simulated or copied,
                with this sweep's tier_floors and add_to_held).

        Raises:
            ValueError: If this sweep adds to held pairs or sells early and
                end_dates has no day for (band, k, population): its curves
                would otherwise end on whatever day (UTC) the cell happens to
                be read.
        """
        out: dict[float, SweepPoint] = {}
        # The eager point is the one the backtest simulated during the run, at
        # its own cap; the cap it shares from does not depend on the cap, so
        # it is this subset's too
        seed = _sharing_floor(eager_point)
        # Every simulated cap ends its curve where the eager point's ended, not
        # on whatever day (UTC) this cell happens to be read; a cell with no
        # eager point takes its day from end_dates
        if eager_point is not None:
            end_date = _curve_end_date(eager_point)
        else:
            end_date = self.end_dates.get((band, k, population))
            if end_date is None and (self.add_to_held or self.sell_at is not None):
                # An add-on or selling sweep has no eager point to fall back
                # on, so a missing day (a key off by float noise included) is
                # refused
                raise ValueError(
                    f"CapSweep {'adds to held pairs' if self.add_to_held else 'sells early'} "
                    f"but end_dates has no day for (band, k, population) = "
                    f"{(band, k, population)!r}")
        # The eager point sizes as every cap at or above the seed ONLY if its
        # own cap is at or above it too — a capped eager point does not
        eager_seeds = seed is not None and self.primary_cap >= seed
        reuse: SweepPoint | None = None
        for cap in self.caps:
            if eager_point is not None and cap == self.primary_cap:
                point = eager_point                              # identity, always
            elif eager_seeds and cap >= seed:
                out[cap] = replace(eager_point, size_cap=cap)   # sizes as the eager point
                self.reused += 1
                continue
            elif reuse is not None:
                out[cap] = replace(reuse, size_cap=cap)
                self.reused += 1
                continue
            else:
                # size_cap explicit (see _sim_options: a cap equal to
                # BUDGET_FRACTION must still name itself); quiet, the pinned
                # end date, this sweep's tier setting and its add-on setting
                # through _sim_options, so a tier-on sweep that never adds
                # forwards neither
                point = _simulate_at_discount(
                    subset, self.start_date, self.initial_balance, k=k, spread_band=band,
                    population=population, size_cap=cap,
                    **_sim_options(None, True, end_date=end_date,
                                   tier_floors=self.tier_floors,
                                   add_to_held=self.add_to_held,
                                   sell_at=self.sell_at,
                                   sell_min_days=self.sell_min_days))
                self.simulated += 1
                if self.checks and population in _CHECKED_POPULATIONS:
                    point.halves = _half_split(
                        _split_halves(subset, self.split_date), self.start_date,
                        self.initial_balance, k, band, population=population,
                        tier_floors=self.tier_floors, size_cap=cap, quiet=True,
                        end_date=end_date, add_to_held=self.add_to_held,
                        sell_at=self.sell_at, sell_min_days=self.sell_min_days)
                    point.ex_top_event = _ex_top_event(
                        point, subset, self.start_date, self.initial_balance, band,
                        population=population, tier_floors=self.tier_floors, quiet=True,
                        end_date=end_date)
                # The first simulated point whose cap reaches the cap it
                # shares from sizes as every larger cap (never set while the
                # eager point seeds, whose floor every simulated cap here
                # sits below)
                floor = _sharing_floor(point)
                if reuse is None and floor is not None and cap >= floor:
                    reuse = point
            out[cap] = point
        return out

    def cell(self, band: tuple[float, float], k: float) -> dict[float, dict[str, SweepPoint]]:
        """
        Every cap's points for one (band, k) cell.

        Args:
            band (tuple[float, float]): One of self.bands.
            k (float): One of self.ks.

        Returns:
            dict[float, dict[str, SweepPoint]]: cap -> population -> point.
                "all" always; with checks, also "time_series", "ladder" and
                "cross", each only when non-empty at this band — the eager
                loop's rule, through the same _population_subsets. The
                primary cap's points are the eager objects themselves, where
                the sweep has an eager point.

        Raises:
            KeyError: If band is not one of self.bands.
            ValueError: If this sweep adds to held pairs and end_dates has no
                day for one of the cell's populations (see _by_cap).
        """
        entries = self.entries_by_band[band]
        subsets = [("all", entries)]
        if self.checks:
            subsets += [(label, sub) for label, sub in _population_subsets(entries) if sub]
        out: dict[float, dict[str, SweepPoint]] = {cap: {} for cap in self.caps}
        for population, subset in subsets:
            points = self._by_cap(subset, band, k, population,
                                  self.eager.get((band, k, population)))
            for cap, point in points.items():
                out[cap][population] = point
        return out

    def same_title(self) -> dict[float, SweepPoint]:
        """
        The same-title entries simulated alone, at every cap.

        Band- and k-independent like BacktestSweep.same_title_point (simulated
        at the primary k, with no band), which is the primary cap's point when
        the sweep holds it (same_title_eager); a sweep that adds to held pairs
        holds none and simulates every cap.

        Returns:
            dict[float, SweepPoint]: cap -> point; {} without checks (no band
                sweep) or without same-title entries, when the eager sweep
                had no same-title population either.

        Raises:
            ValueError: If this sweep adds to held pairs and end_dates has no
                day for (None, primary_k, "same_title") (see _by_cap).
        """
        if not (self.checks and self.st_entries):
            return {}
        return self._by_cap(self.st_entries, None, self.primary_k, "same_title",
                            self.same_title_eager)

    def entry_events(self) -> set[tuple[str, str]]:
        """
        Every (event ticker, fallback category) any cell's trades could carry.

        Built the way _simulate_at_discount fills BacktestTrade.event_ticker and
        .category (market A's event ticker, and infer_category of it), but for
        every qualifying Monday of every band's entries, so the dashboard can
        list every category and tag before any cell is simulated. It is a
        superset of the events the trades carry: it includes pairs no cell
        trades, and both events of a same-title pair whose market A changes
        between Mondays.

        Returns:
            set[tuple[str, str]]: (event ticker, category) pairs; the ticker
                is "" for an entry that carries none.
        """
        return _entry_events(self.entries_by_band)


def _entry_events(entries_by_band: dict) -> set[tuple[str, str]]:
    """
    Every (event ticker, fallback category) a simulation of some entries could file a trade under.

    Market A's event ticker on every qualifying Monday of every entry, and
    infer_category of it — what _simulate_at_discount fills
    BacktestTrade.event_ticker and .category with — so a report can list
    every category and tag before a cell is simulated (CapSweep.entry_events,
    SellSweep.entry_events).

    Args:
        entries_by_band (dict): Band -> that band's entry records.

    Returns:
        set[tuple[str, str]]: (event ticker, category) pairs; the ticker is ""
            for an entry that carries none.
    """
    return {(monday["mA"].get("event_ticker") or "",
             # infer_category maps the event-ticker prefix to the fallback
             # label BacktestTrade.category carries (e.g. "Crypto")
             infer_category(monday["mA"].get("event_ticker", "")))
            for entries in entries_by_band.values() for rec in entries
            # Every Monday a cell could enter the pair on
            for monday in _entry_mondays(rec["entry"])}


@dataclass(frozen=True)
class SameSale:
    """
    A sell cell that is exactly a run SellSweep.sold_grid already yielded, at the same size cap.

    sold_grid yields one of these, instead of a point, for a (level, minimum
    days, cap) that walks exactly as an earlier (level, min_days) of the
    same cell and cap that it simulated and yielded as a point (_sale_cover
    says why the two are one run). A reader reuses whatever it made of that
    earlier point; nothing is simulated or kept for this one.

    Attributes:
        level (float): The sell level of the earlier, simulated cell.
        min_days (int): Its minimum of days before maturity.
    """
    level: float
    min_days: int


@dataclass(frozen=True)
class SellSweep:
    """
    The dashboard's "Sell" family: every sell level and minimum of days of every scenario, simulated on demand.

    The backtest dashboard's Sell select offers "no selling" (the scenario as
    simulated) and each of `levels` (config.TAKE_PROFIT_LEVELS): sell a whole
    position once its realized profit has stayed at or above that share of
    its potential profit for config.TAKE_PROFIT_HOLD_DAYS days in a row
    (_simulate_at_discount's sell_at) — and, for each of `min_days`
    (config.TAKE_PROFIT_MIN_DAYS), only while at least that many days remain
    before the position's last market stops trading (sell_min_days). It
    covers every scenario the filter bar shows — every band, Tier floors
    setting, k, size cap and Add to held pairs setting. Nothing is simulated
    while the backtest runs: sweep() builds, for one level, minimum,
    Tier floors setting and add-on setting, a CapSweep over the same entries
    the size-cap and add-on families hold, and cell() reads one (band, k)
    cell of it, every cap, the "all" population only (what the filter bar
    shows). Like the add-on family it has no eager point to start from, so
    each cell's curves end on the day its eager twin's did (end_dates /
    off_end_dates), and each cap is simulated until one reaches the peak
    Kelly fraction, which every larger cap shares.

    sold_grid() reads one (band, k) cell at every level and minimum while
    simulating as few of them as it can, and exactly: a (level, minimum) no
    position can sell at is the no-selling run (_sale_reach), and one at
    least as strict as a run it already simulated, when every sale of that
    run also meets it, is that run (_sale_cover). Every other one is
    simulated.

    Retention: none of its own. Its entry maps are the very objects
    BacktestSweep's size-cap and add-on families already hold.

    Attributes:
        levels (tuple[float, ...]): The sell levels, ascending, distinct,
            each in (0, 1] (_resolve_sell_levels).
        min_days (tuple[int, ...]): The minimum-days options, ascending,
            each a whole number of at least 1 (_resolve_min_days_options).
        caps (tuple[float, ...]): The size caps of every scenario (the
            size-cap sweep's, or the run's own alone).
        primary_cap (float): The run's own cap.
        bands (tuple): The tier-on bands.
        off_bands (tuple): The bands a deadline-gap tier binds at, re-run
            with the tier floors off; () without the tier-floors-off family.
        ks (tuple[float, ...]): The k grid.
        primary_k (float): The run's own k.
        start_date (date): The backtest's start date.
        initial_balance (float): The balance every simulation starts from.
        entries_by_band (dict): Band -> tier-on entries. Not in repr.
        off_entries_by_band (dict): Binding band -> tier-off entries. Not in repr.
        end_dates (dict): (band, k, "all") -> the day the tier-on eager
            point's curve ended. Not in repr.
        off_end_dates (dict): The same for the tier-off family. Not in repr.
    """
    levels: tuple[float, ...]
    min_days: tuple[int, ...]
    caps: tuple[float, ...]
    primary_cap: float
    bands: tuple
    off_bands: tuple
    ks: tuple[float, ...]
    primary_k: float
    start_date: date
    initial_balance: float
    entries_by_band: dict = field(repr=False)
    off_entries_by_band: dict = field(repr=False)
    end_dates: dict = field(repr=False)
    off_end_dates: dict = field(repr=False)

    def sweep(self, level: float | None, *, min_days: int | None = None,
              tier_floors: bool = True, add_to_held: bool = False,
              caps: Iterable[float] | None = None) -> CapSweep:
        """
        The lazy size-cap sweep that sells at `level` under one minimum of days, Tier floors and add-on setting.

        Args:
            level (float | None): One of self.levels, or None for the same
                sweep never selling (the run each level is measured against).
            min_days (int | None): Keyword-only. One of self.min_days: sell
                only while at least that many days remain before the
                position's last market stops trading. None (default) sets no
                minimum.
            tier_floors (bool): Keyword-only. False for the tier-floors-off
                family's binding bands.
            add_to_held (bool): Keyword-only. Whether every simulation may
                also add to a pair it still holds.
            caps (Iterable[float] | None): Keyword-only. The caps to simulate,
                each one of self.caps; None (default) means every one. A cap
                left out is not simulated, and caps at or above a point's peak
                still share one simulation among those kept.

        Returns:
            CapSweep: Over the tier-on or tier-off entries, the "all"
                population only, selling at `level` with that minimum.

        Raises:
            ValueError: For a level not in self.levels, a minimum not in
                self.min_days or one given without a level, a cap not in
                self.caps, or tier floors off on a run without the
                tier-floors-off family.
        """
        if level is not None and level not in self.levels:
            raise ValueError(f"sell level {level!r} is not one of {self.levels}")
        if min_days is not None:
            if level is None:
                raise ValueError("a minimum of days before maturity needs a sell level: "
                                 "a sweep that never sells has no sale to hold back")
            if (isinstance(min_days, (bool, np.bool_))
                    or min_days not in self.min_days):
                raise ValueError(f"minimum days {min_days!r} is not one of {self.min_days}")
        if not tier_floors and not self.off_bands:
            raise ValueError("this run has no tier-floors-off family to sell in")
        chosen = self.caps if caps is None else tuple(sorted(set(caps)))
        if any(cap not in self.caps for cap in chosen):
            raise ValueError(f"size caps {chosen!r} are not all among {self.caps}")
        bands = self.bands if tier_floors else self.off_bands
        return CapSweep(
            caps=chosen, primary_cap=self.primary_cap, bands=tuple(bands), ks=self.ks,
            primary_k=self.primary_k, start_date=self.start_date,
            initial_balance=self.initial_balance, split_date=None, checks=False,
            entries_by_band=self.entries_by_band if tier_floors else self.off_entries_by_band,
            st_entries=[], eager={}, tier_floors=tier_floors, add_to_held=add_to_held,
            end_dates=self.end_dates if tier_floors else self.off_end_dates,
            sell_at=level, sell_min_days=min_days)

    def cell(self, level: float, band: tuple[float, float], k: float, *,
             min_days: int | None = None, tier_floors: bool = True,
             add_to_held: bool = False) -> dict[float, dict[str, SweepPoint]]:
        """
        One (band, k) cell, every size cap, selling at `level` with an optional minimum of days.

        Args:
            level (float): One of self.levels.
            band (tuple[float, float]): One of the setting's bands.
            k (float): One of self.ks.
            min_days (int | None): Keyword-only. One of self.min_days, or None
                (default) for no minimum.
            tier_floors (bool): Keyword-only. The Tier floors setting.
            add_to_held (bool): Keyword-only. The Add to held pairs setting.

        Returns:
            dict[float, dict[str, SweepPoint]]: cap -> {"all": point}.

        Raises:
            ValueError: As sweep(); and from CapSweep, for a cell with no end day.
            KeyError: For a band the setting does not hold.
        """
        return self.sweep(level, min_days=min_days, tier_floors=tier_floors,
                          add_to_held=add_to_held).cell(band, k)

    def entry_events(self) -> set[tuple[str, str]]:
        """
        Every (event ticker, fallback category) a sell run's trades could carry.

        Over both Tier floors settings' entries, every qualifying Monday of
        each (_entry_events): selling frees cash, so a sell run can trade a
        pair no other scenario traded, and a report must list its category and
        tag before any cell is simulated.

        Returns:
            set[tuple[str, str]]: (event ticker, category) pairs.
        """
        return _entry_events(self.entries_by_band) | _entry_events(self.off_entries_by_band)

    def for_band(self, band: tuple[float, float], *, tier_floors: bool = True) -> "SellSweep":
        """
        This family narrowed to one band of one Tier floors setting.

        A dashboard worker process receives one band's family, never every
        band's entries: the copy holds that band's entries and end days alone
        (the same objects, before pickling), and every other field unchanged.

        Args:
            band (tuple[float, float]): One of self.bands (tier floors on) or
                self.off_bands (off).
            tier_floors (bool): Keyword-only. Which setting's entries to keep.

        Returns:
            SellSweep: A copy whose only band, of that setting, is `band`.

        Raises:
            KeyError: For a band the setting does not hold.
        """
        days = self.end_dates if tier_floors else self.off_end_dates
        kept = {key: day for key, day in days.items() if key[0] == band}
        if tier_floors:
            return replace(self, bands=(band,), off_bands=(),
                           entries_by_band={band: self.entries_by_band[band]},
                           off_entries_by_band={}, end_dates=kept, off_end_dates={})
        return replace(self, bands=(), off_bands=(band,), entries_by_band={},
                       off_entries_by_band={band: self.off_entries_by_band[band]},
                       end_dates={}, off_end_dates=kept)

    def sold_grid(self, band: tuple[float, float], k: float, *, tier_floors: bool = True,
                  add_to_held: bool = False, caps: Iterable[float] | None = None,
                  stats: dict | None = None,
                  ) -> Iterator[tuple[float, int,
                                      dict[float, SweepPoint | SameSale | None]]]:
        """
        Every (sell level, minimum of days) of one (band, k) cell, simulating only the runs that differ.

        First the cell is simulated without selling, at every cap asked for,
        and _sale_reach reads each distinct run once (caps that share one run
        share its reading): for each level, the most days before maturity at
        which one of its positions would have sold there.
        Then the levels are read in ascending order, and each level's
        minimums in ascending order, and every (level, minimum, cap) is one
        of three things:
          * None — the no-selling run itself: the level is not in the cap's
            reach, or the minimum is above it, so the run never sells
            (_sale_reach), and nothing is simulated;
          * SameSale(level0, min_days0) — exactly the run already yielded as
            a point at that earlier (level0, min_days0) and the same cap:
            this level is at or above level0 and at or below the highest
            level all of that run's sales reached, and this minimum is at or
            above min_days0 and at or below the fewest days any of its sales
            had left (_sale_cover). Both days bounds are needed: a run from
            an earlier level can have a larger minimum than this one;
          * a point — simulated, the caps at or above a point's peak sharing
            one run (CapSweep). Every such run sells at least once, since
            its minimum is within the cap's reach (_sale_reach); a run that
            did not would mean the replay and the walk disagree, and raises.
        Only one (level, minimum)'s points are alive at a time: between
        yields the generator keeps, per simulated run, its level, minimum
        and _sale_cover — never a point.

        Args:
            band (tuple[float, float]): One of the setting's bands.
            k (float): One of self.ks.
            tier_floors (bool): Keyword-only. The Tier floors setting.
            add_to_held (bool): Keyword-only. The Add to held pairs setting.
            caps (Iterable[float] | None): Keyword-only. The caps to cover,
                each one of self.caps; None (default) means every one.
            stats (dict | None): Keyword-only. When given, its "simulated"
                and "reused" counts are raised by the cap points this call
                simulated and shared (CapSweep's counters, the no-selling
                runs included), its "same" count by the cells yielded as
                SameSale and its "pruned" count by those yielded as None, for
                a report's log line.

        Yields:
            tuple[float, int, dict[float, SweepPoint | SameSale | None]]:
                (level, minimum, cap -> the "all" point selling at that level
                with that minimum, a SameSale naming the earlier point it
                equals, or None where it equals the no-selling run), for
                every level of self.levels and every minimum of
                self.min_days, levels ascending and each level's minimums
                ascending.

        Raises:
            ValueError: As sweep(); and from CapSweep, for a cell with no end day.
            KeyError: For a band the setting does not hold.
            RuntimeError: When a run simulated because the no-selling run
                showed a position that would sell there made no sale.
        """
        chosen = self.caps if caps is None else tuple(sorted(set(caps)))
        if stats is not None:
            for key in ("simulated", "reused", "same", "pruned"):
                stats.setdefault(key, 0)

        def read(level: float | None, min_days: int | None, wanted) -> dict:
            """One sweep's cell at (level, min_days) over the caps wanted, counted into stats."""
            swept = self.sweep(level, min_days=min_days, tier_floors=tier_floors,
                               add_to_held=add_to_held, caps=wanted)
            cell = swept.cell(band, k)
            if stats is not None:
                stats["simulated"] += swept.simulated
                stats["reused"] += swept.reused
            return cell

        base = read(None, None, chosen)
        # For each cap, the most days left at which each level sells
        # somewhere. Caps at or above a run's sharing floor hold copies of one
        # run, with the same trades list, and _sale_reach's answer depends on
        # the trades alone, so each distinct run is replayed once
        replayed: dict[int, dict[float, int]] = {}
        reach: dict[float, dict[float, int]] = {}
        for cap, pops in base.items():
            point = pops["all"]
            if id(point.trades) not in replayed:
                replayed[id(point.trades)] = _sale_reach(point, self.levels)
            reach[cap] = replayed[id(point.trades)]
        del base, replayed
        # Per cap, every run simulated so far: (level index, minimum, its _sale_cover)
        runs: dict[float, list[tuple[int, int, tuple[int, int]]]] = {cap: [] for cap in chosen}

        def cells_at(li: int, level: float, min_days: int) -> dict:
            """Every cap of one (level, minimum): None, a SameSale or a simulated point."""
            out: dict[float, SweepPoint | SameSale | None] = {}
            needed = []
            for cap in chosen:
                top = reach[cap].get(level)
                if top is None or min_days > top:
                    # No position of the no-selling run sells here
                    out[cap] = None
                    continue
                # An earlier run of this cap that makes exactly these sales;
                # every earlier run's level index is at or below li
                out[cap] = next((SameSale(self.levels[i0], d0)
                                 for i0, d0, (fewest, top_index) in runs[cap]
                                 if d0 <= min_days <= fewest and li <= top_index), None)
                if out[cap] is None:
                    needed.append(cap)
            if needed:
                sold = read(level, min_days, needed)
                for cap in needed:
                    point = sold[cap]["all"]
                    if not point.sales:
                        raise RuntimeError(
                            f"the run selling at {level!r} with at least {min_days} day(s) "
                            f"before maturity at cap {cap!r} made no sale, though the "
                            "no-selling run has a position that would sell there")
                    runs[cap].append((li, min_days, _sale_cover(point.sales, self.levels)))
                    out[cap] = point
            if stats is not None:
                stats["same"] += sum(isinstance(v, SameSale) for v in out.values())
                stats["pruned"] += sum(v is None for v in out.values())
            return {cap: out[cap] for cap in chosen}

        for li, level in enumerate(self.levels):
            for min_days in self.min_days:
                # Built inside the call, so between yields this frame holds no point
                yield level, min_days, cells_at(li, level, min_days)


@dataclass
class BacktestSweep:
    """
    Everything one backtest run produces across every interval discount — and,
    when the spread-band sweep is on, across every band of the grid (and, with
    tier_off_sweep, again with the deadline-gap tier floors off at every band
    they bind).

    Returned by run_backtest_sweep(). One preparation pass (the expensive,
    network-bound half, _prepare_candidates) feeds every point here, so the
    whole aggregate costs one fetch, one _find_entry pass per band (and a
    second at each band the tier floors bind, on a tier-off sweep), and one
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
            DEFAULTED, so a hand-built sweep may omit it — but a caller that
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
            run's pairs were extracted and entered under — the flag
            decides which pairs exist, so a report must say which it was.
            None means not recorded (a hand-built sweep).
        split_date (date | None): The date the split-half check (SweepPoint.
            halves) splits entries at — the median_low of the primary band's
            TIME-SERIES entries' first qualifying Mondays, which do not depend
            on k (same-title entries are left out, so they cannot move the split
            the time-series checks are read at), or the window's midpoint when
            that band has none. One date for every scenario and both checked
            populations, so every cell's halves cover the same two stretches
            of history. None when the band sweep is off or the window was
            infeasible.
        corpus_provenance (CorpusProvenance | None): What this run's
            settled-market corpus covers: when it was assembled or last
            extended — it holds nothing settled after that, while the window
            nominally runs to today — whether it came from an earlier run's
            cache (assembled earlier the same UTC day), when it was last
            assembled in full if it was extended since, and the archive
            cutoff AS OF that assembly (information only). It hangs off the
            sweep for the reason label_coverage does: one fact about one
            corpus, valid at every point. The dashboard renders it under the
            Period line whether healthy or not (absence must never be
            the only signal).
            None when not recorded: the feasibility short-circuit (no fetch),
            a test that stubs the fetch with a plain list, or a hand-built
            sweep. DEFAULTED, like
            label_coverage, so a hand-built sweep may omit it; the one
            production construction that has a corpus
            (_sweep_from_candidates) always passes it.
        config_same_event_ladders (bool | None): The ladder switch as this
            process's config set it (backtester's binding of
            config.TIME_SERIES_SAME_EVENT_LADDERS) — the value this
            checkout's live finder would run with — so a report can say
            whether the run replays that rule. None = not recorded (a
            hand-built sweep). DEFAULTED like label_coverage; both
            production constructions must pass it.
        tier_off_scenarios (list[SweepPoint]): The tier-off family: every
            tier-bound band x k cell (_tier_floors_bind — on the shipped grid
            the bands with a floor below 0.30) simulated again with the
            deadline-gap tier floors off, i.e. at the band floor alone, in the
            same order and shape as scenarios — the "all" point, then its
            non-empty "time_series", "ladder" and "cross" points, the "all"
            and "time_series" points carrying halves split at the ONE
            split_date and, where they traded an event, ex_top_event. Every
            point here has tier_floors False. A band whose floor is at or
            above both tiers has no twin: its tier-on cells ARE its tier-off
            cells. [] unless run_backtest_sweep(tier_off_sweep=True) — which
            the CLI turns on together with the band sweep — ran on a feasible
            window (the Monday-feasibility short-circuit returns it empty).
        tier_off_calibrations_by_band (dict): Band -> the tier-off entries'
            own IntervalCalibration (or None), each bucket labelled with the
            floor alone. Its keys are exactly the tier-bound bands, and the
            dashboard reads them as "simulated again": a band that is absent
            AND where _tier_floors_bind is False counts as its own tier-on
            run. {} unless tier_off_sweep ran on a feasible window.
        cap_sweep (CapSweep | None): Every other per-trade size cap of
            SIZE_CAP_SWEEP, simulated lazily, one (band, k) cell at a time,
            by whoever reads it (see CapSweep). None when the size-cap sweep
            was off — run_backtest_sweep's default, and the infeasible
            window. Every point above is the run's own cap
            (config.BUDGET_FRACTION) whether or not this is set; setting it
            simulates nothing during the run itself. Appended with a
            default, so no construction moves.
        tier_off_cap_sweep (CapSweep | None): The same lazy size-cap sweep
            over the tier-floors-off family (tier_floors False): its binding
            bands, every k, every cap, seeded from tier_off_scenarios' own
            points (never a tier-on one, and cap_sweep is never seeded from
            one of these). None unless BOTH cap_sweep and tier_off_sweep ran
            on a feasible window with at least one binding band; like
            cap_sweep it simulates nothing during the run. Its primary-cap
            points ARE the tier_off_scenarios objects. Appended with a
            default, so no construction moves.
        add_on_cap_sweep (CapSweep | None): The "Add to held pairs" family
            the dashboard's filter bar shows: every band x k x cap of the
            run's grid simulated with add_to_held, the "all" population
            only (no split-half or top-event checks), lazily, by whoever
            reads it. It has no eager point — nothing the run simulates adds
            to held pairs — so every cell is simulated when read, ending its
            curves on the day its eager twin's ended. None unless
            run_backtest_sweep(add_on_sweep=True) ran on a feasible window.
            Appended with a default, so no construction moves.
        add_on_tier_off_cap_sweep (CapSweep | None): The same over the
            tier-floors-off family's binding bands (tier_floors False). None
            unless add_on_sweep ran on a feasible window with at least one
            binding band.
        same_title_size_cap (float | None): The extra cap every same-title
            candidate was sized under (this module's SAME_TITLE_SIZE_CAP).
            None only on a hand-built sweep.
        live_tier_floors (bool | None): The saved live defaults' tier_floors,
            from run_backtest_sweep's one fail-soft pre-fetch read
            (_live_settings_for_report); reporting only, blind to main.py's
            per-run overrides. None, with the other live_* fields, when no
            live defaults are saved, the saved file is refused, or on a
            hand-built sweep.
        live_spread_band (tuple[float, float] | None): The saved live
            defaults' spread_band.
        live_categories (tuple[str, ...] | None): The saved live defaults'
            categories; None means any category when live_tier_floors is
            recorded.
        live_tags (tuple[str, ...] | None): The saved live defaults' tags,
            likewise.
        live_origin (str | None): Where the saved live defaults were read
            (LiveSettings.origin: the file, when and from what it was saved),
            from the same read; reporting only.
        live_interval_discount (float | None): The saved live defaults' k,
            from the same read; reporting only (this run sized at its own k,
            primary.k).
        live_size_cap (float | None): The saved live defaults' per-trade cap,
            from the same read; reporting only (this run sized at its own,
            primary.size_cap).
        live_same_title_size_cap (float | None): The saved live defaults'
            same-title cap, from the same read; reporting only (this run sized
            under same_title_size_cap).
        live_add_to_held_pairs (bool | None): Whether the saved live defaults
            add to held pairs, from the same read; reporting only (this run's
            own points never do).
        entry_checkpoint (str | None): SCHEDULED_RUN.label() of the schedule
            the entry checkpoints were placed by (e.g. "Monday 09:00
            America/Los_Angeles"), for the dashboard header. None = not
            recorded (a hand-built sweep).
        sell_sweep (SellSweep | None): The dashboard's "Sell" family: every
            sell level (config.TAKE_PROFIT_LEVELS), each with every minimum of
            days before maturity (config.TAKE_PROFIT_MIN_DAYS), of every
            scenario, simulated when the dashboard reads it. None unless
            run_backtest_sweep(sell_sweep=True) ran on a feasible window.
        depth_model (DepthModel | None): The depth model every trade's
            synthetic order book came from (which snapshots, how many
            ladders, when they were taken), for the page's header; None when
            the run had none, so every trade filled at the top of the book.
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
    corpus_provenance: CorpusProvenance | None = None
    config_same_event_ladders: bool | None = None
    tier_off_scenarios: list[SweepPoint] = field(default_factory=list)
    tier_off_calibrations_by_band: dict[tuple[float, float], IntervalCalibration | None] = field(
        default_factory=dict)
    cap_sweep: CapSweep | None = None
    tier_off_cap_sweep: CapSweep | None = None
    same_title_size_cap: float | None = None
    live_tier_floors: bool | None = None
    live_spread_band: tuple[float, float] | None = None
    live_categories: tuple[str, ...] | None = None
    live_tags: tuple[str, ...] | None = None
    entry_checkpoint: str | None = None
    live_origin: str | None = None
    live_interval_discount: float | None = None
    live_size_cap: float | None = None
    live_same_title_size_cap: float | None = None
    # The "Add to held pairs" family the dashboard's filter bar shows: every
    # band x k x cap of the run's grid simulated with add_to_held, the "all"
    # population only, lazily (None unless run_backtest_sweep(add_on_sweep=True));
    # and the same over the tier-floors-off family's binding bands
    add_on_cap_sweep: CapSweep | None = None
    add_on_tier_off_cap_sweep: CapSweep | None = None
    # The saved live defaults' add_to_held_pairs, recorded with the other live_* fields
    live_add_to_held_pairs: bool | None = None
    # The dashboard's "Sell" family, lazy (None unless run_backtest_sweep(sell_sweep=True))
    sell_sweep: SellSweep | None = None
    # The depth model the trades' synthetic books came from (None: top of the book)
    depth_model: DepthModel | None = None


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
    These are the top-of-the-book prices: _find_entry and Pass 1's Kelly gate
    read them, and a trade built without fill prices paid them
    (_paid_prices).

    Args:
        pair_type (str): "time_series" or "same_title" — anything but the exact
            string "time_series" is treated as same-title, matching
            scanner.leg_sides.
        pA (float): YES ask of market A, dollars in [0.01, 0.99].
        nA (float): NO ask of market A, dollars in [0.01, 0.99].
        pB (float): YES ask of market B, dollars in [0.01, 0.99].
        nB (float): NO ask of market B, dollars in [0.01, 0.99].

    Returns:
        tuple[float, float]: (price_a, price_b) — the quote of the side bought
            on market A and on market B respectively.
    """
    if pair_type == "time_series":
        return pA, nB
    return nA, pB


def _paid_prices(trade: BacktestTrade) -> tuple[float, float]:
    """
    The average price a trade paid per contract on each leg.

    The one reader of what a trade paid: its fill prices, or, for a trade
    built without them, its two leg quotes (_leg_prices_for), which is what
    a top-of-the-book fill pays.

    Args:
        trade (BacktestTrade): A trade.

    Returns:
        tuple[float, float]: (market A's leg, market B's leg), in dollars.
    """
    if trade.fill_price_a is not None and trade.fill_price_b is not None:
        return trade.fill_price_a, trade.fill_price_b
    return _leg_prices_for(trade.pair_type, trade.entry_pA, trade.entry_nA,
                           trade.entry_pB, trade.entry_nB)


def _open_value(trade: BacktestTrade, day: date) -> float:
    """
    What an open trade is worth at the entry checkpoint on `day`: the one valuation Pass 2 sizes on.

    Each leg counts at the latest usable ask of the side it holds at that
    checkpoint (a YES leg at the YES ask, a NO leg at the NO ask), at its
    payout (1 or 0 per contract) once its market has paid out, and at the
    price it paid (_paid_prices) before that side has had any usable ask
    (LegQuotes.at_checkpoint). A trade with no quotes
    (trade.marks is None, every hand-built trade) is worth exactly its
    total_cost, the value the portfolio carried before trades were valued at
    market. Fees are never part of the value: they are spent.

    Args:
        trade (BacktestTrade): An open trade (it entered on or before `day`
            and pays out after it).
        day (date): A checkpoint date on the trade's legs' weekly grid.

    Returns:
        float: The trade's value in dollars.

    Raises:
        ValueError: From LegQuotes.at_checkpoint, for a day off the legs'
            checkpoint grid (the quotes were taken on another schedule).
    """
    if trade.marks is None:
        return trade.total_cost
    return trade.n * (_leg_mark(trade, 0, day) + _leg_mark(trade, 1, day))


def _leg_mark(trade: BacktestTrade, index: int, day: date) -> float:
    """
    What one contract of one leg of an open trade is worth at the checkpoint on `day`.

    The one reader of a leg's checkpoint quote (LegQuotes.at_checkpoint),
    shared by _open_value (both legs) and _open_leg_stake (one leg): the
    latest usable ask of the side the leg holds, its payout once its market
    has paid out, or the price it paid (_paid_prices) before that side has
    had a usable ask, or when the trade has no quotes at all.

    Args:
        trade (BacktestTrade): An open trade.
        index (int): 0 for its market A leg, 1 for its market B leg.
        day (date): A checkpoint date on the legs' weekly grid.

    Returns:
        float: The leg's value per contract, in dollars.

    Raises:
        ValueError: From LegQuotes.at_checkpoint, for a day off the legs'
            checkpoint grid.
    """
    # Which side the leg holds (scanner.leg_sides, the one definition) and
    # what it paid, for a leg that has had no usable ask yet
    side = leg_sides(trade.pair_type)[index]
    price = _paid_prices(trade)[index]
    if trade.marks is None:
        return price
    return trade.marks[index].at_checkpoint(day, side, price)


def _open_leg_stake(trade: BacktestTrade, ticker: str, day: date) -> float:
    """
    What one leg of an open trade stakes at the checkpoint on `day`: its value at market plus its fee.

    Read for a lone leg: once an open trade's other market has paid out, only
    this leg is still held, as live's scanner.held_pairs counts a lone held
    leg at its own worth plus its own fees (the other leg's payout is already
    cash). The leg counts at the latest usable ask of the side it holds
    (LegQuotes.at_checkpoint), or at the price it paid when the trade has no
    quotes, plus the exact fee paid on it (config.fee_leg_exact at the price
    paid, the fee the trade was charged for that leg).

    Args:
        trade (BacktestTrade): An open trade with a leg on `ticker`.
        ticker (str): The leg's market.
        day (date): A checkpoint date on the leg's weekly grid.

    Returns:
        float: The leg's stake in dollars.
    """
    index = 0 if trade.ticker_a == ticker else 1
    # What the leg paid per contract, for the exact fee it was charged
    price = _paid_prices(trade)[index]
    return trade.n * _leg_mark(trade, index, day) + fee_leg_exact(trade.n, price)


# ─── Selling a position early ─────────────────────────────────────────────────

def _resolve_sell_at(sell_at: float | None) -> float | None:
    """
    Check the share of potential profit at which a simulation sells a position.

    Args:
        sell_at (float | None): A share in (0, 1] (0.25 sells once a position
            has made 25% of the profit it could make), or None to never sell.

    Returns:
        float | None: The share as a builtin float, or None.

    Raises:
        ValueError: If sell_at is a bool, not a real number, NaN or outside (0, 1].
    """
    if sell_at is None:
        return None
    return _validated_cap(sell_at, "sell_at")


def _resolve_sell_levels() -> tuple[float, ...]:
    """
    Check the sell levels the Sell family offers, and return them in ascending order.

    Reads this module's TAKE_PROFIT_LEVELS when called (patch backtester's,
    never config's). run_backtest_sweep calls it before its fetch when the
    sell family is on, so a bad level is refused in milliseconds rather than
    after the fetch, and _sweep_from_candidates calls it again to build the
    SellSweep's levels. Each level is checked as one simulation's sell level
    is (_resolve_sell_at's rule, _validated_cap). A level named twice is
    refused: the dashboard's Sell build finds each level's place among the
    options by its value, so a repeated level would leave one of its two
    places unfilled.

    Returns:
        tuple[float, ...]: The levels, ascending, as builtin floats.

    Raises:
        ValueError: If TAKE_PROFIT_LEVELS is not a collection, is empty, holds
            a value that is not a real number in (0, 1] (a bool or NaN
            included), or names one level twice.
    """
    try:
        levels = tuple(TAKE_PROFIT_LEVELS)
    except TypeError:
        raise ValueError("TAKE_PROFIT_LEVELS must be a tuple of shares in (0, 1], got "
                         f"{TAKE_PROFIT_LEVELS!r}") from None
    if not levels:
        raise ValueError("TAKE_PROFIT_LEVELS must name at least one sell level")
    checked = [_validated_cap(level, "each of TAKE_PROFIT_LEVELS") for level in levels]
    if len(set(checked)) != len(checked):
        raise ValueError(f"TAKE_PROFIT_LEVELS names a level twice: {levels!r}")
    return tuple(sorted(checked))


def _resolve_hold_days() -> int:
    """
    Check how many days in a row a position must stay at its sell level before it is sold.

    Reads this module's TAKE_PROFIT_HOLD_DAYS when called (patch
    backtester's, never config's). Read wherever the sell rule is applied
    (_simulate_at_discount with sell_at, _sale_reach) and by
    run_backtest_sweep before its fetch when the sell family is on.

    Returns:
        int: A whole number from 1 to _HOLD_DAYS_MAX.

    Raises:
        ValueError: For a bool, a number that is not whole, or one outside 1
            to _HOLD_DAYS_MAX.
    """
    days = TAKE_PROFIT_HOLD_DAYS
    if (isinstance(days, (bool, np.bool_)) or not isinstance(days, numbers.Integral)
            or not 1 <= days <= _HOLD_DAYS_MAX):
        raise ValueError(f"TAKE_PROFIT_HOLD_DAYS must be a whole number from 1 to "
                         f"{_HOLD_DAYS_MAX}, got {days!r}")
    return int(days)


def _resolve_sell_min_days(min_days: int | None, sell_at: float | None) -> int | None:
    """
    Check the fewest days before maturity at which a simulation may sell a position.

    The days rule (_simulate_at_discount's sell_min_days, CapSweep's): a
    position that has reached its sell level is sold only if at least this
    many days remain before its last market stops trading (_far_enough);
    nearer than that it is held until it pays out. It means nothing without
    a sell level, so it is refused without one.

    Args:
        min_days (int | None): A whole number of at least 1 (numpy whole
            numbers too, never a bool), or None for no minimum.
        sell_at (float | None): The simulation's resolved sell level, or None
            when it never sells.

    Returns:
        int | None: The minimum as a builtin int, or None.

    Raises:
        ValueError: If min_days is set while sell_at is None, or is a bool,
            not a whole number, or below 1.
    """
    if min_days is None:
        return None
    if sell_at is None:
        raise ValueError("sell_min_days needs sell_at: a run that never sells "
                         "has no sale to hold back")
    return _whole_days(min_days, "sell_min_days")


def _whole_days(days: Any, name: str) -> int:
    """
    Check one minimum of days before maturity: a whole number of at least 1.

    The one test of a days value, shared by _resolve_sell_min_days (one
    simulation's minimum) and _resolve_min_days_options (the Sell family's
    options), so the two can never accept different values.

    Args:
        days (Any): The value to check. A numpy whole number is accepted; a
            bool never is.
        name (str): What the value is, for the error message.

    Returns:
        int: The value as a builtin int.

    Raises:
        ValueError: If days is a bool, not a whole number, or below 1.
    """
    if (isinstance(days, (bool, np.bool_)) or not isinstance(days, numbers.Integral)
            or days < 1):
        raise ValueError(f"{name} must be a whole number of at least 1, got {days!r}")
    return int(days)


def _resolve_min_days_options() -> tuple[int, ...]:
    """
    Check the minimum-days options the Sell family offers, and return them in ascending order.

    Reads this module's TAKE_PROFIT_MIN_DAYS when called (patch
    backtester's, never config's). run_backtest_sweep calls it before its
    fetch when the sell family is on, so a bad option is refused in
    milliseconds rather than after the fetch, and _sweep_from_candidates
    calls it again to build the SellSweep's min_days. Each option is checked
    as one simulation's minimum is (_whole_days).

    Returns:
        tuple[int, ...]: The options, ascending, as builtin ints.

    Raises:
        ValueError: If TAKE_PROFIT_MIN_DAYS is not a collection, is empty,
            holds a value that is not a whole number of at least 1 (a bool
            included), or names one value twice.
    """
    try:
        options = tuple(TAKE_PROFIT_MIN_DAYS)
    except TypeError:
        raise ValueError("TAKE_PROFIT_MIN_DAYS must be a tuple of whole numbers of at "
                         f"least 1, got {TAKE_PROFIT_MIN_DAYS!r}") from None
    if not options:
        raise ValueError("TAKE_PROFIT_MIN_DAYS must name at least one minimum of days")
    checked = [_whole_days(days, "each of TAKE_PROFIT_MIN_DAYS") for days in options]
    if len(set(checked)) != len(checked):
        raise ValueError(f"TAKE_PROFIT_MIN_DAYS names a value twice: {options!r}")
    return tuple(sorted(checked))


def _days_left(position: list[BacktestTrade], day: date) -> int | None:
    """
    Days from `day` to the date the position's last market stops trading: its days to maturity.

    A position matures when the last of its markets closes, so this is the
    latest close date among its trades' markets (BacktestTrade.close_date_a
    and close_date_b, the dates the markets' close_time names — UTC on every
    Kalshi timestamp) less the checkpoint's date, in whole calendar days. It
    is 0 when the last market closes on the checkpoint's own date, and
    negative when it closed before it (a market that has stopped trading
    but not yet paid out). The days rule (_far_enough) reads it, and every
    sale records it (SaleCheck.days_left).

    Args:
        position (list[BacktestTrade]): One position's open trades (_positions).
        day (date): The checkpoint date.

    Returns:
        int | None: The days left; None when any trade lacks a close date
            (only a hand-built trade can: _simulate_at_discount gives every
            trade both) or the position is empty (never in the walk, whose
            positions each hold at least one trade).
    """
    # config.days_to_maturity: the one definition of days to maturity, kept
    # in config so live code can read it too
    return days_to_maturity([d for t in position for d in (t.close_date_a, t.close_date_b)],
                            day)


def _far_enough(position: list[BacktestTrade], day: date, min_days: int | None) -> bool:
    """
    Whether the days rule lets a position be sold at the checkpoint on `day`.

    With no minimum, always. With one, only when the position has at least
    that many days left before its last market stops trading (_days_left);
    a position whose days left are unknown is never sold under a minimum.
    The walk's sales (_simulate_at_discount) and its quick test at a
    checkpoint with no candidate (_position_sells) both ask here before they
    value the position, so they apply one rule.

    Args:
        position (list[BacktestTrade]): One position's open trades.
        day (date): The checkpoint date.
        min_days (int | None): The resolved minimum (_resolve_sell_min_days),
            or None for none.

    Returns:
        bool: True when the position may be sold there.
    """
    if min_days is None:
        return True
    left = _days_left(position, day)
    return left is not None and left >= min_days


def _hold_days_text(hold_days: int) -> str:
    """
    Say, for a log line or the dashboard, when a position is sold at a level.

    Args:
        hold_days (int): A resolved day count (_resolve_hold_days).

    Returns:
        str: "once it reaches a level at a checkpoint" for 1; otherwise
            e.g. "once it has stayed at or above a level for 3 days in a row".
    """
    if hold_days == 1:
        return "once it reaches a level at a checkpoint"
    return f"once it has stayed at or above a level for {hold_days} days in a row"


def _sale_label(sell_at: float, min_days: int | None = None) -> str:
    """
    Name a sell level, and any minimum of days before maturity, for a completion line or a log summary.

    Args:
        sell_at (float): A resolved share in (0, 1].
        min_days (int | None): The resolved minimum of days before maturity
            (_resolve_sell_min_days), or None for none.

    Returns:
        str: "selling at <percent>% of potential profit", the percent through
            _cap_percent, which never prints two levels alike; with a
            minimum, followed by ", at least <N> day(s) before maturity".
    """
    days = ("" if min_days is None
            else f", at least {min_days} day{'' if min_days == 1 else 's'} before maturity")
    return f"selling at {_cap_percent(sell_at)}% of potential profit{days}"


def _ladder_average(ladder: list[list[float]], contracts: float) -> float | None:
    """
    The average price of selling `contracts` down a bid ladder, best bid first.

    Args:
        ladder (list[list[float]]): [[price, contracts], ...], best first.
        contracts (float): How many contracts to sell; above 0.

    Returns:
        float | None: Their average price (exactly the best bid when that
            level holds them all); None when the ladder holds fewer contracts
            than that.
    """
    # scanner.walk_bids: the one walk down a bid ladder, kept in scanner so
    # live code can walk a real book with it too
    walked = walk_bids(ladder, contracts)
    return None if walked is None else walked[0]


def _positions(open_trades: list[BacktestTrade]) -> list[list[BacktestTrade]]:
    """
    Group open trades into positions: trades joined, directly or through others, by a market.

    Without add_to_held every open trade is a position of its own, since no
    two open trades share a market. With it, an add-on shares both markets
    with the pair it adds to, and an add-on to a lone leg shares that leg's
    market, so a position is a pair with everything added to it — what a
    sale sells whole.

    Args:
        open_trades (list[BacktestTrade]): The open trades, in the order they were made.

    Returns:
        list[list[BacktestTrade]]: The positions, in the order of their first
            trade, each holding its trades in open_trades' order.
    """
    parent = list(range(len(open_trades)))

    def root(i: int) -> int:
        """The lowest index of i's group (its root), shortening the path on the way."""
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # The first trade on each market; a later trade on it joins that one's group
    owner: dict[str, int] = {}
    for i, trade in enumerate(open_trades):
        for ticker in (trade.ticker_a, trade.ticker_b):
            first = owner.setdefault(ticker, i)
            a, b = root(i), root(first)
            if a != b:
                # A group's root is its lowest index, so groups list in order
                parent[max(a, b)] = min(a, b)
    groups: dict[int, list[BacktestTrade]] = {}
    for i, trade in enumerate(open_trades):
        groups.setdefault(root(i), []).append(trade)
    return list(groups.values())


def _position_sale_value(position: list[BacktestTrade], day: date,
                         days_back: int = 0) -> tuple | None:
    """
    What selling a whole position at the checkpoint on `day` would return, and what it cost and could pay.

    The one valuation a sale decides on (and _sale_reach replays).
    The position's contracts are totalled per market (by ticker and side
    held: an add-on to a lone leg can hold one market as A in one trade and
    as B in another), and each market gets one sale price:
      * a market that had paid out by the check
        (LegQuotes.paid_at_checkpoint) counts at its payout, with no sale and
        no fee;
      * otherwise the sale starts at the bid of the side held at the check
        (LegQuotes.bid_at_checkpoint). With a modeled bid ladder there
        (LegQuotes.sale_ladder) the price is the average over the position's
        contracts walked down it, and a ladder holding fewer contracts than
        the position means it cannot be sold at that check; with none (no
        model, no volume data, or a bid the model has no ladders for), the
        price is the bid itself, in any size.
    Each trade's leg returns its contracts at that price, less the taker fee
    on selling them (config.fee_leg_exact). The cost is each trade's
    contracts plus entry fees, and the potential total return each trade's
    contract pairs times CONTRACT_PAYOUT_DOLLARS — what one leg pays in a win.
    With days_back, everything is read at the sell rule's daily check that
    many days before the checkpoint (TAKE_PROFIT_HOLD_DAYS): what a sale then
    would have returned. _hold_readings is its only caller.

    Args:
        position (list[BacktestTrade]): One position's open trades (_positions).
        day (date): A checkpoint date on the legs' weekly grid.
        days_back (int): 0 (the default) for the checkpoint; 1 to
            _HOLD_DAYS_MAX - 1 for a daily check before it.

    Returns:
        tuple | None: (each trade's sale as (value, fees, (market A's sale
            price, market B's)), in order — a price None for a leg that paid
            out; the sale value; the total cost; the potential total return).
            None when a trade has no quotes, or a market still to pay out has
            no bid at this check (at the checkpoint, no fresh bid; at an
            earlier check, no quote in the 24 hours before it) or too few
            contracts on its ladder there.

    Raises:
        ValueError: From LegQuotes, for a day off the legs' checkpoint grid
            or a bad days_back.
    """
    # Every leg: its market and side, with the trade's quotes for that market
    legs = []
    for trade in position:
        if trade.marks is None:
            return None
        # Which side each leg holds (scanner.leg_sides, the one definition)
        legs.append(list(zip((trade.ticker_a, trade.ticker_b), trade.marks,
                             leg_sides(trade.pair_type), strict=True)))
    # The contracts held on each market, and its quotes
    held: dict[tuple[str, str], list] = {}
    for trade, trade_legs in zip(position, legs, strict=True):
        for ticker, quotes, side in trade_legs:
            held.setdefault((ticker, side), [quotes, 0])[1] += trade.n
    # One sale price per market; None when it has paid out
    prices: dict[tuple[str, str], float | None] = {}
    for (ticker, side), (quotes, contracts) in held.items():
        if quotes.paid_at_checkpoint(day, days_back):
            prices[(ticker, side)] = None
            continue
        bid = quotes.bid_at_checkpoint(day, side, days_back)
        if bid != bid:
            # No bid at this check (NaN): this market cannot be sold then
            return None
        ladder = quotes.sale_ladder(day, bid, days_back)
        price = bid if ladder is None else _ladder_average(ladder, contracts)
        if price is None:
            # The modeled bids hold fewer contracts than the position
            return None
        prices[(ticker, side)] = price
    per_trade = []
    value = cost = potential = 0.0
    for trade, trade_legs in zip(position, legs, strict=True):
        trade_value = fees = 0.0
        sale_prices: list[float | None] = []
        for ticker, quotes, side in trade_legs:
            price = prices[(ticker, side)]
            if price is None:
                # Paid out: worth its payout, with nothing to sell
                trade_value += trade.n * (quotes.paid_yes if side == "yes" else quotes.paid_no)
                sale_prices.append(None)
                continue
            # config.fee_leg_exact: the taker fee on selling n contracts at the price
            fee = fee_leg_exact(trade.n, price)
            trade_value += trade.n * price - fee
            fees += fee
            sale_prices.append(price)
        sale = (trade_value, fees, (sale_prices[0], sale_prices[1]))
        per_trade.append(sale)
        value += sale[0]
        cost += trade.total_cost + trade.fees
        potential += trade.n * CONTRACT_PAYOUT_DOLLARS
    return per_trade, value, cost, potential


def _sells_at(sell_at: float, realized: float, potential: float) -> bool:
    """
    Whether a position has realized at least sell_at of its potential profit.

    The sell rule: realized profit is the sale value less the total cost,
    potential profit the potential total return less the total cost, and a
    position with a potential profit sells once realized >= sell_at x
    potential (less PRICE_EPSILON of float noise). For a fixed position the
    test holds at every level below one at which it holds, since a float
    product with a positive number never falls as the other factor rises;
    _sale_reach relies on that. The sell rule applies it at every
    daily check (_reached_every_day).

    Args:
        sell_at (float): The share, in (0, 1].
        realized (float): The sale value less the total cost, in dollars.
        potential (float): The potential total return less the total cost.

    Returns:
        bool: True when the position sells.
    """
    # config.take_profit_reached: the one test, kept in config so live code
    # can decide a sale with it too
    return take_profit_reached(sell_at, realized, potential)


def _hold_readings(position: list[BacktestTrade], day: date, hold_days: int, *,
                   level: float | None = None) -> tuple | None:
    """
    A position's sale at the checkpoint on `day`, and its profit at each daily check the sell rule reads.

    The sell rule (TAKE_PROFIT_HOLD_DAYS) sells a position at a checkpoint
    only when it has reached its level there and at the daily check on each
    of the hold_days - 1 days before it — so it has stayed at the level for
    hold_days days in a row, checked once a day, 24 hours apart. Each check
    is valued as a sale then would be (_position_sale_value: at the
    checkpoint the fresh bids, at an earlier check the last quote of the 24
    hours before it; with a depth model, each check walks its own modeled
    bid ladder, built from that check's 24-hour volume). The walk's sales
    (_simulate_at_discount), its quick test at a checkpoint where nothing is
    bought (_position_sells) and the shortcut that replays the walk
    (_sale_reach) all read the checks here and decide through
    _reached_every_day, so they apply one rule.

    Args:
        position (list[BacktestTrade]): One position's open trades (_positions).
        day (date): A checkpoint date on the legs' weekly grid.
        hold_days (int): How many checks, the checkpoint's included (_resolve_hold_days).
        level (float | None): Keyword-only. When given, stop at the first
            check below this share of potential profit and return None: the
            walk asks about its one level, so a check after a failing one is
            never read. _sale_reach asks about every level and
            passes none.

    Returns:
        tuple | None: (the checkpoint's _position_sale_value, [(realized
            profit, potential profit) at each check, the checkpoint's
            first]); None when the position cannot be valued at some check
            (a trade with no quotes, or a leg still to pay out with no bid
            there or too few contracts on its ladder there) or, with level,
            a check falls below it.
    """
    sale = _position_sale_value(position, day)
    profits = []
    for days_back in range(hold_days):
        # days_back 0 is the checkpoint itself — the day of the sale, valued
        # just above as `sale` (it is also the sale's proceeds), so it is
        # reused rather than valued twice; 1, 2, ... are the days before it
        valued = sale if days_back == 0 else _position_sale_value(position, day, days_back)
        if valued is None:
            return None
        _per_trade, value, cost, potential = valued
        profits.append((value - cost, potential - cost))
        if level is not None and not _sells_at(level, value - cost, potential - cost):
            # Below the one level asked about: no later check is read
            return None
    return sale, profits


def _reached_every_day(sell_at: float, profits: list[tuple[float, float]]) -> bool:
    """
    Whether a position reached sell_at of its potential profit at every daily check.

    _sells_at at each check of _hold_readings. Like _sells_at it holds at
    every level below one at which it holds (it holds at each check), which
    _sale_reach relies on.

    Args:
        sell_at (float): The share, in (0, 1].
        profits (list[tuple[float, float]]): (realized profit, potential
            profit) at each check (_hold_readings).

    Returns:
        bool: True when every check reaches it.
    """
    return all(_sells_at(sell_at, realized, potential) for realized, potential in profits)


def _position_sells(sell_at: float, position: list[BacktestTrade], day: date,
                    hold_days: int, min_days: int | None = None) -> bool:
    """
    Whether a position would be sold at the checkpoint on `day`.

    The walk's quick test at a checkpoint with no candidate
    (_simulate_at_discount): it applies the walk's own rule, the days rule
    first (_far_enough), then the checks.

    Args:
        sell_at (float): The share of potential profit that sells it.
        position (list[BacktestTrade]): One position's open trades.
        day (date): A checkpoint date on its legs' weekly grid.
        hold_days (int): How many days in a row it must stay at that share
            (_resolve_hold_days).
        min_days (int | None): The fewest days it must have left before its
            last market stops trading (_resolve_sell_min_days); None (default)
            for no minimum.

    Returns:
        bool: False when it is too near maturity (_far_enough) or cannot be
            valued at some check (_hold_readings is None); otherwise whether
            it reached sell_at at every check (_reached_every_day).
    """
    if not _far_enough(position, day, min_days):
        # Too near maturity: held to pay out, never valued
        return False
    readings = _hold_readings(position, day, hold_days, level=sell_at)
    # Sold only when every check has a price (readings is not None) and each
    # reaches sell_at. With level=sell_at, _hold_readings already stops at
    # the first check below it, so the second test repeats the first; it is
    # kept so this, the walk's sales and _sale_reach all decide
    # through the one function
    return readings is not None and _reached_every_day(sell_at, readings[1])


def _sold_copy(trade: BacktestTrade, day: date, sale: tuple) -> BacktestTrade:
    """
    The record of a trade sold at the checkpoint on `day`, replacing the trade's own.

    The sale is the trade's exit: exit_date is the sale day, actual_payoff
    the sale's value after its fees, and the figures read off them (profit,
    profit_ratio, monthly_profit_ratio, holding_days, slippage) are
    recomputed as the cash walk computes them for a trade that pays out, so
    profit is still actual_payoff - total_cost - fees and the equity curve
    needs nothing new. outcome_*, settled_date_* and everything about the
    entry are kept.

    Args:
        trade (BacktestTrade): The open trade sold.
        day (date): The sale's checkpoint date.
        sale (tuple): Its sale, as _position_sale_value gives each trade's.

    Returns:
        BacktestTrade: A copy marked sold.
    """
    value, fees, (price_a, price_b) = sale
    profit = value - trade.total_cost - trade.fees
    invested = trade.total_cost + trade.fees
    profit_ratio = profit / invested if invested > 0 else 0.0
    holding_days = max(1, (day - trade.entry_date).days)
    return replace(
        trade, exit_date=day, actual_payoff=value, profit=profit,
        profit_ratio=profit_ratio, monthly_profit_ratio=profit_ratio * 30.0 / holding_days,
        slippage=profit - trade.expected_payoff, holding_days=holding_days,
        sold=True, sale_price_a=price_a, sale_price_b=price_b, sale_fees=fees,
    )


def _sale_checkpoints(first: date, last: date) -> list[date]:
    """
    Every entry-checkpoint date from `first` to `last`: the dates a position can be sold on.

    Args:
        first (date): The first date (inclusive).
        last (date): The last date (inclusive).

    Returns:
        list[date]: Each SCHEDULED_RUN weekday in range, in order.
    """
    day = first
    # Advance to the first run weekday, then step a week at a time
    while day.weekday() != SCHEDULED_RUN.weekday:
        day += timedelta(days=1)
    out = []
    while day <= last:
        out.append(day)
        day += timedelta(weeks=1)
    return out


def _sale_stream(candidates: list[dict]):
    """
    The candidates as (date, candidate), with (date, None) at every other checkpoint a position may be sold on.

    The checkpoints run from the first candidate's date to the last pay-out
    date of any candidate (_sale_checkpoints), so a position can be sold on a
    Monday nothing is bought and after the last purchase.

    Args:
        candidates (list[dict]): Pass 2's candidates, in date order.

    Yields:
        tuple[date, dict | None]: In date order; a candidate date is never
            also yielded with None.
    """
    if not candidates:
        return
    grid = _sale_checkpoints(candidates[0]["entry_date"],
                             max(c["exit_date"] for c in candidates))
    at = 0
    for c in candidates:
        day = c["entry_date"]
        while at < len(grid) and grid[at] < day:
            yield grid[at], None
            at += 1
        while at < len(grid) and grid[at] == day:
            at += 1
        yield day, c
    for day in grid[at:]:
        yield day, None


def _sale_reach(point: "SweepPoint", levels: Iterable[float]) -> dict[float, int]:
    """
    For each level, the most days before maturity at which some position of a no-selling run would sell there.

    It replays the sell test on a run that never sold: at every checkpoint
    from the first entry to the last pay-out (_sale_checkpoints) it groups
    the trades open there — entered before it and paying out after it, the
    trades a selling walk would hold — into positions (_positions) and
    reads each at the checkpoint and the daily checks before it, as a sale
    would (_hold_readings, TAKE_PROFIT_HOLD_DAYS). For each position it
    also reads its days left before maturity (_days_left; a position with
    an unknown one is skipped, since a minimum never sells it) and records
    them against every level _reached_every_day holds at — a prefix of the
    ascending levels, since reaching a share means reaching every lower
    one. The value kept per level is the largest such count.

    What it guarantees, for SellSweep.sold_grid: a run selling at level L
    with a minimum of N days is the no-selling run itself whenever L is not a
    key, or N is more than the value at L. A selling run walks exactly as
    the no-selling run does until its first sale, and that first sale would
    have to be one of the (checkpoint, position) pairs replayed here that
    reaches L at every check with at least N days left — there is none.
    (The converse holds too: with N at or below the value the run sells at
    least once. Had it sold nothing before the (checkpoint, position) that
    set the value, it would reach that pair just as the no-selling run did,
    and sell it there.)

    Args:
        point (SweepPoint): A simulation that never sold (sell_at None).
        levels (Iterable[float]): The shares to test, each in (0, 1].

    Returns:
        dict[float, int]: level -> the most days left of a position that
            reaches it; a level no such position reaches is not a key. Empty
            when there is no trade.

    Raises:
        ValueError: For a point that sold, or a bad TAKE_PROFIT_HOLD_DAYS.
    """
    if point.sell_at is not None:
        raise ValueError("_sale_reach reads a run that never sold, "
                         f"not one selling at {point.sell_at!r}")
    hold_days = _resolve_hold_days()
    ascending = sorted(levels)
    reach: dict[float, int] = {}
    if not point.trades or not ascending:
        return reach
    by_entry = sorted(range(len(point.trades)), key=lambda i: point.trades[i].entry_date)
    first = point.trades[by_entry[0]].entry_date
    last = max(t.exit_date for t in point.trades)
    added = 0
    open_ids: list[int] = []
    for day in _sale_checkpoints(first + timedelta(days=1), last - timedelta(days=1)):
        # Open here: entered before this checkpoint, paying out after it
        while added < len(by_entry) and point.trades[by_entry[added]].entry_date < day:
            open_ids.append(by_entry[added])
            added += 1
        open_ids = [i for i in open_ids if point.trades[i].exit_date > day]
        if not open_ids:
            continue
        # In the order the trades were made, as the walk holds them
        open_trades = [point.trades[i] for i in sorted(open_ids)]
        for position in _positions(open_trades):
            left = _days_left(position, day)
            if left is None:
                # Never sold under a minimum (_far_enough)
                continue
            readings = _hold_readings(position, day, hold_days)
            if readings is None:
                continue
            for level in ascending:
                if not _reached_every_day(level, readings[1]):
                    # Nor any higher level
                    break
                reach[level] = max(reach.get(level, left), left)
    return reach


def _sale_cover(sales: tuple[SaleCheck, ...], levels: tuple[float, ...]) -> tuple[int, int]:
    """
    How far a selling run's own sales say it reaches: the fewest days left among them, and the highest level all reached.

    SellSweep.sold_grid reads it to tell, without simulating, which stricter
    sell settings walk exactly as a run it simulated. A run selling at level
    levels[i0] with a minimum of N0 days is the same run as one at
    levels[i], with a minimum of N days, whenever i0 <= i <= the index
    returned and N0 <= N <= the days returned. Two facts make it so. The
    stricter setting sells nothing the simulated run did not: a position
    that reaches levels[i] at every check reaches levels[i0] too, and one
    with at least N days left has at least N0. And the bounds say every sale
    the simulated run made also reaches levels[i] at every check with at
    least N days left, so the stricter setting makes each of those sales
    too. Walking side by side from the same start, the two runs therefore
    decide alike at every checkpoint and position, and stay one walk. This
    needs nothing about how a position's days left change over time (an
    add-on to a lone leg can join a market that closes later).

    Args:
        sales (tuple[SaleCheck, ...]): Every sale of one selling run
            (SweepPoint.sales), at least one.
        levels (tuple[float, ...]): The sell levels, ascending.

    Returns:
        tuple[int, int]: (the fewest days_left among the sales, -1 when one
            is unknown; the index in levels of the highest level every
            sale's profits reach at every check (_reached_every_day), -1
            when not even the lowest is).

    Raises:
        ValueError: For a run that made no sale: it says nothing about any
            other setting (sold_grid never simulates one).
    """
    if not sales:
        raise ValueError("_sale_cover reads a run that sold at least once")
    days = [sale.days_left for sale in sales]
    fewest = -1 if any(d is None for d in days) else min(days)
    top = -1
    for i, level in enumerate(levels):
        if not all(_reached_every_day(level, list(sale.profits)) for sale in sales):
            # Nor any higher level
            break
        top = i
    return fewest, top


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
    Prefilter: could this market ever be one leg of an entered pair, of either type?

    Run on each market before grouping, by the fetch and again in both of
    _prepare_candidates' passes. It must keep exactly the markets some
    checkpoint of _find_entry's scan could reach, and changing it (or
    CANDLESTICK_PERIOD_INTERVAL_MINUTES) requires a new
    config.SETTLED_PREFILTER_CACHE_TAG.

    _find_entry() enters only at a checkpoint dated from start_date to the
    day before the earlier close, where both markets have a candle at or
    before it. Kalshi is observed never to serve a candle ending at or before
    the start of the candle period a market opened in, so a market is kept
    exactly when some checkpoint from start_date to its close date minus a
    day has its FLOOR, the start of its candle period (_checkpoint_floor),
    after open_time. The one-year lookback and max_horizon_days are left to
    _find_entry (one cached list serves every horizon).

    Exact only while each checkpoint falls on its own UTC date, which
    _prepare_candidates checks first. An open_time with no UTC offset (only a
    hand-edited cache) or outside datetime's range, or a checkpoint floor
    outside that range, gets the looser DATE TEST (a checkpoint on the
    opening date counts at any hour). It never raises: it runs in the fetch's
    workers, where an exception would end the run.

    Args:
        m (dict): Market dict as produced by historical._market_to_dict().
        start_date (date): Backtest start date; no earlier checkpoint is scanned.

    Returns:
        bool: False only when its times prove no checkpoint can enter it; True
            when unsure too, e.g. a missing or unreadable open_time/close_time
            (older caches). Never raises.
    """
    open_dt = _parse_iso_datetime(m.get("open_time"))
    close_d = _parse_iso_date(m.get("close_time"))
    if open_dt is None or close_d is None:
        # Can't prove ineligibility — keep it rather than risk dropping a
        # market that could actually enter a pair.
        return True

    # Naive (no offset): no moment to compare, so the docstring's date test.
    open_utc = None
    open_d = open_dt.date()
    if open_dt.utcoffset() is not None:
        try:
            open_utc = open_dt.astimezone(UTC)
        except OverflowError:
            # Outside datetime's range in UTC (only a hand-edited cache): the
            # date test too, never an exception out of an assembly worker.
            pass
        else:
            open_d = open_utc.date()

    # Every step below is a DIFFERENCE of dates, never a date plus a
    # timedelta, so no date near either end of datetime's range can raise
    # OverflowError.
    # The scan starts at the later of the opening date and start_date ...
    first = max(open_d, start_date)
    # ... may reach any checkpoint dated up to close_time's date minus one
    # day, exactly as _find_entry builds scan_end ...
    room = (close_d - first).days - 1
    # ... and its first checkpoint is on the first run weekday on or after
    # `first`, the same date _monday_timestamps's loop reaches.
    ahead = (SCHEDULED_RUN.weekday - first.weekday()) % 7
    # A checkpoint dated after the opening date has its floor after the
    # opening (checkpoints fall on their own UTC date). Only one on the
    # opening date can have its floor at or before it; then the first
    # reachable checkpoint is next week's. A floor outside datetime's range
    # counts as reachable (looser, never tighter).
    if open_utc is not None and ahead == 0 and first == open_d:
        floor = _checkpoint_floor(SCHEDULED_RUN, first)
        if floor is not None and open_utc >= floor:
            ahead = 7
    return ahead <= room


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


def _comparable_closes_dicts(mA: dict, mB: dict) -> tuple[datetime, datetime] | None:
    """
    Both records' close_time parsed, or None when they cannot be compared.

    The dict-world READER of the same-title close gate (DR-74): the verdict
    itself is scanner.closes_apart, called by _extract_pairs on what this
    returns, so the live finder and this one can never disagree about which
    cross-series pair closes at one moment. Readability is split out here,
    rather than left to closes_apart's own fail-closed branch, because the two
    paths do NOT see the same population: the live finder never reaches a
    market without a close_time (_filter_active_markets drops it before
    grouping, with its own WARNING), while the backtest's same-title groups
    are not close-filtered — _can_ever_enter keeps a market whose close_time
    it cannot parse. So an unreadable close is counted on its own
    backtest-only line, and the close-gap line stays verbatim with the live
    one for every READABLE pair. One shape splits the two paths' lines: a
    naive close beside an aware one lands here, on the unreadable line,
    while the live finder, whose gate has one line only, counts it on its
    close-gap line. Both refuse it, and the live path cannot reach it today
    (Kalshi's close strings end in "Z"; every cached close does too). The
    backtest reads the REALIZED close Kalshi recorded for a settled market,
    where the live finder reads the SCHEDULED close of an open one.

    Args:
        mA (dict): A market dict in the compact historical._market_to_dict form.
        mB (dict): A second market dict, same form.

    Returns:
        tuple[datetime, datetime] | None: (close_a, close_b) parsed by
            _parse_iso_datetime. None when either close_time is absent or
            unparseable, or when one parses naive and the other aware (their
            subtraction would raise). Never raises.
    """
    ca = _parse_iso_datetime(mA.get("close_time"))
    cb = _parse_iso_datetime(mB.get("close_time"))
    if ca is None or cb is None or (ca.utcoffset() is None) != (cb.utcoffset() is None):
        return None
    return ca, cb


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
    one on a subtitle-blank corpus, and the k-hat of the rule the switch
    shipped behind (not met — see config.TIME_SERIES_SAME_EVENT_LADDERS) was
    measured on exactly such a corpus (see CLAUDE.md's DR-67 residual
    list and the plan's own "prove the corpus before spending on it" step).

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


def _ladder_keys_dict(m: dict, question_key: str | None = None) -> frozenset:
    """
    Return the ladder labels of one cached market, through scanner.ladder_keys.

    A cached market often has no event title, so its question label can
    match more markets than it would live.

    Args:
        m (dict): One cached market record.
        question_key (str | None): The market's question with its dates
            removed, if already known. None works it out (slower).

    Returns:
        frozenset: Up to two labels, one for the event and one for the question.
    """
    return ladder_keys(m.get("event_ticker"),
                       _ts_group_key(m) if question_key is None else question_key)


def _group_by_exact_title(markets: Iterable[dict]) -> dict[tuple, list[dict]]:
    """
    Group markets by exact (event_title, title, subtitle) tuple for same-title pair detection.

    Three-element key: the event_title component prevents cross-event option-label
    collisions in MVE markets; (title, subtitle) distinguishes markets within an event.
    The key is _st_group_key's, shared with _index_eligible_keys so the
    groupable-subset decision in _prepare_candidates is taken on this exact key.

    Grouping is deliberately unchanged by the one-series rule (DR-02, DR-54)
    and by the close gate (DR-74): two events of one recurring fixture, and
    two games on two series closing hours apart, still land in one group, and
    _extract_pairs is what refuses to pair them. Keeping each rule in one
    place mirrors the live scanner, where find_same_title_pairs groups first
    and filters inside its inner loop.

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

    The count it logs is of candidates: a time-series pair has one per
    passing Monday.

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
    two events of one series worded identically, AND, for a same-title group,
    close within SAME_TITLE_MAX_CLOSE_GAP_SECONDS of each other (DR-74). No
    price filtering at this stage.

    The one-series rule (DR-02, DR-54) mirrors both live finders through
    _same_series_dicts / _identical_wording_dicts: two events resolving to one
    series are two instances of one recurring fixture, so identical wording
    across them is one question about two DIFFERENT events. Two combo tickets
    resolve to one series while sharing no literal prefix (DR-55). The 3-tuple
    (same-title) branch tests the series alone, because its group key already
    guarantees the wording is identical; the string (time-series) branch tests
    the conjunct, because there the wording is only date-stripped-equal and a
    genuine cumulative pair (deadline IN the wording) must survive.

    The same-title close gate (DR-74) mirrors scanner.find_same_title_pairs
    through the ONE definition, scanner.closes_apart, on the close times
    _comparable_closes_dicts parses: identical wording on two DIFFERENT series
    is one question only when both markets close at the same moment — a men's
    and a women's college basketball game between the same two schools
    (KXNCAAMBGAME / KXNCAAWBGAME) share every word of the key and close hours
    apart, and two competitions' fixtures of one matchup days apart. It runs
    after the series test, on the 3-tuple branch only: identical wording can
    never form a time-series pair (DR-67), so the time-series branch needs no
    twin. The backtest reads REALIZED closes where the live finder reads
    SCHEDULED ones (see CLAUDE.md's DR-74 gotcha for what that difference
    admits and refuses).

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
    pair (they are counted, never enumerated — see below). The +1 day margin
    is slack only (it can never cause a pair within the true limit to be
    skipped) — _find_entry() still applies the exact
    `.days > MAX_DEADLINE_GAP_DAYS` cutoff itself. Members with a missing or
    unparseable close_time are dropped from this sweep, and counted
    (group-local only — _group_by_exact_title's same-title groups are
    untouched), because _find_entry() unconditionally requires close_time on
    both legs and returns None immediately without it, regardless of pair
    type.

    3-tuple-keyed (same-title) groups have no deadline-gap concept, so they
    are swept naively — every pair of members is visited — and the
    eligibility prefilter (_can_ever_enter, applied in run_backtest before
    grouping) keeps these groups small in practice. Their pairs ARE gated on
    close time (DR-74 above), but per visited candidate, not by a sorted
    window: the gate is a one-hour bound that refuses most candidates, not a
    performance bound, and these groups are not close-filtered at all — a
    member whose close_time cannot be parsed is kept in the group and
    refused per candidate on its own line.

    EVERY PAIR OF A GROUP'S MEMBERS IS ACCOUNTED FOR (M10). Each count below
    is reported once, at the end of the call, on its own silent-at-zero INFO
    line, and together with the pairs returned they cover every pair of
    members of every group this call receives:
      - a same-title group is swept naively, so each of its pairs is visited
        and is either returned or refused as both markets on one event
        ticker, as two events of one series (DR-02, DR-54), as two markets
        whose close_time cannot be read or compared, or as two markets
        closing more than SAME_TITLE_MAX_CLOSE_GAP_SECONDS apart (DR-74);
      - a time-series group first sets aside its members with no readable
        close_time (counted as MEMBERS, not pairs — no pair involving one is
        ever formed), then splits the pairs of the rest at the sweep's
        close-date window: the pairs beyond it are counted as never visited
        (no rule is evaluated on them — the window is a performance bound:
        _find_entry rejects a cross-event pair that far apart anyway, and a
        same-event one would be refused with the ladder switch off and is
        judged by the ladder sub-pass with it on), and each pair inside it
        is either returned or refused as both markets on one event ticker
        (counted only while the ladder switch is off; with it on the pair
        belongs to the ladder sub-pass, which counts it there), as two
        events of one series worded identically, or for one of the three
        DR-72 wording reasons. With the switch on, the sub-pass's own lines
        count every same-event pair of the group, INCLUDING those the
        sweep's window excludes, so its population overlaps the
        never-visited count by exactly the same-event pairs beyond the
        window.
    The one exception is the `seen` guard, an uncounted `continue` that can
    fire only on a ticker listed twice in one group — a corpus the assembly's
    first-wins ticker dedup never produces — and that cannot cause a zero
    even then, since `seen` holds only pairs already returned for the group
    and so refuses nothing but a repeat of one. And a grouping with NO group
    of two or more members reaches this function as an empty dict, which
    says nothing about which kind of grouping it was, so _prepare_candidates
    logs both groupings' sizes on every run before calling it; between that
    line and these, every zero "Potential pairs" count has a logged cause.

    The one-series and same-event checks used to be bare `continue`s, so a
    corpus whose candidates they refused — the shape a combo-heavy window
    takes, where identically worded KXMVE tickets share a group key —
    reported "Potential pairs: 0" with no logged cause at all (DR-66).
    M10's counting changed no control flow: it added counts only, so every
    check, its order and every pair returned stayed exactly as before it
    (DR-74's close gate, which came later, is a new check and does refuse
    pairs). The same-title one-series line is the verbatim mirror of
    scanner.find_same_title_pairs', and so is the same-title close-gap line
    (DR-74) for READABLE closes; the same-title close-time readability line
    has no live twin, because the live finder drops a market without a
    close_time before grouping and so never reaches such a pair (and
    Kalshi's live close strings are all aware) — a naive close beside an
    aware one lands on this line here, but on the live finder's close-gap
    line, the only line the live gate has; the same-title
    same-event and time-series one-series lines mirror the lines M10 added to
    the two live finders; the time-series same-event line has no live twin
    that counts the same thing — the live finder's disabled-ladder count is
    unwindowed and over actively priced markets — and neither do the
    never-visited and no-close_time lines, since the live finder sweeps
    every pair of a group (its gap cap is a counted rule, applied after the
    wording check) and drops a market without a close_time before grouping,
    with its own WARNING.

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
    switch on — every default run since the 2026-09-26 decision. A per-call aggregate guard
    is the alternative and is deliberately not built; if the cost ever stops
    being negligible, sum len(idxs)*(len(idxs)-1)//2 across buckets here and
    warn past a second threshold.

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
            deadlines, or whose closes, for a 3-tuple-keyed group, are readable
            and within SAME_TITLE_MAX_CLOSE_GAP_SECONDS of each other, and no
            group holds an admissible ladder.
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
    # The one-series and same-event refusals (M10), counted where they fire,
    # per CANDIDATE pair, and reported beside the three above (silent at
    # zero). They were bare `continue`s, so the 2026-09-24 7-day run logged
    # "Potential pairs: 0 time-series, 0 same-title" over 184,178 groupable
    # markets with 748 time-series wording refusals and nothing that named a
    # cause for the rest of that zero. A counter is only ever incremented
    # immediately before the `continue` it explains, so no check moves and
    # no pair changes.
    #
    # ts_same_event_skips counts ONLY while the ladder switch is off: with it
    # on, a same-event candidate is not refused here but handed to the ladder
    # sub-pass below, which counts every one of them (unwindowed) on its own
    # lines, so counting it here as well would report it twice.
    ts_same_event_skips = 0
    ts_series_skips = 0
    st_same_event_skips = 0
    st_series_skips = 0
    # The same-title close gate (DR-74), counted after the series test so the
    # two counts above do not move and each candidate lands on one line.
    # st_undated_skips is backtest-only: a same-title group is not
    # close-filtered (_can_ever_enter keeps a market whose close_time it cannot
    # parse), while the live finder drops such a market before grouping — so
    # readability is counted apart, and st_close_gap_skips stays the verbatim
    # twin of the live finder's close-gap count.
    st_undated_skips = 0
    st_close_gap_skips = 0
    # The two things the time-series sweep drops BEFORE any candidate is
    # visited (M10), so that a zero caused by them has a cause in the log
    # too. ts_undated_members counts MEMBERS (a member without a readable
    # close_time forms no pair at all); ts_beyond_window counts the pairs of
    # dated members the close-date window excludes. Both are tallied once per
    # outer index or per group — never per candidate — so the sweep's
    # O(n * window) cost is unchanged: at the window's `break`, every later
    # index is also beyond it (the members are sorted by close date).
    ts_undated_members = 0
    ts_beyond_window = 0
    # DR-73's same-event ladder sub-pass keeps its OWN counters, for the same
    # reason the live branch does: they count candidates INSIDE one event, a
    # population the three above have never seen, and folding them in blurs
    # exactly the distinction DR-72 split apart. Each is silent at zero.
    #
    # Two of the live branch's counters have no mirror here, deliberately.
    # ladder_disabled_skips is live-only: counting the disabled population
    # here would mean enumerating every same-event pair just to feed a
    # counter, which is 615,266 pairs on one real strike-blind day slice — the
    # switch-off path must do no pairwise work at all. (ts_same_event_skips
    # above is not that census: it counts only the same-event candidates the
    # windowed sweep already visits, so it costs no pairwise work of its own,
    # and it is a windowed subset of that census, not the census itself.)
    # ladder_price_sum_skips has nothing to count: this function applies no
    # price filter of any kind.
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
            # Members the sweep cannot place (M10): counted, never paired.
            ts_undated_members += len(members) - len(dated)
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
                # The first index the window excludes for this mA; n when the
                # window reaches the end of the group (M10's never-visited
                # count — set only at the `break`, so no candidate pays for it).
                stop = n
                for j in range(i + 1, n):
                    close_b, mB = dated[j]
                    if close_b - close_a > margin:
                        # Sorted ascending by close_time — every further j is
                        # at least this far from mA, so nothing later qualifies.
                        stop = j
                        break
                    if mA["event_ticker"] == mB["event_ticker"]:
                        # Two markets of ONE event: refused here with the
                        # ladder switch off; with it on, the ladder sub-pass
                        # below judges (and counts) the candidate instead.
                        if not ladders_on:
                            ts_same_event_skips += 1
                        continue
                    # Mirror of the scanner's time-series conjunct: identical
                    # wording across two events of one series is two instances
                    # of one recurring fixture, not one question at two
                    # deadlines (DR-02, DR-54).
                    if _identical_wording_dicts(mA, mB) and _same_series_dicts(mA, mB):
                        ts_series_skips += 1
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
                # Indexes stop..n-1 were never visited for this mA.
                ts_beyond_window += n - stop

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
            # Same-title: no deadline-gap concept, so swept naively — but each
            # candidate is gated on close-time proximity below (DR-74).
            for i, mA in enumerate(members):
                for mB in members[i + 1:]:
                    if mA["event_ticker"] == mB["event_ticker"]:
                        # Mirror of scanner.find_same_title_pairs' same-event
                        # skip, counted as it now is there (M10).
                        st_same_event_skips += 1
                        continue
                    # Mirror of scanner.find_same_title_pairs' one-series rule:
                    # the group key already guarantees identical wording, so
                    # two events of one series are two fixtures (DR-02, DR-54).
                    if _same_series_dicts(mA, mB):
                        st_series_skips += 1
                        continue
                    # Mirror of find_same_title_pairs' close gate (DR-74): same
                    # wording on two series is one question only when both close
                    # at the same moment. An unreadable close is refused on its
                    # own backtest-only line (no live twin: the live finder drops
                    # a missing close before grouping).
                    closes = _comparable_closes_dicts(mA, mB)
                    if closes is None:
                        st_undated_skips += 1
                        continue
                    # The ONE definition of the gate, shared with the live finder
                    # so the two paths cannot disagree on "one moment".
                    if closes_apart(*closes):
                        st_close_gap_skips += 1
                        continue
                    pair_key = frozenset([mA["ticker"], mB["ticker"]])
                    if pair_key in seen:
                        continue
                    seen.add(pair_key)
                    pairs.append((mA, mB, canon, key))
    # What the time-series sweep set aside before visiting anything (M10),
    # each silent at zero: without these, a zero caused by an unreadable
    # close_time or by a group whose members all close too far apart logged
    # nothing at all. Neither has a live twin (see the docstring). Worded as
    # never VISITED, not refused, and without "gap cap": no rule was evaluated
    # on these pairs, and the live finder's gap-cap line counts something
    # else — pairs already worded as two cumulative deadlines.
    if ts_undated_members:
        logging.info(
            "Time-series group members without a readable close_time, left "
            "out of pair extraction (_find_entry cannot enter a pair without "
            "one; counted as markets, not pairs): %d",
            ts_undated_members,
        )
    if ts_beyond_window:
        logging.info(
            "Time-series candidate pairs the sweep never visits because their "
            "close dates are more than %d days apart (a performance bound, no "
            "rule evaluated — _find_entry rejects a cross-event pair past %d "
            "days; with same-event ladders on, the ladder sub-pass still judges "
            "the same-event ones): %d",
            MAX_DEADLINE_GAP_DAYS + 1, MAX_DEADLINE_GAP_DAYS, ts_beyond_window,
        )
    # The one-series and same-event refusals (M10), each silent at zero.
    # _prepare_candidates hands each call ONE grouping (the time-series or
    # the same-title one), so a production call logs at most one pair of
    # these. The time-series lines say "within the deadline-gap window,
    # before price filters" like the DR-72 lines below, because the windowed
    # sweep never visits a candidate beyond that window; the same-title lines
    # are unwindowed, like the branch that counts them.
    if ts_same_event_skips:
        # Backtest-only: the live finder's nearest line ("Same-event
        # candidates skipped because same-event deadline ladders are
        # disabled ...") counts EVERY same-event candidate of a group, while
        # this counts only those inside the sweep's window, over eligible
        # settled markets — two different numbers, so two different wordings.
        # Logged only with the switch off; with it on, the ladder sub-pass's
        # own lines below account for the same candidates.
        logging.info(
            "Time-series candidates skipped because both markets carry the "
            "same event ticker and same-event deadline ladders are off for "
            "this run (within the deadline-gap window, before price "
            "filters): %d",
            ts_same_event_skips,
        )
    if ts_series_skips:
        # Mirror of the line find_time_series_pairs logs for its one-series
        # conjunct (M10); the parenthesis differs as the DR-72 lines' does.
        logging.info(
            "Time-series candidates skipped as two instances of one event "
            "series (identical wording, different fixture; within the "
            "deadline-gap window, before price filters): %d",
            ts_series_skips,
        )
    if st_same_event_skips:
        # Verbatim mirror of find_same_title_pairs' same-event line (M10).
        logging.info(
            "Same-title candidates skipped because both markets carry the "
            "same event ticker (one event's own markets, not one question "
            "listed by two events): %d",
            st_same_event_skips,
        )
    if st_series_skips:
        # Verbatim mirror of find_same_title_pairs' long-standing one-series
        # line: both count every refused candidate before any price filter
        # (this function applies none), so the two numbers mean the same.
        logging.info(
            "Same-title candidates skipped as two instances of one event series "
            "(identical wording, different fixture): %d", st_series_skips,
        )
    if st_close_gap_skips:
        # Verbatim mirror of find_same_title_pairs' close-gap line (DR-74).
        # The bound is printed through scanner.close_gap_bound_text, which
        # reads the binding closes_apart read — this module holds no copy of
        # the constant, so the line cannot state a bound its gate did not use.
        logging.info(
            "Same-title candidates refused because the two markets close more "
            "than %s apart (two different games or instants, not one "
            "question listed twice): %d",
            close_gap_bound_text(), st_close_gap_skips,
        )
    if st_undated_skips:
        # Backtest-only (DR-74): the live finder never reaches a market
        # without a close_time, so this has no live twin.
        logging.info(
            "Same-title candidate pairs refused because a market's close_time "
            "cannot be read (fail closed; the live finder drops such markets "
            "before grouping): %d",
            st_undated_skips,
        )
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
    # switch off adds no LADDER line to this function's output (the sweep's
    # own same-event line above is not a ladder line). They say
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
        # counts candidates, because which rung of a ladder trades is decided
        # later here, in _simulate_at_discount.
        logging.info(
            "Same-event ladder candidates among the time-series candidates: %d",
            ladder_pairs,
        )
    return pairs


# ─── Entry point detection from candlestick data ──────────────────────────────

def _checkpoint_datetime(d: date) -> datetime:
    """
    The entry checkpoint on date `d`: SCHEDULED_RUN.instant(d), the UTC moment the live bot runs.

    _find_entry's checkpoints and the prefilter both come from SCHEDULED_RUN,
    so they cannot disagree. Tests patch backtester.SCHEDULED_RUN, never
    config.*.

    Args:
        d (date): The checkpoint's date (a run weekday, for every caller).

    Returns:
        datetime: The run's moment on d, tz-aware in UTC.

    Raises:
        zoneinfo.ZoneInfoNotFoundError, ValueError, OSError: Unresolvable zone.
        OverflowError: The moment is outside datetime's range.
    """
    # The same definition the scheduler's startup check uses
    return SCHEDULED_RUN.instant(d)


@functools.lru_cache(maxsize=4096)
def _checkpoint_floor(run: ScheduledRun, d: date) -> datetime | None:
    """
    The checkpoint's FLOOR: the start of the candle period holding the run's moment on date `d`.

    Candle periods are CANDLESTICK_PERIOD_INTERVAL_MINUTES long, counted from
    the Unix epoch. _can_ever_enter keeps a market only if it opened before
    some checkpoint's floor; a candle-period change needs a new
    config.SETTLED_PREFILTER_CACHE_TAG. Cached per (schedule, date), so a
    patched SCHEDULED_RUN never reads another schedule's answer.

    Args:
        run (ScheduledRun): The schedule whose moment is rounded down.
        d (date): The checkpoint's date.

    Returns:
        datetime | None: The floor, tz-aware in UTC; None when the moment is
            outside datetime's range (the prefilter then counts it reachable).

    Raises:
        zoneinfo.ZoneInfoNotFoundError, ValueError, OSError: Unresolvable zone.
    """
    period_seconds = CANDLESTICK_PERIOD_INTERVAL_MINUTES * 60
    try:
        # The run's UTC moment on d (config.ScheduledRun.instant), rounded
        # down to a candle-period boundary as _candle_window_open does
        instant = run.instant(d)
        return instant - timedelta(seconds=int(instant.timestamp()) % period_seconds)
    except OverflowError:
        return None


def _prefilter_cache_tag() -> str:
    """
    Build the cache tag naming the prefilter a backtest market list is assembled under.

    config.SETTLED_PREFILTER_CACHE_TAG plus SCHEDULED_RUN.cache_slug(), read
    when called, so a changed or patched schedule renames the cache. The
    fetch puts it in the cached list's file name.

    Returns:
        str: e.g. "checkpoint-v3-mon0900-America-Los_Angeles".
    """
    # The schedule's file-name-safe slug (config.ScheduledRun.cache_slug)
    return f"{SETTLED_PREFILTER_CACHE_TAG}-{SCHEDULED_RUN.cache_slug()}"


def _monday_timestamps(start_date: date, end_date: date) -> list[int]:
    """
    Generate the Unix timestamp of every entry checkpoint in the given date range.

    One per SCHEDULED_RUN weekday (despite the name, Monday only on the
    shipped schedule), each _checkpoint_datetime of that date. _find_entry
    scans them; _prepare_candidates checks the window holds any.

    Args:
        start_date (date): First date of the scan range (inclusive). The function
            advances to the first run weekday on or after this date.
        end_date (date): Last date of the scan range (inclusive).

    Returns:
        list[int]: One timestamp per run weekday in range; [] if none.

    Raises:
        zoneinfo.ZoneInfoNotFoundError, ValueError, OSError: Unresolvable zone.
        OverflowError: A checkpoint or weekly step outside datetime's range.
    """
    d = start_date
    # Advance to the first run weekday if start_date is not already one
    while d.weekday() != SCHEDULED_RUN.weekday:
        d += timedelta(days=1)
    ts_list = []
    while d <= end_date:
        ts_list.append(int(_checkpoint_datetime(d).timestamp()))
        d += timedelta(weeks=1)
    return ts_list


def _checkpoint_utc_times(start_date: date, end_date: date) -> list[str]:
    """
    List the distinct UTC times of day the entry checkpoints in a date range fall at.

    For run_backtest_sweep's "Entry checkpoint (backtest)" log line.

    Args:
        start_date (date): First date (inclusive).
        end_date (date): Last date (inclusive).

    Returns:
        list[str]: Ascending "HH:MM" UTC times, e.g. ["16:00", "17:00"] for
            09:00 Los Angeles time across a clock change; [] when none.

    Raises:
        ZoneInfoNotFoundError, ValueError, OSError, OverflowError: As _monday_timestamps.
    """
    return sorted({datetime.fromtimestamp(ts, UTC).strftime("%H:%M")
                   for ts in _monday_timestamps(start_date, end_date)})


def _candle_at_or_before(candles: list[dict], ts: int) -> dict | None:
    """
    Find the most recent candlestick at or before a given Unix timestamp.

    Candles are assumed to be sorted ascending by their "ts" field. Returns the
    last candle whose "ts" is <= ts, or None if no such candle exists. ("ts" is
    derived from the API's end_period_ts when the candle is fetched — see
    historical.fetch_candlesticks — so this is equivalently "at or before the
    candle's period end", just read off the dict's own key rather than the
    wire field name.) Nothing in the pipeline calls it: _find_entry looks up
    all of a pair's Mondays at once through _candles_at_or_before, which
    gives the same answers in one pass. It is kept as the plain version the
    tests check that function against (TestCandlesAtOrBefore).

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


def _candles_at_or_before(candles: list[dict], timestamps: list[int]) -> list[dict | None]:
    """
    For each of several moments, find a market's latest candle at or before it.

    A faster way to do what _candle_at_or_before does, for many moments at
    once; _find_entry and _leg_quotes call it. _find_entry checks a pair at the
    entry checkpoint of every Monday of the pair's window and records each
    Monday on which the pair passes its entry checks. To check a Monday it
    needs each market's prices as of that checkpoint: the market's latest
    candle at or before that moment. A candle is one hour of
    a market's price history: the hour's end time ("ts") and the YES and NO
    ask prices at the end of that hour (the NO ask estimated as 1 − the YES
    bid).

    Calling _candle_at_or_before once per Monday would re-read the candle
    list from the start every time. The Mondays come in date order, so here
    each lookup carries on from where the previous one stopped, and one walk
    through the list answers them all. The answers are exactly
    _candle_at_or_before's, even for candles out of order, and a moment
    earlier than the previous one restarts from the first candle
    (TestCandlesAtOrBefore checks both).

    Args:
        candles (list[dict]): One market's candles, each a dict with a "ts"
            key (Unix seconds), normally in time order.
        timestamps (list[int]): The moments to look up, in Unix seconds;
            _find_entry passes its entry checkpoints (one per run weekday,
            Monday on the shipped schedule), in date order.

    Returns:
        list[dict | None]: For each moment, in order, _candle_at_or_before's
            answer — for time-ordered candles, the latest candle at or before
            that moment (the dict itself, not a copy) — or None when there is
            none.
    """
    found: list[dict | None] = []
    i = 0
    n = len(candles)
    previous: int | None = None
    for ts in timestamps:
        if previous is not None and ts < previous:
            # Earlier than the previous moment: start again from the first candle
            i = 0
        # Step forward to the first candle after this moment; the one just before it is the answer
        while i < n and candles[i]["ts"] <= ts:
            i += 1
        found.append(candles[i - 1] if i else None)
        previous = ts
    return found


def _usable_ask(raw: Any, side: str) -> float:
    """
    Read one candle close as an ask that can value a leg, or NaN.

    An ask is usable when it reads as a number strictly between 0 and 1 —
    the rule live applies to a held pair's ask (scanner._held_leg_worth) —
    so a sub-cent ask on a fine grid counts, while a YES ask of 1.00 (no one
    offering YES) does not. A NO ask must also be below
    _CANDLE_NO_ASK_CEILING: the candle stores the NO ask as 1 - the YES bid,
    clamped to 0.99 at most, so 0.99 is also what a YES-bid book with no bids
    reads as. Each bound is held PRICE_EPSILON inside, so float noise on a
    price sitting on a bound reads as that bound.

    Args:
        raw: A candle's "yes_ask_close" or "no_ask_close" (normally a float).
        side (str): "yes" for a YES ask, "no" for a NO ask.

    Returns:
        float: The ask when usable; NaN otherwise, which _leg_quotes skips,
            so the side keeps its last usable ask.
    """
    # historical.usable_candle_ask: the one definition, kept in historical
    # so live code can read candles with it too
    return usable_candle_ask(raw, side)


def _leg_quotes(market: dict, candles: list[dict], start_date: date,
                depth_model: DepthModel | None = None) -> tuple[LegQuotes | None, int]:
    """
    Sample one market's candles into the prices that value an open leg in it (a LegQuotes).

    Day-end samples run from first_day — the later of start_date and the day
    the first candle can be read at a day's end — through the market's
    settlement date (or its last candle's date when it has no readable
    settlement time), each side's latest usable ask (_usable_ask) on a candle
    ending at or before the next UTC midnight. Checkpoint samples run over
    the same days, one at each entry checkpoint (_monday_timestamps), each
    side's latest usable ask on a candle ending at or before it. Both use
    _candles_at_or_before, the one candle lookup _find_entry uses, over the
    candles whose ask on that side is usable — so a side keeps its last
    usable ask through an empty book or an unreadable candle, and a side
    with no usable ask yet is NaN. A sample at or after the settlement time
    is the market's payout when the result is known (from its "result":
    "yes" pays YES, "no" pays NO); with an unknown result the asks go on.

    At each checkpoint it also records what selling would fetch (read only
    when a sale is simulated, _simulate_at_discount's sell_at): each side's
    bid, 1 - the other side's usable ask on the latest candle at or before
    the checkpoint, when that candle ended at most one candle period before
    it (NaN otherwise, and never carried forward), and whether the market
    had paid out by then (its exact settlement time, with a known payout).
    It records the same once a day over the day-end samples' days, for the
    days the sell rule checks before a sale (TAKE_PROFIT_HOLD_DAYS): each
    day's check is at the moment of the next checkpoint on or after it, less
    whole days, and reads the latest candle that ended in the 24 hours
    before it.

    And at each checkpoint and each day's check, what a synthetic order book
    reads: the contracts traded in the 24 hours up to it
    (depth_model.volume_24h, the snapshot's own definition; NaN when
    unknown), beside the depth model itself.

    Args:
        market (dict): The market's cached record ("ticker", "settlement_ts", "result").
        candles (list[dict]): Its hourly candles ("ts", "yes_ask_close",
            "no_ask_close", "volume"), in time order.
        start_date (date): The backtest's first trading date.
        depth_model (DepthModel | None): The depth model the quotes carry;
            None means no synthetic book.

    Returns:
        tuple[LegQuotes | None, int]: The market's quotes, None when it has no
            candle; and how many day-end samples before its payout were read
            from a candle more than _STALE_QUOTE_DAYS days old (reporting only).
    """
    if not candles:
        return None, 0
    ticker = market.get("ticker") or ""
    settled = _parse_iso_datetime(market.get("settlement_ts"))
    settle_ts = None
    if settled is not None:
        # Kalshi's times end in "Z"; a naive one (a hand-edited cache) is read
        # as UTC, and one outside the representable range as unknown
        if settled.tzinfo is None:
            settled = settled.replace(tzinfo=UTC)
        try:
            settle_ts = settled.timestamp()
            settled = settled.astimezone(UTC)
        except (OverflowError, ValueError, OSError):
            settled, settle_ts = None, None
    result = market.get("result")
    paid_yes = (1.0 if result == "yes" else 0.0 if result == "no" else float("nan"))
    if settle_ts is None:
        # No payout without a known settlement time
        paid_yes = float("nan")
    paid_no = 1.0 - paid_yes
    # From this moment every sample is the payout; None when the payout is unknown
    pays_from = settle_ts if paid_yes == paid_yes else None
    first_ts = min(c["ts"] for c in candles)
    last_ts = max(c["ts"] for c in candles)
    # The first day whose end (the next UTC midnight) a candle has ended by
    first_day = max(start_date, datetime.fromtimestamp(first_ts - 1, UTC).date())
    last_day = (settled.date() if settled is not None
                else datetime.fromtimestamp(last_ts, UTC).date())
    last_day = max(first_day, last_day)
    # Day ends as Unix times: the UTC midnight after each day
    day_ends = [(first_day.toordinal() + 1 + i - _EPOCH_ORDINAL) * _DAY_SECONDS
                for i in range((last_day - first_day).days + 1)]
    # The entry checkpoints over the same days, and the first one's date
    first_checkpoint = first_day
    while first_checkpoint.weekday() != SCHEDULED_RUN.weekday:
        first_checkpoint += timedelta(days=1)
    checkpoints = _monday_timestamps(first_day, last_day)
    stale_limit = _STALE_QUOTE_DAYS * _DAY_SECONDS
    # Each side's candles with a usable ask, in the candles' own order, as
    # small {"ts", "ask"} dicts, so a lookup lands on the latest usable ask
    usable: dict[str, list[dict]] = {}
    for side, key in (("yes", "yes_ask_close"), ("no", "no_ask_close")):
        usable[side] = [{"ts": c["ts"], "ask": ask} for c in candles
                        if (ask := _usable_ask(c.get(key), side)) == ask]

    def sample(moments: list[int], count_stale: bool) -> tuple[list[float], list[float], int]:
        """
        Read each side's latest usable ask at each moment (the payout from settlement on).

        Args:
            moments (list[int]): Unix times, in time order.
            count_stale (bool): Whether to count samples read from an old candle.

        Returns:
            tuple: (YES asks, NO asks, how many samples came from a candle
                more than _STALE_QUOTE_DAYS days before their moment, on
                either side).
        """
        # The one candle lookup _find_entry reads prices through
        found_yes = _candles_at_or_before(usable["yes"], moments)
        found_no = _candles_at_or_before(usable["no"], moments)
        yes_out: list[float] = []
        no_out: list[float] = []
        stale = 0
        for moment, yes_candle, no_candle in zip(moments, found_yes, found_no, strict=True):
            if pays_from is not None and pays_from <= moment:
                yes_out.append(paid_yes)
                no_out.append(paid_no)
                continue
            yes_out.append(float("nan") if yes_candle is None else yes_candle["ask"])
            no_out.append(float("nan") if no_candle is None else no_candle["ask"])
            if count_stale and any(c is not None and moment - c["ts"] > stale_limit
                                   for c in (yes_candle, no_candle)):
                stale += 1
        return yes_out, no_out, stale

    def bids(moments: list[int], recent: Callable[[int], bool]
             ) -> tuple[list[float], list[float], list[bool]]:
        """
        What a sale would fetch at each moment, and whether the market had paid out by then.

        A side's bid is 1 - the other side's usable ask on the latest candle
        at or before the moment, read only when that candle is recent
        enough (`recent` of its age in seconds) and never carried forward.
        Used twice below, with one bid rule for both: at the weekly
        checkpoints, where a sale happens (a candle at most one candle
        period old), and at each day's check of the sell rule's days before
        a sale (TAKE_PROFIT_HOLD_DAYS: that day's last quote, a candle from
        the 24 hours before it). Only the `recent` test differs.

        Args:
            moments (list[int]): Unix times, in time order.
            recent (Callable[[int], bool]): Whether a candle that ended that
                many seconds before a moment may be read there.

        Returns:
            tuple: (YES bids, NO bids, paid-out markers), one per moment;
                NaN where there is no bid.
        """
        yes_out: list[float] = []
        no_out: list[float] = []
        paid_out: list[bool] = []
        for moment, candle in zip(moments, _candles_at_or_before(candles, moments),
                                  strict=True):
            # Paid out by then: its exact settlement time, with a known payout
            is_paid = pays_from is not None and pays_from <= moment
            paid_out.append(is_paid)
            if is_paid or candle is None or not recent(moment - candle["ts"]):
                yes_out.append(float("nan"))
                no_out.append(float("nan"))
                continue
            # historical.candle_sale_bids: the one candle bid rule (1 - the
            # other side's usable ask, to six decimals; NaN with no usable
            # ask), kept in historical so live code can read candles with it
            yes_bid, no_bid = candle_sale_bids(candle)
            yes_out.append(yes_bid)
            no_out.append(no_bid)
        return yes_out, no_out, paid_out

    yes_days, no_days, stale_days = sample(day_ends, True)
    yes_checkpoints, no_checkpoints, _ = sample(checkpoints, False)
    # What a sale would fetch at each checkpoint: a bid from a candle that
    # ended at most one candle period before it
    period = CANDLESTICK_PERIOD_INTERVAL_MINUTES * 60
    yes_bids, no_bids, paid = bids(checkpoints, lambda age: age <= period)
    # ... and at one check a day, over the day-end samples' days, for the
    # days the sell rule reads before a sale: each at the moment of the next
    # checkpoint on or after that day, less whole days (so 24 hours apart,
    # a checkpoint's own date at the checkpoint), reading that day's last
    # quote — a candle that ended in the 24 hours before the check
    run_moments: dict[date, int] = {}
    day_checks: list[int] = []
    for offset in range((last_day - first_day).days + 1):
        day = first_day + timedelta(days=offset)
        ahead = (SCHEDULED_RUN.weekday - day.weekday()) % 7
        run_day = day + timedelta(days=ahead)
        if run_day not in run_moments:
            run_moments[run_day] = int(_checkpoint_datetime(run_day).timestamp())
        day_checks.append(run_moments[run_day] - ahead * _DAY_SECONDS)
    yes_daily, no_daily, paid_daily = bids(day_checks, lambda age: age < _DAY_SECONDS)
    # The 24-hour volume at each checkpoint and at each day's check, read the
    # way a depth snapshot reads it (depth_model.volume_24h needs the candles
    # in time order)
    ordered = sorted(candles, key=lambda c: c["ts"])

    def volumes_at(moments: list[int]) -> list[float]:
        """The contracts traded in the 24 hours up to each moment (NaN: unknown)."""
        return [float("nan") if (v := volume_24h(ordered, moment)) is None else v
                for moment in moments]

    volumes, volumes_daily = volumes_at(checkpoints), volumes_at(day_checks)
    quotes = LegQuotes(ticker, first_day, yes_days, no_days, first_checkpoint,
                       yes_checkpoints, no_checkpoints, paid_yes, paid_no,
                       yes_bids, no_bids, paid, yes_daily, no_daily, paid_daily,
                       volumes, volumes_daily, depth_model)
    return quotes, stale_days


def _attach_leg_quotes(records: Iterable[dict], candles_by_ticker: dict,
                       start_date: date, depth_model: DepthModel | None = None) -> None:
    """
    Give each entry record the prices that value its trades at market: the one writer of "leg_quotes".

    For each record whose two markets both have candles it sets
    rec["leg_quotes"] = {ticker_a: LegQuotes, ticker_b: LegQuotes} on the
    record itself (never on its entry dict, whose keys are the Monday's).
    One LegQuotes is built per ticker per call and shared by every record,
    band and tier setting that holds the market, and no candle list is kept
    alive by it. A record whose markets lack candles (every hand-built
    record) is left without the key, so its trades are valued at cost and
    filled at the top of the book. The quotes also carry the depth model,
    from which Pass 2 builds each trade's synthetic book.
    _simulate_at_discount's Pass 1b is the key's one reader. Called by
    _prepare_entries after its entry pass and by _sweep_from_candidates just
    before it releases the candles.

    Args:
        records (Iterable[dict]): Entry records (_entries_for_band output); a
            record may appear more than once.
        candles_by_ticker (dict): Ticker -> hourly candles, from _fetch_candles_parallel.
        start_date (date): The backtest's first trading date.
        depth_model (DepthModel | None): The depth model; None means every
            trade fills at the top of the book.
    """
    by_ticker: dict[str, LegQuotes | None] = {}
    by_pair: dict[tuple[str, str], dict] = {}
    stale_legs = 0
    for rec in records:
        entry = rec["entry"]
        tickers = (entry["mA"]["ticker"], entry["mB"]["ticker"])
        pair = by_pair.get(tickers)
        if pair is None:
            for m in (entry["mA"], entry["mB"]):
                ticker = m["ticker"]
                if ticker not in by_ticker:
                    quotes, stale_days = _leg_quotes(m, candles_by_ticker.get(ticker) or [],
                                                     start_date, depth_model)
                    by_ticker[ticker] = quotes
                    stale_legs += 1 if stale_days else 0
            if by_ticker[tickers[0]] is None or by_ticker[tickers[1]] is None:
                continue
            pair = by_pair[tickers] = {tickers[0]: by_ticker[tickers[0]],
                                       tickers[1]: by_ticker[tickers[1]]}
        rec["leg_quotes"] = pair
    if stale_legs:
        # Reporting only: no quote is refused for its age
        logging.debug("Legs valued on a quote more than %d days old on some day before they "
                      "pay out: %d", _STALE_QUOTE_DAYS, stale_legs)


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
    *,
    tier_floors: bool = True,
) -> dict | None:
    """
    Find every Monday on which a potential pair was tradeable at the required threshold.

    Scans weekly Monday snapshots up to and including the day before the
    earlier of the two market close dates. The scan's start is the LATER of
    the backtest start date and one calendar year before that end point — the
    one-year figure bounds how far back the window can reach, it does not
    describe where the window ends. At each Monday, reads the candlestick
    prices, applies the price gap, price-sum, and fee filters from the live
    trading logic, and returns the first qualifying Monday's entry data with
    every later qualifying Monday listed under "later". It uses only price
    and date rules — no probability model — so it knows nothing about k or
    the size cap, and one pass per spread band (and tier setting) serves
    every k and cap; _simulate_at_discount decides, per k, which of them the
    pair may be traded on.
    Each leg's candles are read in one pass (_candles_at_or_before) up to
    just past the window's last Monday, so a malformed candle in that
    stretch can raise (KeyError, TypeError).

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
        are not traded for this pair type (the Kelly gate in
        _simulate_at_discount reads all four quotes for the mid spread its
        forecast prices at). The deadline gap driving
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
    In both cases the two legs' top-of-the-book quotes come from
    _leg_prices_for, and both of them must be live [0.01, 0.99] quotes.

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
    default reproduces the tier rule alone, and the live band and tier
    setting reproduce config.time_series_spread_refusal
    (TestLiveBacktestSpreadParity). same_title pairs never read the band.

    tier_floors (BACKTEST-only, like the band) switches the deadline-gap tier
    itself off: the threshold is then the band floor alone, which drives the
    gap test and the leg-price-sum ceiling exactly as max(tier, floor) does,
    so the sum ceiling stays tied to the floor. Nothing else reads it — the
    30-day gap cap, the ceiling, the live-quote and fee checks, the ladder
    rule and every same-title pair are the same either way. At a floor of 0
    what is left of the gap test is that pB must EXCEED pA: a time-series
    Monday whose spread is not strictly positive (pB − pA <= PRICE_EPSILON)
    is refused whatever the tiers, because a pair with no in-between mass
    has nothing to dispute, as the live spread rule refuses it
    (config.time_series_spread_refusal). That refusal is inert with the
    tiers on, where every tier already demands more; at a floor of 0 with
    them off it is the one gap condition left, beside a sum ceiling of 1 the
    fee check implies.

    Scanning stops at the earlier close date (not the later one) because after
    the first market closes, the pair is no longer open for entry.

    Args:
        candles_a (list[dict]): Market A's hourly price history, oldest first:
            one dict per hour of data with "ts" (the hour's end, Unix seconds)
            and the YES ask and NO ask at its close ("yes_ask_close",
            "no_ask_close"; the NO ask estimated as 1 − the YES bid).
            _entries_for_band passes the candles _fetch_candles_parallel
            fetched for mA's ticker.
        candles_b (list[dict]): The same for market B.
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
        tier_floors (bool): Keyword-only, BACKTEST-only. False enters
            time-series pairs at the band floor alone, the deadline-gap tier
            not applied (the band sweep's tier-floors-off family) — still
            only at a strictly positive spread; True (the default) layers
            the floor on the tier. Ignored by same_title pairs.

    Returns:
        Optional[dict]: A dict with keys "entry_date" (date), "pA" (float), "pB"
            (float), "nA" (float), "nB" (float), "mA" (dict), "mB" (dict),
            "gap_days" (int | None) for the first qualifying Monday, and
            "later" (tuple[dict, ...]). pA, pB, nA and nB are all four
            quotes of the canonicalized A and B (YES ask and NO ask of each),
            of which _leg_prices_for picks the two legs' quotes.
            "gap_days" is the loop-invariant deadline gap the price tier and
            the MAX_DEADLINE_GAP_DAYS cutoff were applied on, carried out for
            reporting so a report cannot bucket a pair under a gap it was not
            filtered by; it is None for same_title, which has no deadline-gap
            concept.
            "later" holds one dict per later qualifying Monday, earliest
            first, with every key above but "later": that Monday's date,
            prices and mA/mB (for a same-title pair, which market is A is
            decided afresh each Monday, so the two can swap; the market dicts
            are the same objects, not copies) and the same gap_days. () when
            the pair qualified on one Monday only. Read them through
            _entry_mondays().
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
        # untouched — and, with tier_floors False (backtest-only), drops the
        # tier and returns band_lo alone. This one threshold drives BOTH the
        # gap test and the leg-price-sum ceiling below, which is what keeps
        # the sum ceiling at 1 - floor when the band raises the floor (the
        # live pairing of the two, applied to the raised floor rather than to
        # the tier alone), and at 1 - floor when the tier is dropped too.
        threshold = min_price_diff_for_gap(gap_days, spread_min=band_lo,
                                           tier_floors=tier_floors)
    else:
        # same_title pairs have no deadline-gap concept — flat 5% threshold
        threshold = SAME_TITLE_MIN_PRICE_DIFF
        # No deadline gap to report for same_title. The time_series branch
        # above assigns gap_days on both its try and except paths, so this is
        # the only branch where the name would otherwise be unbound below.
        gap_days = None

    # Every qualifying Monday, earliest first (see Returns)
    mondays: list[dict] = []
    # The Mondays to test (SCHEDULED_RUN's weekday), each at its checkpoint
    checkpoints = _monday_timestamps(scan_start, scan_end)
    # candles_a is market A's hourly price history and candles_b market B's
    # (see Args). If the time-series branch above swapped the two markets so
    # that A is the earlier contract, it swapped these lists too. For each
    # Monday, candles_at_a and candles_at_b hold that market's latest candle
    # at or before that Monday's checkpoint (its prices at that moment), or
    # None if it had no candle yet.
    candles_at_a = _candles_at_or_before(candles_a, checkpoints)
    candles_at_b = _candles_at_or_before(candles_b, checkpoints)
    for ts, ca, cb in zip(checkpoints, candles_at_a, candles_at_b, strict=True):
        entry_date = datetime.fromtimestamp(ts, tz=UTC).date()

        # Optional opt-in bet-horizon cap: skip checkpoints where the
        # later-closing leg would close further out than max_horizon_days from
        # THIS simulated checkpoint — a cheap comparison done before reading
        # this Monday's quotes. Deliberately max(close_a, close_b) rather than close_b:
        # same_title has no ordering at all, and since DR-73 a same-event
        # LADDER is ordered by STATED deadline, so close_b >= close_a no longer
        # holds for every time_series pair either (in the archive 13 dated
        # same-event pairs are ordered the wrong way round by realized close,
        # and 681 have a close gap of ZERO days). The max() keeps this correct
        # under all three; close_time is read here ON PURPOSE, because this
        # bounds when a market is still OPEN, not when its deadline falls.
        if max_horizon_days is not None and (max(close_a, close_b) - entry_date).days > max_horizon_days:
            continue

        # A leg with no candle at or before this Monday has no quote to test
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
        # the profit model's mid spread, so a dead quote there is just as
        # disqualifying.
        if not (0.01 <= p_a_raw <= 0.99 and 0.01 <= p_b_raw <= 0.99):
            continue

        if pair_type == "time_series":
            # A stays the earlier-closing market — no price canonicalization.
            # The anomaly is the LATER contract priced higher: gap = pB - pA.
            mA_i, mB_i = mA, mB
            pA, pB, nA, nB = p_a_raw, p_b_raw, n_a_raw, n_b_raw
            gap = pB - pA
            # A Monday with no in-between mass (pB - pA not strictly positive)
            # has nothing to dispute, as the live spread rule says
            # (config.time_series_spread_refusal). Inert with the tiers
            # on, since every tier > 0 already demands more; it keeps the
            # tier-floors-off family, whose floor-0 threshold is 0.0, from
            # entering such a pair. PRICE_EPSILON sits on the REJECT side here
            # and TIGHTENS, like scanner's pA + nB guard: a zero spread that
            # evaluates a hair above 0 is still refused, while a genuine
            # one-tick spread (>= $0.0001) is two orders of magnitude clear.
            if gap <= PRICE_EPSILON:
                continue
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

        # The two legs' top-of-the-book quotes — (nA, pB) for same_title,
        # (pA, nB) for time_series — via the module's single leg mapping
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
        # floor, so under a raised floor this is 1 - max(tier, floor) (and
        # 1 - floor with the tier floors off).
        if price_a + price_b > 1.0 - threshold + PRICE_EPSILON:
            continue

        # Check that the gross spread on the leg prices exceeds the continuous fee estimate
        if (1.0 - price_a - price_b) <= fee_per_pair_approx(price_a, price_b):
            continue

        mondays.append({
            "entry_date": entry_date,
            "pA": pA, "pB": pB, "nA": nA, "nB": nB,
            # This Monday's legs (for a same-title pair, A is chosen afresh each Monday)
            "mA": mA_i, "mB": mB_i,
            # The gap the tier above was selected from, carried out so the
            # calibration report buckets each pair under exactly the gap it
            # was filtered by. None for same_title. Reporting only — every
            # decision this value drives was already made above.
            "gap_days": gap_days,
        })

    if not mondays:
        return None
    # The first qualifying Monday is the entry itself (the calibration and the
    # split date read only this one); the rest go under "later"
    entry = mondays[0]
    entry["later"] = tuple(mondays[1:])
    return entry


def _entry_mondays(entry: dict) -> tuple[dict, ...]:
    """
    Every qualifying Monday of one _find_entry() result, earliest first.

    The entry itself is the first Monday and its "later" list holds the rest
    (DR-75). An entry with no "later" key (tests build some by hand) counts
    as one Monday. _simulate_at_discount's Kelly gate, _ex_top_event,
    CapSweep.entry_events and _log_qualifying_mondays read the Mondays
    through this function; an AST test in tests/test_strategy.py checks that
    only _find_entry, this function and _split_halves use the "later" key.

    Args:
        entry (dict): A _find_entry() result — a record's "entry".

    Returns:
        tuple[dict, ...]: (entry, *entry["later"]); never empty. Each item
            carries "entry_date", "pA", "pB", "nA", "nB", "mA", "mB" and
            "gap_days" for its own Monday.
    """
    return (entry, *entry.get("later", ()))


def _log_qualifying_mondays(entries: list[dict], band: tuple[float, float] | None,
                            *, tier_floors: bool = True) -> None:
    """
    Log how many qualifying Mondays one entry pass recorded (DR-75).

    Called by _prepare_entries after its entry pass, and by
    _sweep_from_candidates once per band (tier floors on and off), right
    after that band's prepared-entries line. It logs even at zero, so the log
    always shows whether any pair qualified on more than one Monday (DR-66).
    The line starts differently from the log lines tests count ("Prepared ",
    "Spread band ", "Tier floors off", "Split-half check").

    Args:
        entries (list[dict]): One _entries_for_band() output (or a band's
            time-series + same-title concatenation) — records carrying an
            "entry".
        band (tuple[float, float] | None): The band the pass ran at; None
            names the default band.
        tier_floors (bool): Keyword-only; False appends " with the tier
            floors off" after the band, as the completion lines do.

    Returns:
        None
    """
    counts = [len(_entry_mondays(rec["entry"])) for rec in entries]
    logging.info("Qualifying Mondays at band %s%s: %d over %d entries (%d qualify on more "
                 "than one)",
                 # None means the default band; label it the way the
                 # completion lines do
                 _band_label(time_series_spread_band(band)),
                 " with the tier floors off" if tier_floors is False else "",
                 sum(counts), len(counts), sum(1 for n in counts if n > 1))


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


def _settled_after(settled: datetime | None, cutoff_ts: int | None) -> bool:
    """
    Whether a market settled at or after Kalshi's archive cutoff (its candles are on the live API).

    Args:
        settled (datetime | None): The market's settlement time; a naive one
            is read as UTC.
        cutoff_ts (int | None): The archive cutoff, Unix seconds; None when
            not known.

    Returns:
        bool: True only when both are known and the settlement is at or after
            the cutoff.
    """
    if settled is None or cutoff_ts is None:
        return False
    if settled.tzinfo is None:
        settled = settled.replace(tzinfo=UTC)
    try:
        return settled.timestamp() >= cutoff_ts
    except (OverflowError, ValueError, OSError):
        return False


def _fetch_candles_parallel(
    hist_client: Any,
    needed_tickers: dict[str, dict],
    start_date: date,
    use_cache: bool,
    *,
    cutoff_ts: int | None = None,
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
    the main thread before any worker starts (BS-07). Since DR-74 no candidate
    pair of either type carries such a market — the time-series sweep sets it
    aside and the same-title close gate refuses the pair — so this branch is
    defence in depth for a caller that hands one over directly.

    Worker exceptions are deliberately NOT caught: fetch_candlesticks already
    fail-softs network errors to an empty list internally, so anything that
    still escapes is a real defect (e.g. a market that should have been
    prefiltered out) and must surface rather than be silently degraded into
    "this ticker has no prices".

    Before returning, the tickers that resolved to an empty series are counted
    and reported in ONE summary WARNING (silent at zero) — the per-run signal
    that replaces reading hundreds of individual 404 lines (TS-02).

    Each ticker is handed its series (historical.series_ticker of its event
    ticker — the literal prefix, never scanner.event_series, which collapses
    the KXMVE* family), so fetch_candlesticks can ask Kalshi's live
    candlestick endpoint for a market the archive does not hold yet: one that
    settled at or after the archive cutoff, which is asked there first
    (live_first). The routing only saves requests — a 404 from the endpoint
    asked first asks the other one — so a cutoff that moved since cutoff_ts
    was read changes no result. A market with no event ticker is asked of the
    archive only, with exactly the call this function always made.

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
        cutoff_ts (int | None): Keyword-only. Kalshi's archive cutoff (Unix
            seconds): a market that settled at or after it is asked of the
            live endpoint first. None (default, and any corpus with no
            recorded cutoff) asks the archive first.

    Returns:
        dict[str, list[dict]]: Ticker -> candle list (keys: ts, yes_ask_close,
        no_ask_close, volume). Every key of needed_tickers is present; the value is an
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
    work: list[tuple[str, int, int, dict]] = []
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
        # Which endpoint to ask first: the live API for a market settled at or
        # after the archive cutoff (the archive answers 404 for it). Passed
        # only with a series, so a market with no event ticker gets exactly
        # the archive-only call it always got.
        series = series_ticker(m.get("event_ticker") or "")
        endpoint: dict = {}
        if series:
            settled = _parse_iso_datetime(m.get("settlement_ts"))
            endpoint = {"series": series, "live_first": _settled_after(settled, cutoff_ts)}
        work.append((ticker, open_ts, close_ts, endpoint))

    if work:
        with ThreadPoolExecutor(max_workers=CANDLESTICK_FETCH_MAX_WORKERS) as pool:
            # Returns list[dict] with keys: ts (unix int), yes_ask_close (float),
            # no_ask_close (float), volume (float or None) — cached per ticker,
            # so a second run is much faster
            futures = {
                pool.submit(fetch_candlesticks, hist_client, ticker,
                            open_ts, close_ts, use_cache, **endpoint): ticker
                for ticker, open_ts, close_ts, endpoint in work
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

    # Summarize the misses ONCE: the count is the useful signal, not one
    # warning per ticker (TS-02). A ticker neither endpoint serves is never
    # cached, so it is asked again on every run. Counted off the RESULT dict
    # rather than off caught exceptions, because fetch_candlesticks already
    # fail-softs a failure to [] internally; deliberately outside the `if work`
    # block so the tickers resolved to [] above for a missing or unparseable
    # close_time are counted too.
    empty = sum(1 for series in candles_by_ticker.values() if not series)
    if empty:
        logging.warning(
            "Candlestick fetch: %d of %d tickers returned no candles "
            "(neither Kalshi's archive nor its live candlestick endpoint served them; "
            "never cached, so asked again next run)",
            empty, len(candles_by_ticker),
        )
    # Tickers whose every candle lacks a traded-volume count: one line, so a
    # renamed API field shows up instead of passing as quiet markets.
    with_candles = [series for series in candles_by_ticker.values() if series]
    no_volume = sum(1 for series in with_candles
                    if all(c.get("volume") is None for c in series))
    if no_volume:
        logging.info(
            "Candlestick fetch: %d of %d tickers' candles carry no traded volume",
            no_volume, len(with_candles),
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
    # that simply has no records — a cause the surrounding "Markets to
    # analyze" and "Eligibility prefilter" lines already name. The census still
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
        # The "equivalently" clause names BOTH assembled-cache spellings: since
        # SS-1 the assembled cache is the streamed settled_markets_*.jsonl.gz,
        # and a legacy settled_markets_*.json may still sit beside it, so
        # naming the .json alone left the streamed cache to be served again.
        # dashboard._label_coverage_html renders the same remedy and must be
        # kept in step with this sentence.
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
            "assembled backtest_cache/settled_markets_*.jsonl.gz and any legacy "
            "settled_markets_*.json); --no-cache ALONE "
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
            figure "Markets to analyze" reports.
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


def _log_corpus_prefilter(total: int, eligible: int,
                          provenance: CorpusProvenance | None) -> None:
    """
    Log what the corpus holds and what the eligibility prefilter did to it.

    The corpus is the settled markets the backtest analyses. The fetch
    already applied _can_ever_enter, so the first pass's re-check normally
    rejects nothing. The corpus's provenance decides the wording:

      * provenance present (the corpus came from fetch_all_settled_markets):
        N ELIGIBLE markets; the line names the cache tag and the prefilter's
        rejections, if the corpus recorded them. Anything the re-check
        rejects is a WARNING: _can_ever_enter changed without a
        config.SETTLED_PREFILTER_CACHE_TAG bump, or the cache was altered.
      * no provenance (a plain list: a test stub or a hand-built corpus): the
        re-check IS its prefilter, and the second line reads "Eligibility
        prefilter: skipping X/N ...".

    Both lines are logged on every run, healthy or not: the first
    always at INFO, the re-check at INFO when it rejects nothing and at
    WARNING otherwise. A zero-trade run's first question — did the prefilter
    leave anything? — is answered by the first line on its own, which on
    that WARNING path names both counts ("N assembled as eligible, E still
    eligible after the re-check below") rather than call all N eligible on
    the line above a WARNING that says some of them are not.

    Args:
        total (int): Every record the first pass walked.
        eligible (int): Those that passed _can_ever_enter in that pass.
        provenance (CorpusProvenance | None): The corpus's provenance, read by
            type before the corpus is released; None for a plain list.
    """
    rejected_here = total - eligible
    # The cache tag the fetch assembled this corpus under
    tag = _prefilter_cache_tag()
    if provenance is None:
        logging.info(
            "Markets to analyze: %d (no assembly record — whether a prefilter "
            "ran while this corpus was assembled is unknown)", total,
        )
        logging.info(
            "Eligibility prefilter: skipping %d/%d markets that cannot appear "
            "in any tradeable pair", rejected_here, total,
        )
        return
    counts = provenance.assembly_counts
    when = "at this cache's assembly" if provenance.from_cache else "by this run"
    # The corpus is N eligible markets only while the re-check agrees. When
    # it rejects anything (the WARNING below), name both counts: the line
    # above that WARNING must not call all N eligible.
    if rejected_here == 0:
        size, size_args = "%d eligible", (total,)
    else:
        size = "%d assembled as eligible, %d still eligible after the re-check below"
        size_args = (total, eligible)
    if counts is not None:
        logging.info(
            "Markets to analyze: " + size + " — the eligibility prefilter (%s) "
            "ran during assembly and rejected %d of the %d records settled in "
            "the window (%d more were duplicate or blank tickers; counted %s)",
            *size_args, tag, counts.rejected,
            counts.settled, counts.duplicates, when,
        )
    else:
        logging.info(
            "Markets to analyze: " + size + " — the eligibility prefilter (%s) "
            "ran during assembly, but this cache records no count of the "
            "records it rejected%s",
            *size_args, tag,
            # An extended cache drops its counts; its extension logged its own
            "" if provenance.full_assembly_at is None
            else " (it was extended since its full assembly; the extension's own "
                 "counts are on the line logged when it ran)",
        )
    if rejected_here == 0:
        logging.info(
            "Eligibility prefilter re-check: 0 of %d markets rejected — none "
            "expected, since it already ran during assembly", total,
        )
    else:
        logging.warning(
            "Eligibility prefilter re-check: %d of %d markets rejected, although "
            "the prefilter (%s) already ran during this corpus's assembly and "
            "should have left none — backtester._can_ever_enter has changed "
            "without a config.SETTLED_PREFILTER_CACHE_TAG bump, or the cache "
            "file was altered. They are dropped here; bump the tag so the cache "
            "is rebuilt under the current predicate",
            rejected_here, total, tag,
        )


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

    The schedule and feasibility checks, the settled-market fetch, the
    prefilter, the outcome-label census, both groupings, pair extraction and
    the candlestick fetch depend only on which markets exist and when they
    traded, not on the spread band or k, so one call can feed an entry pass
    per band through _entries_for_band().

    The corpus (the settled markets the fetch returns, normally streamed off
    disk) is walked TWICE: _index_eligible_keys counts it, prefilters it,
    feeds the census and hashes each eligible record's two grouping keys;
    _materialize_groupable then keeps only records whose key another shares
    (any other would be a single-member group, which both groupings drop),
    so the groups and pairs are exactly those of the whole eligible list.

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
            of ONE event may pair. None (the default) resolves this
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
            OutcomeLabelCoverage, the corpus's provenance (None for a plain
            list corpus), and the start date / horizon / ladder flag every
            entry pass must reuse. None — the codebase's
            return-None-on-validation-failure convention — when the
            feasibility pre-check fails (no entry checkpoint in [start_date,
            today]), a "no simulation is possible in this window at all"
            signal distinct from "no pair was ever tradeable"; the fetch
            never ran on that path, so no census exists either.

    Raises:
        ValueError: Before any fetch, when SCHEDULED_RUN cannot place the
            entry checkpoints: an unresolvable zone, or a run date (from
            start_date to _SCHEDULE_CHECK_DAYS_AHEAD past today) that a clock
            change skips or repeats, or whose UTC moment is on another date or
            out of datetime's range. A configuration error, so it raises
            rather than returning None.
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

    # Feasibility pre-check, BEFORE any network call: if [start_date, today]
    # holds no run weekday, no trade can be entered, so skip the fetch. The
    # end is today, NOT yesterday (a market that settled early can still
    # carry a future close_time), and in UTC, not local (the local date lags
    # UTC for hours west of it): this guard must be strictly conservative.
    feasibility_end = datetime.now(UTC).date()
    # Every run date the backtest may scan must put the run's UTC moment on
    # that same UTC date, exactly once. A breach is a configuration error:
    # it raises before any fetch, never the None meaning "nothing to simulate".
    schedule_end = (date.max if (date.max - feasibility_end).days < _SCHEDULE_CHECK_DAYS_AHEAD
                    else feasibility_end + timedelta(days=_SCHEDULE_CHECK_DAYS_AHEAD))
    try:
        problems = SCHEDULED_RUN.date_problems(start_date, schedule_end)
    except (ZoneInfoNotFoundError, ValueError, OSError) as exc:
        raise ValueError(
            f"config.SCHEDULED_RUN ({SCHEDULED_RUN.label()}) cannot place the "
            f"backtest's entry checkpoints: its zone cannot be resolved "
            f"({type(exc).__name__}: {exc})"
        ) from exc
    if problems:
        raise ValueError(
            f"config.SCHEDULED_RUN ({SCHEDULED_RUN.label()}) cannot serve as the "
            f"backtest's entry checkpoint: {len(problems)} run date(s) in "
            f"[{start_date}, {schedule_end}] do not put the run at one UTC "
            f"instant on the same date, e.g. {'; '.join(problems[:3])}"
        )
    if not _monday_timestamps(start_date, feasibility_end):
        logging.warning(
            "No %s entry checkpoint (config.SCHEDULED_RUN) exists in [%s, %s] "
            "— no trade can ever be entered; skipping the fetch entirely",
            SCHEDULED_RUN.label(), start_date, feasibility_end,
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
    # The cache tag names the prefilter's version and schedule; bump
    # config.SETTLED_PREFILTER_CACHE_TAG whenever _can_ever_enter changes.
    markets = fetch_all_settled_markets(
        hist_client, live_client, start_date, use_cache,
        prefilter=lambda m: _can_ever_enter(m, start_date),
        prefilter_tag=_prefilter_cache_tag(),
    )
    # What the corpus says about itself — assembly time, cache or fresh, and
    # the archive cutoff as of assembly — taken now, before `markets` is
    # released, for the dashboard header (DR-13).
    # Read by TYPE, never by attribute probing: a plain list (a test stub) has
    # no provenance, and a MagicMock would answer any attribute with nonsense.
    corpus_provenance = (
        markets.provenance if isinstance(markets, SettledCorpus) else None
    )
    # Pass 1 of the two walks the docstring describes; the corpus may be any
    # re-iterable of market dicts (a test stub's list is resident until the
    # `del markets` below). The prefilter is re-applied
    # although the fetch already applied it: it is idempotent, costs no extra
    # walk, and keeps the guarantee beside the O(n^2) pairing that needs it.
    census = _OutcomeLabelTally()
    key_index = _index_eligible_keys(markets, start_date, census)
    # What the corpus is (eligible markets, when the fetch prefiltered it) and
    # what the prefilter rejected — during assembly, as the corpus recorded
    # it, and in the re-check just run (M9)
    _log_corpus_prefilter(key_index.total, key_index.eligible, corpus_provenance)

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
    # corpus held as a list (a test stub) this is
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
    # A fetched corpus is never resident (a legacy settled_markets_*.json
    # cache, the one format handed over as a list, is no longer served), so
    # the groupable count is what this warning has to cover. It deliberately carries only THIS
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
    # Both groupings' sizes, logged on EVERY run, zero included (M10). A
    # grouping with no group of two or more members reaches _extract_pairs as
    # an empty dict, which cannot say which kind of grouping it was, and
    # every per-candidate line there is silent at zero — so without this
    # line an empty grouping and one whose every candidate was refused
    # would read the same ("Potential pairs: 0 ..." and nothing else),
    # absence of a warning being the only signal (DR-66). Linear in the
    # number of groups; the groupable-subset line above counts the markets
    # that share a key, this one what those keys actually grouped.
    logging.info(
        "Groups of two or more markets: %d time-series (%d markets), "
        "%d same-title (%d markets) — pairs form only inside a group",
        len(ts_groups), sum(len(v) for v in ts_groups.values()),
        len(same_groups), sum(len(v) for v in same_groups.values()),
    )
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
    # The archive cutoff the corpus was assembled (or extended) under routes
    # each ticker to the endpoint that holds its candles; None (a corpus with
    # no recorded cutoff) asks the archive first and the live API on a 404
    cutoff = None if corpus_provenance is None else corpus_provenance.archive_cutoff
    candles_by_ticker = _fetch_candles_parallel(
        hist_client, needed_tickers, start_date, use_cache,
        cutoff_ts=None if cutoff is None else int(cutoff.timestamp()),
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
        corpus_provenance=corpus_provenance,
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
    Render a resolved spread band as "floor-ceiling" (log lines, the dashboard's tier-off labels).

    One definition, so the Phase-1 announcement of a band sweep and the
    completion line of every simulation at that band spell it identically —
    a reader can match them by text. dashboard.py labels a band's
    tier-floors-off view with it too (that run was gated on its floor alone,
    so the page's own "max(tier,<floor>)" would misname it), so the page and
    the log name that run alike. Each bound is formatted with :g, which
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


# _sweep_from_candidates' default for live: read the saved live defaults there
# (a direct caller); None is a read run_backtest_sweep took that found none
# usable.
_LIVE_NOT_READ = object()


def _live_settings_for_report() -> LiveSettings | None:
    """
    Read the saved live defaults for this run's report, failing soft.

    Reporting only: the backtest itself sizes at this module's own k and caps
    (config.py's, bound by value), never the saved defaults'. With none saved
    it logs one INFO line, and with a refused file one WARNING, instead of
    aborting the run.

    Returns:
        LiveSettings | None: The saved defaults; None when none are saved or
            the saved file is refused.
    """
    try:
        # The file a live run starts from (config.LIVE_DEFAULTS_FILE)
        return live_defaults()
    except LiveDefaultsMissing:
        logging.info("No live defaults are saved, so live runs refuse to start — the "
                     "live rule is not recorded on this run's report")
        return None
    except LiveDefaultsError as e:
        logging.warning("The saved live defaults are refused (%s) — the live rule is "
                        "not recorded on this run's report", e)
        return None


def _live_rule_fields(live: LiveSettings | None) -> dict:
    """
    Name the BacktestSweep keywords that record the saved live defaults.

    Both production BacktestSweep constructions spread this (beside
    same_title_size_cap), so the nine fields are recorded together from one
    read or not at all.

    Args:
        live (LiveSettings | None): _live_settings_for_report()'s result.

    Returns:
        dict: The nine live_* keywords from live; {} when live is None.
    """
    if live is None:
        return {}
    return {"live_tier_floors": live.tier_floors, "live_spread_band": live.spread_band,
            "live_categories": live.categories, "live_tags": live.tags,
            "live_origin": live.origin, "live_interval_discount": live.interval_discount,
            "live_size_cap": live.size_cap,
            "live_same_title_size_cap": live.same_title_size_cap,
            "live_add_to_held_pairs": live.add_to_held_pairs}


# The label a recorded live rule is named with, in the log line and page header
_LIVE_RULE_LABEL = "saved live defaults"
# The line and header when the run recorded none
_LIVE_RULE_NONE = ("none recorded — no usable live defaults were saved when this run "
                   "started, and live runs refuse to start without them")


def _live_sizing_note(sweep: BacktestSweep) -> str:
    """
    State the saved defaults' k and caps when this run's primary sized at others.

    A backtest sizes at config.py's k and caps (this module's by-value
    bindings), not the saved defaults', so its primary scenario can hold the
    live rule's band but not its sizing. The page adds where its filter bar
    shows the live sizing (dashboard._live_rule_html); the log line cannot
    know.

    Args:
        sweep (BacktestSweep): The run's sweep.

    Returns:
        str: A "; ..." clause, or "" when a value is unrecorded or all three agree.

    Raises:
        Nothing. It only formats values the sweep already holds.
    """
    live = (sweep.live_interval_discount, sweep.live_size_cap, sweep.live_same_title_size_cap)
    run = (sweep.primary.k, sweep.primary.size_cap, sweep.same_title_size_cap)
    if None in live or None in run or live == run:
        return ""

    def sizing(k: float, cap: float, same_title: float) -> str:
        """
        Name one k and pair of caps as the note words them.

        Args:
            k (float): An interval discount, printed exactly.
            cap (float): A per-trade cap as a fraction; 1.0 reads
                "100% (no cap)".
            same_title (float): A same-title cap as a fraction; 1.0 reads
                "100% (no extra cap)".

        Returns:
            str: "k X, per-trade cap Y and same-title cap Z".

        Raises:
            Nothing.
        """
        return (f"k {_exact_number(k)}, per-trade cap {_cap_text(cap, 'no cap')} and "
                f"same-title cap {_cap_text(same_title, 'no extra cap')}")

    return (f"; the live defaults size at {sizing(*live)}, where this run's primary sized "
            f"at {sizing(*run)}")


def _live_add_on_note(sweep: BacktestSweep) -> str:
    """
    Say when the saved live defaults add to held pairs and this run's headline figures do not.

    The backtest's own simulations (its primary scenario and every scenario
    it runs) never add to held pairs. Only the dashboard's "Add to held
    pairs: on" view does, simulated when the page is built. So when the
    saved live defaults add to held pairs, the live-rule log line ends with
    this clause, saying the primary differs from live there.

    Args:
        sweep (BacktestSweep): The run's sweep.

    Returns:
        str: "; the live defaults add to held pairs, which this run's primary
            does not" when live_add_to_held_pairs is True, else "" (off, or
            not recorded).
    """
    if sweep.live_add_to_held_pairs is not True:
        return ""
    return "; the live defaults add to held pairs, which this run's primary does not"


def _live_filter_text(categories: tuple[str, ...] | None,
                      tags: tuple[str, ...] | None) -> str:
    """
    Name a recorded category/tag filter as config.describe_trade_filter does.

    Same words from the two recorded fields, through config._names_text (pinned
    equal by tests/test_backtester.py::TestLiveRuleLine).

    Args:
        categories (tuple[str, ...] | None): BacktestSweep.live_categories.
        tags (tuple[str, ...] | None): BacktestSweep.live_tags.

    Returns:
        str: e.g. "categories Economics, Sports; tags any".
    """
    return f"categories {_names_text(categories)}; tags {_names_text(tags)}"


def _live_filter_is_one_slice(categories: tuple[str, ...] | None,
                              tags: tuple[str, ...] | None) -> bool:
    """
    Say whether a live category/tag filter is ONE of the filter bar's options.

    The bar offers one Category, or one "C · T" Tag option, at a time: one
    category with at most one tag is one option; anything else may span
    several.

    Args:
        categories (tuple[str, ...] | None): The live categories, None for any.
        tags (tuple[str, ...] | None): The live tags, None for any.

    Returns:
        bool: True when the filter is one Category or one Tag option.
    """
    return categories is not None and len(categories) == 1 and len(tags or ()) <= 1


# _LiveRuleView.where: this run's primary scenario holds the saved live
# defaults' time-series rule, another cell of its grid does, or none does.
_LIVE_RULE_PRIMARY = "primary"
_LIVE_RULE_GRID = "grid"
_LIVE_RULE_NOT_SIMULATED = "not simulated"


@dataclass(frozen=True)
class _LiveRuleView:
    """
    Where a run's own grid holds the saved live defaults' time-series rule.

    Attributes:
        where (str): One of the three _LIVE_RULE_* constants.
        tier_floors (bool): The holding cell's Tier floors setting: the live
            one, except True where no tier binds (its on and off cells are one).
    """
    where: str
    tier_floors: bool


def _live_rule_view(sweep: BacktestSweep) -> _LiveRuleView | None:
    """
    Say where a run's own grid holds the saved live defaults' time-series rule.

    The one definition, shared by _live_rule_line and dashboard._live_rule_html
    and judged only on what the sweep RECORDED: the primary scenario when its
    band is the live band and the live tiers are on or never bind there; another
    cell when the live band was simulated and, tier-off at a binding band, the
    tier-off family covers every binding band (dashboard._tier_off_binds).

    Args:
        sweep (BacktestSweep): The run's sweep.

    Returns:
        _LiveRuleView | None: The verdict; None when no live rule is recorded.
    """
    tier_floors, band = sweep.live_tier_floors, sweep.live_spread_band
    if tier_floors is None or band is None:
        return None
    shown_on = bool(tier_floors) or not _tier_floors_bind(band)
    if shown_on and band == sweep.primary.spread_band:
        return _LiveRuleView(_LIVE_RULE_PRIMARY, True)
    bands = sweep.calibrations_by_band
    if band not in bands:
        reachable = False
    elif shown_on:
        reachable = True
    else:
        reachable = bool(sweep.tier_off_scenarios) and all(
            b is not None and (b in sweep.tier_off_calibrations_by_band
                               or not _tier_floors_bind(b))
            for b in bands)
    return _LiveRuleView(_LIVE_RULE_GRID if reachable else _LIVE_RULE_NOT_SIMULATED,
                         shown_on)


def _live_rule_ladder_note(sweep: BacktestSweep) -> str:
    """
    Qualify a live-rule verdict whose run departs from config.py's ladder switch.

    Which pairs exist is the ladder switch's (DR-73), not the entry rule's.

    Args:
        sweep (BacktestSweep): The run's sweep.

    Returns:
        str: A "; ..." clause, or "" when the two agree or either is unrecorded.
    """
    run, configured = sweep.same_event_ladders, sweep.config_same_event_ladders
    if run is None or configured is None or bool(run) == bool(configured):
        return ""
    return (f"; this run's same-event ladders are {'on' if run else 'off'} and "
            f"config.py's {'on' if configured else 'off'}, so its pairs are not the "
            "live bot's")


def _live_rule_line(sweep: BacktestSweep) -> str:
    """
    Word the "Live time-series rule (saved live defaults): ..." log line from a built sweep.

    Always a line, so a report never silently omits the live rule: with none
    recorded it says so (_LIVE_RULE_NONE). A recorded rule's line ends with
    _live_sizing_note when the saved defaults size differently from this
    run's primary, then _live_add_on_note when they add to held pairs.

    Args:
        sweep (BacktestSweep): The run's sweep.

    Returns:
        str: The line.
    """
    view = _live_rule_view(sweep)
    if view is None:
        return f"Live time-series rule: {_LIVE_RULE_NONE}"
    prefix = f"Live time-series rule ({_LIVE_RULE_LABEL}): "
    rule = describe_time_series_rule(sweep.live_tier_floors, sweep.live_spread_band)
    categories, tags = sweep.live_categories, sweep.live_tags
    filtered = categories is not None or tags is not None
    if filtered:
        rule += f"; category/tag filter ({_live_filter_text(categories, tags)})"
    if view.where == _LIVE_RULE_NOT_SIMULATED:
        return (f"{prefix}{rule} — not simulated by this run{_live_sizing_note(sweep)}"
                f"{_live_add_on_note(sweep)}")
    # Tier floors off at a band no tier binds at: the tier-on cell holds it
    never_binds = ("" if sweep.live_tier_floors or not view.tier_floors else
                   " (no tier floor binds at this band, so off and on are one rule)")
    if view.where == _LIVE_RULE_PRIMARY:
        tail = ("this run's primary scenario applies " + ("its time-series rule" if filtered
                                                          else "it") + never_binds)
        subject = "it"
    else:
        primary = sweep.primary.spread_band
        tail = (f"this run's primary scenario does not (tier floors on, band "
                f"{'not recorded' if primary is None else _band_label(primary)}); its "
                f"grid simulated the live rule as band {_band_label(sweep.live_spread_band)} "
                f"with the tier floors {'on' if view.tier_floors else 'off'}{never_binds}, "
                "which the dashboard's filter bar shows")
        subject = "that scenario"
    if filtered:
        # Only the page knows which options the bar offers, so it names them
        tail += ("; the dashboard's filter bar shows the live category/tag filter as one "
                 f"Category or Tag option of {subject} (offered where this run filed a "
                 "pair under it)"
                 if _live_filter_is_one_slice(categories, tags) else
                 "; the dashboard's filter bar shows the live category/tag filter one "
                 f"Category or Tag option of {subject} at a time (each offered where this "
                 "run filed a pair under it), never as their union")
    return (f"{prefix}{rule} — {tail}{_live_rule_ladder_note(sweep)}"
            f"{_live_sizing_note(sweep)}{_live_add_on_note(sweep)}")


def _tier_floors_bind(band: tuple[float, float]) -> bool:
    """
    Report whether a deadline-gap tier floor ever sits above a band's floor.

    With the tiers on, a time-series entry at this band must clear
    min_price_diff_for_gap(gap, spread_min=floor) = max(tier, floor); with
    them off, the floor alone. The two differ for some gap exactly when the
    floor is below a tier — on the shipped grid floors 0, 0.20 and 0.25 (both
    tiers at 0, the 0.30 tier alone at the other two). A band whose floor is
    at or above both tiers enters the same pairs on the same Mondays either
    way, so a tier-off sweep simulates only the binding bands again and the
    dashboard reuses the tier-on run for the rest (dashboard.py imports this
    one test to decide which bands may). Decided through
    min_price_diff_for_gap itself, over every gap _find_entry admits, so it
    cannot disagree with the rule it names.

    Args:
        band (tuple[float, float]): A (floor, ceiling) resolved by
            config.time_series_spread_band.

    Returns:
        bool: True when some gap in 0..MAX_DEADLINE_GAP_DAYS has a tier above
            the floor.
    """
    floor = band[0]
    # The one threshold helper, asked both ways at every admissible gap: the
    # tiers bind at this band exactly where the two answers differ
    return any(min_price_diff_for_gap(gap, spread_min=floor)
               != min_price_diff_for_gap(gap, spread_min=floor, tier_floors=False)
               for gap in range(MAX_DEADLINE_GAP_DAYS + 1))


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


def _population_subsets(entries: list[dict]) -> list[tuple[str, list[dict]]]:
    """
    Split one band's entries into the band sweep's standalone time-series populations.

    The ONE definition of those subsets, read by both _sweep_from_candidates'
    eager loops (through _band_populations, tier-on and tier-off) and
    CapSweep.cell, so a size-cap cell simulates exactly the
    populations the eager scenarios did. Split on _is_ladder_pair — the rule
    each trade's same_event_ladder label follows — so a trade and the
    population it was simulated in always agree. "time_series" is ladders and
    cross-event together, same-title excluded: the population the band and k
    act on alone.

    Args:
        entries (list[dict]): One band's entries (time-series then
            same-title), as entries_by_band holds them.

    Returns:
        list[tuple[str, list[dict]]]: ("time_series", ...), ("ladder", ...),
            ("cross", ...), in that order, each subset in entry order; any
            may be empty — a caller skips an empty one rather than simulating
            an empty scenario.
    """
    ladder_flags = [_is_ladder_pair(rec["pair_type"], rec["entry"]["mA"], rec["entry"]["mB"])
                    for rec in entries]
    return [
        ("time_series", [rec for rec in entries if rec["pair_type"] == "time_series"]),
        ("ladder", [rec for rec, is_ladder in zip(entries, ladder_flags, strict=True)
                    if is_ladder]),
        ("cross", [rec for rec, is_ladder in zip(entries, ladder_flags, strict=True)
                   if rec["pair_type"] == "time_series" and not is_ladder]),
    ]


def _entries_for_band(
    candidates: _Candidates,
    spread_band: tuple[float, float] | None = None,
    *,
    pair_types: tuple[str, ...] = _PAIR_TYPES,
    tier_floors: bool = True,
    _pairs: list | None = None,
) -> list[dict]:
    """
    Locate each candidate pair's tradeable Mondays under one spread band.

    The Pass-1a sweep: one _find_entry() call per pair in candidates.all_pairs
    (or in _pairs, when given) whose type is in pair_types, in scan order.
    _find_entry applies price,
    deadline and band thresholds only — it holds no probability model — so
    the result is identical at every interval discount; only the band (and
    the backtest-only tier_floors switch) can change it, and only for
    time-series pairs (a same-title pair reads neither). A caller sweeping
    many bands can therefore compute the
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
            config.BACKTEST_DEFAULT_SPREAD_BAND there — (0.0, 1.0), no band.
        pair_types (tuple[str, ...]): Which pair types to scan — any subset
            of ("time_series", "same_title"). Keyword-only. Defaults to both.
        tier_floors (bool): Keyword-only, BACKTEST-only; handed to every
            _find_entry() call. False enters time-series pairs at the band
            floor alone, the deadline-gap tier not applied (the tier-off
            family of _sweep_from_candidates); True (the default) is the live
            rule. A same-title pair never reads it.
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

        # Find every Monday where this pair was tradeable at the threshold
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
            # Backtest-only: False drops the deadline-gap tier for this pass
            tier_floors=tier_floors,
        )
        if entry is None:
            continue

        # Carry the group identity alongside the entry: _simulate_at_discount
        # keeps one same-title pair per group, and reads a time-series pair's
        # group key as its markets' question. mA/mB ride inside the entry.
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
    *,
    depth_model: DepthModel | None = None,
) -> tuple[list[dict] | None, OutcomeLabelCoverage | None]:
    """
    Run the half of the backtest that does not depend on the interval discount.

    It is _prepare_candidates() plus one _entries_for_band() pass at the
    DEFAULT band, config's BACKTEST_DEFAULT_SPREAD_BAND (0.0, 1.0), i.e. no
    band. None of it uses a probability model, so _simulate_at_discount()
    can be re-run at many discounts k over this one expensive pass.
    run_backtest() is its one production caller (run_backtest_sweep() runs
    one entry pass per band itself). Output pinned by
    tests/test_backtester.py::TestPrepareEntriesGolden.

    Args:
        hist_client (Any): Signed client for the historical archive/live endpoints.
        live_client: Client passed through to fetch_all_settled_markets.
        start_date (date): Earliest settlement date to include.
        use_cache (bool): Whether to reuse the disk-cached assembled market list.
        max_horizon_days (int | None): Optional opt-in bet-horizon cap mirroring
            scanner.filter_markets_within_horizon on the live path, but relative
            to each simulated checkpoint rather than real-world now: at a given
            checkpoint, a pair can only enter if the later-closing leg
            closes within max_horizon_days of THAT checkpoint. None applies no
            cap. Passed straight through to _find_entry() for each pair.
        same_event_ladders (bool | None): Whether two dated cumulative rungs
            of ONE event may pair. None (the default) resolves this
            module's TIME_SERIES_SAME_EVENT_LADDERS (bound from config at
            import) at call time; patching config itself is a silent no-op —
            see _extract_pairs' own entry. Handed
            verbatim to _prepare_candidates(), which gives it to BOTH
            _extract_pairs() calls and carries it, unresolved, to every
            _find_entry() call of the entry pass — load-bearing: the two must
            agree, or a pair this function proposes is replayed under the
            other rule's ordering.
        depth_model (DepthModel | None): Keyword-only. The depth model the
            entries' quotes carry, so their trades walk synthetic books; None
            (default) fills every trade at the top of the book.

    Returns:
        tuple[list[dict] | None, OutcomeLabelCoverage | None]: The prepared
            entries and this run's outcome-label census.

            Element 0 is one record per pair that produced an entry, in scan
            order (time-series pairs first, then same-title), each shaped
            {"pair_type": str, "canon": str, "group_key": object, "entry": dict}
            where "entry" is _find_entry()'s return dict (which already carries
            the possibly-swapped mA/mB). An empty list means no pair was ever
            tradeable. It is None — the codebase's
            return-None-on-validation-failure convention — when the
            feasibility pre-check fails, a "no simulation is possible in this
            window at all" signal distinct from "nothing entered". The
            sentinel lives on element 0: a caller that forgets to unpack
            holds a 2-tuple, which is never None, so its
            `if raw_entries is None` guard would silently go false.

            Element 1 is the OutcomeLabelCoverage the census measured over the
            eligible-market corpus — carried out so the dashboard can render
            the same caveat the log warns about — and is None on
            exactly the feasibility-short-circuit path, where the fetch never
            ran and there was no corpus to census. That is distinct from a
            censused corpus of zero records, which carries total=0.

    Raises:
        ValueError: Before any fetch, from _prepare_candidates(), when
            SCHEDULED_RUN cannot place the entry checkpoints (a configuration
            error).
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

    # ── Pass 1a: locate each pair's tradeable Mondays (k-independent) ──
    # _find_entry applies price and deadline thresholds only — it holds no
    # probability model — so this sweep yields identical entries at every
    # interval discount and is run exactly once, ahead of any sizing. At the
    # default spread band (spread_band=None, which _find_entry resolves to
    # config.BACKTEST_DEFAULT_SPREAD_BAND, i.e. no band), with the tier floors
    # on, it is the tier rule alone.
    raw_entries = _entries_for_band(candidates, spread_band=None)
    # The prices that value each entry's trades at market while open, taken
    # from the candles before they are released with this function's locals,
    # with the depth model their synthetic books are built from
    _attach_leg_quotes(raw_entries, candidates.candles_by_ticker, candidates.start_date,
                       depth_model)

    logging.info("Prepared %d candidate entries for sizing", len(raw_entries))
    _log_qualifying_mondays(raw_entries, None)
    return raw_entries, candidates.label_coverage


# Why _size_trade refused a candidate whose walked book left no size worth buying
_NO_SIZE_FITS = "no size fits"
# Why _size_trade skipped a walked candidate the cash left cannot buy one
# contract pair of (the book did not cause it, so it is counted apart)
_NO_CASH = "no cash for one contract pair"


def _cents(dollars: float) -> int:
    """
    Dollars as whole cents, rounded down, as the live sizer takes them.

    Rounded to 6 decimals first (config.leg_cash_cents' idiom), so float
    noise such as 0.29 * 100 = 28.999999999999996 never costs a cent; a real
    fraction of a cent is dropped. The same rounding reads an amount a hair
    under a whole cent (0.09999999999999998) as that cent.

    Args:
        dollars (float): An amount in dollars, at least 0 up to float noise
            (a hair below 0 reads as 0).

    Returns:
        int: Whole cents.
    """
    return math.floor(round(dollars * 100, 6))


def _debug_log(level: int, msg: str, *args: Any) -> None:
    """
    Log one of the live enrichment's lines at DEBUG, whatever its own level.

    The backtest walks many books, so scanner._enrich_pair's lines are kept
    out of the INFO log; level is the line's own level, unused.

    Args:
        level (int): The level the live run logs the line at.
        msg (str): The message format.
        *args (Any): Its arguments.
    """
    logging.debug(msg, *args)


def _candidate_pair(c: dict, held: HeldPair | None, markets: dict) -> CandidatePair:
    """
    The live sizer's CandidatePair for one Pass 2 candidate, at that Monday's quotes.

    Each market is built from its cached record by the live parser
    (scanner._market_from_dict), once per record per simulation (`markets`),
    so the live code reads each market's own close time and price grid. A
    time-series pair carries the deadline gap _find_entry tiered it on as its
    stated gap (scanner.pair_gap_days reads it), so the walk applies the same
    tier, and its mid spread at that Monday's four quotes
    (config.time_series_mid_spread), the input the live sizer's forecast
    reads, as the Kelly gate does, never at the market records' own quotes,
    which are their last ones, from when they closed. The live enrichment
    writes the mid spread again from a walked book, whose top is those
    quotes. The candle fetch clamps every NO ask into 0.01-0.99, so a YES bid
    never reads below 0.01 here. Live enrichment reads an earlier market's
    YES bid as it is, a missing one as 0, so where that bid is under a cent
    or missing, the mid spread here can sit up to 0.005 below what live
    enrichment would read off the same book, and the chance of profit up to
    k x 0.005 above it.

    Args:
        c (dict): A Pass 2 candidate.
        held (HeldPair | None): What an add-on adds to; None for a new pair.
        markets (dict): id(market record) -> ApiMarket, the simulation's memo.

    Returns:
        CandidatePair: Tradeable, priced at the quotes, not yet walked.
    """
    built = []
    for m in (c["mA"], c["mB"]):
        market = markets.get(id(m))
        if market is None:
            market = markets[id(m)] = _market_from_dict(m, m.get("event_title") or "")
        built.append(market)
    gap = c["gap_days"]
    stated = (gap if c["pair_type"] == "time_series" and type(gap) is int else None)
    # The mid spread at that Monday's quotes, which the sizer's forecast
    # reads: the one definition live enrichment also writes it with; None for
    # a same-title pair
    mid = (time_series_mid_spread(c["pA"], c["nA"], c["pB"], c["nB"])
           if c["pair_type"] == "time_series" else None)
    return CandidatePair(
        market_a=built[0], market_b=built[1], pA=c["pA"], pB=c["pB"], nA=c["nA"],
        nB=c["nB"], tradeable=True, canonical_title=str(c["canon"]),
        pair_type=c["pair_type"], stated_gap_days=stated, held=held, mid_spread=mid)


def _books_for(c: dict, d: date) -> tuple[dict, dict] | None:
    """
    Both markets' synthetic books at the candidate's checkpoint, anchored at its quotes.

    Market A's book starts at its YES ask pA and YES bid 1 - nA, market B's at
    pB and 1 - nB, so the top of each book is the candle's own prices.

    Args:
        c (dict): A Pass 2 candidate.
        d (date): Its checkpoint date.

    Returns:
        tuple[dict, dict] | None: (market A's book, market B's), or None when
            either cannot be built (no quotes, no depth model, or no volume
            data that Monday).
    """
    marks = c["marks"]
    if marks is None:
        return None
    book_a = marks[0].book_at(d, c["pA"], 1.0 - c["nA"])
    book_b = marks[1].book_at(d, c["pB"], 1.0 - c["nB"])
    if book_a is None or book_b is None:
        return None
    return book_a, book_b


def _size_trade(c: dict, d: date, checkpoint_value: float, cash: float,
                settings: LiveSettings, held: HeldPair | None,
                markets: dict) -> tuple[TradeSpec | None, str | None]:
    """
    Size one Pass 2 candidate as the live bot would.

    With a synthetic book for both markets (_books_for), the live enrichment
    (scanner._enrich_pair) prices the pair off it; then the live sizer
    (strategy.compute_trade) picks the count, on the portfolio value and the
    cash in whole cents, so a trade bets its Kelly share of the value but
    never spends more than the cash. With no book it sizes at the candle
    prices, the top of the book.

    Args:
        c (dict): A Pass 2 candidate.
        d (date): Its checkpoint date.
        checkpoint_value (float): The portfolio value at the checkpoint, in dollars.
        cash (float): The cash left, in dollars.
        settings (LiveSettings): The simulation's settings.
        held (HeldPair | None): What an add-on adds to; None for a new pair.
        markets (dict): The simulation's market memo (_candidate_pair).

    Returns:
        tuple[TradeSpec | None, str | None]: (the spec, None); (None, why)
            when a walked book refused it (an ENRICH_* or SPREAD_* code, or
            "no size fits"), or _NO_CASH when the cash left, not the book,
            cannot buy one contract pair; (None, None) when the top of the
            book had no size worth buying.
    """
    pair = _candidate_pair(c, held, markets)
    value_cents, cash_cents = _cents(checkpoint_value), _cents(cash)
    books = _books_for(c, d)
    if books is not None:
        # The live enrichment's own step, its lines at DEBUG
        pair, refusal = _enrich_pair(pair, *books, value_cents, settings=settings,
                                     cash_cents=cash_cents, log=_debug_log)
        if not pair.tradeable:
            # The cash left, not the book, cannot buy one contract pair (the
            # test behind enrichment's "cash binds" note): counted apart from
            # the book's refusals
            if refusal == ENRICH_UNAFFORDABLE and _cash_binds(
                    value_cents, max_kelly_fraction(pair.pair_type, settings), cash_cents):
                return None, _NO_CASH
            return None, refusal or ENRICH_UNPROFITABLE
    # The live sizer, quiet: its "Trade computed" lines at DEBUG
    spec = compute_trade(pair, value_cents, settings=settings, cash_cents=cash_cents, quiet=True)
    if spec is None:
        return None, (_NO_SIZE_FITS if books is not None else None)
    return spec, None


def _held_pair(c: dict, stake: float, trades: list[BacktestTrade],
               lone_ticker: str | None) -> HeldPair:
    """
    What an add-on adds to, as the live sizer reads it.

    compute_trade reads only its stake (stake_dollars), here the backtest's
    own: the held pair's open trades at market plus the fees paid for them,
    or a lone leg's alone.

    Args:
        c (dict): The add-on's candidate.
        stake (float): The held stake, in dollars.
        trades (list[BacktestTrade]): The held trades.
        lone_ticker (str | None): The held market of a lone leg; None for an exact pair.

    Returns:
        HeldPair: Its sides (one for a lone leg), contracts held, what they
            cost with their fees, and the stake.
    """
    sides = dict(zip((c["mA"]["ticker"], c["mB"]["ticker"]), leg_sides(c["pair_type"]),
                     strict=True))
    if lone_ticker is None:
        held = sorted(sides.items())
        cost = sum(t.total_cost + t.fees for t in trades)
    else:
        held = [(lone_ticker, sides[lone_ticker])]
        # The held leg's own contracts and fee, in each trade
        cost = 0.0
        for t in trades:
            price = _paid_prices(t)[0 if t.ticker_a == lone_ticker else 1]
            cost += t.n * price + fee_leg_exact(t.n, price)
    return HeldPair(sides=tuple(held), count=float(sum(t.n for t in trades)),
                    cost_dollars=cost, value_dollars=stake, fees_dollars=0.0)


def _lone_leg_records(open_legs: dict, active_tickers: set, market_a: dict,
                      market_b: dict, day: date, pair_type: str) -> list[dict] | None:
    """
    Return the open leg records a new pair would add to as a lone leg, or None.

    The backtest's twin of live's lone held leg (scanner.held_pairs): exactly
    one of the pair's two markets is held, every open trade holding it has
    had its other market pay out by `day`, and each holds the side this pair
    buys there. The other market must be held by no open trade. Anything
    else is no lone leg (None), and the pair is refused as before.

    Args:
        open_legs (dict): ticker -> open leg records (see _simulate_at_discount).
        active_tickers (set): Tickers held by an open trade.
        market_a (dict): The pair's market A.
        market_b (dict): Its market B.
        day (date): The checkpoint date.
        pair_type (str): The pair's type; scanner.leg_sides says which side
            each leg buys.

    Returns:
        list[dict] | None: The held market's open leg records, or None.
    """
    tickers = (market_a["ticker"], market_b["ticker"])
    held = [index for index, ticker in enumerate(tickers) if ticker in active_tickers]
    # Exactly one held market, which only open legs hold
    if len(held) != 1 or tickers[held[0]] not in open_legs:
        return None
    index = held[0]
    legs = open_legs[tickers[index]]
    side = leg_sides(pair_type)[index]
    # Every trade on it has had its other market pay out, and holds the side bought here
    if all(leg["partner_paid_out"] <= day < leg["paid_out"] and leg["side"] == side
           for leg in legs):
        return legs
    return None


def _simulate_at_discount(
    raw_entries: list[dict],
    start_date: date,
    initial_balance: float,
    k: float | None = None,
    spread_band: tuple[float, float] | None = None,
    population: str = "all",
    *,
    tier_floors: bool = True,
    size_cap: float | None = None,
    quiet: bool = False,
    end_date: date | None = None,
    add_to_held: bool = False,
    sell_at: float | None = None,
    sell_min_days: int | None = None,
) -> SweepPoint:
    """
    Choose, size and settle trades from prepared entries at one interval
    discount k (the share of the market's implied in-between chance the model
    believes), and build the equity curve.

    Scores each entry's Mondays with the Kelly rule (the formula that sets how
    much to bet) at the top of the book, a time-series pair's chance of paying
    read from that Monday's mid spread (config.time_series_mid_spread), as
    live sizing reads it, and a Monday whose quotes are crossed on either
    market (a YES ask below its own YES bid) skipped and counted. It drops
    pairs with no usable pay-out, then walks the Mondays in date order. Each
    trade is sized by the live
    code (_size_trade): with a depth model and volume data that Monday, the
    live enrichment walks a synthetic order book and the live sizer
    (strategy.compute_trade) picks the count; otherwise the sizer fills at
    the candle prices. Either way a trade bets a share of that Monday's
    opening portfolio value (cash plus open trades at market, _open_value)
    but never spends more than the cash left, so a trade the cash cannot
    fully buy is shrunk to fit. One config.LiveSettings per simulation, built
    from the arguments, carries k, the caps, the band and the tier setting
    to the live code. A time-series
    pair that cannot be taken is tried again on its next passing Monday; a
    same-title pair is not, and only the best one per title group is kept. At
    most one time-series pair is open per ladder (one question at several
    deadlines). The Kelly check runs before the pay-out checks, so keep that
    order: the count of impossible pay-outs covers only pairs that passed it.

    spread_band and tier_floors must be the ones the entries were found
    under: a walked book applies the live spread rule with them. population
    only labels the log lines and the returned point. size_cap changes trade
    sizes, and so which trades find cash, but never which Mondays pass the
    Kelly check or peak_kelly_fraction (CapSweep relies on this); with a
    walked book it also bounds the live search, which cap_free_from accounts
    for.

    add_to_held lets the walk add to what it still holds, as live does, each
    add-on a new trade (BacktestTrade.add_on):
      * a pair with an open trade, neither market paid out, may trade again
        the same way round;
      * once one market of an open trade has paid out, a new pair may add to
        the other (a lone leg, _lone_leg_records) on its held side, beside a
        market no open trade holds;
      * a same-title pair also keeps its later passing Mondays with the same
        legs the same way round, which can only add to its open trade.
    An add-on is refused when its markets share no ladder label or another
    open trade holds one, and sized by config.held_pair_fraction on what is
    held (open trades at market plus their fees; a lone leg alone,
    _open_leg_stake), so it never stakes more than a new pair would, and is
    skipped when that is already a full share. The cap reaches it only
    through min(pair cap, f*) (and, with a walked book, the live search's
    bound), so cap_free_from still marks the cap above which the walk is
    the same. Off by default: no add-on state is kept and no add-on line is
    logged.

    sell_at sells a whole position before it pays out (None, the default,
    never sells, and the walk is then unchanged):
      * a position is an open trade with every open trade joined to it by a
        market (_positions): a pair with everything added to it;
      * at every entry checkpoint from the first candidate's to the last
        pay-out — including those with no candidate (_sale_stream) — after
        that day's pay-outs and before its valuation, a position sells when
        its realized profit has stayed at or above sell_at of its potential
        profit for TAKE_PROFIT_HOLD_DAYS days in a row: at the checkpoint and
        at a daily check on each day before it, 24 hours apart
        (_hold_readings, _reached_every_day). Realized profit is what
        selling returns at that check (_position_sale_value: each market's
        contracts sold down its modeled bid ladder from the bid of the side
        held, or at that bid with no ladder, less the taker fee on the sale;
        a leg whose market has paid out, at its payout) less its cost
        (contracts plus entry fees); potential profit is its contract pairs
        at CONTRACT_PAYOUT_DOLLARS less that cost;
      * a position with no quotes, or with a market still to pay out and no
        fresh bid at the checkpoint (or no quote in the 24 hours before an
        earlier check) or too few contracts on its ladder at some check, is
        not sold at that checkpoint;
      * each of its trades exits at the sale (_sold_copy), its markets and
        ladders are freed, and its pair may be bought again at a later
        checkpoint, never at the sale's (nor may a pair touching its markets
        be bought there);
      * sell_min_days (None, the default, sets no minimum) holds back the
        sale of a position with fewer than that many days left before it
        matures — the date its last market stops trading, the latest close
        date among its trades' markets, counted in calendar days from the
        checkpoint's date (_days_left). Such a position is not even valued
        there (_far_enough is asked first). Its days left only shrink at
        later checkpoints, so it is held until it pays out — unless, with
        add_to_held, an add-on brings in a market that closes later.
    Sales come before every purchase at a checkpoint: the walk pays out the
    day's settlements (release), then sells (sell), then values the
    portfolio, then takes the day's candidates, new pairs and add-ons alike.
    So a sale's proceeds are cash for that checkpoint's trades, and every
    trade there is sized on the portfolio value after the sales.
    A checkpoint with no candidate changes nothing unless a position sells
    there, so at a level no position reaches the walk is exactly the one
    that never sells (_sale_reach). The cap still reaches the walk
    only through the sizes, so CapSweep's reuse above cap_free_from holds
    with selling on. With add_to_held as well, a
    position not sold may still be added to; one sold at a checkpoint is not.
    Every position sold is recorded as a SaleCheck on the returned point's
    sales.

    Args:
        raw_entries (list[dict]): Prepared entries, one record per pair.
        start_date (date): First trading date of the window.
        initial_balance (float): Starting cash in dollars.
        k (float | None): The interval discount, in (0, 1]; None means config.py's.
        spread_band (tuple[float, float] | None): The band the entries were found under.
        population (str): Which entries these are (e.g. "all", "time_series"); a label.
        tier_floors (bool): Keyword-only. Whether the entries were found with the tier floors.
        size_cap (float | None): Keyword-only. Per-trade cap in (0, 1] on the 5% grid;
            None for BUDGET_FRACTION.
        quiet (bool): Keyword-only. True logs this run's lines at DEBUG.
        end_date (date | None): Keyword-only. Last day of the equity curve; None for today (UTC).
        add_to_held (bool): Keyword-only. Whether a pair still held may trade again (see
            above); True ends the completion line's label with ", adding to held pairs".
        sell_at (float | None): Keyword-only. The share of potential profit, in (0, 1],
            at which a position is sold (see above); None never sells. Set, it ends the
            completion line's label with ", selling at <percent>% of potential profit".
        sell_min_days (int | None): Keyword-only. The fewest days a position must have
            left before its last market stops trading to be sold (see above): a whole
            number of at least 1, set only with sell_at; None sets no minimum. Set, it
            adds ", at least <N> day(s) before maturity" to the label.

    Returns:
        SweepPoint: The trades and daily equity curve, stamped with the resolved
            k and size cap, the band, population, tier setting, the largest
            Kelly fraction seen, whether it could add to held pairs, the
            level it sold at and its minimum of days, the cap it is the same
            at and above (cap_free_from), and, when it could sell, a
            SaleCheck per position sold (sales).

    Raises:
        ValueError: For an unknown population, a bad spread band, a bad size
            cap, a bad sell_at, a sell_min_days without sell_at or not a
            whole number of at least 1, a k outside (0, 1], or, with
            sell_at, a bad TAKE_PROFIT_HOLD_DAYS.
        TypeError: For a spread band that is not a pair of numbers.
    """
    if population not in _SIMULATION_LABELS:
        raise ValueError(
            f"population must be one of {_SIMULATION_LABELS}, got {population!r}"
        )
    # The band the entries were found under; it goes into the live spread
    # rule (the settings below). config.time_series_spread_band owns the
    # default and the validation; resolving here (once, before the loop)
    # also means a bad band fails before any work rather than after it.
    band_lo, band_hi = time_series_spread_band(spread_band)
    # The discount actually in force, recorded on the result so no caller has
    # to re-derive it from the None sentinel.
    effective_k = TIME_SERIES_INTERVAL_PROB_DISCOUNT if k is None else k
    # The per-trade cap in force — config.BUDGET_FRACTION unless a size-cap
    # sweep asks for another — resolved and validated before any entry.
    cap = _resolve_size_cap(size_cap)
    # ... and the extra same-title cap, held to the same rule
    st_cap = _resolve_same_title_size_cap()
    # The share of potential profit that sells a position (None: never sell),
    # and how many days in a row a position must stay at it (read only when
    # selling, so a run that never sells never reads it)
    sell_level = _resolve_sell_at(sell_at)
    hold_days = None if sell_level is None else _resolve_hold_days()
    # ...and the fewest days before maturity a sale needs (None: no minimum)
    min_days = _resolve_sell_min_days(sell_min_days, sell_level)
    # The live code's settings for this simulation: its k and caps, and the
    # band and tier setting the entries were found under (validated here,
    # before any entry is scored)
    settings = LiveSettings(
        tier_floors=tier_floors is not False, spread_band=(band_lo, band_hi),
        interval_discount=effective_k, size_cap=cap, same_title_size_cap=st_cap,
        categories=None, tags=None, add_to_held_pairs=bool(add_to_held))

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
    # WARNING) — do not reorder the two. Counted once per pair.
    premise_violations = 0
    # Time-series Mondays the Kelly gate skipped because a market's quote was
    # crossed (its YES ask below its own YES bid), once per pair per Monday;
    # reported once after the walk
    crossed_mondays = 0
    # SweepPoint.peak_kelly_fraction: the largest uncapped fraction over
    # every Monday a pair may be traded on
    peak_kelly = 0.0
    # The pair types with a candidate whose Monday can walk a synthetic book,
    # read off the candidates (never the trades the cap lets a walk reach)
    walk_types: set[str] = set()
    # Same-title markets' ladder labels, worked out once per market
    same_title_ladders: dict[int, frozenset] = {}

    for pair_id, rec in enumerate(raw_entries):
        # Group identity and the _find_entry result, exactly as recorded by
        # _entries_for_band (inside _prepare_entries, or once per band in a
        # sweep) — mA/mB inside the entry may have been swapped there to
        # canonicalize which leg is A.
        pair_type = rec["pair_type"]
        canon     = rec["canon"]
        group_key = rec["group_key"]
        entry     = rec["entry"]

        # The pair's qualifying Mondays with a positive Kelly fraction at this
        # k, in date order: all of them for a time-series pair, the first for
        # a same-title pair (all of them too when adding to held pairs).
        passing = []
        for monday in _entry_mondays(entry):
            # Unpack this Monday — mA/mB may have been swapped inside _find_entry to canonicalize
            mA = monday["mA"]
            mB = monday["mB"]
            pA, pB, nA, nB = monday["pA"], monday["pB"], monday["nA"], monday["nB"]
            entry_date = monday["entry_date"]

            # The two legs' top-of-the-book quotes — (nA, pB) for same_title,
            # (pA, nB) for time_series — the backtester's mirror of
            # scanner.leg_prices; Pass 2 finds the prices actually paid
            price_a, price_b = _leg_prices_for(pair_type, pA, nA, pB, nB)

            # ── Kelly fraction (sizing happens in Pass 2 against the checkpoint) ──
            # Compute the net spread on the leg prices after the continuous fee approximation
            fee_approx = fee_per_pair_approx(price_a, price_b)
            net_spread = (1.0 - price_a - price_b) - fee_approx
            # REPORTED/RANKED return on the contracts' cost (feeds
            # entry_monthly_ratio, the ranking key — see the note above
            # expected_days) — the mirror of strategy.TradeSpec.profit_ratio,
            # fee-less denominator and all.
            profit_ratio_entry = net_spread / (price_a + price_b) if net_spread > 0 else 0.0
            # Kelly's "b": the SAME numerator over the dollars actually at risk,
            # which include the fee — a losing pair loses cost + fees, not cost
            # (DR-62). A DIFFERENT quantity from profit_ratio_entry above; mirrors
            # strategy._evaluate_size's kelly_b exactly, so live and backtest admit
            # the same pairs. Do not collapse the two back together.
            kelly_b_entry = (net_spread / (price_a + price_b + fee_approx)
                             if net_spread > 0 else 0.0)

            # Probability model. time_series: the discounted market-implied
            # in-between mass at the midpoints, 1 - k * the mid spread, from
            # config.time_series_profit_prob — the single definition
            # strategy._kelly_p_at and dashboard._kelly_fraction also call, so
            # the three can never drift (called directly by name here; a test
            # pins the two-link chain run_backtest -> _simulate_at_discount ->
            # the helper). k is this function's override, and None — what
            # run_backtest passes — is the sentinel the helper resolves to
            # config.TIME_SERIES_INTERVAL_PROB_DISCOUNT, config.py's k (live
            # sizing prices through the same helper at the saved live
            # defaults' k). same_title: the fixed co-resolution prior.
            if pair_type == "time_series":
                # A quote whose YES ask sits below its own YES bid is a crossed
                # book: live enrichment drops it, and its midpoint could let
                # Kelly pass 1 - k (config.max_kelly_fraction relies on this)
                if pA + nA < 1.0 - PRICE_EPSILON or pB + nB < 1.0 - PRICE_EPSILON:
                    crossed_mondays += 1
                    continue
                # The market's in-between chance at the midpoints, the input
                # live sizing reads (config.time_series_mid_spread, DR-78). On
                # these uncrossed quotes a Monday that passes the fee check
                # has a mid spread above its fee less PRICE_EPSILON, so above
                # zero: the model's zero clamp, which would read a spread at
                # or below zero as riskless, never decides a Monday here
                spread = time_series_mid_spread(pA, nA, pB, nB)
                p = time_series_profit_prob(spread, k=k)
            else:
                p = SAME_TITLE_CO_RESOLVE_PROB
            q = 1.0 - p

            # Kelly formula: f* = p - q/b; non-positive means no positive expected
            # value once the fee is counted on the losing side too (DR-62)
            kelly_f = (p - q / kelly_b_entry) if kelly_b_entry > 0 else -1.0
            if kelly_f > 0:
                passing.append((monday, kelly_f, profit_ratio_entry))
                if pair_type != "time_series" and not add_to_held:
                    # A same-title pair is tried on its first passing Monday only
                    break
        if not passing:
            # No qualifying Monday has positive expected value at this k
            continue
        if pair_type != "time_series" and len(passing) > 1:
            # Adding to held pairs: a same-title pair's later passing Mondays
            # can only add to its first one's trade, so only those with the
            # first's legs the same way round are kept (its pricier side is
            # decided again every Monday, and the other way round is the
            # opposite trade)
            legs = (passing[0][0]["mA"]["ticker"], passing[0][0]["mB"]["ticker"])
            passing = [passing[0]] + [row for row in passing[1:]
                                      if (row[0]["mA"]["ticker"], row[0]["mB"]["ticker"]) == legs]
        # Any passing Monday may be the one traded (or, for a same-title pair
        # adding to held pairs, added on)
        peak_kelly = max(peak_kelly, *(kelly_f for _m, kelly_f, _ratio in passing))
        # config.pair_size_cap: the one definition live sizing caps through
        size_cap_for_pair = pair_size_cap(pair_type, cap, st_cap)

        # Checked once per pair: a time-series pair's markets are the same on
        # every Monday, and a same-title pair's kept Mondays all have its
        # first one's legs
        mA, mB = passing[0][0]["mA"], passing[0][0]["mB"]

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

        # Use title > subtitle > ticker as the display label for each market
        title_a = mA.get("title") or mA.get("subtitle") or mA.get("ticker", "")
        title_b = mB.get("title") or mB.get("subtitle") or mB.get("ticker", "")

        # A time-series pair's group key is its markets' question; a
        # same-title pair's is not, so its questions are worked out here
        if pair_type == "time_series":
            ladders_a = _ladder_keys_dict(mA, group_key)
            ladders_b = _ladder_keys_dict(mB, group_key)
        else:
            for m in (mA, mB):
                if id(m) not in same_title_ladders:
                    same_title_ladders[id(m)] = _ladder_keys_dict(m)
            ladders_a, ladders_b = same_title_ladders[id(mA)], same_title_ladders[id(mB)]

        # Each leg's prices over time (the one reader of what _attach_leg_quotes
        # wrote), in this pair's A/B order; None values its trades at cost
        quotes = rec.get("leg_quotes")
        marks = None if quotes is None else (quotes[mA["ticker"]], quotes[mB["ticker"]])

        # One candidate per passing Monday, priced and ranked on that Monday
        for index, (monday, kelly_f, profit_ratio_entry) in enumerate(passing):
            entry_date = monday["entry_date"]
            holding_days = max(1, (exit_date - entry_date).days)

            # Ranking metric: the expected return at this Monday's prices per
            # 30 days to the later close. A paid-out market's close time is
            # when it actually closed, which can be earlier than scheduled, so
            # this can use something not known on that Monday.
            expected_days = max(1, (max(close_a_d, close_b_d) - entry_date).days)
            entry_monthly_ratio = profit_ratio_entry * 30.0 / expected_days

            # Whether this Monday can walk a synthetic book (both markets
            # have one there): decides cap_free_from below
            if (marks is not None and marks[0].has_book_at(entry_date)
                    and marks[1].has_book_at(entry_date)):
                walk_types.add(pair_type)

            candidates.append({
                "pair_type": pair_type,
                "canon": canon,
                "group_key": group_key,
                "mA": mA, "mB": mB,
                "pA": monday["pA"], "pB": monday["pB"],
                "nA": monday["nA"], "nB": monday["nB"],
                "entry_date": entry_date,
                "exit_date": exit_date,
                "outcome_a": outcome_a,
                "outcome_b": outcome_b,
                "kelly_f_capped": min(size_cap_for_pair, kelly_f),
                # A same-title pair's later Monday: it can only add to the
                # pair's open trade (never with add_to_held off: there is none)
                "add_on_only": pair_type != "time_series" and index > 0,
                "entry_monthly_ratio": entry_monthly_ratio,
                "holding_days": holding_days,
                "title_a": title_a,
                "title_b": title_b,
                # Per-leg dates for the dashboard's trade tables (reporting only)
                "close_date_a": close_a_d, "close_date_b": close_b_d,
                "settled_date_a": exit_date_a, "settled_date_b": exit_date_b,
                # Carried straight from _find_entry (None for same_title) so the
                # recorded trade reports the same gap the tier was chosen from
                "gap_days": entry["gap_days"],
                # So Pass 2 opens the pair at most once (and, adding to held
                # pairs, knows which open trade a later Monday adds to)
                "pair_id": pair_id,
                "ladder_keys_a": ladders_a, "ladder_keys_b": ladders_b,
                # How Pass 2 and the equity curve value the trade while open
                "marks": marks,
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
        # A tier-floors-off simulation (its entries were found with the
        # deadline-gap tiers not applied) appends
        # " [tier floors off]", so a count from that backtest-only family is
        # never read as one from the run's tier-on scenarios; the tier-on
        # text is unchanged.
        # The count is cap-independent (taken after the Kelly gate, before any
        # sizing), so a quiet size-cap re-run only repeats what the
        # primary-cap run already warned: it goes to DEBUG, text unchanged.
        logging.log(
            logging.DEBUG if quiet else logging.WARNING,
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
            "outcome-label coverage line)%s",
            premise_violations,
            " [tier floors off]" if tier_floors is False else "",
        )

    # The cap at and above which no trade here depends on the cap. A walked
    # book is also averaged only as deep as config.max_kelly_fraction allows,
    # which stops moving once the cap reaches the pair type's own ceiling
    # (its value with no per-trade cap)
    uncapped = replace(settings, size_cap=1.0)
    cap_free_from = max([peak_kelly, *(max_kelly_fraction(pair_type, uncapped)
                                       for pair_type in sorted(walk_types))])

    # Keep only the best same-title candidate per title group, as the live
    # finders keep one pair per group; this ranks by expected monthly return
    # (what Pass 2 orders on), the live finders by price gap. The key includes
    # the event title, so one option label in two events stays two groups.
    # The winner can be dated after a group-mate that passed the gate earlier,
    # which the live bot could not have known. Time-series candidates skip
    # this: the ladder rule in Pass 2 decides which rung trades. A same-title
    # pair's add-on-only Mondays never contest; the winner's are kept after
    # it, and a loser's are dropped with it.
    best_by_group: dict = {}
    for c in candidates:
        if c["pair_type"] == "time_series" or c["add_on_only"]:
            continue
        key = (c["pair_type"], c["group_key"])
        cur = best_by_group.get(key)
        if cur is None or c["entry_monthly_ratio"] > cur["entry_monthly_ratio"]:
            best_by_group[key] = c
    winners = {c["pair_id"] for c in best_by_group.values()}
    candidates = ([c for c in candidates if c["pair_type"] == "time_series"]
                  + list(best_by_group.values())
                  # Empty unless add_to_held is on (only then does a same-title
                  # pair keep a later Monday)
                  + [c for c in candidates if c["add_on_only"] and c["pair_id"] in winners])

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
    # Walk the candidates in date order with a running cash balance. Each date
    # is one checkpoint (the Monday-morning moment trades are entered, like one
    # live run). Each candidate is sized by the live code (_size_trade): a
    # share of the checkpoint's opening portfolio value (cash after that day's
    # pay-outs plus every open trade at market, _open_value), never more than
    # the cash left, walking a synthetic book when one can be built. A
    # candidate the cash cannot fully buy is shrunk to fit, and skipped if
    # not one contract pair fits or a win no longer pays after fees.
    #
    # Unlike a live run, an open trade is valued at the ask of the side each
    # leg holds (live reads Kalshi's own value of the positions), no extra
    # cash is held back for the orders' worst-case prices, and a shrunk trade
    # is not re-checked for a positive expected value.
    #
    # As live, a market is in the open trades of at most one pair (freed when
    # that pair pays out) and at most one time-series pair is open per ladder
    # (freed the day each market pays out). Nothing is sold before it pays
    # out unless sell_at is set (see the docstring). With add_to_held, a pair
    # still held may trade again as a new trade of its own, and a new pair may
    # add to a held leg whose partner has paid out (see the docstring).
    trades: list[BacktestTrade] = []
    active_tickers: set[str] = set()
    cash = initial_balance
    # Portfolio value at the current checkpoint; each trade that day bets a share of it
    checkpoint_date: date | None = None
    checkpoint_value = cash
    # (exit_date, settlement receipt, trade); the trade lets a sale take its
    # receipt back, and every list below carries it for the same reason
    pending_exits: list[tuple[date, float, BacktestTrade]] = []
    # Every open trade, counted in the portfolio value (at market) until it pays out
    open_trades: list[BacktestTrade] = []
    # (exit_date, ticker, trade) for every leg of a still-open trade — the
    # release ledger for active_tickers, kept alongside pending_exits so cash
    # and ticker availability are always freed on exactly the same day.
    active_until: list[tuple[date, str, BacktestTrade]] = []
    traded_pairs: set[int] = set()
    # Open markets per ladder label, and when each market's labels free up
    open_ladders: dict = {}
    ladders_until: list[tuple[date, frozenset, BacktestTrade]] = []
    ladder_refusals = 0
    # Adding to held pairs (filled only when add_to_held, so the off path
    # records nothing new): each open pair's trades with what each one paid
    # (its contracts plus fees), the day it pays out, and how many ladder
    # holds its own trades have taken
    open_pairs: dict[int, dict] = {}
    # ...and each open leg by its market: the trade, the side it holds, its
    # ladder labels, the day its market pays out and the day its partner's
    # does — what a later add-on to a lone leg (its partner paid out) reads
    open_legs: dict[str, list[dict]] = {}
    add_ons = add_on_cap_skips = add_on_ladder_skips = 0
    # Selling early (filled only when sell_at is set, so the off path records
    # nothing new): each trade's pair, the sold copy that replaces a sold
    # trade in the results, the markets sold at the current checkpoint (not
    # bought again there), every pair ever sold, and the counts it logs
    pair_of: dict[int, int] = {}
    sold_copies: dict[int, BacktestTrade] = {}
    sold_here: set[str] = set()
    sold_pairs: set[int] = set()
    positions_sold = bought_again = 0
    # One SaleCheck per position sold, in the order sold (SweepPoint.sales)
    sale_checks: list[SaleCheck] = []
    # The live code's market objects, built once per record (_candidate_pair),
    # trades filled at the top of the book, walked books' refusals by reason,
    # and walked candidates the cash left could not buy one contract pair of
    markets: dict[int, Any] = {}
    top_of_book = 0
    walk_refusals: Counter = Counter()
    cash_skips = 0

    def release(d: date) -> None:
        """
        Pay out every trade that paid out by `d`, and free its markets and ladders.

        Args:
            d (date): The checkpoint date.
        """
        nonlocal cash, pending_exits, open_trades, active_until, ladders_until
        # Release settlement receipts from trades that exited on or before this entry
        cash += sum(amt for ed, amt, _trade in pending_exits if ed <= d)
        pending_exits = [row for row in pending_exits if row[0] > d]
        # ...and stop counting them in the portfolio value
        open_trades = [t for t in open_trades if t.exit_date > d]
        # Release the tickers of those same settled trades — the position is
        # closed, so (as live) the ticker is no longer blocked. Set difference
        # is safe because the conflict filter below guarantees a ticker is in
        # the open trades of at most one pair, which share one exit date.
        # NOTE the `<= d` (not `< d`): a trade exiting ON date d frees its
        # tickers for a later candidate entering that same date d. This is a
        # deliberate symmetry with the cash rule directly above, which likewise
        # returns that trade's settlement receipt on d — both resources are
        # freed on exactly the same day, so a same-day re-entry is funded and
        # unblocked together rather than one without the other.
        active_tickers.difference_update(tk for ed, tk, _trade in active_until if ed <= d)
        active_until = [row for row in active_until if row[0] > d]
        # Free the ladders of markets paid out by this day
        if ladders_until:
            still_held = []
            for row in ladders_until:
                ed, keys, _trade = row
                if ed > d:
                    still_held.append(row)
                    continue
                for key in keys:
                    open_ladders[key] -= 1
                    if not open_ladders[key]:
                        del open_ladders[key]
            ladders_until = still_held
        if add_to_held:
            # Release the open-pair record of trades paid out by this day, on
            # the same schedule as their cash
            for pid in [pid for pid, rec in open_pairs.items() if rec["until"] <= d]:
                del open_pairs[pid]
            # ...and every leg whose own market has paid out
            for ticker in [t for t, legs in open_legs.items()
                           if any(leg["paid_out"] <= d for leg in legs)]:
                legs = [leg for leg in open_legs[ticker] if leg["paid_out"] > d]
                if legs:
                    open_legs[ticker] = legs
                else:
                    del open_legs[ticker]

    def sell(d: date) -> None:
        """
        Sell, whole, every open position that has stayed at sell_at of its potential profit for hold_days days.

        A position with fewer than min_days days left before it matures is
        skipped before it is valued (_far_enough). Each sold trade's proceeds
        come in now, its record is replaced by its sold copy (_sold_copy),
        its pay-out, markets and ladder labels are freed, and its pair may be
        bought again at a later checkpoint (never at this one: its markets go
        into sold_here). Each position sold adds a SaleCheck to sale_checks.

        Args:
            d (date): The checkpoint date, after its pay-outs.
        """
        nonlocal cash, pending_exits, open_trades, active_until, ladders_until
        nonlocal positions_sold
        sold_here.clear()
        sold_ids: set[int] = set()
        for position in _positions(open_trades):
            if not _far_enough(position, d, min_days):
                # Too near maturity: held to pay out, never valued
                continue
            # The checkpoint and the daily checks before it (_hold_readings)
            readings = _hold_readings(position, d, hold_days, level=sell_level)
            if readings is None or not _reached_every_day(sell_level, readings[1]):
                continue
            per_trade = readings[0][0]
            positions_sold += 1
            # What the sale had: its days left and its profit at every check
            sale_checks.append(SaleCheck(_days_left(position, d), tuple(readings[1])))
            for trade, sale in zip(position, per_trade, strict=True):
                # The sale's proceeds, trade by trade
                cash += sale[0]
                sold_copies[id(trade)] = _sold_copy(trade, d, sale)
                sold_ids.add(id(trade))
                sold_here.update((trade.ticker_a, trade.ticker_b))
                # Free to trade again, at a later checkpoint only
                traded_pairs.discard(pair_of[id(trade)])
                sold_pairs.add(pair_of[id(trade)])
        if not sold_ids:
            return
        # A sold trade pays out no more, and holds no market or ladder: a
        # position is all the trades sharing its markets, so no open trade
        # still holds any of them
        pending_exits = [row for row in pending_exits if id(row[2]) not in sold_ids]
        open_trades = [t for t in open_trades if id(t) not in sold_ids]
        active_tickers.difference_update(tk for _ed, tk, t in active_until if id(t) in sold_ids)
        active_until = [row for row in active_until if id(row[2]) not in sold_ids]
        still_held = []
        for row in ladders_until:
            if id(row[2]) not in sold_ids:
                still_held.append(row)
                continue
            for key in row[1]:
                open_ladders[key] -= 1
                if not open_ladders[key]:
                    del open_ladders[key]
        ladders_until = still_held
        if add_to_held:
            # Its pairs' records and its legs go too: nothing is held to add to
            for pid in [pid for pid, rec in open_pairs.items()
                        if any(id(t) in sold_ids for t, _paid in rec["trades"])]:
                del open_pairs[pid]
            for ticker in list(open_legs):
                legs = [leg for leg in open_legs[ticker] if id(leg["trade"]) not in sold_ids]
                if legs:
                    open_legs[ticker] = legs
                else:
                    del open_legs[ticker]

    # Without selling, the candidates alone; with it, every other checkpoint
    # too, up to the last pay-out (_sale_stream), so a position can be sold
    # on a Monday nothing is bought
    stream = (((c["entry_date"], c) for c in candidates) if sell_level is None
              else _sale_stream(candidates))
    for d, c in stream:
        if c is None:
            # A checkpoint with no candidate, visited only to sell. Nothing
            # changes here unless a position sells, so a run that never sells
            # makes exactly the moves of one that cannot (_sale_reach)
            open_now = [t for t in open_trades if t.exit_date > d]
            if not any(_position_sells(sell_level, position, d, hold_days, min_days)
                       for position in _positions(open_now)):
                continue
            release(d)
            sell(d)
            continue
        release(d)

        if d != checkpoint_date:
            if sell_level is not None:
                # Sales come after the day's pay-outs and before its valuation
                # and every candidate (new pairs and add-ons alike), so a
                # sale's cash funds this checkpoint's trades
                sell(d)
            # New checkpoint: value the portfolio as cash plus every open trade
            # at market (_open_value, the one valuation; a trade with no
            # quotes counts at its cost)
            checkpoint_date = d
            checkpoint_value = cash + sum(_open_value(t, d) for t in open_trades)

        held = None
        # Set instead of held for an add-on to a lone leg: that leg's open records
        lone_legs = None
        mA, mB = c["mA"], c["mB"]
        ladders_a, ladders_b = c["ladder_keys_a"], c["ladder_keys_b"]
        if sold_here and (mA["ticker"] in sold_here or mB["ticker"] in sold_here):
            # Sold at this checkpoint: not bought or added to again here
            continue
        if c["pair_id"] in traded_pairs:
            # Traded before: only an add-on to its still-open trade may follow,
            # and only while neither of its markets has paid out (live finds a
            # held pair only while it holds both)
            held = open_pairs.get(c["pair_id"]) if add_to_held else None
            if held is None or not (c["settled_date_a"] > d and c["settled_date_b"] > d):
                continue
        elif c["add_on_only"]:
            # A same-title pair's later Monday with nothing to add to
            continue
        elif add_to_held and c["settled_date_a"] > d and c["settled_date_b"] > d:
            # A new pair on one held market whose partner has paid out may add
            # to that lone leg (live, scanner.held_pairs finds it alone)
            lone_legs = _lone_leg_records(open_legs, active_tickers, mA, mB, d,
                                          c["pair_type"])

        # Skip if either ticker is still committed to a trade that hasn't
        # settled; an add-on's own trade holds its tickers, and the conflict
        # filter kept every other trade off them while it is open
        if held is None and lone_legs is None and (mA["ticker"] in active_tickers
                                                   or mB["ticker"] in active_tickers):
            continue

        if lone_legs is not None:
            # The lone leg's own trades still hold its ladder labels (their
            # other legs' labels were freed when those paid out); any more
            # holds on a label of either market belong to another open trade,
            # which refuses it, as live adds only beside no other held ladder
            own = Counter(key for leg in lone_legs for key in leg["ladders"])
            if not (ladders_a & ladders_b) or any(
                    open_ladders.get(key, 0) > own[key]
                    for key in (*ladders_a, *ladders_b)):
                add_on_ladder_skips += 1
                continue
        elif held is not None:
            # As live: its two markets must share a ladder, and a hold by any
            # other open trade (beyond its own, held["ladders"]) refuses it,
            # counted on its own line
            if not (ladders_a & ladders_b) or any(
                    open_ladders.get(key, 0) > held["ladders"][key]
                    for key in (*ladders_a, *ladders_b)):
                add_on_ladder_skips += 1
                continue
        elif c["pair_type"] == "time_series" and any(
                key in open_ladders for key in (*ladders_a, *ladders_b)):
            # At most one open time-series pair per ladder
            ladder_refusals += 1
            continue

        fraction = c["kelly_f_capped"]
        # What an add-on adds to, as the live sizer reads it (None: a new pair)
        held_pair = None
        if held is not None or lone_legs is not None:
            if lone_legs is not None:
                # A lone leg's stake: that one leg, at market plus its fee, in
                # each of its open trades (_open_leg_stake); its partner's
                # payout is already cash
                stake = sum(_open_leg_stake(leg["trade"], leg["ticker"], d)
                            for leg in lone_legs)
            else:
                # The pair's stake: what each of its open trades paid
                # (contracts plus fees), moved by what its contracts have
                # gained or lost since (_open_value less their cost, zero for
                # a trade with no quotes) — their value at market plus the
                # fees paid for them
                stake = 0.0
                for held_trade, paid in held["trades"]:
                    stake += paid + (_open_value(held_trade, d) - held_trade.total_cost)
            # Kelly sizes the whole position: buy what the pair is missing of
            # its Kelly share of the portfolio value (config.held_pair_fraction,
            # the live sizer's own rule). Checked here, at the top of the book,
            # so a pair already at its share is counted on its own line
            fraction = held_pair_fraction(fraction, stake, checkpoint_value)
            if fraction <= 0:
                add_on_cap_skips += 1
                continue
            held_pair = (_held_pair(c, stake, [t for t, _paid in held["trades"]], None)
                         if lone_legs is None else
                         _held_pair(c, stake, [leg["trade"] for leg in lone_legs],
                                    lone_legs[0]["ticker"]))

        # Sized by the live code: a share of the portfolio value, never more
        # than the cash left, over a walked synthetic book when there is one
        spec, refusal = _size_trade(c, d, checkpoint_value, cash, settings, held_pair, markets)
        if spec is None:
            if refusal == _NO_CASH:
                cash_skips += 1
            elif refusal is not None:
                walk_refusals[refusal] += 1
            continue
        n = spec.x
        # The average price each leg pays, the fees there and the cash spent
        # (the spec's own figures); every dollar figure below reads these
        fill_a, fill_b = leg_prices(spec.pair)
        walked = len(spec.pair.depth_levels) > 0
        fee_a, fee_b = fee_leg_exact(n, fill_a), fee_leg_exact(n, fill_b)
        total_cost = spec.total_cost
        # Exact ceiling-rounded taker fees for both legs, charged at entry
        fees = fee_a + fee_b
        invested = spec.total_cost_with_fees
        # Win-scenario NET profit after exact fees (same_title: the floor every
        # co-resolution outcome clears; time_series: the profit of either win
        # cell); the sizer already refused a trade whose fees ate it
        expected_payoff = n * (1.0 - fill_a - fill_b) - fees
        if expected_payoff <= 0:
            continue
        # Safety check only: the sizer never spends more than the cash it was
        # handed, in whole cents (a float cash a hair below a cent reads as
        # that cent, as live's cash does)
        if invested > _cents(cash) / 100.0:
            continue

        # Realized P&L from the settlement outcomes — n per leg whose market
        # resolved to the side bought (sides depend on the pair type)
        receipt      = _settlement_receipt(n, c["outcome_a"], c["outcome_b"], c["pair_type"])
        profit       = receipt - total_cost - fees
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

        trade = BacktestTrade(
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
            # The capped Kelly fraction the sizer sized at; on an add-on, the
            # share of the portfolio value held_pair_fraction allowed
            kelly_fraction=spec.kelly_fraction,
            expected_payoff=expected_payoff,
            slippage=slippage,
            holding_days=c["holding_days"],
            balance_at_entry=checkpoint_value,
            deadline_gap_days=c["gap_days"],
            event_ticker=event_a,
            same_event_ladder=is_ladder,
            subtitle_a=mA.get("subtitle") or "",
            subtitle_b=mB.get("subtitle") or "",
            close_date_a=c["close_date_a"],
            close_date_b=c["close_date_b"],
            settled_date_a=c["settled_date_a"],
            settled_date_b=c["settled_date_b"],
            add_on=held is not None or lone_legs is not None,
            marks=c["marks"],
            fill_price_a=fill_a,
            fill_price_b=fill_b,
            book_walked=walked,
        )
        trades.append(trade)
        if not walked:
            top_of_book += 1

        # Cash out the door: contracts plus fees; the receipt comes back at exit
        cash -= invested
        pending_exits.append((c["exit_date"], receipt, trade))
        # Counted in the portfolio value (at market) until it pays out
        open_trades.append(trade)

        # Mark both tickers as active so no OVERLAPPING pair is added later;
        # the release ledger frees them again on this trade's exit date.
        active_tickers.add(mA["ticker"])
        active_tickers.add(mB["ticker"])
        active_until.append((c["exit_date"], mA["ticker"], trade))
        active_until.append((c["exit_date"], mB["ticker"], trade))

        if sell_level is not None:
            # Which pair the trade belongs to, for a sale to free; a new trade
            # (not an add-on) of a pair sold before is it bought again
            pair_of[id(trade)] = c["pair_id"]
            if not trade.add_on and c["pair_id"] in sold_pairs:
                bought_again += 1

        if add_to_held:
            if trade.add_on:
                add_ons += 1
            # Add this trade, with what it paid (its contracts plus fees), and
            # its ladder holds to its pair's record (a new record for a new
            # pair, an add-on to a lone leg included)
            if held is None:
                held = open_pairs[c["pair_id"]] = {"trades": [], "until": c["exit_date"],
                                                   "ladders": Counter()}
            held["trades"].append((trade, invested))
            held["ladders"].update((*ladders_a, *ladders_b))
            # ...and each leg to its market's record, for a later add-on once
            # the other leg has paid out
            side_a, side_b = leg_sides(c["pair_type"])
            for market, side, keys, paid_out, partner in (
                    (mA, side_a, ladders_a, c["settled_date_a"], c["settled_date_b"]),
                    (mB, side_b, ladders_b, c["settled_date_b"], c["settled_date_a"])):
                open_legs.setdefault(market["ticker"], []).append(
                    {"trade": trade, "ticker": market["ticker"], "side": side,
                     "ladders": keys, "paid_out": paid_out, "partner_paid_out": partner})

        # Each market holds its ladders until it pays out
        traded_pairs.add(c["pair_id"])
        for keys, paid_out in ((ladders_a, c["settled_date_a"]),
                               (ladders_b, c["settled_date_b"])):
            for key in keys:
                open_ladders[key] = open_ladders.get(key, 0) + 1
            ladders_until.append((paid_out, keys, trade))

    if sold_copies:
        # Each sold trade's record is its sold copy, in the order it was made
        trades = [sold_copies.get(id(t), t) for t in trades]

    # Named with the RESOLVED discount, the resolved band and the population.
    # A default run emits this line once per swept k, and a band sweep once
    # per (band, k, population) plus its split-half and ex-top runs, so
    # without all three a reader could not tell which simulation a trade
    # count belonged to (TS-21); every prefix up to the ':' is unique within
    # one sweep — k and band are printed through _exact_label, so an off-grid
    # value that rounds onto a grid member still prints distinctly.
    # effective_k and the resolved band, never the arguments, so the None
    # sentinels are never printed. A tier-off sweep's simulation carries
    # " with the tier floors off" after the band, so the two families never
    # share a prefix — no colon in it (the prefix ends at the first ':'). A
    # cap other than BUDGET_FRACTION is named after the population, BEFORE
    # the colon (", cap 35%" / ", no cap" — injective, so the prefix stays
    # unique across caps too); the default cap adds nothing, so every pre-cap
    # line is byte-identical and the population stays the last ", "-separated
    # field of a default-cap prefix that does not add to held pairs. A run
    # that adds to held pairs ends its prefix on ", adding to held pairs",
    # after the cap, so it never shares a prefix with one that does not. The
    # lazy size-cap runs are quiet (DEBUG).
    run_label = "k={}, band {}{}, {}{}{}{}".format(
        _exact_label(effective_k, ".3f"),
        _band_label((band_lo, band_hi)),
        " with the tier floors off" if tier_floors is False else "",
        population,
        "" if cap == BUDGET_FRACTION else f", {_cap_label(cap)}",
        ", adding to held pairs" if add_to_held else "",
        "" if sell_level is None else f", {_sale_label(sell_level, min_days)}",
    )
    # Once per pair per Monday the Kelly gate skipped for a crossed quote
    # (silent at zero)
    if crossed_mondays:
        logging.log(
            logging.DEBUG if quiet else logging.INFO,
            "Time-series Mondays skipped because a market's YES ask sat below its "
            "own YES bid (a crossed book) (%s): %d",
            run_label, crossed_mondays,
        )
    # Once per pair per Monday a busy ladder held it back (silent at zero)
    if ladder_refusals:
        logging.log(
            logging.DEBUG if quiet else logging.INFO,
            "Time-series pairs skipped on a Monday because we still held a trade on "
            "the same ladder (%s): %d",
            run_label, ladder_refusals,
        )
    # Adding to held pairs: what was added, and why an add was refused (each
    # silent at zero, and never logged with add_to_held off)
    if add_ons:
        logging.log(
            logging.DEBUG if quiet else logging.INFO,
            "Trades that added to a held pair (%s): %d",
            run_label, add_ons,
        )
    if add_on_cap_skips:
        logging.log(
            logging.DEBUG if quiet else logging.INFO,
            "Adds skipped because the held pair is already at its target size (%s): %d",
            run_label, add_on_cap_skips,
        )
    if add_on_ladder_skips:
        logging.log(
            logging.DEBUG if quiet else logging.INFO,
            "Adds skipped because another open trade is on the same ladder, or the "
            "two markets share no ladder (%s): %d",
            run_label, add_on_ladder_skips,
        )
    # Selling early: positions sold, and pairs bought again after one (each
    # silent at zero, and never logged with sell_at unset)
    if positions_sold:
        logging.log(
            logging.DEBUG if quiet else logging.INFO,
            "Positions sold before they paid out (%s): %d",
            run_label, positions_sold,
        )
    if bought_again:
        logging.log(
            logging.DEBUG if quiet else logging.INFO,
            "Pairs bought again after a sale (%s): %d",
            run_label, bought_again,
        )
    # How trades were sized: at the top of the book (no depth model, or no
    # volume data for a leg that Monday), walked books' refusals by reason,
    # and walked candidates skipped for want of cash (each silent at zero)
    if top_of_book:
        logging.log(
            logging.DEBUG if quiet else logging.INFO,
            "Trades sized at the top of the book (no depth data) (%s): %d",
            run_label, top_of_book,
        )
    for reason in sorted(walk_refusals):
        logging.log(
            logging.DEBUG if quiet else logging.INFO,
            "Trades refused when their book was walked (%s) (%s): %d",
            reason, run_label, walk_refusals[reason],
        )
    if cash_skips:
        logging.log(
            logging.DEBUG if quiet else logging.INFO,
            "Trades skipped with no cash left for one contract pair (%s): %d",
            run_label, cash_skips,
        )
    logging.log(
        logging.DEBUG if quiet else logging.INFO,
        "Backtest complete at %s: %d trades, %d profitable",
        run_label,
        len(trades),
        sum(1 for t in trades if t.profit > 0),
    )

    # end_date is None on every eager call (today, UTC); a lazy size-cap run
    # pins its eager point's last day so neighbouring caps share one span
    equity_df = _build_equity_curve(trades, start_date, initial_balance, end_date=end_date)
    return SweepPoint(
        k=effective_k, trades=trades, equity_df=equity_df,
        # The resolved tuple when a band was given, so (0, 1) and (0.0, 1.0)
        # stamp the same scenario; None stays None ("given no band").
        spread_band=None if spread_band is None else (band_lo, band_hi),
        population=population,
        # Only an explicit False marks a tier-off point
        tier_floors=tier_floors is not False,
        size_cap=cap, peak_kelly_fraction=peak_kelly,
        add_to_held=bool(add_to_held), sell_at=sell_level,
        cap_free_from=cap_free_from, sell_min_days=min_days,
        # Recorded only by a run that could sell
        sales=None if sell_level is None else tuple(sale_checks),
    )


# ─── Interval-discount calibration ────────────────────────────────────────────

def _calibration_bucket(
    label: str,
    tier: float,
    observations: Sequence[CalibrationObservation],
) -> IntervalCalibrationBucket:
    """
    Reduce a set of time-series observations to one calibration report row.

    Computes the realised in-between rate, the mean market-implied in-between
    mass (the mean mid spread, CalibrationObservation.implied), and their
    ratio — the empirical interval discount k_hat, i.e. how much of the mass
    the market priced actually materialized. The one
    definition of that arithmetic: _interval_calibration builds every row of
    its report here, and dashboard._khat_band, which regroups
    IntervalCalibration.observations by category, tag and spread band,
    reduces each group here too (through dashboard._khat_stat). Reducing the
    carried tuple itself reproduces the pooled row exactly — the same
    observations in the same order through the same arithmetic — while a
    group, or a reordered or reassembled copy of the whole, agrees with it
    only to float rounding.

    Args:
        label (str): Row label for the bucket ("0-7d", "POOLED", ...).
        tier (float): Price tier for the bucket, or 0.0 for the pooled row,
            which spans every tier (see IntervalCalibrationBucket.tier).
        observations (Sequence[CalibrationObservation]): The bucket's
            candidates, a list or a tuple. May be empty, which yields a
            zeroed row rather than a division error.

    Returns:
        IntervalCalibrationBucket: The row. empirical_k is None whenever
            mean_implied is not strictly positive — with a zero or negative
            mean mid spread (only entries whose earlier quote is crossed can
            pull it there; see CalibrationObservation.implied) the ratio is
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
    *,
    tier_floors: bool = True,
) -> IntervalCalibration | None:
    """
    Measure the empirical interval discount k over the prepared entries.

    The time-series bet loses exactly one settlement cell: the event first
    happens BETWEEN the two deadlines (earlier NO, later YES). The market's
    price of that cell is the mid spread: the later market's midpoint minus
    the earlier one's, a market's midpoint being halfway between its YES ask
    and its YES bid (config.time_series_mid_spread, DR-78). The forecast
    (config.time_series_profit_prob) believes only
    TIME_SERIES_INTERVAL_PROB_DISCOUNT of that price; this function measures
    the fraction that actually materialized:

        k_hat = P(earlier NO, later YES) / mean(mid spread)

    pooled and per deadline-gap band, so an operator can compare the hand-set
    constant against what the history did. k_hat and the forecast's k are
    fractions of the same quantity, so they compare directly.

    The population is deliberately k-INDEPENDENT: it reads the prepared
    entries directly (_prepare_entries()' output, or one spread band's
    _entries_for_band() output inside a sweep), NOT _simulate_at_discount()'s
    surviving candidates, so it is NOT filtered by the Kelly gate. Filtering
    by Kelly would make the estimate circular — the in-between rate would be
    measured only among the pairs the CURRENT k already liked, so a wrong k
    would confirm itself. Being k-independent also means one computation is
    valid for every k at one band, which is why BacktestSweep holds one of
    these per band rather than each SweepPoint holding its own. It is not
    band-independent: a band decides which pairs enter at all. It reads each
    pair's mid spread at its first qualifying Monday's quotes (the entry's own
    pA, nA, pB and nB), never at the Monday a simulation trades it on: that
    Monday depends on k, so using it would bring back the circularity
    described above. A first Monday whose quote is crossed (a market's YES
    ask below its own YES bid) is measured like any other, at its mid
    spread, though the Kelly gate never trades a crossed Monday.

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
        whose YES-ask gap already cleared its tier ever produced an entry.
        That is a feature, not a sampling flaw: it is exactly the conditional
        distribution the live sizer faces under the same entry rule, so k_hat
        is measured on the population TIME_SERIES_INTERVAL_PROB_DISCOUNT is
        applied to, and against the quantity it multiplies. One pair's mid
        spread can still be zero or negative, but only when its earlier
        market's quote is crossed (see CalibrationObservation.implied), so a
        bucket's mean can be too, and then it reports no k_hat
        (_calibration_bucket).

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
        tier_floors (bool): Keyword-only, labelling only, like spread_min:
            whether the entries were detected with the deadline-gap tiers
            applied. False (a tier-off sweep's calibrations) labels each gap
            band with the floor alone — the only floor those entries cleared —
            so at a floor of 0.0 every bucket's tier is 0.0. A renderer that
            follows IntervalCalibrationBucket.tier's `tier <= 0` sentinel —
            _log_interval_calibration's "-", the dashboard's calibration
            tables — would print such a bucket's tier like the pooled row's,
            because there is no floor to print: that is the documented
            reading of the sentinel. No tier-off calibration is logged today
            (only the primary band's tier-on calibration is). True (default)
            labels max(tier, spread_min).

    Returns:
        IntervalCalibration | None: The report — carrying, on `observations`,
            the candidates its pooled row was computed over, so a report can
            regroup them — or None when there is nothing
            to report — no time-series candidate produced a usable
            observation AND none was excluded as a premise violation (the
            codebase's return-None-on-nothing-to-say convention). Unlike most
            such conventions here, this None is NOT silent at the caller
            (DR-72): _log_interval_calibration logs one explanatory line for
            it rather than nothing at all.
    """
    observations: list[CalibrationObservation] = []
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

        # Market A's event ticker as the entry carries it — _find_entry's
        # canonicalized leg, the one BacktestTrade.event_ticker records for a
        # traded pair — so a report files an observation under the same
        # series its trade would be filed under. Read by TYPE: a non-string
        # (only a hand-edited cache can hold one) reads as absent rather than
        # raising out of infer_category and ending the whole sweep, the
        # fail-safe reading scanner.event_series gives the same field.
        raw_event = mA.get("event_ticker")
        event_ticker = raw_event if isinstance(raw_event, str) else ""
        observations.append(CalibrationObservation(
            # Carried out of _find_entry rather than recomputed, so a
            # candidate is always bucketed under the very gap its price tier
            # and the MAX_DEADLINE_GAP_DAYS cutoff were applied on.
            gap_days=entry["gap_days"],
            # The market's in-between chance at the midpoints, from the first
            # qualifying Monday's four quotes: the input the forecast's k
            # multiplies (config.time_series_mid_spread, DR-78), so k-hat and
            # k are fractions of one quantity
            implied=time_series_mid_spread(entry["pA"], entry["nA"],
                                           entry["pB"], entry["nB"]),
            in_between=(outcome_a == "no" and outcome_b == "yes"),
            event_ticker=event_ticker,
            # The ticker-prefix label BacktestTrade.category carries, which a
            # report falls back to when the series has no Kalshi category
            category=infer_category(event_ticker),
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
            # is the floor these entries were actually detected under — the
            # floor alone when they were detected with the tiers off.
            min_price_diff_for_gap(hi, spread_min=spread_min, tier_floors=tier_floors),
            band,
        ))

    # 0.0 tier: the pooled row spans every band, so it has no single tier.
    pooled = _calibration_bucket(_CALIBRATION_POOLED_LABEL, 0.0, observations)
    return IntervalCalibration(
        pooled=pooled,
        buckets=buckets,
        excluded_premise_violations=excluded,
        # The very list the pooled row was reduced from, in measurement
        # order: reducing it again reproduces the pooled row exactly (a
        # regrouped or reordered copy agrees only to float rounding)
        observations=tuple(observations),
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
    config.py or the saved live defaults, and the live sizer keeps reading
    the saved live defaults' k (or main.py's --interval-discount) whatever
    this prints.

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
        "market-implied gap at the midpoints)"
    )
    logging.info("  %-14s%4s%9s%12s%11s%10s",
                 "bucket", "tier", "n", "realised", "implied", "k_hat")

    for b in [*calibration.buckets, calibration.pooled]:
        # tier <= 0 marks a row with no floor to print: the pooled row, which
        # spans every tier, or a tier-off bucket at a band floor of 0
        # (IntervalCalibrationBucket.tier).
        tier_txt = "-" if b.tier <= 0 else f"{b.tier:.2f}"
        k_txt = "-" if b.empirical_k is None else f"{b.empirical_k:.3f}"
        logging.info("  %-14s%4s%9d%12.4f%11.4f%10s",
                     b.label, tier_txt, b.n, b.realised_rate, b.mean_implied, k_txt)

    pooled_k = calibration.pooled.empirical_k
    logging.info(
        "  Configured k = %.3f (config.TIME_SERIES_INTERVAL_PROB_DISCOUNT) | "
        "pooled empirical k_hat = %s",
        # The CONFIG constant, not any sweep point's override: this line
        # compares the measurement against config.py's k, the one this
        # backtest defaults to (a live run sizes at the saved live defaults' k)
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
    config.TIME_SERIES_INTERVAL_PROB_DISCOUNT (config.py's k, the backtest's
    default; a live run prices at the saved live defaults' k):
    _prepare_entries() does the k-independent work and
    _simulate_at_discount(..., k=None) does the k-dependent work.

    Algorithm (the step split between the two helpers is noted):
      1. Fetch all settled markets since start_date.                [prepare]
      2. Drop markets that provably can never enter (_can_ever_enter). [prepare]
      3. Group into potential time-series and same-title pairs.     [prepare]
      4. Fetch hourly candlesticks for every ticker appearing in a
         potential pair, in parallel across CANDLESTICK_FETCH_MAX_WORKERS
         threads.                                                   [prepare]
      5. Find every entry checkpoint (the live run's weekly moment) where
         the pair was tradeable at the threshold.                   [prepare]
      6. Keep each time-series pair's qualifying Mondays that pass the
         Kelly gate, and each same-title pair's first; exclude (and count,
         with one summary WARNING) any time-series candidate whose
         settlement was earlier-YES/later-NO — impossible for a
         cumulative-deadline pair, so a premise violation rather than a
         payout; keep only the best same-title entry per title group (live
         one-pair-per-group rule), then drop any time-series candidate whose
         ticker pair was also found as a same-title candidate (live
         main._dedup_pairs rule).                                  [simulate]
      7. Walk the entries in date order with a running cash balance: each
         trade bets a share of that Monday's portfolio value (cash plus open
         trades at market) but spends at most the cash left, shrinking to fit;
         at most one open time-series trade per ladder; each pair trades at
         most once (only the add-on family of run_backtest_sweep,
         _simulate_at_discount(add_to_held=True), adds to a pair it still
         holds); record the actual profit or loss.                  [simulate]
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
            checkpoint, a pair can only enter if the later-closing leg
            closes within max_horizon_days of THAT checkpoint. None (default)
            applies no cap. Passed straight through to _find_entry().

    Returns:
        tuple[list[BacktestTrade], pd.DataFrame]: (trades, equity_df).
            trades is one BacktestTrade per entered pair, in entry-date order
            (empty if none were ever entered). equity_df has columns
            [date, portfolio_value, daily_return], one row per day from
            start_date - 1 day (the untouched initial balance) through today,
            flat at initial_balance if trades is empty. portfolio_value is cash
            plus open positions at market (see _build_equity_curve), so
            deploying capital moves it only by the fees.

    Raises:
        ValueError: Before any fetch, from _prepare_candidates(), when
            SCHEDULED_RUN cannot place the entry checkpoints (a configuration
            error).
        KeyError: Propagates out of the candlestick-fetch pool
            (_fetch_candles_parallel) if a ticker needed by a candidate pair
            was not properly excluded by the eligibility prefilter — this is
            treated as a real defect (a market that should never have reached
            this stage), not degraded into "no price history".

    Note:
        Before any network call, _prepare_entries() checks whether [start_date,
        today] holds at least one entry checkpoint (a SCHEDULED_RUN weekday).
        If not, no trade can ever be entered regardless of what the fetch would return,
        so the fetch is skipped entirely, that helper returns None, and this
        returns the same empty-result shape as the zero-trade path ([], an
        equity curve flat at initial_balance — for a future start_date, its
        leading row plus start_date's own) with a WARNING logged.
    """
    logging.info("Starting backtest from %s with $%.2f", start_date, initial_balance)

    # The k-independent half: fetch, group, pair and locate each pair's
    # tradeable Mondays. None means the feasibility pre-check failed.
    #
    # The outcome-label census rides out alongside the entries (DR-66b), but
    # this entry point returns the historical two-tuple and feeds no dashboard,
    # so it is discarded here — the census has already logged itself.
    #
    # No same_event_ladders argument, deliberately: this function keeps its
    # exact pre-DR-73 signature for every existing caller, and omitting the
    # keyword leaves _prepare_entries' None sentinel to resolve this module's
    # TIME_SERIES_SAME_EVENT_LADDERS (bound from config at import) at call
    # time — the value the live finder uses. A harness that wants ladders on or
    # off here (they follow the switch, on by the 2026-09-26 decision) must patch
    # backtester.TIME_SERIES_SAME_EVENT_LADDERS, not the config attribute, which
    # this module never reads; the supported lever is
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
    # TIME_SERIES_INTERVAL_PROB_DISCOUNT, config.py's k, the one this backtest
    # defaults to
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

    It uses each pair's first qualifying Monday (entry_date), never the
    Monday a simulation trades it on, which depends on k: every scenario of
    the sweep must split at the same date.

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

    Each pair goes to the half its first qualifying Monday falls in. A
    first-half (H1) pair keeps only its Mondays before the split, so the H1
    simulation cannot enter it on a Monday from the second period, and it is
    not copied into the second half (H2), where a same-title pair could enter
    on a Monday the full run never tries. So each half's candidates are a
    subset of the full run's and its peak Kelly fraction is never higher,
    which lets CapSweep reuse the halves. The cost: an H1 pair the full run
    can only take after the split trades in neither half.

    Trimming a pair's Mondays builds a new record rather than editing the
    old one, because records are shared (by a band's populations, checks and
    CapSweep, and each same-title record by every band of both tier
    settings). Records that lose nothing are returned unchanged.

    Args:
        entries (list[dict]): Prepared entry records (each with
            ["entry"]["entry_date"], the first qualifying Monday).
        split_date (date): BacktestSweep.split_date.

    Returns:
        tuple[list[dict], list[dict]]: (every entry whose first qualifying
            Monday is before split_date, with its later Mondays cut to those
            before it; every other entry, unchanged) — either may be empty.
    """
    first_half: list[dict] = []
    second_half: list[dict] = []
    for rec in entries:
        entry = rec["entry"]
        if entry["entry_date"] >= split_date:
            # Every Monday of this pair is on or after the split
            second_half.append(rec)
            continue
        # Keep only this H1 pair's Mondays before the split
        later = entry.get("later", ())
        kept = tuple(m for m in later if m["entry_date"] < split_date)
        # A new record and entry dict — never entry["later"] = kept, which
        # would truncate the entry everywhere it is shared
        first_half.append(rec if len(kept) == len(later)
                          else {**rec, "entry": {**entry, "later": kept}})
    return first_half, second_half


def _half_split(
    halves: tuple[list[dict], list[dict]],
    start_date: date,
    initial_balance: float,
    k: float,
    band: tuple[float, float],
    population: str = "all",
    *,
    tier_floors: bool = True,
    size_cap: float | None = None,
    quiet: bool = False,
    end_date: date | None = None,
    add_to_held: bool = False,
    sell_at: float | None = None,
    sell_min_days: int | None = None,
) -> HalfSplit:
    """
    Simulate each half of one scenario's entries alone and keep three numbers each.

    Both halves are simulated at the scenario's own size cap, tier-floor
    setting, add-on setting, sell level and minimum of days before maturity,
    forwarded through _sim_options — which
    forwards NOTHING on a default call, so the eager tier-on band sweep calls
    _simulate_at_discount with exactly the keywords it always did.

    Args:
        halves (tuple[list[dict], list[dict]]): The scenario's entries as
            _split_halves splits them at BacktestSweep.split_date — (before
            it, on or after it), with each first-half pair's later Mondays
            cut at the split.
        start_date (date): The backtest's start date.
        initial_balance (float): The balance EACH half starts from, in dollars
            — the halves are two independent runs, never one run's two parts.
        k (float): The scenario's resolved interval discount.
        band (tuple[float, float]): The scenario's resolved band, the one its
            entries were found under; it goes into the live spread rule.
        population (str): The checked population the halves belong to —
            "all" (default) or "time_series" — which names the two runs
            "<population>/H1" and "<population>/H2" on their completion lines,
            so the two populations' split-half runs never share a prefix.
        tier_floors (bool): Keyword-only. The scenario's tier-floor setting,
            the one its entries were found under, handed to both halves'
            simulations (it goes into the live spread rule, and names a
            tier-off scenario's halves as tier-off runs).
        size_cap (float | None): Keyword-only. The scenario's per-trade size
            cap; None (default) or BUDGET_FRACTION for the run's own.
        quiet (bool): Keyword-only. Log both halves' completion lines (and
            any premise WARNING) at DEBUG — the lazy size-cap runs.
        end_date (date | None): Keyword-only. The day both halves' equity
            curves end on (a lazy size-cap run pins its eager point's); None
            (default) for today (UTC), forwarded only when given.
        add_to_held (bool): Keyword-only. Whether both halves may add to a
            pair they still hold, as the scenario did; forwarded only when
            True. Default False.
        sell_at (float | None): Keyword-only. The share of potential profit
            at which both halves sell a position, as the scenario did;
            forwarded only when set. Default None.
        sell_min_days (int | None): Keyword-only. The fewest days before
            maturity at which both halves sell a position, as the scenario
            did; forwarded only when set. Default None.

    Returns:
        HalfSplit: Each half's total return, trade count and entry count. The
            halves' equity curves are dropped; an entry count of 0 marks a
            half whose 0.0 return is not a measurement.
    """
    first, second = halves
    # Only the options that differ from the defaults, so a default call is
    # byte-for-byte the call it always was
    options = _sim_options(size_cap, quiet, end_date=end_date, tier_floors=tier_floors,
                           add_to_held=add_to_held, sell_at=sell_at,
                           sell_min_days=sell_min_days)
    h1 = _simulate_at_discount(first, start_date, initial_balance, k=k,
                               spread_band=band, population=f"{population}/H1", **options)
    h2 = _simulate_at_discount(second, start_date, initial_balance, k=k,
                               spread_band=band, population=f"{population}/H2", **options)
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
    *,
    tier_floors: bool = True,
    quiet: bool = False,
    end_date: date | None = None,
) -> tuple[str, float] | None:
    """
    Measure how much of one scenario's result a single event carried.

    Finds the event whose trades made the most profit on the point (by market
    A's event ticker, skipping trades without one; ties go to the
    alphabetically first), then re-runs the simulation without every pair
    whose market A is in that event on any of its qualifying Mondays, at the
    same starting balance, k, band and size cap. It re-runs rather than
    subtracting that event's profit, because the other trades would have been
    sized differently without it. The re-run also adds to held pairs exactly
    when the point did (point.add_to_held), and sells early at the point's
    level and minimum of days before maturity (point.sell_at,
    point.sell_min_days).

    Args:
        point (SweepPoint): The scenario's "all" or "time_series" point.
        entries (list[dict]): The entries that point was simulated from.
        start_date (date): The backtest's start date.
        initial_balance (float): The balance the re-run starts from.
        band (tuple[float, float]): The scenario's band, the one its entries
            were found under.
        population (str): The point's population, used to name the re-run's log line.
        tier_floors (bool): Keyword-only. The point's tier-floor setting, the
            one its entries were found under.
        quiet (bool): Keyword-only. Log the re-run's lines at DEBUG.
        end_date (date | None): Keyword-only. Last day of the re-run's equity curve; None for today.

    Returns:
        tuple[str, float] | None: (event ticker, total return without it); None when no trade
            names an event.
    """
    pnl_by_event: dict[str, float] = defaultdict(float)
    for t in point.trades:
        if t.event_ticker:
            pnl_by_event[t.event_ticker] += t.profit
    if not pnl_by_event:
        return None
    # Largest summed profit first; the ticker breaks exact ties.
    top = min(pnl_by_event, key=lambda ev: (-pnl_by_event[ev], ev))
    # Drop every entry whose market A names the event on any qualifying
    # Monday (see the docstring)
    rest = [rec for rec in entries
            if all((monday["mA"].get("event_ticker") or "") != top
                   for monday in _entry_mondays(rec["entry"]))]
    without = _simulate_at_discount(rest, start_date, initial_balance, k=point.k,
                                    spread_band=band, population=f"{population}/ex-top",
                                    **_sim_options(point.size_cap, quiet,
                                                   end_date=end_date,
                                                   tier_floors=tier_floors,
                                                   add_to_held=point.add_to_held,
                                                   sell_at=point.sell_at,
                                                   sell_min_days=point.sell_min_days))
    return top, _total_return(without, initial_balance)


def _entered_pairs(candidates: _Candidates, entries: list[dict]) -> list:
    """
    The time-series candidate pairs that produced one of these entries.

    The rescan list of a band sweep's no-band pre-pass (the tier-on one, or
    the tier-floors-off family's own — the subset holds per tier setting,
    never across them): every band's accepted Mondays are a subset of the
    no-band band's at the same tier setting (see _sweep_from_candidates),
    so a pair that entered nowhere there can enter at no band of that
    setting, and every other band of it rescans only what this returns.
    Matched on the legs' TICKERS,
    not on object identity: whatever _find_entry hands back (the legs it was
    given, possibly swapped), the two tickers name the pair, and a ticker
    pair can only ever OVER-include a pair here (a duplicate is rescanned,
    never lost).

    Args:
        candidates (_Candidates): The sweep's candidates, before its pair
            list is released.
        entries (list[dict]): A pre-pass's _entries_for_band() output.

    Returns:
        list: The time-series items of candidates.all_pairs whose ticker pair
            produced one of the entries, in all_pairs' order. A new list;
            all_pairs is never mutated.
    """
    entered = {frozenset((rec["entry"]["mA"]["ticker"], rec["entry"]["mB"]["ticker"]))
               for rec in entries}
    return [item for item in candidates.all_pairs
            if item[1] == "time_series"
            and frozenset((item[0][0]["ticker"], item[0][1]["ticker"])) in entered]


def _band_populations(
    entries: list[dict],
    split_date: date,
) -> tuple[list[tuple[str, list[dict]]], dict[str, tuple[list[dict], list[dict]]]]:
    """
    Split one band's entries into its standalone populations and their halves.

    The populations are k-independent subsets, so a band sweep splits them
    once per band through _population_subsets — the ONE definition of the
    split, on _is_ladder_pair, the same rule that labels each trade's
    same_event_ladder, so a trade and the population it was simulated in
    always agree, and a lazy CapSweep cell simulates exactly these subsets.
    "time_series" is ladders + cross-event together, same-title excluded: the
    population the band and k actually act on, and the one the dashboard's
    heatmap and fragility banner read, so a same-title result (band- and
    k-independent) can never dilute them. The two checked populations'
    halves are split at the ONE split date.

    Args:
        entries (list[dict]): One band's entries — its time-series entries
            then the shared same-title ones.
        split_date (date): BacktestSweep.split_date.

    Returns:
        tuple: (populations, halves_by_population). populations is
            [("time_series", ...), ("ladder", ...), ("cross", ...)], each an
            ordered subset of entries (possibly empty — the caller skips an
            empty one rather than simulating an empty scenario);
            halves_by_population maps "all" and "time_series" to their
            _split_halves at split_date.
    """
    # The one split CapSweep.cell also reads, so an eager band sweep cell and
    # a lazy size-cap cell always simulate the same populations
    populations = _population_subsets(entries)
    ts_only = populations[0][1]

    # Both checked populations' halves, at the ONE split date.
    halves_by_population = {"all": _split_halves(entries, split_date),
                            "time_series": _split_halves(ts_only, split_date)}
    return populations, halves_by_population


def _band_sweep_cell(
    point: SweepPoint,
    entries: list[dict],
    populations: list[tuple[str, list[dict]]],
    halves_by_population: dict[str, tuple[list[dict], list[dict]]],
    start_date: date,
    initial_balance: float,
    point_k: float,
    band: tuple[float, float],
    *,
    tier_floors: bool = True,
) -> list[SweepPoint]:
    """
    Run one band x k cell's robustness checks and population simulations.

    The two robustness checks are set on the "all" point itself (the primary
    included — same object everywhere it is held) and on the "time_series"
    point: the dashboard reads the latter's, and keeps the former's for its
    own "All" row. Every other population gets a standalone simulation from
    the initial balance, never a slice of the "all" run, so its return,
    drawdown and Sharpe are defined. A population with no entry at this band
    is skipped rather than simulated as an empty scenario.

    Args:
        point (SweepPoint): The cell's "all" point, already simulated.
            MUTATED: its halves and ex_top_event are set here.
        entries (list[dict]): The entries the "all" point was simulated from.
        populations (list[tuple[str, list[dict]]]): _band_populations' first
            element for this band.
        halves_by_population (dict[str, tuple[list[dict], list[dict]]]):
            _band_populations' second element for this band.
        start_date (date): The backtest's start date.
        initial_balance (float): The balance every simulation here starts
            from, in dollars.
        point_k (float): The cell's resolved interval discount.
        band (tuple[float, float]): The cell's resolved band, the one its
            entries were found under; it goes into the live spread rule.
        tier_floors (bool): Keyword-only. The cell's tier-floor setting, the
            one its entries were found under, handed to every
            simulation here through _sim_options — so only an explicit False
            is ever forwarded. The tier-on sweep never passes it.

    Returns:
        list[SweepPoint]: [point, *population points] — the order
            BacktestSweep.scenarios has always held them in.
    """
    cell = [point]
    point.halves = _half_split(halves_by_population["all"], start_date,
                               initial_balance, point_k, band, population="all",
                               tier_floors=tier_floors)
    point.ex_top_event = _ex_top_event(point, entries, start_date,
                                       initial_balance, band, population="all",
                                       tier_floors=tier_floors)
    for label, subset in populations:
        if not subset:
            continue
        # tier_floors through _sim_options: forwarded only when False, so a
        # tier-on cell calls the simulator with exactly its old keywords
        pop_point = _simulate_at_discount(
            subset, start_date, initial_balance, k=point_k,
            spread_band=band, population=label,
            **_sim_options(None, False, tier_floors=tier_floors),
        )
        if label in _CHECKED_POPULATIONS:
            pop_point.halves = _half_split(
                halves_by_population[label], start_date, initial_balance,
                point_k, band, population=label, tier_floors=tier_floors)
            pop_point.ex_top_event = _ex_top_event(
                pop_point, subset, start_date, initial_balance, band,
                population=label, tier_floors=tier_floors)
        cell.append(pop_point)
    return cell


def _sweep_from_candidates(
    candidates: _Candidates,
    initial_balance: float,
    *,
    interval_discount: float | None,
    sweep: bool,
    spread_band: tuple[float, float] | None,
    band_sweep: bool,
    tier_off_sweep: bool = False,
    cap_sweep: bool = False,
    add_on_sweep: bool = False,
    sell_sweep: bool = False,
    live: LiveSettings | None | object = _LIVE_NOT_READ,
    depth_model: DepthModel | None = None,
) -> BacktestSweep:
    """
    Run every entry pass and every simulation of one backtest over one fetch.

    The second half of run_backtest_sweep(): everything after
    _prepare_candidates(). It runs in two phases so the candle series are
    held no longer than a single-band run holds them.

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

    The tier-off family (tier_off_sweep, backtest-only; a band sweep is
    required) runs every band where a deadline-gap tier floor binds
    (_tier_floors_bind — on the shipped grid the 18 bands with a floor below
    0.30) a second time with the tiers not applied, so that band's floor
    alone gates the spread and sets the leg-price-sum ceiling (1 − floor).
    In Phase 1, after the tier-on band loop and before the candles are
    released, it runs its own no-band pre-pass at (0, 1) with the tiers off
    and rescans only the pairs that entered there at every other binding
    band (17 of them on the shipped grid) — the pre-pass argument holds
    unchanged, since with the tiers off the threshold is the floor alone,
    which still only rises from the no-band band's 0.0, the ceiling still
    only drops, and the refusal of a spread that is not strictly positive
    reads no band. The tier-ON pre-pass cannot serve here: a pair the tiers
    refuse everywhere can still enter with them off. In Phase 2, after
    the same-title point, every binding band gets its own tier-off
    calibration (labelled with the floor alone) and, at every k of the same
    grid, the same scenario block as the tier-on sweep (_band_sweep_cell:
    populations, halves at the ONE split date, the excluding-top-event
    check), returned as BacktestSweep.tier_off_scenarios and
    tier_off_calibrations_by_band. A band whose floor sits at or above both
    tiers enters the same pairs on the same Mondays either way, so it is
    neither entered nor simulated twice. The tier-on payload — primary,
    points, scenarios, every calibration, same_title_point, split_date — is
    exactly what the run returns without it. Its announcement lines begin
    "Tier floors off", never "Spread band " or "Split-half check: split
    date"; its simulations' completion lines carry " with the tier floors
    off" after the band, and a premise-violation WARNING from one of them
    ends " [tier floors off]".

    With cap_sweep, every TIER-ON point simulated above (never a
    tier_off_scenarios point) is recorded as the eager seed of its (band, k,
    population) as it is simulated — the "all" point before the
    band-sweep-only work, so a single-band run records its points too, then
    each population point of _band_sweep_cell, and the same-title point —
    and a CapSweep over the same entries is returned on
    BacktestSweep.cap_sweep. It simulates NOTHING here: every other size cap
    is simulated only when a reader asks for a cell, so this function's cost
    and every point it returns are unchanged by the flag. Its seeds are
    TIER-ON points only: a tier-off point (tier_floors False) is never
    recorded as one, because
    the CapSweep's entries are the tier-on entries_by_band and a tier-off
    seed would stand in for a simulation of different entries. With
    tier_off_sweep too (and at least one binding band), every point of the
    tier-off family's cells is recorded instead as a seed of a SECOND
    CapSweep (tier_floors False, over the binding bands' tier-off entries,
    no same-title population of its own), returned on
    BacktestSweep.tier_off_cap_sweep and announced by one INFO line
    ("Tier floors off: size-cap sweep: …") — so every tier x band x k x cap
    scenario is a real simulation, and neither sweep ever holds the other's
    seeds. It too simulates nothing here.

    With add_on_sweep, two more CapSweeps are returned (add_on_cap_sweep over
    the tier-on entries, and add_on_tier_off_cap_sweep over the tier-floors-off
    family's binding bands when there are any): every simulation they run adds
    to held pairs, and only the "all" population is read (checks off). No
    eager run adds to held pairs, so they hold no eager point; each cell ends
    its curves on the day its eager twin's ended, which is recorded here as
    each eager "all" point is simulated (the tier-on loop's and the tier-off
    loop's, in separate maps). They simulate nothing here either: the run's
    simulations, log lines and every figure are those of a run without the
    flag. They keep
    entries_by_band and tier_off_entries alive for the reader, as the size-cap
    sweeps do.

    With sell_sweep, a SellSweep is returned (BacktestSweep.sell_sweep) over
    the same entries, caps, bands and k grid as the add-on family, tier floors
    on and off, with the same end-date maps, announced by one INFO line. It
    too simulates nothing here.

    Args:
        candidates (_Candidates): _prepare_candidates() output. CONSUMED: its
            candles_by_ticker and all_pairs attributes are deleted after
            Phase 1, so one _Candidates feeds one sweep.
        initial_balance (float): Simulated starting cash balance in dollars;
            every scenario, half and re-simulation starts from it.
        interval_discount (float | None): The primary k, in (0, 1]. None means
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
        tier_off_sweep (bool): When True (a band sweep required), also run
            the tier-off family described above. False (default) runs none
            of it: tier_off_scenarios is [] and tier_off_calibrations_by_band
            is {}.
        cap_sweep (bool): When True, return a lazy CapSweep over
            SIZE_CAP_SWEEP (unioned with the run's own cap) on
            BacktestSweep.cap_sweep, seeded from every TIER-ON point
            simulated here (the primary, each swept k, each band-sweep
            scenario and the same-title point; never a tier_off_scenarios
            point) and keeping entries_by_band alive for it; its cells carry the
            populations and checks this run computed (all four populations
            and the checks with band_sweep, "all" alone without). With
            tier_off_sweep as well, also a tier-floors-off CapSweep on
            BacktestSweep.tier_off_cap_sweep, seeded from the family's own
            points only and keeping its entries alive. False (default)
            returns cap_sweep=None and tier_off_cap_sweep=None.
        add_on_sweep (bool): When True, also return the add-on family
            described above on BacktestSweep.add_on_cap_sweep and
            add_on_tier_off_cap_sweep (the caps of the size-cap grid with
            cap_sweep, the run's own cap alone without). False (default)
            returns both as None.
        sell_sweep (bool): When True, also return the sell family described
            above on BacktestSweep.sell_sweep (the same caps as the add-on
            family). False (default) returns None.
        live (LiveSettings | None): Keyword-only. run_backtest_sweep's read of
            the saved live defaults (None: none saved, or the file refused);
            left out, read here.
        depth_model (DepthModel | None): Keyword-only. The depth model every
            entry's quotes carry, so every simulation's trades walk synthetic
            books; None (default) fills every trade at the top of the book.

    Returns:
        BacktestSweep: primary, points (the primary band's k sweep),
            calibration (the primary band's), label_coverage (carried from
            candidates), scenarios, same_title_point, calibrations_by_band,
            same_event_ladders (resolved), split_date, corpus_provenance
            (carried from candidates), config_same_event_ladders (the
            configured switch, read beside same_event_ladders's
            resolution), tier_off_scenarios, tier_off_calibrations_by_band,
            cap_sweep, tier_off_cap_sweep, add_on_cap_sweep,
            add_on_tier_off_cap_sweep, same_title_size_cap, the nine live_*
            fields, entry_checkpoint, sell_sweep and depth_model — see
            BacktestSweep.

    Raises:
        ValueError: If tier_off_sweep is set without band_sweep (the tier-off
            family is a band-sweep family: its split date, rescans and cells
            are the band sweep's), checked first; from
            config.time_series_spread_band for an invalid band; or from the
            first simulation, for a bad SAME_TITLE_SIZE_CAP.
        AttributeError: If candidates has already fed a sweep (its
            all_pairs and candles_by_ticker were deleted) — loud rather than
            a silent sweep with no pairs and therefore no entries.
    """
    if tier_off_sweep and not band_sweep:
        raise ValueError("tier_off_sweep needs band_sweep: the tier-off family "
                         "re-runs the band sweep's binding bands")
    start_date = candidates.start_date
    primary_band = time_series_spread_band(spread_band)
    # The saved live defaults: run_backtest_sweep's read, or our own for a
    # direct caller
    if live is _LIVE_NOT_READ:
        live = _live_settings_for_report()
    # The ladder setting this sweep's pairs were extracted under, resolved the
    # way _extract_pairs and _find_entry resolve the unresolved flag carried on
    # candidates (see _Candidates for when those can disagree), for the report.
    ladders = bool(TIME_SERIES_SAME_EVENT_LADDERS if candidates.same_event_ladders is None
                   else candidates.same_event_ladders)
    # The configured switch this run is judged against — read beside the
    # resolution above so one pass reads both (see BacktestSweep.config_same_event_ladders).
    config_ladders = bool(TIME_SERIES_SAME_EVENT_LADDERS)
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
    #     horizon, the candle lookups, the refusal of a spread that is not
    #     strictly positive, the live-quote checks and the fee check — never
    #     reads the band.
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
        # The pairs that entered at the no-band band, matched on the legs'
        # TICKERS (a duplicate is rescanned, never lost) — a new list;
        # candidates.all_pairs itself is never mutated.
        rescan = _entered_pairs(candidates, no_band_entries)
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
        # How many Mondays this band's entries qualified on (DR-75), zero included
        _log_qualifying_mondays(entries_by_band[band], band)

    # ── Tier floors off (tier_off_sweep): the bands where a deadline-gap tier
    # binds, entered again with the tiers not applied. A band whose floor is
    # at or above both tiers enters the same pairs on the same Mondays either
    # way (_tier_floors_bind), so it is not entered twice; the dashboard reuses
    # its tier-on run. The pre-pass argument above holds unchanged with the
    # tiers off — the threshold is the floor alone, which still only rises from
    # the no-band band's 0.0, and the ceiling still only drops — so the tier-off
    # no-band band's entries bound every tier-off band's. They need a pre-pass
    # of their OWN: the tier-on one cannot serve, since a pair the tiers refuse
    # at every band can still enter with them off (off_rescan, never rescan).
    tier_off_bands = [band for band in bands if _tier_floors_bind(band)] if tier_off_sweep else []
    tier_off_entries: dict[tuple[float, float], list[dict]] = {}
    off_rescan = off_no_band_entries = None
    if tier_off_bands:
        logging.info(
            "Tier floors off (backtest only): %d of %d spread bands have a floor below a "
            "deadline-gap tier and are entered again without the tiers; the other %d enter "
            "the same pairs either way", len(tier_off_bands), len(bands),
            len(bands) - len(tier_off_bands))
        logging.info("Tier floors off: no-band pre-pass: scanning all %d time-series pairs at %s",
                     n_ts_pairs, _band_label(no_band))
        # The one full tier-off scan, at (0, 1): with the tiers off its
        # threshold is 0.0, no higher than any other band's floor
        off_no_band_entries = _entries_for_band(candidates, no_band, pair_types=("time_series",),
                                                tier_floors=False)
        off_rescan = _entered_pairs(candidates, off_no_band_entries)
        logging.info("Tier floors off: no-band pre-pass: %d of %d time-series pairs produced an "
                     "entry; every other tier-off band rescans only those",
                     len(off_rescan), n_ts_pairs)
        for i, band in enumerate(tier_off_bands, start=1):
            logging.info("Tier floors off: spread band %d/%d: %s", i, len(tier_off_bands),
                         _band_label(band))
            ts_entries = (off_no_band_entries if band == no_band
                          else _entries_for_band(candidates, band, pair_types=("time_series",),
                                                 tier_floors=False, _pairs=off_rescan))
            # The same-title entries never read the tiers: shared, as above
            tier_off_entries[band] = ts_entries + st_entries
            logging.info("Tier floors off: prepared %d candidate entries for sizing "
                         "(%d time-series, %d same-title)", len(tier_off_entries[band]),
                         len(ts_entries), len(st_entries))
            _log_qualifying_mondays(tier_off_entries[band], band, tier_floors=False)
    # The prices that value every entry's trades at market while open (every
    # band's records, tier floors on and off; the same-title records sit in
    # each band's list), taken from the candles before they are released,
    # with the depth model their synthetic books are built from. Each
    # market's LegQuotes is shared by every record that holds it.
    _attach_leg_quotes(
        [rec for entries in (*entries_by_band.values(), *tier_off_entries.values())
         for rec in entries],
        candidates.candles_by_ticker, candidates.start_date, depth_model)
    # Nothing below reads a candle or the pair list. Releasing both here keeps
    # the peak at a single-band run's entry-pass peak and, like the
    # single-band path (whose pair list dies with _prepare_entries' locals),
    # holds no pair tuple through the simulations: the entry dicts carry the
    # market records they need (mA/mB) and never a candle, and the records'
    # quotes keep only small sampled arrays. Only the scalar fields —
    # label_coverage, start_date, same_event_ladders, corpus_provenance — are
    # read after this point. The pre-passes' rescan lists are lists of pair
    # tuples too, so they go with them (their entries already live on in
    # entries_by_band and tier_off_entries).
    del candidates.candles_by_ticker, candidates.all_pairs
    del rescan, no_band_entries, off_rescan, off_no_band_entries

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
    # Each entry's date is its pair's first qualifying Monday, which no k moves
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
    # (band, k, population) -> the point simulated here at the run's own cap —
    # the seed a size-cap sweep reuses (CapSweep). Filled only with cap_sweep;
    # it holds references to points already kept above, never a copy.
    eager: dict[tuple, SweepPoint] = {}
    # (band, k, "all") -> the last day the eager "all" point's curve ends on.
    # The add-on and sell sweeps end every cell's curves there too, so a cell
    # and its eager twin cover one span. Filled only with add_on_sweep or
    # sell_sweep (the tier-off loop fills its own map, since the two share keys).
    add_on_end_dates: dict[tuple, date | None] = {}
    add_on_off_end_dates: dict[tuple, date | None] = {}
    lazy_families = add_on_sweep or sell_sweep
    for bi, band in enumerate(bands, start=1):
        entries = entries_by_band[band]
        # The primary's is the object already measured and logged above.
        calibrations_by_band[band] = (
            calibration if band == primary_band
            else _interval_calibration(entries, spread_min=band[0])
        )
        if band_sweep:
            # Standalone populations, split once per band (they are
            # k-independent subsets) through _population_subsets — the one
            # definition, shared with CapSweep.cell — and both checked
            # populations' halves at the ONE split date. A population with no
            # entry at this band is skipped rather than simulated as an empty
            # scenario.
            populations, halves_by_population = _band_populations(entries, split_date)
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
            if cap_sweep:
                # Recorded BEFORE the band-sweep-only work below, so a
                # single-band run (band_sweep False) still seeds its cells
                eager[(band, point_k, "all")] = point
            if lazy_families:
                # Keyed by the exact band and k objects the add-on and sell
                # cells are read with (the tuples this loop and
                # CapSweep.bands/.ks hold)
                add_on_end_dates[(band, point_k, "all")] = _curve_end_date(point)
            if not band_sweep:
                continue

            # The robustness checks on the "all" point (the primary included —
            # same object everywhere it is held), then its population
            # simulations with theirs: [point, *population points], the
            # order scenarios has always held them in
            cell = _band_sweep_cell(
                point, entries, populations, halves_by_population, start_date,
                initial_balance, point_k, band)
            scenarios.extend(cell)
            if cap_sweep:
                # Each population point is its (band, k, population) cell's
                # size-cap seed; cell[0] is the "all" point, recorded above
                for pop_point in cell[1:]:
                    eager[(band, point_k, pop_point.population)] = pop_point

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

    # ── Tier floors off: every binding band at every k of the SAME grid, the
    # same scenario block as above — populations, halves at the ONE split date,
    # the excluding-top-event check — so the dashboard can swap a tier-off cell
    # in for a tier-on one like for like.
    tier_off_scenarios: list[SweepPoint] = []
    tier_off_calibrations: dict[tuple[float, float], IntervalCalibration | None] = {}
    # (band, k, population) -> the family's own point at the run's own cap —
    # the seed of the TIER-OFF size-cap sweep below. Kept apart from `eager`
    # (the tier-on seeds) because the two share keys: a tier-off point filed
    # there would stand in for a simulation of the tier-on entries. Filled
    # only with cap_sweep; references to points already kept, never copies.
    off_eager: dict[tuple, SweepPoint] = {}
    for bi, band in enumerate(tier_off_bands, start=1):
        entries = tier_off_entries[band]
        # Labelled with the floor alone: the only floor these entries cleared
        tier_off_calibrations[band] = _interval_calibration(entries, spread_min=band[0],
                                                            tier_floors=False)
        populations, halves_by_population = _band_populations(entries, split_date)
        logging.info("Tier floors off: simulating spread band %d/%d: %s (%d entries)",
                     bi, len(tier_off_bands), _band_label(band), len(entries))
        for point_k in grid:
            point = _simulate_at_discount(entries, start_date, initial_balance, k=point_k,
                                          spread_band=band, population="all", tier_floors=False)
            if lazy_families:
                add_on_off_end_dates[(band, point_k, "all")] = _curve_end_date(point)
            # Never recorded in eager: the tier-on CapSweep below re-simulates
            # the TIER-ON entries_by_band, so a tier-off seed would stand in
            # for a simulation of different entries. Recorded in off_eager
            # instead, the seeds of the tier-off CapSweep over these entries.
            cell = _band_sweep_cell(
                point, entries, populations, halves_by_population, start_date,
                initial_balance, point_k, band, tier_floors=False)
            tier_off_scenarios.extend(cell)
            if cap_sweep:
                # Every point of the cell — "all" first — seeds its own
                # (band, k, population), exactly as the tier-on loop records
                for off_point in cell:
                    off_eager[(band, point_k, off_point.population)] = off_point

    # The size caps the grid offers: the run's own cap (every point above
    # carries it, resolved at simulation time) unioned into SIZE_CAP_SWEEP, as
    # the k grid unions its primary, so the eager points are always exact
    # members. Read by the size-cap sweeps and, with the size-cap sweep on, by
    # the add-on sweeps; a run without the size-cap sweep never reads a
    # point's stamped cap here.
    caps = (tuple(sorted(set(SIZE_CAP_SWEEP) | {primary.size_cap}))
            if cap_sweep else ())
    capped = None
    if cap_sweep:
        # Every other cap, simulated only when a reader asks for a cell. Built
        # after the tier-off family and over the tier-on entries and seeds
        # alone (the family gets its own, below).
        capped = CapSweep(caps=caps, primary_cap=primary.size_cap, bands=tuple(bands),
                          ks=tuple(grid), primary_k=effective_k, start_date=start_date,
                          initial_balance=initial_balance, split_date=split_date,
                          checks=band_sweep, entries_by_band=entries_by_band,
                          st_entries=st_entries, eager=eager,
                          same_title_eager=same_title_point)
        logging.info("Size-cap sweep: %d caps (%s) x %d band(s) x %d k, simulated on "
                     "demand, one (band, k) cell at a time, when a report reads them",
                     len(caps),
                     ", ".join(_cap_label(c) for c in caps), len(bands), len(grid))

    off_capped = None
    if cap_sweep and tier_off_bands:
        # The same caps over the tier-floors-off family: its binding bands'
        # tier-off entries (a same-title population of its own would repeat
        # the tier-on one's — the same entries, which never read the tiers —
        # so st_entries is []), seeded from the family's own points only,
        # with the band sweep's checks (the family exists only on one). It
        # keeps tier_off_entries alive for the reader — see CapSweep's
        # retention note.
        off_capped = CapSweep(caps=caps, primary_cap=primary.size_cap,
                              bands=tuple(tier_off_bands), ks=tuple(grid), primary_k=effective_k,
                              start_date=start_date, initial_balance=initial_balance,
                              split_date=split_date, checks=True,
                              entries_by_band=tier_off_entries, st_entries=[], eager=off_eager,
                              same_title_eager=None, tier_floors=False)
        logging.info("Tier floors off: size-cap sweep: %d caps x %d binding band(s) x %d k, "
                     "simulated on demand, one (band, k) cell at a time, when a report "
                     "reads them", len(caps), len(tier_off_bands), len(grid))

    add_on_capped = add_on_off_capped = None
    if add_on_sweep:
        # The same grid with every simulation adding to held pairs: the size
        # cap sweep's caps when it ran, else the run's own cap. Only the "all"
        # population is read (checks off, no same-title population of its
        # own, since "all" already holds the same-title entries), and no eager
        # point exists to start from, so eager is empty and each cell takes
        # its curves' last day from the end-date map built above.
        add_on_caps = caps if cap_sweep else (primary.size_cap,)
        add_on_capped = CapSweep(
            caps=add_on_caps, primary_cap=primary.size_cap, bands=tuple(bands),
            ks=tuple(grid), primary_k=effective_k, start_date=start_date,
            initial_balance=initial_balance, split_date=None, checks=False,
            entries_by_band=entries_by_band, st_entries=[], eager={},
            add_to_held=True, end_dates=add_on_end_dates)
        logging.info("Adding to held pairs: %d size cap(s) x %d band(s) x %d k, all trades "
                     "together, each simulated when the dashboard reads it",
                     len(add_on_caps), len(bands), len(grid))
        if tier_off_bands:
            add_on_off_capped = CapSweep(
                caps=add_on_caps, primary_cap=primary.size_cap,
                bands=tuple(tier_off_bands), ks=tuple(grid), primary_k=effective_k,
                start_date=start_date, initial_balance=initial_balance, split_date=None,
                checks=False, entries_by_band=tier_off_entries, st_entries=[], eager={},
                tier_floors=False, add_to_held=True, end_dates=add_on_off_end_dates)
            logging.info("Adding to held pairs, tier floors off: %d size cap(s) x %d band(s) "
                         "the tiers bind at x %d k, each simulated when the dashboard "
                         "reads it", len(add_on_caps), len(tier_off_bands), len(grid))

    sold = None
    if sell_sweep:
        # Every sell level and minimum of days over the same grid as the
        # add-on family, tier floors on and off, adding to held pairs or not;
        # nothing simulated here
        sell_caps = caps if cap_sweep else (primary.size_cap,)
        sold = SellSweep(
            levels=_resolve_sell_levels(), min_days=_resolve_min_days_options(),
            caps=sell_caps, primary_cap=primary.size_cap, bands=tuple(bands),
            off_bands=tuple(tier_off_bands), ks=tuple(grid), primary_k=effective_k,
            start_date=start_date, initial_balance=initial_balance,
            entries_by_band=entries_by_band, off_entries_by_band=tier_off_entries,
            end_dates=add_on_end_dates, off_end_dates=add_on_off_end_dates)
        logging.info("Selling early: %d level(s) x %d minimum-days option(s) x %d size cap(s) "
                     "x %d band(s) (%d with the tier floors off) x %d k, adding to held "
                     "pairs or not, each simulated when the dashboard reads it",
                     len(sold.levels), len(sold.min_days), len(sell_caps), len(bands),
                     len(tier_off_bands), len(grid))

    return BacktestSweep(
        primary=primary, points=points, calibration=calibration,
        label_coverage=candidates.label_coverage,
        scenarios=scenarios,
        same_title_point=same_title_point,
        calibrations_by_band=calibrations_by_band,
        same_event_ladders=ladders,
        split_date=split_date,
        # One fact about the one corpus, like label_coverage: the header's
        # corpus line and the run's closing line read it (DR-13)
        corpus_provenance=candidates.corpus_provenance,
        config_same_event_ladders=config_ladders,
        tier_off_scenarios=tier_off_scenarios,
        tier_off_calibrations_by_band=tier_off_calibrations,
        cap_sweep=capped,
        tier_off_cap_sweep=off_capped,
        add_on_cap_sweep=add_on_capped,
        add_on_tier_off_cap_sweep=add_on_off_capped,
        # Resolved as every simulation, lazy cap cells included, resolves it
        same_title_size_cap=_resolve_same_title_size_cap(),
        # The saved live defaults' rule, filter, k and caps, from the one read above
        **_live_rule_fields(live),
        # For the dashboard header: the schedule every entry pass scanned at
        entry_checkpoint=SCHEDULED_RUN.label(),
        sell_sweep=sold,
        # Where every trade's synthetic book came from, for the page's header
        depth_model=depth_model,
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
    tier_off_sweep: bool = False,
    cap_sweep: bool = False,
    add_on_sweep: bool = False,
    sell_sweep: bool = False,
    *,
    depth_model: DepthModel | None = None,
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
    points) — the backtest-only scenario explorer. run_backtest() remains the
    two-tuple entry point; this is what backtest.py calls when it needs the
    sweep payload.

    It is the composition _prepare_candidates() + _sweep_from_candidates().
    The expensive half runs ONCE: _prepare_candidates() (fetch, prefilter,
    grouping, pair extraction, candlesticks) depends on neither the band nor
    k. Only the _find_entry pass is repeated per band (the band acts there and
    nowhere else), and only _simulate_at_discount() — Kelly gate, dedups,
    Pass 2, equity curve — per simulated scenario; it must be a full
    re-simulation rather than a re-score: k decides which candidates exist,
    and they then compete for the same simulated cash and ladders.
    _interval_calibration() is computed once per band.

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

    With tier_off_sweep (a band sweep required) every band where a
    deadline-gap tier floor binds — on the shipped grid the 18 bands with a
    floor below 0.30 — is entered and simulated a second time with the tiers
    not applied: that band's floor alone gates pB − pA (which must still be
    strictly positive) and sets the leg-price-sum ceiling (1 − floor), and
    nothing else moves (the 30-day gap cap, the band ceiling, the live-quote
    and fee checks, the Kelly gate, ladders, same-title pairs and the primary
    scenario). The family comes
    back as BacktestSweep.tier_off_scenarios and
    tier_off_calibrations_by_band — backtest-only reporting data, like the
    band sweep's own — and leaves every tier-on figure exactly as it is
    without it. It is opt-in here (False by default, the band_sweep
    pattern); backtest.py turns it on together with the band sweep.

    This function never writes config.py or the saved live defaults. The
    calibration it reports is a recommendation for a human to act on: live
    sizing keeps reading the saved live defaults' k (or main.py's
    --interval-discount) whatever is passed here; no band or tier setting
    here reaches live.

    It reports the saved live defaults' time-series rule, category/tag filter,
    k and caps: one fail-soft read before the fetch, recorded on the nine
    live_* fields and, worded by _live_rule_line, logged last ("none
    recorded" when no usable defaults are saved).

    With add_on_sweep, the result also carries the dashboard's "Add to held
    pairs" family (BacktestSweep.add_on_cap_sweep, and
    add_on_tier_off_cap_sweep over the tier-floors-off family): lazy CapSweeps
    over the run's grid whose every simulation adds to held pairs, for the
    "all" population only. Nothing is simulated during the run, so every
    point, figure and completion line is what the run reports without the
    flag; the run logs one setting line for it and, from
    _sweep_from_candidates, one summary line per sweep.

    With sell_sweep, the result also carries the dashboard's "Sell" family
    (BacktestSweep.sell_sweep): every level of config.TAKE_PROFIT_LEVELS, each
    with every minimum of days before maturity of config.TAKE_PROFIT_MIN_DAYS,
    over the same grid, tier floors on and off, adding to held pairs or not,
    simulated only when the dashboard reads it. The run logs one setting line
    for it and, from _sweep_from_candidates, one summary line.

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
            point, in (0, 1]. None (default) means "no override", which
            resolves to config.TIME_SERIES_INTERVAL_PROB_DISCOUNT (config.py's
            k, not necessarily the saved live defaults' k live sizing reads).
        sweep (bool): When True (default), also simulate every discount in
            config.INTERVAL_DISCOUNT_SWEEP. When False, points holds the
            primary alone (and a band sweep simulates each band at the
            primary k only) — the escape hatch for a full-history run where
            the extra passes are not worth their time.
        same_event_ladders (bool | None): Whether two dated cumulative rungs
            of ONE event may pair for this run. None (the default)
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
            BacktestSweep.same_event_ladders, alongside the configured switch
            itself (BacktestSweep.config_same_event_ladders) — the ladder log
            line below names whether this run's resolved value replays that
            switch or departs from it.
        spread_band (tuple[float, float] | None): The primary scenario's
            BACKTEST-only time-series spread band (floor, ceiling) on pB − pA.
            None (default) resolves config.BACKTEST_DEFAULT_SPREAD_BAND —
            (0.0, 1.0), no band. Resolved and validated at the
            TOP of this function, before anything is logged or fetched, so a
            bad band fails in milliseconds rather than after the fetch.
        band_sweep (bool): When True, also sweep every band of the config
            grid and compute BacktestSweep.scenarios, same_title_point,
            split_date and every band's calibration. False (default) sweeps
            k at the primary band alone.
        tier_off_sweep (bool): When True (band_sweep required), also compute
            the tier-off family above — BacktestSweep.tier_off_scenarios and
            tier_off_calibrations_by_band. False (default) computes none of
            it. Validated at the TOP of this function, beside the band,
            before anything is logged or fetched.
        cap_sweep (bool): When True, also return BacktestSweep.cap_sweep — a
            lazy CapSweep over SIZE_CAP_SWEEP (the per-trade Kelly size caps
            5%..95% and no cap; same-title pairs stay under
            SAME_TITLE_SIZE_CAP at every cap) that simulates each other cap
            only when a reader asks for a (band, k) cell. Every point this function
            returns is still sized at the run's own cap
            (config.BUDGET_FRACTION), and the flag adds no simulation to the
            run itself. With tier_off_sweep too, also
            BacktestSweep.tier_off_cap_sweep — the same over the
            tier-floors-off family, seeded from its own points. False
            (default) returns cap_sweep=None and tier_off_cap_sweep=None, as
            does the infeasible window. Backtest-only: live sizing never reads
            either.
        add_on_sweep (bool): When True, also return
            BacktestSweep.add_on_cap_sweep — the "Add to held pairs" family:
            a lazy CapSweep over the run's grid (every cap of SIZE_CAP_SWEEP
            with cap_sweep, the run's own cap alone without) whose every
            simulation adds to held pairs, for the "all" population only —
            and, with tier_off_sweep too, add_on_tier_off_cap_sweep over the
            tier-floors-off family's binding bands. The flag adds no
            simulation to the run itself. False (default) returns both as
            None, as does the infeasible window. Backtest-only: live sizing
            never reads either.
        sell_sweep (bool): When True, also return BacktestSweep.sell_sweep —
            a lazy SellSweep over every level of config.TAKE_PROFIT_LEVELS and
            every minimum of days of config.TAKE_PROFIT_MIN_DAYS. The flag
            adds no simulation to the run itself. False (default)
            returns None, as does the infeasible window. Backtest-only: live
            trading never sells.
        depth_model (DepthModel | None): Keyword-only. The depth model every
            trade's synthetic order book is built from
            (depth_model.load_depth_model()); None (default) fills every trade
            at the top of the book. Recorded on BacktestSweep.depth_model and
            named on one log line.

    Returns:
        BacktestSweep: primary (the effective-discount, primary-band result),
            points (the primary band's k sweep, ascending, always containing
            primary), calibration (the primary band's; None when no
            time-series candidate was measurable), label_coverage (the run's
            outcome-label census, None when the feasibility short-circuit
            skipped the fetch), and the band-sweep payload — scenarios,
            same_title_point, calibrations_by_band, split_date — plus the
            resolved same_event_ladders, the configured switch it is judged
            against (config_same_event_ladders), the corpus's provenance,
            None when not recorded, the tier-off family, empty unless
            tier_off_sweep, the lazy cap_sweep and tier_off_cap_sweep, the
            lazy add-on family (add_on_cap_sweep and
            add_on_tier_off_cap_sweep), same_title_size_cap, the live_*
            fields, entry_checkpoint, the lazy sell_sweep and depth_model —
            see BacktestSweep.

    Raises:
        ValueError: Before any fetch or log line, if tier_off_sweep is set
            without band_sweep, if interval_discount is not a number in
            (0, 1], if this module's SAME_TITLE_SIZE_CAP is not on the 5%
            grid in (0, 1], if sell_sweep is set and this module's
            TAKE_PROFIT_HOLD_DAYS is not a whole number from 1 to
            _HOLD_DAYS_MAX, its TAKE_PROFIT_LEVELS is not a non-empty
            collection of distinct real numbers in (0, 1] or its
            TAKE_PROFIT_MIN_DAYS is not a non-empty collection of distinct
            whole numbers of at least 1, or if
            spread_band is not a valid band; and,
            before any fetch, from _prepare_candidates() when SCHEDULED_RUN
            cannot place the entry checkpoints (a configuration error).
        TypeError: From config.time_series_spread_band, before any fetch, if
            spread_band is not a pair of numbers.
        KeyError: Propagates out of the candlestick-fetch pool
            (_fetch_candles_parallel) if a ticker needed by a candidate pair
            was not properly excluded by the eligibility prefilter — a real
            defect rather than a ticker with no price history.

    Note:
        When _prepare_candidates()'s feasibility pre-check fails, no
        simulation is possible at any discount or band: the result is a sweep
        holding one empty point (built by the same _simulate_at_discount()
        call every other point comes from, over an empty entry list, so its
        shape, its resolved k and its band stamp cannot drift from a real
        one), calibration=None, label_coverage=None, scenarios=[],
        calibrations_by_band={}, an empty tier-off family and the resolved
        same_event_ladders, with the configured switch
        (config_same_event_ladders), same_title_size_cap, the entry checkpoint
        and, when usable live defaults are saved, the live_* fields recorded,
        and the live-rule line logged either way, so callers need no special
        case. No add-on or sell family is built there (all three are None).

        Before the fetch, one INFO line names the entry checkpoint and its UTC
        times over [start_date, today]; it never raises. _prepare_candidates
        then raises ValueError for a bad schedule, or OverflowError for a
        start_date whose next run weekday is past date.max.
    """
    # Resolved and validated FIRST — before anything is logged or fetched: an
    # invalid band is a caller bug, and it must surface in milliseconds, not
    # after a multi-hour fetch. config owns the default and the rule.
    primary_band = time_series_spread_band(spread_band)
    # The same for the tier-off family, a band-sweep family: without the band
    # sweep there is no grid, split date or rescan for it to re-run
    if tier_off_sweep and not band_sweep:
        raise ValueError("tier_off_sweep needs band_sweep: the tier-off family "
                         "re-runs the band sweep's binding bands")
    # The same for the extra same-title cap: refused here, not after the fetch
    same_title_cap = _resolve_same_title_size_cap()
    # ... and, when the sell family rides the result, for the sell rule's day
    # count, the sell levels and the minimum-days options, which the
    # dashboard otherwise reads only after the run
    hold_days = _resolve_hold_days() if sell_sweep else None
    if sell_sweep:
        _resolve_sell_levels()
        _resolve_min_days_options()
    # And for k: every simulation hands it to the live sizer, whose
    # LiveSettings takes only a k in (0, 1]
    if interval_discount is not None and (
            isinstance(interval_discount, bool)
            or not isinstance(interval_discount, numbers.Real)
            or not 0.0 < interval_discount <= 1.0):
        raise ValueError(f"interval_discount (k) must be in (0, 1], got {interval_discount!r}")

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
    # ... and judged against the switch this checkout's live finder runs with:
    # an override the other way does not replay that rule, and the line says so.
    config_ladders = bool(TIME_SERIES_SAME_EVENT_LADDERS)
    if same_event_ladders is None:
        source = "config.TIME_SERIES_SAME_EVENT_LADDERS"
    elif bool(ladders) == config_ladders:
        source = "run-level override, same as this checkout's config"
    else:
        source = ("run-level override; config.TIME_SERIES_SAME_EVENT_LADDERS is "
                  f"{'on' if config_ladders else 'off'} in this "
                  "checkout, so this run does not replay its live rule")
    logging.info(
        "Same-event deadline ladders (DR-73): %s (%s)",
        "on" if ladders else "off", source,
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
    # And for the per-trade size cap: the cap every point below is sized
    # under (this module's BUDGET_FRACTION, resolved at call time, so a
    # monkeypatched value is the one printed), the same-title cap only when it
    # binds tighter, and whether the lazy size-cap sweep rides the result.
    run_cap = _resolve_size_cap(None)
    same_title_clause = (
        f", same-title never above {_cap_percent(same_title_cap)}% "
        "(config.SAME_TITLE_SIZE_CAP)"
        if same_title_cap < run_cap else "")
    logging.info(
        "Per-trade size cap (backtest): %s%% (config.BUDGET_FRACTION)%s; size-cap sweep %s",
        _cap_percent(run_cap),
        same_title_clause,
        "on" if cap_sweep else "off",
    )
    # And the entry checkpoint, with its UTC times over [start_date, today].
    # This line never raises (it reports the times as not computable);
    # _prepare_candidates just below refuses a bad schedule.
    try:
        times = _checkpoint_utc_times(start_date, datetime.now(UTC).date())
    except (ZoneInfoNotFoundError, ValueError, OSError, OverflowError) as exc:
        where = f"UTC times not computable ({type(exc).__name__})"
    else:
        where = (f"{'/'.join(times)} UTC in this window" if times
                 else "no checkpoint in this window")
    logging.info(
        "Entry checkpoint (backtest): %s (config.SCHEDULED_RUN, the live scheduler's "
        "run time): %s",
        SCHEDULED_RUN.label(), where,
    )
    # And for the add-on family: it is simulated when the dashboard reads it,
    # never during the run, so the line says whether it rides the result
    logging.info(
        "Adding to held pairs (backtest): %s",
        "on — the dashboard's Add to held pairs select is simulated when the "
        "dashboard is built" if add_on_sweep else
        "off — the dashboard's Add to held pairs select stays disabled",
    )
    # And for the sell family, the same way, with when a position is sold
    logging.info(
        "Selling early (backtest): %s",
        f"on — a position is sold {_hold_days_text(hold_days)} "
        "(config.TAKE_PROFIT_HOLD_DAYS); the dashboard's Sell select is simulated "
        "when the dashboard is built"
        if sell_sweep else "off — the dashboard's Sell select stays disabled",
    )
    # And the depth model this run builds every trade's synthetic book from
    # (load_depth_model's own line says it was loaded)
    if depth_model is None:
        logging.info("Depth model: none — every trade fills at the top of the book")
    else:
        logging.info("Depth model: %d snapshot(s), %d ladders, taken %s to %s",
                     depth_model.snapshots, depth_model.ladders,
                     depth_model.first_taken or "unknown", depth_model.last_taken or "unknown")
    # The saved live defaults (never main.py's per-run overrides), read ONCE
    # before the fetch, so a refused file warns at the top of the run
    live = _live_settings_for_report()

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
        # empty scenarios list here apart from a band sweep that was off. No
        # corpus was fetched either, so there is no provenance to report: the
        # header says "not recorded" rather than inventing one.
        result = BacktestSweep(primary=empty, points=[empty], calibration=None,
                               label_coverage=None, scenarios=[],
                               calibrations_by_band={},
                               same_event_ladders=bool(ladders),
                               corpus_provenance=None,
                               config_same_event_ladders=config_ladders,
                               same_title_size_cap=same_title_cap,
                               # For the dashboard header (config.ScheduledRun.label)
                               entry_checkpoint=SCHEDULED_RUN.label(),
                               depth_model=depth_model,
                               **_live_rule_fields(live))
    else:
        # Every entry pass and simulation. It deletes the candles and pair list
        # after the last entry pass, so holding `candidates` pins neither
        result = _sweep_from_candidates(
            candidates, initial_balance,
            interval_discount=interval_discount, sweep=sweep,
            spread_band=primary_band, band_sweep=band_sweep,
            tier_off_sweep=tier_off_sweep, cap_sweep=cap_sweep,
            add_on_sweep=add_on_sweep, sell_sweep=sell_sweep, live=live,
            depth_model=depth_model,
        )
    # On every path, "none recorded" included, so the report never omits it
    logging.info("%s", _live_rule_line(result))
    return result


# ─── Equity curve construction ────────────────────────────────────────────────

def _open_value_path(trade: BacktestTrade) -> np.ndarray | None:
    """
    What an open trade is worth at the end of each day it is held: the one day-end path.

    One value per UTC day from the entry day to the day before its pay-out
    day, each n x (the day-end value of each leg): the latest usable ask of
    the side it holds at that day's end, its payout once its market has paid
    out, or the price it paid (_paid_prices) before that side has had any
    usable ask (LegQuotes.day_values). _carry_steps and
    _value_steps are built from it, so the equity curve, the dashboard's
    per-type lines and its risk-free hurdle all read the same path.

    Args:
        trade (BacktestTrade): A completed trade.

    Returns:
        np.ndarray | None: One float per held day; None for a trade with no
            quotes (valued at cost) or one that pays out on its entry day.
    """
    if trade.marks is None:
        return None
    held_days = (trade.exit_date - trade.entry_date).days
    if held_days <= 0:
        return None
    quotes_a, quotes_b = trade.marks
    # Which side each leg holds (scanner.leg_sides) and what it paid
    side_a, side_b = leg_sides(trade.pair_type)
    price_a, price_b = _paid_prices(trade)
    last = trade.exit_date - timedelta(days=1)
    return trade.n * (quotes_a.day_values(trade.entry_date, last, side_a, price_a)
                      + quotes_b.day_values(trade.entry_date, last, side_b, price_b))


def _path_steps(trade: BacktestTrade, values: list[float], first: float,
                last: float) -> tuple[tuple[date, float], ...]:
    """
    Turn a day-end path into dated steps: `first` on entry, each day's change, `last` on exit.

    A day whose value did not change adds no step.

    Args:
        trade (BacktestTrade): The trade the path belongs to.
        values (list[float]): Its day-end values (_open_value_path), entry day first.
        first (float): The step on the entry day.
        last (float): The step on the exit day.

    Returns:
        tuple[tuple[date, float], ...]: (day, amount) pairs in date order.
    """
    steps = [(trade.entry_date, first)]
    for i in range(1, len(values)):
        change = values[i] - values[i - 1]
        if change:
            steps.append((trade.entry_date + timedelta(days=i), change))
    steps.append((trade.exit_date, last))
    return tuple(steps)


def _carry_steps(trade: BacktestTrade) -> tuple[tuple[date, float], ...]:
    """
    What the portfolio carries in an open trade, as dated steps: its position on the curve.

    The steps add up, day by day, to the trade's day-end value while it is
    open (_open_value_path) and to zero from its pay-out day on. With no
    quotes they are exactly ((entry_date, total_cost), (exit_date,
    -total_cost)): the trade carried at its cost.

    Args:
        trade (BacktestTrade): A completed trade.

    Returns:
        tuple[tuple[date, float], ...]: (day, amount) pairs in date order.
    """
    path = _open_value_path(trade)
    if path is None:
        return ((trade.entry_date, trade.total_cost), (trade.exit_date, -trade.total_cost))
    values = path.tolist()
    return _path_steps(trade, values, values[0], -values[-1])


def _value_steps(trade: BacktestTrade) -> tuple[tuple[date, float], ...]:
    """
    What an open trade adds to the portfolio value each day, as dated steps.

    Cash falls by total_cost + fees on the entry day while the position
    enters at its day-end value; each later day adds the change in that
    value; the pay-out day adds the receipt less the last value. The steps
    sum to the trade's profit. With no quotes they are exactly ((entry_date,
    -fees), (exit_date, actual_payoff - total_cost)).

    Args:
        trade (BacktestTrade): A completed trade.

    Returns:
        tuple[tuple[date, float], ...]: (day, amount) pairs in date order.
    """
    path = _open_value_path(trade)
    if path is None:
        return ((trade.entry_date, -trade.fees),
                (trade.exit_date, trade.actual_payoff - trade.total_cost))
    values = path.tolist()
    return _path_steps(trade, values, (values[0] - trade.total_cost) - trade.fees,
                       trade.actual_payoff - values[-1])


def _build_equity_curve(
    trades: list[BacktestTrade],
    start_date: date,
    initial_balance: float,
    *,
    end_date: date | None = None,
) -> pd.DataFrame:
    """
    Construct a daily equity curve DataFrame from the list of backtest trades.

    "portfolio_value" is a PORTFOLIO VALUE, not a cash balance: it is cash plus
    the value of every position still open at that day's end. An open trade
    is valued AT MARKET: each leg at the latest usable ask of the side it
    holds at the end of the day (on a candle ending at or before the next UTC
    midnight; LegQuotes), at its payout once its market has paid out, and at
    the price it paid before that side has had any usable ask — the same
    valuation Pass 2 sizes on, read at the
    day's end rather than at the checkpoint (_open_value_path, through
    _carry_steps). A trade with no quotes (trade.marks is None, every
    hand-built trade) is carried at its COST, which leaves it exactly two
    moves:
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
    A quoted trade moves the curve on every day its value changes as well,
    but its steps still sum to its own profit, so the curve's endpoints — the
    start plus every trade's profit — do not depend on the marks.

    The value at market is the ask, which overstates what a leg would sell
    for by the spread; a leg that stops trading keeps its last quote (there
    is no age limit) until it pays out, so it shows a flat line and then a
    jump. Swings inside a day are not seen.

    This matters because the curve is the sole input to every risk figure on the
    dashboard: the "Max Drawdown" KPI, the "Drawdown (%)" chart, _sharpe and
    _sortino (which read the derived "daily_return" column), the per-k sweep
    table's drawdown and Sharpe columns, and the benchmark row that sits in the
    same column as ^GSPC's own daily drawdown. Carrying an open position at
    zero (a cash-only curve) would read every trade's deployment as a loss.

    The curve opens one day before start_date at the untouched initial balance,
    so a trade entering on start_date itself shows its day-0 cost as a real
    pct_change and a real decline from the cummax peak, and the performance
    card and the per-k sweep table both divide by that same untouched opening.

    Args:
        trades (list[BacktestTrade]): Completed backtest trades with entry_date,
            exit_date, total_cost, fees and actual_payoff populated.
        start_date (date): The first TRADING date of the window; the curve opens
            one row earlier, on start_date - 1 day, at the untouched initial
            balance.
        initial_balance (float): Starting portfolio value in dollars.
        end_date (date | None): Keyword-only. The day the curve runs to in
            place of today (UTC). None (default) reads today at call time.
            CapSweep passes the last day of the eager point a lazy size-cap
            run belongs to, which is today (UTC) as it stood when that eager
            point was simulated, or start_date itself for a future window —
            either way the same axis the eager curve has.

    Returns:
        pd.DataFrame: DataFrame with one leading row for start_date - 1 day at
            the initial balance, followed by one row per calendar day from
            start_date to today (UTC), or to end_date when one is given — and,
            when start_date is itself after that day, exactly those two rows —
            with columns:
            - "date" (date): Calendar date.
            - "portfolio_value" (float): Cash plus open positions at market
              (at cost for a trade with no quotes), in dollars (see above).
            - "daily_return" (float): Fractional daily return (pct_change of portfolio_value).
            Never zero rows: a column-less DataFrame would violate this contract
            and crash the "daily_return" assignment below, as well as every
            .iloc[0]/.iloc[-1] read in dashboard.py.
    """
    # entry_date and exit_date come from UTC-derived timestamps, so use UTC today
    # here as well — otherwise `date.today()` in a non-UTC timezone can drop or add
    # a day around the boundary and misalign the equity curve. A caller may pin
    # the day instead (CapSweep's lazy runs pin their eager point's), so a
    # simulation that runs after UTC midnight ends where its sibling did.
    today = datetime.now(UTC).date() if end_date is None else end_date
    # Floored at 1: a start_date after today (reachable through run_backtest /
    # run_backtest_sweep, whose Monday-feasibility short-circuit builds an empty
    # curve for whatever window it was handed) makes the raw span zero or
    # negative. The leading row below already keeps the frame from being the
    # column-less pd.DataFrame([]), so the floor is what guarantees start_date
    # itself is on the axis — the documented future-window shape is exactly the
    # leading row plus start_date.
    span_days = max((today - start_date).days + 1, 1)
    # The curve opens one day BEFORE start_date at the untouched initial
    # balance. start_date can carry an entry (when it is a run weekday), and
    # that day's charges must land on a row after the first, or pct_change and
    # cummax never see them. No trade can enter before start_date, so the
    # leading row is always flat.
    dates = [start_date - timedelta(days=1)] + [
        start_date + timedelta(days=i) for i in range(span_days)
    ]

    # Two accumulators, because a portfolio is cash PLUS whatever is still
    # open. Tracking cash alone would carry every open position at zero,
    # which makes the curve dive on entry and recover at settlement whether
    # the trade won or lost — deployment reported as drawdown.
    cash_changes: dict[date, float] = defaultdict(float)
    position_changes: dict[date, float] = defaultdict(float)
    for t in trades:
        # Cash leaves the portfolio on entry day (contract cost + taker fees)
        # and the gross receipt comes back on the pay-out day
        cash_changes[t.entry_date] -= t.total_cost + t.fees
        cash_changes[t.exit_date]  += t.actual_payoff
        # ...while the contracts it bought are an ASSET held until then: they
        # enter the portfolio at their day-end value, move with it each day
        # and leave it on the pay-out day (at cost, entry and exit only, for
        # a trade with no quotes). Fees are never part of that value: they
        # are gone the moment the order fills.
        for when, amount in _carry_steps(t):
            position_changes[when] += amount

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
