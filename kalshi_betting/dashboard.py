"""
File: dashboard.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Generates a self-contained interactive HTML performance dashboard from the
    results of a backtest run. Assembles nine sections — portfolio performance
    (equity curve, Sharpe, drawdown), returns decomposition (by month, category,
    entry price), calibration analysis (Brier score, reliability diagram),
    interval-discount (k) calibration (empirical k-hat plus a native Plotly
    dropdown that switches the equity curve between the swept k values),
    empirical k-hat broken down by Kalshi category, tag and spread band (a bar
    chart and a table with a "Group by" <select> of their own), a
    scenario explorer (a fragility banner, a spread-band x k heatmap and a
    per-population KPI table over BacktestSweep.scenarios, with two <select>s
    and a short inline script driving a Plotly.restyle'd equity curve — a
    native updatemenus dropdown cannot express two independent axes of
    selection), trade-level diagnostics
    (distribution, slippage, best/worst trades), risk metrics (Kelly sizing
    scatter, capital deployment), and benchmark comparison (S&P 500 via
    yfinance) — into a single HTML file with embedded Plotly charts. The file
    is written to PROJECT_ROOT and can be opened directly in any browser.

    A sticky filter bar at the top of the page — Spread band, Tier floors,
    Category, Tag — re-scopes every trade-derived section (performance,
    decomposition, calibration, diagnostics, risk, the benchmark's strategy
    row) to the run at another spread band, with the deadline-gap tier floors
    on (the run as simulated) or off (the band sweep's tier-floors-off run of
    that band), and/or one Kalshi category or category · tag of it, and
    moves the k-hat breakdown to the same band, tier setting and selection.
    Every figure a selection shows is computed here in Python by the
    helpers the sections themselves render with (_filter_payload), packed
    into one gzip + base64 data block, and swapped in by a small inline
    script (_FILTER_JS) that draws nothing of its own.

Dependencies:
    Imports BacktestSweep, BacktestTrade, CorpusProvenance (historical.py's,
    re-exported by backtester.py), IntervalCalibration, OutcomeLabelCoverage
    and SweepPoint from backtester.py, plus its _calibration_bucket() (the one
    definition of the k-hat arithmetic, which the k-hat breakdown runs over
    each category's, tag's or band's observations), _exact_label() — the
    injective float formatter its completion lines use, reused so no two
    scenario-explorer labels can collide — _build_equity_curve() (the one
    definition of an equity curve, which the page-wide filter runs over a
    category's or tag's trades alone for that slice's attributed curve),
    _leg_prices_for(), max_trades_simulated() (the one test of a carried
    post-cutoff verdict against the run's own trades, shared with
    backtest.py's closing line), _band_label() (the bare "floor-ceiling" a
    tier-floors-off run is labelled with, since its floor alone gated it) and
    _tier_floors_bind() (the one test of whether a deadline-gap tier sits
    above a band's floor: for a band ABSENT from the tier-off family it
    decides whether that band's tier-on run may stand in for its
    tier-floors-off view, or the whole off view is withheld — the family's
    calibration keys, never this test, decide which bands were simulated
    again), and
    BACKTEST_OUTCOME_LABEL_WARN_FRACTION, PROJECT_ROOT,
    SAME_TITLE_CO_RESOLVE_PROB, CALENDAR_DAYS_PER_YEAR, TRADING_DAYS_PER_YEAR,
    MIN_PRICE_DIFF_SHORT_GAP, MIN_PRICE_DIFF_LONG_GAP, SHORT_DEADLINE_GAP_DAYS
    and MAX_DEADLINE_GAP_DAYS (so the filter bar names the tier floors from
    config, never as literals), fee_per_pair_approx() and
    time_series_profit_prob() from config.py — the latter is the single
    definition of the time-series Kelly probability shared with strategy.py
    and backtester.py, so the Kelly scatter here shows the same fraction the
    live sizer computes. Uses plotly, numpy, pandas, and yfinance (all
    external). Called by backtest.py after run_backtest_sweep() completes.

Notes:
    The HTML file loads Plotly.js from the CDN (cdn.plot.ly), so an internet
    connection is required to view the charts. If yfinance fails to fetch S&P 500
    data (e.g. network unavailable), the benchmark section degrades gracefully
    and shows only the strategy equity curve.

    _sharpe/_sortino annualize on whatever periodicity the caller names. The
    strategy equity curve is CALENDAR-daily (backtester._build_equity_curve),
    so every call on that curve takes their CALENDAR_DAYS_PER_YEAR default; the
    ^GSPC benchmark row is the one TRADING-day series on the page and is the
    single site that passes TRADING_DAYS_PER_YEAR explicitly. The two must never
    share a factor — at rf = 0 the mismatch is exactly sqrt(365/252) = 1.2035 of
    magnitude.

    The interval-discount section's k selector is a NATIVE Plotly `updatemenus`
    dropdown over one trace per swept k — no extra dependency and no hand-rolled
    JavaScript, so it works inside the same self-contained page every other
    chart renders into. Its scope is deliberately that one section: the
    scenario-explorer section (below) carries its own independent band/k
    selectors, and the remaining seven — the six trade-derived sections and
    the k-hat breakdown — are rendered at the run's primary scenario (the
    k-hat breakdown is k-independent: the primary k is only its reference
    line): its primary k (the CLI's --interval-discount, or
    config.TIME_SERIES_INTERVAL_PROB_DISCOUNT when it was not passed) AND its
    primary spread band (--spread-min/--spread-max, or
    config.BACKTEST_DEFAULT_SPREAD_BAND when neither was passed) — until the
    page-wide filter bar re-scopes them.

    The page-wide filter bar (_filter_bar_html) is a third, separate set of
    controls, and the only one that reaches beyond its own section. Its
    spread band choice shows that band's OWN run at the primary k (the band
    sweep's "all" point there — a standalone simulation, so every figure is
    genuine). Its tier floors choice switches every band between that run
    (on: the deadline-gap tier floors applied, as the run simulated it) and,
    when the run carries the band sweep's tier-floors-off family
    (BacktestSweep.tier_off_scenarios), the band's run with the tier floors
    NOT applied (off: the band's own floor alone — its tier-off "all" point
    at the primary k, also a standalone simulation, labelled with the bare
    floor-ceiling since that floor alone gated it; _tier_off_runs). A band
    whose floor sits at or above both tiers is never simulated again,
    because the tiers never bind there (backtester._tier_floors_bind): its
    off view IS its tier-on run, and the summary line says so. A run without
    the family, or with one missing a binding band's run, has no off view at
    all — never a relabelled tier-on run: the select stays disabled, with a
    note beside it. Its category and tag choices show a SLICE of the run the
    band and tier choices name, whose
    figures drawn from an equity curve (return, drawdown, Sharpe, Sortino,
    the median monthly return, the benchmark's strategy row) come from the
    attributed curve — the starting balance plus the slice's P&L as the run
    booked it (backtester._build_equity_curve over the slice) — i.e. its
    contribution, not a standalone simulation, and the bar's summary line
    says so. The header's trade count follows the selection too. A tag is
    Kalshi's FIRST tag of the series (_series_labels), so every breakdown
    partitions. It never re-scopes the interval-discount section or the
    scenario explorer, which keep their own controls; the bar names them.
    Every view is computed by the same helpers the sections render with
    (_performance_kpis, _performance_series, _decomposition_aggregates,
    _category_table, _reliability, _best_and_worst, _kelly_points,
    _capital_deployed, _strategy_row), so the script only draws what Python
    computed. The page as rendered IS the primary band's unfiltered view with
    the tier floors on: the
    script inflates the data block as the page loads, sets the bar back to
    that view (a browser can restore a stale choice on reload) and keeps its
    selects disabled until the data is ready, and redraws only when a
    <select> changes — each chart from its layout as Python drew it, so a
    zoom never carries over into another selection. A failure to build the
    payload costs the bar, never the page: the page is written without it,
    with a notice in its place and a WARNING in the log.

    The k-hat breakdown (_section_khat) is the one chart that follows the
    filter bar without describing trades. It regroups each band's carried
    k-hat population — IntervalCalibration.observations: every time-series
    candidate ENTRY at the band, measured before the Kelly gate and
    independent of k, so it covers entries the run never traded — through
    backtester._calibration_bucket, the one definition of k-hat, filed on
    market A's event ticker by _series_labels, the rule trades are filed by,
    so a category's k-hat and its trades describe the same events. Its own
    "Group by" <select> picks the axis (category, tag or spread band); the
    filter bar picks the band, its tier floors setting (a tier-off band
    regroups its tier-off run's calibration) and the category or tag, and
    grouping by one of them shows every value of it with the selection
    highlighted. Each bar
    states its entries and the distinct events behind them. Every figure and
    word it shows comes from Python (_khat_stat's pre-formatted cells, bar
    labels and hover figures, the _KHAT_TEXT templates), so neither the script
    nor Plotly's hover can round or word one differently. A band whose calibration does not carry its population
    (len(observations) != pooled.n — a hand-built one) offers its pooled row
    alone (_khat_band), since there is nothing to break down.

    The scenario-explorer section's band x k grid is too large, and its two
    axes of selection too independent, for the same native-dropdown idiom: a
    Plotly `updatemenus` button can only toggle trace VISIBILITY or REPLACE a
    trace's data wholesale from a fixed list baked in at render time, not
    combine two independently-chosen indices into one lookup. It therefore
    carries its own small (~100-line) inline vanilla-JS script that reads one
    `<script type="application/json">` data block and drives a `Plotly.restyle`
    call plus two plain HTML table re-renders — no new dependency, and no
    hand-rolled charting: Plotly still owns every pixel that gets drawn.

    Directly under the Period line the header says what settled-market corpus
    the run read (BacktestSweep.corpus_provenance): when it was assembled (a
    legacy settled_markets_*.json's file time, named as such) — the Period
    runs to today, the corpus only to that moment — whether it was served
    from an earlier run's cache, and the archive cutoff as of that assembly,
    with a red banner when the window starts at or after it and so could
    never enter a trade (DR-13, M2) — or, when some simulated point DID
    trade (backtester.max_trades_simulated), an amber line saying that
    verdict is stale instead. It renders on every run, "not recorded"
    included, never as a silence (DR-66).

    The page header also names the run's primary spread band and its same-event
    ladder setting (DR-73) under the Period line, or "not recorded" when the
    run passed no sweep: the ladder setting decides which pairs exist and the
    band which of them are ever entered, so, like DR-66b's strike-blind
    notice, they qualify every section rather than only the explorer.

    The scenario explorer's heatmap, fragility banner and equity curve read
    the "time_series" population — every time-series entry simulated alone,
    same-title excluded — never the "all" one, so a same-title result (band-
    and k-independent) cannot dilute the band x k comparison; "all" keeps its
    own labelled KPI row.
"""
import base64
import gzip
import html
import json
import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from plotly.subplots import make_subplots

from .backtester import (
    BacktestSweep,
    BacktestTrade,
    CorpusProvenance,
    IntervalCalibration,
    OutcomeLabelCoverage,
    SweepPoint,
    _band_label,
    _build_equity_curve,
    _calibration_bucket,
    _exact_label,
    _leg_prices_for,
    _tier_floors_bind,
    max_trades_simulated,
)
from .config import (
    BACKTEST_OUTCOME_LABEL_WARN_FRACTION,
    CALENDAR_DAYS_PER_YEAR,
    MAX_DEADLINE_GAP_DAYS,
    MIN_PRICE_DIFF_LONG_GAP,
    MIN_PRICE_DIFF_SHORT_GAP,
    PROJECT_ROOT,
    SAME_TITLE_CO_RESOLVE_PROB,
    SHORT_DEADLINE_GAP_DAYS,
    TRADING_DAYS_PER_YEAR,
    fee_per_pair_approx,
    time_series_profit_prob,
)
from .scanner import leg_sides

# The one dashboard file every backtest run writes (and overwrites) in PROJECT_ROOT.
DASHBOARD_FILENAME = "backtest_dashboard.html"

# ─── Metric computation ───────────────────────────────────────────────────────

def _sharpe(daily_returns: pd.Series, rf: float = 0.0, *,
            periods_per_year: int = CALENDAR_DAYS_PER_YEAR) -> float:
    """
    Compute the annualized Sharpe ratio from a series of per-period returns.

    Annualizes by multiplying the RATIO of the mean per-period excess return to
    its standard deviation by sqrt(periods_per_year) — not the mean alone, as
    an earlier version of this docstring claimed.

    `periods_per_year` governs BOTH halves of the annualization: it converts the
    annual `rf` hurdle into a per-period hurdle AND supplies the sqrt scaling
    factor. It must therefore match the actual periodicity of the series passed.
    Use CALENDAR_DAYS_PER_YEAR (365) for a backtester._build_equity_curve()
    output, which has one row per calendar day, and TRADING_DAYS_PER_YEAR (252)
    for a trading-day series such as the ^GSPC benchmark's pct_change. Mixing
    the two scales the calendar-day series' MAGNITUDE down by exactly
    sqrt(365/252) = 1.2035 at rf = 0, leaving the sign alone (at rf != 0 it is
    not a constant rescale, since the per-period hurdle moves too).

    The default is the CALENDAR base because four of the five calls to this
    helper and _sortino in this module consume _build_equity_curve output; the
    single trading-day consumer (_section_benchmark's ^GSPC row) passes
    TRADING_DAYS_PER_YEAR explicitly. The parameter is keyword-only so it can
    never be passed positionally into `rf`'s slot.

    Args:
        daily_returns (pd.Series): Series of per-period fractional returns
            (e.g. 0.01 for 1%).
        rf (float): Annual hurdle rate (the T-bill "rf" term of the Sharpe
            formula) as a decimal (e.g. 0.05 for 5%). Defaults to 0.0.
        periods_per_year (int): Periods per year in `daily_returns`. Must be
            positive; not validated, since every value that reaches it is a
            config constant — three of this helper's four in-module call sites
            take the CALENDAR_DAYS_PER_YEAR default and the fourth
            (_section_benchmark's ^GSPC row) passes TRADING_DAYS_PER_YEAR
            explicitly. Defaults to CALENDAR_DAYS_PER_YEAR (365).

    Returns:
        float: Annualized Sharpe ratio. Returns 0.0 if the standard deviation is zero.

    Raises:
        ZeroDivisionError: If `periods_per_year` is 0 (the per-period hurdle
            divides by it).
    """
    excess = daily_returns - rf / periods_per_year
    std = excess.std()
    return float(excess.mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0.0


def _sortino(daily_returns: pd.Series, rf: float = 0.0, *,
             periods_per_year: int = CALENDAR_DAYS_PER_YEAR) -> float:
    """
    Compute the annualized Sortino ratio from a series of per-period returns.

    Like the Sharpe ratio but uses only downside (negative) deviations in the
    denominator, avoiding penalization for upside volatility. Annualizes by
    multiplying that RATIO by sqrt(periods_per_year).

    `periods_per_year` governs BOTH the conversion of the annual `rf` hurdle to
    a per-period hurdle AND the sqrt annualization factor, so it must match the
    actual periodicity of the series passed: CALENDAR_DAYS_PER_YEAR (365) for a
    backtester._build_equity_curve() output, TRADING_DAYS_PER_YEAR (252) for a
    trading-day series such as the ^GSPC benchmark. It defaults to the calendar
    base for the same reason _sharpe does — every in-module caller of this
    helper consumes the calendar-day equity curve — and is keyword-only so it
    can never land in `rf`'s positional slot.

    Args:
        daily_returns (pd.Series): Series of per-period fractional returns.
        rf (float): Annual hurdle rate (the T-bill "rf" term) as a decimal.
            Defaults to 0.0.
        periods_per_year (int): Periods per year in `daily_returns`. Must be
            positive; not validated, since its one call site takes the default.
            Defaults to CALENDAR_DAYS_PER_YEAR (365).

    Returns:
        float: Annualized Sortino ratio. Returns 0.0 if there are no negative excess returns.

    Raises:
        ZeroDivisionError: If `periods_per_year` is 0 (the per-period hurdle
            divides by it).
    """
    excess = daily_returns - rf / periods_per_year
    # Standard downside deviation: RMS of the negative excess returns over ALL
    # periods (positives clipped to 0). Using the sample std of only the
    # negative values returns NaN with a single loss and is not Sortino.
    downside = np.minimum(excess, 0.0)
    dd = float(np.sqrt(np.mean(np.square(downside))))
    return float(excess.mean() / dd * np.sqrt(periods_per_year)) if dd > 0 else 0.0


def _max_drawdown(equity: pd.Series) -> tuple[float, date | None]:
    """
    Compute the maximum peak-to-trough drawdown for an equity curve.

    Args:
        equity (pd.Series): Time-indexed series of portfolio values.

    Returns:
        tuple[float, date | None]: A pair of (max_drawdown, trough_date) where
            max_drawdown is the largest fractional decline from any prior peak
            (expressed as a negative number, e.g. -0.15 for a 15% drawdown),
            and trough_date is the index label at the trough. Returns (0.0, None)
            if equity is empty or entirely NaN, if the drawdown series itself
            is entirely NaN (e.g. an all-zero equity curve, where every point
            divides 0 by a running peak of 0), or if the curve never falls
            below a prior peak (a flat curve — the page-wide filter's view of
            a selection with no trade) — there is no trough to report.
    """
    # idxmin() raises ValueError on an empty or all-NaN Series rather than
    # returning None, so that case must be handled before calling it.
    if equity.empty or equity.isna().all():
        return 0.0, None
    rolling_max = equity.cummax()
    dd = (equity - rolling_max) / rolling_max
    # An all-zero (or zero-peaked) curve makes every element 0/0 → NaN, so the
    # emptiness check above is not sufficient: re-check AFTER the division so
    # this function is total for any numeric input.
    if dd.isna().all():
        return 0.0, None
    max_dd = float(dd.min())
    # idxmin() of an all-zero series is its FIRST date, which is no trough
    when = None if max_dd == 0 else dd.idxmin()
    return max_dd, when


# Trade-type lines of the performance chart, in display order: (label, colour).
_TRADE_TYPE_LINES = (
    ("Same-title", "#8E24AA"),
    ("Time-series: ladder", "#FB8C00"),
    ("Time-series: cross-event", "#43A047"),
)

# How every trade-type line is drawn beside the solid total-return line; the
# page's filter script draws a type line from this too, so the two can never
# disagree.
_TYPE_LINE_WIDTH = 1.5
_TYPE_LINE_DASH = "dot"


def _trade_type_label(trade: BacktestTrade) -> str:
    """
    Name the trade-type line a trade belongs to.

    Args:
        trade (BacktestTrade): A completed trade.

    Returns:
        str: "Same-title" for any pair type other than time_series (the same
            fail-safe reading scanner.leg_sides applies), otherwise
            "Time-series: ladder" or "Time-series: cross-event" by
            BacktestTrade.same_event_ladder.
    """
    if trade.pair_type != "time_series":
        return "Same-title"
    return "Time-series: ladder" if trade.same_event_ladder else "Time-series: cross-event"


def _return_by_trade_type(
    trades: list[BacktestTrade], equity_df: pd.DataFrame, initial_balance: float,
) -> list[tuple[str, str, list[float]]]:
    """
    Attribute the equity curve's cumulative return to each trade type.

    Books every trade exactly as backtester._build_equity_curve does — minus
    its fees on the entry date, plus (actual_payoff - total_cost) on the exit
    date — so each type's line is that type's share of the curve and the lines
    sum to the total return on every date (up to float noise). A step dated
    outside the curve's own dates is booked on the first curve date on or
    after it, or dropped if there is none, the same days the curve itself
    would have seen it.

    Args:
        trades (list[BacktestTrade]): The run's completed trades.
        equity_df (pd.DataFrame): The run's equity curve (_build_equity_curve).
        initial_balance (float): Starting balance the percentages divide by.

    Returns:
        list[tuple[str, str, list[float]]]: (label, colour, cumulative return in
            percent per curve date), in _TRADE_TYPE_LINES order, for the types
            that have at least one trade. Empty when there are no trades, the
            curve is empty, or initial_balance is not positive.
    """
    if not trades or equity_df.empty or initial_balance <= 0:
        return []
    dates = pd.to_datetime(equity_df["date"]).to_numpy()
    steps: dict[str, np.ndarray] = {}
    for t in trades:
        label = _trade_type_label(t)
        row = steps.setdefault(label, np.zeros(len(dates)))
        for when, amount in ((t.entry_date, -t.fees),
                             (t.exit_date, t.actual_payoff - t.total_cost)):
            i = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(when)), side="left"))
            if i < len(dates):
                row[i] += amount
    return [
        (label, color, list(np.cumsum(steps[label]) / initial_balance * 100))
        for label, color in _TRADE_TYPE_LINES if label in steps
    ]


def _median_monthly_return(equity_df: pd.DataFrame | None) -> float | None:
    """
    Compute the median calendar-month return of an equity curve.

    Each month's return is its last portfolio value over the previous month's
    last value, minus one; the first month is measured against the curve's
    opening row, which _build_equity_curve always sets to the untouched
    initial balance (DR-03), so the months chain from the same base every
    other figure on the page uses. Every calendar month the curve spans counts,
    including months with no trade in them (a 0.0 return): the median describes
    the run's months as lived, not only its active ones.

    Args:
        equity_df (pd.DataFrame | None): Daily equity curve with columns
            [date, portfolio_value, ...] as produced by _build_equity_curve().

    Returns:
        float | None: The median monthly return as a fraction (e.g. 0.012 for
            +1.2%). None when the curve is absent or empty, or its opening
            value is not a positive number (no return is defined from it).
    """
    if equity_df is None or equity_df.empty:
        return None
    values = equity_df["portfolio_value"].set_axis(pd.to_datetime(equity_df["date"]))
    opening = float(values.iloc[0])
    if not math.isfinite(opening) or opening <= 0:
        return None
    month_end = values.groupby(values.index.to_period("M")).last().to_numpy(dtype=float)
    previous = np.concatenate(([opening], month_end[:-1]))
    returns = month_end / previous - 1.0
    returns = returns[np.isfinite(returns)]
    return float(np.median(returns)) if returns.size else None


def _brier_score(trades: list[BacktestTrade]) -> float:
    """
    Compute the mean Brier score across all market predictions in the trade list.

    The Brier score is the mean squared error between predicted probabilities and
    binary outcomes (1 for "yes", 0 for "no"). Lower scores indicate better calibration.
    Each trade contributes two predictions: one for market A and one for market B.

    Args:
        trades (list[BacktestTrade]): List of completed backtest trades with entry prices
            and settlement outcomes.

    Returns:
        float: Mean Brier score in [0, 1]. Returns 0.0 if no trades are provided.
    """
    scores = []
    for t in trades:
        for prob, outcome in [(t.entry_pA, t.outcome_a), (t.entry_pB, t.outcome_b)]:
            actual = 1.0 if outcome == "yes" else 0.0
            scores.append((prob - actual) ** 2)
    return float(np.mean(scores)) if scores else 0.0


def _log_loss(trades: list[BacktestTrade]) -> float:
    """
    Compute the mean binary cross-entropy (log loss) across all market predictions.

    Measures how well predicted probabilities match binary settlement outcomes. Each
    trade contributes two (probability, outcome) pairs. Probabilities are clipped to
    [1e-7, 1-1e-7] to avoid log(0).

    Args:
        trades (list[BacktestTrade]): List of completed backtest trades with entry prices
            and settlement outcomes.

    Returns:
        float: Mean log loss. Lower values indicate better calibration. Returns 0.0 if
            no trades are provided.
    """
    eps = 1e-7
    losses = []
    for t in trades:
        for prob, outcome in [(t.entry_pA, t.outcome_a), (t.entry_pB, t.outcome_b)]:
            actual = 1.0 if outcome == "yes" else 0.0
            p = max(eps, min(1 - eps, prob))
            losses.append(-(actual * np.log(p) + (1 - actual) * np.log(1 - p)))
    return float(np.mean(losses)) if losses else 0.0


def _kelly_fraction(pA: float, nA: float, pB: float, nB: float, pair_type: str,
                    k: float | None = None) -> float:
    """
    Uncapped Kelly fraction f* = p - (1-p)/b for one pair trade.

    Mirrors strategy._kelly_p and strategy.compute_trade so the dashboard scatter
    shows the same theoretical Kelly the live sizer would compute (before the
    BUDGET_FRACTION cap) — including the fee-inclusive Kelly denominator
    b = net_spread / (price_a + price_b + fee), since the losing cell loses the
    fee too (DR-62). The legs are mapped exactly like scanner.leg_prices:
    a same_title pair costs nA + pB (NO on A, YES on B) and is priced on the
    SAME_TITLE_CO_RESOLVE_PROB prior; a time_series pair costs pA + nB (YES on
    the earlier contract A, NO on the later contract B) and is priced on
    config.time_series_profit_prob(pA, pB) — one minus the discounted
    market-implied probability of the single loss cell (A=NO, B=YES; the event
    first happens between the deadlines). The two win cells are event by A
    (A=YES, hence B=YES) and never by B (A=NO, B=NO); A=YES/B=NO is impossible
    for a cumulative-deadline pair, a premise both pair-finders now screen for
    in the legs' wording (scanner.deadline_phrasing). Returns 0.0 when there is
    no edge.

    The optional k must be whatever interval discount the plotted trades were
    actually SIZED at (backtester.SweepPoint.k). Leaving it None on a run that
    passed --interval-discount would plot the config-constant Kelly against
    trades sized at the override — a wrong chart, not merely a missing feature.

    Args:
        pA (float): YES ask price of market A at entry (a leg price for time_series).
        nA (float): NO ask price of market A at entry (a leg price for same_title).
        pB (float): YES ask price of market B at entry (a leg price for same_title;
            feeds the probability model for time_series).
        nB (float): NO ask price of market B at entry (a leg price for time_series).
        pair_type (str): "time_series" or "same_title" — selects the leg prices
            and the probability model; anything else is treated as same_title,
            matching scanner.leg_sides.
        k (float | None): Interval-discount override in [0, 1] handed straight to
            config.time_series_profit_prob. None (default) means "no override",
            which that helper resolves at call time to
            config.TIME_SERIES_INTERVAL_PROB_DISCOUNT — the value live sizing
            reads. Ignored for same_title, which prices on a fixed prior.

    Returns:
        float: Uncapped Kelly fraction, clamped to be >= 0.
    """
    if pair_type == "time_series":
        price_a, price_b = pA, nB
        # Shared definition with strategy._kelly_p / backtester._simulate_at_discount
        # so the dashboard can never show a Kelly the live sizer would not compute
        p = time_series_profit_prob(pA, pB, k=k)
    else:
        price_a, price_b = nA, pB
        p = SAME_TITLE_CO_RESOLVE_PROB
    cost = price_a + price_b
    fee_approx = fee_per_pair_approx(price_a, price_b)
    net_spread = (1.0 - price_a - price_b) - fee_approx
    if cost <= 0 or net_spread <= 0:
        return 0.0
    # Kelly's "b" divides by the dollars AT RISK, which include the fee — a
    # losing pair loses cost + fees (DR-62). Mirrors strategy._evaluate_size's
    # kelly_b and backtester._simulate_at_discount's kelly_b_entry; if this kept
    # the fee-less denominator the Risk section would plot a model the live
    # sizer no longer uses.
    b = net_spread / (cost + fee_approx)
    q = 1.0 - p
    return max(0.0, p - q / b)


# ─── Section builders ─────────────────────────────────────────────────────────

_COLORS = {
    "strategy": "#2196F3",
    "sp500":    "#FF9800",
    "naive":    "#9E9E9E",
    "profit":   "#4CAF50",
    "loss":     "#F44336",
    "dd":       "#F44336",
    "cash":     "#66BB6A",
    "invested": "#42A5F5",
}

_SECTION_STYLE = """
<div style="margin:40px 0 0 0; padding:0 0 8px 0;
            border-bottom:2px solid #E0E0E0; font-size:22px;
            font-weight:700; color:#212121; font-family:sans-serif;">
{title}
</div>
"""

_KPI_TEMPLATE = """
<div style="display:inline-block; margin:12px 16px 12px 0; padding:16px 24px;
            background:#F5F5F5; border-radius:10px; min-width:140px;
            font-family:sans-serif;">
  <div style="font-size:12px; color:#757575; text-transform:uppercase;
              letter-spacing:0.5px;">{label}</div>
  <div{value_attr} style="font-size:26px; font-weight:700; color:{color};">{value}</div>
</div>
"""


# The default colour of a KPI card's value, spelled once: _kpi's default, and
# what a card list names for the cards that take it, so both render alike.
_KPI_DEFAULT_COLOR = "#212121"


def _kpi(label: str, value: str, color: str = _KPI_DEFAULT_COLOR,
         key: str | None = None) -> str:
    """
    Render a single KPI card as an HTML snippet using the _KPI_TEMPLATE.

    Args:
        label (str): Short label displayed above the value (e.g. "Total Return").
        value (str): Pre-formatted value string to display (e.g. "+12.3%").
        color (str): CSS hex color for the value text. Defaults to near-black "#212121".
        key (str | None): When given, the value element carries id="kpi-<key>",
            so the page's filter script can rewrite it for another selection.
            None (default) renders the card exactly as before, with no id.

    Returns:
        str: Rendered HTML string for one KPI card block.
    """
    value_attr = f' id="kpi-{key}"' if key else ""
    return _KPI_TEMPLATE.format(label=label, value=value, color=color,
                                value_attr=value_attr)


def _filterable_body(prefix: str, has_trades: bool, body: str) -> str:
    """
    Wrap a trade-derived section's body so the filter script can show it or a
    "No trades." line in its place.

    The body is always rendered — with no trades its charts are simply empty —
    because a selection the filter bar offers (another spread band, say) can
    have trades where the run's primary scenario has none, and the script can
    only redraw a chart that exists.

    Args:
        prefix (str): Id prefix: the elements are "<prefix>-empty" and
            "<prefix>-body".
        has_trades (bool): Whether the rendered (primary) view has any trade;
            decides which of the two starts visible.
        body (str): The section's HTML below its title.

    Returns:
        str: The "No trades." paragraph followed by the body's wrapper, exactly
            one of them visible.
    """
    hidden = ' style="display:none"'
    return (f'<p id="{prefix}-empty"{hidden if has_trades else ""}>No trades.</p>'
            f'<div id="{prefix}-body"{"" if has_trades else hidden}>{body}</div>')


def _fig_html(fig: go.Figure, height: int = 400, div_id: str | None = None) -> str:
    """
    Apply a standard layout to a Plotly figure and return it as an inline HTML string.

    Configures common layout properties (height, margins, background colors, font,
    legend position) then serializes to HTML without the full Plotly.js bundle
    (assumes the CDN script tag is already present in the page <head>).

    Args:
        fig (go.Figure): Plotly figure to render.
        height (int): Desired figure height in pixels. Defaults to 400.
        div_id (str | None): Optional fixed id for the figure's wrapping <div>,
            passed straight through to Plotly's own to_html(). None (default)
            lets Plotly generate its usual random UUID id — the same behaviour
            every pre-existing caller of this helper still gets. A caller that
            needs to drive the figure from separate client-side JS (the
            scenario-explorer section's Plotly.restyle calls) passes a fixed
            id here instead of scraping a random one out of the rendered HTML.

    Returns:
        str: HTML string fragment (no <html>/<body> wrapper, no Plotly.js script tag).
    """
    fig.update_layout(
        height=height,
        margin={"l": 60, "r": 20, "t": 40, "b": 40},
        plot_bgcolor="white",
        paper_bgcolor="white",
        font={"family": "sans-serif", "size": 12},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    kwargs = {"full_html": False, "include_plotlyjs": False}
    if div_id is not None:
        kwargs["div_id"] = div_id
    return fig.to_html(**kwargs)


# ─── Section 1: Portfolio Performance ────────────────────────────────────────

def _performance_kpis(
    equity_df: pd.DataFrame,
    trades: list[BacktestTrade],
    initial_balance: float,
) -> list[tuple[str, str, str, str]]:
    """
    Compute the Portfolio Performance KPI cards, formatted, in render order.

    The one definition of every figure on those cards, read both by
    _section_performance and by any other view of the same page that shows
    them for a different set of trades — so no second copy of a metric or of
    its formatting can drift from this one.

    Args:
        equity_df (pd.DataFrame): Daily equity curve with columns [date,
            portfolio_value, daily_return] as produced by _build_equity_curve().
        trades (list[BacktestTrade]): The trades the win rate and the per-trade
            returns are taken over.
        initial_balance (float): Starting portfolio value in dollars, the base
            of the total return.

    Returns:
        list[tuple[str, str, str, str]]: (key, label, value, colour) per card:
            total return, Sharpe, Sortino, max drawdown (with its trough date),
            win rate, mean and median return per trade, median monthly return
            and the trade count. The key is a stable identifier for the card.
    """
    final_value  = float(equity_df["portfolio_value"].iloc[-1])
    total_return = (final_value - initial_balance) / initial_balance
    daily_ret    = equity_df["daily_return"]
    sharpe       = _sharpe(daily_ret)
    sortino      = _sortino(daily_ret)
    # _max_drawdown reports the trough via the Series' index, so it must be
    # indexed by date rather than equity_df's default RangeIndex — otherwise
    # the KPI shows a meaningless row number instead of a calendar date.
    max_dd, dd_when = _max_drawdown(
        equity_df["portfolio_value"].set_axis(equity_df["date"])
    )
    win_rate     = sum(1 for t in trades if t.profit > 0) / len(trades) if trades else 0
    avg_ret      = np.mean([t.profit_ratio for t in trades]) if trades else 0
    med_ret      = np.median([t.profit_ratio for t in trades]) if trades else 0
    med_month    = _median_monthly_return(equity_df)

    dd_str = f" ({dd_when})" if dd_when else ""
    med_month_str = "—" if med_month is None else f"{med_month:+.1%}"

    return [
        ("total_return", "Total Return", f"{total_return:+.1%}", "#2196F3"),
        ("sharpe", "Sharpe Ratio", f"{sharpe:.2f}", _KPI_DEFAULT_COLOR),
        ("sortino", "Sortino Ratio", f"{sortino:.2f}", _KPI_DEFAULT_COLOR),
        ("max_drawdown", "Max Drawdown", f"{max_dd:.1%}{dd_str}", "#F44336"),
        ("win_rate", "Win Rate", f"{win_rate:.1%}", "#4CAF50"),
        ("avg_return", "Avg Return/Trade", f"{avg_ret:.1%}", _KPI_DEFAULT_COLOR),
        ("median_return", "Median Return/Trade", f"{med_ret:.1%}", _KPI_DEFAULT_COLOR),
        ("median_monthly", "Median Monthly Return", med_month_str, _KPI_DEFAULT_COLOR),
        ("trades", "Total Trades", str(len(trades)), _KPI_DEFAULT_COLOR),
    ]


def _performance_series(
    equity_df: pd.DataFrame,
    trades: list[BacktestTrade],
    initial_balance: float,
) -> tuple[pd.Series, list[tuple[str, str, list[float]]], pd.Series]:
    """
    Compute the Portfolio Performance charts' series, on equity_df's rows.

    Args:
        equity_df (pd.DataFrame): Daily equity curve (_build_equity_curve).
        trades (list[BacktestTrade]): The trades the per-type lines attribute.
        initial_balance (float): Starting balance the percentages divide by.

    Returns:
        tuple: (total, type_lines, drawdown) — the cumulative return in percent
            of the starting balance, the per-trade-type lines
            (_return_by_trade_type: label, colour, percent series; they sum to
            the total), and the drawdown from the running peak, in percent.
    """
    # Cumulative return, total and per trade type: one line each, in percent of
    # the starting balance. The per-type lines attribute each trade exactly as
    # _build_equity_curve books it, so they add up to the total line.
    total = (equity_df["portfolio_value"] / initial_balance - 1.0) * 100
    type_lines = _return_by_trade_type(trades, equity_df, initial_balance)
    rolling_max = equity_df["portfolio_value"].cummax()
    drawdown = (equity_df["portfolio_value"] - rolling_max) / rolling_max * 100
    return total, type_lines, drawdown


def _section_performance(
    equity_df: pd.DataFrame,
    trades: list[BacktestTrade],
    start_date: date,
    initial_balance: float,
) -> str:
    """
    Build the "Portfolio Performance" HTML section.

    Renders the summary KPIs (_performance_kpis: total return, Sharpe,
    Sortino, max drawdown, win rate, mean and median return per trade, median
    monthly return) and two charts (_performance_series): cumulative return as
    a series of lines — the total plus one per trade type (which sum to the
    total) — and a drawdown percentage plot. The median per
    trade sits beside the mean because a few large wins or total losses can
    carry the mean on their own; the two disagreeing is the signal.

    Args:
        equity_df (pd.DataFrame): Daily equity curve with columns [date, portfolio_value,
            daily_return] as produced by _build_equity_curve().
        trades (list[BacktestTrade]): Completed backtest trades for win rate and avg return.
        start_date (date): Backtest start date for display context.
        initial_balance (float): Starting portfolio value in dollars.

    Returns:
        str: Self-contained HTML section string including KPI cards and two Plotly charts.
    """
    # Keyed, so the filter script can rewrite each card for another selection
    kpis = "".join(_kpi(label, value, color, key=key) for key, label, value, color
                   in _performance_kpis(equity_df, trades, initial_balance))
    total, type_lines, drawdown = _performance_series(equity_df, trades, initial_balance)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_df["date"], y=total,
        name="Total return", line={"color": _COLORS["strategy"], "width": 3},
    ))
    for label, color, series in type_lines:
        fig.add_trace(go.Scatter(
            x=equity_df["date"], y=series, name=label,
            line={"color": color, "width": _TYPE_LINE_WIDTH, "dash": _TYPE_LINE_DASH},
        ))
    fig.update_layout(title="Cumulative Return by Trade Type (% of starting balance)",
                      yaxis_title="Cumulative return (%)", xaxis_title="Date",
                      legend={"orientation": "h", "y": -0.2})

    # Drawdown chart
    fig2 = go.Figure(go.Scatter(
        x=equity_df["date"], y=drawdown,
        fill="tozeroy", name="Drawdown %",
        line={"color": _COLORS["dd"]}, fillcolor="rgba(244,67,54,0.2)",
    ))
    fig2.update_layout(title="Drawdown (%)", yaxis_title="Drawdown (%)", xaxis_title="Date")

    return (
        _SECTION_STYLE.format(title="Portfolio Performance")
        + kpis
        + _fig_html(fig, div_id="perf-cum")
        + _fig_html(fig2, height=280, div_id="perf-dd")
    )


# ─── Section 2: Returns Decomposition ────────────────────────────────────────

def _series_ticker(event_ticker: str) -> str:
    """
    The series part of an event ticker (everything before its first hyphen).

    Deliberately NOT scanner.event_series, which collapses every KXMVE* combo
    series onto one family for the one-series pairing rule: Kalshi files each
    literal series under its own category, and that is what is looked up here.

    Args:
        event_ticker (str): An event ticker, e.g. "KXNCAAMBGAME-26JAN13WIUEIU".

    Returns:
        str: The series ticker ("KXNCAAMBGAME"), or "" for an empty ticker.
    """
    return (event_ticker or "").split("-", 1)[0]


def _series_labels(
    event_ticker: str,
    fallback_category: str,
    series_categories: dict[str, tuple[str, tuple[str, ...]]] | None,
) -> tuple[str, str]:
    """
    Name the Kalshi category and FIRST tag an event's series is filed under.

    The one filing rule behind every category and tag on the page — the
    Returns Decomposition, the page-wide filter and the k-hat breakdown — so
    a trade and a k-hat observation of the same event can never be filed
    apart. First tag only, so every breakdown PARTITIONS what it breaks down: a
    series can carry several tags, and counting it under each would make the
    groups add up to more than the whole.

    Args:
        event_ticker (str): The event ticker whose series is looked up
            (_series_ticker).
        fallback_category (str): The label to use when there is no map or the
            series is missing from it — the ticker-prefix category
            (BacktestTrade.category, CalibrationObservation.category).
        series_categories (dict | None): historical.load_series_categories'
            series ticker -> (category, tags), or None when not loaded.

    Returns:
        tuple[str, str]: (category, tag). The tag reads "General" when the
            series has none (or is not in the map); the category
            "Uncategorised" when Kalshi gives it none.
    """
    entry = (series_categories or {}).get(_series_ticker(event_ticker))
    if entry is None:
        return fallback_category, "General"
    return entry[0] or "Uncategorised", entry[1][0] if entry[1] else "General"


def _trade_category(
    trade: BacktestTrade,
    series_categories: dict[str, tuple[str, tuple[str, ...]]] | None,
) -> tuple[str, str]:
    """
    File a trade under Kalshi's official category and its first tag.

    Looks up market A's series in historical.load_series_categories' map,
    through _series_labels. Market A speaks for the pair: a same-title pair's
    two legs ask one question, a same-event ladder's share one event, and a
    cross-event time-series pair's legs ask one question at two deadlines —
    usually within one series, and when not, the earlier leg (A) is the one
    filed. Without a map, or for a series missing from it, it falls back to
    BacktestTrade.category (the ticker-prefix label) so the breakdown still
    renders.

    Args:
        trade (BacktestTrade): A completed trade.
        series_categories (dict | None): series ticker -> (category, tags), or
            None when the listing was not loaded.

    Returns:
        tuple[str, str]: (category, "category · tag"). The tag reads "General"
            when the series has none; the category "Uncategorised" when Kalshi
            gives it none.
    """
    category, tag = _series_labels(trade.event_ticker, trade.category, series_categories)
    return category, f"{category} · {tag}"


def _category_table(df: pd.DataFrame) -> str:
    """
    Tabulate returns per category · tag, largest P&L first.

    Args:
        df (pd.DataFrame): One row per trade with columns subcategory, profit
            and profit_ratio.

    Returns:
        str: An HTML table: trades, win rate, P&L, share of total P&L, and the
            mean and median return per trade (each trade's profit over its own
            fee-inclusive stake).
    """
    total = df["profit"].sum()
    grouped = df.groupby("subcategory").agg(
        trades=("profit", "size"),
        wins=("profit", lambda p: int((p > 0).sum())),
        pnl=("profit", "sum"),
        mean_ret=("profit_ratio", "mean"),
        median_ret=("profit_ratio", "median"),
    ).sort_values("pnl", ascending=False)
    td = "<td style='padding:4px 12px;'>"
    rows = "".join(
        "<tr style='border-bottom:1px solid #E0E0E0'>"
        + td + html.escape(str(name)) + "</td>"
        + td + str(int(r.trades)) + "</td>"
        + td + f"{r.wins / r.trades:.0%}</td>"
        + f"<td style='padding:4px 12px;color:{_COLORS['profit'] if r.pnl >= 0 else _COLORS['loss']};'>"
        + f"${r.pnl:+,.2f}</td>"
        + td + ("—" if total == 0 else f"{r.pnl / total:.0%}") + "</td>"
        + td + f"{r.mean_ret:+.1%}</td>"
        + td + f"{r.median_ret:+.1%}</td></tr>"
        for name, r in grouped.iterrows()
    )
    return (
        "<div style='font-family:sans-serif;font-size:13px;margin:8px 0 16px;'>"
        f"<b>Returns by category · tag</b> ({len(grouped)} groups, Kalshi's own series "
        "categories and tags)"
        "<table style='border-collapse:collapse;margin-top:8px;width:auto;'>"
        "<tr style='background:#E8F5E9;font-weight:bold;'>"
        "<th style='padding:6px 12px;'>Category · tag</th><th style='padding:6px 12px;'>Trades</th>"
        "<th style='padding:6px 12px;'>Win rate</th><th style='padding:6px 12px;'>P&amp;L</th>"
        "<th style='padding:6px 12px;'>Share of P&amp;L</th>"
        "<th style='padding:6px 12px;'>Mean/trade</th><th style='padding:6px 12px;'>Median/trade</th>"
        "</tr>" + rows + "</table></div>"
    )


# Entry-price buckets of the decomposition's price chart: right-closed bins
# over market A's YES ask at entry, and the label each one renders under.
_PRICE_BUCKET_BINS = [0, 0.20, 0.40, 0.60, 0.80, 1.01]
_PRICE_BUCKET_LABELS = ["<20¢", "20–40¢", "40–60¢", "60–80¢", ">80¢"]


def _pnl_colors(values) -> list[str]:
    """
    Colour each P&L bar by its sign: profit colour at or above zero, loss below.

    Args:
        values: An iterable of dollar amounts, one per bar.

    Returns:
        list[str]: One CSS colour per value.
    """
    return [_COLORS["profit"] if v >= 0 else _COLORS["loss"] for v in values]


def _subcategory_chart_height(rows: int) -> int:
    """
    Height of the "P&L by Category · Tag" chart for a given number of bars.

    Args:
        rows (int): Bars (category · tag groups) the chart draws.

    Returns:
        int: Pixels — at least 350, and 28 per bar plus room for the axes.
    """
    return max(350, 28 * rows + 120)


# The decomposition frame's columns, named so an EMPTY trade list still yields
# a frame with them (a list of no dicts would carry no columns at all).
_DECOMPOSITION_COLUMNS = ["entry_date", "exit_date", "profit_ratio", "profit", "category",
                          "subcategory", "holding_days", "entry_pA", "n", "pair_type"]


def _decomposition_frame(
    trades: list[BacktestTrade],
    series_categories: dict[str, tuple[str, tuple[str, ...]]] | None,
) -> pd.DataFrame:
    """
    Build the Returns Decomposition's one-row-per-trade frame.

    Args:
        trades (list[BacktestTrade]): The trades to decompose; may be empty,
            which yields an empty frame with every column.
        series_categories (dict | None): historical.load_series_categories'
            map, handed to _trade_category; None files each trade under its
            ticker-prefix category.

    Returns:
        pd.DataFrame: Columns entry_date, exit_date, profit_ratio, profit,
            category, subcategory ("category · tag"), holding_days, entry_pA,
            n, pair_type and month (the entry date's "YYYY-MM").
    """
    df = pd.DataFrame([{
        "entry_date":    t.entry_date,
        "exit_date":     t.exit_date,
        "profit_ratio":  t.profit_ratio,
        "profit":        t.profit,
        "category":      _trade_category(t, series_categories)[0],
        "subcategory":   _trade_category(t, series_categories)[1],
        "holding_days":  t.holding_days,
        "entry_pA":      t.entry_pA,
        "n":             t.n,
        "pair_type":     t.pair_type,
    } for t in trades], columns=_DECOMPOSITION_COLUMNS)

    df["month"] = pd.to_datetime(df["entry_date"]).dt.to_period("M").astype(str)
    return df


def _decomposition_aggregates(df: pd.DataFrame) -> dict:
    """
    Sum the decomposition frame's P&L four ways, for the section's bar charts.

    Entry price buckets read entry_pA, which is market A's YES ask at entry for
    both pair types, but its meaning differs: for a time_series row it is the
    price actually PAID for the YES leg on the earlier contract, while for a
    same_title row it is the pricier side's quote (the NO leg costs nA).

    Args:
        df (pd.DataFrame): _decomposition_frame's output. Not modified.

    Returns:
        dict: "monthly" — a frame of [month, profit] by entry month, in month
            order; "category" and "subcategory" — P&L per Kalshi category and
            per category · tag, ascending; "price" — P&L per entry-price
            bucket (_PRICE_BUCKET_LABELS), buckets with no trade omitted.
    """
    buckets = pd.cut(df["entry_pA"], bins=_PRICE_BUCKET_BINS, labels=_PRICE_BUCKET_LABELS)
    return {
        "monthly": df.groupby("month")["profit"].sum().reset_index(),
        "category": df.groupby("category")["profit"].sum().sort_values(),
        "subcategory": df.groupby("subcategory")["profit"].sum().sort_values(),
        "price": (df.assign(price_bucket=buckets)
                  .groupby("price_bucket", observed=True)["profit"].sum()),
    }


def _section_decomposition(
    trades: list[BacktestTrade],
    series_categories: dict[str, tuple[str, tuple[str, ...]]] | None = None,
) -> str:
    """
    Build the "Returns Decomposition" HTML section.

    Shows a monthly P&L bar chart, P&L by Kalshi's official market category,
    P&L by category · tag (the finer breakdown) with a table of trades, win
    rate, P&L and mean/median return per category · tag, P&L by entry price
    bucket, and a holding-period histogram.

    Args:
        trades (list[BacktestTrade]): Completed backtest trades to decompose.
        series_categories (dict | None): historical.load_series_categories'
            series ticker -> (category, tags) map. None (default) files every
            trade under its ticker-prefix BacktestTrade.category instead.

    Returns:
        str: Self-contained HTML section string. With no trades it shows a
            "No trades." line, its (empty) body rendered but hidden for the
            page's filter script (_filterable_body).
    """
    df = _decomposition_frame(trades, series_categories)
    agg = _decomposition_aggregates(df)

    # Monthly returns bar chart
    monthly = agg["monthly"]
    fig_monthly = go.Figure(go.Bar(
        x=monthly["month"], y=monthly["profit"],
        marker_color=_pnl_colors(monthly["profit"]),
    ))
    fig_monthly.update_layout(title="Monthly P&L ($)", xaxis_title="Month", yaxis_title="P&L ($)")

    # Category breakdown
    cat = agg["category"]
    fig_cat = go.Figure(go.Bar(
        y=cat.index, x=cat.values, orientation="h",
        marker_color=_pnl_colors(cat.values),
    ))
    fig_cat.update_layout(title="P&L by Category ($)", xaxis_title="P&L ($)")

    # The finer breakdown: category · tag, same colouring
    sub = agg["subcategory"]
    fig_sub = go.Figure(go.Bar(
        y=sub.index, x=sub.values, orientation="h",
        marker_color=_pnl_colors(sub.values),
    ))
    fig_sub.update_layout(title="P&L by Category · Tag ($)", xaxis_title="P&L ($)")

    # Entry price bucket (see _decomposition_aggregates for what entry_pA means
    # for each pair type)
    price_grp = agg["price"]
    fig_price = go.Figure(go.Bar(
        x=price_grp.index.astype(str), y=price_grp.values,
        marker_color=_pnl_colors(price_grp.values),
    ))
    fig_price.update_layout(title="P&L by Entry Price Bucket", xaxis_title="Price Bucket",
                            yaxis_title="P&L ($)")

    # Holding duration histogram
    fig_dur = go.Figure(go.Histogram(
        x=df["holding_days"], nbinsx=20,
        marker_color=_COLORS["strategy"],
    ))
    fig_dur.update_layout(title="Holding Period Distribution", xaxis_title="Days",
                           yaxis_title="Count")

    body = (
        _fig_html(fig_monthly, div_id="dec-monthly")
        + _fig_html(fig_cat, height=350, div_id="dec-cat")
        + _fig_html(fig_sub, height=_subcategory_chart_height(len(sub)), div_id="dec-sub")
        # Wrapped so the filter script can replace the table for a selection
        + f'<div id="dec-table">{_category_table(df) if trades else ""}</div>'
        + _fig_html(fig_price, div_id="dec-price")
        + _fig_html(fig_dur, div_id="dec-hold")
    )
    return (_SECTION_STYLE.format(title="Returns Decomposition")
            + _filterable_body("dec", bool(trades), body))


# ─── Section 3: Calibration Analysis ─────────────────────────────────────────

def _reliability(trades: list[BacktestTrade]) -> dict:
    """
    Compute the price-calibration figures: Brier score, log loss and the
    reliability diagram's points.

    Each trade contributes two predictions — market A's and market B's YES ask
    at entry — against the side each market settled on. The diagram bins them
    into 10 equal-width probability bins and keeps the non-empty ones.

    Args:
        trades (list[BacktestTrade]): Completed trades; may be empty.

    Returns:
        dict: "brier" and "log_loss" (_brier_score / _log_loss, 0.0 with no
            trades), and per non-empty bin, in ascending order: "mean_pred"
            (mean predicted probability), "mean_act" (share that resolved YES),
            "counts" (predictions in the bin), "labels" ("0.3–0.4"), and how the
            diagram draws each bin: "sizes" (marker px) and "texts" (hover).
    """
    # Collect (predicted_prob, actual_outcome) pairs
    probs, actuals = [], []
    for t in trades:
        probs.append(t.entry_pA)
        actuals.append(1 if t.outcome_a == "yes" else 0)
        probs.append(t.entry_pB)
        actuals.append(1 if t.outcome_b == "yes" else 0)

    # Reliability diagram — 10 equal-width bins
    bins   = np.linspace(0, 1, 11)
    labels = []
    mean_pred, mean_act = [], []
    counts = []
    for i in range(len(bins) - 1):
        mask = [(bins[i] <= p < bins[i + 1]) for p in probs]
        if sum(mask) == 0:
            continue
        bin_probs = [p for p, m in zip(probs, mask, strict=True) if m]
        bin_acts  = [a for a, m in zip(actuals, mask, strict=True) if m]
        mean_pred.append(np.mean(bin_probs))
        mean_act.append(np.mean(bin_acts))
        counts.append(len(bin_probs))
        labels.append(f"{bins[i]:.1f}–{bins[i+1]:.1f}")

    return {
        "brier": _brier_score(trades), "log_loss": _log_loss(trades),
        "mean_pred": mean_pred, "mean_act": mean_act, "counts": counts, "labels": labels,
        # How the diagram draws each bin: a marker growing with its count
        # (never below 6 px) and "n=<count>" on hover
        "sizes": [max(6, c // 2) for c in counts],
        "texts": [f"n={c}" for c in counts],
    }


def _calibration_title(brier: float, log_loss: float) -> str:
    """
    Title of the reliability diagram, which states the two scores.

    Args:
        brier (float): The Brier score.
        log_loss (float): The log loss.

    Returns:
        str: "Calibration Curve (Brier=0.1234, LogLoss=0.5678)".
    """
    return f"Calibration Curve (Brier={brier:.4f}, LogLoss={log_loss:.4f})"


def _section_calibration(trades: list[BacktestTrade]) -> str:
    """
    Build the "Calibration Analysis" HTML section.

    Computes Brier score and log loss KPIs and renders a reliability diagram
    (actual resolution rate vs. predicted probability per bin).

    Args:
        trades (list[BacktestTrade]): Completed backtest trades with entry prices
            and settlement outcomes.

    Returns:
        str: Self-contained HTML section string with KPIs and calibration curve.
            With no trades it shows a "No trades." line, its (empty) body
            rendered but hidden for the page's filter script
            (_filterable_body).
    """
    rel = _reliability(trades)
    brier, ll = rel["brier"], rel["log_loss"]

    fig_cal = go.Figure()
    fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Perfect calibration",
                                 line={"dash": "dash", "color": "#9E9E9E"}))
    fig_cal.add_trace(go.Scatter(
        x=rel["mean_pred"], y=rel["mean_act"], mode="lines+markers",
        name="Actual", line={"color": _COLORS["strategy"]},
        marker={"size": rel["sizes"]},
        text=rel["texts"], hoverinfo="text+x+y",
    ))
    fig_cal.update_layout(
        title=_calibration_title(brier, ll),
        xaxis_title="Predicted probability",
        yaxis_title="Actual resolution rate",
        xaxis={"range": [0, 1]}, yaxis={"range": [0, 1]},
    )

    kpis = "".join([
        _kpi("Brier Score", f"{brier:.4f}", "#2196F3", key="brier"),
        _kpi("Log Loss",    f"{ll:.4f}",    "#2196F3", key="log_loss"),
    ])

    return (
        _SECTION_STYLE.format(title="Calibration Analysis")
        + _filterable_body("cal", bool(trades), kpis + _fig_html(fig_cal, div_id="cal-curve"))
    )


# ─── Section 4: Interval Discount (k) Calibration ────────────────────────────

def _deadline_phrasing_html(coverage: OutcomeLabelCoverage | None) -> str:
    """
    Render how this run's eligible markets are worded: cumulative, snapshot, or
    no deadline wording the classifier recognises.

    A time-series pair is only formed between two CUMULATIVE-deadline markets
    ("will X happen BY <date>"), because only there does the earlier deadline's
    event nest inside the later one's. This line is what makes the size of that
    filter visible on the page rather than only in the log.

    DESCRIPTIVE, with no floor and no banner — deliberately unlike the
    outcome-label census above. The cumulative FRACTION has no healthy
    baseline: most Kalshi markets are not deadline markets at all, so any
    threshold would be arbitrary and would fire on every run, training the
    operator to ignore it. The reading that IS actionable needs no threshold —
    zero cumulative markets on a non-empty corpus means this run could not have
    produced a time-series trade whatever else the page says — so that case
    gets its own sentence, but ONLY when the three counts actually add up to
    the corpus (DR-71): a hand-built or stale coverage whose counts don't sum
    to `total` is not a measurement this line can vouch for, so it reports the
    raw counts and says so rather than asserting the verdict.

    The two refusal reasons are reported separately (DR-71), because they are
    refused for different reasons: a snapshot probability need not nest inside
    another's ("after <date>" nests the wrong way; a shared-start "after X and
    before Y" window CAN nest — a known, accepted over-refusal), while
    unrecognised wording is refused because nesting simply cannot be shown.

    Args:
        coverage (backtester.OutcomeLabelCoverage | None): The census carried
            on the sweep. None or an empty corpus renders nothing, because the
            block above has already said why there is nothing to report.

    Returns:
        str: One grey HTML line, or "" when there is no corpus to describe.
    """
    if coverage is None or not coverage.total:
        # The outcome-label block above already states which of these it was;
        # a second "nothing to report" line would only add noise.
        return ""

    total = coverage.total
    cumulative = coverage.cumulative_markets
    snapshot = coverage.snapshot_markets
    unknown = coverage.unknown_deadline_markets
    # The three counts are a fresh census on a genuine run, so they always sum
    # to total — but a hand-built or stale coverage (a test fixture, a carrier
    # from before DR-71) can disagree, and reporting a verdict off numbers that
    # don't even add up would be worse than saying nothing.
    consistent = (cumulative + snapshot + unknown == total)

    body = (
        f"Deadline phrasing: {cumulative:,} of {total:,} eligible markets are "
        f"worded as a cumulative &ldquo;by &lt;date&gt;&rdquo; deadline "
        f"({cumulative / total * 100.0:.2f}%), {snapshot:,} are snapshots "
        f"(&ldquo;on &lt;date&gt;&rdquo;, &ldquo;in &lt;month&gt;&rdquo;, "
        f"&ldquo;after &lt;date&gt;&rdquo;), and {unknown:,} carry no deadline "
        "wording the classifier recognises. "
        "Only a pair of cumulative markets can become a time-series candidate: "
        "snapshot wording is refused because such probabilities need not nest "
        "(&ldquo;after &lt;date&gt;&rdquo; nests the wrong way; shared-start "
        "&ldquo;after X and before Y&rdquo; windows can nest and are a known "
        "over-refusal), and unrecognised wording because nesting cannot be "
        "shown."
    )
    if cumulative == 0 and consistent:
        body += (
            " <strong>No eligible market was read as cumulative, so this run "
            "could not have produced a time-series trade at all.</strong>"
        )
    if not consistent:
        body += " (counts do not sum to the corpus — not a verdict)"
    return (
        "<p style='font-family:sans-serif;font-size:13px;color:#616161;'>"
        f"{body}</p>"
    )


def _label_coverage_html(coverage: OutcomeLabelCoverage | None) -> str:
    """
    Render the outcome-label census as a line, or as a banner when it is low.

    The figure this prints is the one backtester's outcome-label census
    already logged (backtester._report_outcome_label_coverage(), which
    _log_outcome_label_coverage() and _prepare_candidates both report through)
    — the SAME measurement from the same single pass over the same corpus,
    carried through BacktestSweep.label_coverage — so the page and
    the log can never report two different numbers. The below-floor verdict is
    likewise the carried one, never re-derived here: the constant is imported
    only to PRINT where the floor sits.

    Rendered even when coverage is healthy (DR-66b). Absence of a warning must
    not be the only signal a run was clean, because that is indistinguishable
    from the check not existing — which is precisely the silence DR-66 closed
    on the log side.

    The population is the run's ELIGIBLE-MARKET CORPUS, not the far smaller
    subset the empirical k-hat is measured over, so every string here is
    phrased over "eligible markets" and never over "the pairs behind k̂". The
    one claim that legitimately spans both populations is the banner's
    consequence sentence, which says what a strike-blind corpus does to the
    cards beside it.

    Every number is this run's own count, its own percentage, or the
    configured floor — no measurement from another run is baked in (TS-07) —
    and the text is static prose besides, with no Kalshi-controlled free text,
    so it needs no html.escape (unlike _crow's bucket label).

    Args:
        coverage (backtester.OutcomeLabelCoverage | None): The census carried
            on the sweep. None means no census was taken (the Monday
            feasibility short-circuit skipped the fetch, or the sweep was hand
            built), which is rendered as "not measured" rather than as either
            healthy or low coverage.

    Returns:
        str: One HTML block — a red banner when the census fell below the
            floor, otherwise a single grey line. Never the empty string, so the
            section always says something about this run's corpus.
    """
    if coverage is None:
        return ("<p style='font-family:sans-serif;font-size:13px;color:#616161;'>"
                "Outcome-label coverage was not measured for this run.</p>")

    if not coverage.total:
        return ("<p style='font-family:sans-serif;font-size:13px;color:#616161;'>"
                "Outcome-label coverage: no eligible markets to census.</p>")

    # Both fractions are non-None whenever total is non-zero (the carrier's own
    # contract), so these format calls are safe and no "None%" can render.
    subtitle_pct = coverage.subtitle_fraction * 100.0
    event_pct = coverage.event_title_fraction * 100.0
    figures = (
        f"subtitle on {coverage.with_subtitle:,} of {coverage.total:,} "
        f"eligible markets ({subtitle_pct:.2f}%), event_title on "
        f"{coverage.with_event_title:,} ({event_pct:.2f}%)"
    )

    if not coverage.below_floor:
        # The floor applies to SUBTITLE coverage alone, and the event_title
        # figure beside it is not merely uninformative — it runs the WRONG WAY.
        # Measured full-file on the two assembled caches: 12.11% event_title on
        # the label-BLIND 2026-05-01 corpus against 2.87% on the healthy
        # 2026-09-07 one, because event titles are resolved separately by
        # historical._load_or_build_event_titles under a capped per-ticker
        # fallback over an overwhelmingly MVE corpus. So a reader who takes the
        # lower number as the worse one reads a clean run as broken and a
        # broken run as clean. config.py records why it never escalates; that
        # comment reaches a codebase reader, and this clause reaches the
        # operator this page exists for.
        return ("<p style='font-family:sans-serif;font-size:13px;color:#616161;'>"
                f"Outcome-label coverage: {figures}. The floor applies to "
                "subtitle coverage only — near-zero event_title coverage is "
                "normal on an MVE-heavy corpus, does not escalate, and is not "
                "a health signal in either direction.</p>")

    # Below the floor: the same consequence and remedy the WARNING carries,
    # placed where a reader of the k-hat card cannot miss it.
    return (
        "<div style=\"font-family:sans-serif;font-size:14px;color:#B71C1C;"
        "background:#FFEBEE; border-left:6px solid #F44336; border-radius:6px;"
        "padding:14px 18px; margin:16px 0;\">"
        f"<b>Outcome-label coverage is {subtitle_pct:.2f}%, below the "
        f"{BACKTEST_OUTCOME_LABEL_WARN_FRACTION * 100.0:.2f}% floor</b> "
        f"({figures}). Most eligible markets carry no subtitle, so the "
        "time-series grouping key fell back to the bare normalized title — the "
        "strike-blind grouping the live scanner no longer uses — and the "
        "same-title key lost its outcome discriminator. The pairs behind the "
        "k&#770; below, the per-k table and every other strategy-derived "
        "section of this page (the &#94;GSPC benchmark trace, having no pair "
        "population behind it, is unaffected) "
        "were therefore formed from a different pair population than the "
        "shipped scanner would produce: treat this run's potential-pair "
        "counts, trades, returns and empirical k&#770; as describing a "
        "different strategy. "
        "<b>Remedy:</b> delete <code>backtest_cache/archive_days/</code> and "
        "<code>backtest_cache/live_days/</code>, then re-run with "
        # Names BOTH assembled-cache spellings (the streamed .jsonl.gz SS-1
        # writes and any legacy .json beside it), in step with the WARNING
        # backtester._report_outcome_label_coverage logs for the same verdict.
        "<code>--no-cache</code> (equivalently, also delete the assembled "
        "<code>backtest_cache/settled_markets_*.jsonl.gz</code> and any legacy "
        "<code>settled_markets_*.json</code>); "
        "<code>--no-cache</code> ALONE does not refresh the day slices, which "
        "are reused unconditionally."
        "</div>"
    )


def _section_interval_discount(sweep: BacktestSweep | None) -> str:
    """
    Build the "Interval Discount (k) Calibration" HTML section.

    Reports how the hand-set time-series interval discount k compares with what
    the replayed history actually did, and lets the reader switch the equity
    curve between every k the run simulated. Four parts:

      0. The outcome-label coverage of the run's eligible-market corpus — a
         grey line when healthy, and a red banner ABOVE the KPI cards when it
         fell below config.BACKTEST_OUTCOME_LABEL_WARN_FRACTION (DR-66b). The
         k-hat beside it is a recommendation for the real-money constant
         TIME_SERIES_INTERVAL_PROB_DISCOUNT, and a label-less cache makes it
         describe a strategy the shipped scanner does not implement; the
         backtest log has warned about that since DR-66 while this page said
         nothing at all. The banner sits with the cards, and the k-hat card is
         recoloured and its label suffixed, so the caveat travels with the
         number even in a screenshot of the cards alone.
      1. KPI cards — the run's configured (effective) k, the pooled empirical
         k-hat, and the delta between them.
      2. The calibration table — one row per deadline-gap bucket plus the
         pooled row, showing n, the realised in-between rate, the mean
         market-implied gap, and that bucket's k-hat.
      3. The k selector — ONE Plotly figure holding an equity trace per swept
         k, switched by a native `updatemenus` dropdown (no extra dependency,
         no hand-rolled JS) — followed by a table of each k's trade count,
         total return, max drawdown and Sharpe, computed from that point's own
         equity curve with this module's existing _max_drawdown/_sharpe.

    Deliberately named for the interval discount rather than "calibration"
    alone: _section_calibration already exists and means price calibration
    (Brier / log loss), which is a different measurement entirely.

    Takes the BacktestSweep whole rather than its parts — it already carries
    .calibration, .points, .primary.k and .label_coverage, and passing those
    separately would create copies that could disagree with each other. That
    is why carrying the census needed no signature change here.

    Scope limit: the dropdown drives THIS section only, and the page-wide
    filter bar does not reach this section at all: it always reports the
    primary spread band's calibration and k sweep. Every other section
    reflects the primary k, since return / drawdown / trade count is what one
    actually compares k values on.

    Args:
        sweep (BacktestSweep | None): The sweep payload from
            backtester.run_backtest_sweep(). None (or a sweep with no points)
            renders the same short placeholder every other builder emits for
            empty input — the case where the caller ran the plain
            run_backtest() path and has no sweep to show.

    Returns:
        str: Self-contained HTML section string.
    """
    if sweep is None or not sweep.points:
        return (_SECTION_STYLE.format(title="Interval Discount (k) Calibration")
                + "<p>No interval-discount sweep for this run.</p>")

    points = sweep.points
    # The run's effective discount, read off the primary point rather than from
    # config: on an --interval-discount run it is the override, and it is the k
    # every OTHER section on this page was rendered at.
    configured_k = sweep.primary.k
    cal = sweep.calibration
    # The outcome-label census, carried k-independently on the sweep exactly as
    # the calibration is. None means no census was taken, not healthy coverage.
    coverage = sweep.label_coverage
    coverage_html = _label_coverage_html(coverage) + _deadline_phrasing_html(coverage)
    # One verdict, read off the carrier rather than re-derived from the
    # constant, so this page and the backtest log fire on the same condition.
    tainted = coverage is not None and coverage.below_floor

    # ── KPI cards ────────────────────────────────────────────────────────────
    pooled_k = cal.pooled.empirical_k if cal is not None else None
    # Labelled for the RUN, not for config.py: on an --interval-discount run
    # this is the override. Calling it "configured" would misattribute the
    # override to config.py, which this feature never writes.
    kpi_parts = [_kpi("k used (this run)", f"{configured_k:.3f}", "#2196F3")]
    # On a label-less corpus the k-hat is arithmetically correct for the pairs
    # it was handed and those are not the pairs the shipped scanner forms, so
    # the card itself carries the caveat: a reader who screenshots the cards,
    # or who reads only the number, must not see a bare recommendation. The
    # label is SUFFIXED, never replaced — "Pooled empirical k̂" stays intact.
    khat_label = "Pooled empirical k̂" + (" (see caveat above)" if tainted else "")
    khat_color = "#F44336" if tainted else "#2196F3"
    if pooled_k is None:
        kpi_parts.append(_kpi(khat_label, "—"))
        kpi_parts.append(_kpi("k̂ − k", "—"))
    else:
        delta = pooled_k - configured_k
        # p = 1 - k*(pB - pA), so a LARGER k is the more conservative belief.
        # k̂ above the configured k means the in-between cell landed more often
        # than the sizer assumed (it was sizing too big) — flag that red; a
        # negative delta means the run was conservative.
        kpi_parts.append(_kpi(khat_label, f"{pooled_k:.3f}", khat_color))
        kpi_parts.append(_kpi("k̂ − k", f"{delta:+.3f}",
                              "#F44336" if delta > 0 else "#4CAF50"))
    kpis = "".join(kpi_parts)

    # ── Calibration table ────────────────────────────────────────────────────
    if cal is None:
        cal_table = ("<p style='font-family:sans-serif;font-size:14px;color:#616161;'>"
                     "No time-series candidate was measurable in this window, so there "
                     "is no empirical k̂ to report.</p>")
    else:
        def _crow(b) -> str:
            """
            Render one HTML table row for an IntervalCalibrationBucket.

            Args:
                b (backtester.IntervalCalibrationBucket): Bucket to display.

            Returns:
                str: An HTML <tr>...</tr> string with the bucket label, its
                    price tier ("-" for the pooled row, which spans every
                    band), candidate count, realised in-between rate, mean
                    market-implied gap, and empirical k-hat ("-" when it could
                    not be computed).
            """
            # label is the only free-text field rendered here; escape it the
            # same way _trow escapes a Kalshi-controlled market title, so no
            # future producer of a bucket can inject markup into the page.
            safe_label = html.escape(b.label)
            # tier <= 0 marks the pooled row: it spans every band and so has no
            # single tier (backtester.IntervalCalibrationBucket.tier).
            tier_txt = "-" if b.tier <= 0 else f"{b.tier:.2f}"
            k_txt = "-" if b.empirical_k is None else f"{b.empirical_k:.3f}"
            return (f"<tr style='border-bottom:1px solid #E0E0E0'>"
                    f"<td style='padding:6px 16px;'>{safe_label}</td>"
                    f"<td style='padding:6px 16px;'>{tier_txt}</td>"
                    f"<td style='padding:6px 16px;'>{b.n}</td>"
                    f"<td style='padding:6px 16px;'>{b.realised_rate:.4f}</td>"
                    f"<td style='padding:6px 16px;'>{b.mean_implied:.4f}</td>"
                    f"<td style='padding:6px 16px;'>{k_txt}</td>"
                    f"</tr>")

        cal_table = """
<table style="font-family:sans-serif;font-size:14px;border-collapse:collapse;
              margin:16px 0; width:auto;">
<tr style="background:#E3F2FD; font-weight:bold;">
  <th style="padding:8px 16px;">Gap bucket</th>
  <th style="padding:8px 16px;">Tier</th>
  <th style="padding:8px 16px;">n</th>
  <th style="padding:8px 16px;">Realised in-between rate</th>
  <th style="padding:8px 16px;">Mean implied gap</th>
  <th style="padding:8px 16px;">k&#770;</th>
</tr>
""" + "".join(_crow(b) for b in [*cal.buckets, cal.pooled]) + "</table>"

        if cal.excluded_premise_violations:
            # Summary-line idiom, silent at zero — mirrors _log_interval_calibration.
            cal_table += (
                f"<p style='font-family:sans-serif;font-size:13px;color:#616161;'>"
                f"Excluded {cal.excluded_premise_violations} premise violation(s) "
                f"(earlier YES / later NO) from the denominator.</p>"
            )

    cal_table += (
        "<p style='font-family:sans-serif;font-size:13px;color:#616161;'>"
        "Recommendation only — the backtester never writes config.py, and live "
        "sizing always reads config.TIME_SERIES_INTERVAL_PROB_DISCOUNT.</p>"
    )

    # ── The k selector: one trace per point, switched by a native dropdown ───
    # points always contains primary as the SAME object (BacktestSweep), but a
    # hand-built sweep may hold an equal copy — fall back to matching on k, and
    # to the first point if even that fails, so the figure always has exactly
    # one visible trace.
    primary_idx = next(
        (i for i, pt in enumerate(points)
         if pt is sweep.primary or pt.k == sweep.primary.k),
        0,
    )

    fig = go.Figure()
    for i, pt in enumerate(points):
        fig.add_trace(go.Scatter(
            x=pt.equity_df["date"], y=pt.equity_df["portfolio_value"],
            name=f"k = {pt.k:.2f}",
            visible=(i == primary_idx),
            line={"color": _COLORS["strategy"], "width": 2},
        ))

    # Plotly's own "update" method rewrites trace visibility and the title
    # client-side, so the whole selector is static config in the serialized
    # figure — nothing here needs a script of ours.
    buttons = [
        {
            "label": f"k = {pt.k:.2f}",
            "method": "update",
            "args": [
                {"visible": [j == i for j in range(len(points))]},
                {"title": f"Equity Curve at interval discount k = {pt.k:.2f}"},
            ],
        }
        for i, pt in enumerate(points)
    ]
    fig.update_layout(
        title=f"Equity Curve at interval discount k = {points[primary_idx].k:.2f}",
        yaxis_title="Portfolio Value ($)", xaxis_title="Date",
        # The title is left-aligned, so the selector is anchored to the RIGHT
        # and the top margin is widened to give both their own room: anchored
        # left at the default margin, the dropdown rendered on top of the
        # title text (verified in a browser before this was corrected).
        margin={"t": 90},
        updatemenus=[{
            "type": "dropdown",
            "buttons": buttons,
            "active": primary_idx,
            "direction": "down",
            "showactive": True,
            "x": 1.0, "xanchor": "right", "y": 1.16, "yanchor": "top",
        }],
    )

    # ── Sweep metrics table ──────────────────────────────────────────────────
    def _srow(pt, is_primary: bool) -> str:
        """
        Render one HTML table row of per-k sweep metrics.

        Computes total return, max drawdown and Sharpe from the point's OWN
        equity curve, using this module's existing helpers exactly as
        _section_performance calls them (_max_drawdown needs the date axis and
        returns a (drawdown, trough_date) pair, not a scalar). The return base
        is the curve's opening value, which is always the run's initial balance:
        _build_equity_curve opens every curve one day before start_date, before
        any trade can have entered.

        Args:
            pt (backtester.SweepPoint): One simulated interval discount.
            is_primary (bool): True for the run's effective k — bolded, since
                every other section on the page reflects that point.

        Returns:
            str: An HTML <tr>...</tr> string, or a row of "—" placeholders when
                the point's equity curve is too short to measure.
        """
        eq = pt.equity_df
        weight = "700" if is_primary else "400"
        label = f"k = {pt.k:.2f}" + (" (primary)" if is_primary else "")
        if eq.empty:
            # Four unmeasurable cells: return, final balance, drawdown, Sharpe.
            cells = "<td style='padding:6px 16px;'>—</td>" * 4
            return (f"<tr style='border-bottom:1px solid #E0E0E0'>"
                    f"<td style='padding:6px 16px; font-weight:{weight}'>{label}</td>"
                    f"<td style='padding:6px 16px;'>{len(pt.trades)}</td>{cells}</tr>")
        # iloc[0] is the curve's leading pre-start_date row, i.e. the untouched
        # initial balance — the same base _section_performance divides by — so
        # this row's total return and the performance card's agree exactly. It
        # used to be the post-outflow balance whenever a trade entered on
        # start_date, which reported one run two ways on one page (DR-03).
        opening = float(eq["portfolio_value"].iloc[0])
        final = float(eq["portfolio_value"].iloc[-1])
        total_return = (final - opening) / opening if opening else 0.0
        # Same call shape as _section_performance: the date axis is what makes
        # the trough label a calendar date, and the result is a 2-tuple.
        max_dd, _ = _max_drawdown(eq["portfolio_value"].set_axis(eq["date"]))
        # A one-row curve has no pct_change to speak of; _sharpe returns 0.0 on
        # a zero standard deviation, so no extra guard is needed here.
        sharpe = _sharpe(eq["daily_return"]) if "daily_return" in eq else 0.0
        return (f"<tr style='border-bottom:1px solid #E0E0E0'>"
                f"<td style='padding:6px 16px; font-weight:{weight}'>{label}</td>"
                f"<td style='padding:6px 16px;'>{len(pt.trades)}</td>"
                f"<td style='padding:6px 16px;'>{total_return:+.1%}</td>"
                f"<td style='padding:6px 16px;'>${final:,.2f}</td>"
                f"<td style='padding:6px 16px;'>{max_dd:.1%}</td>"
                f"<td style='padding:6px 16px;'>{sharpe:.2f}</td>"
                f"</tr>")

    sweep_table = """
<table style="font-family:sans-serif;font-size:14px;border-collapse:collapse;
              margin:16px 0; width:auto;">
<tr style="background:#E8F5E9; font-weight:bold;">
  <th style="padding:8px 16px;">Interval discount</th>
  <th style="padding:8px 16px;">Trades</th>
  <th style="padding:8px 16px;">Total Return</th>
  <th style="padding:8px 16px;">Final Balance</th>
  <th style="padding:8px 16px;">Max Drawdown</th>
  <th style="padding:8px 16px;">Sharpe</th>
</tr>
""" + "".join(_srow(pt, i == primary_idx) for i, pt in enumerate(points)) + "</table>"

    return (
        _SECTION_STYLE.format(title="Interval Discount (k) Calibration")
        # Above the cards, so the caveat is read before the number it qualifies.
        + coverage_html
        + kpis
        + cal_table
        + _fig_html(fig, height=450)
        + sweep_table
    )


# ─── Section 4b: Empirical k-hat by category, tag and spread band ───────────

# The k-hat chart's "Group by" choices, in menu order: (value, label).
_KHAT_GROUPS = (("category", "Category"), ("tag", "Tag"), ("band", "Spread band"))

# The two deadline-gap tier floors as the page names them ("0.15/0.30"), read
# from config — never a literal — so the page cannot name a tier the backtest
# did not apply: the filter bar's tier-floors-off views (_tier_off_where) and
# the k-hat chart's title for them (_KHAT_TEXT) both read it.
_TIER_FLOORS = (f"{_exact_label(MIN_PRICE_DIFF_SHORT_GAP, '.2f')}/"
                f"{_exact_label(MIN_PRICE_DIFF_LONG_GAP, '.2f')}")

# The k-hat chart's words — its title, its "whole population" rows and the
# notice shown when there is nothing to draw — as templates the page's script
# fills too (D.text), like the filter bar's summary line, so the chart Python
# renders and the chart the script redraws for the same view read the same.
_KHAT_TEXT = {
    "khat_title": "Empirical k̂ by {group} — {scope}",
    "khat_group_words": {"category": "category", "tag": "tag", "band": "spread band"},
    "khat_all_categories": "All categories",
    "khat_all_tags": "All tags",
    "khat_all_in": "All {category}",
    "khat_every_category": "all categories",
    # Grouped by spread band with the tier floors off, the title's scope says
    # so (grouped by category or tag, its scope is a tier-off band's own
    # _tier_off_where phrase, which already does)
    "khat_scope_tier_off": f"{{scope}}, with the {_TIER_FLOORS} tier floors off",
    "khat_none": ("No time-series candidate entry counts toward k̂ for this selection: "
                  "none was entered at this band, or every one settled in the excluded "
                  "earlier-YES / later-NO cell. With same-event ladders off, a run can form "
                  "none at all."),
    # No calibration behind the band: either the run measured none there
    # (backtester._interval_calibration returns None when no time-series
    # candidate entry at the band had a readable settlement), or the page has
    # no sweep to read one from
    "khat_not_recorded": ("k̂ was not recorded for this spread band: the run carries no "
                          "calibration for it — either no time-series candidate entry at the "
                          "band had a readable settlement to measure, or the dashboard was "
                          "built without a sweep (or from a hand-built one)."),
}

# The k-hat chart's height: (minimum, pixels per bar, room for the axes) —
# read by _khat_chart_height and, through the payload, by the page's script.
_KHAT_HEIGHT = (300, 26, 120)


def _khat_cells(stat: dict | None) -> list[str]:
    """
    Format one group's k-hat figures as the table's cells.

    The one formatting of these figures on the page: Python's table uses it,
    and the payload ships its output for the page's script, so the two can
    never round one figure differently (JavaScript's toFixed rounds a binary
    tie away from zero where Python rounds it to even).

    Args:
        stat (dict | None): A _khat_stat, or None for a group with none.

    Returns:
        list[str]: Entries, events, realised rate (4 dp), mean implied gap
            (4 dp) and k-hat (3 dp) — "—" wherever a figure is undefined.
    """
    def fmt(value, spec: str) -> str:
        return "—" if value is None else format(value, spec)
    st = stat or {}
    return [fmt(st.get("n"), "d"), fmt(st.get("events"), "d"), fmt(st.get("rate"), ".4f"),
            fmt(st.get("implied"), ".4f"), fmt(st.get("k"), ".3f")]


def _khat_bar_text(stat: dict | None) -> str:
    """
    The label on one k-hat bar: its entries and distinct events.

    Args:
        stat (dict | None): A _khat_stat's figures, or None.

    Returns:
        str: "n=12 · 3 ev" ("?" events when unknown), or "" for no stat.
    """
    if stat is None:
        return ""
    events = "?" if stat["events"] is None else stat["events"]
    return f"n={stat['n']} · {events} ev"


def _khat_finish(stat: dict) -> dict:
    """
    Add a group's rendered forms — its bar label and table cells — to its figures.

    Args:
        stat (dict): "n", "events", "rate", "implied" and "k".

    Returns:
        dict: The same dict, with "text" (_khat_bar_text) and "cells"
            (_khat_cells) added, so both renderers show Python's strings.
    """
    stat["text"] = _khat_bar_text(stat)
    stat["cells"] = _khat_cells(stat)
    return stat


def _khat_stat(observations) -> dict:
    """
    Reduce a group of k-hat observations to the figures the chart shows.

    The arithmetic is backtester._calibration_bucket's — the one definition of
    k-hat — so a group holding a band's whole carried population, in its
    carried order, reproduces that band's pooled row exactly.

    Args:
        observations: backtester.CalibrationObservation records, a list or
            the carried tuple itself (the "all" group passes the tuple, so its
            sums run in the pooled row's own order).

    Returns:
        dict: "n" (entries), "events" (distinct market-A event tickers; a
            missing ticker counts as one "" event), "rate" (realised
            in-between rate), "implied" (mean market-implied gap), "k"
            (k-hat, None when the implied gap is not positive), plus "text"
            and "cells" (_khat_finish). The event count is a better guide
            to how much evidence a bar holds than the entry count — the
            rungs of one ladder share one event and settle together — but it
            is not a count of independent outcomes: one question listed as
            several events (Oct, Nov, Dec) still counts each.
    """
    bucket = _calibration_bucket("", 0.0, observations)
    return _khat_finish({"n": bucket.n, "events": len({o.event_ticker for o in observations}),
                         "rate": bucket.realised_rate, "implied": bucket.mean_implied,
                         "k": bucket.empirical_k})


def _khat_band(
    calibration: IntervalCalibration | None,
    series_categories: dict[str, tuple[str, tuple[str, ...]]] | None,
    cat_index: dict[str, int],
    sub_index: dict[tuple[str, str], int],
) -> dict | None:
    """
    Break one band's k-hat population down by Kalshi category and first tag.

    Observations are filed by _series_labels — the rule trades are filed by
    — on market A's event ticker, which is the leg BacktestTrade.event_ticker
    records, so a category's k-hat and its trades describe the same events.
    Each group is reduced by _khat_stat.

    Args:
        calibration (IntervalCalibration | None): The band's measurement.
        series_categories (dict | None): The series-category map.
        cat_index (dict[str, int]): Category -> its index in the payload.
        sub_index (dict[tuple[str, str], int]): (category, tag) -> its index.

    Returns:
        dict | None: None when the band has no calibration; otherwise
            "carried" and "groups" (view key -> _khat_stat, keys as the
            filter's: "all", "c<i>", "s<i>"). A calibration that does not
            carry its population (len(observations) != pooled.n — a
            hand-built one) keeps only "all", taken from its pooled row, with
            "events" None, since there is nothing to break down.
    """
    if calibration is None:
        return None
    observations = calibration.observations
    if len(observations) != calibration.pooled.n:
        pooled = calibration.pooled
        return {"carried": False, "groups": {_ALL_VIEW: _khat_finish({
            "n": pooled.n, "events": None, "rate": pooled.realised_rate,
            "implied": pooled.mean_implied, "k": pooled.empirical_k})}}
    groups: dict[str, list] = {}
    for o in observations:
        category, tag = _series_labels(o.event_ticker, o.category, series_categories)
        groups.setdefault(f"c{cat_index[category]}", []).append(o)
        groups.setdefault(f"s{sub_index[(category, tag)]}", []).append(o)
    # The carried tuple itself, never its groups put back together: only it
    # reproduces the pooled row to the last bit
    stats = {_ALL_VIEW: _khat_stat(observations)}
    stats.update((key, _khat_stat(members)) for key, members in groups.items())
    return {"carried": True, "groups": stats}


def _khat_chart_height(rows: int) -> int:
    """
    Height of the k-hat bar chart for a number of bars.

    Args:
        rows (int): Bars drawn.

    Returns:
        int: Pixels, from _KHAT_HEIGHT: at least its minimum, and its pixels
            per bar plus room for the axes.
    """
    least, per_row, axes = _KHAT_HEIGHT
    return max(least, per_row * rows + axes)


def _khat_row_html(label: str, stat: dict | None) -> str:
    """
    One row of the k-hat table.

    Args:
        label (str): The group's name (Kalshi-controlled — escaped here).
        stat (dict | None): Its _khat_stat, or None.

    Returns:
        str: A <tr> with the group and its _khat_cells.
    """
    cells = stat["cells"] if stat else _khat_cells(None)
    return ("<tr style='border-bottom:1px solid #E0E0E0'>"
            f"<td style='padding:4px 12px;'>{html.escape(label)}</td>"
            + "".join(f"<td style='padding:4px 12px;'>{c}</td>" for c in cells)
            + "</tr>")


def _khat_customdata(stat: dict | None) -> list[str]:
    """
    One bar's hover figures: the table's own cells, already formatted.

    The hover shows these strings as they are (no Plotly number format), so
    it can never round a figure differently from the table beside it.

    Args:
        stat (dict | None): A _khat_stat, or None.

    Returns:
        list[str]: _khat_cells — entries, events, realised rate, mean
            implied gap, k-hat.
    """
    return stat["cells"] if stat else _khat_cells(None)


def _section_khat(payload: dict | None, k_used: float | None) -> str:
    """
    Build the "Empirical k̂ by Category, Tag and Spread Band" section.

    k-hat is the realised in-between rate divided by the mean market-implied
    gap (pB − pA), over every time-series candidate ENTRY at a band — the
    population backtester._interval_calibration pools (k-independent, before
    the Kelly gate, premise violations excluded), regrouped here. One chart
    and one table, driven by the page-wide filter bar — its band, its Tier
    floors choice (with the tiers off, each band's tier-off calibration, or
    its own where the tiers never bind), its category and tag — and a "Group
    by" <select>: by Category shows every category at the selected band, by Tag
    every tag at the selected band (within the selected category, if any),
    and by Spread band every band for the selected category or tag — the
    grouping's own filter is ignored and its selected value highlighted. Each
    bar states the entries it pools and the distinct events behind them, and
    a dashed line marks the k the run was sized at. Rendered here for the
    default (by category, primary band, no filter, tier floors on); the
    filter script redraws it for every other choice from the same payload.
    The "Group by" <select> sits outside the chart's body, so it stays
    reachable when a selection has nothing to draw.

    Args:
        payload (dict | None): _filter_payload's output (its "khat" per
            band), or None when it could not be built — the section then says
            so and draws nothing.
        k_used (float | None): The run's interval discount, drawn as the
            reference line; None draws none.

    Returns:
        str: Self-contained HTML section string.
    """
    title = _SECTION_STYLE.format(title="Empirical k̂ by Category, Tag and Spread Band")
    intro = (
        "<p style='font-family:sans-serif;font-size:13px;color:#616161;'>"
        "k&#770; = realised in-between rate ÷ mean market-implied gap (pB − pA), over "
        "every time-series candidate entry at the band — measured before the Kelly gate, "
        "so it covers entries the run never traded, and independent of k; premise "
        "violations (earlier YES, later NO) are excluded. Follows the filter bar above, "
        "its Tier floors choice included: "
        "grouping by category shows every category at the selected band, by tag every "
        "tag (within the selected category), by spread band every band for the selected "
        "category or tag — the selection is highlighted. A bar rests on its entries, but "
        "entries of one event (a ladder's rungs) settle together, so its event count is "
        "the better measure of how much evidence it holds. Recommendation only: live "
        "sizing always reads config.TIME_SERIES_INTERVAL_PROB_DISCOUNT.</p>"
    )
    notice_style = "font-family:sans-serif;font-size:14px;color:#616161;"
    if payload is None:
        return (title + intro + f'<p style="{notice_style}">This breakdown reads the '
                "page-wide filter's data, which could not be built for this run (the log "
                "names the error).</p>")

    primary = payload["primary"]
    band = payload["khat"][primary]
    rows = []
    if band is not None:
        rows.append((_KHAT_TEXT["khat_all_categories"], band["groups"].get(_ALL_VIEW), "all"))
        rows += [(name, band["groups"][f"c{ci}"], "bar")
                 for ci, name in enumerate(payload["categories"])
                 if f"c{ci}" in band["groups"]]
    has = any(st is not None and st["n"] > 0 for _, st, _ in rows)
    colors = payload["styles"]["khat"]

    fig = go.Figure(go.Bar(
        orientation="h",
        y=[label for label, _, _ in rows],
        x=[None if st is None else st["k"] for _, st, _ in rows],
        text=[_khat_bar_text(st) for _, st, _ in rows],
        # Outside the bar, and never clipped, so a zero or tiny k-hat still
        # shows its entries and events
        textposition="outside",
        cliponaxis=False,
        customdata=[_khat_customdata(st) for _, st, _ in rows],
        # Python's cells, unformatted here: the hover reads as the table does
        hovertemplate=("%{y}<br>k̂=%{customdata[4]}<br>entries=%{customdata[0]}"
                       " · events=%{customdata[1]}<br>realised=%{customdata[2]}"
                       " · implied=%{customdata[3]}<extra></extra>"),
        marker_color=[colors[kind] for _, _, kind in rows],
    ))
    layout = {
        "title": _KHAT_TEXT["khat_title"].format(
            group=_KHAT_TEXT["khat_group_words"]["category"],
            scope=payload["bands"][primary]["where"]),
        "xaxis_title": "Empirical k̂ (realised in-between rate ÷ mean implied gap)",
        # The first row (the whole population) on top, like the table below
        "yaxis": {"autorange": "reversed"},
    }
    if k_used is not None:
        layout["shapes"] = [{"type": "line", "x0": k_used, "x1": k_used, "yref": "paper",
                             "y0": 0, "y1": 1, "line": {"dash": "dash", "color": "#616161"}}]
        layout["annotations"] = [{"x": k_used, "y": 1, "yref": "paper", "yanchor": "bottom",
                                  "text": f"sized at {_k_label(k_used)}", "showarrow": False}]
    fig.update_layout(**layout)

    group_opts = "".join(f'<option value="{value}">{label}</option>'
                         for value, label in _KHAT_GROUPS)
    # Disabled until the page's script has its data, and never restored by
    # the browser on reload — the same rules as the filter bar's selects
    group_by = (
        "<div style='font-family:sans-serif;font-size:14px;margin:12px 0;'>"
        '<label>Group by: <select id="khat-group" disabled autocomplete="off">'
        f"{group_opts}</select></label></div>"
    )
    body = (
        _fig_html(fig, height=_khat_chart_height(len(rows)), div_id="khat-fig")
        + "<table style='font-family:sans-serif;font-size:13px;border-collapse:collapse;"
          "margin:8px 0 16px;width:auto;'>"
          "<tr style='background:#E3F2FD;font-weight:bold;'>"
          "<th style='padding:6px 12px;'>Group</th><th style='padding:6px 12px;'>Entries</th>"
          "<th style='padding:6px 12px;'>Events</th>"
          "<th style='padding:6px 12px;'>Realised in-between rate</th>"
          "<th style='padding:6px 12px;'>Mean implied gap</th>"
          "<th style='padding:6px 12px;'>k&#770;</th></tr>"
          '<tbody id="khat-rows">'
          + "".join(_khat_row_html(label, st) for label, st, _ in rows)
          + "</tbody></table>"
    )
    notice = _KHAT_TEXT["khat_none"] if band is not None else _KHAT_TEXT["khat_not_recorded"]
    # The body is always rendered — its chart empty when there is nothing to
    # show — so the filter script can reveal it for a view that has entries
    empty_style = notice_style + ("display:none;" if has else "")
    body_style = "" if has else ' style="display:none"'
    return (
        title + intro + group_by
        + f'<p id="khat-empty" style="{empty_style}">{html.escape(notice)}</p>'
        + f'<div id="khat-body"{body_style}>{body}</div>'
    )


# ─── Section 5: Scenario Explorer ────────────────────────────────────────────

# Above this many rows, a scenario's equity curve is embedded at one point per
# calendar week instead of one per day (see _equity_axis). A band sweep embeds
# one curve per band x k cell, so on a multi-year window the curves, not the
# metrics, are what decides the page size.
_EQUITY_DAILY_MAX_ROWS = 400

# The standalone populations one band x k cell can carry, in the ORDER the
# page's data block indexes them by (it ships this tuple as "populations", and
# the inline script resolves names through it rather than hard-coding
# positions — so this order is an index, not a display order).
_SCENARIO_POPULATIONS = ("all", "ladder", "cross", "time_series")

# The population the heatmap, the fragility banner and the equity curve read:
# every time-series entry (ladders + cross-event) simulated alone, same-title
# excluded. The band and k act on time-series pairs only, and a same-title
# pair prices on the fixed co-resolution prior, so an "all" cell carrying
# same-title trades would dilute the very comparison the explorer exists for
# — and the DR-73 calibration evidence the grid is read against was
# time-series only (its "All" row was 330 = 299 ladder + 31 cross-event
# entries). "all" keeps its own labelled KPI row.
_HEADLINE_POPULATION = "time_series"

# Every population's label on the page, spelled once so the KPI rows, the
# banner, the heatmap title and the curve title can never name one
# population two ways.
_POPULATION_LABELS = {
    "time_series": "Time-series (ladders + cross-event; same-title excluded)",
    "all": "All (time-series + same-title)",
    "ladder": "Ladders (same-event)",
    "cross": "Cross-event",
    "same_title": "Same-title (independent of band and k)",
}


def _row_label(band: tuple[float, float]) -> str:
    """
    Render a resolved spread band as a heatmap row / band <select> label.

    Spells the floor as "max(tier,<floor>)" rather than the bare floor,
    because on a run simulated with the deadline-gap tier floors applied —
    every run's primary scenario and every point of BacktestSweep.scenarios —
    the band's floor only ever applies ON TOP OF the tier
    (config.min_price_diff_for_gap(gap_days, spread_min=...)): a floor at or
    below both tiers (0.15, 0.30) is inert for every pair, and a bare
    "0.2-0.6" would hide that. A tier-floors-off run
    (BacktestSweep.tier_off_scenarios) is gated on the floor alone, so the
    page labels its bands with backtester._band_label's bare "0.2-0.6"
    instead (_tier_off_runs), never with this. Each bound goes through
    backtester._exact_label — the same injective formatter the completion
    lines use — so two DIFFERENT bands can never share a label. That matters
    here more than in a log: the heatmap's y axis is categorical, and Plotly
    merges equal category labels into one row.

    Args:
        band (tuple[float, float]): A (floor, ceiling) already resolved by
            config.time_series_spread_band.

    Returns:
        str: "max(tier,<floor>)-<ceiling>", e.g. "max(tier,0.3)-0.6".
    """
    lo, hi = band
    return f"max(tier,{_exact_label(lo, 'g')})-{_exact_label(hi, 'g')}"


def _k_label(k: float) -> str:
    """
    Render an interval discount as a heatmap column / k <select> label.

    Two decimals for every grid member ("k = 0.65"), but through
    backtester._exact_label, so an off-grid primary within a rounding of a
    grid member (--interval-discount 0.651 beside the grid's 0.65) prints
    exactly ("k = 0.651") instead of repeating the member's label. The
    heatmap's x axis is categorical: two equal labels would merge two columns,
    shift every later column under the wrong label and drop the last one.

    Args:
        k (float): A resolved interval discount.

    Returns:
        str: "k = <k>", distinct for every distinct k.
    """
    return f"k = {_exact_label(k, '.2f')}"


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    """
    Spearman rank correlation of two paired samples, ties at average rank.

    Computed as the Pearson correlation of the two samples' average ranks,
    which is Spearman's definition with ties — the same value
    scipy.stats.spearmanr returns (tests/test_dashboard.py checks it against
    scipy directly). Done by hand only so a constant sample yields None here
    rather than scipy's NaN plus a ConstantInputWarning.

    Args:
        xs (list[float]): First sample.
        ys (list[float]): Second sample, paired positionally with xs.

    Returns:
        float | None: The correlation in [-1, 1], or None when the lengths
            disagree, fewer than two pairs have BOTH values finite (pairs with
            a non-finite member are dropped, never ranked), either sample is
            constant over those pairs (its ranks have zero variance, so the
            ratio is undefined), or the result is itself non-finite.
    """
    if len(xs) != len(ys):
        return None
    pairs = [(x, y) for x, y in zip(xs, ys, strict=True)
             if x is not None and y is not None and math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return None
    rx = pd.Series([x for x, _ in pairs]).rank()
    ry = pd.Series([y for _, y in pairs]).rank()
    if rx.std() == 0 or ry.std() == 0:
        return None
    corr = rx.corr(ry)
    return float(corr) if corr is not None and math.isfinite(corr) else None


def _json_safe(obj):
    """
    Recursively replace non-finite floats with None so a payload can be
    serialised with json.dumps(..., allow_nan=False).

    A browser's JSON.parse() has no representation for NaN/Infinity — Python's
    json module emits the bare (invalid-JSON) tokens NaN/Infinity/-Infinity
    for them unless allow_nan=False, which then RAISES instead of silently
    emitting unparseable output. This walks the payload first so the raise
    never fires on a legitimately non-finite metric (a NaN in an equity curve,
    a zero-variance Sharpe denominator, ...) — those render as JS `null`, which
    every reader in the inline script renders as an em dash.

    Args:
        obj: Any JSON-serialisable structure (nested dicts/lists/tuples of
            str/int/float/bool/None).

    Returns:
        The same structure with every non-finite float replaced by None.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _equity_axis(eq: pd.DataFrame | None) -> pd.DatetimeIndex:
    """
    Decide, ONCE, the date axis every scenario's curve is embedded on.

    Called on the primary scenario's curve. Every scenario of one run spans
    the same calendar (backtester._build_equity_curve always runs from
    start_date - 1 to "today"), so one axis serves them all and the page ships
    one shared date array — but "today" is read per simulation, so a band
    sweep that crosses 00:00 UTC hands later scenarios one more row than the
    primary. Deciding the axis here, and placing every curve on it by DATE
    (_curve_on_axis), is what keeps such a cell from being drawn against the
    wrong dates — including across the downsampling threshold, where a
    per-curve decision would put a weekly curve on a daily axis.

    At or below _EQUITY_DAILY_MAX_ROWS rows the axis is every date. Above it,
    it is the LAST OBSERVED date of each calendar week, plus the curve's
    opening row (the untouched initial balance DR-03 anchors every curve on),
    so no point is dated past the curve's real end — a 2,459-row curve
    becomes 353 points, about a seventh.

    Args:
        eq (pd.DataFrame | None): The primary scenario's equity curve, columns
            [date, portfolio_value, daily_return], or None / empty.

    Returns:
        pd.DatetimeIndex: The axis, ascending; empty when eq is None or empty.
    """
    if eq is None or eq.empty:
        return pd.DatetimeIndex([])
    idx = pd.DatetimeIndex(pd.to_datetime(eq["date"])).unique().sort_values()
    if len(idx) <= _EQUITY_DAILY_MAX_ROWS:
        return idx
    positions = pd.Series(np.arange(len(idx)), index=idx)
    week_ends = positions.groupby(idx.to_period("W")).max().to_numpy()
    keep = np.union1d([0], week_ends)
    return idx[keep]


def _curve_on_axis(eq: pd.DataFrame | None, axis: pd.DatetimeIndex) -> list[float | None]:
    """
    Place one scenario's equity curve on the shared axis, by date, in cents.

    Values are rounded to cents: a portfolio value is a dollar amount, the
    curve is display-only, and a full-precision float (~19 characters) instead
    of a cents one (~8) roughly doubles the page's dominant term on a long
    band sweep. A date the curve does not have — or a non-finite value — is
    None (a gap in the line, JS null), never a neighbour's value shifted into
    its slot.

    Args:
        eq (pd.DataFrame | None): One scenario's equity curve, or None / empty.
        axis (pd.DatetimeIndex): The shared axis from _equity_axis.

    Returns:
        list[float | None]: One value per axis date; [] when eq is None or
            empty or the axis is empty.
    """
    if eq is None or eq.empty or len(axis) == 0:
        return []
    s = pd.Series(eq["portfolio_value"].to_numpy(dtype=float),
                  index=pd.DatetimeIndex(pd.to_datetime(eq["date"])))
    s = s[~s.index.duplicated(keep="last")]
    return [round(float(v), 2) if math.isfinite(v) else None
            for v in s.reindex(axis).to_numpy(dtype=float)]


def _point_kpis(point: SweepPoint) -> dict:
    """
    Compute one standalone SweepPoint's KPI-table row.

    Every field is derived from the point's OWN trades and OWN equity curve,
    never sliced out of a joint run: a band sweep's "ladder" and "cross"
    points and the same-title point are each their own simulation from the
    initial balance, and only a simulation that ran alone has a return, a
    drawdown or a Sharpe of its own — slicing the joint "all" run's trades by
    population would give a P&L sum with no curve behind it.

    Args:
        point (SweepPoint): One simulated scenario, band sweep or not.

    Returns:
        dict: {trades, win_rate, mean_per_trade, median_per_trade,
            total_return, final_balance, max_drawdown, sharpe, sortino}.
            win_rate counts trades with profit STRICTLY above zero.
            mean_per_trade is the mean of BacktestTrade.profit_ratio, which
            the backtester defines as profit / (total_cost + fees) — the
            return on each trade's own fee-inclusive stake, equal-weighted
            across trades; median_per_trade is the median of the same
            quantity, which a few outsized trades cannot move. Sharpe and
            Sortino are annualised on the calendar-day base (365), like every
            other figure computed on a strategy curve. Every value is None
            where the underlying quantity is undefined (no trades, or an
            empty/absent equity curve) rather than a misleading 0.0.
    """
    trades = point.trades
    n = len(trades)
    win_rate = (sum(1 for t in trades if t.profit > 0) / n) if n else None
    mean_per_trade = float(np.mean([t.profit_ratio for t in trades])) if n else None
    median_per_trade = float(np.median([t.profit_ratio for t in trades])) if n else None
    eq = point.equity_df
    if eq is None or eq.empty:
        total_return = final_balance = max_dd = sharpe = sortino = None
    else:
        # Same base as _section_interval_discount's _srow: the curve's OWN
        # opening row, always the untouched initial balance (DR-03), so this
        # figure never disagrees with the run's own performance card.
        opening = float(eq["portfolio_value"].iloc[0])
        final_balance = float(eq["portfolio_value"].iloc[-1])
        total_return = (final_balance - opening) / opening if opening else None
        max_dd, _ = _max_drawdown(eq["portfolio_value"].set_axis(eq["date"]))
        sharpe = _sharpe(eq["daily_return"]) if "daily_return" in eq else None
        sortino = _sortino(eq["daily_return"]) if "daily_return" in eq else None
    return {
        "trades": n, "win_rate": win_rate, "mean_per_trade": mean_per_trade,
        "median_per_trade": median_per_trade, "total_return": total_return, "final_balance": final_balance,
        "max_drawdown": max_dd, "sharpe": sharpe, "sortino": sortino,
    }


def _measured_half(half_return: float, half_entries: int | None) -> float | None:
    """
    Return a split-half return, or None when that half had no entries.

    An empty half's simulation enters nothing, so its return reads 0.0 —
    indistinguishable from a half that traded and broke even. Rendering that
    0.0 would put a non-measurement into the heatmap and, worse, into the
    split-half rank correlation, where a column of tied zeros is scored as if
    it were data. HalfSplit carries the entry counts precisely so this can be
    told apart.

    Args:
        half_return (float): HalfSplit.h1_return or h2_return.
        half_entries (int | None): The matching h1_entries / h2_entries. None
            (not recorded — a hand-built HalfSplit) keeps the return, since
            nothing says the half was empty.

    Returns:
        float | None: half_return, or None when half_entries is exactly 0.
    """
    return None if half_entries == 0 else half_return


def _robustness_extras(point: SweepPoint) -> dict:
    """
    Compute the split-half and concentration fields a checked point carries.

    A band sweep runs both checks on its "all" and "time_series" points
    (backtester._sweep_from_candidates); every other population carries
    neither.

    Args:
        point (SweepPoint): An "all" or "time_series" point (band sweep or
            not).

    Returns:
        dict: {h1_return, h2_return} read off point.halves — None when halves
            was never computed (a run without the band sweep) and, for each
            half on its own, None when that half had NO entries
            (_measured_half), since an empty half's 0.0 is not a return — and
            {top_event, top_event_share, ex_top_return}. top_event and
            ex_top_return are read straight off point.ex_top_event — the event
            and the return of a RE-SIMULATION without its entries — never
            re-derived. top_event_share is that event's summed profit divided
            by the sum of every event's POSITIVE summed profit: that
            denominator stays positive, and the share stays in (0, 1], even
            when the run as a whole lost money (a raw net-P&L denominator
            would be negative or near zero there). None when no event has
            positive P&L. All three are None when ex_top_event is None (not
            a band-sweep point, or no trade on the point names an event).
    """
    h1 = h2 = None
    if point.halves is not None:
        h1 = _measured_half(point.halves.h1_return, point.halves.h1_entries)
        h2 = _measured_half(point.halves.h2_return, point.halves.h2_entries)
    top_name = top_share = ex_top_return = None
    if point.ex_top_event is not None:
        top_name, ex_top_return = point.ex_top_event
        # Same grouping _ex_top_event itself uses (BacktestTrade.event_ticker,
        # market A's event as traded), recomputed here only for the SHARE this
        # carrier does not keep.
        pnl_by_event: dict[str, float] = defaultdict(float)
        for t in point.trades:
            if t.event_ticker:
                pnl_by_event[t.event_ticker] += t.profit
        positive_sum = sum(v for v in pnl_by_event.values() if v > 0)
        top_pnl = pnl_by_event.get(top_name, 0.0)
        top_share = (top_pnl / positive_sum) if positive_sum > 0 else None
    return {
        "h1_return": h1, "h2_return": h2,
        "top_event": top_name, "top_event_share": top_share,
        "ex_top_return": ex_top_return,
    }


# The inline script the scenario explorer drives its selects with. A raw
# string, so the — escapes reach the browser as JS escapes. It reads ONE
# JSON block (id="scn-data") and writes only through textContent-escaped HTML
# and Plotly.restyle — it draws nothing itself.
_SCENARIO_EXPLORER_JS = r"""
<script>
(function() {
  var data = JSON.parse(document.getElementById('scn-data').textContent);
  var bandSel = document.getElementById('scn-band-select');
  var kSel = document.getElementById('scn-k-select');
  // Population name -> its index in every cell's array.
  var P = {};
  data.populations.forEach(function(name, i) { P[name] = i; });

  function isNum(x) { return typeof x === 'number' && isFinite(x); }
  function fmtPct(x) { return isNum(x) ? (x * 100).toFixed(1) + '%' : '—'; }
  function fmtFixed(x, d) { return isNum(x) ? x.toFixed(d) : '—'; }
  function fmtInt(x) { return isNum(x) ? String(x) : '—'; }
  function fmtMoney(x) {
    return isNum(x)
      ? '$' + x.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})
      : '—';
  }
  // Every data-sourced STRING goes through this before innerHTML: the top
  // event's ticker is Kalshi-controlled (BacktestTrade.event_ticker), and
  // escaping the backtester's own bucket labels too costs nothing.
  function esc(s) {
    var d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }
  var TD = '<td style="padding:6px 16px;">';

  function kpiRow(label, k) {
    k = k || {};
    return '<tr style="border-bottom:1px solid #E0E0E0">' + TD + label + '</td>'
      + TD + fmtInt(k.trades) + '</td>' + TD + fmtPct(k.win_rate) + '</td>'
      + TD + fmtPct(k.mean_per_trade) + '</td>' + TD + fmtPct(k.median_per_trade) + '</td>'
      + TD + fmtPct(k.total_return) + '</td>'
      + TD + fmtMoney(k.final_balance) + '</td>' + TD + fmtPct(k.max_drawdown) + '</td>'
      + TD + fmtFixed(k.sharpe, 2) + ' / ' + fmtFixed(k.sortino, 2) + '</td></tr>';
  }

  // A checked row's split-half and concentration figures, on a sub-row of it
  // (the time-series row and the All row carry them; H1/H2 read "—" for a
  // half that had no entries, since its 0.0 would not be a return).
  function extrasRow(a) {
    var line = '—';
    if (a) {
      line = 'H1 return: ' + fmtPct(a.h1_return) + ' | H2 return: ' + fmtPct(a.h2_return);
      if (a.top_event) {
        line += ' | Top event: ' + esc(a.top_event) + ' (' + fmtPct(a.top_event_share)
          + ' of positive event P&amp;L) | return re-simulated without it: '
          + fmtPct(a.ex_top_return);
      }
    }
    return '<tr style="border-bottom:1px solid #E0E0E0;font-size:13px;color:#616161;">'
      + TD + '</td><td colspan="7" style="padding:4px 16px 8px;">' + line + '</td></tr>';
  }

  function calRows(rows) {
    if (!rows || !rows.length) {
      return '<p style="font-family:sans-serif;font-size:14px;color:#616161;">'
        + 'No time-series candidate was measurable at this band.</p>';
    }
    var th = '<th style="padding:8px 16px;">';
    var t = '<table style="font-family:sans-serif;font-size:14px;border-collapse:collapse;'
      + 'margin:16px 0; width:auto;"><tr style="background:#E3F2FD; font-weight:bold;">'
      + th + 'Gap bucket</th>' + th + 'Tier</th>' + th + 'n</th>'
      + th + 'Realised in-between rate</th>' + th + 'Mean implied gap</th>'
      + th + 'k&#770;</th></tr>';
    rows.forEach(function(r) {
      t += '<tr style="border-bottom:1px solid #E0E0E0">'
        + TD + esc(r.label) + '</td>' + TD + fmtFixed(r.tier, 2) + '</td>'
        + TD + fmtInt(r.n) + '</td>' + TD + fmtFixed(r.realised_rate, 4) + '</td>'
        + TD + fmtFixed(r.mean_implied, 4) + '</td>' + TD + fmtFixed(r.empirical_k, 3) + '</td>'
        + '</tr>';
    });
    return t + '</table>';
  }

  function render() {
    var bi = parseInt(bandSel.value, 10), ki = parseInt(kSel.value, 10);
    var cell = data.cells[bi][ki];
    var L = data.labels;
    // The headline (time-series) population first — the one the heatmap and
    // banner read — then its two parts, then All and Same-title, each row
    // named by its own label so no two populations can be mistaken.
    var ts = cell[P.time_series];
    document.getElementById('scn-kpi-body').innerHTML =
      kpiRow(esc(L.time_series), ts) + extrasRow(ts)
      + kpiRow(esc(L.ladder), cell[P.ladder]) + kpiRow(esc(L.cross), cell[P.cross])
      + kpiRow(esc(L.all), cell[P.all]) + extrasRow(cell[P.all])
      + kpiRow(esc(L.same_title), data.same_title);
    document.getElementById('scn-cal-body').innerHTML = calRows(data.calibration_by_band[bi]);

    // Every curve is already on the shared date axis (data.dates), so x is
    // the axis itself, never a slice of it. The curve is the headline
    // population's, like the heatmap.
    var values = (ts && ts.equity) ? ts.equity : [];
    if (window.Plotly && document.getElementById('scn-equity')) {
      Plotly.restyle('scn-equity', {x: [values.length ? data.dates : []], y: [values]});
    }
  }

  bandSel.addEventListener('change', render);
  kSel.addEventListener('change', render);
  render();
})();
</script>
"""


def _scenario_explorer_empty_reason(sweep: BacktestSweep) -> str:
    """
    Name why a non-None sweep still has no scenarios to show.

    Args:
        sweep (BacktestSweep): A sweep whose .scenarios is empty.

    Returns:
        str: One of two causes, told apart by label_coverage. It is None only
            on the Monday-feasibility short-circuit (run_backtest_sweep()'s
            infeasible-window branch, which censuses nothing) or on a
            hand-built sweep that never had one; every FEASIBLE run carries a
            real census, even over an empty corpus (OutcomeLabelCoverage with
            total=0), so a present census means the window was fine and the
            band sweep was simply not requested. The band-sweep cause names
            both the operator's CLI switch and the keyword it sets.
    """
    if sweep.label_coverage is None:
        return "infeasible window (no trades and no census)"
    return "band sweep off (--no-band-sweep / band_sweep=False)"


def _run_settings_html(sweep: BacktestSweep | None) -> str:
    """
    Render the page-header line naming the run's primary spread band and ladder setting.

    Both settings shape the population every section reports on — the
    same-event ladder switch (DR-73) decides which pairs exist, admitting or
    refusing a whole pair population, and the band decides which of them are
    ever entered (it acts inside backtester._find_entry, after pair
    extraction) — so, like DR-66b's strike-blind notice,
    they taint every section of the page, not only the scenario explorer, and
    belong in the header a reader sees before any figure. "not recorded" is
    printed rather than a guess whenever the sweep does not carry the value
    (no sweep at all — the four-positional generate_dashboard call — or a
    hand-built sweep).

    Args:
        sweep (BacktestSweep | None): The run's sweep payload, or None.

    Returns:
        str: One <p> line: "Primary spread band: <label> | same-event
            ladders: on / off / not recorded".
    """
    band = ladders = "not recorded"
    if sweep is not None:
        if sweep.primary.spread_band is not None:
            band = _row_label(sweep.primary.spread_band)
        if sweep.same_event_ladders is True:
            ladders = "on"
        elif sweep.same_event_ladders is False:
            ladders = "off"
    return (
        '<p style="color:#616161; font-size:14px;">'
        f"Primary spread band: {html.escape(band)} | same-event ladders: {ladders}</p>"
    )


def _corpus_provenance_html(sweep: BacktestSweep | None) -> str:
    """
    Render the page-header lines saying what settled-market corpus the run read.

    DR-13 and M2/M3 of the 2026-09-24 review. The Period line above it prints
    start_date → today because the equity curve runs to today, but the corpus
    holds no market settled after its assembly — and a cached re-run serves a
    corpus assembled by an earlier run, so the two can be days apart. And a
    window starting at or after the archive cutoff cannot enter any trade
    (post-cutoff markets have no historical candlesticks), which used to
    reach only the log, and only on a cache miss: a cached re-run, and the
    HTML on every run, showed a flat 0.0% result with no caveat.

    Always renders a line, healthy or not — absence must never be the only
    signal (DR-66): the assembly time (a legacy settled_markets_*.json's
    file time, named as such), whether it came from an earlier run's cache
    (and that --no-cache extends it), and the archive cutoff at assembly.
    When the carried verdict says the window starts at or after that cutoff,
    a second line follows, in one of two forms decided by
    backtester.max_trades_simulated — the one definition the log's closing
    WARNING also reads, so page and log agree:
      * no simulated point traded: a red banner stating a BOUND, not a cause
        — no trade could be entered whatever pairs formed; such a run may
        also have formed no pairs at all (the 2026-09-17 window formed 0).
      * some simulated point traded: the verdict is proven stale (the cutoff
        has since moved past the start date, or the run could not have
        traded), so the page says THAT instead of a red "no trade could be
        entered" beside "Trades found: N".
    The verdict is the one historical._corpus_provenance CARRIED, never
    re-derived here.

    Args:
        sweep (BacktestSweep | None): The run's sweep payload, or None.

    Returns:
        str: One grey <p> line, plus a red or amber <p> when the carried
            post-cutoff verdict is True. "not recorded" when there is no
            sweep or it carries no provenance (the window's fetch was
            skipped, a stubbed corpus, a hand-built sweep).
    """
    prov: CorpusProvenance | None = None if sweep is None else sweep.corpus_provenance
    grey = '<p style="color:#616161; font-size:14px;">'
    if prov is None:
        return (
            f"{grey}Settled-market corpus: assembly time and archive cutoff not "
            "recorded for this run (no sweep was passed to the report, the "
            "window's fetch was skipped, or the corpus did not come from an "
            "assembled cache).</p>"
        )
    if prov.assembled_at is None:
        assembled = "assembly time not recorded"
    elif prov.legacy:
        assembled = (f"last written {prov.assembled_at:%Y-%m-%d %H:%M} UTC (the "
                     "file time of a legacy settled_markets_*.json, which records "
                     "no assembly stamp) — it holds no market settled after that")
    else:
        assembled = (f"assembled {prov.assembled_at:%Y-%m-%d %H:%M} UTC — it holds "
                     "no market settled after that")
    if not prov.from_cache:
        source = "assembled by this run"
    elif prov.legacy:
        source = ("served from an earlier run's cache; --no-cache extends it and "
                  "rebuilds it in the streamed format")
    else:
        source = "served from an earlier run's cache; --no-cache extends it"
    if prov.archive_cutoff is not None:
        cutoff = f"archive cutoff at assembly: {prov.archive_cutoff:%Y-%m-%d}"
    elif prov.legacy:
        cutoff = ("archive cutoff at assembly: not recorded (the legacy format "
                  "records none; --no-cache re-checks it)")
    else:
        cutoff = "archive cutoff at assembly: not recorded (--no-cache re-checks it)"
    line = (f"{grey}Settled-market corpus: {html.escape(assembled)} "
            f"({html.escape(source)}) | {html.escape(cutoff)}</p>")
    if not prov.post_cutoff:
        return line
    cutoff_day = (f"{prov.archive_cutoff:%Y-%m-%d}" if prov.archive_cutoff is not None
                  else "not recorded")
    # A trade at any simulated point disproves "no trade could be entered":
    # the one test the log's closing WARNING applies too
    traded = max_trades_simulated(sweep)
    if traded:
        recorded = "at this corpus's assembly" if prov.from_cache else "by this run"
        notice = (
            f"The archive cutoff recorded {recorded} ({cutoff_day}) is at or "
            "after this window's start date, which would mean no trade could be "
            f"entered — but this run entered trades (up to {traded} in one "
            "simulated scenario), so that verdict is stale: the cutoff has since "
            "moved past the start date. --no-cache re-reads the cutoff and "
            "re-stamps the cache."
        )
        return line + (
            '<p style="color:#E65100; font-size:14px; font-weight:700;">'
            f"{html.escape(notice)}</p>"
        )
    stale = (" If the cutoff has since moved past the start date this may no "
             "longer hold — a cached run does not re-read it; --no-cache "
             "re-checks." if prov.from_cache else "")
    return line + (
        '<p style="color:#B71C1C; font-size:14px; font-weight:700;">'
        f"This window starts at or after the archive cutoff ({cutoff_day}, as "
        "of the corpus's assembly). Post-cutoff markets have no historical "
        "candlesticks, so no trade could be entered in this window whatever "
        "pairs formed: a zero-trade result on this page is structural and "
        f"says nothing about the strategy.{html.escape(stale)}</p>"
    )


def _section_scenario_explorer(sweep: BacktestSweep | None) -> str:
    """
    Build the "Scenario Explorer" HTML section.

    Renders BacktestSweep.scenarios — every (spread band, k) cell of a band
    sweep, each with standalone "all" / "time_series" / "ladder" / "cross"
    simulations — so that choosing a band and k from a backtest happens with
    the grid's fragility on screen rather than from one flattering cell.

    The heatmap, the banner and the equity curve all read ONE population,
    _HEADLINE_POPULATION — "time_series", every time-series entry (ladders +
    cross-event) simulated alone with same-title excluded — and say so on
    the page. The band and k act on time-series pairs only; same-title pairs
    price on the fixed co-resolution prior, so reading the "all" cells here
    would let a band- and k-independent result dilute the comparison, and
    the evidence the grid is read against (the DR-73 calibration corpus) was
    time-series only. A cell with no time-series entry has no time-series
    point and reads "—" everywhere below; nothing falls back to "all". In
    order:

      1. A fragility banner, first: how many band x k cells were computed
         (and, when some have no time-series entry, how many do), the share
         of time-series cells with a positive total return, the split-half
         rank correlation (Spearman) of the
         time-series cells' H1 vs H2 returns — over the cells whose two
         halves BOTH had entries, since an empty half's 0.0 is not a return —
         and the sentence this section exists to put on the page — the best
         of that many correlated cells overstates what a reader should
         expect, where "that many" counts the time-series cells the heatmap
         shows a value in, never the whole grid. It says in words that every
         figure in it, the heatmap and the curve is the time-series
         population's, and that the KPI table below labels its own rows.
      2. A band (row) x k (column) heatmap of the time-series population,
         titled with it, with a native Plotly `updatemenus` metric toggle:
         mean per trade (the default), median per trade, total return, H1
         return, H2 return, trade count and empirical k-hat. Each button is
         an "update" — it swaps the trace's z, colour scale, hover format and
         customdata AND the chart title together (a "restyle" button's second
         argument is read as trace indices, so a title placed there would be
         silently dropped). Row labels read "max(tier,<floor>)-<ceiling>"
         (_row_label). A half with no entries reads null (rendered "—") in the
         H1/H2 views. The k-hat view repeats each band's POOLED k-hat
         (BacktestSweep.calibrations_by_band) across every k column, because
         k-hat is measured over the band's entries and cannot depend on k; it
         is centred on the primary k, its hover shows the entries pooled, and
         a band with no calibration reads null. The same figures follow as a
         static one-row-per-band table ("Empirical k-hat by spread band").
      3. Two <select>s (band, k), preselected to the primary scenario and
         marked "(primary)", driving — through the small inline script
         _SCENARIO_EXPLORER_JS — a KPI table with one row per population,
         each labelled from _POPULATION_LABELS (Time-series, with a sub-row of
         its H1/H2 and top-event figures; Ladders; Cross-event; All —
         time-series + same-title, the run's actual result — with its own
         sub-row; Same-title, which is independent of band and k), the
         selected band's own calibration table, and the time-series equity
         curve (rendered once via _fig_html(div_id="scn-equity") and
         restyled in place). A native updatemenus dropdown cannot express two
         independent axes of selection, which is why this part is scripted.

    Every number the script reads comes from one
    `<script type="application/json" id="scn-data">` block: cells are
    addressed by integer index into the ordered band / k / population arrays
    it also carries, and its only strings are values that are themselves
    names (the date axis, a calibration bucket's label, a cell's top event
    ticker); every value is passed through _json_safe() and the
    payload serialised with json.dumps(..., allow_nan=False), so a non-finite
    metric reaches the browser as null (rendered as an em dash) rather than as
    an invalid NaN token; and every "</" is escaped, because the payload
    carries each cell's top event ticker (Kalshi-controlled) and an unescaped
    "</script>" inside a JSON string would end the block early. The run's
    primary band and ladder setting are not repeated here — they are on the
    page header (_run_settings_html), since they shape every section.

    Args:
        sweep (BacktestSweep | None): The sweep payload from
            backtester.run_backtest_sweep(). None renders "No sweep for this
            run." — the same shape _section_interval_discount's None branch
            takes, and deliberately free of any "strike-blind" /
            "Outcome-label coverage" text, since that caveat belongs to a
            sweep this run never produced. A sweep with no scenarios renders a
            one-line note naming the cause (_scenario_explorer_empty_reason).

    Returns:
        str: Self-contained HTML section string.
    """
    title = _SECTION_STYLE.format(title="Scenario Explorer")

    if sweep is None:
        return title + "<p>No sweep for this run.</p>"
    if not sweep.scenarios:
        return (title + "<p>No scenarios were computed: "
                f"{_scenario_explorer_empty_reason(sweep)}.</p>")

    # ── Index the (band, k) grid and bucket every scenario into it ──────────
    all_points = [pt for pt in sweep.scenarios if pt.population == "all"]
    bands = sorted({pt.spread_band for pt in all_points if pt.spread_band is not None})
    ks = sorted({pt.k for pt in all_points})
    band_idx = {b: i for i, b in enumerate(bands)}
    k_idx = {k: i for i, k in enumerate(ks)}

    cell_points: list[list[dict]] = [
        [dict.fromkeys(_SCENARIO_POPULATIONS) for _ in ks] for _ in bands
    ]
    for pt in sweep.scenarios:
        if (pt.spread_band not in band_idx or pt.k not in k_idx
                or pt.population not in _SCENARIO_POPULATIONS):
            continue
        cell_points[band_idx[pt.spread_band]][k_idx[pt.k]][pt.population] = pt

    # Cached per point, so every metric is computed exactly once however many
    # places (heatmap, banner, data block) read it.
    kpi_cache: dict[int, dict] = {}

    def kpis(pt: SweepPoint) -> dict:
        key = id(pt)
        if key not in kpi_cache:
            kpi_cache[key] = _point_kpis(pt)
            if pt.population in ("all", _HEADLINE_POPULATION):
                kpi_cache[key].update(_robustness_extras(pt))
        return kpi_cache[key]

    headline_label = _POPULATION_LABELS[_HEADLINE_POPULATION]

    # ── Fragility banner — the headline population's cells only ─────────────
    # n_cells counts the grid (every band x k cell has an "all" point); every
    # FIGURE below is the time-series population's, and a cell with no
    # time-series point simply contributes no return — it never falls back
    # to the "all" point, which is how a same-title result would reach here.
    # The multiple-comparison count ("the best of N") is n_headline, the
    # cells the heatmap actually shows a value in, never the grid: a band
    # where no time-series pair enters has an "all" cell but no time-series
    # one (the backtester skips an empty population).
    n_cells = sum(1 for row in cell_points for cp in row if cp["all"] is not None)
    headline_points = [cp[_HEADLINE_POPULATION] for row in cell_points for cp in row
                       if cp[_HEADLINE_POPULATION] is not None]
    n_headline = len(headline_points)
    finite_returns = [r for r in (kpis(pt)["total_return"] for pt in headline_points)
                      if r is not None and math.isfinite(r)]
    positive_share = (sum(1 for r in finite_returns if r > 0) / len(finite_returns)
                      if finite_returns else None)
    checked = [kpis(pt) for pt in headline_points if pt.halves is not None]
    # A cell whose H1 or H2 had no entries carries None there
    # (_robustness_extras), and _spearman drops any pair with a None, so the
    # correlation is over the cells whose two halves both had entries.
    corr = _spearman([k["h1_return"] for k in checked], [k["h2_return"] for k in checked])
    n_empty_half = sum(1 for k in checked if k["h1_return"] is None or k["h2_return"] is None)
    if corr is not None:
        corr_txt = f"{corr:+.3f}"
        if n_empty_half:
            corr_txt += (f" (over the {len(checked) - n_empty_half} of {len(checked)} cells "
                         "whose two halves both had entries)")
    elif n_empty_half:
        corr_txt = ("not measurable — the split date leaves a half without entries in "
                    f"{n_empty_half} of the {len(checked)} cells")
    else:
        corr_txt = "not enough data"
    share_txt = f"{positive_share:.1%}" if positive_share is not None else "—"
    coverage_txt = ("" if n_headline == n_cells else
                    f"; {n_headline} of the {n_cells} cells have a time-series entry and "
                    "the rest are blank on the heatmap")
    best_txt = (f"The best of {n_headline} correlated cells overstates what you should expect."
                if n_headline else
                "No cell has a time-series entry, so nothing on this grid measures the band "
                "or k.")
    banner = (
        "<div style='background:#FFF3E0;border:1px solid #FFB74D;border-radius:8px;"
        "padding:12px 16px;margin:12px 0;font-family:sans-serif;font-size:14px;"
        "color:#5D4037;'>"
        + f"<b>{n_cells} band x k cells computed</b> ({len(sweep.scenarios)} scenario "
        "points across the time-series, all, ladder and cross-event populations"
        f"{coverage_txt}). Every figure in this banner, the heatmap and the "
        f"equity curve is the <b>{html.escape(headline_label)}</b> population's; the KPI "
        f"table below labels each row with its own population. {share_txt} of "
        "the cells with a measurable return had a positive total return. Split-half "
        f"rank correlation (Spearman) of cell returns, H1 vs H2: {corr_txt}. {best_txt}"
        "</div>"
    )

    # ── Heatmap: band rows x k columns, one "update" button per metric ──────
    def metric_matrix(field: str) -> list[list]:
        return [
            [
                (kpis(cell_points[bi][ki][_HEADLINE_POPULATION])[field]
                 if cell_points[bi][ki][_HEADLINE_POPULATION] is not None else None)
                for ki in range(len(ks))
            ]
            for bi in range(len(bands))
        ]

    band_labels = [_row_label(b) for b in bands]
    k_labels = [_k_label(k) for k in ks]
    # Colour scales are read back off a trace plotly.py has already coerced,
    # never passed to the browser by name: plotly.py and plotly.js define
    # "RdBu" in OPPOSITE directions, so a named scale in a button would flip
    # the colours the first time a metric is switched.
    diverging = go.Heatmap(colorscale="RdBu").colorscale
    sequential = go.Heatmap(colorscale="Blues").colorscale
    pct_hover = ("band=%{y}<br>%{x}<br>value=%{z:.2%}<br>trades=%{customdata}"
                 "<extra></extra>")
    count_hover = "band=%{y}<br>%{x}<br>trades=%{z}<extra></extra>"
    # Empirical k-hat is measured ONCE per spread band — over that band's
    # time-series entries, read off the prepared entries rather than any
    # k-sized simulation (backtester._interval_calibration), so it cannot
    # depend on k — and its row therefore repeats one figure across every k
    # column. Its hover names the entries it pooled, since a k-hat over a
    # handful of entries is not a measurement worth reading.
    def band_pooled(band: tuple[float, float]):
        cal = sweep.calibrations_by_band.get(band)
        return None if cal is None else cal.pooled

    pooled_by_band = [band_pooled(b) for b in bands]
    khat_matrix = [[None if p is None else p.empirical_k] * len(ks) for p in pooled_by_band]
    khat_n_matrix = [[None if p is None else p.n] * len(ks) for p in pooled_by_band]
    khat_hover = ("band=%{y}<br>k̂=%{z:.3f} (the same at every k)"
                  "<br>entries pooled=%{customdata}<extra></extra>")

    # (field, label, colour scale, zmid, hover, customdata field). zmid None
    # lets the scale auto-range: a trade count is never negative, so centring
    # it on zero would spend half the colour scale on values that cannot
    # occur. k-hat is centred on the run's primary k, so the colour says which
    # side of the discount the sizing assumed each band's evidence falls on.
    heatmap_fields = [
        ("mean_per_trade", "Mean per trade (equal stake)", diverging, 0, pct_hover, "trades"),
        ("median_per_trade", "Median per trade (equal stake)", diverging, 0, pct_hover,
         "trades"),
        ("total_return", "Total return", diverging, 0, pct_hover, "trades"),
        ("h1_return", "H1 return", diverging, 0, pct_hover, "trades"),
        ("h2_return", "H2 return", diverging, 0, pct_hover, "trades"),
        ("trades", "Trade count", sequential, None, count_hover, "trades"),
        ("empirical_k", "Empirical k̂ (pooled per band)", diverging, sweep.primary.k,
         khat_hover, "empirical_k_n"),
    ]
    # _json_safe here too: the button args are free-form JSON that no Plotly
    # validator touches, so a NaN cell must already be None when it gets there.
    matrices = {field: _json_safe(metric_matrix(field))
                for field, *_ in heatmap_fields if field != "empirical_k"}
    matrices["empirical_k"] = _json_safe(khat_matrix)
    matrices["empirical_k_n"] = _json_safe(khat_n_matrix)

    _, default_label, default_scale, default_zmid, default_hover, default_custom = (
        heatmap_fields[0])
    hfig = go.Figure()
    hfig.add_trace(go.Heatmap(
        z=matrices["mean_per_trade"], x=k_labels, y=band_labels,
        customdata=matrices[default_custom], colorscale=default_scale, zmid=default_zmid,
        hovertemplate=default_hover,
    ))
    # The title names the population, on every metric (each button re-sets
    # it), so a screenshot of the heatmap alone still says whose cells these
    # are.
    heat_title = f"by spread band x k — {headline_label}"
    hfig.update_layout(
        title=f"{default_label} {heat_title}",
        xaxis_title="k", yaxis_title="Spread band",
        updatemenus=[{
            "type": "dropdown", "direction": "down", "active": 0, "showactive": True,
            "x": 1.0, "xanchor": "right", "y": 1.16, "yanchor": "top",
            "buttons": [
                {
                    "label": label, "method": "update",
                    "args": [
                        # customdata travels with every button: k-hat's hover
                        # reads entry counts, every other metric's trade counts
                        {"z": [matrices[field]], "colorscale": [scale],
                         "zmid": [zmid], "hovertemplate": [hover],
                         "customdata": [matrices[custom]]},
                        {"title.text": f"{label} {heat_title}"},
                    ],
                }
                for field, label, scale, zmid, hover, custom in heatmap_fields
            ],
        }],
    )

    # ── Empirical k-hat for EVERY band at once (the heatmap's k-hat metric in
    # table form, with the counts behind it). One row per band, pooled over
    # that band's time-series entries; the per-gap-bucket breakdown of the
    # SELECTED band stays in the calibration table the selects drive. ───────
    def fmt(value, spec: str) -> str:
        return "—" if value is None or not math.isfinite(value) else format(value, spec)

    td = "<td style='padding:4px 12px;'>"
    khat_rows = "".join(
        "<tr style='border-bottom:1px solid #E0E0E0'>"
        + td + html.escape(label) + "</td>"
        + td + ("—" if p is None else str(p.n)) + "</td>"
        + td + fmt(None if p is None else p.realised_rate, ".4f") + "</td>"
        + td + fmt(None if p is None else p.mean_implied, ".4f") + "</td>"
        + td + fmt(None if p is None else p.empirical_k, ".3f") + "</td></tr>"
        for label, p in zip(band_labels, pooled_by_band, strict=True)
    )
    khat_table = (
        "<details open style='font-family:sans-serif;font-size:13px;margin:8px 0 16px;'>"
        f"<summary><b>Empirical k&#770; by spread band</b> (pooled over each band's "
        f"time-series entries; the same at every k — compare with the primary "
        f"k = {sweep.primary.k:.3f})</summary>"
        "<table style='border-collapse:collapse;margin-top:8px;width:auto;'>"
        "<tr style='background:#E8F5E9;font-weight:bold;'>"
        "<th style='padding:6px 12px;'>Spread band</th>"
        "<th style='padding:6px 12px;'>n</th>"
        "<th style='padding:6px 12px;'>Realised in-between rate</th>"
        "<th style='padding:6px 12px;'>Mean implied gap</th>"
        "<th style='padding:6px 12px;'>Pooled k&#770;</th></tr>"
        + khat_rows + "</table></details>"
    )

    # ── The <select>s, preselected to (and marking) the primary scenario ─────
    primary_band_idx = band_idx.get(sweep.primary.spread_band, 0)
    primary_k_idx = k_idx.get(sweep.primary.k, 0)

    def options(labels: list[str], primary: int) -> str:
        return "".join(
            f'<option value="{i}"{" selected" if i == primary else ""}>'
            f'{html.escape(lbl)}{" (primary)" if i == primary else ""}</option>'
            for i, lbl in enumerate(labels)
        )

    selects = (
        "<div style='font-family:sans-serif;font-size:14px;margin:16px 0;'>"
        "<label>Spread band: <select id='scn-band-select'>"
        + options(band_labels, primary_band_idx) + "</select></label>"
        "&nbsp;&nbsp;"
        "<label>k: <select id='scn-k-select'>"
        + options(k_labels, primary_k_idx) + "</select></label>"
        "</div>"
    )

    # ── KPI table skeleton (filled by the inline script) ─────────────────────
    kpi_table = """
<table style="font-family:sans-serif;font-size:14px;border-collapse:collapse;
              margin:16px 0; width:auto;">
<tr style="background:#E8F5E9; font-weight:bold;">
  <th style="padding:8px 16px;">Population</th>
  <th style="padding:8px 16px;">Trades</th>
  <th style="padding:8px 16px;">Win Rate</th>
  <th style="padding:8px 16px;">Mean/Trade</th>
  <th style="padding:8px 16px;">Median/Trade</th>
  <th style="padding:8px 16px;">Total Return</th>
  <th style="padding:8px 16px;">Final Balance</th>
  <th style="padding:8px 16px;">Max Drawdown</th>
  <th style="padding:8px 16px;">Sharpe / Sortino (365-day base)</th>
</tr>
<tbody id="scn-kpi-body"></tbody>
</table>
<div id="scn-cal-body"></div>
"""

    # ── The headline population's equity curve: rendered once for the
    # primary cell, then restyled in place by the inline script. The axis is
    # decided once, from the primary (every scenario of one run spans the same
    # calendar), and every cell's curve is placed on it by date. Only the
    # headline population ships a curve per cell: the curves are the page's
    # dominant term, and a second population's would roughly double it. ─────
    axis = _equity_axis(sweep.primary.equity_df)
    axis_dates = [d.date().isoformat() for d in axis]
    primary_cell = cell_points[primary_band_idx][primary_k_idx] if bands and ks else {}
    primary_headline = primary_cell.get(_HEADLINE_POPULATION)
    primary_values = (_curve_on_axis(primary_headline.equity_df, axis)
                      if primary_headline is not None else [])
    efig = go.Figure()
    efig.add_trace(go.Scatter(
        x=axis_dates if primary_values else [], y=primary_values,
        name=headline_label,
        line={"color": _COLORS["strategy"], "width": 2},
    ))
    efig.update_layout(title=f"Equity curve — selected band x k — {headline_label}",
                       yaxis_title="Portfolio Value ($)", xaxis_title="Date")

    # ── The data block every select, table and chart above reads from ────────
    def cell_entry(pt: SweepPoint | None, population: str) -> dict | None:
        if pt is None:
            return None
        d = dict(kpis(pt))
        if population == _HEADLINE_POPULATION:
            d["equity"] = _curve_on_axis(pt.equity_df, axis)
        return d

    def cal_json(band: tuple[float, float]) -> list[dict] | None:
        cal = sweep.calibrations_by_band.get(band)
        if cal is None:
            return None
        return [
            {"label": b.label, "tier": (None if b.tier <= 0 else b.tier), "n": b.n,
             "realised_rate": b.realised_rate, "mean_implied": b.mean_implied,
             "empirical_k": b.empirical_k}
            for b in [*cal.buckets, cal.pooled]
        ]

    payload = {
        "bands": [[lo, hi] for lo, hi in bands],
        "ks": list(ks),
        "populations": list(_SCENARIO_POPULATIONS),
        # Every row's label, spelled once (_POPULATION_LABELS); the script
        # escapes each before it reaches innerHTML.
        "labels": dict(_POPULATION_LABELS),
        "primary_band_idx": primary_band_idx,
        "primary_k_idx": primary_k_idx,
        "dates": axis_dates,
        # cells[band index][k index][population index]
        "cells": [
            [
                [cell_entry(cell_points[bi][ki][pop], pop) for pop in _SCENARIO_POPULATIONS]
                for ki in range(len(ks))
            ]
            for bi in range(len(bands))
        ],
        "same_title": (kpis(sweep.same_title_point)
                       if sweep.same_title_point is not None else None),
        "calibration_by_band": [cal_json(b) for b in bands],
    }
    # _json_safe() has already replaced every non-finite float, so
    # allow_nan=False never fires in practice — it is the backstop that makes
    # a missed one raise here instead of shipping unparseable JSON. "</" is
    # escaped so a Kalshi-controlled top-event ticker can never close this
    # <script> block early.
    json_text = json.dumps(_json_safe(payload), allow_nan=False).replace("</", "<\\/")
    data_block = f'<script type="application/json" id="scn-data">{json_text}</script>'

    return (
        title
        + banner
        + _fig_html(hfig, height=450)
        + khat_table
        + selects
        + kpi_table
        + _fig_html(efig, height=400, div_id="scn-equity")
        + data_block
        + _SCENARIO_EXPLORER_JS
    )


# ─── Section 6: Trade-Level Diagnostics ──────────────────────────────────────

def _fmt_day(d: date | None) -> str:
    """
    Format a calendar date for the trade tables, e.g. "Jan 14, 2026".

    Args:
        d (date | None): The date, or None when the trade does not carry it.

    Returns:
        str: The formatted date, or "date unknown" for None.
    """
    return "date unknown" if d is None else f"{d:%b} {d.day}, {d.year}"


def _leg_market(title: str, subtitle: str, ticker: str) -> str:
    """
    Describe one leg's market as escaped HTML: its question, its outcome label,
    and its ticker in small grey text.

    The outcome label is appended unless the title already contains it, because
    it is what tells two same-titled markets apart ("Western Illinois" vs
    "Eastern Illinois" under one game question). The ticker is shown because
    it names the series: KXNCAAMBGAME and KXNCAAWBGAME are the men's and the
    women's game under identical wording.

    Args:
        title (str): The market's question (Kalshi-controlled text).
        subtitle (str): The market's outcome label ("" when absent).
        ticker (str): The market's ticker.

    Returns:
        str: An HTML fragment, every Kalshi-controlled string escaped.
    """
    text = title if not subtitle or subtitle in title else f"{title} — {subtitle}"
    return (f"{html.escape(text)} "
            f"<span style='color:#9E9E9E;font-size:11px;'>{html.escape(ticker)}</span>")


def _trade_row(t: BacktestTrade, color: str) -> str:
    """
    Render one best/worst-trade table row for a BacktestTrade.

    Each leg is described in market order (A, then B): the side it bought, the
    market, the date that market closed for trading and the price paid per
    contract — and, separately, the side the market settled on and when, marked
    won or lost for the side this trade held. Which side each leg bought comes
    from scanner.leg_sides and the prices from backtester._leg_prices_for, the
    same single sources the simulation priced and paid out with.

    Args:
        t (BacktestTrade): Trade to display.
        color (str): CSS background color for the row (e.g. "#F9FBE7").

    Returns:
        str: An HTML <tr>...</tr> string: entry date, pair type, the YES and NO
            prices paid, the trade details, the outcome, contract count,
            fee-inclusive cost and profit/return.
    """
    side_a, side_b = leg_sides(t.pair_type)
    price_a, price_b = _leg_prices_for(t.pair_type, t.entry_pA, t.entry_nA,
                                       t.entry_pB, t.entry_nB)
    paid = {side_a: price_a, side_b: price_b}
    legs = (
        (side_a, price_a, t.title_a, t.subtitle_a, t.ticker_a, t.close_date_a,
         t.outcome_a, t.settled_date_a),
        (side_b, price_b, t.title_b, t.subtitle_b, t.ticker_b, t.close_date_b,
         t.outcome_b, t.settled_date_b),
    )
    details = "<br>".join(
        f"<b>{side.upper()}</b> on {_leg_market(title, sub, ticker)} "
        f"(closes {_fmt_day(closes)}) at ${price:.2f}"
        for side, price, title, sub, ticker, closes, _, _ in legs
    )
    outcome = "<br>".join(
        f"{name}: settled <b>{str(result).upper()}</b> on {_fmt_day(settled)} "
        f"({'won' if result == side else 'lost'})"
        for name, (side, _, _, _, _, _, result, settled) in zip(("A", "B"), legs, strict=True)
    )
    cell = "<td style='padding:4px 8px;vertical-align:top;'>"
    return (f"<tr style='background:{color}'>"
            f"{cell}{t.entry_date}</td>"
            f"{cell}{html.escape(t.pair_type)}</td>"
            f"{cell}${paid['yes']:.2f}</td>"
            f"{cell}${paid['no']:.2f}</td>"
            f"{cell}{details}</td>"
            f"{cell}{outcome}</td>"
            f"{cell}{t.n}</td>"
            f"{cell}${t.total_cost + t.fees:.2f}</td>"
            f"<td style='padding:4px 8px;vertical-align:top;"
            f"color:{'#2E7D32' if t.profit >= 0 else '#C62828'}'>"
            f"${t.profit:+.2f} ({t.profit_ratio:.1%})</td>"
            f"</tr>")


# Row backgrounds of the best- and worst-five trade tables.
_BEST_ROW_COLOR = "#F9FBE7"
_WORST_ROW_COLOR = "#FFF8F8"


def _best_and_worst(trades: list[BacktestTrade]) -> tuple[list, list]:
    """
    Pick the five most and five least profitable trades.

    A stable sort by dollar profit, descending, so trades tied on profit keep
    their order; the two slices overlap below 11 trades (see
    _section_diagnostics for why that is accepted).

    Args:
        trades (list[BacktestTrade]): Completed trades; may be empty.

    Returns:
        tuple[list, list]: (best five, most profitable first; worst five, in
            the same descending order — the last one is the biggest loss).
    """
    sorted_trades = sorted(trades, key=lambda t: t.profit, reverse=True)
    return sorted_trades[:5], sorted_trades[-5:]


def _section_diagnostics(trades: list[BacktestTrade]) -> str:
    """
    Build the "Trade-Level Diagnostics" HTML section.

    Renders a per-trade return distribution histogram, a slippage histogram, and
    HTML tables listing the top 5 and worst 5 trades by dollar profit. Each row
    (_trade_row) shows the YES and NO prices paid, each leg's side, market,
    close date and price, and how each leg settled and when.

    Args:
        trades (list[BacktestTrade]): Completed backtest trades to diagnose.

    Returns:
        str: Self-contained HTML section string. With no trades it shows a
            "No trades." line, its (empty) body rendered but hidden for the
            page's filter script (_filterable_body).
    """
    # Return distribution histogram
    ratios = [t.profit_ratio for t in trades]
    fig_hist = go.Figure(go.Histogram(
        x=[r * 100 for r in ratios], nbinsx=30,
        marker_color=_COLORS["strategy"],
    ))
    fig_hist.update_layout(title="Per-Trade Return Distribution (%)",
                            xaxis_title="Return (%)", yaxis_title="Count")

    # Slippage distribution
    slippages = [t.slippage for t in trades]
    fig_slip = go.Figure(go.Histogram(x=slippages, nbinsx=20,
                                      marker_color="#7986CB"))
    # "Expected" is the win-scenario payoff: the co-resolution floor for a
    # same-title trade, the profit of either win cell for a time-series one —
    # so a time-series loss cell shows as large negative slippage, never positive
    fig_slip.update_layout(title="Slippage Distribution (Actual − Win-Scenario Payoff, $)",
                            xaxis_title="Slippage ($)", yaxis_title="Count")

    # Best and worst trades table.
    #
    # Below 11 trades these two slices OVERLAP — identical rows under both
    # headings at fewer than 5, partial overlap at 5-10 — with nothing on the
    # page saying so. That is a recorded WONTFIX (operator decision,
    # 2026-09-13), not an oversight: on a low-frequency bot's early dashboards
    # the overlap is accepted, and a reader of a 5-trade dashboard is looking
    # at every trade either way. Do not "fix" it into a single merged table
    # without asking; the sweep raised it as TS-29 and it was declined.
    best, worst = _best_and_worst(trades)

    header = ("<th>Entry</th><th>Type</th><th>YES paid</th><th>NO paid</th>"
              "<th>Trade details</th><th>Outcome</th><th>n</th>"
              "<th>Cost incl. fees</th><th>Profit</th>")
    table_html = f"""
<div style="margin:16px 0; font-family:sans-serif;">
<b>Top 5 Trades</b>
<table style="width:100%;border-collapse:collapse;font-size:13px;margin-top:8px;">
<tr style="background:#E8F5E9;font-weight:bold;">{header}</tr>
<tbody id="diag-best">""" + "".join(_trade_row(t, _BEST_ROW_COLOR) for t in best) + f"""</tbody>
</table>
<br>
<b>Worst 5 Trades</b>
<table style="width:100%;border-collapse:collapse;font-size:13px;margin-top:8px;">
<tr style="background:#FFEBEE;font-weight:bold;">{header}</tr>
<tbody id="diag-worst">""" + "".join(_trade_row(t, _WORST_ROW_COLOR) for t in worst) + """</tbody>
</table>
</div>
"""

    body = (_fig_html(fig_hist, div_id="diag-ret")
            + _fig_html(fig_slip, height=280, div_id="diag-slip")
            + table_html)
    return (_SECTION_STYLE.format(title="Trade-Level Diagnostics")
            + _filterable_body("diag", bool(trades), body))


# ─── Section 7: Risk Metrics ──────────────────────────────────────────────────

def _kelly_points(trades: list[BacktestTrade],
                  k: float | None) -> tuple[list[float], list[float]]:
    """
    Compute the Kelly-vs-actual scatter's two coordinates, one pair per trade.

    Args:
        trades (list[BacktestTrade]): Completed trades; may be empty.
        k (float | None): The interval discount the trades were SIZED at,
            handed to _kelly_fraction (None resolves to the config constant).

    Returns:
        tuple[list[float], list[float]]: (uncapped Kelly fraction, fraction of
            the entry-date balance actually committed — fee-inclusive cost over
            BacktestTrade.balance_at_entry, 0.0 when that balance is not
            positive), each in trade order.
    """
    # Pass all four entry quotes — _kelly_fraction picks the leg prices per pair
    # type — plus the run's interval discount, so an --interval-discount run
    # plots the Kelly its trades were actually sized at rather than the config one
    kelly_fracs = [
        _kelly_fraction(t.entry_pA, t.entry_nA, t.entry_pB, t.entry_nB, t.pair_type, k=k)
        for t in trades
    ]
    # The actual fraction uses the simulated balance at each trade's entry (the
    # base its Kelly budget was computed from) — dividing by the initial
    # balance would distort as equity drifts.
    actual_fracs = [
        (t.total_cost + t.fees) / t.balance_at_entry if t.balance_at_entry > 0 else 0.0
        for t in trades
    ]
    return kelly_fracs, actual_fracs


def _one_to_one_extent(kelly_fracs: list[float]) -> float:
    """
    How far the Kelly scatter's dashed 1:1 reference line runs.

    Args:
        kelly_fracs (list[float]): The scatter's x values; may be empty.

    Returns:
        float: 10% past the largest Kelly fraction, and never less than 0.011
            (the line is drawn even when every fraction is 0).
    """
    return max(kelly_fracs + [0.01]) * 1.1


def _capital_deployed(trades: list[BacktestTrade], equity_df: pd.DataFrame) -> list[float]:
    """
    Compute the capital tied up in open trades on each row of the equity curve.

    Args:
        trades (list[BacktestTrade]): Completed trades; may be empty.
        equity_df (pd.DataFrame): The equity curve whose "date" column is the
            axis — datetime.date values, as _build_equity_curve writes them;
            a timestamp (datetime, pd.Timestamp) is read as its date.

    Returns:
        list[float]: One value per equity_df row — the running sum of every
            trade's fee-inclusive cost from its entry date until its exit date,
            floored at 0.
    """
    entry_by_date: dict[date, float] = {}
    exit_by_date: dict[date, float] = {}
    for t in trades:
        # Fee-inclusive: fees are cash out the door at entry, so they are part
        # of the capital deployed. The Kelly-vs-actual scatter
        # (_kelly_points) already uses (total_cost + fees); this used not to,
        # so the two charts on one dashboard disagreed by the fee rate (TS-12).
        cost = t.total_cost + t.fees
        entry_by_date[t.entry_date] = entry_by_date.get(t.entry_date, 0.0) + cost
        exit_by_date[t.exit_date]   = exit_by_date.get(t.exit_date,   0.0) + cost

    invested_by_date: list[float] = []
    running_invested = 0.0
    # The date column alone, not iterrows(): building a Series per row made
    # this the slowest step of the page-wide filter, which runs it per view
    for d in equity_df["date"]:
        # A datetime (pd.Timestamp included) IS a date subclass, so test for
        # it: kept whole it would match no trade's entry or exit date
        d = d.date() if isinstance(d, datetime) else d
        running_invested += entry_by_date.get(d, 0.0) - exit_by_date.get(d, 0.0)
        invested_by_date.append(max(0.0, running_invested))
    return invested_by_date


def _section_risk(trades: list[BacktestTrade], equity_df: pd.DataFrame,
                  initial_balance: float, k: float | None = None) -> str:
    """
    Build the "Risk Metrics" HTML section.

    Renders a Kelly fraction vs. actual fraction scatter plot (to assess sizing
    discipline) and a portfolio value over time chart showing capital deployment.

    Args:
        trades (list[BacktestTrade]): Completed backtest trades for sizing analysis.
        equity_df (pd.DataFrame): Daily equity curve with columns [date, portfolio_value].
        initial_balance (float): Starting portfolio value in dollars.
        k (float | None): The interval discount these trades were SIZED at,
            passed through to _kelly_fraction so the scatter's x-axis is the
            Kelly the run actually used. None (default) means "no override" and
            resolves to config.TIME_SERIES_INTERVAL_PROB_DISCOUNT — correct for
            every run that did not pass --interval-discount, and wrong for one
            that did, which is why the caller threads it.

    Returns:
        str: Self-contained HTML section string. With no trades it shows a
            "No trades." line, its (empty) body rendered but hidden for the
            page's filter script (_filterable_body).
    """
    # Kelly vs actual sizing scatter, at the discount the trades were sized at
    kelly_fracs, actual_fracs = _kelly_points(trades, k)

    fig_kelly = go.Figure(go.Scatter(
        x=kelly_fracs, y=actual_fracs, mode="markers",
        marker={"color": _COLORS["strategy"], "size": 7, "opacity": 0.6},
        # title_a is Kalshi-controlled and Plotly's hover text renders an HTML
        # subset — escape it the same as the table row above (_trow).
        text=[html.escape(t.title_a[:40]) for t in trades],
    ))
    extent = _one_to_one_extent(kelly_fracs)
    fig_kelly.add_trace(go.Scatter(
        x=[0, extent], y=[0, extent],
        name="1:1 line", line={"dash": "dash", "color": "#9E9E9E"},
    ))
    fig_kelly.update_layout(
        title="Kelly Fraction vs Actual Fraction of Balance",
        xaxis_title="Kelly fraction", yaxis_title="Actual fraction",
    )

    # Capital deployment over time: net running position size (cost still tied up
    # in unsettled trades). Positive on entry days, back down to zero on exit days.
    invested_by_date = _capital_deployed(trades, equity_df)

    fig_dep = make_subplots()
    fig_dep.add_trace(go.Scatter(
        x=equity_df["date"], y=invested_by_date,
        name="Capital deployed", fill="tozeroy",
        line={"color": _COLORS["invested"]},
        fillcolor="rgba(66,165,245,0.15)",
    ))
    fig_dep.update_layout(title="Capital Deployed Over Time",
                           yaxis_title="Capital in open trades ($)", xaxis_title="Date")

    return (_SECTION_STYLE.format(title="Risk Metrics")
            + _filterable_body("risk", bool(trades),
                               _fig_html(fig_kelly, div_id="risk-kelly")
                               + _fig_html(fig_dep, div_id="risk-dep")))


# ─── Section 8: Benchmark Comparison ─────────────────────────────────────────

def _strategy_row(equity_df: pd.DataFrame, initial_balance: float) -> dict[str, str]:
    """
    Compute the benchmark table's strategy row, formatted.

    Args:
        equity_df (pd.DataFrame): The strategy's equity curve
            (_build_equity_curve: one row per CALENDAR day).
        initial_balance (float): Starting balance the return divides by.

    Returns:
        dict[str, str]: "return" (total return), "sharpe" (annualised on the
            calendar base) and "max_dd" (max drawdown), each formatted.
    """
    strat_ret    = float(equity_df["portfolio_value"].iloc[-1] / initial_balance - 1)
    # Calendar-daily by construction (_build_equity_curve emits one row per
    # calendar day), so this takes _sharpe's CALENDAR_DAYS_PER_YEAR default
    # while the ^GSPC row overrides it to the trading-day base.
    strat_sharpe = _sharpe(equity_df["daily_return"])
    strat_dd     = _max_drawdown(equity_df["portfolio_value"])[0]
    return {
        "return": f"{strat_ret:+.1%}",
        "sharpe": f"{strat_sharpe:.2f}",
        "max_dd": f"{strat_dd:.1%}",
    }


def _bench_cell_id(row_name: str, field: str) -> str:
    """
    The id attribute of one benchmark-table cell: the strategy row's cells are
    "bench-<field>", every other row's none.

    Args:
        row_name (str): The row's name ("Kalshi Arbitrage Strategy" or
            "S&P 500").
        field (str): "return", "sharpe" or "max_dd".

    Returns:
        str: ' id="bench-<field>"' for the strategy row, "" otherwise.
    """
    return f' id="bench-{field}"' if row_name == "Kalshi Arbitrage Strategy" else ""


def _section_benchmark(equity_df: pd.DataFrame, start_date: date,
                        initial_balance: float) -> str:
    """
    Build the "Benchmark Comparison" HTML section.

    Fetches S&P 500 data via yfinance, normalizes it to the same starting balance,
    and plots both the strategy and the benchmark on the same chart. Also renders
    a summary table with total return, Sharpe, and max drawdown for each.

    If yfinance fails (network error, no data), the S&P 500 line is omitted and
    only the strategy is shown.

    The table's two rows are annualized on DIFFERENT bases, because they are two
    different periodicities: the strategy row takes _sharpe's calendar default
    (its curve has one row per calendar day) while the ^GSPC row passes
    TRADING_DAYS_PER_YEAR, since yfinance serves trading days only (DR-56).
    Sharing one factor would make the comparison this section exists for
    apples-to-oranges by exactly sqrt(365/252) = 1.2035 of magnitude.

    Args:
        equity_df (pd.DataFrame): Daily equity curve with columns
            [date, portfolio_value, daily_return].
        start_date (date): The backtest's first trading date. The benchmark
            download opens one day earlier so its window starts on the equity
            curve's leading row. The alignment is of the WINDOW only: that
            leading row is flat by construction, while the S&P's first bar is a
            live trading day, so when start_date - 1 is itself a trading day the
            benchmark carries one extra day of market P&L the strategy cannot
            have. yfinance serves trading days only, so the first S&P bar can
            also land later than start_date - 1.
        initial_balance (float): Starting portfolio value in dollars.

    Returns:
        str: Self-contained HTML section string with comparison table and chart.
    """
    # Fetch S&P 500
    sp_raw = None
    try:
        # One day earlier, so the benchmark's WINDOW opens on the same date as
        # the strategy trace's leading flat row (see _build_equity_curve). Only
        # the window aligns: that row is flat by construction while the S&P's
        # first bar is a live trading day, so a start_date - 1 that is itself a
        # trading day moves the normalization anchor by one bar. yfinance
        # returns trading days only, so the first bar served can also be later.
        sp_raw = yf.download("^GSPC", start=(start_date - timedelta(days=1)).isoformat(),
                             progress=False, auto_adjust=True)
    except Exception as e:
        logging.warning("Could not fetch S&P 500: %s", e)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_df["date"], y=equity_df["portfolio_value"],
        name="Strategy", line={"color": _COLORS["strategy"], "width": 2},
    ))

    bench_rows: list[dict] = []

    if sp_raw is not None and not sp_raw.empty:
        # The whole benchmark computation degrades gracefully (per the section
        # docstring): yfinance's return shape varies by version — modern
        # releases use MultiIndex columns, making sp_raw["Close"] a DataFrame —
        # so any surprise here must not crash the dashboard.
        try:
            sp = sp_raw["Close"]
            if hasattr(sp, "columns"):
                # Single-ticker download with MultiIndex columns → flatten to a Series
                sp = sp.iloc[:, 0]
            sp = sp.dropna()
            sp_norm = sp / float(sp.iloc[0]) * initial_balance
            fig.add_trace(go.Scatter(
                x=sp.index, y=sp_norm.values,
                name="S&P 500 (normalized)", line={"color": _COLORS["sp500"], "width": 2},
            ))
            sp_ret = float(sp_norm.iloc[-1] / initial_balance - 1)
            sp_daily = sp.pct_change().dropna()
            bench_rows.append({
                "name": "S&P 500",
                "return": f"{sp_ret:+.1%}",
                # yfinance serves TRADING days only, so this is the one series
                # on the page that is not calendar-daily: it must override
                # _sharpe's calendar default or the two rows of this very table
                # would be annualized on different bases (DR-56).
                "sharpe": f"{_sharpe(sp_daily, periods_per_year=TRADING_DAYS_PER_YEAR):.2f}",
                "max_dd": f"{_max_drawdown(sp_norm)[0]:.1%}",
            })
        except Exception as e:
            logging.warning("Benchmark computation failed: %s — omitting S&P 500", e)

    # "Kalshi Arbitrage Strategy" / "Kalshi Arbitrage Backtest" (here, the
    # bold-row match below, and the page <title>/<h1>) are the PRODUCT NAME,
    # kept deliberately after the 2026-09 time-series inversion — the row name
    # here is string-matched by the table renderer below, so both must agree.
    # They are not a claim that the time-series leg is an arbitrage.
    bench_rows.insert(0, {"name": "Kalshi Arbitrage Strategy",
                          **_strategy_row(equity_df, initial_balance)})

    fig.update_layout(title="Strategy vs Benchmarks", yaxis_title="Portfolio Value ($)",
                      xaxis_title="Date")

    bench_table = """
<table style="font-family:sans-serif;font-size:14px;border-collapse:collapse;
              margin:16px 0; width:auto;">
<tr style="background:#E3F2FD; font-weight:bold;">
  <th style="padding:8px 16px;">Benchmark</th>
  <th style="padding:8px 16px;">Total Return</th>
  <th style="padding:8px 16px;">Sharpe</th>
  <th style="padding:8px 16px;">Max Drawdown</th>
</tr>
""" + "".join(
    f"<tr style='border-bottom:1px solid #E0E0E0'>"
    f"<td style='padding:8px 16px; font-weight:{'700' if r['name']=='Kalshi Arbitrage Strategy' else '400'}'>{r['name']}</td>"
    # The strategy row's cells carry ids, so the filter script can rewrite
    # them for another selection; the S&P row's never change
    + "".join(f"<td{_bench_cell_id(r['name'], field)} style='padding:8px 16px;'>{r[field]}</td>"
              for field in ("return", "sharpe", "max_dd"))
    + "</tr>"
    for r in bench_rows
) + "</table>"

    return (
        _SECTION_STYLE.format(title="Benchmark Comparison")
        + bench_table
        + _fig_html(fig, height=450, div_id="bench-fig")
    )


# ─── Page-wide filter: spread band x tier floors x Kalshi category x tag ─────

# The view every trade list carries: all of its trades, no category or tag.
_ALL_VIEW = "all"

# The two sections the filter bar does NOT drive, named once, for the bar's
# note and its tests: the interval-discount section (its own k dropdown, at
# the primary band) and the scenario explorer (its own band and k selects).
_UNFILTERED_SECTIONS = "Interval Discount (k) Calibration and the Scenario Explorer"

# The filter bar's "Tier floors" options: each band's run as simulated (the
# tiers applied, config.min_price_diff_for_gap's rule) or its run with them
# off, where the band's own floor alone gates the spread (BacktestSweep's
# tier-off family). Kept short, so the sticky bar is less likely to wrap; the
# select's title (_TIER_SELECT_TITLE) spells the rule out. Both read the tiers
# from config (as _TIER_FLOORS, above, does), never as literals.
_TIER_OPTION_ON = (f"on ({_exact_label(MIN_PRICE_DIFF_SHORT_GAP, '.2f')} / "
                   f"{_exact_label(MIN_PRICE_DIFF_LONG_GAP, '.2f')} by deadline gap)")
_TIER_OPTION_OFF = "off (each band's own floor alone)"

# The Tier floors select's tooltip (its title attribute, escaped where the bar
# renders it): what each setting admits, and that the choice is a backtest
# what-if — every number in it read from config.
_TIER_SELECT_TITLE = (
    "on — a time-series pair needs pB − pA of at least "
    f"{_exact_label(MIN_PRICE_DIFF_SHORT_GAP, '.2f')} when its deadlines are up to "
    f"{SHORT_DEADLINE_GAP_DAYS} days apart and {_exact_label(MIN_PRICE_DIFF_LONG_GAP, '.2f')} "
    f"for {SHORT_DEADLINE_GAP_DAYS + 1}–{MAX_DEADLINE_GAP_DAYS} days, and at least the "
    "band's floor; off — the band's floor alone (the spread must still be positive); a "
    "backtest what-if: live trading always applies the tier floors.")

# The filter bar's summary line, as templates: _filter_summary_text fills them
# for the page as rendered and the page's script fills them (D.text) for every
# other selection, so the two can never word one selection differently.
# {scenario} is a band's _band_scenario (or, with the tier floors off, its
# _tier_off_scenario), {count} and {band_count} a _trade_count, {n} a bare
# count and {selection} a category or "Category · Tag". "other_band" and
# "other_run" are the closing notes the payload names per band (its "note"):
# a band other than the primary is its own simulation, and so is the primary
# band's run with the tier floors off wherever they bind there.
_SUMMARY_TEMPLATES = {
    "all": "Showing every trade of the run at {scenario}: {count}.",
    "other_band": " This band is its own simulation, not a slice of the primary run.",
    "other_run": " This is its own simulation, not a slice of the primary run.",
    "slice": ("Showing {selection} within the run at {scenario}: {n} of its {band_count}. "
              "Every figure drawn from an equity curve (return, drawdown, Sharpe, "
              "Sortino, the median monthly return, the benchmark's strategy row) is this "
              "selection's contribution: the starting balance plus these trades' P&L as "
              "that run booked it, not a standalone simulation."),
    "unfiltered": f" Not filtered by this bar: {_UNFILTERED_SECTIONS}.",
}

# Shown in the bar's place when the filter's data cannot be built (DR-66: the
# absence of a control must not be the only sign that it failed)
_FILTER_UNAVAILABLE_HTML = (
    '<p id="flt-unavailable" style="color:#B71C1C; font-size:14px; font-family:sans-serif;">'
    "The page-wide filter could not be built for this run (the log names the error); "
    "every section shows the primary spread band's full run.</p>")


@dataclass(frozen=True)
class _BandRun:
    """
    One spread band's own run at the run's primary k, with the deadline-gap
    tier floors on (as simulated, _band_runs) or, as a tier-floors-off view,
    off (_tier_off_runs): what choosing that band and that Tier floors
    setting in the page's filter bar shows.

    Attributes:
        band (tuple[float, float] | None): The resolved band, or None when the
            run recorded none (no sweep was passed, or a hand-built one).
        label (str): How the band reads on the page (_row_label, or "not
            recorded"; a tier-floors-off view's is _band_label's bare
            "floor-ceiling", since its floor alone gated it).
        trades (list[BacktestTrade]): That band's "all"-population trades — for
            the primary band's tier-on run, the very list every section
            renders by default; for a binding band's tier-off view, its
            tier-off "all" point's trades at the primary k.
        equity_df (pd.DataFrame): That band's own standalone equity curve (a
            binding band's tier-off view: its tier-off point's curve).
        calibration (IntervalCalibration | None): That band's k-hat
            measurement, whose observations the k-hat breakdown regroups (a
            binding band's tier-off view: its tier-off calibration,
            BacktestSweep.tier_off_calibrations_by_band).
        same_as_tier_on (bool): True only on a tier-floors-off view
            (_tier_off_runs) of a band whose floor sits at or above both
            deadline-gap tiers: the tiers never bind there, so the run was
            not simulated again and this IS its tier-on run (same trades,
            curve and calibration), relabelled. Appended with a default of
            False, so a positional five-argument construction still builds.
    """
    band: tuple[float, float] | None
    label: str
    trades: list
    equity_df: pd.DataFrame
    calibration: IntervalCalibration | None
    same_as_tier_on: bool = False


def _band_runs(
    sweep: BacktestSweep | None,
    trades: list[BacktestTrade],
    equity_df: pd.DataFrame,
) -> tuple[list[_BandRun], int]:
    """
    List every spread band the filter bar offers, each with its own run.

    A band sweep simulates every band at every k; the band the bar shows is
    each band's "all"-population point at the run's PRIMARY k — the result of
    running the backtest at that band, same-title trades included — so a
    band choice is a real, standalone simulation, never a slice. The primary
    band's run is the trades and curve the rest of the page renders (the
    caller's own arguments), not a copy looked up again. These are the runs
    with the tier floors on: a point stamped tier_floors False (the
    tier-floors-off family's, which _tier_off_runs reads) is never taken for
    one, just as _tier_off_runs never takes a tier-on point for a tier-off
    run.

    Args:
        sweep (BacktestSweep | None): The run's sweep, or None.
        trades (list[BacktestTrade]): The trades the page renders by default
            (the primary scenario's).
        equity_df (pd.DataFrame): Their equity curve.

    Returns:
        tuple[list[_BandRun], int]: The runs in ascending band order, and the
            index of the primary one. A single run labelled "not recorded"
            when there is no sweep or its primary records no band; the primary
            band alone when the band sweep was off.
    """
    if sweep is None or sweep.primary.spread_band is None:
        calibration = None if sweep is None else sweep.calibration
        return [_BandRun(None, "not recorded", trades, equity_df, calibration)], 0
    primary_band = sweep.primary.spread_band
    others = {pt.spread_band: pt for pt in sweep.scenarios
              if pt.population == "all" and pt.k == sweep.primary.k
              and pt.spread_band is not None and pt.spread_band != primary_band
              and pt.tier_floors is not False}
    bands = sorted({primary_band, *others})
    runs = []
    for band in bands:
        if band == primary_band:
            runs.append(_BandRun(band, _row_label(band), trades, equity_df,
                                 sweep.calibrations_by_band.get(band, sweep.calibration)))
        else:
            point = others[band]
            runs.append(_BandRun(band, _row_label(band), point.trades, point.equity_df,
                                 sweep.calibrations_by_band.get(band)))
    return runs, bands.index(primary_band)


def _tier_off_binds(sweep: BacktestSweep | None, bands: list) -> list[bool] | None:
    """
    Say, per band, whether its tier-floors-off view is a simulation of its own.

    The band sweep's tier-off family (BacktestSweep.tier_off_scenarios)
    simulates again exactly the bands where a deadline-gap tier floor binds,
    and its calibrations' keys (tier_off_calibrations_by_band) are exactly
    those bands. A band outside them whose floor sits at or above both tiers
    (backtester._tier_floors_bind False) enters the same pairs on the same
    Mondays either way, so its tier-on run IS its tier-off run. Anything else
    fails CLOSED — the whole off view is withheld rather than one band's
    guessed: a view the run did not simulate is never shown as if it had
    been.

    Args:
        sweep (BacktestSweep | None): The run's sweep, or None.
        bands (list): Each offered band's resolved (floor, ceiling), in the
            filter bar's order — None for an unrecorded band.

    Returns:
        list[bool] | None: Per band, True when the run simulated it again with
            the tier floors off, False when the tiers never bind there.
            Returns None — no off view anywhere — when there is no sweep, it
            carries no tier-off family, a band is unrecorded, or a band the
            tiers bind at is missing from the family.
    """
    if sweep is None or not sweep.tier_off_scenarios:
        return None
    binds = []
    for band in bands:
        if band is None:
            return None
        if band in sweep.tier_off_calibrations_by_band:
            binds.append(True)
        # The one test of "a tier sits above this floor", shared with the
        # backtester that decided which bands to simulate again
        elif _tier_floors_bind(band):
            return None
        else:
            binds.append(False)
    return binds


def _tier_off_runs(sweep: BacktestSweep | None,
                   runs: list[_BandRun]) -> list[_BandRun] | None:
    """
    Each offered band's run with the deadline-gap tier floors off, parallel to _band_runs' runs.

    The filter bar's "Tier floors: off" choice shows these. A band the tiers
    bind at shows its tier-off "all"-population point at the run's PRIMARY k
    — a standalone simulation of the entries detected at the band's floor
    alone (config.min_price_diff_for_gap(..., tier_floors=False)) — with that
    band's tier-off calibration. A band the tiers never bind at shows its own
    tier-on run, the same trades, curve and calibration objects, marked
    same_as_tier_on. Both are labelled with backtester._band_label's bare
    "floor-ceiling" (the same text the backtest log names the band with):
    with the tiers off the floor is the only floor, so _row_label's
    "max(tier,<floor>)" would misname it.

    Args:
        sweep (BacktestSweep | None): The run's sweep, or None.
        runs (list[_BandRun]): _band_runs' runs, in band order.

    Returns:
        list[_BandRun] | None: One run per entry of runs, in the same order.
            Returns None — no off view — when _tier_off_binds is None, or a
            band the tiers bind at has no tier-off "all" point at the primary
            k (a point not stamped tier_floors False is never read as one).
    """
    binds = _tier_off_binds(sweep, [run.band for run in runs])
    if binds is None:
        return None
    # Each binding band's tier-off run at the primary k, as _band_runs picks
    # the tier-on ones: the "all" population, never another k's point
    points = {pt.spread_band: pt for pt in sweep.tier_off_scenarios
              if pt.population == "all" and pt.k == sweep.primary.k
              and pt.tier_floors is False}
    twins = []
    for run, own in zip(runs, binds, strict=True):
        # The backtester's own band text: the floor alone gated this run
        label = _band_label(run.band)
        if not own:
            twins.append(replace(run, label=label, same_as_tier_on=True))
            continue
        point = points.get(run.band)
        if point is None:
            return None
        twins.append(_BandRun(run.band, label, point.trades, point.equity_df,
                              sweep.tier_off_calibrations_by_band[run.band]))
    return twins


def _sparse_on_axis(dates, values, axis: pd.DatetimeIndex, ndigits: int) -> list[list]:
    """
    Place a series on the page's date axis, by date, keeping only its change points.

    Every curve on the page is flat between trade dates, so shipping one point
    per change instead of one per day is what keeps a view per band x
    category x tag affordable; the filter script expands it back
    (expand(): each value holds until the next change point). A date the
    series does not have, or a non-finite value, is None — a gap in the line.

    Args:
        dates: The series' dates, one per value — a pd.DatetimeIndex (used
            as is, so a caller placing several series of one curve parses
            its dates once) or any iterable of dates or timestamps.
        values: The series' values.
        axis (pd.DatetimeIndex): The page's shared date axis.
        ndigits (int): Decimal places kept — 2 for dollars, 4 for percent.

    Returns:
        list[list]: [[axis index, value], ...] ascending, always starting at
            index 0 (an empty list for an empty axis).
    """
    if not len(axis):
        return []
    index = (dates if isinstance(dates, pd.DatetimeIndex)
             else pd.DatetimeIndex(pd.to_datetime(list(dates))))
    s = pd.Series(np.asarray(values, dtype=float), index=index)
    s = s[~s.index.duplicated(keep="last")]
    arr = np.round(s.reindex(axis).to_numpy(dtype=float), ndigits)
    finite = np.isfinite(arr)
    # A non-finite value compares equal to the next one here (both inf), so a
    # run of gaps is one change point, like a run of any other value.
    comparable = np.where(finite, arr, np.inf)
    change = np.ones(len(arr), dtype=bool)
    change[1:] = comparable[1:] != comparable[:-1]
    return [[int(i), float(arr[i]) if finite[i] else None] for i in np.flatnonzero(change)]


class _StringTable:
    """
    Deduplicated HTML fragments for the filter payload: each distinct string
    is shipped once and referred to by index (the same trade row appears in
    many views' best/worst tables, and many views share a category table).
    """

    def __init__(self) -> None:
        """Start with no fragment stored."""
        self.items: list[str] = []
        self._index: dict[str, int] = {}

    def add(self, text: str) -> int:
        """
        Store a fragment once and return its index.

        Args:
            text (str): Already-escaped HTML.

        Returns:
            int: The fragment's index in self.items.
        """
        i = self._index.get(text)
        if i is None:
            i = self._index[text] = len(self.items)
            self.items.append(text)
        return i


def _view_payload(
    sel: list[BacktestTrade],
    idx: list[int],
    equity_df: pd.DataFrame,
    axis: pd.DatetimeIndex,
    initial_balance: float,
    series_categories: dict[str, tuple[str, tuple[str, ...]]] | None,
    kelly_x: list[float],
    row_of,
    strings: _StringTable,
) -> dict:
    """
    Compute everything the filtered sections show for one selection of trades.

    Every figure comes from the SAME helper the server-rendered section uses
    (_performance_kpis, _performance_series, _decomposition_aggregates,
    _category_table, _reliability, _best_and_worst, _capital_deployed,
    _strategy_row), so a view and the section it redraws cannot disagree on a
    definition — the filter script only draws what this computes.

    Args:
        sel (list[BacktestTrade]): The selection's trades, in list order.
        idx (list[int]): Their indexes in the band's trade list, which the
            script uses to pick the per-trade arrays (histograms, scatter).
        equity_df (pd.DataFrame): The selection's equity curve: the band's own
            curve for the whole band, the attributed curve for a slice.
        axis (pd.DatetimeIndex): The page's shared date axis.
        initial_balance (float): Starting balance in dollars.
        series_categories (dict | None): The series-category map.
        kelly_x (list[float]): The band list's per-trade Kelly fractions.
        row_of: Callable (trade, which) -> string index of that trade's
            best ("best") or worst ("worst") table row.
        strings (_StringTable): Where HTML fragments are stored.

    Returns:
        dict: "n", "idx", "kpi" (card key -> formatted value), the sparse
            series "total", "types", "dd", "eq" and "dep", "bench" (the
            benchmark strategy row), and — only with at least one trade —
            "monthly", "cat", "sub", "price" (bar specs: x, y, colours c),
            "table" (string index), "cal", "best", "worst" and "k11".
    """
    if len(axis):
        # A curve built after the page's own (a slice's, built here) can run
        # one day further when midnight UTC passed in between; the page's axis
        # decides where every curve ends, for the figures AND the metrics.
        equity_df = equity_df[pd.to_datetime(equity_df["date"]) <= axis[-1]]
    # Parsed once: every series below is placed on the axis by these dates
    dates = pd.DatetimeIndex(pd.to_datetime(list(equity_df["date"])))
    total, type_lines, drawdown = _performance_series(equity_df, sel, initial_balance)
    view = {
        "n": len(sel),
        "idx": idx,
        "kpi": {key: value for key, _, value, _ in
                _performance_kpis(equity_df, sel, initial_balance)},
        "total": _sparse_on_axis(dates, total, axis, 4),
        "types": [[label, _sparse_on_axis(dates, series, axis, 4)]
                  for label, _, series in type_lines],
        "dd": _sparse_on_axis(dates, drawdown, axis, 4),
        "eq": _sparse_on_axis(dates, equity_df["portfolio_value"], axis, 2),
        "dep": _sparse_on_axis(dates, _capital_deployed(sel, equity_df), axis, 2),
        "bench": _strategy_row(equity_df, initial_balance),
    }
    if not sel:
        return view

    df = _decomposition_frame(sel, series_categories)
    agg = _decomposition_aggregates(df)
    monthly, cat, sub, price = agg["monthly"], agg["category"], agg["subcategory"], agg["price"]
    view["monthly"] = {"x": [str(m) for m in monthly["month"]],
                       "y": [float(v) for v in monthly["profit"]],
                       "c": _pnl_colors(monthly["profit"])}
    view["cat"] = {"x": [float(v) for v in cat.values], "y": [str(c) for c in cat.index],
                   "c": _pnl_colors(cat.values)}
    view["sub"] = {"x": [float(v) for v in sub.values], "y": [str(c) for c in sub.index],
                   "c": _pnl_colors(sub.values), "h": _subcategory_chart_height(len(sub))}
    view["price"] = {"x": [str(b) for b in price.index], "y": [float(v) for v in price.values],
                     "c": _pnl_colors(price.values)}
    view["table"] = strings.add(_category_table(df))

    rel = _reliability(sel)
    view["cal"] = {
        "brier": f"{rel['brier']:.4f}", "log_loss": f"{rel['log_loss']:.4f}",
        "title": _calibration_title(rel["brier"], rel["log_loss"]),
        "x": [float(v) for v in rel["mean_pred"]], "y": [float(v) for v in rel["mean_act"]],
        "size": rel["sizes"], "text": rel["texts"],
    }
    best, worst = _best_and_worst(sel)
    view["best"] = [row_of(t, "best") for t in best]
    view["worst"] = [row_of(t, "worst") for t in worst]
    view["k11"] = _one_to_one_extent([kelly_x[i] for i in idx])
    return view


def _list_payload(
    trades: list[BacktestTrade],
    equity_df: pd.DataFrame,
    axis: pd.DatetimeIndex,
    start_date: date,
    initial_balance: float,
    series_categories: dict[str, tuple[str, tuple[str, ...]]] | None,
    k: float | None,
    cat_index: dict[str, int],
    sub_index: dict[tuple[str, str], int],
    strings: _StringTable,
) -> dict:
    """
    Compute one distinct trade list's per-trade arrays and every view of it.

    A view exists for the whole list and for each category and category ·
    tag that has at least one trade in it. A slice's equity curve is
    backtester._build_equity_curve over the slice's trades alone — the
    starting balance plus those trades' P&L on the days the run booked it —
    so its return, drawdown and Sharpe are the slice's CONTRIBUTION to the
    run, not a standalone simulation (the sizes are the joint run's).

    Args:
        trades (list[BacktestTrade]): The band's trades.
        equity_df (pd.DataFrame): The band's own equity curve.
        axis (pd.DatetimeIndex): The page's shared date axis.
        start_date (date): The backtest's start date (a slice's curve opens
            the day before it, like every curve on the page).
        initial_balance (float): Starting balance in dollars.
        series_categories (dict | None): The series-category map.
        k (float | None): The interval discount the trades were sized at, for
            the Kelly scatter (_kelly_points).
        cat_index (dict[str, int]): Category -> its index in the payload.
        sub_index (dict[tuple[str, str], int]): (category, tag) -> its index.
        strings (_StringTable): Where HTML fragments are stored.

    Returns:
        dict: Per-trade arrays "ret" (return in percent), "slip", "hold",
            "kx", "ky" and "kt" (Kelly scatter x, y and escaped hover text),
            and "views": view key ("all", "c<category index>", "s<category ·
            tag index>") -> _view_payload.
    """
    kelly_x, kelly_y = _kelly_points(trades, k)
    position = {id(t): i for i, t in enumerate(trades)}
    rows: dict[tuple[int, str], int] = {}

    def row_of(trade: BacktestTrade, which: str) -> int:
        """
        Store a trade's best- or worst-table row once and return its index.

        Args:
            trade (BacktestTrade): One of this list's trades.
            which (str): "best" or "worst" (the row's background colour).

        Returns:
            int: The row's index in `strings`.
        """
        key = (position[id(trade)], which)
        if key not in rows:
            color = _BEST_ROW_COLOR if which == "best" else _WORST_ROW_COLOR
            rows[key] = strings.add(_trade_row(trade, color))
        return rows[key]

    groups: dict[str, list[int]] = {_ALL_VIEW: list(range(len(trades)))}
    for i, t in enumerate(trades):
        category, tag = _series_labels(t.event_ticker, t.category, series_categories)
        groups.setdefault(f"c{cat_index[category]}", []).append(i)
        groups.setdefault(f"s{sub_index[(category, tag)]}", []).append(i)

    views = {}
    for key, idx in groups.items():
        sel = [trades[i] for i in idx]
        # A slice's curve comes from backtester's one definition of a curve,
        # over the slice's trades alone: its contribution to the band's run
        curve = (equity_df if key == _ALL_VIEW
                 else _build_equity_curve(sel, start_date, initial_balance))
        views[key] = _view_payload(sel, idx, curve, axis, initial_balance,
                                   series_categories, kelly_x, row_of, strings)
    return {
        # The histogram's x values, exactly as _section_diagnostics draws them
        "ret": [t.profit_ratio * 100 for t in trades],
        "slip": [t.slippage for t in trades],
        "hold": [t.holding_days for t in trades],
        "kx": kelly_x, "ky": kelly_y,
        # Hover text renders an HTML subset: escaped like _section_risk's
        "kt": [html.escape(t.title_a[:40]) for t in trades],
        "views": views,
    }


def _filter_payload(
    runs: list[_BandRun],
    primary_idx: int,
    start_date: date,
    initial_balance: float,
    series_categories: dict[str, tuple[str, tuple[str, ...]]] | None,
    k: float | None,
    k_text: str,
    *,
    off_runs: list[_BandRun] | None = None,
) -> dict:
    """
    Build the data block the page's filter bar and script read.

    Bands whose trade lists are equal share one list (and its views), which is
    what keeps a band sweep whose bands rarely differ cheap: on a run with no
    time-series trade every band's list is the same same-title list. The
    primary band's list is built first, so it keeps the page's own curve;
    then every other band's; then each band's tier-floors-off run, so a band
    the tiers never bind at (whose off run IS its tier-on run) shares its
    band's list, and a tier-off run that traded exactly what some band did
    shares that list too.

    Args:
        runs (list[_BandRun]): _band_runs' runs.
        primary_idx (int): The primary run's index.
        start_date (date): The backtest's start date.
        initial_balance (float): Starting balance in dollars.
        series_categories (dict | None): The series-category map.
        k (float | None): The interval discount the trades were sized at.
        k_text (str): How that k reads on the page ("k = 0.75", or "k not
            recorded"), for each band's summary scenario.
        off_runs (list[_BandRun] | None): Keyword-only. _tier_off_runs' runs
            — each band's run with the deadline-gap tier floors off, parallel
            to runs — or None (default) when the run has no such view; the
            payload then carries none ("bands_off" and "khat_off" are None)
            and the bar's Tier floors select stays disabled.

    Returns:
        dict: "dates" (the shared axis, ISO dates), "bands" ([{label, option,
            list, scenario, where, note}] — the band's label, its option text
            in the bar (" (primary)" on the primary), its list, _band_scenario's
            and _band_where's phrases, and the whole-run view's closing note:
            None on the primary band, "other_band" on every other),
            "bands_off" (None, or the same entries per band for its tier-off
            run — _tier_off_scenario's and _tier_off_where's phrases, and a
            note that is None only where the primary band's tier-off run IS
            the primary run, "other_run" for the primary band's own tier-off
            simulation and "other_band" on every other band),
            "primary", "categories" (sorted names — every trade's, and every
            k-hat observation's, tier-off runs' included), "subcats"
            ([[category index, tag], ...], sorted), "lists" (_list_payload
            per distinct list), "empty" (the view of a selection with no trade
            — a flat curve), "strings" (HTML fragments), "text" (the
            templates: _SUMMARY_TEMPLATES and _KHAT_TEXT), "styles" (the
            trade-type lines' and the k-hat bars' drawing, and the k-hat
            chart's height formula), "khat" (_khat_band per band, in band
            order), "khat_off" (None, or _khat_band per band's tier-off run)
            and "khat_blank" (the table cells of a group with no k-hat).

    Raises:
        ValueError: If off_runs is not parallel to runs (one per band, the
            same bands in the same order) — a view filed under the wrong band
            must never reach the page (generate_dashboard then writes the page
            without the bar).
    """
    if off_runs is not None and [r.band for r in off_runs] != [r.band for r in runs]:
        raise ValueError("off_runs must hold one run per band of runs, in the same order")
    axis = pd.DatetimeIndex(pd.to_datetime(list(runs[primary_idx].equity_df["date"])))
    strings = _StringTable()
    # Every run the bar can show: each band's, then each band's tier-off run.
    # One category and tag list serves both Tier floors settings, so the
    # options keep their indices across a toggle — a category only a tier-off
    # run trades is offered (at 0) with the tiers on too
    every = runs + (off_runs or [])

    pairs = {_series_labels(t.event_ticker, t.category, series_categories)
             for run in every for t in run.trades}
    # A category or tag seen only in some band's k-hat population is offered
    # too: the k-hat breakdown shows it even where no trade was made
    pairs.update(_series_labels(o.event_ticker, o.category, series_categories)
                 for run in every if run.calibration is not None
                 for o in run.calibration.observations)
    categories = sorted({c for c, _ in pairs})
    cat_index = {c: i for i, c in enumerate(categories)}
    subcats = sorted(pairs)
    sub_index = {pair: i for i, pair in enumerate(subcats)}

    # Distinct lists, the primary's first so it keeps the page's own curve,
    # then every other band's, then the tier-off runs'
    sources: list[tuple[list, pd.DataFrame]] = []
    run_list = [0] * len(every)
    order = [primary_idx, *(j for j in range(len(runs)) if j != primary_idx),
             *range(len(runs), len(every))]
    for i in order:
        for li, (listed, _) in enumerate(sources):
            if listed == every[i].trades:
                run_list[i] = li
                break
        else:
            run_list[i] = len(sources)
            sources.append((every[i].trades, every[i].equity_df))

    lists = [_list_payload(listed, curve, axis, start_date, initial_balance,
                           series_categories, k, cat_index, sub_index, strings)
             for listed, curve in sources]
    # A selection with no trade: the flat curve backtester draws for no trade
    empty = _view_payload([], [], _build_equity_curve([], start_date, initial_balance),
                          axis, initial_balance, series_categories, [], None, strings)

    def option(label: str, primary: bool) -> str:
        """
        One band's option text in the bar's Spread band select.

        Args:
            label (str): The band's label under the current tier setting.
            primary (bool): Whether it is the run's primary band.

        Returns:
            str: The label, with " (primary)" on the primary band.
        """
        return label + (" (primary)" if primary else "")

    bands_off = None
    if off_runs is not None:
        bands_off = []
        for i, run in enumerate(off_runs):
            primary = i == primary_idx
            # The primary band's tier-off run is the primary run itself only
            # where the tiers never bind; elsewhere it is its own simulation
            note = ((None if run.same_as_tier_on else "other_run") if primary
                    else "other_band")
            bands_off.append({
                "label": run.label, "option": option(run.label, primary),
                "list": run_list[len(runs) + i],
                "scenario": _tier_off_scenario(run.label, primary, run.same_as_tier_on,
                                               k_text),
                "where": _tier_off_where(run.label, primary, run.same_as_tier_on),
                "note": note})
    return {
        "dates": [d.date().isoformat() for d in axis],
        "bands": [{"label": run.label, "option": option(run.label, i == primary_idx),
                   "list": run_list[i],
                   "scenario": _band_scenario(run.label, i == primary_idx,
                                              run.band is not None, k_text),
                   "where": _band_where(run.label, i == primary_idx, run.band is not None),
                   # Today's rule: every band but the primary is its own simulation
                   "note": None if i == primary_idx else "other_band"}
                  for i, run in enumerate(runs)],
        "bands_off": bands_off,
        "primary": primary_idx,
        "categories": categories,
        "subcats": [[cat_index[c], tag] for c, tag in subcats],
        "lists": lists,
        "empty": empty,
        "strings": strings.items,
        "text": {**_SUMMARY_TEMPLATES, **_KHAT_TEXT},
        "styles": {"types": {label: {"color": color, "width": _TYPE_LINE_WIDTH,
                                     "dash": _TYPE_LINE_DASH}
                             for label, color in _TRADE_TYPE_LINES},
                   # The k-hat chart's bars — the grouping's whole population,
                   # the filter's current choice, every other group — and its
                   # height formula (_khat_chart_height)
                   "khat": {"all": _COLORS["naive"], "selected": _COLORS["sp500"],
                            "bar": _COLORS["strategy"]},
                   "khat_height": list(_KHAT_HEIGHT)},
        # Per band, its k-hat population broken down like its trades
        # (_khat_band), and the cells of a group with none
        "khat": [_khat_band(run.calibration, series_categories, cat_index, sub_index)
                 for run in runs],
        # The same per band's tier-off run (a band the tiers never bind at
        # regroups its own calibration)
        "khat_off": (None if off_runs is None else
                     [_khat_band(run.calibration, series_categories, cat_index, sub_index)
                      for run in off_runs]),
        "khat_blank": _khat_cells(None),
    }


def _trade_count(n: int) -> str:
    """
    Count trades in words, as the summary line does.

    Args:
        n (int): A number of trades.

    Returns:
        str: "1 trade" or "N trades" (the filter script's trades() is the same).
    """
    return f"{n} trade" if n == 1 else f"{n} trades"


def _band_where(label: str, primary: bool, recorded: bool) -> str:
    """
    Name a spread band on the page, as the filter bar's summary line reads it.

    Args:
        label (str): The band's label.
        primary (bool): Whether it is the run's primary band.
        recorded (bool): Whether the run recorded a band at all (False for a
            dashboard built without a sweep, or from a hand-built one).

    Returns:
        str: "the primary spread band <label>" or "spread band <label>"; "the
            primary spread band (not recorded)" when no band was recorded.
    """
    if not recorded:
        return "the primary spread band (not recorded)"
    which = "the primary spread band" if primary else "spread band"
    return f"{which} {label}"


def _band_scenario(label: str, primary: bool, recorded: bool, k_text: str) -> str:
    """
    Name the run a spread band choice shows: the band and the run's k.

    Args:
        label (str): The band's label.
        primary (bool): Whether it is the run's primary band.
        recorded (bool): Whether the run recorded a band at all.
        k_text (str): "k = 0.75", or "k not recorded".

    Returns:
        str: _band_where's phrase, then ", " and k_text — e.g. "the primary
            spread band max(tier,0)-1, k = 0.75".
    """
    return f"{_band_where(label, primary, recorded)}, {k_text}"


def _tier_off_where(label: str, primary: bool, same_as_tier_on: bool) -> str:
    """
    Name a spread band's tier-floors-off run, as _band_where names its run.

    Args:
        label (str): The band's tier-off label (backtester._band_label's
            "0.2-0.6": with the tiers off, the floor is the only floor).
        primary (bool): Whether it is the run's primary band.
        same_as_tier_on (bool): Whether the tiers never bind at this band, so
            its tier-off run IS its tier-on run (_BandRun.same_as_tier_on).

    Returns:
        str: "spread band 0.2-0.6 with the 0.15/0.30 tier floors off" ("the
            primary spread band ..." on the primary), plus " (they never bind
            at this band: its run with them on)" where the tiers never bind.
    """
    which = "the primary spread band" if primary else "spread band"
    where = f"{which} {label} with the {_TIER_FLOORS} tier floors off"
    if same_as_tier_on:
        where += " (they never bind at this band: its run with them on)"
    return where


def _tier_off_scenario(label: str, primary: bool, same_as_tier_on: bool, k_text: str) -> str:
    """
    Name the run a band's tier-floors-off view shows: the band and the run's k.

    Args:
        label (str): The band's tier-off label.
        primary (bool): Whether it is the run's primary band.
        same_as_tier_on (bool): Whether the tiers never bind at this band.
        k_text (str): "k = 0.75", or "k not recorded".

    Returns:
        str: _tier_off_where's phrase, then ", " and k_text, as _band_scenario
            joins its own — e.g. "the primary spread band 0-1 with the
            0.15/0.30 tier floors off, k = 0.75".
    """
    return f"{_tier_off_where(label, primary, same_as_tier_on)}, {k_text}"


def _filter_summary_text(text: dict, scenario: str, primary: bool,
                         selection: str | None, n: int, n_band: int, *,
                         note: str | None = None) -> str:
    """
    Say, under the filter bar, what the page is showing.

    Fills the same templates the page's script fills for every other
    selection (_FILTER_JS's summary(), from D.text); this renders the primary
    band's unfiltered view, so the page reads correctly before anything is
    chosen.

    Args:
        text (dict): The templates (_SUMMARY_TEMPLATES, as the payload
            carries them).
        scenario (str): The band's _band_scenario phrase (a tier-off view's
            _tier_off_scenario).
        primary (bool): Whether it is the run's primary band.
        selection (str | None): "Sports" or "Sports · Basketball", or None for
            the whole band.
        n (int): Trades in the selection.
        n_band (int): Trades in the whole band.
        note (str | None): Keyword-only. The whole-band view's closing note,
            as the payload names it for the view shown (its "note":
            "other_band" or "other_run", the key of a template) — what the
            script appends. None (default) applies the rule the tier-on bands
            carry: "other_band" on every band but the primary, nothing on it.

    Returns:
        str: Plain text — escape it before putting it in HTML.
    """
    if selection is None:
        out = text["all"].format(scenario=scenario, count=_trade_count(n))
        closing = note if note is not None else (None if primary else "other_band")
        if closing is not None:
            out += text[closing]
    else:
        out = text["slice"].format(scenario=scenario, selection=selection, n=n,
                                   band_count=_trade_count(n_band))
    return out + text["unfiltered"]


def _filter_bar_html(payload: dict) -> str:
    """
    Render the sticky filter bar: four <select>s and a summary line.

    Options carry the primary band's trade counts; the script rewrites them on
    every band or tier change, and each band's name (Python's "option" text)
    on every tier change. Tag options list every "Category · Tag" while the
    category is "All"; choosing one sets the category to match. The Tier
    floors select offers each band's run as simulated ("on", selected) or its
    tier-floors-off run ("off"), its title spelling out both rules
    (_TIER_SELECT_TITLE); a payload with no tier-off view ("bands_off" None)
    puts a grey "(not simulated for this run)" note beside it, and the
    script never enables it. The selects
    are rendered DISABLED, and autocomplete="off" so a browser does not
    restore a stale choice on reload: the script enables them once it has
    inflated the data, so without it (or without a browser that can inflate
    it) they cannot promise a view the page will not show.

    Args:
        payload (dict): _filter_payload's output.

    Returns:
        str: The bar's HTML, every Kalshi-controlled name escaped.
    """
    primary = payload["primary"]
    views = payload["lists"][payload["bands"][primary]["list"]]["views"]

    def count(key: str) -> int:
        """
        Trades in one view of the primary band's list.

        Args:
            key (str): A view key ("all", "c<i>", "s<i>").

        Returns:
            int: The view's trade count; 0 when the band has no such view.
        """
        view = views.get(key)
        return view["n"] if view else 0

    band_opts = "".join(
        f'<option value="{i}"{" selected" if i == primary else ""}>'
        f'{html.escape(b["option"])}</option>'
        for i, b in enumerate(payload["bands"]))
    # Each band's run as simulated, or with the deadline-gap tier floors off
    tier_opts = (f'<option value="on" selected>{html.escape(_TIER_OPTION_ON)}</option>'
                 f'<option value="off">{html.escape(_TIER_OPTION_OFF)}</option>')
    # A run with no tier-off view says so beside the select it keeps shut
    tier_note = ("" if payload.get("bands_off") is not None else
                 '&nbsp;<span id="flt-tier-note" style="color:#9E9E9E; font-size:13px;">'
                 "(not simulated for this run)</span>")
    cat_opts = '<option value="">All categories</option>' + "".join(
        f'<option value="{i}">{html.escape(c)} ({count(f"c{i}")})</option>'
        for i, c in enumerate(payload["categories"]))
    tag_opts = '<option value="">All tags</option>' + "".join(
        f'<option value="{i}">{html.escape(payload["categories"][ci] + " · " + tag)} '
        f'({count(f"s{i}")})</option>'
        for i, (ci, tag) in enumerate(payload["subcats"]))
    # The primary band's own closing note (none), as the script appends a
    # view's — one rule for the rendered line and every redrawn one
    summary = html.escape(_filter_summary_text(
        payload["text"], payload["bands"][primary]["scenario"], True, None,
        count(_ALL_VIEW), count(_ALL_VIEW), note=payload["bands"][primary]["note"]))
    # Escaped whole, quotes included, so the attribute can never end early
    tier_title = html.escape(_TIER_SELECT_TITLE)
    return (
        '<div id="flt-bar" style="position:sticky; top:0; z-index:1000; background:#FFFFFF;'
        ' border-bottom:1px solid #E0E0E0; padding:10px 0 8px; font-family:sans-serif;'
        ' font-size:14px;">'
        f'<label>Spread band: <select id="flt-band" disabled autocomplete="off">'
        f'{band_opts}</select></label>&nbsp;&nbsp;'
        f'<label>Tier floors: <select id="flt-tier" disabled autocomplete="off" '
        f'title="{tier_title}">{tier_opts}</select></label>{tier_note}&nbsp;&nbsp;'
        f'<label>Category: <select id="flt-cat" disabled autocomplete="off">'
        f'{cat_opts}</select></label>&nbsp;&nbsp;'
        f'<label>Tag: <select id="flt-tag" disabled autocomplete="off">'
        f'{tag_opts}</select></label>'
        '<div id="flt-summary" style="color:#616161; font-size:13px; margin-top:6px;">'
        f'{summary}</div></div>'
    )


def _packed_json_script(element_id: str, payload: dict) -> str:
    """
    Embed a payload as gzip-compressed, base64-encoded strict JSON.

    The filter payload holds a view per band x category x tag (per distinct
    trade list — a band's tier-floors-off run adds one only where it traded
    differently), and its largest part is HTML the page shows verbatim (each
    trade's best/worst table row, each view's category table), which
    compresses many times over. Measured 2026-09-26 on a synthetic 36-band
    run whose bands each traded a different 200-trade list over ~2,460 days
    (16 series in 8 categories, a 300-entry k-hat population per band): 11.7
    MB as compact JSON and 2.19 MB packed (the base64 block), a 3.02 MB page;
    with a tier-floors-off family whose 18 binding bands each traded yet
    another such list (54 lists in all, each binding band with its own
    300-entry population) — the worst case now — 17.6 MB, 3.30 MB packed and
    a 4.13 MB page. On the DR-73 calibration corpus's own band sweep (start
    2020-01-01, ladders on) the family took the block from 0.45 MB to 0.67
    MB and the page from 4.07 MB to 4.29 MB. The script inflates it with the
    browser's own DecompressionStream, so nothing is added to the page but
    the bytes.

    Non-finite floats become null first (_json_safe) and allow_nan=False
    makes a missed one raise instead of shipping unparseable JSON. The block
    is base64, whose alphabet has no "<", so no text inside it — Kalshi's or
    anyone's — can close the <script> element early; and it is typed
    text/plain, so the browser never runs it. gzip's mtime is pinned to 0, so
    the same payload always encodes to the same bytes.

    Args:
        element_id (str): The block's id.
        payload (dict): JSON-serialisable data.

    Returns:
        str: The <script type="text/plain" data-encoding="gzip+base64"> element.
    """
    raw = json.dumps(_json_safe(payload), allow_nan=False, separators=(",", ":"))
    packed = base64.b64encode(gzip.compress(raw.encode("utf-8"), mtime=0)).decode("ascii")
    return (f'<script type="text/plain" id="{element_id}" data-encoding="gzip+base64">'
            f"{packed}</script>")


# The page-wide filter's script. A raw string, so every backslash in it (a JS
# escape or a regular expression) reaches the browser as written. It reads ONE
# data block (id="dash-data", built by _filter_payload and packed by
# _packed_json_script) and draws nothing of its own: every figure it shows
# was computed in Python, every sentence about the data is a Python template
# it fills (D.text) — its only words of its own are the line it shows when
# that data cannot be loaded — every trace it draws copies the styling of a
# trace Python drew
# (traceOf; the trade-type lines, which a band can hold where the primary
# holds none, from the styles Python drew them with, D.styles), and it writes
# only through textContent, the options API and Python-escaped HTML
# fragments. On load it inflates the block, sets the bar back to the view
# Python rendered and enables it (the Tier floors select only when the data
# carries a tier-off view, D.bands_off) — it redraws nothing until a <select>
# changes.
_FILTER_JS = r"""
<script>
(function() {
  var dataEl = document.getElementById('dash-data');
  var bandSel = document.getElementById('flt-band');
  var catSel = document.getElementById('flt-cat');
  var tagSel = document.getElementById('flt-tag');
  if (!dataEl || !bandSel || !catSel || !tagSel) { return; }
  var SELECTS = [bandSel, catSel, tagSel];
  // "Tier floors": each band's run as simulated (on), or its run with the
  // deadline-gap tier floors not applied (off, D.bands_off)
  var tierSel = document.getElementById('flt-tier');
  if (tierSel) { SELECTS.push(tierSel); }
  // The k-hat chart's own "Group by" select follows the bar's rules
  var khatGroup = document.getElementById('khat-group');
  if (khatGroup) { SELECTS.push(khatGroup); }
  var D = null, N = 0;

  function byId(id) { return document.getElementById(id); }
  function setText(id, text) { var el = byId(id); if (el) { el.textContent = text; } }

  // A sparse series ([[axis index, value], ...], one point wherever the value
  // changes, always from index 0) back onto every date of the shared axis.
  function expand(sparse) {
    var out = new Array(N), j = 0, cur = null;
    for (var i = 0; i < N; i++) {
      while (j < sparse.length && sparse[j][0] <= i) { cur = sparse[j][1]; j++; }
      out[i] = cur;
    }
    return out;
  }
  function pick(arr, idx) { return idx.map(function(i) { return arr[i]; }); }

  function bandIndex() { return parseInt(bandSel.value, 10); }
  // The bands as the Tier floors choice reads them: Python's tier-off runs
  // only when the data carries them, else the runs as simulated
  function tiersOff() { return !!(tierSel && D.bands_off && tierSel.value === 'off'); }
  function bands() { return tiersOff() ? D.bands_off : D.bands; }
  function khats() { return tiersOff() ? D.khat_off : D.khat; }
  function list() { return D.lists[bands()[bandIndex()].list]; }
  function viewKey() {
    if (tagSel.value !== '') { return 's' + tagSel.value; }
    if (catSel.value !== '') { return 'c' + catSel.value; }
    return 'all';
  }
  function count(key) { var v = list().views[key]; return v ? v.n : 0; }
  function currentView() { return list().views[viewKey()] || D.empty; }
  function subName(i, withCategory) {
    var sc = D.subcats[i];
    return (withCategory ? D.categories[sc[0]] + ' · ' : '') + sc[1];
  }
  function selectionName(key) {
    if (key.charAt(0) === 'c') { return D.categories[parseInt(key.slice(1), 10)]; }
    return subName(parseInt(key.slice(1), 10), true);
  }

  // Python's _trade_count, and a template's {name} fields filled as
  // str.format fills them there
  function trades(n) { return n + (n === 1 ? ' trade' : ' trades'); }
  function fill(template, values) {
    return template.replace(/\{(\w+)\}/g, function(field, name) {
      return name in values ? String(values[name]) : field;
    });
  }
  // The summary line: the templates _filter_summary_text fills for the view
  // Python rendered, filled here for every other one
  function summary(v) {
    var key = viewKey(), T = D.text, band = bands()[bandIndex()], text;
    if (key === 'all') {
      text = fill(T.all, {scenario: band.scenario, count: trades(v.n)});
      // Python names each view's closing note (none for the primary run itself)
      if (band.note) { text += T[band.note]; }
    } else {
      text = fill(T.slice, {scenario: band.scenario, selection: selectionName(key),
                            n: v.n, band_count: trades(count('all'))});
    }
    setText('flt-summary', text + T.unfiltered);
  }

  // Trace i of a figure Python drew, with new data: its styling (colours,
  // widths, fills, bins, hover) stays exactly what Python drew, and so do its
  // visibility and point selection — a legend click or a box/lasso selection
  // is not carried into another view (its point indexes would name other
  // points there), just as the trade-type lines drawn fresh beside it start
  // visible and unselected.
  function traceOf(id, i, data) {
    var gd = byId(id);
    var base = (gd && gd.data && gd.data[i]) ? gd.data[i] : {};
    var t = Object.assign({}, base, data);
    delete t.uid;
    delete t.visible;
    delete t.selectedpoints;
    return t;
  }
  function markerOf(id, i, extra) {
    var gd = byId(id);
    var base = (gd && gd.data && gd.data[i] && gd.data[i].marker) ? gd.data[i].marker : {};
    return Object.assign({}, base, extra);
  }
  // Every chart the script redraws, with its layout exactly as Python drew
  // it, captured now — as the page finishes loading, before any redraw, so
  // normally before the reader has zoomed or panned (a zoom made while the
  // page is still loading would be kept). A redraw starts from that copy,
  // never from the live layout, where a zoom leaves a fixed axis range that
  // would clip the next selection's data.
  var CHARTS = ['perf-cum', 'perf-dd', 'dec-monthly', 'dec-cat', 'dec-sub', 'dec-price',
                'dec-hold', 'cal-curve', 'diag-ret', 'diag-slip', 'risk-kelly', 'risk-dep',
                'bench-fig', 'khat-fig'];
  var drawn = {};
  CHARTS.forEach(function(id) {
    var gd = byId(id);
    if (gd && gd.layout) { drawn[id] = JSON.stringify(gd.layout); }
  });
  function redraw(id, traces, layoutPatch, title) {
    var gd = byId(id);
    if (!gd || !drawn[id] || !window.Plotly) { return; }
    var layout = Object.assign(JSON.parse(drawn[id]), layoutPatch || {});
    if (title !== undefined) {
      var base = (layout.title && typeof layout.title === 'object') ? layout.title : {};
      layout.title = Object.assign({}, base, {text: title});
    }
    Plotly.react(gd, traces, layout);
    // A body that was hidden has laid its charts out at no width at all
    Plotly.Plots.resize(gd);
  }
  // A chart whose height depends on its rows. A resize re-reads the chart's
  // box, and plotly.py writes the height on one of two boxes by version: on
  // the one around the chart (the chart itself 100% of it, 6.9) or on the
  // chart's own (older) — so both get it.
  function sizeTo(id, height) {
    var gd = byId(id);
    if (!gd) { return; }
    gd.style.height = height + 'px';
    if (gd.parentElement) { gd.parentElement.style.height = height + 'px'; }
  }
  function bars(id, spec, layoutPatch) {
    redraw(id, [traceOf(id, 0, {x: spec.x, y: spec.y,
                                marker: markerOf(id, 0, {color: spec.c})})], layoutPatch);
  }
  function show(prefix, has) {
    var empty = byId(prefix + '-empty'), body = byId(prefix + '-body');
    if (empty) { empty.style.display = has ? 'none' : ''; }
    if (body) { body.style.display = has ? '' : 'none'; }
  }
  function rows(ids) { return ids.map(function(i) { return D.strings[i]; }).join(''); }

  function renderPerformance(v) {
    Object.keys(v.kpi).forEach(function(k) { setText('kpi-' + k, v.kpi[k]); });
    var traces = [traceOf('perf-cum', 0, {x: D.dates, y: expand(v.total)})];
    v.types.forEach(function(line) {
      var st = D.styles.types[line[0]] || {};
      traces.push({type: 'scatter', x: D.dates, y: expand(line[1]), name: line[0],
                   line: {color: st.color, width: st.width, dash: st.dash}});
    });
    redraw('perf-cum', traces);
    redraw('perf-dd', [traceOf('perf-dd', 0, {x: D.dates, y: expand(v.dd)})]);
  }
  function renderDecomposition(v, L) {
    bars('dec-monthly', v.monthly);
    bars('dec-cat', v.cat);
    sizeTo('dec-sub', v.sub.h);
    bars('dec-sub', v.sub, {height: v.sub.h});
    var table = byId('dec-table');
    if (table) { table.innerHTML = D.strings[v.table]; }
    bars('dec-price', v.price);
    redraw('dec-hold', [traceOf('dec-hold', 0, {x: pick(L.hold, v.idx)})]);
  }
  function renderCalibration(v) {
    setText('kpi-brier', v.cal.brier);
    setText('kpi-log_loss', v.cal.log_loss);
    redraw('cal-curve', [
      traceOf('cal-curve', 0, {}),
      traceOf('cal-curve', 1, {x: v.cal.x, y: v.cal.y, text: v.cal.text,
                               marker: markerOf('cal-curve', 1, {size: v.cal.size})})],
      null, v.cal.title);
  }
  function renderDiagnostics(v, L) {
    redraw('diag-ret', [traceOf('diag-ret', 0, {x: pick(L.ret, v.idx)})]);
    redraw('diag-slip', [traceOf('diag-slip', 0, {x: pick(L.slip, v.idx)})]);
    var best = byId('diag-best'), worst = byId('diag-worst');
    if (best) { best.innerHTML = rows(v.best); }
    if (worst) { worst.innerHTML = rows(v.worst); }
  }
  function renderRisk(v, L) {
    redraw('risk-kelly', [
      traceOf('risk-kelly', 0, {x: pick(L.kx, v.idx), y: pick(L.ky, v.idx),
                                text: pick(L.kt, v.idx)}),
      traceOf('risk-kelly', 1, {x: [0, v.k11], y: [0, v.k11]})]);
    redraw('risk-dep', [traceOf('risk-dep', 0, {x: D.dates, y: expand(v.dep)})]);
  }
  function renderBenchmark(v) {
    setText('bench-return', v.bench['return']);
    setText('bench-sharpe', v.bench.sharpe);
    setText('bench-max_dd', v.bench.max_dd);
    // The strategy trace is redrawn; the S&P trace after it keeps its data
    var gd = byId('bench-fig'), traces = [traceOf('bench-fig', 0, {x: D.dates, y: expand(v.eq)})];
    for (var i = 1; gd && gd.data && i < gd.data.length; i++) { traces.push(traceOf('bench-fig', i, {})); }
    redraw('bench-fig', traces);
  }

  // The k-hat chart's rows for a grouping: [{label, st, kind}], kind "all"
  // (the grouping's whole population), "selected" (the filter's current
  // choice) or "bar". By category or tag: every group at the selected band
  // (tags within the selected category); by band: every band for the
  // selected category or tag — every band as the Tier floors choice reads it.
  // _section_khat renders the same rows for the default (by category,
  // primary band, tier floors on, no filter), from the same payload.
  function khatRows(group) {
    var bi = bandIndex(), cat = catSel.value, tag = tagSel.value, T = D.text, out = [];
    var B = bands(), K = khats();
    if (group === 'band') {
      var key = viewKey();
      K.forEach(function(b, i) {
        out.push({label: B[i].label, st: (b && b.groups[key]) || null,
                  kind: i === bi ? 'selected' : 'bar'});
      });
      return out;
    }
    var band = K[bi];
    if (!band) { return out; }
    if (group === 'tag') {
      out.push({label: cat === '' ? T.khat_all_tags
                                  : fill(T.khat_all_in, {category: D.categories[parseInt(cat, 10)]}),
                st: band.groups[cat === '' ? 'all' : 'c' + cat] || null, kind: 'all'});
      D.subcats.forEach(function(sc, si) {
        if (cat !== '' && String(sc[0]) !== cat) { return; }
        var st = band.groups['s' + si];
        if (st) {
          out.push({label: subName(si, cat === ''), st: st,
                    kind: String(si) === tag ? 'selected' : 'bar'});
        }
      });
      return out;
    }
    out.push({label: T.khat_all_categories, st: band.groups.all || null, kind: 'all'});
    D.categories.forEach(function(name, ci) {
      var st = band.groups['c' + ci];
      if (st) { out.push({label: name, st: st, kind: String(ci) === cat ? 'selected' : 'bar'}); }
    });
    return out;
  }
  // The title _section_khat renders for the default, from the same template.
  // Grouped by band while the tiers are off, Python's khat_scope_tier_off
  // names that setting (grouped otherwise, the scope is the band's own
  // "where", which already names it)
  function khatTitle(group) {
    var T = D.text, scope;
    if (group === 'band') {
      var key = viewKey();
      scope = key === 'all' ? T.khat_every_category : selectionName(key);
      if (tiersOff()) { scope = fill(T.khat_scope_tier_off, {scope: scope}); }
    } else {
      scope = bands()[bandIndex()].where;
    }
    return fill(T.khat_title, {group: T.khat_group_words[group], scope: scope});
  }
  // One table row: the group's name and the cells Python formatted
  function khatRow(label, cells) {
    var tr = document.createElement('tr');
    tr.style.borderBottom = '1px solid #E0E0E0';
    [label].concat(cells).forEach(function(text) {
      var td = document.createElement('td');
      td.style.padding = '4px 12px';
      td.textContent = text;
      tr.appendChild(td);
    });
    return tr;
  }
  function renderKhat() {
    var groupSel = document.getElementById('khat-group');
    if (!groupSel || !byId('khat-fig')) { return; }
    var group = groupSel.value, list_ = khatRows(group);
    var has = list_.some(function(r) { return r.st && r.st.n > 0; });
    // "Not recorded" (no calibration behind the rows) is not "none measured"
    var K = khats();
    var recorded = group === 'band' ? K.some(function(b) { return b !== null; })
                                    : K[bandIndex()] !== null;
    setText('khat-empty', recorded ? D.text.khat_none : D.text.khat_not_recorded);
    show('khat', has);
    if (!has) { return; }
    var colors = D.styles.khat, h = D.styles.khat_height;
    // _khat_chart_height, from the same constants
    var height = Math.max(h[0], h[1] * list_.length + h[2]);
    sizeTo('khat-fig', height);
    redraw('khat-fig', [traceOf('khat-fig', 0, {
      y: list_.map(function(r) { return r.label; }),
      x: list_.map(function(r) { return r.st ? r.st.k : null; }),
      text: list_.map(function(r) { return r.st ? r.st.text : ''; }),
      customdata: list_.map(function(r) { return r.st ? r.st.cells : D.khat_blank; }),
      marker: markerOf('khat-fig', 0, {color: list_.map(function(r) { return colors[r.kind]; })})
    })], {height: height}, khatTitle(group));
    var body = byId('khat-rows');
    if (body) {
      body.textContent = '';
      list_.forEach(function(r) {
        body.appendChild(khatRow(r.label, r.st ? r.st.cells : D.khat_blank));
      });
    }
  }

  // Option labels carry the selected band's trade counts; the tag list holds
  // the selected category's tags, or every "Category · Tag" under "All".
  function refreshOptions() {
    for (var i = 1; i < catSel.options.length; i++) {
      var ci = catSel.options[i].value;
      catSel.options[i].text = D.categories[parseInt(ci, 10)] + ' (' + count('c' + ci) + ')';
    }
    var cat = catSel.value, keep = tagSel.value;
    while (tagSel.options.length > 1) { tagSel.remove(1); }
    D.subcats.forEach(function(sc, i) {
      if (cat !== '' && String(sc[0]) !== cat) { return; }
      tagSel.add(new Option(subName(i, cat === '') + ' (' + count('s' + i) + ')', String(i)));
    });
    tagSel.value = keep;
    if (tagSel.value !== keep) { tagSel.value = ''; }
  }
  // Each band option named as the Tier floors choice reads it (Python's
  // "option" text: "max(tier,0.2)-0.6" with the tiers on, "0.2-0.6" off)
  function relabelBands() {
    var B = bands();
    for (var i = 0; i < bandSel.options.length; i++) {
      bandSel.options[i].text = B[parseInt(bandSel.options[i].value, 10)].option;
    }
  }

  function render() {
    var v = currentView(), L = list(), has = v.n > 0;
    summary(v);
    setText('hdr-trades', String(v.n));
    // Shown before drawing, so every chart is laid out at its real width
    ['dec', 'cal', 'diag', 'risk'].forEach(function(p) { show(p, has); });
    renderPerformance(v);
    if (has) {
      renderDecomposition(v, L);
      renderCalibration(v);
      renderDiagnostics(v, L);
      renderRisk(v, L);
    }
    renderBenchmark(v);
    renderKhat();
  }

  // The block is gzip-compressed JSON in base64 (_packed_json_script),
  // inflated once by the browser's own DecompressionStream.
  function inflate() {
    var bin = atob(dataEl.textContent.trim());
    var bytes = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) { bytes[i] = bin.charCodeAt(i); }
    var stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    return new Response(stream).text().then(function(text) { return JSON.parse(text); });
  }
  // The script's one sentence of its own: the data it would fill Python's
  // templates from could not be loaded
  function unavailable(reason) {
    SELECTS.forEach(function(s) { s.disabled = true; });
    setText('flt-summary', 'The filter could not load its data (' + reason
      + '); every section shows the primary spread band’s full run.');
  }
  // A browser can restore a <select>'s last choice on a reload, or on going
  // back; the page as rendered is the primary band's unfiltered view with the
  // tier floors on, so the bar is set back to it. Python renders the selects
  // disabled: they are enabled once the data is inflated, so no choice can be
  // made (or lost) before it can be drawn.
  SELECTS.forEach(function(s) {
    s.selectedIndex = 0;
    for (var i = 0; i < s.options.length; i++) {
      if (s.options[i].defaultSelected) { s.selectedIndex = i; }
    }
  });
  if (!window.DecompressionStream || !window.Response || !window.Blob) {
    unavailable('this browser cannot decompress it');
    return;
  }
  // Started from a resolved promise, so an error thrown while the inflate is
  // starting (atob on a damaged block) lands in the same handler as a later one
  Promise.resolve().then(inflate).then(function(data) {
    D = data;
    N = D.dates.length;
    // A run with no tier-off view keeps the Tier floors select disabled
    SELECTS.forEach(function(s) { s.disabled = s === tierSel && !D.bands_off; });
  }, function(err) { unavailable(String(err)); });

  bandSel.addEventListener('change', function() {
    if (!D) { return; }
    refreshOptions();
    render();
  });
  if (tierSel) {
    tierSel.addEventListener('change', function() {
      // Nothing to switch to without a tier-off view (the select stays shut)
      if (!D || !D.bands_off) { return; }
      relabelBands();
      refreshOptions();
      render();
    });
  }
  catSel.addEventListener('change', function() {
    if (!D) { return; }
    tagSel.value = '';
    refreshOptions();
    render();
  });
  if (khatGroup) {
    khatGroup.addEventListener('change', function() {
      if (!D) { return; }
      renderKhat();
    });
  }
  tagSel.addEventListener('change', function() {
    if (!D) { return; }
    var t = tagSel.value;
    if (t !== '' && catSel.value === '') {
      // A "Category · Tag" picked under "All" selects its category too
      catSel.value = String(D.subcats[parseInt(t, 10)][0]);
      refreshOptions();
      tagSel.value = t;
    }
    render();
  });
})();
</script>
"""


# ─── Main entry ──────────────────────────────────────────────────────────────

def generate_dashboard(
    trades: list[BacktestTrade],
    equity_df: pd.DataFrame,
    start_date: date,
    initial_balance: float,
    *,
    sweep: BacktestSweep | None = None,
    interval_discount: float | None = None,
    series_categories: dict[str, tuple[str, tuple[str, ...]]] | None = None,
) -> Path:
    """
    Assemble all nine dashboard sections into a single self-contained HTML file.

    Calls each _section_*() builder in order, concatenates the resulting HTML
    fragments into a full page with an embedded Plotly CDN script tag — plus
    the page-wide filter: the sticky bar under the header lines
    (_filter_bar_html), the packed data block of every band x tier floors x
    category x tag view after the sections (_filter_payload,
    _packed_json_script), and the
    script that swaps a selection in (_FILTER_JS). If that data cannot be
    built, the page is written without the bar and its script, with a notice
    in the bar's place (and in the k-hat breakdown's, which reads the same
    data) and a WARNING in the log — then
    writes the file to PROJECT_ROOT as backtest_dashboard.html, REPLACING the
    previous run's page (operator decision, 2026-09-25: one current dashboard
    rather than a timestamped one per run, which TS-18 had made collision-free).
    The page is written to a temporary file beside it first and then renamed
    over it (os.replace, atomic on one filesystem), so a browser or a second
    reader never sees a half-written page and a failed write leaves the
    previous dashboard intact. Two runs finishing together each write a
    complete page and the later rename wins.

    The two sweep-related parameters are keyword-only WITH defaults, so the
    existing four-argument positional call still works verbatim. Omit both and
    the interval-discount and scenario-explorer sections each show the same
    kind of short placeholder every other builder emits for empty input, and
    the header's run-settings line reads "not recorded" for both the spread
    band and the ladder setting — with no coverage line and no strike-blind
    notice, since that path has no census to report.

    Args:
        trades (list[BacktestTrade]): Completed backtest trades from
            run_backtest() (or run_backtest_sweep()'s primary point). May be
            empty, in which case all charts show placeholder messages.
        equity_df (pd.DataFrame): Daily equity curve DataFrame with columns
            [date, portfolio_value, daily_return], produced by _build_equity_curve().
        start_date (date): Backtest start date shown in the page title and header.
        initial_balance (float): Starting portfolio value in dollars, used for
            return calculations and benchmark normalization.
        sweep (BacktestSweep | None): The full sweep payload from
            backtester.run_backtest_sweep(), rendered by the
            interval-discount section and the scenario-explorer section, and
            read by the page-wide filter and the k-hat breakdown (every
            band's run and calibration, via _band_runs, and — when it carries
            the tier-floors-off family — every band's tier-off run, via
            _tier_off_runs).
            Passed whole rather than unpacked — it already carries the
            calibration, every swept point, the primary k, the band x k x
            population scenarios and the run's outcome-label census, and
            splitting it would create copies that could disagree. It also
            feeds the header's run-settings line (_run_settings_html). None
            (default) renders both sections' placeholders and the k-hat
            breakdown's "not recorded" notice — and therefore no coverage line
            either, which is honest: that path shows no k̂ card
            to caveat — and the run-settings line says "not recorded" rather
            than guessing.

            When its label_coverage is below
            config.BACKTEST_OUTCOME_LABEL_WARN_FRACTION, a one-line notice is
            also emitted under the Period line, because a strike-blind corpus
            changes which pairs exist and so taints all nine sections, not
            just the one that renders the census (DR-66b).

            Its corpus_provenance is rendered directly under the Period line
            on every run (_corpus_provenance_html): when the corpus was
            assembled — the Period runs to today, the corpus only to that
            moment — whether it came from an earlier run's cache, and the
            archive cutoff at assembly, with a red banner when the window
            starts at or after it (DR-13, M2), or an amber stale-verdict line
            when a simulated point traded anyway. "not recorded" when the
            sweep carries none or there is no sweep.
        interval_discount (float | None): The interval discount `trades` were
            SIZED at, threaded into the Risk section's Kelly scatter and the
            filter's views of it. Separate from `sweep` because that scatter
            needs it even on a run that produced no sweep. None (default)
            means "no override": the sweep's primary k when a sweep is passed
            (the k its points were simulated at), else
            config.TIME_SERIES_INTERVAL_PROB_DISCOUNT.
        series_categories (dict | None): historical.load_series_categories'
            series ticker -> (category, tags) map, which the Returns
            Decomposition section and the page-wide filter bar file each
            trade under (its category and first tag). None (default) falls
            back to each trade's ticker-prefix category.

    Returns:
        Path: Absolute path to the HTML file written,
            PROJECT_ROOT / "backtest_dashboard.html".
    """
    out_path = PROJECT_ROOT / DASHBOARD_FILENAME
    # The window's last day for the Period line. UTC, like every other
    # "today" the backtest reads (TS-13).
    today = datetime.now(UTC).date()

    # A label-less corpus taints EVERY section, not just the interval-discount
    # one: it changes which pairs were formed, so the trades, the returns and
    # the risk figures on this page all describe a different pair population
    # (DR-66b). The section carries the full caveat and the remedy; this is the
    # one-line pointer at the top so a reader who never scrolls that far still
    # knows. Read off the sweep this function already receives — no new
    # parameter — so the four-positional call renders exactly as before.
    header_note = ""
    if sweep is not None and sweep.label_coverage is not None \
            and sweep.label_coverage.below_floor:
        header_note = (
            '<p style="color:#B71C1C; font-size:14px; font-weight:700;">'
            "Outcome-label coverage for this run is below the floor — the "
            "pairs behind every strategy-derived figure on this page were "
            "grouped strike-blind. See Interval Discount (k) Calibration below."
            "</p>"
        )

    # The ladder setting decides which pairs exist and the primary spread band
    # which of them are ever entered, so both are named in the header above
    # every section, not only inside the scenario explorer; "not recorded"
    # when there is no sweep.
    run_settings = _run_settings_html(sweep)

    # Directly under the Period line, which it qualifies: the corpus holds
    # nothing settled after its assembly even though the period runs to today,
    # and a window at or after the archive cutoff could never enter a trade
    # (unless a simulated point traded, which proves that verdict stale).
    # Rendered on every run, healthy or not (DR-13, M2; DR-66's rule).
    corpus_note = _corpus_provenance_html(sweep)

    # One k for the whole page: the discount these trades were sized at — the
    # override when one was passed, else the sweep's primary k (the k its
    # points were simulated at). The Risk section's Kelly scatter, the
    # filter's views of it and the filter bar's summary all read this one
    # value, so they cannot name two. With neither it is None: the scatter
    # then prices at config.TIME_SERIES_INTERVAL_PROB_DISCOUNT and the summary
    # says "k not recorded".
    k_used = (interval_discount if interval_discount is not None
              else (sweep.primary.k if sweep is not None else None))
    k_text = "k not recorded" if k_used is None else _k_label(k_used)

    # The page-wide filter: every spread band's own run at the primary k (and,
    # when the run simulated one, its run with the tier floors off), and
    # within it every Kalshi category and category · tag, each view computed
    # here by the same helpers the sections below render with. It is an
    # extra: a failure to build it costs the bar and its script, never the
    # page — every section below renders from its own arguments.
    try:
        runs, primary_idx = _band_runs(sweep, trades, equity_df)
        # Each band's run with the tier floors off, or None (no off view:
        # the run carries no complete tier-off family)
        off_runs = _tier_off_runs(sweep, runs)
        filter_data = _filter_payload(runs, primary_idx, start_date, initial_balance,
                                      series_categories, k_used, k_text, off_runs=off_runs)
    except Exception:
        logging.warning("The page-wide filter could not be built; the dashboard is "
                        "written without it", exc_info=True)
        filter_data = None
    if filter_data is None:
        filter_bar, filter_block = _FILTER_UNAVAILABLE_HTML, ""
    else:
        filter_bar = _filter_bar_html(filter_data)
        filter_block = _packed_json_script("dash-data", filter_data) + _FILTER_JS

    sections = [
        _section_performance(equity_df, trades, start_date, initial_balance),
        _section_decomposition(trades, series_categories),
        _section_calibration(trades),
        # Takes the sweep whole (calibration + every point + the primary k)
        _section_interval_discount(sweep),
        # The same k-hat, broken down by category, tag and spread band — read
        # off the filter payload, so it follows the filter bar like the
        # trade sections do (None when the payload could not be built)
        _section_khat(filter_data, k_used),
        # Also takes the sweep whole — it reads .scenarios, .same_title_point
        # and .calibrations_by_band, none of which _section_interval_discount
        # renders, and passing pieces could let the two sections (and the
        # header's run-settings line) drift onto different bands or settings.
        _section_scenario_explorer(sweep),
        _section_diagnostics(trades),
        # k must be the discount these trades were sized at, or the Kelly
        # scatter plots the config model against override-sized trades
        _section_risk(trades, equity_df, initial_balance, k=k_used),
        _section_benchmark(equity_df, start_date, initial_balance),
    ]

    # Named page_html (not `html`) so it can't shadow the `html` module used
    # by the _section_* helpers above (html.escape()).
    page_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Kalshi Arbitrage Backtest — {start_date} to today</title>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <style>
    body {{ font-family: sans-serif; max-width: 1200px; margin: 0 auto; padding: 24px; }}
    h1 {{ color: #1A237E; }}
  </style>
</head>
<body>
<h1>Kalshi Arbitrage Backtest</h1>
<p style="color:#616161; font-size:14px;">
  Period: {start_date} → {today} &nbsp;|&nbsp;
  Starting balance: ${initial_balance:,.2f} &nbsp;|&nbsp;
  Trades found: <span id="hdr-trades">{len(trades)}</span>
</p>
{corpus_note}
{run_settings}
{header_note}
{filter_bar}
{''.join(sections)}
{filter_block}
</body>
</html>"""

    # Written beside the target, then renamed over it: the rename is atomic,
    # so the previous dashboard is replaced only by a complete page. The
    # temporary name carries the pid so two concurrent runs never share one.
    tmp_path = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_bytes(page_html.encode("utf-8"))
        os.replace(tmp_path, out_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    logging.info("Dashboard written: %s", out_path)
    return out_path
