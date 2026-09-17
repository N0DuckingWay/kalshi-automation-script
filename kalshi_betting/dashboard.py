"""
File: dashboard.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Generates a self-contained interactive HTML performance dashboard from the
    results of a backtest run. Assembles seven sections — portfolio performance
    (equity curve, Sharpe, drawdown), returns decomposition (by month, category,
    entry price), calibration analysis (Brier score, reliability diagram),
    interval-discount (k) calibration (empirical k-hat plus a native Plotly
    dropdown that switches the equity curve between the swept k values), trade-
    level diagnostics (distribution, slippage, best/worst trades), risk metrics
    (Kelly sizing scatter, capital deployment), and benchmark comparison (S&P 500
    via yfinance) — into a single HTML file with embedded Plotly charts. The file
    is written to PROJECT_ROOT and can be opened directly in any browser.

Dependencies:
    Imports BacktestSweep and BacktestTrade from backtester.py, and PROJECT_ROOT,
    SAME_TITLE_CO_RESOLVE_PROB, CALENDAR_DAYS_PER_YEAR, TRADING_DAYS_PER_YEAR,
    create_new_output(), fee_per_pair_approx() and
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
    chart renders into. Its scope is deliberately that one section: the other
    six always reflect the run's primary k (the CLI's --interval-discount, or
    config.TIME_SERIES_INTERVAL_PROB_DISCOUNT when it was not passed).
"""
import html
import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from plotly.subplots import make_subplots

from .backtester import BacktestSweep, BacktestTrade, OutcomeLabelCoverage
from .config import (
    BACKTEST_OUTCOME_LABEL_WARN_FRACTION,
    CALENDAR_DAYS_PER_YEAR,
    PROJECT_ROOT,
    SAME_TITLE_CO_RESOLVE_PROB,
    TRADING_DAYS_PER_YEAR,
    create_new_output,
    fee_per_pair_approx,
    time_series_profit_prob,
)

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
            if equity is empty or entirely NaN, or if the drawdown series itself
            is entirely NaN (e.g. an all-zero equity curve, where every point
            divides 0 by a running peak of 0) — there is no trough to report.
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
    when = dd.idxmin()
    return max_dd, when


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
    for a cumulative-deadline pair. Returns 0.0 when there is no edge.

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
  <div style="font-size:26px; font-weight:700; color:{color};">{value}</div>
</div>
"""


def _kpi(label: str, value: str, color: str = "#212121") -> str:
    """
    Render a single KPI card as an HTML snippet using the _KPI_TEMPLATE.

    Args:
        label (str): Short label displayed above the value (e.g. "Total Return").
        value (str): Pre-formatted value string to display (e.g. "+12.3%").
        color (str): CSS hex color for the value text. Defaults to near-black "#212121".

    Returns:
        str: Rendered HTML string for one KPI card block.
    """
    return _KPI_TEMPLATE.format(label=label, value=value, color=color)


def _fig_html(fig: go.Figure, height: int = 400) -> str:
    """
    Apply a standard layout to a Plotly figure and return it as an inline HTML string.

    Configures common layout properties (height, margins, background colors, font,
    legend position) then serializes to HTML without the full Plotly.js bundle
    (assumes the CDN script tag is already present in the page <head>).

    Args:
        fig (go.Figure): Plotly figure to render.
        height (int): Desired figure height in pixels. Defaults to 400.

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
    return fig.to_html(full_html=False, include_plotlyjs=False)


# ─── Section 1: Portfolio Performance ────────────────────────────────────────

def _section_performance(
    equity_df: pd.DataFrame,
    trades: list[BacktestTrade],
    start_date: date,
    initial_balance: float,
) -> str:
    """
    Build the "Portfolio Performance" HTML section.

    Computes summary KPIs (total return, Sharpe, Sortino, max drawdown, win rate)
    and renders two charts: the equity curve and a drawdown percentage plot.

    Args:
        equity_df (pd.DataFrame): Daily equity curve with columns [date, portfolio_value,
            daily_return] as produced by _build_equity_curve().
        trades (list[BacktestTrade]): Completed backtest trades for win rate and avg return.
        start_date (date): Backtest start date for display context.
        initial_balance (float): Starting portfolio value in dollars.

    Returns:
        str: Self-contained HTML section string including KPI cards and two Plotly charts.
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

    dd_str = f"({dd_when})" if dd_when else ""

    kpis = "".join([
        _kpi("Total Return",  f"{total_return:+.1%}", "#2196F3"),
        _kpi("Sharpe Ratio",  f"{sharpe:.2f}"),
        _kpi("Sortino Ratio", f"{sortino:.2f}"),
        _kpi("Max Drawdown",  f"{max_dd:.1%} {dd_str}", "#F44336"),
        _kpi("Win Rate",      f"{win_rate:.1%}", "#4CAF50"),
        _kpi("Avg Return/Trade", f"{avg_ret:.1%}"),
        _kpi("Total Trades",  str(len(trades))),
    ])

    # Equity curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_df["date"], y=equity_df["portfolio_value"],
        name="Strategy", line={"color": _COLORS["strategy"], "width": 2},
    ))
    fig.update_layout(title="Equity Curve", yaxis_title="Portfolio Value ($)",
                      xaxis_title="Date")

    # Drawdown chart
    rolling_max = equity_df["portfolio_value"].cummax()
    dd_series   = (equity_df["portfolio_value"] - rolling_max) / rolling_max
    fig2 = go.Figure(go.Scatter(
        x=equity_df["date"], y=dd_series * 100,
        fill="tozeroy", name="Drawdown %",
        line={"color": _COLORS["dd"]}, fillcolor="rgba(244,67,54,0.2)",
    ))
    fig2.update_layout(title="Drawdown (%)", yaxis_title="Drawdown (%)", xaxis_title="Date")

    return (
        _SECTION_STYLE.format(title="Portfolio Performance")
        + kpis
        + _fig_html(fig)
        + _fig_html(fig2, height=280)
    )


# ─── Section 2: Returns Decomposition ────────────────────────────────────────

def _section_decomposition(trades: list[BacktestTrade]) -> str:
    """
    Build the "Returns Decomposition" HTML section.

    Shows four charts: monthly P&L bar chart, P&L by market category, P&L by entry
    price bucket, and a holding-period histogram.

    Args:
        trades (list[BacktestTrade]): Completed backtest trades to decompose.

    Returns:
        str: Self-contained HTML section string. Returns a "No trades" placeholder
            if the trades list is empty.
    """
    if not trades:
        return _SECTION_STYLE.format(title="Returns Decomposition") + "<p>No trades.</p>"

    df = pd.DataFrame([{
        "entry_date":    t.entry_date,
        "exit_date":     t.exit_date,
        "profit_ratio":  t.profit_ratio,
        "profit":        t.profit,
        "category":      t.category,
        "holding_days":  t.holding_days,
        "entry_pA":      t.entry_pA,
        "n":             t.n,
        "pair_type":     t.pair_type,
    } for t in trades])

    df["month"] = pd.to_datetime(df["entry_date"]).dt.to_period("M").astype(str)

    # Monthly returns bar chart
    monthly = df.groupby("month")["profit"].sum().reset_index()
    fig_monthly = go.Figure(go.Bar(
        x=monthly["month"], y=monthly["profit"],
        marker_color=[_COLORS["profit"] if v >= 0 else _COLORS["loss"] for v in monthly["profit"]],
    ))
    fig_monthly.update_layout(title="Monthly P&L ($)", xaxis_title="Month", yaxis_title="P&L ($)")

    # Category breakdown
    cat = df.groupby("category")["profit"].sum().sort_values()
    fig_cat = go.Figure(go.Bar(
        y=cat.index, x=cat.values, orientation="h",
        marker_color=[_COLORS["profit"] if v >= 0 else _COLORS["loss"] for v in cat.values],
    ))
    fig_cat.update_layout(title="P&L by Category ($)", xaxis_title="P&L ($)")

    # Entry price bucket. entry_pA is market A's YES ask at entry for both pair
    # types, but its meaning differs: for a time_series row it is the price
    # actually PAID for the YES leg on the earlier contract, while for a
    # same_title row it is the pricier side's quote (the NO leg costs nA).
    bins   = [0, 0.20, 0.40, 0.60, 0.80, 1.01]
    labels = ["<20¢", "20–40¢", "40–60¢", "60–80¢", ">80¢"]
    df["price_bucket"] = pd.cut(df["entry_pA"], bins=bins, labels=labels)
    price_grp = df.groupby("price_bucket", observed=True)["profit"].sum()
    fig_price = go.Figure(go.Bar(
        x=price_grp.index.astype(str), y=price_grp.values,
        marker_color=[_COLORS["profit"] if v >= 0 else _COLORS["loss"] for v in price_grp.values],
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

    return (
        _SECTION_STYLE.format(title="Returns Decomposition")
        + _fig_html(fig_monthly)
        + _fig_html(fig_cat, height=350)
        + _fig_html(fig_price)
        + _fig_html(fig_dur)
    )


# ─── Section 3: Calibration Analysis ─────────────────────────────────────────

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
            Returns a "No trades" placeholder if the list is empty.
    """
    if not trades:
        return _SECTION_STYLE.format(title="Calibration Analysis") + "<p>No trades.</p>"

    # Collect (predicted_prob, actual_outcome) pairs
    probs, actuals = [], []
    for t in trades:
        probs.append(t.entry_pA)
        actuals.append(1 if t.outcome_a == "yes" else 0)
        probs.append(t.entry_pB)
        actuals.append(1 if t.outcome_b == "yes" else 0)

    brier = _brier_score(trades)
    ll    = _log_loss(trades)

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

    fig_cal = go.Figure()
    fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Perfect calibration",
                                 line={"dash": "dash", "color": "#9E9E9E"}))
    fig_cal.add_trace(go.Scatter(
        x=mean_pred, y=mean_act, mode="lines+markers",
        name="Actual", line={"color": _COLORS["strategy"]},
        marker={"size": [max(6, c // 2) for c in counts]},
        text=[f"n={c}" for c in counts], hoverinfo="text+x+y",
    ))
    fig_cal.update_layout(
        title=f"Calibration Curve (Brier={brier:.4f}, LogLoss={ll:.4f})",
        xaxis_title="Predicted probability",
        yaxis_title="Actual resolution rate",
        xaxis={"range": [0, 1]}, yaxis={"range": [0, 1]},
    )

    kpis = "".join([
        _kpi("Brier Score", f"{brier:.4f}", "#2196F3"),
        _kpi("Log Loss",    f"{ll:.4f}",    "#2196F3"),
    ])

    return (
        _SECTION_STYLE.format(title="Calibration Analysis")
        + kpis
        + _fig_html(fig_cal)
    )


# ─── Section 4: Interval Discount (k) Calibration ────────────────────────────

def _label_coverage_html(coverage: OutcomeLabelCoverage | None) -> str:
    """
    Render the outcome-label census as a line, or as a banner when it is low.

    The figure this prints is the one backtester._log_outcome_label_coverage()
    already logged — the SAME measurement from the same single pass over the
    same corpus, carried through BacktestSweep.label_coverage — so the page and
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
        "<code>--no-cache</code> (equivalently, also delete the assembled "
        "<code>backtest_cache/settled_markets_*.json</code>); "
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

    Scope limit: the dropdown drives THIS section only. Every other section
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
    coverage_html = _label_coverage_html(coverage)
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


# ─── Section 5: Trade-Level Diagnostics ──────────────────────────────────────

def _section_diagnostics(trades: list[BacktestTrade]) -> str:
    """
    Build the "Trade-Level Diagnostics" HTML section.

    Renders a per-trade return distribution histogram, a slippage histogram, and
    HTML tables listing the top 5 and worst 5 trades by dollar profit.

    Args:
        trades (list[BacktestTrade]): Completed backtest trades to diagnose.

    Returns:
        str: Self-contained HTML section string. Returns a "No trades" placeholder
            if the list is empty.
    """
    if not trades:
        return _SECTION_STYLE.format(title="Trade-Level Diagnostics") + "<p>No trades.</p>"

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
    sorted_trades = sorted(trades, key=lambda t: t.profit, reverse=True)
    best  = sorted_trades[:5]
    worst = sorted_trades[-5:]

    def _trow(t: BacktestTrade, color: str) -> str:
        """
        Render a single HTML table row for a BacktestTrade.

        Args:
            t (BacktestTrade): Trade to display.
            color (str): CSS background color string for the row (e.g. "#F9FBE7").

        Returns:
            str: An HTML <tr>...</tr> string with entry date, title, pair type,
                contract count, total cost, and profit/return.
        """
        # title_a is Kalshi-controlled (market question text) and rendered
        # into raw HTML below — escape it so a market title can't inject
        # markup or break out of the <td>.
        safe_title = html.escape(t.title_a[:40])
        return (f"<tr style='background:{color}'>"
                f"<td>{t.entry_date}</td>"
                f"<td style='max-width:200px;overflow:hidden;white-space:nowrap;'>{safe_title}</td>"
                f"<td>{t.pair_type}</td>"
                f"<td>{t.n}</td>"
                f"<td>${t.total_cost + t.fees:.2f}</td>"
                f"<td style='color:{'#2E7D32' if t.profit>=0 else '#C62828'}'>"
                f"${t.profit:+.2f} ({t.profit_ratio:.1%})</td>"
                f"</tr>")

    table_html = """
<div style="margin:16px 0; font-family:sans-serif;">
<b>Top 5 Trades</b>
<table style="width:100%;border-collapse:collapse;font-size:13px;margin-top:8px;">
<tr style="background:#E8F5E9;font-weight:bold;">
  <th>Entry</th><th>Title</th><th>Type</th><th>n</th><th>Cost incl. fees</th><th>Profit</th>
</tr>
""" + "".join(_trow(t, "#F9FBE7") for t in best) + """
</table>
<br>
<b>Worst 5 Trades</b>
<table style="width:100%;border-collapse:collapse;font-size:13px;margin-top:8px;">
<tr style="background:#FFEBEE;font-weight:bold;">
  <th>Entry</th><th>Title</th><th>Type</th><th>n</th><th>Cost incl. fees</th><th>Profit</th>
</tr>
""" + "".join(_trow(t, "#FFF8F8") for t in worst) + """
</table>
</div>
"""

    return (
        _SECTION_STYLE.format(title="Trade-Level Diagnostics")
        + _fig_html(fig_hist)
        + _fig_html(fig_slip, height=280)
        + table_html
    )


# ─── Section 6: Risk Metrics ──────────────────────────────────────────────────

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
        str: Self-contained HTML section string. Returns a "No trades" placeholder
            if the trades list is empty.
    """
    if not trades:
        return _SECTION_STYLE.format(title="Risk Metrics") + "<p>No trades.</p>"

    # Kelly vs actual sizing scatter. The actual fraction uses the simulated
    # balance at each trade's entry (the base its Kelly budget was computed
    # from) — dividing by the initial balance would distort as equity drifts.
    # Pass all four entry quotes — _kelly_fraction picks the leg prices per pair
    # type — plus the run's interval discount, so an --interval-discount run
    # plots the Kelly its trades were actually sized at rather than the config one
    kelly_fracs = [
        _kelly_fraction(t.entry_pA, t.entry_nA, t.entry_pB, t.entry_nB, t.pair_type, k=k)
        for t in trades
    ]
    actual_fracs = [
        (t.total_cost + t.fees) / t.balance_at_entry if t.balance_at_entry > 0 else 0.0
        for t in trades
    ]

    fig_kelly = go.Figure(go.Scatter(
        x=kelly_fracs, y=actual_fracs, mode="markers",
        marker={"color": _COLORS["strategy"], "size": 7, "opacity": 0.6},
        # title_a is Kalshi-controlled and Plotly's hover text renders an HTML
        # subset — escape it the same as the table row above (_trow).
        text=[html.escape(t.title_a[:40]) for t in trades],
    ))
    fig_kelly.add_trace(go.Scatter(
        x=[0, max(kelly_fracs + [0.01]) * 1.1],
        y=[0, max(kelly_fracs + [0.01]) * 1.1],
        name="1:1 line", line={"dash": "dash", "color": "#9E9E9E"},
    ))
    fig_kelly.update_layout(
        title="Kelly Fraction vs Actual Fraction of Balance",
        xaxis_title="Kelly fraction", yaxis_title="Actual fraction",
    )

    # Capital deployment over time: net running position size (cost still tied up
    # in unsettled trades). Positive on entry days, back down to zero on exit days.
    entry_by_date: dict[date, float] = {}
    exit_by_date: dict[date, float] = {}
    for t in trades:
        # Fee-inclusive: fees are cash out the door at entry, so they are part
        # of the capital deployed. The Kelly-vs-actual scatter twenty-six lines
        # above already uses (total_cost + fees); this used not to, so the two
        # charts on one dashboard disagreed by the fee rate (TS-12).
        cost = t.total_cost + t.fees
        entry_by_date[t.entry_date] = entry_by_date.get(t.entry_date, 0.0) + cost
        exit_by_date[t.exit_date]   = exit_by_date.get(t.exit_date,   0.0) + cost

    invested_by_date: list[float] = []
    running_invested = 0.0
    for _, row in equity_df.iterrows():
        d = row["date"] if isinstance(row["date"], date) else row["date"].date()
        running_invested += entry_by_date.get(d, 0.0) - exit_by_date.get(d, 0.0)
        invested_by_date.append(max(0.0, running_invested))

    fig_dep = make_subplots()
    fig_dep.add_trace(go.Scatter(
        x=equity_df["date"], y=invested_by_date,
        name="Capital deployed", fill="tozeroy",
        line={"color": _COLORS["invested"]},
        fillcolor="rgba(66,165,245,0.15)",
    ))
    fig_dep.update_layout(title="Capital Deployed Over Time",
                           yaxis_title="Capital in open trades ($)", xaxis_title="Date")

    return (
        _SECTION_STYLE.format(title="Risk Metrics")
        + _fig_html(fig_kelly)
        + _fig_html(fig_dep)
    )


# ─── Section 7: Benchmark Comparison ─────────────────────────────────────────

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

    strat_ret    = float(equity_df["portfolio_value"].iloc[-1] / initial_balance - 1)
    # Calendar-daily by construction (_build_equity_curve emits one row per
    # calendar day), so this takes _sharpe's CALENDAR_DAYS_PER_YEAR default
    # while the ^GSPC row above overrides it to the trading-day base.
    strat_sharpe = _sharpe(equity_df["daily_return"])
    strat_dd     = _max_drawdown(equity_df["portfolio_value"])[0]

    # "Kalshi Arbitrage Strategy" / "Kalshi Arbitrage Backtest" (here, the
    # bold-row match below, and the page <title>/<h1>) are the PRODUCT NAME,
    # kept deliberately after the 2026-09 time-series inversion — the row name
    # here is string-matched by the table renderer below, so both must agree.
    # They are not a claim that the time-series leg is an arbitrage.
    bench_rows.insert(0, {
        "name":   "Kalshi Arbitrage Strategy",
        "return": f"{strat_ret:+.1%}",
        "sharpe": f"{strat_sharpe:.2f}",
        "max_dd": f"{strat_dd:.1%}",
    })

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
    f"<td style='padding:8px 16px;'>{r['return']}</td>"
    f"<td style='padding:8px 16px;'>{r['sharpe']}</td>"
    f"<td style='padding:8px 16px;'>{r['max_dd']}</td>"
    f"</tr>"
    for r in bench_rows
) + "</table>"

    return (
        _SECTION_STYLE.format(title="Benchmark Comparison")
        + bench_table
        + _fig_html(fig, height=450)
    )


# ─── Main entry ──────────────────────────────────────────────────────────────

def generate_dashboard(
    trades: list[BacktestTrade],
    equity_df: pd.DataFrame,
    start_date: date,
    initial_balance: float,
    *,
    sweep: BacktestSweep | None = None,
    interval_discount: float | None = None,
) -> Path:
    """
    Assemble all seven dashboard sections into a single self-contained HTML file.

    Calls each _section_*() builder in order, concatenates the resulting HTML
    fragments into a full page with an embedded Plotly CDN script tag, then
    writes the file to PROJECT_ROOT. The output file is timestamped to
    microsecond precision AND created exclusively, so multiple backtest runs can
    be compared without overwriting previous results even when two runs finish in
    the same second — which was observed happening under the old second-precision
    name (TS-18).

    The two sweep-related parameters are keyword-only WITH defaults, so the
    existing four-argument positional call still works verbatim: omit both and
    the page renders exactly as before, with the interval-discount section
    showing the same kind of short placeholder every other builder emits for
    empty input.

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
            backtester.run_backtest_sweep(), rendered by the interval-discount
            section. Passed whole rather than unpacked — it already carries the
            calibration, every swept point, the primary k and the run's
            outcome-label census, and splitting it would create copies that
            could disagree. None (default) renders that section's placeholder —
            and therefore no coverage line either, which is honest: that path
            shows no k̂ card to caveat.

            When its label_coverage is below
            config.BACKTEST_OUTCOME_LABEL_WARN_FRACTION, a one-line notice is
            also emitted under the Period line, because a strike-blind corpus
            changes which pairs exist and so taints all seven sections, not
            just the one that renders the census (DR-66b).
        interval_discount (float | None): The interval discount `trades` were
            SIZED at, threaded into the Risk section's Kelly scatter. Separate
            from `sweep` because that scatter needs it even on a run that
            produced no sweep. None (default) means "no override" and resolves
            to config.TIME_SERIES_INTERVAL_PROB_DISCOUNT.

    Returns:
        Path: Absolute path to the HTML file actually created
            (PROJECT_ROOT / "backtest_dashboard_YYYY-MM-DD_HHMMSS_ffffff.html",
            with a "-1", "-2", … stem suffix on collision).
    """
    ts = datetime.now(UTC).astimezone().strftime("%Y-%m-%d_%H%M%S_%f")
    out_path = PROJECT_ROOT / f"backtest_dashboard_{ts}.html"

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

    sections = [
        _section_performance(equity_df, trades, start_date, initial_balance),
        _section_decomposition(trades),
        _section_calibration(trades),
        # Takes the sweep whole (calibration + every point + the primary k)
        _section_interval_discount(sweep),
        _section_diagnostics(trades),
        # k must be the discount these trades were sized at, or the Kelly
        # scatter plots the config model against override-sized trades
        _section_risk(trades, equity_df, initial_balance, k=interval_discount),
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
  Period: {start_date} → {datetime.now(UTC).date()} &nbsp;|&nbsp;
  Starting balance: ${initial_balance:,.2f} &nbsp;|&nbsp;
  Trades found: {len(trades)}
</p>
{header_note}
{''.join(sections)}
</body>
</html>"""

    # create_new_output hands back a binary handle, so encode explicitly rather
    # than adding a text-mode parameter to the shared helper (TS-18)
    out_path, fh = create_new_output(out_path)
    with fh:
        fh.write(page_html.encode("utf-8"))
    logging.info("Dashboard written: %s", out_path)
    return out_path
