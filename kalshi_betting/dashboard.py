"""
File: dashboard.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Generates a self-contained interactive HTML performance dashboard from the
    results of a backtest run. Assembles six sections — portfolio performance
    (equity curve, Sharpe, drawdown), returns decomposition (by month, category,
    entry price), calibration analysis (Brier score, reliability diagram), trade-
    level diagnostics (distribution, slippage, best/worst trades), risk metrics
    (Kelly sizing scatter, capital deployment), and benchmark comparison (S&P 500
    via yfinance) — into a single HTML file with embedded Plotly charts. The file
    is written to PROJECT_ROOT and can be opened directly in any browser.

Dependencies:
    Imports BacktestTrade from backtester.py and PROJECT_ROOT from config.py.
    Uses plotly, numpy, pandas, and yfinance (all external). Called by backtest.py
    after run_backtest() completes.

Notes:
    The HTML file loads Plotly.js from the CDN (cdn.plot.ly), so an internet
    connection is required to view the charts. If yfinance fails to fetch S&P 500
    data (e.g. network unavailable), the benchmark section degrades gracefully
    and shows only the strategy equity curve.
"""
import html
import logging
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from plotly.subplots import make_subplots

from .backtester import BacktestTrade
from .config import PROJECT_ROOT, SAME_TITLE_CO_RESOLVE_PROB, fee_per_pair_approx

# ─── Metric computation ───────────────────────────────────────────────────────

def _sharpe(daily_returns: pd.Series, rf: float = 0.0) -> float:
    """
    Compute the annualized Sharpe ratio from a series of daily returns.

    Annualizes by multiplying the mean daily excess return by sqrt(252).

    Args:
        daily_returns (pd.Series): Series of daily fractional returns (e.g. 0.01 for 1%).
        rf (float): Annual risk-free rate as a decimal (e.g. 0.05 for 5%). Defaults to 0.0.

    Returns:
        float: Annualized Sharpe ratio. Returns 0.0 if the standard deviation is zero.
    """
    excess = daily_returns - rf / 252
    std = excess.std()
    return float(excess.mean() / std * np.sqrt(252)) if std > 0 else 0.0


def _sortino(daily_returns: pd.Series, rf: float = 0.0) -> float:
    """
    Compute the annualized Sortino ratio from a series of daily returns.

    Like the Sharpe ratio but uses only downside (negative) deviations in the
    denominator, avoiding penalization for upside volatility.

    Args:
        daily_returns (pd.Series): Series of daily fractional returns.
        rf (float): Annual risk-free rate as a decimal. Defaults to 0.0.

    Returns:
        float: Annualized Sortino ratio. Returns 0.0 if there are no negative excess returns.
    """
    excess = daily_returns - rf / 252
    # Standard downside deviation: RMS of the negative excess returns over ALL
    # periods (positives clipped to 0). Using the sample std of only the
    # negative values returns NaN with a single loss and is not Sortino.
    downside = np.minimum(excess, 0.0)
    dd = float(np.sqrt(np.mean(np.square(downside))))
    return float(excess.mean() / dd * np.sqrt(252)) if dd > 0 else 0.0


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
            if equity is empty or entirely NaN — there is no trough to report.
    """
    # idxmin() raises ValueError on an empty or all-NaN Series rather than
    # returning None, so that case must be handled before calling it.
    if equity.empty or equity.isna().all():
        return 0.0, None
    rolling_max = equity.cummax()
    dd = (equity - rolling_max) / rolling_max
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


def _kelly_fraction(pA: float, nA: float, pB: float, pair_type: str) -> float:
    """
    Uncapped Kelly fraction f* = p - (1-p)/b for the combined arbitrage trade.

    Mirrors strategy._kelly_p and strategy.compute_trade so the dashboard scatter
    shows the same theoretical Kelly the live sizer would compute (before the
    BUDGET_FRACTION cap). Returns 0.0 when there is no edge.

    Args:
        pA (float): YES ask price of market A at entry.
        nA (float): NO ask price of market A at entry.
        pB (float): YES ask price of market B at entry.
        pair_type (str): "time_series" or "same_title" — selects the probability model.

    Returns:
        float: Uncapped Kelly fraction, clamped to be >= 0.
    """
    cost = nA + pB
    net_spread = (1.0 - nA - pB) - fee_per_pair_approx(nA, pB)
    if cost <= 0 or net_spread <= 0:
        return 0.0
    b = net_spread / cost
    p = SAME_TITLE_CO_RESOLVE_PROB if pair_type == "same_title" else (1.0 - pA * (1.0 - pB))
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

    # Entry price bucket
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


# ─── Section 4: Trade-Level Diagnostics ──────────────────────────────────────

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
    fig_slip.update_layout(title="Slippage Distribution (Actual − Expected Payoff, $)",
                            xaxis_title="Slippage ($)", yaxis_title="Count")

    # Best and worst trades table
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
                f"<td>${t.total_cost:.2f}</td>"
                f"<td style='color:{'#2E7D32' if t.profit>=0 else '#C62828'}'>"
                f"${t.profit:+.2f} ({t.profit_ratio:.1%})</td>"
                f"</tr>")

    table_html = """
<div style="margin:16px 0; font-family:sans-serif;">
<b>Top 5 Trades</b>
<table style="width:100%;border-collapse:collapse;font-size:13px;margin-top:8px;">
<tr style="background:#E8F5E9;font-weight:bold;">
  <th>Entry</th><th>Title</th><th>Type</th><th>n</th><th>Cost</th><th>Profit</th>
</tr>
""" + "".join(_trow(t, "#F9FBE7") for t in best) + """
</table>
<br>
<b>Worst 5 Trades</b>
<table style="width:100%;border-collapse:collapse;font-size:13px;margin-top:8px;">
<tr style="background:#FFEBEE;font-weight:bold;">
  <th>Entry</th><th>Title</th><th>Type</th><th>n</th><th>Cost</th><th>Profit</th>
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


# ─── Section 5: Risk Metrics ──────────────────────────────────────────────────

def _section_risk(trades: list[BacktestTrade], equity_df: pd.DataFrame,
                  initial_balance: float) -> str:
    """
    Build the "Risk Metrics" HTML section.

    Renders a Kelly fraction vs. actual fraction scatter plot (to assess sizing
    discipline) and a portfolio value over time chart showing capital deployment.

    Args:
        trades (list[BacktestTrade]): Completed backtest trades for sizing analysis.
        equity_df (pd.DataFrame): Daily equity curve with columns [date, portfolio_value].
        initial_balance (float): Starting portfolio value in dollars.

    Returns:
        str: Self-contained HTML section string. Returns a "No trades" placeholder
            if the trades list is empty.
    """
    if not trades:
        return _SECTION_STYLE.format(title="Risk Metrics") + "<p>No trades.</p>"

    # Kelly vs actual sizing scatter. The actual fraction uses the simulated
    # balance at each trade's entry (the base its Kelly budget was computed
    # from) — dividing by the initial balance would distort as equity drifts.
    kelly_fracs = [_kelly_fraction(t.entry_pA, t.entry_nA, t.entry_pB, t.pair_type) for t in trades]
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
        entry_by_date[t.entry_date] = entry_by_date.get(t.entry_date, 0.0) + t.total_cost
        exit_by_date[t.exit_date]   = exit_by_date.get(t.exit_date,   0.0) + t.total_cost

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


# ─── Section 6: Benchmark Comparison ─────────────────────────────────────────

def _section_benchmark(equity_df: pd.DataFrame, start_date: date,
                        initial_balance: float) -> str:
    """
    Build the "Benchmark Comparison" HTML section.

    Fetches S&P 500 data via yfinance, normalizes it to the same starting balance,
    and plots both the strategy and the benchmark on the same chart. Also renders
    a summary table with total return, Sharpe, and max drawdown for each.

    If yfinance fails (network error, no data), the S&P 500 line is omitted and
    only the strategy is shown.

    Args:
        equity_df (pd.DataFrame): Daily equity curve with columns
            [date, portfolio_value, daily_return].
        start_date (date): Start date used for downloading benchmark data.
        initial_balance (float): Starting portfolio value in dollars.

    Returns:
        str: Self-contained HTML section string with comparison table and chart.
    """
    # Fetch S&P 500
    sp_raw = None
    try:
        sp_raw = yf.download("^GSPC", start=start_date.isoformat(),
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
                "sharpe": f"{_sharpe(sp_daily):.2f}",
                "max_dd": f"{_max_drawdown(sp_norm)[0]:.1%}",
            })
        except Exception as e:
            logging.warning("Benchmark computation failed: %s — omitting S&P 500", e)

    strat_ret    = float(equity_df["portfolio_value"].iloc[-1] / initial_balance - 1)
    strat_sharpe = _sharpe(equity_df["daily_return"])
    strat_dd     = _max_drawdown(equity_df["portfolio_value"])[0]

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
) -> Path:
    """
    Assemble all six dashboard sections into a single self-contained HTML file.

    Calls each _section_*() builder in order, concatenates the resulting HTML
    fragments into a full page with an embedded Plotly CDN script tag, then
    writes the file to PROJECT_ROOT. The output file is timestamped so multiple
    backtest runs can be compared without overwriting previous results.

    Args:
        trades (list[BacktestTrade]): Completed backtest trades from run_backtest().
            May be empty, in which case all charts show placeholder messages.
        equity_df (pd.DataFrame): Daily equity curve DataFrame with columns
            [date, portfolio_value, daily_return], produced by _build_equity_curve().
        start_date (date): Backtest start date shown in the page title and header.
        initial_balance (float): Starting portfolio value in dollars, used for
            return calculations and benchmark normalization.

    Returns:
        Path: Absolute path to the generated HTML file
            (PROJECT_ROOT / "backtest_dashboard_YYYY-MM-DD_HHMMSS.html").
    """
    ts = datetime.now(UTC).astimezone().strftime("%Y-%m-%d_%H%M%S")
    out_path = PROJECT_ROOT / f"backtest_dashboard_{ts}.html"

    sections = [
        _section_performance(equity_df, trades, start_date, initial_balance),
        _section_decomposition(trades),
        _section_calibration(trades),
        _section_diagnostics(trades),
        _section_risk(trades, equity_df, initial_balance),
        _section_benchmark(equity_df, start_date, initial_balance),
    ]

    html = f"""<!DOCTYPE html>
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
  Period: {start_date} → {date.today()} &nbsp;|&nbsp;
  Starting balance: ${initial_balance:,.2f} &nbsp;|&nbsp;
  Trades found: {len(trades)}
</p>
{''.join(sections)}
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    logging.info("Dashboard written: %s", out_path)
    return out_path
