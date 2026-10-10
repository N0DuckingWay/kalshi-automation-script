"""Tests for dashboard.py — HTML escaping of Kalshi-controlled titles (BS-20),
the _max_drawdown empty/all-NaN guard (BS-30), the _sharpe/_sortino
annualization base and its per-row use in the benchmark table (DR-56), the
interval-discount (k) section, whose curve and per-k table follow the page-wide
filter bar's k and size cap, the risk-free rate every Sharpe and Sortino
subtracts (the TestRiskFree* classes and TestCapitalDeployedParity), and the
filter bar's "Save as live defaults…" button (TestSaveLiveDefaultsButton), whose
clicked address is read back through defaults_server's own request handler and
proposal parser, and its "Trade using defaults…" link (TestTradeUsingDefaultsLink),
whose id defaults_server's own scan must find near the top of the page.

generate_dashboard() pulls in yfinance (network) and Plotly's full HTML
serialization; the escaping and drawdown fixes are exercised directly against
the cheapest reliable seam instead — _section_diagnostics() (the HTML table
render site, _trow) and _section_risk() (the Plotly hover-text render site) —
so these tests stay fully offline. Both sites are Kalshi-controlled
(BacktestTrade.title_a is a market question straight from the API). The
interval-discount section is tested the same way: _section_interval_discount()
is driven from a hand-built BacktestSweep, never through generate_dashboard().

The header's Fills line (TestFillsHeader) says how many of the primary scenario's
trades walked a modeled order book, and the trade rows and Kelly scatter read what
each trade paid (TestFillsOnRowsAndScatter) while the entry-price buckets and the
spread calibration stay on the entry quotes.

The scenario explorer (PB5) is pinned by VALUE on a non-square band x k grid
whose primary sits off index 0 on both axes, and the seven sections that
predate it are pinned against digests captured on main by
tests/dashboard_golden.py (TestGoldenSections). Tests that do render a whole
page stub dashboard.yf.download and redirect dashboard.PROJECT_ROOT.
"""
import ast
import base64
import dataclasses
import gzip
import hashlib
import html
import inspect
import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
import warnings
from array import array
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock
from urllib.parse import parse_qs

import numpy as np
import pandas as pd
import pytest

from kalshi_betting import backtester, config, dashboard, defaults_server, historical, scanner
from kalshi_betting.backtester import (
    BacktestSweep,
    BacktestTrade,
    CorpusProvenance,
    HalfSplit,
    IntervalCalibration,
    IntervalCalibrationBucket,
    OutcomeLabelCoverage,
    SweepPoint,
)
from kalshi_betting.config import (
    SAME_TITLE_CO_RESOLVE_PROB,
    fee_leg_exact,
    fee_per_pair_approx,
    time_series_mid_spread,
    time_series_profit_prob,
)
from kalshi_betting.dashboard import (
    _kelly_fraction,
    _max_drawdown,
    _section_diagnostics,
    _section_interval_discount,
    _section_risk,
    _section_scenario_explorer,
)
from kalshi_betting.depth_model import DepthModel
from kalshi_betting.treasury import (
    SOURCE_API,
    SOURCE_CACHE,
    SOURCE_UNAVAILABLE,
    RiskFreeRates,
    day_numbers,
)

# test_backtester as a module, never its classes: a Test* class imported
# here would be collected twice
from . import dashboard_golden
from . import test_backtester as _tb
from .conftest import save_config_live_defaults

_XSS_TITLE = "<script>alert(1)</script>Will BTC exceed $80k by December 2026 or later?"


def make_trade(title_a: str = "Will BTC exceed $80k?", profit: float | None = None,
               deadline_gap_days: int | None = None) -> BacktestTrade:
    """Factory for a coherent time-series BacktestTrade covering every field the
    dashboard section builders under test read.

    YES on the earlier contract at 0.30 and NO on the later at 0.40 (later YES
    ask 0.60, earlier NO ask 0.70: both books with no width, so the mid spread
    the forecast reads is 0.30), n=5, settled in the "event by A" win cell
    (A=YES, hence B=YES). _section_risk calls _kelly_fraction on these entry
    prices live, threading the run's
    interval discount. deadline_gap_days is reporting-only on BacktestTrade and
    defaults to None (the same-title / no-gap-recorded shape).
    """
    n = 5
    entry_pA, entry_pB, entry_nA, entry_nB = 0.30, 0.60, 0.70, 0.40
    total_cost = n * (entry_pA + entry_nB)
    fees = fee_leg_exact(n, entry_pA) + fee_leg_exact(n, entry_nB)
    expected_payoff = n * (1.0 - entry_pA - entry_nB) - fees
    if profit is None:
        profit = expected_payoff
    return BacktestTrade(
        pair_type="time_series",
        ticker_a="TICK-A",
        ticker_b="TICK-B",
        title_a=title_a,
        title_b="Will BTC exceed $90k?",
        category="Crypto",
        entry_date=date(2026, 1, 5),
        exit_date=date(2026, 1, 12),
        entry_pA=entry_pA,
        entry_pB=entry_pB,
        entry_nA=entry_nA,
        entry_nB=entry_nB,
        n=n,
        total_cost=total_cost,
        fees=fees,
        outcome_a="yes",
        outcome_b="yes",
        actual_payoff=float(n),
        profit=profit,
        profit_ratio=profit / (total_cost + fees),
        monthly_profit_ratio=0.1,
        kelly_fraction=0.1,
        expected_payoff=expected_payoff,
        slippage=profit - expected_payoff,
        holding_days=7,
        balance_at_entry=1000.0,
        deadline_gap_days=deadline_gap_days,
    )


def make_equity(values: list[float], start: date = date(2026, 1, 5)) -> pd.DataFrame:
    """Build an equity curve in _build_equity_curve's shape (one row per day,
    columns [date, portfolio_value, daily_return]) from raw portfolio values.

    `start` here is the curve's own first date. The real builder prepends a
    leading row one day before the backtest's start_date holding the untouched
    initial balance (DR-03); these section tests only need the column shape and
    a distinguishable series, so they pass the values they want directly rather
    than modelling that row.
    """
    df = pd.DataFrame({
        "date": [start + timedelta(days=i) for i in range(len(values))],
        "portfolio_value": [float(v) for v in values],
    })
    df["daily_return"] = df["portfolio_value"].pct_change().fillna(0.0)
    return df


class TestKellyFraction:
    """dashboard._kelly_fraction maps the legs like scanner.leg_prices and
    prices time-series pairs through config.time_series_profit_prob of the
    mid spread."""

    @pytest.mark.usefixtures("pre_toggle_defaults")
    def test_time_series_flow_through_fixture(self):
        # YES 0.30 + NO 0.40, later YES ask 0.60, both books with no width: a
        # mid spread of 0.30, p = 0.775, f* ≈ 0.1620 at k 0.75. b's denominator
        # carries the fee — the dollars at risk include it, because a losing
        # pair loses cost + fees (DR-62).
        pA, nA, pB, nB = 0.30, 0.70, 0.60, 0.40
        fee = fee_per_pair_approx(pA, nB)
        net_spread = (1.0 - pA - nB) - fee
        b = net_spread / (pA + nB + fee)
        p = time_series_profit_prob(time_series_mid_spread(pA, nA, pB, nB))
        assert _kelly_fraction(pA, nA, pB, nB, "time_series") == pytest.approx(p - (1 - p) / b)
        assert _kelly_fraction(pA, nA, pB, nB, "time_series") == pytest.approx(0.1620, abs=1e-4)

    def test_time_series_wide_book_clamps_to_zero(self):
        assert _kelly_fraction(0.30, 0.70, 0.60, 0.50, "time_series") == 0.0

    def test_same_title_prices_nA_pB_on_the_prior(self):
        nA, pB = 0.20, 0.30
        fee = fee_per_pair_approx(nA, pB)
        net_spread = (1.0 - nA - pB) - fee
        b = net_spread / (nA + pB + fee)
        p = SAME_TITLE_CO_RESOLVE_PROB
        assert _kelly_fraction(0.70, nA, pB, 0.65, "same_title") == pytest.approx(p - (1 - p) / b)

    def test_risk_section_renders_time_series_trade(self):
        # End-to-end through _section_risk: the scatter is built from the
        # five-argument helper on the trade's own entry prices
        equity_df = pd.DataFrame({
            "date": [date(2026, 1, 5), date(2026, 1, 12)],
            "portfolio_value": [1000.0, 1001.33],
        })
        html_out = _section_risk([make_trade()], equity_df, initial_balance=1000.0)
        assert "Kelly Fraction vs Actual Fraction of Balance" in html_out


class TestKellyFractionIntervalDiscount:
    """_kelly_fraction must price at the discount the plotted trades were SIZED
    at: an explicit k wins, and k=None still resolves to the config constant at
    call time (so a monkeypatched constant is honoured)."""

    _PA, _NA, _PB, _NB = 0.30, 0.70, 0.60, 0.40

    def test_none_matches_the_no_argument_call(self):
        explicit_none = _kelly_fraction(self._PA, self._NA, self._PB, self._NB,
                                        "time_series", k=None)
        assert explicit_none == _kelly_fraction(self._PA, self._NA, self._PB, self._NB,
                                                "time_series")

    def test_none_resolves_to_the_config_constant(self):
        assert _kelly_fraction(self._PA, self._NA, self._PB, self._NB, "time_series") == (
            _kelly_fraction(self._PA, self._NA, self._PB, self._NB, "time_series",
                            k=config.TIME_SERIES_INTERVAL_PROB_DISCOUNT)
        )

    def test_explicit_k_of_one_clamps_to_zero(self):
        # k = 1 takes the market at face value: p = 1 - (pB - pA) = 0.70, f* < 0
        assert _kelly_fraction(self._PA, self._NA, self._PB, self._NB,
                               "time_series", k=1.0) == 0.0

    def test_explicit_k_wins_over_a_monkeypatched_constant(self, monkeypatch):
        # With the constant at the never-trade value, k=0.75 must still give
        # test_time_series_flow_through_fixture's ~0.1620 fraction
        monkeypatch.setattr(config, "TIME_SERIES_INTERVAL_PROB_DISCOUNT", 1.0)
        assert _kelly_fraction(self._PA, self._NA, self._PB, self._NB,
                               "time_series", k=0.75) == pytest.approx(0.1620, abs=1e-4)
        # ...and with no override the patched constant governs, proving the
        # sentinel is resolved at call time rather than bound at def time.
        assert _kelly_fraction(self._PA, self._NA, self._PB, self._NB,
                               "time_series", k=None) == 0.0

    def test_same_title_ignores_k(self):
        # same_title prices on the fixed co-resolution prior — no interval
        # discount appears in its model at all.
        base = _kelly_fraction(0.70, 0.20, 0.30, 0.65, "same_title")
        assert _kelly_fraction(0.70, 0.20, 0.30, 0.65, "same_title", k=1.0) == base

    def test_risk_section_threads_k_into_the_scatter(self):
        # At k = 1 every time-series Kelly clamps to 0.0, so the scatter's x
        # values collapse to zero — the visible difference that proves
        # _section_risk hands its k down to _kelly_fraction.
        equity_df = make_equity([1000.0, 1001.33])
        trades = [make_trade()]
        at_config = _section_risk(trades, equity_df, initial_balance=1000.0)
        at_one = _section_risk(trades, equity_df, initial_balance=1000.0, k=1.0)
        assert at_config != at_one
        assert '"x":[0.0]' in at_one.replace(" ", "")


def _sweep_points(ks: list[float]) -> list[SweepPoint]:
    """One SweepPoint per k, each with a distinguishable equity curve."""
    return [
        SweepPoint(k=k, trades=[make_trade()] * i,
                   equity_df=make_equity([1000.0, 1000.0 + 10 * i, 1000.0 + 5 * i]))
        for i, k in enumerate(ks, start=1)
    ]


def _calibration(label: str = "0-7d") -> IntervalCalibration:
    """A two-row calibration: one gap band plus the POOLED row (tier 0.0)."""
    return IntervalCalibration(
        pooled=IntervalCalibrationBucket(
            label="POOLED", tier=0.0, n=10, realised_rate=0.12,
            mean_implied=0.20, empirical_k=0.60,
        ),
        buckets=[IntervalCalibrationBucket(
            label=label, tier=0.15, n=6, realised_rate=0.10,
            mean_implied=0.18, empirical_k=None,
        )],
        excluded_premise_violations=3,
    )


class TestSectionIntervalDiscount:
    """The interval-discount section renders the calibration table, ONE equity
    trace at the k and size cap shown (div id "kd-equity" — the page-wide
    filter bar's k select replaced the native Plotly updatemenus dropdown it
    used to carry) and a per-k table (tbody "kd-rows") with the k shown in
    bold."""

    def test_none_returns_the_placeholder(self):
        out = _section_interval_discount(None)
        assert "Interval Discount (k) Calibration" in out
        assert "No interval-discount sweep for this run." in out
        # The placeholder must not carry a chart or its table
        assert "Plotly.newPlot" not in out and 'id="kd-rows"' not in out

    def test_empty_points_returns_the_placeholder(self):
        point = _sweep_points([0.75])[0]
        sweep = BacktestSweep(primary=point, points=[], calibration=None)
        assert "No interval-discount sweep for this run." in _section_interval_discount(sweep)

    def test_one_trace_at_the_primary_k_and_no_dropdown(self):
        points = _sweep_points([0.60, 0.75, 0.90])
        sweep = BacktestSweep(primary=points[1], points=points,
                              calibration=_calibration())
        out = _section_interval_discount(sweep)

        # The bar's k select replaced the dropdown: no updatemenus, one chart,
        # one trace — the primary's own curve — under a fixed id the filter
        # script redraws
        assert "updatemenus" not in out
        assert out.count("Plotly.newPlot(") == 1 and 'id="kd-equity"' in out
        data, layout = _nth_figure(out, 0)
        assert len(data) == 1 and "visible" not in data[0]
        eq = points[1].equity_df
        assert data[0]["x"] == [d.isoformat() for d in eq["date"]]
        assert data[0]["y"] == pytest.approx(list(eq["portfolio_value"]))
        assert layout["title"]["text"] == (
            "Equity Curve at interval discount k = 0.75, cap not recorded")
        # One row per point, the k shown (the primary here) in bold and
        # marked; the table's rows sit in the body the script rewrites
        body = re.search(r'<tbody id="kd-rows">(.*?)</tbody>', out, re.S).group(1)
        labels = re.findall(r"font-weight:(\d+)'>(.*?)</td>", body)
        assert labels == [("400", "k = 0.60"), ("700", "k = 0.75 (primary)"),
                          ("400", "k = 0.90")]

    def test_primary_matched_by_k_when_points_holds_an_equal_copy(self):
        # BacktestSweep normally shares the object, but a copy must still be
        # located (by k) rather than silently defaulting to the first point.
        points = _sweep_points([0.60, 0.75, 0.90])
        copy = SweepPoint(k=points[2].k, trades=list(points[2].trades),
                          equity_df=points[2].equity_df.copy())
        sweep = BacktestSweep(primary=copy, points=points, calibration=None)
        out = _section_interval_discount(sweep)
        data, _ = _nth_figure(out, 0)
        assert data[0]["y"] == pytest.approx(list(points[2].equity_df["portfolio_value"]))
        assert "font-weight:700'>k = 0.90 (primary)</td>" in out

    def test_kpis_and_calibration_table_render(self):
        points = _sweep_points([0.75])
        sweep = BacktestSweep(primary=points[0], points=points,
                              calibration=_calibration())
        out = _section_interval_discount(sweep)

        # Labelled for the page, not for config.py — the k shown: the run's
        # own as rendered (on an --interval-discount run the override, which
        # config.py never receives), the filter bar's once one is chosen.
        assert "k selected" in out and "0.750" in out
        assert "k used (this run)" not in out and "Configured k" not in out
        assert 'id="kpi-kd_k" style="font-size:26px; font-weight:700; color:#2196F3;">0.750' \
            in out
        assert "Pooled empirical k̂" in out and "0.600" in out
        # Delta = 0.600 - 0.750: the sizer was conservative, so green
        assert "k̂ − k (primary band, pooled)" in out
        assert 'id="kpi-kd_delta" style="font-size:26px; font-weight:700; color:#4CAF50;">' \
               '-0.150' in out
        # Bucket row: label, tier, n, realised, implied; empirical_k of None
        # renders as "-", and the pooled tier of 0.0 does too.
        assert "0-7d" in out and "0.15" in out and "0.1800" in out
        assert "POOLED" in out
        assert "Excluded 3 premise violation(s)" in out
        assert "never writes config.py" in out

    def test_a_caption_says_the_implied_gap_is_the_mid_spread(self):
        # DR-78: k-hat divides by the mean mid spread. The table keeps its
        # "Mean implied gap" header, and one grey line right under the table
        # says what that gap is, in plain words
        points = _sweep_points([0.75])
        sweep = BacktestSweep(primary=points[0], points=points,
                              calibration=_calibration())
        out = _section_interval_discount(sweep)
        caption = ("<p style='font-family:sans-serif;font-size:13px;color:#616161;'>"
                   "k&#770; = realised in-between rate ÷ mean implied gap, where a "
                   "pair's implied gap is its mid spread (B's midpoint minus A's).</p>")
        assert out.count(caption) == 1
        header = '<th style="padding:8px 16px;">Mean implied gap</th>'
        assert out.count(header) == 1
        # Straight after the calibration table, before the premise-violation
        # note and the "Recommendation only" line
        table_end = out.index("</table>", out.index(header))
        assert out.index(caption) == table_end + len("</table>")
        assert out.index(caption) < out.index("Excluded 3 premise violation(s)")
        assert out.index(caption) < out.index("Recommendation only")
        assert "pB − pA" not in out
        # No calibration, no table, so no caption
        bare = BacktestSweep(primary=points[0], points=points, calibration=None)
        assert "mean implied gap, where" not in _section_interval_discount(bare)

    def test_sweep_table_reports_per_k_metrics(self):
        points = _sweep_points([0.60, 0.75])
        sweep = BacktestSweep(primary=points[1], points=points, calibration=None)
        out = _section_interval_discount(sweep)

        # One row per point, primary flagged; trade counts come off the points
        assert "k = 0.60" in out and "k = 0.75 (primary)" in out
        # Point 2's curve is 1000 -> 1020 -> 1010: +1.0% total, -1.0% drawdown
        assert "+1.0%" in out
        assert "-1.0%" in out
        # No calibration to show, but the sweep table and its chart still render
        assert "No time-series candidate was measurable" in out
        assert 'id="kd-equity"' in out
        # No k-hat to compare: the delta is a dash in the default colour
        assert 'id="kpi-kd_delta" style="font-size:26px; font-weight:700; color:#212121;">—' \
            in out

    def test_calibration_label_is_escaped(self):
        points = _sweep_points([0.75])
        sweep = BacktestSweep(primary=points[0], points=points,
                              calibration=_calibration(label=_XSS_TITLE))
        out = _section_interval_discount(sweep)
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in out
        assert "<script>alert(1)" not in out


def _coverage(with_subtitle: int, total: int = 100) -> OutcomeLabelCoverage:
    """An OutcomeLabelCoverage shaped exactly as the census produces one.

    below_floor is computed here the same way the census computes it, so a
    fixture can never claim a verdict its own numbers contradict — but the
    DASHBOARD never recomputes it: it branches on the carried flag.

    The phrasing counts (7/11/82, always used with the default total=100 every
    call site passes) are fixed, real numbers deliberately distinct from the
    with_subtitle values the subtitle-coverage tests substring-match (2, 97,
    1, ...), so the two families of assertions can never accidentally collide
    (DR-71).
    """
    fraction = with_subtitle / total if total else None
    return OutcomeLabelCoverage(
        total=total,
        with_subtitle=with_subtitle,
        with_event_title=with_subtitle,
        subtitle_fraction=fraction,
        event_title_fraction=fraction,
        below_floor=(fraction is not None
                     and fraction < config.BACKTEST_OUTCOME_LABEL_WARN_FRACTION),
        cumulative_markets=7, snapshot_markets=11, unknown_deadline_markets=82,
    )


class TestOutcomeLabelCoverageIsRendered:
    """DR-66b: the census that DR-66 taught the LOG to emit must also reach the
    page.

    The k̂ card beside it is a recommendation for the real-money constant
    TIME_SERIES_INTERVAL_PROB_DISCOUNT, and backtest.py's closing line points
    the operator at the HTML — so a reader of that page must be able to tell a
    label-less run from a good one. Before this, the strings "subtitle",
    "coverage" and "strike-blind" were all absent from a 140kB dashboard
    generated from a run whose census had logged 0.00% coverage.
    """

    @staticmethod
    def _sweep(coverage) -> BacktestSweep:
        points = _sweep_points([0.75])
        return BacktestSweep(primary=points[0], points=points,
                             calibration=_calibration(), label_coverage=coverage)

    # ── Below the floor: the caveat, its consequence and its remedy ──────────

    def test_the_banner_renders_below_the_floor(self):
        out = _section_interval_discount(self._sweep(_coverage(2, total=100)))

        # This run's own numbers, and the configured floor — never a figure
        # measured on some other run (TS-07).
        assert "2.00%" in out
        assert f"{config.BACKTEST_OUTCOME_LABEL_WARN_FRACTION * 100.0:.2f}%" in out
        # The consequence: the pair population is not the shipped scanner's.
        assert "strike-blind" in out
        assert "different strategy" in out
        # The remedy, including the trap that --no-cache alone is not enough.
        assert "backtest_cache/archive_days/" in out
        assert "backtest_cache/live_days/" in out
        assert "--no-cache" in out
        assert "ALONE does not refresh the day slices" in out

    def test_the_caveat_travels_with_the_khat_card(self):
        # A reader who sees only the KPI cards — or screenshots them — must not
        # get a bare recommendation. The label is SUFFIXED, never replaced.
        out = _section_interval_discount(self._sweep(_coverage(2)))
        assert "Pooled empirical k̂" in out
        assert "Pooled empirical k̂ (see caveat above)" in out
        # ...and the number is recoloured to the warning colour, which the
        # healthy render does not do to it.
        assert 'color:#F44336;">0.600' in out

    def test_the_banner_precedes_the_cards(self):
        out = _section_interval_discount(self._sweep(_coverage(2)))
        assert out.index("strike-blind") < out.index("Pooled empirical k̂")

    def test_the_section_still_renders_everything_else(self):
        # The banner is additive: the table, the equity chart, the per-k
        # table and the k cards must all survive it.
        out = _section_interval_discount(self._sweep(_coverage(2)))
        assert 'id="kd-equity"' in out and 'id="kd-rows"' in out
        assert "POOLED" in out
        assert "k selected" in out

    # ── Healthy: the figure is still rendered, and the caveat is not ─────────

    def test_healthy_coverage_renders_the_figure_without_a_banner(self):
        out = _section_interval_discount(self._sweep(_coverage(97, total=100)))

        # Present, so a reader can CONFIRM the run was clean. Absence of a
        # warning must not be the only signal — that is indistinguishable from
        # the feature not existing.
        assert "Outcome-label coverage" in out
        assert "97.00%" in out
        assert "97 of 100 eligible markets" in out
        # ...and no caveat anywhere.
        assert "strike-blind" not in out
        assert "different strategy" not in out
        assert "see caveat above" not in out
        assert "Pooled empirical k̂" in out

    def test_the_verdict_is_carried_not_recomputed(self):
        # The page must branch on the census's own flag so it and the log can
        # never fire on different conditions. A carrier whose numbers look low
        # but whose verdict says otherwise renders NO banner.
        lying = OutcomeLabelCoverage(
            total=100, with_subtitle=1, with_event_title=1,
            subtitle_fraction=0.01, event_title_fraction=0.01,
            below_floor=False,
            cumulative_markets=7, snapshot_markets=11, unknown_deadline_markets=82,
        )
        out = _section_interval_discount(self._sweep(lying))
        assert "1.00%" in out            # the figure is still reported
        assert "strike-blind" not in out  # but the verdict was not re-derived

    # ── The two "nothing to report" states ──────────────────────────────────

    def test_coverage_none_renders_no_banner_and_no_none(self):
        # An older caller, a hand-built sweep, or the Monday-feasibility
        # short-circuit: no census was taken. That is neither healthy nor low.
        out = _section_interval_discount(self._sweep(None))
        assert "was not measured for this run" in out
        assert "strike-blind" not in out
        assert "see caveat above" not in out
        # No "None%" — and no bare "None" anywhere in the rendered fragment.
        assert "None" not in out
        # Everything the section rendered before is untouched.
        assert 'id="kd-equity"' in out and "POOLED" in out
        assert "Pooled empirical k̂" in out and "0.600" in out

    def test_a_defaulted_sweep_still_renders(self):
        # label_coverage is defaulted, so a construction that predates it must
        # render the not-measured line rather than crash.
        points = _sweep_points([0.75])
        out = _section_interval_discount(
            BacktestSweep(primary=points[0], points=points, calibration=None))
        assert "was not measured for this run" in out
        assert "None" not in out

    def test_an_empty_corpus_is_not_reported_as_zero_percent(self):
        # total == 0 makes the fraction UNDEFINED, not 0%: warning there would
        # manufacture a drift alarm out of a corpus that simply has no records.
        empty = OutcomeLabelCoverage(
            total=0, with_subtitle=0, with_event_title=0,
            subtitle_fraction=None, event_title_fraction=None,
            below_floor=False,
            cumulative_markets=0, snapshot_markets=0, unknown_deadline_markets=0,
        )
        out = _section_interval_discount(self._sweep(empty))
        assert "no eligible markets to census" in out
        assert "0.00%" not in out
        assert "strike-blind" not in out
        assert "None" not in out

    def test_the_placeholder_path_is_untouched(self):
        # The sweep-less placeholder must gain nothing: that page shows no k̂
        # card either, so there is no number there to caveat.
        out = _section_interval_discount(None)
        assert "No interval-discount sweep for this run." in out
        assert "Outcome-label coverage" not in out
        assert "Plotly.newPlot" not in out


class TestGenerateDashboardHeaderNotice:
    """A strike-blind corpus changes WHICH PAIRS EXIST, so it taints every
    STRATEGY-DERIVED section — the one-line header notice is the pointer for a
    reader who never scrolls to the interval-discount section.

    Deliberately not "all seven": _section_benchmark plots a yfinance ^GSPC
    download, an external index series with no pair population behind it, so it
    is unaffected. The notice said "every figure on this page" until that was
    corrected; over-warning is the safe direction, but a caveat that overstates
    its own scope is the thing a reader learns to discount.

    generate_dashboard() is exercised here rather than only the section builder
    because a unit test that does not prove the string reaches the rendered
    page is exactly the gap DR-66b is about. yfinance is stubbed out, so this
    stays offline; _section_benchmark already degrades on a failed download.
    """

    @staticmethod
    def _offline(monkeypatch, tmp_path):
        monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(dashboard.yf, "download",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))

    def _page(self, monkeypatch, tmp_path, **kwargs) -> str:
        self._offline(monkeypatch, tmp_path)
        out_path = dashboard.generate_dashboard(
            [make_trade()], make_equity([1000.0, 1010.0, 1005.0]),
            date(2026, 1, 5), 1000.0, **kwargs)
        return out_path.read_text(encoding="utf-8")

    def test_the_notice_renders_below_the_floor(self, monkeypatch, tmp_path):
        points = _sweep_points([0.75])
        sweep = BacktestSweep(primary=points[0], points=points,
                              calibration=_calibration(),
                              label_coverage=_coverage(2))
        page = self._page(monkeypatch, tmp_path, sweep=sweep)

        assert ("the pairs behind every strategy-derived figure on this page "
                "were grouped") in page
        assert "strike-blind" in page
        # The section's own full caveat is there too, with the remedy.
        assert "ALONE does not refresh the day slices" in page

    def test_no_notice_at_healthy_coverage(self, monkeypatch, tmp_path):
        points = _sweep_points([0.75])
        sweep = BacktestSweep(primary=points[0], points=points,
                              calibration=_calibration(),
                              label_coverage=_coverage(97))
        page = self._page(monkeypatch, tmp_path, sweep=sweep)

        assert "strike-blind" not in page
        # ...but the figure itself is on the page, so a clean run is confirmable.
        assert "97.00%" in page

    def test_the_four_positional_call_still_works(self, monkeypatch, tmp_path):
        # Constraint: no new parameter, and the pre-existing positional call
        # renders as before — placeholder section, no coverage line, no notice.
        page = self._page(monkeypatch, tmp_path)
        assert "Kalshi Arbitrage Backtest" in page
        assert "No interval-discount sweep for this run." in page
        assert "strike-blind" not in page
        assert "Outcome-label coverage" not in page


class TestTitleEscaping:
    def test_diagnostics_table_escapes_title(self):
        trades = [make_trade(title_a=_XSS_TITLE)]
        html_out = _section_diagnostics(trades)

        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_out
        assert "<script>alert(1)" not in html_out

    def test_risk_plotly_text_escapes_title(self):
        equity_df = pd.DataFrame({
            "date": [date(2026, 1, 5), date(2026, 1, 12)],
            "portfolio_value": [1000.0, 1005.0],
        })
        trades = [make_trade(title_a=_XSS_TITLE)]
        html_out = _section_risk(trades, equity_df, initial_balance=1000.0)

        # Plotly's own JSON serializer additionally escapes the "/" in
        # "</script>" (e.g. to "/"), so the exact escaped substring
        # varies by Plotly version — what matters is that the "<" is gone
        # (via our html.escape) and the raw unescaped tag never appears.
        assert "&lt;script&gt;alert(1)&lt;" in html_out
        assert "<script>alert(1)" not in html_out

    def test_diagnostics_normal_title_unaffected(self):
        trades = [make_trade(title_a="Will BTC exceed $80k?")]
        html_out = _section_diagnostics(trades)
        assert "Will BTC exceed $80k?" in html_out


class TestMaxDrawdown:
    def test_empty_series_returns_zero_and_none(self):
        result = _max_drawdown(pd.Series([], dtype=float))
        assert result == (0.0, None)

    def test_all_nan_series_returns_zero_and_none(self):
        idx = [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)]
        series = pd.Series([float("nan")] * 3, index=idx)
        result = _max_drawdown(series)
        assert result == (0.0, None)

    def test_all_zero_series_returns_zero_and_none(self):
        # Every point is 0/0 after the cummax division, so the drawdown series
        # is all-NaN even though the equity series itself is not — idxmin()
        # would raise ValueError without the post-division guard.
        idx = [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)]
        series = pd.Series([0.0, 0.0, 0.0], index=idx)
        result = _max_drawdown(series)
        assert result == (0.0, None)

    def test_normal_declining_series_reports_negative_drawdown(self):
        idx = [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3), date(2026, 1, 4)]
        series = pd.Series([100.0, 120.0, 90.0, 110.0], index=idx)
        max_dd, when = _max_drawdown(series)

        # Peak 120 -> trough 90 = -25%
        assert max_dd == pytest.approx(-0.25)
        assert when == date(2026, 1, 3)

    def test_a_curve_that_never_falls_has_no_trough(self):
        # idxmin() of an all-zero drawdown would name the FIRST date — a
        # "trough" on a curve that never fell (the filter's empty view)
        idx = [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)]
        assert _max_drawdown(pd.Series([100.0, 100.0, 105.0], index=idx)) == (0.0, None)


class TestCapitalDeployedIsFeeInclusive:
    """
    TS-12: the Kelly-vs-actual scatter already divided by (total_cost + fees),
    but the trades tables and the capital-deployed chart summed total_cost
    alone — so two charts on one dashboard disagreed by the fee rate.
    """

    def test_trades_table_cell_includes_fees(self):
        t = make_trade()
        out = dashboard._section_diagnostics([t])
        assert f"${t.total_cost + t.fees:.2f}" in out
        assert f">${t.total_cost:.2f}<" not in out

    def test_trades_table_header_says_incl_fees(self):
        out = dashboard._section_diagnostics([make_trade()])
        assert "Cost incl. fees" in out

    def test_capital_deployed_sums_fee_inclusive_cost(self):
        t = make_trade()
        out = dashboard._section_risk([t], make_equity([1000.0, 1010.0]), 1000.0)
        assert f"{t.total_cost + t.fees:.2f}" in out.replace(",", "")

    def test_a_timestamp_date_column_is_read_as_its_date(self):
        # A pd.Timestamp IS a date subclass: kept whole, it matched no trade's
        # entry or exit date and every row read as nothing deployed
        t = make_trade()
        curve = make_equity([1000.0] * 10)
        deployed = dashboard._capital_deployed([t], curve)
        assert max(deployed) == pytest.approx(t.total_cost + t.fees)
        stamped = curve.assign(date=pd.to_datetime(curve["date"]))
        assert dashboard._capital_deployed([t], stamped) == deployed


class TestBenchmarkDownloadWindow:
    """
    DR-03: the ^GSPC download must open on the equity curve's leading
    initial-balance row, one day before the backtest's start_date, so the two
    traces on the benchmark chart cover the same window.

    This is the only pin on _section_benchmark anywhere; without it the
    download's `start` argument is invisible to the gates (reverting it leaves
    the suite green). yfinance is stubbed out, so the test stays offline.
    """

    def _capture(self, monkeypatch, frame: pd.DataFrame) -> dict:
        seen: dict = {}

        def fake_download(ticker, **kwargs):
            seen["ticker"] = ticker
            seen.update(kwargs)
            return frame

        monkeypatch.setattr(dashboard.yf, "download", fake_download)
        return seen

    def test_download_opens_one_day_before_start_date(self, monkeypatch):
        seen = self._capture(monkeypatch, pd.DataFrame())

        out = dashboard._section_benchmark(
            make_equity([1000.0, 1010.0]), date(2026, 6, 1), 1000.0)

        assert seen["ticker"] == "^GSPC"
        assert seen["start"] == "2026-05-31"
        # An empty download still renders the strategy-only section.
        assert "Kalshi Arbitrage Strategy" in out

    def test_download_start_crosses_a_month_boundary_correctly(self, monkeypatch):
        # A plain string slice of the isoformat would give "2026-06-00"; the
        # date arithmetic has to do it.
        seen = self._capture(monkeypatch, pd.DataFrame())

        dashboard._section_benchmark(
            make_equity([1000.0]), date(2026, 1, 1), 1000.0)

        assert seen["start"] == "2025-12-31"


# A series with both a nonzero standard deviation and at least one negative
# value, so _sharpe's std > 0 branch AND _sortino's dd > 0 branch both run.
_RETURNS = pd.Series([0.02, -0.01, 0.015, -0.03, 0.004, 0.011, -0.008, 0.02])


class TestAnnualizationBase:
    """
    DR-56: _sharpe/_sortino hardcoded sqrt(252) and rf/252, but four of their
    five call sites are handed backtester._build_equity_curve output, which has
    one row per CALENDAR day (~365/yr). The ^GSPC benchmark row rendered beside
    the strategy row in the SAME table is the one trading-day series.

    These helpers had essentially no coverage before this class, so a
    "tests must not drop" gate was satisfiable by adding nothing.
    """

    def test_sharpe_scales_by_the_exact_sqrt_ratio(self):
        # The EXACT identity, never an approximate literal: at rf = 0 the
        # helper reduces to mean/std * sqrt(P), so changing only P rescales by
        # sqrt(365/252) = 1.2035002. A wrong-but-close base (e.g. 350, giving
        # 1.1785) would slip through a loose "~1.18" assertion.
        at_252 = dashboard._sharpe(_RETURNS, periods_per_year=252)
        at_365 = dashboard._sharpe(_RETURNS, periods_per_year=365)
        assert at_252 != 0.0
        assert at_365 == pytest.approx(at_252 * math.sqrt(365 / 252))

    def test_sortino_scales_by_the_exact_sqrt_ratio(self):
        at_252 = dashboard._sortino(_RETURNS, periods_per_year=252)
        at_365 = dashboard._sortino(_RETURNS, periods_per_year=365)
        assert at_252 != 0.0
        assert at_365 == pytest.approx(at_252 * math.sqrt(365 / 252))

    def test_magnitude_grows_without_changing_sign(self):
        # The rescale is of MAGNITUDE: a losing strategy's Sharpe gets MORE
        # negative, it does not "improve".
        losing = -_RETURNS
        at_252 = dashboard._sharpe(losing, periods_per_year=252)
        at_365 = dashboard._sharpe(losing, periods_per_year=365)
        assert at_252 < 0 and at_365 < 0
        assert at_365 < at_252

    def test_bare_sharpe_call_annualizes_on_the_calendar_base(self):
        # Pins the DEFAULT to 365 against an independently computed value, not
        # against the constant the implementation happens to read.
        expected = float(_RETURNS.mean() / _RETURNS.std() * math.sqrt(365))
        assert dashboard._sharpe(_RETURNS) == pytest.approx(expected)
        assert dashboard._sharpe(_RETURNS) != pytest.approx(
            dashboard._sharpe(_RETURNS, periods_per_year=252))

    def test_bare_sortino_call_annualizes_on_the_calendar_base(self):
        downside = _RETURNS.where(_RETURNS < 0, 0.0)
        dd = float((downside**2).mean() ** 0.5)
        expected = float(_RETURNS.mean() / dd * math.sqrt(365))
        assert dashboard._sortino(_RETURNS) == pytest.approx(expected)
        assert dashboard._sortino(_RETURNS) != pytest.approx(
            dashboard._sortino(_RETURNS, periods_per_year=252))

    def test_default_is_the_calendar_constant(self):
        assert config.CALENDAR_DAYS_PER_YEAR == 365
        assert config.TRADING_DAYS_PER_YEAR == 252
        assert dashboard._sharpe(_RETURNS) == pytest.approx(
            dashboard._sharpe(_RETURNS,
                              periods_per_year=config.CALENDAR_DAYS_PER_YEAR))

    def test_periods_per_year_is_keyword_only(self):
        # Positionally it would land in rf's slot and silently reinterpret a
        # periodicity as a 252%-per-year hurdle.
        with pytest.raises(TypeError):
            dashboard._sharpe(_RETURNS, 0.0, 252)
        with pytest.raises(TypeError):
            dashboard._sortino(_RETURNS, 0.0, 252)

    def test_nonzero_rf_is_not_a_constant_rescale(self):
        # periods_per_year divides the annual hurdle as well as supplying the
        # sqrt factor, so the sqrt(365/252) identity holds only at rf = 0.
        ratio = (dashboard._sharpe(_RETURNS, 0.05, periods_per_year=365)
                 / dashboard._sharpe(_RETURNS, 0.05, periods_per_year=252))
        assert ratio != pytest.approx(math.sqrt(365 / 252))


class TestBenchmarkAnnualizationPerRow:
    """
    DR-56 call-site pin: the Benchmark Comparison table's two rows must be
    annualized on their own periodicities — ^GSPC (yfinance trading days) at
    252, the strategy's calendar-day equity curve at 365. The strategy call
    takes the value by DEFAULT rather than passing it, so the recording shim
    must READ the real default (`__kwdefaults__`) rather than restate it: a
    shim that re-declares `periods_per_year=CALENDAR_DAYS_PER_YEAR` records its
    own constant for that call and stays green even when the production default
    is flipped to 252 (measured — the same oracle-replays-the-bug shape
    CLAUDE.md records for the archive-walk parity tests). Each call is recorded
    with the SERIES it received, so the pin survives a reordering of the two
    rows; the default's VALUE is pinned separately by TestAnnualizationBase.
    """

    def test_gspc_row_gets_252_and_strategy_row_gets_365(self, monkeypatch):
        idx = pd.to_datetime(["2026-05-31", "2026-06-01", "2026-06-02", "2026-06-03"])
        monkeypatch.setattr(
            dashboard.yf, "download",
            lambda ticker, **kw: pd.DataFrame(
                {"Close": [100.0, 101.0, 99.5, 102.0]}, index=idx),
        )

        seen: list[tuple[pd.Series, int]] = []
        real_sharpe = dashboard._sharpe
        # The REAL default, read off the real function — never restated here.
        real_default = real_sharpe.__kwdefaults__["periods_per_year"]

        def recording_sharpe(series, rf=0.0, *, periods_per_year=real_default):
            seen.append((series, periods_per_year))
            return real_sharpe(series, rf, periods_per_year=periods_per_year)

        monkeypatch.setattr(dashboard, "_sharpe", recording_sharpe)

        equity = make_equity([1000.0, 1010.0, 1005.0, 1020.0])
        out = dashboard._section_benchmark(equity, date(2026, 6, 1), 1000.0)

        # The S&P row must actually have rendered — otherwise the benchmark
        # block was swallowed by its own except and there is nothing to pin.
        assert "S&P 500" in out
        assert "Kalshi Arbitrage Strategy" in out
        assert len(seen) == 2

        strategy_calls = [p for s, p in seen if s.equals(equity["daily_return"])]
        benchmark_calls = [p for s, p in seen if not s.equals(equity["daily_return"])]
        assert strategy_calls == [config.CALENDAR_DAYS_PER_YEAR]
        assert benchmark_calls == [config.TRADING_DAYS_PER_YEAR]


def _recording(monkeypatch, name: str) -> list[int]:
    """Replace dashboard.<name> with a shim recording the periods_per_year each
    call received, and return the list it appends to.

    Like TestBenchmarkAnnualizationPerRow's shim, the default is READ off the
    real function (`__kwdefaults__`) rather than restated: a shim that
    re-declares `periods_per_year=CALENDAR_DAYS_PER_YEAR` would record its own
    constant and stay green even after the production default was flipped.
    """
    real = getattr(dashboard, name)
    real_default = real.__kwdefaults__["periods_per_year"]
    seen: list[int] = []

    def shim(series, rf=0.0, *, periods_per_year=real_default):
        seen.append(periods_per_year)
        return real(series, rf, periods_per_year=periods_per_year)

    monkeypatch.setattr(dashboard, name, shim)
    return seen


class TestCalendarAnnualizationAtTheUnpinnedCallSites:
    """
    DR-56 call-site pin for the three calendar-base _sharpe/_sortino calls not
    already pinned elsewhere: the performance card's Sharpe and Sortino, and
    the interval-discount section's per-k sweep row. FOUR of the module's five
    calls consume a _build_equity_curve output and so must annualize on the
    CALENDAR base (see dashboard._sharpe's own paragraph on its calendar
    default); the fourth of them — the Benchmark table's STRATEGY row, which
    reads equity_df["daily_return"] — and the one TRADING-base call (that
    table's ^GSPC row) are both pinned by TestBenchmarkAnnualizationPerRow
    above.

    All three take the value by DEFAULT, so nothing at the call site would
    break if one of them started passing TRADING_DAYS_PER_YEAR instead; only a
    recorder can see it. Pinning the count as well as the value is what makes
    a call quietly relocated onto the wrong base visible here.
    """

    def test_performance_card_annualizes_both_ratios_on_365(self, monkeypatch):
        sharpe_seen = _recording(monkeypatch, "_sharpe")
        sortino_seen = _recording(monkeypatch, "_sortino")

        equity = make_equity([1000.0, 1010.0, 1005.0, 1020.0])
        out = dashboard._section_performance(
            equity, [make_trade()], date(2026, 1, 5), 1000.0)

        assert "Portfolio Performance" in out
        assert sharpe_seen == [config.CALENDAR_DAYS_PER_YEAR]
        assert sortino_seen == [config.CALENDAR_DAYS_PER_YEAR]

    def test_per_k_sweep_row_annualizes_on_365(self, monkeypatch):
        sharpe_seen = _recording(monkeypatch, "_sharpe")

        points = _sweep_points([0.60, 0.75, 0.90])
        sweep = BacktestSweep(primary=points[1], points=points,
                              calibration=_calibration())
        out = _section_interval_discount(sweep)

        # One Sharpe per swept point, every one of them on the calendar base:
        # each point's equity_df is a _build_equity_curve-shaped calendar-day
        # series, exactly like the performance card's.
        assert "Interval Discount (k) Calibration" in out
        assert sharpe_seen == [config.CALENDAR_DAYS_PER_YEAR] * len(points)


class TestDeploymentIsNotRenderedAsDrawdown:
    """DR-61, at the render sites: the "Max Drawdown" KPI and the per-k sweep
    table must report realized loss, not capital deployment.

    backtester._build_equity_curve used to accumulate cash alone, so an open
    position was carried at ZERO and the curve dived on entry and recovered at
    settlement whatever the outcome. A real 2026-05-01 run rendered "Max
    Drawdown -60.0%" for a k=1.00 point with three trades, all three
    profitable and a +4.8% return.

    This fixture is that shape in miniature: two winning time-series pairs on
    $10, committing $7.34 of it on day one. Cash-only accounting renders
    -73.4%; cost-basis carry renders the $0.34 of taker fees, -3.4%. The curve
    is built by the REAL builder rather than make_equity(), because what is
    under test is what that builder puts in the column.
    """

    _START = date(2026, 1, 5)
    _INITIAL = 10.0

    def _trades(self) -> list[BacktestTrade]:
        first = make_trade()                       # entered 01-05, exits 01-12
        second = dataclasses.replace(first, exit_date=date(2026, 1, 19),
                                     holding_days=14)
        return [first, second]

    def _equity(self) -> pd.DataFrame:
        return backtester._build_equity_curve(
            self._trades(), self._START, self._INITIAL)

    def test_the_fixture_is_all_winners_and_mostly_deployed(self):
        trades = self._trades()
        assert all(t.profit > 0 for t in trades)
        assert all(t.entry_date == self._START for t in trades)
        assert sum(t.total_cost + t.fees for t in trades) == pytest.approx(7.34)
        assert sum(t.fees for t in trades) == pytest.approx(0.34)

    def test_performance_card_renders_the_fees_not_the_deployment(self):
        out = dashboard._section_performance(
            self._equity(), self._trades(), self._START, self._INITIAL)

        assert "Max Drawdown" in out
        # The fees, on the day they were charged — not the -73.4% the
        # deployment used to read as.
        assert "-3.4% (2026-01-05)" in out
        assert "-73.4%" not in out
        # The endpoint is untouched by DR-61, so the headline return is the
        # same number cash-only accounting produced.
        assert "+26.6%" in out

    def test_per_k_sweep_row_renders_the_same_drawdown(self):
        point = SweepPoint(k=config.TIME_SERIES_INTERVAL_PROB_DISCOUNT,
                           trades=self._trades(), equity_df=self._equity())
        out = _section_interval_discount(
            BacktestSweep(primary=point, points=[point], calibration=None))

        assert "-3.4%" in out
        assert "-73.4%" not in out
        # Same base as the performance card's (DR-03's leading row).
        assert "+26.6%" in out


class TestDeadlinePhrasingIsRendered:
    """The phrasing census must reach the PAGE, not only the log.

    DR-66b's lesson: a run whose log said 0.00% coverage produced a dashboard
    containing none of the words that would have warned its reader. The same
    applies here — the cumulative-deadline rule decides which pairs exist at
    all, so a reader of the k̂ card has to be able to see how much of the
    corpus it admitted.
    """

    @staticmethod
    def _coverage(**kw):
        base = {
            "total": 1000, "with_subtitle": 1000, "with_event_title": 30,
            "subtitle_fraction": 1.0, "event_title_fraction": 0.03,
            "below_floor": False, "cumulative_markets": 120,
            "snapshot_markets": 300, "unknown_deadline_markets": 580,
        }
        base.update(kw)
        return backtester.OutcomeLabelCoverage(**base)

    def test_healthy_run_states_the_mix(self):
        html = dashboard._deadline_phrasing_html(self._coverage())
        assert "Deadline phrasing" in html
        assert "120 of 1,000 eligible markets" in html
        assert "12.00%" in html
        assert "300" in html and "580" in html

    def test_zero_cumulative_says_so_in_its_own_sentence(self):
        # The one reading that is actionable without a threshold.
        html = dashboard._deadline_phrasing_html(
            self._coverage(cumulative_markets=0, snapshot_markets=400,
                           unknown_deadline_markets=600)
        )
        assert "could not have produced a time-series trade" in html

    def test_a_healthy_mix_carries_no_such_claim(self):
        html = dashboard._deadline_phrasing_html(self._coverage())
        assert "could not have produced" not in html

    @pytest.mark.parametrize("coverage", [None, "empty"])
    def test_no_corpus_renders_nothing(self, coverage):
        # The outcome-label block above already says which case it was; a
        # second "nothing to report" line would be noise.
        arg = None if coverage is None else self._coverage(
            total=0, with_subtitle=0, with_event_title=0,
            subtitle_fraction=None, event_title_fraction=None,
            cumulative_markets=0, snapshot_markets=0,
            unknown_deadline_markets=0,
        )
        assert dashboard._deadline_phrasing_html(arg) == ""

    def test_it_reaches_the_rendered_section(self):
        # The helper is wired into the section, not merely defined — the whole
        # point of DR-66b is that a measurement which never reaches the page is
        # indistinguishable from one that was never taken.
        points = _sweep_points([0.75])
        sweep = BacktestSweep(primary=points[0], points=points,
                              calibration=_calibration(),
                              label_coverage=self._coverage())
        html = _section_interval_discount(sweep)
        assert "Deadline phrasing" in html
        assert "120 of 1,000 eligible markets" in html

    # ── DR-71: honest labels, and a verdict only when the counts add up ──────

    @staticmethod
    def _corpus():
        # A shared corpus with pairwise-distinct non-zero phrasing counts,
        # rendered from the CENSUS's own return value rather than a hand-built
        # carrier, so a swapped label pair (D1) is caught by the same object
        # the log measured, not by a fixture that might itself assume it.
        cumulative = [{"subtitle": "Yes", "event_title": "E",
                       "title": "Will X happen by Dec 31, 2026?"}
                      for _ in range(3)]
        snapshot = [{"title": "Bitcoin price on Sep 15, 2026?"}
                    for _ in range(2)]
        unknown = [{"title": "Q"} for _ in range(5)]
        return cumulative + snapshot + unknown

    def test_from_census_states_each_bucket_with_the_right_label(self):
        coverage = backtester._log_outcome_label_coverage(self._corpus())
        html = dashboard._deadline_phrasing_html(coverage)
        assert "3 of 10 eligible markets are worded as a cumulative" in html
        assert "2 are snapshots" in html
        assert "5 carry no deadline wording" in html
        # The two refusal reasons are split, not folded into one sentence: a
        # snapshot is refused because its probability need not nest, an
        # unrecognised wording because nesting cannot be shown at all — and
        # the old folded sentence ("their probabilities do not nest") must be
        # gone, not merely joined by the new text.
        assert "need not nest" in html
        assert "known over-refusal" in html
        assert "nesting cannot be shown" in html
        assert "do not nest" not in html
        # A consistent, non-zero corpus renders no mismatch caveat.
        assert "counts do not sum to the corpus" not in html

    @pytest.mark.parametrize("cumulative_markets,snapshot_markets,unknown_deadline_markets", [
        (0, 1, 1),      # sum 2, UNDER total 1000
        (0, 600, 600),  # sum 1200, OVER total 1000 — a `>=` consistency check
                         # would wrongly accept this direction (C5-ADV-1)
    ])
    def test_mismatched_counts_render_no_verdict(
        self, cumulative_markets, snapshot_markets, unknown_deadline_markets,
    ):
        # A carrier whose phrasing counts don't add up to the corpus (a stale
        # carrier, a hand-built fixture) must not assert a verdict its own
        # numbers cannot support, even when cumulative_markets is 0. Total is
        # 1000 (the base fixture's default) in both directions.
        mismatched = self._coverage(
            cumulative_markets=cumulative_markets,
            snapshot_markets=snapshot_markets,
            unknown_deadline_markets=unknown_deadline_markets,
        )
        html = dashboard._deadline_phrasing_html(mismatched)
        assert "could not have produced a time-series trade" not in html
        assert "counts do not sum to the corpus" in html


# ═══ PB5: the "Scenario Explorer" section (band x k x population sweep) ══════

def _scn_trade(profit: float = 5.0, event_ticker: str = "") -> BacktestTrade:
    """make_trade's time-series trade with a settable profit and event ticker
    (BacktestTrade.event_ticker drives the explorer's concentration figures)."""
    return dataclasses.replace(make_trade(profit=profit), event_ticker=event_ticker)


def _scn_point(band, k, population="all", trades=None, values=None,
               halves=None, ex_top=None, *, tier_floors: bool = True) -> SweepPoint:
    """One scenario SweepPoint (tier_floors=False: a tier-floors-off one)."""
    return SweepPoint(
        k=k, trades=trades if trades is not None else [_scn_trade()],
        equity_df=make_equity(values if values is not None else [1000.0, 1010.0]),
        spread_band=band, population=population, halves=halves, ex_top_event=ex_top,
        tier_floors=tier_floors,
    )


def _scn_calibration(label: str = "0-7d", n: int = 6) -> IntervalCalibration:
    return IntervalCalibration(
        pooled=IntervalCalibrationBucket(label="POOLED", tier=0.0, n=10,
                                         realised_rate=0.12, mean_implied=0.20,
                                         empirical_k=0.60),
        buckets=[IntervalCalibrationBucket(label=label, tier=0.15, n=n,
                                           realised_rate=0.10, mean_implied=0.18,
                                           empirical_k=None)],
        excluded_premise_violations=0,
    )


def _scn_coverage() -> OutcomeLabelCoverage:
    return OutcomeLabelCoverage(
        total=10, with_subtitle=10, with_event_title=10,
        subtitle_fraction=1.0, event_title_fraction=1.0, below_floor=False,
        cumulative_markets=1, snapshot_markets=1, unknown_deadline_markets=8,
    )


def _fail_on_constant(token):  # pragma: no cover - only runs on a failure
    pytest.fail(f"JSON payload contained a non-finite constant token: {token}")


def _scn_blocks(section_html: str) -> dict[str, dict]:
    """Every packed block of the scenario explorer ("scn-data" and each
    "scn-cap-<i>"), inflated as its script unpacks them (base64, then gzip)
    and parsed strictly (a NaN/Infinity token fails the test), by id."""
    return {element_id: _decode_block(body)
            for element_id, body in _PACKED.findall(section_html)
            if element_id.startswith("scn-")}


def _scn_data(section_html: str) -> dict:
    """The explorer's base block (id="scn-data"): axes, labels, populations,
    dates, each band's calibration, the metrics and the cap-independent
    matrices, and the primary [band, k, cap] indexes."""
    return _scn_blocks(section_html)["scn-data"]


def _scn_cap(section_html: str, ci: int | None = None) -> dict:
    """One size cap's block (id="scn-cap-<ci>"): its cells, same-title row,
    banner, cap-dependent matrices and titles — the primary cap's when ci is
    None."""
    blocks = _scn_blocks(section_html)
    if ci is None:
        ci = blocks["scn-data"]["primary"][2]
    return blocks[f"scn-cap-{ci}"]


def _first_figure(section_html: str) -> tuple[list, dict]:
    """(data, layout) of the first Plotly.newPlot call in a fragment — the
    explorer's heatmap — with plotly's typed arrays decoded to lists."""
    decoder = json.JSONDecoder()
    i = section_html.index("Plotly.newPlot(") + len("Plotly.newPlot(")
    args = []
    while len(args) < 3:
        while section_html[i] in " \n\t,":
            i += 1
        value, i = decoder.raw_decode(section_html, i)
        args.append(value)
    return (dashboard_golden._decode_typed_arrays(args[1]),
            dashboard_golden._decode_typed_arrays(args[2]))


def _options(section_html: str, select_id: str) -> list[tuple[str, bool, str]]:
    """(value, selected, text) of every <option> of one <select> — double-quoted
    id, and whatever attributes follow it (disabled, autocomplete)."""
    body = re.search(rf'<select id="{select_id}"[^>]*>(.*?)</select>', section_html).group(1)
    return [(v, bool(sel), txt) for v, sel, txt in
            re.findall(r'<option value="(\d+)"( selected)?>(.*?)</option>', body)]


class _Grid:
    """A NON-square 3-band x 2-k grid whose primary sits at index 1 on BOTH
    axes, so a transposed payload, a primary hard-coded to index 0, or a
    population read off the wrong point all show up as wrong numbers.

    Cell c = band_index * 2 + k_index. Its "all" point has c + 1 trades, its
    "ladder" point 10 + c and its "cross" point 20 + c — except that cell 5
    has no ladder and cell 0 no cross. The "all" returns are NaN / 0 / +1% /
    +2% / +3% / -1% (the NaN from a NaN final value), and the H1 / H2 halves
    have DIFFERENT rank orders, so the split-half correlation is a specific
    non-trivial number rather than +/-1.

    PB7: each cell also carries a "time_series" point — the population the
    heatmap, banner and curve read — whose figures MIRROR its "all" point, as
    on the DR-73 calibration corpus, which has no same-title pair (there
    time_series == all). The tests below therefore read the same numbers they
    always did; TestScenarioExplorerHeadlinePopulation proves, on a fixture
    where the two differ, that the headline reads "time_series".
    """

    BANDS = [(0.0, 1.0), (0.3, 0.6), (0.35, 0.8)]
    KS = [0.65, 0.75]
    H1 = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    H2 = [0.03, -0.01, 0.05, 0.00, 0.02, 0.02]
    FINALS = [float("nan"), 1000.0, 1010.0, 1020.0, 1030.0, 990.0]
    PRIMARY_TRADES = [(10.0, "EVT-A"), (30.0, "EVT-C"), (-5.0, "EVT-B"), (0.0, "")]
    PRIMARY_VALUES = [1000.0, 1040.0, 980.0, 1020.0]

    @classmethod
    def all_trades(cls, c: int) -> list[BacktestTrade]:
        if c == 3:
            return [_scn_trade(p, e) for p, e in cls.PRIMARY_TRADES]
        return [_scn_trade(5.0, "EVT-A") for _ in range(c + 1)]

    @classmethod
    def sweep(cls, *, top_event: str = "EVT-C") -> BacktestSweep:
        scenarios, primary = [], None
        for bi, band in enumerate(cls.BANDS):
            for ki, k in enumerate(cls.KS):
                c = bi * 2 + ki
                trades = cls.all_trades(c)
                values = (cls.PRIMARY_VALUES if c == 3
                          else [1000.0, 1000.0 + 5 * c, cls.FINALS[c]])
                ex_top = (top_event, -0.015) if c == 3 else ("EVT-A", 0.01)
                if c == 3 and top_event != "EVT-C":
                    trades = [dataclasses.replace(t, event_ticker=top_event)
                              if t.event_ticker == "EVT-C" else t for t in trades]
                pt = _scn_point(band, k, "all", trades, values,
                                HalfSplit(cls.H1[c], cls.H2[c], 1, 1), ex_top)
                scenarios.append(pt)
                scenarios.append(dataclasses.replace(pt, population="time_series"))
                if c == 3:
                    primary = pt
                if c != 5:
                    scenarios.append(_scn_point(
                        band, k, "ladder", [_scn_trade(1.0)] * (10 + c),
                        [1000.0, 1000.0 + c, 1000.0 + 2 * c]))
                if c != 0:
                    scenarios.append(_scn_point(
                        band, k, "cross", [_scn_trade(-1.0)] * (20 + c),
                        [1000.0, 999.0 - c]))
        same_title = _scn_point(None, cls.KS[1], "same_title",
                                [_scn_trade(2.0)] * 7, [1000.0, 1002.0])
        return BacktestSweep(
            primary=primary, points=[primary], calibration=_scn_calibration(),
            label_coverage=_scn_coverage(), scenarios=scenarios,
            same_title_point=same_title,
            calibrations_by_band={cls.BANDS[0]: _scn_calibration(),
                                  cls.BANDS[1]: _scn_calibration("8-15d", 9),
                                  cls.BANDS[2]: None},
            same_event_ladders=True, split_date=date(2026, 1, 6),
        )

    @classmethod
    def section(cls, **kwargs) -> str:
        return _section_scenario_explorer(cls.sweep(**kwargs))

    # The tier-floors-off family: BANDS[0] (floor 0, below both tiers) is the
    # one band a tier binds at; BANDS[1] and BANDS[2] (floors 0.3 and 0.35)
    # sit at or above both, so they were never simulated again. Band 0's
    # tier-off cells differ from its tier-on ones on every figure, and every
    # population's count differs from every other's: at k index i, "all"
    # holds 40 + i trades, "time_series" 30 + i, "ladder" 50 + i and "cross"
    # 60 + i. Its time-series returns are -5% and +10%.
    OFF_TS_FINALS = [950.0, 1100.0]
    OFF_H1 = [0.07, -0.02]
    OFF_H2 = [0.01, 0.04]

    @classmethod
    def off_points(cls) -> list[SweepPoint]:
        band, points = cls.BANDS[0], []
        for ki, k in enumerate(cls.KS):
            halves = HalfSplit(cls.OFF_H1[ki], cls.OFF_H2[ki], 1, 1)
            values = [1000.0, 1000.0 + 7 * (ki + 1), cls.OFF_TS_FINALS[ki]]
            for population, n in (("all", 40), ("time_series", 30)):
                points.append(_scn_point(band, k, population,
                                         [_scn_trade(7.0, "EVT-OFF")] * (n + ki), values,
                                         halves, ("EVT-OFF", 0.05 + ki), tier_floors=False))
            points.append(_scn_point(band, k, "ladder", [_scn_trade(3.0)] * (50 + ki),
                                     [1000.0, 1003.0 + ki], tier_floors=False))
            points.append(_scn_point(band, k, "cross", [_scn_trade(-3.0)] * (60 + ki),
                                     [1000.0, 997.0 - ki], tier_floors=False))
        return points

    @staticmethod
    def off_calibration() -> IntervalCalibration:
        """Band 0's tier-off calibration, labelled with its floor alone
        (backtester._interval_calibration with tier_floors False): at floor 0
        every bucket's tier is 0.0."""
        return IntervalCalibration(
            pooled=IntervalCalibrationBucket(label="POOLED", tier=0.0, n=12,
                                             realised_rate=0.15, mean_implied=0.30,
                                             empirical_k=0.50),
            buckets=[IntervalCalibrationBucket(label="16-30d", tier=0.0, n=7,
                                               realised_rate=0.20, mean_implied=0.35,
                                               empirical_k=0.571)],
            excluded_premise_violations=0,
        )

    @classmethod
    def sweep_tiers(cls) -> BacktestSweep:
        """sweep() plus a tier-floors-off family for BANDS[0] at both k, in
        all four populations, with its tier-off calibration."""
        return dataclasses.replace(
            cls.sweep(), tier_off_scenarios=cls.off_points(),
            tier_off_calibrations_by_band={cls.BANDS[0]: cls.off_calibration()})


class TestScenarioExplorerPayload:
    """The data blocks the inline script reads: their indexing, their
    per-population rows and their values, all pinned by value on a
    non-square grid. The direct call renders the sweep's own eager points,
    at the run's own (here unrecorded) size cap: one cap block."""

    def test_the_grid_is_band_major_and_not_transposed(self):
        section = _Grid.section()
        data, cap = _scn_data(section), _scn_cap(section)
        assert data["bands"] == [[0.0, 1.0], [0.3, 0.6], [0.35, 0.8]]
        assert data["ks"] == [0.65, 0.75]
        assert data["caps"] == [None] and data["cap_labels"] == ["not recorded"]
        assert data["populations"] == ["all", "ladder", "cross", "time_series"]
        pop = {name: i for i, name in enumerate(data["populations"])}
        assert len(cap["cells"]) == 3
        assert all(len(row) == 2 for row in cap["cells"])
        for bi in range(3):
            for ki in range(2):
                c = bi * 2 + ki
                cell = cap["cells"][bi][ki]
                assert cell[pop["all"]]["trades"] == c + 1
                # Each population row is its OWN point's figures, never the
                # All point's — 10 + c and 20 + c trades, not c + 1.
                if c == 5:
                    assert cell[pop["ladder"]] is None
                else:
                    assert cell[pop["ladder"]]["trades"] == 10 + c
                if c == 0:
                    assert cell[pop["cross"]] is None
                else:
                    assert cell[pop["cross"]]["trades"] == 20 + c

    def test_the_primary_cell_kpis_by_value(self):
        sweep = _Grid.sweep()
        all_row = _scn_cap(_section_scenario_explorer(sweep))["cells"][1][1][0]
        trades = _Grid.all_trades(3)
        assert all_row["trades"] == 4
        # profit > 0 strictly: the zero-profit trade is not a win.
        assert all_row["win_rate"] == pytest.approx(0.5)
        # Mean per trade = mean of profit / (total_cost + fees), fee-inclusive.
        expected_mean = sum(t.profit / (t.total_cost + t.fees) for t in trades) / 4
        assert all_row["mean_per_trade"] == pytest.approx(expected_mean)
        assert all_row["total_return"] == pytest.approx(0.02)
        assert all_row["final_balance"] == pytest.approx(1020.0)
        assert all_row["max_drawdown"] == pytest.approx((980.0 - 1040.0) / 1040.0)
        eq = sweep.primary.equity_df
        assert all_row["sharpe"] == pytest.approx(dashboard._sharpe(eq["daily_return"]))
        assert all_row["sortino"] == pytest.approx(dashboard._sortino(eq["daily_return"]))
        assert all_row["sortino"] != pytest.approx(all_row["sharpe"])
        assert (all_row["h1_return"], all_row["h2_return"]) == (0.04, 0.00)
        assert all_row["top_event"] == "EVT-C"
        # 30 / (10 + 30): the share is over POSITIVE event P&L only; a net-P&L
        # denominator would read 30 / 35.
        assert all_row["top_event_share"] == pytest.approx(0.75)
        assert all_row["ex_top_return"] == pytest.approx(-0.015)

    def test_population_rows_are_their_own_simulations(self):
        cap = _scn_cap(_Grid.section())
        ladder, cross = cap["cells"][1][1][1], cap["cells"][1][1][2]
        assert ladder["total_return"] == pytest.approx(0.006)   # 1000 -> 1006
        assert ladder["final_balance"] == pytest.approx(1006.0)
        assert cross["total_return"] == pytest.approx(-0.004)   # 1000 -> 996
        assert "h1_return" not in ladder and "equity" not in ladder
        # The same-title row is its cap's own (one cap here: the run's own)
        same_title = cap["same_title"]
        assert same_title["trades"] == 7
        assert same_title["total_return"] == pytest.approx(0.002)

    def test_calibration_is_the_selected_bands_own(self):
        data = _scn_data(_Grid.section())
        labels = [[row["label"] for row in cal] if cal else None
                  for cal in data["calibration_by_band"]]
        assert labels == [["0-7d", "POOLED"], ["8-15d", "POOLED"], None]
        assert data["calibration_by_band"][1][0]["n"] == 9
        # The POOLED row's tier of 0.0 means "no tier" and ships as null.
        assert data["calibration_by_band"][0][1]["tier"] is None

    def test_json_is_strict_and_carries_no_non_finite_value(self):
        section = _Grid.section()
        # Every packed block decodes strictly (_decode_block: a NaN or
        # Infinity token fails the test) — a grep for "NaN" would read the
        # base64 alphabet, which can spell it by chance
        blocks = _scn_blocks(section)
        assert set(blocks) == {"scn-data", "scn-cap-0"}
        cap, n = blocks["scn-cap-0"], len(blocks["scn-data"]["dates"])
        assert cap["cells"][0][0][0]["total_return"] is None
        # The per-cell curve is the headline (time-series) population's, as
        # change points: the NaN final value is a gap, never a number
        assert _expand(cap["cells"][0][0][3]["equity"], n)[2] is None
        assert cap["matrices"]["total_return"][0][0] is None
        # ... and outside the blocks — the heatmap's and the curve's figure
        # JSON Python rendered — no non-finite token either (the blocks'
        # bodies stripped first, since base64 can spell "NaN")
        stripped = re.sub(r'(data-encoding="gzip\+base64">)[^<]*', r"\1", section)
        assert stripped.count('data-encoding="gzip+base64">') == 2
        assert "NaN" not in stripped and "Infinity" not in stripped

    def test_python_s_curve_is_the_one_the_script_expands(self, monkeypatch):
        # 18783.345 sits a hair above a half cent: Python's round() reads it
        # 18783.35 and numpy's 18783.34. The primary curve Python draws and
        # the one the script expands from the primary cap's block when a
        # reader returns to it must be one rounding, not a cent apart
        values = [1000.0, 18783.345, 999.995, 1001.005]
        assert round(values[1], 2) != float(np.round(values[1], 2))   # the case is real
        monkeypatch.setattr(_Grid, "PRIMARY_VALUES", values)
        section = _Grid.section()
        drawn, _ = _nth_figure(section, 1)
        data, cap = _scn_data(section), _scn_cap(section)
        ts = data["populations"].index("time_series")
        shipped = _expand(cap["cells"][1][1][ts]["equity"], len(data["dates"]))
        assert drawn[0]["y"] == shipped
        assert shipped[1] == float(np.round(values[1], 2))

    def test_a_script_closing_ticker_cannot_end_the_data_block(self):
        hostile = "EVT-</script><b>x"
        section = _Grid.section(top_event=hostile)
        assert hostile not in section and "<b>x" not in section
        # Base64 has no "<", so nothing inside a block can close it early:
        # each block ends exactly where its encoded bytes end
        ids = re.findall(r'id="(scn-[a-z0-9-]+)" data-encoding="gzip\+base64">', section)
        assert ids == ["scn-data", "scn-cap-0"]
        for element_id in ids:
            opening = f'id="{element_id}" data-encoding="gzip+base64">'
            body = section[section.index(opening) + len(opening):]
            assert re.fullmatch(r"[A-Za-z0-9+/=]+", body[:body.index("</script>")])
        assert _scn_cap(section)["cells"][1][1][0]["top_event"] == hostile

    def test_every_curve_is_on_the_shared_axis(self):
        sweep = _Grid.sweep()
        section = _section_scenario_explorer(sweep)
        data, cap = _scn_data(section), _scn_cap(section)
        assert data["dates"] == [d.isoformat() for d in sweep.primary.equity_df["date"]]
        axis = dashboard._equity_axis(sweep.primary.equity_df)
        points = {(p.spread_band, p.k): p for p in sweep.scenarios
                  if p.population == "time_series"}
        for bi, row in enumerate(cap["cells"]):
            for ki, cell in enumerate(row):
                # The per-cell curve is the headline (time-series)
                # population's, change points on the shared axis, expanded by
                # the script; the "all" entry ships none
                curve = _expand(cell[3]["equity"], len(data["dates"]))
                point = points[(_Grid.BANDS[bi], _Grid.KS[ki])]
                assert curve == dashboard._curve_on_axis(point.equity_df, axis)
                assert "equity" not in cell[0]


class TestScenarioExplorerControls:
    """The selects, the heatmap's metric select and the fragility banner."""

    def test_selects_are_labelled_and_preselect_only_the_primary(self):
        section = _Grid.section()
        assert _options(section, "scn-band-select") == [
            ("0", False, "max(tier,0)-1"),
            ("1", True, "max(tier,0.3)-0.6 (primary)"),
            ("2", False, "max(tier,0.35)-0.8"),
        ]
        assert _options(section, "scn-k-select") == [
            ("0", False, "k = 0.65"),
            ("1", True, "k = 0.75 (primary)"),
        ]
        # The run's own cap alone (the direct call walks the eager points);
        # _Grid's points record none
        assert _options(section, "scn-cap-select") == [("0", True, "not recorded (primary)")]
        data = _scn_data(section)
        assert data["primary"] == [1, 1, 0]
        # The labels the filter bar's calls are matched against: the bar's own
        assert data["band_labels"] == [dashboard._band_option(b) for b in _Grid.BANDS]
        assert data["k_labels"] == [dashboard._k_option(k) for k in _Grid.KS]
        # Rendered disabled (the script enables them once its data is
        # unpacked), and never restored by a browser
        for sid in ("scn-band-select", "scn-k-select", "scn-cap-select", "scn-metric"):
            assert f'<select id="{sid}" disabled autocomplete="off">' in section

    def test_labels_stay_distinct_for_off_grid_values(self):
        # An off-grid --interval-discount within a rounding of a grid member,
        # and an off-grid band floor, must not share a label with the member:
        # the heatmap's axes are categorical and would merge them.
        bands = [(0.3, 0.6), (0.3000001, 0.6)]
        ks = [0.65, 0.651, 0.70]
        scenarios = [_scn_point(b, k) for b in bands for k in ks]
        sweep = BacktestSweep(primary=scenarios[1], points=[scenarios[1]],
                              calibration=None, label_coverage=_scn_coverage(),
                              scenarios=scenarios)
        heat, _ = _first_figure(_section_scenario_explorer(sweep))
        assert heat[0]["x"] == ["k = 0.65", "k = 0.651", "k = 0.70"]
        assert heat[0]["y"] == ["max(tier,0.3)-0.6", "max(tier,0.3000001)-0.6"]

    def test_the_heatmap_metrics_carry_z_scale_and_title_together(self):
        section = _Grid.section()
        heat, layout = _first_figure(section)
        data, cap = _scn_data(section), _scn_cap(section)
        # PB7: the title names the population the cells are; C4: and the cap
        pop = "Time-series (ladders + cross-event; same-title excluded)"
        assert layout["title"]["text"] == cap["titles"][0] == (
            f"Mean per trade (equal stake) by spread band x k, cap not recorded — {pop}")
        # A scripted select replaced the native dropdown: one figure, no
        # updatemenus (it could not follow the size cap)
        assert "updatemenus" not in layout
        metrics = data["metrics"]
        assert [m["label"] for m in metrics] == [html.unescape(o[2]) for o in
                                                 _options(section, "scn-metric")]
        labels = [m["label"] for m in metrics]
        for m, t in zip(metrics, cap["titles"], strict=True):
            # A figure of the trades names the cap; k-hat does not depend on it
            if m["key"] in ("empirical_k", "khat_minus_k"):
                assert t == f"{m['label']} by spread band x k — {pop}"
            else:
                assert t == f"{m['label']} by spread band x k, cap not recorded — {pop}"
        z = {label: (cap["matrices"] | data["matrices"])[m["key"]]
             for label, m in zip(labels, metrics, strict=True)}
        assert z["Trade count"] == [[1, 2], [3, 4], [5, 6]]
        assert z["Total return"][0][0] is None
        assert z["Total return"][1] == pytest.approx([0.01, 0.02])
        assert z["H1 return"] == [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]]
        assert z["H2 return"] == [[0.03, -0.01], [0.05, 0.0], [0.02, 0.02]]
        # The figure as rendered: the default metric's z
        assert heat[0]["z"] == z["Mean per trade (equal stake)"]
        # A count is never negative: its own colour scale, auto-ranged.
        count = metrics[labels.index("Trade count")]
        assert count["zmid"] is None
        assert count["colorscale"] != metrics[0]["colorscale"]
        assert metrics[0]["zmid"] == 0
        assert heat[0]["colorscale"] == metrics[0]["colorscale"]
        assert heat[0]["customdata"] == cap["matrices"]["trades"]

    def test_banner_is_first_and_carries_this_runs_figures(self):
        from scipy.stats import spearmanr
        section = _Grid.section()
        expected_rho = spearmanr(_Grid.H1, _Grid.H2).statistic
        # The halves were chosen so the figure is specific (neither +/-1 nor 0).
        assert f"{expected_rho:+.3f}" == "-0.116"
        banner_at = section.index("band x k cells computed")
        assert banner_at < section.index("Plotly.newPlot(")
        assert "<b>6 band x k cells computed</b>" in section
        # 6 all + 6 time-series + 5 ladder + 5 cross points (the stored
        # scenario points, not every simulation the sweep ran).
        assert "(22 scenario points" in section
        # Every cell has a time-series entry, so no "N of M" clause
        assert "cells have a time-series entry" not in section
        # Finite returns 0, +1%, +2%, +3%, -1% (the NaN cell excluded): 3 of 5.
        assert "60.0% of the cells with a measurable return" in section
        assert "H1 vs H2: -0.116." in section
        assert "The best of 6 correlated cells overstates what you should expect." in section

    def test_equity_div_id_is_the_fixed_token(self):
        assert 'id="scn-equity"' in _Grid.section()


def _nth_figure(section_html: str, n: int) -> tuple[list, dict]:
    """(data, layout) of the n-th (0-based) Plotly.newPlot call in a fragment:
    0 is the explorer's heatmap, 1 its equity curve."""
    decoder = json.JSONDecoder()
    i = -1
    for _ in range(n + 1):
        i = section_html.index("Plotly.newPlot(", i + 1)
    i += len("Plotly.newPlot(")
    args = []
    while len(args) < 3:
        while section_html[i] in " \n\t,":
            i += 1
        value, i = decoder.raw_decode(section_html, i)
        args.append(value)
    return (dashboard_golden._decode_typed_arrays(args[1]),
            dashboard_golden._decode_typed_arrays(args[2]))


_TS_LABEL = "Time-series (ladders + cross-event; same-title excluded)"


class TestScenarioExplorerHeadlinePopulation:
    """PB7: the heatmap, the fragility banner and the equity curve read the
    "time_series" population — every time-series entry simulated alone,
    same-title excluded — never the "all" one, and say so. The fixture makes
    the two DISAGREE on everything: every "all" cell is +50% (a same-title
    windfall, band- and k-independent), every time-series cell loses, and the
    last cell has no time-series point at all."""

    BANDS = [(0.0, 1.0), (0.3, 0.6)]
    KS = [0.65, 0.75]
    TS_FINALS = [900.0, 880.0, 860.0]           # cells 0..2; cell 3 has none
    TS_H1 = [-0.10, -0.30, -0.20]
    TS_H2 = [-0.05, -0.02, -0.40]

    @classmethod
    def sweep(cls) -> BacktestSweep:
        scenarios, primary = [], None
        for bi, band in enumerate(cls.BANDS):
            for ki, k in enumerate(cls.KS):
                c = bi * 2 + ki
                # Every curve spans one calendar, as every scenario of one run
                # does (the page decides its date axis from the primary)
                all_pt = _scn_point(band, k, "all", [_scn_trade(50.0, "EVT-ST")] * 3,
                                    [1000.0, 1250.0, 1500.0],
                                    HalfSplit(0.9, 0.8, 1, 1, 1, 1), ("EVT-ST", 0.4))
                scenarios.append(all_pt)
                if c == 0:
                    primary = all_pt
                if c < 3:
                    scenarios.append(_scn_point(
                        band, k, "time_series", [_scn_trade(-5.0, "EVT-TS")],
                        [1000.0, 950.0, cls.TS_FINALS[c]],
                        HalfSplit(cls.TS_H1[c], cls.TS_H2[c], 1, 1, 2, 2),
                        ("EVT-TS", -0.05)))
        return BacktestSweep(
            primary=primary, points=[primary], calibration=None,
            label_coverage=_scn_coverage(), scenarios=scenarios,
            same_title_point=_scn_point(None, 0.75, "same_title",
                                        [_scn_trade(50.0)] * 3, [1000.0, 1500.0]),
            calibrations_by_band=dict.fromkeys(cls.BANDS),
            same_event_ladders=True, split_date=date(2026, 1, 6))

    def test_the_heatmap_reads_time_series_and_never_falls_back(self):
        section = _section_scenario_explorer(self.sweep())
        heat, layout = _first_figure(section)
        matrices = _scn_cap(section)["matrices"]
        expected = [(f - 1000.0) / 1000.0 for f in self.TS_FINALS]
        assert matrices["total_return"][0] == pytest.approx(expected[:2])
        assert matrices["total_return"][1][0] == pytest.approx(expected[2])
        # Cell 3 has no time-series point: null, never the "all" cell's +50%
        assert matrices["total_return"][1][1] is None
        assert matrices["trades"] == [[1, 1], [1, None]]
        assert matrices["h1_return"] == [[-0.10, -0.30], [-0.20, None]]
        # The figure as rendered: the default metric's, from the same cells
        assert heat[0]["z"] == matrices["mean_per_trade"]
        assert heat[0]["z"][1][1] is None
        # The title names the population on every metric
        assert layout["title"]["text"].endswith(f"— {_TS_LABEL}")
        assert all(title.endswith(f"— {_TS_LABEL}") for title in _scn_cap(section)["titles"])

    def test_same_title_dilution_cannot_reach_the_banner(self):
        from scipy.stats import spearmanr
        section = _section_scenario_explorer(self.sweep())
        # Every time-series cell lost; every "all" cell won
        assert "0.0% of the cells with a measurable return" in section
        rho = spearmanr(self.TS_H1, self.TS_H2).statistic
        assert f"H1 vs H2: {rho:+.3f}." in section
        # The "all" halves are constant (0.9 / 0.8): read, they would make
        # the correlation undefined instead
        assert "not enough data" not in section
        # The grid still counts every cell, and the banner names its population
        assert "<b>4 band x k cells computed</b>" in section
        assert (f"Every figure in this banner, the heatmap and the equity curve is the "
                f"<b>{_TS_LABEL}</b> population's; the KPI table below labels each row "
                "with its own population.") in section
        # ... but its multiple-comparison count is the cells the heatmap
        # shows a value in (cell 3 has no time-series point), never the grid
        assert "; 3 of the 4 cells have a time-series entry and the rest are blank" in section
        assert "The best of 3 correlated cells overstates what you should expect." in section
        assert "The best of 4 correlated cells" not in section

    def test_a_grid_with_no_time_series_cell_says_so(self):
        # Every cell carries an "all" point only (a same-title-only run): the
        # banner must not claim a "best of 0" and must not read the "all"
        # cells in their place.
        scenarios = [_scn_point(band, k, "all", [_scn_trade(50.0, "EVT-ST")],
                                [1000.0, 1500.0], HalfSplit(0.4, 0.5, 1, 1, 1, 1))
                     for band in self.BANDS for k in self.KS]
        sweep = BacktestSweep(primary=scenarios[0], points=[scenarios[0]], calibration=None,
                              label_coverage=_scn_coverage(), scenarios=scenarios)
        section = _section_scenario_explorer(sweep)
        assert "<b>4 band x k cells computed</b>" in section
        assert "; 0 of the 4 cells have a time-series entry" in section
        assert ("No cell has a time-series entry, so nothing on this grid measures the "
                "band or k.") in section
        assert "The best of" not in section
        assert "— of the cells with a measurable return" in section

    def test_the_curve_and_payload_follow_the_headline(self):
        section = _section_scenario_explorer(self.sweep())
        curve, layout = _nth_figure(section, 1)
        # The primary cell's TIME-SERIES curve, not its "all" curve
        assert curve[0]["y"] == [1000.0, 950.0, 900.0]
        assert _TS_LABEL in layout["title"]["text"]
        data, cap = _scn_data(section), _scn_cap(section)
        pop = {name: i for i, name in enumerate(data["populations"])}
        cell = cap["cells"][0][0]
        # Shipped as change points, expanded by the script onto the axis
        assert _expand(cell[pop["time_series"]]["equity"], len(data["dates"])) \
            == [1000.0, 950.0, 900.0]
        assert "equity" not in cell[pop["all"]]
        # ... and both checked populations carry their own extras
        assert cell[pop["time_series"]]["top_event"] == "EVT-TS"
        assert cell[pop["all"]]["top_event"] == "EVT-ST"
        assert cap["cells"][1][1][pop["time_series"]] is None

    def test_every_population_is_labelled(self):
        data = _scn_data(_section_scenario_explorer(self.sweep()))  # the base block
        assert data["labels"] == {
            "time_series": _TS_LABEL,
            "all": "All (time-series + same-title)",
            "ladder": "Ladders (same-event)",
            "cross": "Cross-event",
            # Same-title reads the same under both Tier floors settings
            "same_title": "Same-title (independent of band, k and the tier floors)",
        }
        # The script builds every KPI row from those labels, the headline
        # population first, and no longer spells a bare 'All'
        js = dashboard._SCENARIO_EXPLORER_JS
        assert js.index("L.time_series") < js.index("L.ladder") < js.index("L.all")
        assert "kpiRow('All'" not in js

    def test_the_script_reads_the_headline_population(self):
        # The KPI table's headline row, its sub-row and the redrawn curve
        # exist only in the inline script (TestScenarioExplorerScript runs
        # it); on a corpus with no same-title pair "all" equals
        # "time_series" cell for cell, so a script reading "all" would look
        # right there. Pin the reads themselves.
        js = dashboard._SCENARIO_EXPLORER_JS
        assert "var ts = cell[P.time_series];" in js
        assert "kpiRow(esc(L.time_series), ts) + extrasRow(ts)" in js
        assert ("var values = (ts && ts.equity && ts.equity.length) ? expand(ts.equity) : [];"
                in js)
        assert "kpiRow(esc(L.all), cell[P.all]) + extrasRow(cell[P.all])" in js
        # ... and the population is resolved through the payload's names, so
        # the index the script reads is the time-series point's
        section = _section_scenario_explorer(self.sweep())
        data, cap = _scn_data(section), _scn_cap(section)
        ts_idx = data["populations"].index("time_series")
        cell = cap["cells"][0][0]
        assert _expand(cell[ts_idx]["equity"], len(data["dates"])) == [1000.0, 950.0, 900.0]
        assert cell[ts_idx]["final_balance"] == pytest.approx(900.0)


class TestSplitHalfDegeneracy:
    """PB7: a half with no entries has a 0.0 return that is NOT a measurement.
    The page renders it as null ("—") and leaves the cell out of the
    split-half correlation."""

    @staticmethod
    def _sweep(halves: list[HalfSplit]) -> BacktestSweep:
        scenarios = []
        for i, h in enumerate(halves):
            band = (0.0, 0.5 + 0.1 * i)
            for population in ("all", "time_series"):
                scenarios.append(_scn_point(band, 0.75, population, halves=h))
        return BacktestSweep(primary=scenarios[0], points=[scenarios[0]], calibration=None,
                             label_coverage=_scn_coverage(), scenarios=scenarios)

    def test_an_empty_half_is_null_and_left_out_of_the_correlation(self):
        from scipy.stats import spearmanr
        h1 = [0.01, 0.02, 0.03, 0.04]
        h2 = [0.03, 0.01, 0.04, 0.02]
        halves = [HalfSplit(a, b, 1, 1, 3, 3) for a, b in zip(h1, h2, strict=True)]
        # A fifth cell whose H1 had no entries: its 0.0 would move the rank
        # correlation if it were counted
        halves.append(HalfSplit(0.0, 0.10, 0, 2, 0, 3))
        section = _section_scenario_explorer(self._sweep(halves))
        measured = spearmanr(h1, h2).statistic
        with_zero = spearmanr(h1 + [0.0], h2 + [0.10]).statistic
        assert f"{measured:+.3f}" != f"{with_zero:+.3f}"
        assert f"H1 vs H2: {measured:+.3f} (over the 4 of 5 cells whose two halves both " \
               "had entries)." in section
        heat, _ = _first_figure(section)
        data, cap = _scn_data(section), _scn_cap(section)
        z = cap["matrices"]
        assert [row[0] for row in z["h1_return"]] == [*h1, None]
        # ... while H2 of that cell, which had entries, is still a measurement
        assert [row[0] for row in z["h2_return"]] == [*h2, 0.10]
        # The figure as rendered: the default metric's z
        assert heat[0]["z"] == z["mean_per_trade"]
        pop = {name: i for i, name in enumerate(data["populations"])}
        assert cap["cells"][4][0][pop["time_series"]]["h1_return"] is None
        assert cap["cells"][4][0][pop["all"]]["h1_return"] is None

    def test_every_half_empty_says_not_measurable(self):
        halves = [HalfSplit(0.0, r, 0, 1, 0, 2) for r in (0.1, 0.2, 0.3)]
        section = _section_scenario_explorer(self._sweep(halves))
        assert ("H1 vs H2: not measurable — the split date leaves a half without "
                "entries in 3 of the 3 cells.") in section

    def test_unrecorded_entry_counts_keep_the_return(self):
        # A hand-built four-field HalfSplit records no entry counts; nothing
        # says a half was empty, so its return is kept
        assert dashboard._measured_half(0.0, None) == 0.0
        assert dashboard._measured_half(0.0, 0) is None
        assert dashboard._measured_half(-0.2, 3) == -0.2

    def test_the_golden_fixture_s_empty_h1_reaches_the_page_as_null(self, monkeypatch):
        # The backtester's golden fixture: four of the primary band's five
        # entries (three of its four time-series ones) enter on the first
        # Monday, so the median_low split date IS that Monday and every
        # cell's H1 is empty. Run through the real sweep and rendered.
        from . import test_backtester as tb
        golden = tb.TestPrepareEntriesGolden()
        golden._patch(monkeypatch)
        sweep = backtester.run_backtest_sweep(
            hist_client=tb.MagicMock(), live_client=tb.MagicMock(),
            start_date=golden._START, initial_balance=10_000.0,
            same_event_ladders=True, sweep=False, band_sweep=True)
        assert sweep.split_date == date(2026, 1, 5)
        assert all(p.halves.h1_entries == 0 for p in sweep.scenarios
                   if p.population in ("all", "time_series"))
        section = _section_scenario_explorer(sweep)
        data, cap = _scn_data(section), _scn_cap(section)
        pop = {name: i for i, name in enumerate(data["populations"])}
        for row in cap["cells"]:
            for cell in row:
                for name in ("all", "time_series"):
                    assert cell[pop[name]]["h1_return"] is None
                    assert cell[pop[name]]["h2_return"] is not None
        assert "H1 vs H2: not measurable — the split date leaves a half without entries" \
            in section


class TestNoCandleCapNotice:
    """PB7 put a red notice on any window long enough to need a candlestick
    request over Kalshi's 5,000-candle cap, because every market needing one
    silently had no candles and could never enter. historical.fetch_candlesticks
    now pages such a window, so the page must not claim a truncated
    population on a long window — in the header or in the explorer's banner."""

    def test_a_long_window_carries_no_candle_cap_notice(self, monkeypatch, tmp_path):
        from datetime import UTC, datetime
        monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(dashboard.yf, "download",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
        start = datetime.now(UTC).date() - timedelta(days=1_000)
        page = dashboard.generate_dashboard(
            [make_trade()], make_equity([1000.0, 1010.0, 1005.0]), start, 1000.0,
            sweep=_Grid.sweep()).read_text(encoding="utf-8")
        assert "This window spans" not in page
        assert "candles per request" not in page
        assert "could never enter" not in page
        # The explorer's banner still opens on its own first line
        assert "band x k cells computed" in page


class TestSpearman:
    """dashboard._spearman against scipy, with ties, and its None cases."""

    @pytest.mark.parametrize("xs, ys", [
        ([1, 2, 3, 4, 5], [5, 6, 7, 8, 7]),
        ([0.1, 0.1, 0.3, -0.2, 0.5, 0.5], [2.0, 1.0, 1.0, 4.0, -3.0, 0.0]),
        (_Grid.H1, _Grid.H2),
    ])
    def test_matches_scipy_with_ties(self, xs, ys):
        from scipy.stats import spearmanr
        assert dashboard._spearman(xs, ys) == pytest.approx(spearmanr(xs, ys).statistic)

    @pytest.mark.parametrize("xs, ys", [
        ([1.0], [2.0]),                         # fewer than two pairs
        ([1.0, 2.0], [1.0]),                    # lengths disagree
        ([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]),     # constant sample
        ([1.0, float("nan"), None], [1.0, 2.0, 3.0]),  # one finite pair left
    ])
    def test_undefined_cases_are_none(self, xs, ys):
        assert dashboard._spearman(xs, ys) is None

    def test_non_finite_pairs_are_dropped_not_ranked(self):
        from scipy.stats import spearmanr
        xs = [1.0, 2.0, float("nan"), 4.0, 5.0]
        ys = [2.0, 1.0, 9.0, 4.0, 3.0]
        kept = ([1.0, 2.0, 4.0, 5.0], [2.0, 1.0, 4.0, 3.0])
        assert dashboard._spearman(xs, ys) == pytest.approx(spearmanr(*kept).statistic)


class TestEquityAxis:
    """One date axis per page, decided from the primary; every curve placed on
    it by date, in cents."""

    @staticmethod
    def _curve(n: int, start: date = date(2019, 12, 31)) -> pd.DataFrame:
        return make_equity([10000.0 + i * 1.2345678901 for i in range(n)], start=start)

    def test_short_curves_keep_every_date(self):
        eq = self._curve(400)
        assert list(dashboard._equity_axis(eq).date) == list(eq["date"])

    def test_long_curves_keep_the_opening_and_real_week_ends(self):
        eq = self._curve(2459)
        axis = dashboard._equity_axis(eq)
        assert len(axis) == 353
        # The DR-03 opening row survives, and the last point is the curve's
        # real last date — never the future Sunday that closes its week.
        assert axis[0].date() == eq["date"].iloc[0]
        assert axis[-1].date() == eq["date"].iloc[-1]
        assert set(axis.date) <= set(eq["date"])
        assert all(d.weekday() == 6 for d in axis.date[1:-1])

    def test_a_longer_cell_is_placed_by_date_on_the_primarys_axis(self):
        # A band sweep that crosses 00:00 UTC hands later cells one more row.
        primary = self._curve(400)
        later = self._curve(401)
        axis = dashboard._equity_axis(primary)
        values = dashboard._curve_on_axis(later, axis)
        assert len(values) == 400
        # In cents, rounded as every shipped curve is (_sparse_on_axis)
        assert values == [float(np.round(v, 2)) for v in later["portfolio_value"].iloc[:400]]

    def test_values_are_cents_and_non_finite_is_none(self):
        eq = make_equity([1000.123456, float("nan"), 1001.5])
        values = dashboard._curve_on_axis(eq, dashboard._equity_axis(eq))
        assert values == [1000.12, None, 1001.5]

    def test_the_curve_is_the_change_points_every_scenario_ships(self):
        # One rounding for the curve Python draws and the change points the
        # script expands (_sparse_on_axis), even a hair above a half cent,
        # where Python's round() and numpy's disagree
        eq = make_equity([10000.0, 18783.345, 9999.995, 10001.005])
        axis = dashboard._equity_axis(eq)
        sparse = dashboard._sparse_on_axis(list(eq["date"]), eq["portfolio_value"], axis, 2)
        assert dashboard._curve_on_axis(eq, axis) == _expand(sparse, len(axis))
        assert dashboard._curve_on_axis(eq, axis)[1] == float(np.round(18783.345, 2))

    def test_missing_curves_and_dates_are_empty_or_none(self):
        axis = dashboard._equity_axis(self._curve(5))
        assert dashboard._curve_on_axis(None, axis) == []
        shorter = self._curve(3)
        assert dashboard._curve_on_axis(shorter, axis)[3:] == [None, None]
        assert len(dashboard._equity_axis(None)) == 0


class TestScenarioExplorerEmptyStates:
    """The section's own placeholders: no sweep, and the two empty-scenario
    causes, each named in the operator's own terms."""

    def test_no_sweep(self):
        out = _section_scenario_explorer(None)
        assert "Scenario Explorer" in out
        assert "No sweep for this run." in out
        for absent in ("strike-blind", "Outcome-label coverage", "Primary spread band"):
            assert absent not in out

    def test_infeasible_window(self):
        pt = _scn_point((0.0, 1.0), 0.75, trades=[], values=[1000.0])
        sweep = BacktestSweep(primary=pt, points=[pt], calibration=None,
                              label_coverage=None, scenarios=[])
        assert ("No scenarios were computed: infeasible window (no trades and no "
                "census).") in _section_scenario_explorer(sweep)

    def test_band_sweep_off(self):
        pt = _scn_point((0.0, 1.0), 0.75)
        sweep = BacktestSweep(primary=pt, points=[pt], calibration=None,
                              label_coverage=_scn_coverage(), scenarios=[])
        out = _section_scenario_explorer(sweep)
        assert ("No scenarios were computed: band sweep off (--no-band-sweep / "
                "band_sweep=False).") in out
        assert "infeasible window" not in out


class TestRunSettingsHeader:
    """The page header names the primary spread band, the ladder setting and
    the per-trade size cap, under the Period line and above every section —
    the first two decide which pairs exist and the cap how big every trade
    is, so they qualify the whole page; whether the size-cap sweep ran says
    why the filter bar's Size cap select offers one cap or many."""

    @staticmethod
    def _page(monkeypatch, tmp_path, **kwargs) -> str:
        monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(dashboard.yf, "download",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
        out_path = dashboard.generate_dashboard(
            [make_trade()], make_equity([1000.0, 1010.0, 1005.0]),
            date(2026, 1, 5), 1000.0, **kwargs)
        return out_path.read_text(encoding="utf-8")

    def test_no_sweep_says_not_recorded(self, monkeypatch, tmp_path):
        page = self._page(monkeypatch, tmp_path)
        assert ("Primary spread band: not recorded | same-event ladders: not recorded"
                " | per-trade cap: not recorded (size-cap sweep not recorded)</p>" in page)
        assert "strike-blind" not in page
        assert "Outcome-label coverage" not in page

    @pytest.mark.parametrize("ladders, word", [(True, "on"), (False, "off"),
                                               (None, "not recorded")])
    def test_each_ladder_state(self, monkeypatch, tmp_path, ladders, word):
        pt = _scn_point((0.3, 0.6), 0.75)
        sweep = BacktestSweep(primary=pt, points=[pt], calibration=None,
                              label_coverage=_scn_coverage(), scenarios=[],
                              same_event_ladders=ladders)
        page = self._page(monkeypatch, tmp_path, sweep=sweep)
        # A hand-built point records no cap; a run with a census but no cap
        # sweep ran with --no-cap-sweep
        line = (f"Primary spread band: max(tier,0.3)-0.6 | same-event ladders: {word}"
                " | per-trade cap: not recorded (size-cap sweep off, --no-cap-sweep)</p>")
        assert line in page
        assert page.index("Period:") < page.index(line) < page.index("Portfolio Performance")

    @pytest.mark.parametrize("cap, cap_sweep, coverage, clause", [
        (0.2, True, True, "per-trade cap: 20% (size-cap sweep on)"),
        (0.2, False, True, "per-trade cap: 20% (size-cap sweep off, --no-cap-sweep)"),
        # No census: this may be the infeasible window, which carries no cap
        # sweep either — so it is not called "off"
        (0.2, False, False, "per-trade cap: 20% (size-cap sweep not recorded)"),
        (1.0, False, True, "per-trade cap: off (full Kelly) (size-cap sweep off, "
                           "--no-cap-sweep)"),
        (None, True, True, "per-trade cap: not recorded (size-cap sweep on)"),
    ])
    def test_the_cap_clause(self, cap, cap_sweep, coverage, clause):
        pt = dataclasses.replace(_scn_point((0.3, 0.6), 0.75), size_cap=cap)
        sweep = BacktestSweep(primary=pt, points=[pt], calibration=None,
                              label_coverage=_scn_coverage() if coverage else None,
                              cap_sweep=object() if cap_sweep else None,
                              same_event_ladders=True)
        line = dashboard._run_settings_html(sweep)
        assert line.endswith(f"same-event ladders: on | {clause}</p>")

    def test_a_cap_sweep_the_bar_could_not_use_says_so(self):
        # The run carried a size-cap sweep, but the bar fell back to the
        # run's own cap: the one-option Size cap select's reason is on the
        # page, not only in the log (DR-66)
        pt = dataclasses.replace(_scn_point((0.3, 0.6), 0.75), size_cap=0.2)
        sweep = BacktestSweep(primary=pt, points=[pt], calibration=None,
                              label_coverage=_scn_coverage(), cap_sweep=object(),
                              same_event_ladders=True)
        line = dashboard._run_settings_html(sweep, cap_sweep_unused=True)
        assert line.endswith(
            "per-trade cap: 20% (size-cap sweep on, but it could not be used — the filter "
            "bar offers the run's own cap only; the log names why)</p>")
        # Without a size-cap sweep there is nothing it could not use
        off = dataclasses.replace(sweep, cap_sweep=None)
        assert dashboard._run_settings_html(off, cap_sweep_unused=True) \
            == dashboard._run_settings_html(off)

    @pytest.mark.parametrize("ladders, configured, tail", [
        (True, True, "on (same as this checkout's config)"),
        (False, False, "off (same as this checkout's config)"),
        (False, True, "off (departs from this checkout's config.TIME_SERIES_SAME_EVENT_LADDERS, "
                      "which was on: not a replay of its live rule)"),
        (True, False, "on (departs from this checkout's config.TIME_SERIES_SAME_EVENT_LADDERS, "
                      "which was off: not a replay of its live rule)"),
    ])
    def test_ladder_reading_is_judged_against_the_configured_switch(
        self, monkeypatch, tmp_path, ladders, configured, tail,
    ):
        # A run's resolved setting is read beside the switch this checkout's
        # config actually carries — agreeing renders one clause, departing
        # renders another, naming the CONFIGURED value so a reader knows what
        # the live finder would have traded.
        pt = _scn_point((0.3, 0.6), 0.75)
        sweep = BacktestSweep(primary=pt, points=[pt], calibration=None,
                              label_coverage=_scn_coverage(), scenarios=[],
                              same_event_ladders=ladders,
                              config_same_event_ladders=configured)
        page = self._page(monkeypatch, tmp_path, sweep=sweep)
        # The ladder reading, then the per-trade cap clause the line goes on with
        assert f"| same-event ladders: {tail} | per-trade cap:" in page

    def test_an_unrecorded_setting_stays_not_recorded_whatever_the_config(
        self, monkeypatch, tmp_path,
    ):
        # same_event_ladders itself unrecorded still reads "not recorded",
        # whatever config_same_event_ladders happens to carry — there is
        # nothing to compare it against.
        pt = _scn_point((0.3, 0.6), 0.75)
        sweep = BacktestSweep(primary=pt, points=[pt], calibration=None,
                              label_coverage=_scn_coverage(), scenarios=[],
                              same_event_ladders=None,
                              config_same_event_ladders=True)
        page = self._page(monkeypatch, tmp_path, sweep=sweep)
        assert "| same-event ladders: not recorded | per-trade cap:" in page


class TestLiveRuleHeader:
    """dashboard._live_rule_html(sweep, bar=...): the saved live defaults' rule,
    as run_backtest_sweep read it before its fetch, and where this page shows it
    (backtester._live_rule_view's verdict, in the filter bar's option texts),
    on a <p> of its own: _run_settings_html's exact "</p>" suffix is pinned.
    When the saved k or caps are not the run's own, it says so and how the
    filter bar shows them."""

    _P = '<p style="color:#616161; font-size:14px;">'
    _CAP = "same-title trades capped at the size cap shown, like every pair (20% at this run's own cap)"

    @staticmethod
    def _sweep(*, live_tier_floors=True, live_spread_band=(0.0, 1.0),
               primary_band=(0.0, 1.0), size_cap=0.2, same_title_size_cap=1.0,
               calibrations_by_band=None, tier_off_calibrations_by_band=None,
               tier_off_scenarios=None, live_categories=None, live_tags=None,
               same_event_ladders=None, config_same_event_ladders=None,
               live_sizing=(None, None, None)) -> BacktestSweep:
        """A sweep at k 0.75 recording these live_* fields; live_sizing is the
        saved defaults' (k, per-trade cap, same-title cap), unrecorded by default."""
        pt = dataclasses.replace(_scn_point(primary_band, 0.75), size_cap=size_cap)
        live_k, live_cap, live_st = live_sizing
        return BacktestSweep(
            primary=pt, points=[pt], calibration=None,
            calibrations_by_band=(calibrations_by_band if calibrations_by_band is not None
                                  else {primary_band: None}),
            tier_off_calibrations_by_band=tier_off_calibrations_by_band or {},
            tier_off_scenarios=tier_off_scenarios or [],
            same_title_size_cap=same_title_size_cap,
            same_event_ladders=same_event_ladders,
            config_same_event_ladders=config_same_event_ladders,
            live_tier_floors=live_tier_floors, live_spread_band=live_spread_band,
            live_categories=live_categories, live_tags=live_tags,
            live_interval_discount=live_k, live_size_cap=live_cap,
            live_same_title_size_cap=live_st,
        )

    @staticmethod
    def _bar(bands=((0.0, 1.0),), primary=(0.0, 1.0), off=True, categories=(),
             subcats=(), ks=(0.75,), caps=(0.2,), off_caps=None) -> dict:
        """The base-block keys _live_rule_html reads, labelled as _filter_payload labels
        them; the primary k and cap are the first of each. Every tier-on cell holds a
        chunk; with the tiers off only the caps in off_caps do (every cap when None),
        as on a page that could not read a tier-floors-off size-cap sweep."""
        def mark(b):
            return " (primary)" if b == primary else ""

        def grid(cap_held):
            """
            A [band][k][cap] grid of chunk ids, shaped as _filter_payload ships one.

            Args:
                cap_held (Callable[[float], bool]): Whether a cell at this cap holds
                    a chunk; a cell that does not is None, as the bar ships one the
                    run never simulated.

            Returns:
                list: The grid.
            """
            return [[[f"chunk-{bi}-{ki}-{ci}" if cap_held(c) else None
                      for ci, c in enumerate(caps)] for ki in range(len(ks))]
                    for bi in range(len(bands))]

        return {
            "grid": grid(lambda c: True),
            "grid_off": None if not off else grid(
                lambda c: off_caps is None or c in off_caps),
            "ks": [{"label": dashboard._k_option(k), "value": k} for k in ks],
            "caps": [{"label": dashboard._cap_option(c), "value": c} for c in caps],
            "primary": [list(bands).index(primary) if primary in bands else 0, 0, 0],
            "bands": [{"label": dashboard._band_option(b),
                       "option": dashboard._band_option(b) + mark(b)} for b in bands],
            "bands_off": None if not off else [
                {"label": dashboard._band_label(b), "option": dashboard._band_label(b) + mark(b)}
                for b in bands],
            "categories": list(categories),
            "subcats": [[list(categories).index(c), t] for c, t in subcats],
        }

    def _text(self, sweep, bar) -> str:
        out = dashboard._live_rule_html(sweep, bar=bar)
        assert out.startswith(self._P) and out.endswith("</p>")
        return out[len(self._P):-len("</p>")]

    # ── Not recorded ────────────────────────────────────────────────────────

    def test_no_sweep_is_not_recorded(self):
        assert dashboard._live_rule_html(None, bar=None) == (
            self._P + "Live rule: not recorded</p>")

    @pytest.mark.parametrize("live_tier_floors, live_spread_band",
                             [(None, (0.0, 1.0)), (True, None), (None, None)])
    def test_either_unrecorded_field_is_none_recorded(self, live_tier_floors, live_spread_band):
        sweep = self._sweep(live_tier_floors=live_tier_floors,
                            live_spread_band=live_spread_band)
        assert dashboard._live_rule_html(sweep, bar=self._bar()) == (
            self._P + "Live rule: none recorded — no usable live defaults were saved when "
            "this run started, and live runs refuse to start without them</p>")

    # ── The primary scenario ────────────────────────────────────────────────

    @pytest.mark.parametrize("bar", [True, False])
    def test_the_primary_scenario_is_the_rule(self, bar):
        rule = config.describe_time_series_rule(True, (0.0, 1.0))
        # The page as rendered IS the primary scenario: no bar needed
        assert self._text(self._sweep(), self._bar() if bar else None) == (
            f"Live rule (saved live defaults): {rule}; {self._CAP} — this run's primary")

    def test_the_primary_is_matched_by_its_own_band(self):
        rule = config.describe_time_series_rule(True, (0.3, 0.6))
        sweep = self._sweep(live_spread_band=(0.3, 0.6), primary_band=(0.3, 0.6))
        assert self._text(sweep, self._bar(bands=[(0.3, 0.6)], primary=(0.3, 0.6))) == (
            f"Live rule (saved live defaults): {rule}; {self._CAP} — this run's primary")
        # ... so a live band equal to the DEFAULT band is elsewhere on it
        rule = config.describe_time_series_rule(True, (0.0, 1.0))
        sweep = self._sweep(live_spread_band=(0.0, 1.0), primary_band=(0.3, 0.6),
                            calibrations_by_band={(0.0, 1.0): None, (0.3, 0.6): None})
        bar = self._bar(bands=[(0.0, 1.0), (0.3, 0.6)], primary=(0.3, 0.6))
        assert self._text(sweep, bar) == (
            f"Live rule (saved live defaults): {rule}; {self._CAP} — choose Spread band "
            "max(tier,0)-1 and Tier floors on in the filter bar")

    # ── Elsewhere on the grid, named as the bar names it ────────────────────

    def test_a_tier_on_band_elsewhere_names_the_bars_option(self):
        rule = config.describe_time_series_rule(True, (0.3, 0.6))
        sweep = self._sweep(live_spread_band=(0.3, 0.6),
                            calibrations_by_band={(0.0, 1.0): None, (0.3, 0.6): None})
        bar = self._bar(bands=[(0.0, 1.0), (0.3, 0.6)])
        assert self._text(sweep, bar) == (
            f"Live rule (saved live defaults): {rule}; {self._CAP} — choose Spread band "
            "max(tier,0.3)-0.6 and Tier floors on in the filter bar")
        assert "max(tier,0.3)-0.6" in [e["option"] for e in bar["bands"]]

    def test_a_tier_off_band_is_chosen_after_the_tier_floors(self):
        # Tiers off relabel the bar's bands: "off" first, then the bare band label
        band = (0.0, 0.5)
        rule = config.describe_time_series_rule(False, band)
        sweep = self._sweep(live_tier_floors=False, live_spread_band=band,
                            calibrations_by_band={(0.0, 1.0): None, band: None},
                            tier_off_calibrations_by_band={(0.0, 1.0): None, band: None},
                            tier_off_scenarios=[object()])
        bar = self._bar(bands=[(0.0, 0.5), (0.0, 1.0)])
        assert self._text(sweep, bar) == (
            f"Live rule (saved live defaults): {rule}; {self._CAP} — choose Tier floors off and "
            "Spread band 0-0.5 in the filter bar")
        # ... and at the primary band, its tier-off option carries the mark
        sweep = self._sweep(live_tier_floors=False, live_spread_band=(0.0, 1.0),
                            calibrations_by_band={(0.0, 1.0): None},
                            tier_off_calibrations_by_band={(0.0, 1.0): None},
                            tier_off_scenarios=[object()])
        assert self._text(sweep, self._bar()).endswith(
            "— choose Tier floors off and Spread band 0-1 (primary) in the filter bar")

    def test_off_the_grid_is_not_simulated(self):
        sweep = self._sweep(live_spread_band=(0.3, 0.6),
                            calibrations_by_band={(0.0, 1.0): None})
        assert self._text(sweep, self._bar()).endswith("— not simulated by this run")

    def test_tier_floors_off_needs_the_whole_family(self):
        band = (0.0, 1.0)
        assert backtester._tier_floors_bind(band) is True
        grid = {band: None, (0.2, 0.6): None}
        # On the grid, but no tier-off family ran at all
        sweep = self._sweep(live_tier_floors=False, calibrations_by_band=grid)
        assert self._text(sweep, self._bar()).endswith("— not simulated by this run")
        # The family lacks ANOTHER binding band: the page withholds the whole
        # off view (dashboard._tier_off_binds)
        assert backtester._tier_floors_bind((0.2, 0.6)) is True
        sweep = self._sweep(live_tier_floors=False, calibrations_by_band=grid,
                            tier_off_calibrations_by_band={band: None},
                            tier_off_scenarios=[object()])
        assert dashboard._tier_off_binds(sweep, list(grid)) is None
        assert self._text(sweep, self._bar()).endswith("— not simulated by this run")
        # The family names every binding band: reachable
        sweep = self._sweep(live_tier_floors=False, calibrations_by_band=grid,
                            tier_off_calibrations_by_band=dict(grid),
                            tier_off_scenarios=[object()])
        assert self._text(sweep, self._bar(bands=[band, (0.2, 0.6)])).endswith(
            "— choose Tier floors off and Spread band 0-1 (primary) in the filter bar")

    def test_tier_floors_off_where_no_tier_binds_is_the_tier_on_rule(self):
        band = (0.35, 0.5)
        assert backtester._tier_floors_bind(band) is False
        note = " (no tier floor binds at this band, so off and on are one rule)"
        # The primary at that band IS it
        sweep = self._sweep(live_tier_floors=False, live_spread_band=band,
                            primary_band=band)
        assert self._text(sweep, self._bar(bands=[band], primary=band)).endswith(
            f"— this run's primary{note}")
        # Elsewhere, its tier-on cell holds it, with or without the family
        for family in ([], [object()]):
            sweep = self._sweep(live_tier_floors=False, live_spread_band=band,
                                calibrations_by_band={(0.0, 1.0): None, band: None},
                                tier_off_scenarios=family)
            assert self._text(sweep, self._bar(bands=[(0.0, 1.0), band], off=False)).endswith(
                f"— choose Spread band max(tier,0.35)-0.5 and Tier floors on{note} in the "
                "filter bar")

    # ── What this page can show ─────────────────────────────────────────────

    def test_without_a_bar_a_grid_cell_is_not_shown(self):
        sweep = self._sweep(live_spread_band=(0.3, 0.6),
                            calibrations_by_band={(0.0, 1.0): None, (0.3, 0.6): None})
        assert self._text(sweep, None).endswith(
            "— not shown: this page's filter bar could not be built")

    def test_a_bar_without_the_option_does_not_show_it(self):
        band = (0.0, 0.5)
        sweep = self._sweep(live_tier_floors=False, live_spread_band=band,
                            calibrations_by_band={(0.0, 1.0): None, band: None},
                            tier_off_calibrations_by_band={(0.0, 1.0): None, band: None},
                            tier_off_scenarios=[object()])
        # No tier-off view on the page (bands_off null), then no such band
        for bar in (self._bar(bands=[band, (0.0, 1.0)], off=False), self._bar()):
            assert self._text(sweep, bar).endswith(
                "— not shown: this page's filter bar does not offer that scenario")

    # ── The same-title cap at every cap the page offers ─────────────────────

    @pytest.mark.parametrize("size_cap, st_cap, clause", [
        (0.35, 1.0, "same-title trades capped at the size cap shown, like every pair "
                    "(35% at this run's own cap)"),
        (0.35, 0.10, "same-title trades capped at the lower of the size cap shown and 10% "
                     "(10% at this run's own cap)"),
        (1.0, 0.20, "same-title trades capped at the lower of the size cap shown and 20% "
                    "(20% at this run's own cap)"),
        (None, 1.0, "same-title cap not recorded"),
        (0.2, None, "same-title cap not recorded"),
    ])
    def test_the_same_title_cap_clause(self, size_cap, st_cap, clause):
        sweep = self._sweep(size_cap=size_cap, same_title_size_cap=st_cap)
        assert f"; {clause} — this run's primary" in self._text(sweep, self._bar())

    # ── The saved defaults' own sizing ──────────────────────────────────────

    # The run's own sizing, as _sweep records it: k 0.75, a 20% cap, no extra
    # same-title cap
    _RUN_SIZING = "k 0.75, per-trade cap 20% and same-title cap 100% (no extra cap)"

    def test_no_sizing_note_when_the_saved_sizing_is_the_runs(self):
        for live_sizing in ((0.75, 0.2, 1.0), (None, None, None), (0.8, None, 1.0)):
            text = self._text(self._sweep(live_sizing=live_sizing),
                              self._bar(ks=(0.75, 0.8)))
            assert text.endswith("— this run's primary"), (live_sizing, text)

    def test_a_bar_offering_the_saved_sizing_names_its_options(self):
        sweep = self._sweep(live_sizing=(0.8, 0.1, 1.0))
        bar = self._bar(ks=(0.75, 0.8), caps=(0.2, 0.1))
        assert self._text(sweep, bar).endswith(
            "— this run's primary; the live defaults size at k 0.8, per-trade cap 10% and "
            f"same-title cap 100% (no extra cap), where this run's primary sized at "
            f"{self._RUN_SIZING}; choose {dashboard._k_option(0.8)} and Size cap 10% in the "
            "filter bar to see it")

    def test_an_offered_primary_option_is_named_as_the_bar_shows_it(self):
        # The saved k is the run's own, the cap is not: the k option is the primary's
        sweep = self._sweep(live_sizing=(0.75, 0.1, 1.0))
        text = self._text(sweep, self._bar(ks=(0.75, 0.8), caps=(0.2, 0.1)))
        assert text.endswith(f"; choose {dashboard._k_option(0.75)} (primary) and Size cap "
                             "10% in the filter bar to see it")

    @pytest.mark.parametrize("ks, caps, live_st, clause", [
        ((0.75,), (0.2, 0.1), 1.0,
         "this page's filter bar does not offer k = 0.80"),
        ((0.75, 0.8), (0.2,), 1.0,
         "this page's filter bar does not offer size cap 10%"),
        ((0.75,), (0.2,), 1.0,
         "this page's filter bar does not offer k = 0.80 or size cap 10%"),
        ((0.75, 0.8), (0.2, 0.1), 0.2,
         "the same-title cap is not a filter-bar choice"),
        ((0.75,), (0.2, 0.1), 0.2,
         "this page's filter bar does not offer k = 0.80, and the same-title cap is not "
         "a filter-bar choice"),
    ])
    def test_what_the_bar_cannot_show_is_named(self, ks, caps, live_st, clause):
        assert dashboard._k_option(0.8) == "k = 0.80"
        sweep = self._sweep(live_sizing=(0.8, 0.1, live_st))
        text = self._text(sweep, self._bar(ks=ks, caps=caps))
        assert text.endswith(f"where this run's primary sized at {self._RUN_SIZING}; "
                             f"{clause}"), text

    def test_only_the_same_title_cap_differing_is_named(self):
        # k and the per-trade cap are the run's own: the bar shows those already
        sweep = self._sweep(live_sizing=(0.75, 0.2, 0.2))
        assert self._text(sweep, self._bar()).endswith(
            "; the live defaults size at k 0.75, per-trade cap 20% and same-title cap 20%, "
            f"where this run's primary sized at {self._RUN_SIZING}; the same-title cap is "
            "not a filter-bar choice")

    def test_without_a_bar_the_saved_sizing_is_not_shown(self):
        sweep = self._sweep(live_sizing=(0.8, 0.1, 1.0))
        assert self._text(sweep, None).endswith(
            f"where this run's primary sized at {self._RUN_SIZING}; this page's filter bar "
            "could not be built, so it does not show that sizing")

    # The note as _live_sizing_note words a seed-like saved sizing against _sweep's run
    _NOTE = ("; the live defaults size at k 0.8, per-trade cap 10% and same-title cap "
             "100% (no extra cap), where this run's primary sized at k 0.75, per-trade cap "
             "20% and same-title cap 100% (no extra cap)")

    def test_the_note_follows_every_verdict(self):
        # Not simulated, and a grid cell elsewhere: the note closes the line either way
        assert self._NOTE.endswith(self._RUN_SIZING)
        sweep = self._sweep(live_spread_band=(0.3, 0.6), live_sizing=(0.8, 0.1, 1.0),
                            calibrations_by_band={(0.0, 1.0): None})
        assert self._text(sweep, self._bar()).endswith(
            f"— not simulated by this run{self._NOTE}")
        sweep = self._sweep(live_spread_band=(0.3, 0.6), live_sizing=(0.8, 0.1, 1.0),
                            calibrations_by_band={(0.0, 1.0): None, (0.3, 0.6): None})
        bar = self._bar(bands=[(0.0, 1.0), (0.3, 0.6)], ks=(0.75, 0.8), caps=(0.2, 0.1))
        assert self._text(sweep, bar).endswith(
            f"Tier floors on in the filter bar{self._NOTE}; choose k = 0.80 and Size cap 10% "
            "in the filter bar to see it")

    def test_a_page_that_does_not_show_the_rule_never_offers_its_sizing(self):
        # A bar that offers the saved k and cap still cannot show a rule the run never
        # simulated, or one at a band the bar does not offer: no "choose ..." clause
        bar = self._bar(ks=(0.75, 0.8), caps=(0.2, 0.1))
        not_simulated = self._sweep(live_spread_band=(0.3, 0.6), live_sizing=(0.8, 0.1, 1.0),
                                    calibrations_by_band={(0.0, 1.0): None})
        assert self._text(not_simulated, bar).endswith(
            f"— not simulated by this run{self._NOTE}")
        not_offered = self._sweep(live_spread_band=(0.3, 0.6), live_sizing=(0.8, 0.1, 1.0),
                                  calibrations_by_band={(0.0, 1.0): None, (0.3, 0.6): None})
        assert self._text(not_offered, bar).endswith(
            "— not shown: this page's filter bar does not offer that scenario"
            f"{self._NOTE}")
        not_built = self._sweep(live_spread_band=(0.3, 0.6), live_sizing=(0.8, 0.1, 1.0),
                                calibrations_by_band={(0.0, 1.0): None, (0.3, 0.6): None})
        assert self._text(not_built, None).endswith(
            f"— not shown: this page's filter bar could not be built{self._NOTE}")

    def test_a_tier_off_cell_the_page_does_not_hold_is_named(self):
        # Tiers off at a binding band, on a page holding tier-off cells at its own cap
        # only (no tier-floors-off size-cap sweep to read): the saved 10% cap is on the
        # bar, but its tier-off view there is the "missing" template
        band = (0.0, 0.5)
        sweep = self._sweep(live_tier_floors=False, live_spread_band=band,
                            live_sizing=(0.8, 0.1, 1.0),
                            calibrations_by_band={(0.0, 1.0): None, band: None},
                            tier_off_calibrations_by_band={(0.0, 1.0): None, band: None},
                            tier_off_scenarios=[object()])
        bar = self._bar(bands=[band, (0.0, 1.0)], ks=(0.75, 0.8), caps=(0.2, 0.1),
                        off_caps=(0.2,))
        assert self._text(sweep, bar).endswith(
            f"Spread band 0-0.5 in the filter bar{self._NOTE}; this page's filter bar offers "
            "k = 0.80 and Size cap 10% but holds no scenario of the live rule there")
        # ... and with a tier-off cell at every cap, it is chosen like any other
        bar = self._bar(bands=[band, (0.0, 1.0)], ks=(0.75, 0.8), caps=(0.2, 0.1))
        assert self._text(sweep, bar).endswith(
            f"Spread band 0-0.5 in the filter bar{self._NOTE}; choose k = 0.80 and Size cap "
            "10% in the filter bar to see it")

    # ── The category/tag filter ─────────────────────────────────────────────

    _CATS = ("Economics", "Oil & Gas", "Sports")
    _SUBCATS = (("Economics", "Fed"), ("Economics", "Inflation"), ("Oil & Gas", "Fed"),
                ("Sports", "Basketball"))

    def _filtered(self, categories, tags, *, bar=True, **kw) -> str:
        sweep = self._sweep(live_categories=categories, live_tags=tags, **kw)
        return self._text(sweep, self._bar(categories=self._CATS, subcats=self._SUBCATS)
                          if bar else None)

    @pytest.mark.parametrize("categories, tags, tail", [
        (("economics",), None, "this run's primary, with Category Economics ticked in the "
                               "filter bar"),
        # A category narrowed to one of its two tags: tick the tag as well
        (("Economics",), ("fed",), "this run's primary, with Category Economics and Tag "
                                   "Economics · Fed ticked in the filter bar"),
        # A plain tag under two categories: both are ticked, and only the category the
        # tag narrows needs the tag ticked (Oil & Gas has no other tag)
        (None, ("Fed",), "this run's primary, with Category Economics, Oil &amp; Gas and "
                         "Tag Economics · Fed ticked in the filter bar"),
        # Several categories at once are shown together
        (("Economics", "Sports"), None, "this run's primary, with Category Economics, "
                                        "Sports ticked in the filter bar"),
        # A tied tag narrows its own category; the others count in full
        (("Economics", "Oil & Gas"), ("Economics · Inflation",),
         "this run's primary, with Category Economics, Oil &amp; Gas and Tag Economics · "
         "Inflation ticked in the filter bar"),
        (("Politics",), None, "this run's primary; no scenario of this run has a pair filed "
                              "under the live category/tag filter"),
        # A tag no pair of the run carries under that category
        (("Economics",), ("Basketball",), "this run's primary; no scenario of this run has a "
                                          "pair filed under the live category/tag filter"),
    ])
    def test_a_filter_names_what_to_tick(self, categories, tags, tail):
        text = self._filtered(categories, tags)
        words = backtester._live_filter_text(categories, tags)
        rule = config.describe_time_series_rule(True, (0.0, 1.0))
        assert text == (f"Live rule (saved live defaults): {rule}; category/tag filter "
                        f"({html.escape(words, quote=False)}); {self._CAP} — {tail}")
        assert "never their union" not in text and "one of them at a time" not in text

    _TICK_CATS = ("Economics", "Sports")
    _TICK_SUBCATS = (("Economics", "Fed"), ("Sports", "Basketball"), ("Sports", "Soccer"))

    @pytest.mark.parametrize("categories, tags, ticked", [
        # The example of the multi-choice filter: one category whole, one narrowed
        (("Economics", "Sports"), ("Sports · Basketball",),
         (["Economics", "Sports"], ["Sports · Basketball"])),
        # Letter case never matters, tied halves included
        (("economics", "SPORTS"), ("sports · BASKETBALL",),
         (["Economics", "Sports"], ["Sports · Basketball"])),
        # Every tag of a category ticked is the category alone
        (("Sports",), ("Sports · Basketball", "Sports · Soccer"), (["Sports"], [])),
        # A plain tag is looked for under each listed category
        (("Economics", "Sports"), ("Soccer",), (["Sports"], ["Sports · Soccer"])),
        (None, ("Soccer",), (["Sports"], ["Sports · Soccer"])),
        # No category listed and no tag kept: nothing to tick
        (None, ("Nope",), ([], [])),
        # A category the bar does not offer is left out
        (("Economics", "Politics"), None, (["Economics"], [])),
    ])
    def test_the_ticks_follow_the_one_live_rule(self, categories, tags, ticked):
        bar = self._bar(categories=self._TICK_CATS, subcats=self._TICK_SUBCATS)
        assert dashboard._live_filter_ticks(categories, tags, bar) == ticked

    def test_a_tied_tag_in_the_header_names_the_tag_to_tick(self):
        sweep = self._sweep(live_categories=("Economics", "Sports"),
                            live_tags=("Sports · Basketball",))
        text = self._text(sweep, self._bar(categories=self._TICK_CATS,
                                           subcats=self._TICK_SUBCATS))
        assert text.endswith("— this run's primary, with Category Economics, Sports and Tag "
                             "Sports · Basketball ticked in the filter bar")

    def test_the_ticks_agree_with_the_rule_on_every_bar_pair(self):
        # Whatever the ticks say, the pairs they show are exactly the pairs the live
        # rule keeps (the page counts a ticked category in full unless a tag of it is ticked)
        bar = self._bar(categories=self._TICK_CATS, subcats=self._TICK_SUBCATS)
        for categories, tags in [(("Economics", "Sports"), ("Sports · Basketball",)),
                                 (None, ("Soccer", "Fed")), (("Sports",), None)]:
            keeps = config.trade_filter_for(categories, tags)
            cats, ticked_tags = dashboard._live_filter_ticks(categories, tags, bar)
            for ci, tag in bar["subcats"]:
                name = bar["categories"][ci]
                own = [t for c, t in map(config.split_tag, ticked_tags) if c == name]
                shown = name in cats and (not own or tag in own)
                assert shown == keeps(name, tag)

    def test_a_filter_on_a_grid_cell_follows_its_band_choice(self):
        sweep = self._sweep(live_spread_band=(0.3, 0.6), live_categories=("Sports",),
                            calibrations_by_band={(0.0, 1.0): None, (0.3, 0.6): None})
        bar = self._bar(bands=[(0.0, 1.0), (0.3, 0.6)], categories=self._CATS,
                        subcats=self._SUBCATS)
        assert self._text(sweep, bar).endswith(
            "— choose Spread band max(tier,0.3)-0.6 and Tier floors on in the filter bar, "
            "and tick Category Sports there")
        # A bar that does not offer the band names no slice of it either
        bar = self._bar(categories=self._CATS, subcats=self._SUBCATS)
        assert self._text(sweep, bar).endswith(
            "— not shown: this page's filter bar does not offer that scenario")

    def test_without_a_bar_the_filters_slice_is_not_shown(self):
        assert self._filtered(("Economics",), None, bar=False).endswith(
            "— this run's primary; this page's filter bar could not be built, so the live "
            "category/tag filter cannot be shown")

    def test_no_filter_says_nothing_about_one(self):
        assert "categor" not in self._text(self._sweep(), self._bar(categories=self._CATS))

    def test_kalshi_names_are_escaped_and_apostrophes_kept(self):
        text = self._filtered(("Oil & Gas",), None)
        assert "Oil &amp; Gas" in text and "Oil & Gas" not in text
        assert "this run's primary" in text

    # ── The ladder switch ───────────────────────────────────────────────────

    def test_a_ladder_departure_is_named(self):
        sweep = self._sweep(same_event_ladders=False, config_same_event_ladders=True)
        assert self._text(sweep, self._bar()).endswith(
            "— this run's primary; this run's same-event ladders are off and config.py's "
            "on, so its pairs are not the live bot's")
        for run, configured in ((True, True), (None, True), (False, None)):
            sweep = self._sweep(same_event_ladders=run, config_same_event_ladders=configured)
            assert self._text(sweep, self._bar()).endswith("— this run's primary")

    # ── On the page ─────────────────────────────────────────────────────────

    def test_it_renders_right_after_run_settings_on_the_page(self, monkeypatch, tmp_path):
        pt = dataclasses.replace(_scn_point((0.0, 1.0), 0.75), size_cap=0.2)
        sweep = BacktestSweep(primary=pt, points=[pt], calibration=None,
                              label_coverage=_scn_coverage(),
                              live_tier_floors=True, live_spread_band=(0.0, 1.0),
                              same_title_size_cap=1.0)
        page = TestRunSettingsHeader._page(monkeypatch, tmp_path, sweep=sweep)
        run_settings = dashboard._run_settings_html(sweep)
        # The primary needs no bar, so the line is the same whatever the page built
        live_rule = dashboard._live_rule_html(sweep, bar=None)
        assert live_rule.endswith("— this run's primary</p>")
        assert run_settings in page
        assert live_rule in page
        assert (page.index("Period:") < page.index(run_settings)
                < page.index(live_rule) < page.index("Portfolio Performance"))

    def test_the_page_names_its_own_bars_option(self, monkeypatch, tmp_path):
        # A band on the grid: the line names the Spread band option the bar offers
        band = (0.3, 0.6)
        points = [dataclasses.replace(_scn_point(b, 0.75), size_cap=0.2)
                  for b in ((0.0, 1.0), band)]
        sweep = BacktestSweep(primary=points[0], points=[points[0]], calibration=None,
                              label_coverage=_scn_coverage(), scenarios=points,
                              calibrations_by_band={(0.0, 1.0): None, band: None},
                              live_tier_floors=True, live_spread_band=band,
                              same_title_size_cap=1.0)
        page = TestRunSettingsHeader._page(monkeypatch, tmp_path, sweep=sweep)
        assert ("— choose Spread band max(tier,0.3)-0.6 and Tier floors on in the filter "
                "bar</p>") in page
        assert re.search(r'<select id="flt-band"[^>]*>.*?<option value="\d+">'
                         r'max\(tier,0\.3\)-0\.6</option>', page)


class TestEntryCheckpointHeader:
    """The page header names the entry checkpoint (the live scheduler's run
    time, BacktestSweep.entry_checkpoint) under the run-settings line, or
    says it was not recorded."""

    _LABEL = "Monday 09:00 America/Los_Angeles"
    _RECORDED = (f"Entry checkpoint: {_LABEL} — the live scheduler's run "
                 "time</p>")
    _NOT_RECORDED = "Entry checkpoint: not recorded</p>"

    def test_a_recorded_checkpoint_is_named_under_the_run_settings(
        self, monkeypatch, tmp_path,
    ):
        # A below-floor census renders the strike-blind notice too, so the
        # regex below pins the line's place among all the header lines
        pt = _scn_point((0.3, 0.6), 0.75)
        sweep = BacktestSweep(primary=pt, points=[pt], calibration=None,
                              label_coverage=_coverage(0), scenarios=[],
                              entry_checkpoint=self._LABEL)
        page = TestRunSettingsHeader._page(monkeypatch, tmp_path, sweep=sweep)
        assert self._RECORDED in page
        assert self._NOT_RECORDED not in page
        assert (page.index("Period:") < page.index("Primary spread band:")
                < page.index("Entry checkpoint:") < page.index("Portfolio Performance"))
        assert re.search(
            r"\(size-cap sweep [^<]*\)</p>\n"
            r"<p [^>]*>Live rule: none recorded[^\n]*</p>\n"
            rf'<p style="color:#616161; font-size:14px;">{re.escape(self._RECORDED)}\n'
            r"<p [^>]*>Fills: [^\n]*</p>\n"
            r"<p [^>]*>Risk-free rate[^\n]*</p>\n"
            r"<p [^>]*>Outcome-label coverage for this run is below the floor",
            page)

    def test_no_sweep_says_not_recorded(self, monkeypatch, tmp_path):
        page = TestRunSettingsHeader._page(monkeypatch, tmp_path)
        assert self._NOT_RECORDED in page

    def test_a_sweep_without_one_says_not_recorded(self, monkeypatch, tmp_path):
        pt = _scn_point((0.3, 0.6), 0.75)
        sweep = BacktestSweep(primary=pt, points=[pt], calibration=None)
        page = TestRunSettingsHeader._page(monkeypatch, tmp_path, sweep=sweep)
        assert self._NOT_RECORDED in page
        assert "the live scheduler&#x27;s run time" not in page
        assert "the live scheduler's run time" not in page

    @pytest.mark.parametrize("value", [None, "", MagicMock(), 7])
    def test_anything_but_a_label_reads_not_recorded(self, value):
        pt = _scn_point((0.3, 0.6), 0.75)
        sweep = BacktestSweep(primary=pt, points=[pt], calibration=None,
                              entry_checkpoint=value)
        assert dashboard._entry_checkpoint_html(sweep).endswith(self._NOT_RECORDED)

    def test_the_label_is_escaped(self):
        pt = _scn_point((0.3, 0.6), 0.75)
        sweep = BacktestSweep(primary=pt, points=[pt], calibration=None,
                              entry_checkpoint="Monday 09:00 <Zone>")
        line = dashboard._entry_checkpoint_html(sweep)
        assert "Entry checkpoint: Monday 09:00 &lt;Zone&gt; — the live" in line
        assert "<Zone>" not in line

    def test_a_run_s_sweep_reaches_the_page(self, monkeypatch, tmp_path):
        # The label run_backtest_sweep records is the one the header prints
        monkeypatch.setattr(backtester, "_prepare_candidates", lambda *a, **k: None)
        sweep = backtester.run_backtest_sweep(MagicMock(), MagicMock(), date(2026, 1, 1),
                                              1000.0, sweep=False)
        page = TestRunSettingsHeader._page(monkeypatch, tmp_path, sweep=sweep)
        assert (f"Entry checkpoint: {backtester.SCHEDULED_RUN.label()} — the live "
                "scheduler's run time</p>") in page


class TestFillsHeader:
    """The header's Fills line says how many of the primary scenario's trades
    walked a modeled order book and how many filled at the top of the book,
    which snapshots the depth came from, or, with no depth model, that every
    trade filled at the top of the book."""

    _NO_MODEL = ("Fills: no usable depth snapshot — every trade fills at the top of the "
                 "book, in any size (see the log; run python3 -m kalshi_betting.depth_model "
                 "snapshot)</p>")

    @staticmethod
    def _model(snapshots: int = 3, first: str = "2026-10-02T21:30:00Z",
               last: str = "2026-10-09T21:30:00Z") -> DepthModel:
        return DepthModel(cells={}, volume_rows={}, overall=(0.0,) * 7, snapshots=snapshots,
                          ladders=90, first_taken=first, last_taken=last, digest="test")

    @staticmethod
    def _sweep(walked: int, top: int, model: DepthModel | None) -> BacktestSweep:
        trades = ([dataclasses.replace(make_trade(), book_walked=True,
                                       fill_price_a=0.31, fill_price_b=0.41)] * walked
                  + [make_trade()] * top)
        pt = _scn_point((0.3, 0.6), 0.75, trades=trades)
        return BacktestSweep(primary=pt, points=[pt], calibration=None, depth_model=model)

    def test_a_walked_run_counts_walked_and_top_of_book_trades(self):
        line = dashboard._fills_html(self._sweep(5, 2, self._model()))
        assert line == (
            '<p style="color:#616161; font-size:14px;">Fills: 5 of 7 trades walked a '
            "modeled order book (3 snapshots, taken 2026-10-02 to 2026-10-09); 2 at the "
            "top of the book (no depth data)</p>")

    def test_one_snapshot_on_one_day_reads_in_the_singular(self):
        line = dashboard._fills_html(self._sweep(
            1, 0, self._model(1, "2026-10-02T21:30:00Z", "2026-10-02T22:30:00Z")))
        assert ("Fills: 1 of 1 trades walked a modeled order book "
                "(1 snapshot, taken 2026-10-02); 0 at the top of the book") in line

    @pytest.mark.parametrize("first, last", [("", ""), ("unknown", "2026-10-09T21:30:00Z"),
                                             (None, None), ("2026-13-45", "2026-10-09")])
    def test_a_stamp_that_is_not_a_date_leaves_the_dates_out(self, first, last):
        line = dashboard._fills_html(self._sweep(2, 0, self._model(2, first, last)))
        assert "(2 snapshots); 0 at the top of the book" in line
        assert "taken" not in line

    def test_no_model_is_a_red_line_naming_the_command(self):
        line = dashboard._fills_html(self._sweep(0, 4, None))
        assert line.endswith(self._NO_MODEL)
        assert "color:#B71C1C" in line and "font-weight:700" in line
        assert "walked a modeled" not in line

    def test_no_sweep_says_not_recorded(self):
        assert dashboard._fills_html(None) == (
            '<p style="color:#616161; font-size:14px;">Fills: not recorded</p>')

    def test_a_run_with_no_trades_counts_zero_of_zero(self):
        line = dashboard._fills_html(self._sweep(0, 0, self._model()))
        assert "Fills: 0 of 0 trades walked a modeled order book" in line

    def test_the_count_reads_the_primary_points_trades_only(self):
        # Another point's trades (a swept k, a band's scenario) are not counted
        sweep = self._sweep(1, 1, self._model())
        other = _scn_point((0.3, 0.6), 0.5, trades=[make_trade()] * 9)
        sweep = dataclasses.replace(sweep, points=[sweep.primary, other], scenarios=[other])
        assert "Fills: 1 of 2 trades" in dashboard._fills_html(sweep)

    def test_the_page_places_it_under_the_entry_checkpoint(self, monkeypatch, tmp_path):
        page = TestRunSettingsHeader._page(monkeypatch, tmp_path,
                                           sweep=self._sweep(1, 1, self._model()))
        assert "Fills: 1 of 2 trades walked a modeled order book" in page
        assert page.count("Fills: ") == 1
        assert (page.index("Period:") < page.index("Primary spread band:")
                < page.index("Entry checkpoint:") < page.index("Fills: ")
                < page.index("Portfolio Performance"))

    def test_a_page_without_a_depth_model_carries_the_red_line(self, monkeypatch, tmp_path):
        page = TestRunSettingsHeader._page(monkeypatch, tmp_path,
                                           sweep=self._sweep(0, 1, None))
        assert self._NO_MODEL in page

    def test_a_page_without_a_sweep_says_not_recorded(self, monkeypatch, tmp_path):
        page = TestRunSettingsHeader._page(monkeypatch, tmp_path)
        assert "Fills: not recorded</p>" in page

    def test_the_model_a_run_records_reaches_the_page(self, monkeypatch, tmp_path):
        # The model run_backtest_sweep is handed is the one the header names
        monkeypatch.setattr(backtester, "_prepare_candidates", lambda *a, **k: None)
        sweep = backtester.run_backtest_sweep(
            MagicMock(), MagicMock(), date(2026, 1, 1), 1000.0, sweep=False,
            depth_model=self._model(2, "2026-10-02T21:30:00Z", "2026-10-02T21:30:00Z"))
        page = TestRunSettingsHeader._page(monkeypatch, tmp_path, sweep=sweep)
        assert ("Fills: 0 of 0 trades walked a modeled order book "
                "(2 snapshots, taken 2026-10-02); 0 at the top of the book") in page

    def test_a_run_without_a_model_reaches_the_page_as_the_red_line(self, monkeypatch,
                                                                    tmp_path):
        monkeypatch.setattr(backtester, "_prepare_candidates", lambda *a, **k: None)
        sweep = backtester.run_backtest_sweep(MagicMock(), MagicMock(), date(2026, 1, 1),
                                              1000.0, sweep=False)
        page = TestRunSettingsHeader._page(monkeypatch, tmp_path, sweep=sweep)
        assert self._NO_MODEL in page


class TestStartingBalanceHeader:
    """The Period line's starting balance says where the amount came from when
    backtest.py names it — the account's value when the run started, or
    --balance — and is the amount alone on a call that names nothing."""

    _SOURCE = ("the account's value at 2026-10-02 21:30 UTC: cash $116.15 + open "
               "positions $95.27")

    def test_the_source_is_printed_beside_the_amount(self, monkeypatch, tmp_path):
        page = TestRunSettingsHeader._page(monkeypatch, tmp_path, balance_source=self._SOURCE)
        assert ("Starting balance: $1,000.00 (the account's value at 2026-10-02 "
                "21:30 UTC: cash $116.15 + open positions $95.27) &nbsp;|&nbsp;") in page
        # On the Period line, above every section
        assert (page.index("Period:") < page.index("Starting balance:")
                < page.index("Trades found:") < page.index("Portfolio Performance"))

    def test_no_source_shows_the_amount_alone(self, monkeypatch, tmp_path):
        page = TestRunSettingsHeader._page(monkeypatch, tmp_path)
        assert "Starting balance: $1,000.00 &nbsp;|&nbsp;" in page

    @pytest.mark.parametrize("source", [None, "", MagicMock(), 7])
    def test_anything_but_a_note_shows_the_amount_alone(self, source):
        assert dashboard._starting_balance_text(211.42, source) == "$211.42"

    def test_the_note_is_escaped(self):
        # &, < and > are escaped; an apostrophe, harmless in page text, is kept
        assert (dashboard._starting_balance_text(10_000.0, "set by <--balance> & 'x'")
                == "$10,000.00 (set by &lt;--balance&gt; &amp; 'x')")


class TestCorpusProvenanceHeader:
    """DR-13 (P2): directly under the Period line the header says what
    settled-market corpus the run read — its assembly time (the Period runs to
    today, the corpus only to that moment), whether it came from an earlier
    run's cache, and the archive cutoff as of assembly — on EVERY run, healthy
    or not (DR-66). The cutoff is information only: a post-cutoff market is
    priced from the live candlestick endpoint, so the red "structurally
    0-trade" banner is gone. A legacy .json hit shows its file time. Rendered
    through generate_dashboard(), because a unit test that does not prove the
    string reaches the page is the gap DR-66b was about."""

    ASSEMBLED = datetime(2026, 9, 24, 12, 37, 49, tzinfo=UTC)
    CUTOFF = datetime(2026, 7, 25, tzinfo=UTC)

    def _page(self, monkeypatch, tmp_path, prov=None, *, sweep=True,
              n_trades=0, other_point_trades=0) -> str:
        # The page's trades and the sweep's primary point agree, as they do in
        # production (backtest.py passes result.primary.trades); a second k
        # point can carry trades of its own, as the filter bar's k select can show.
        monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(dashboard.yf, "download",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
        trades = [make_trade() for _ in range(n_trades)]
        pt = _scn_point((0.3, 0.6), 0.75, trades=trades)
        other = _scn_point((0.3, 0.6), 0.5,
                           trades=[make_trade() for _ in range(other_point_trades)])
        kwargs = {}
        if sweep:
            kwargs["sweep"] = BacktestSweep(primary=pt, points=[other, pt],
                                            calibration=None, corpus_provenance=prov)
        out_path = dashboard.generate_dashboard(
            trades, make_equity([1000.0, 1010.0, 1005.0]), date(2026, 1, 5),
            1000.0, **kwargs)
        return out_path.read_text(encoding="utf-8")

    def _prov(self, **overrides):
        fields = {"from_cache": False, "assembled_at": self.ASSEMBLED,
                  "archive_cutoff": self.CUTOFF}
        fields.update(overrides)
        return CorpusProvenance(**fields)

    @staticmethod
    def _corpus_line(page: str) -> str:
        start = page.index("Settled-market corpus:")
        return page[start: page.index("</p>", start)]

    def test_a_healthy_fresh_run_still_shows_its_assembly(self, monkeypatch, tmp_path):
        page = self._page(monkeypatch, tmp_path, self._prov())
        line = self._corpus_line(page)
        assert ("assembled 2026-09-24 12:37 UTC — it holds no market settled "
                "after that") in line
        assert "(assembled by this run)" in line
        assert "archive cutoff at assembly: 2026-07-25" in line
        assert "archive cutoff (" not in page  # no banner on a pre-cutoff window
        # Directly under the Period line, above the run settings and every section
        assert (page.index("Period:") < page.index("Settled-market corpus:")
                < page.index("Primary spread band:") < page.index("Portfolio Performance"))

    def test_a_cached_run_names_its_cache_and_the_remedy(self, monkeypatch, tmp_path):
        # A cache is served as it is only on the UTC day it was assembled;
        # an earlier day's is extended before it is served
        line = self._corpus_line(self._page(monkeypatch, tmp_path,
                                            self._prov(from_cache=True)))
        assert ("served from an earlier run&#x27;s cache, assembled earlier today; "
                "--no-cache re-assembles it in full") in line

    def test_a_cache_that_could_not_be_brought_up_to_date_says_so(
            self, monkeypatch, tmp_path):
        # An earlier day's cache served as it was after its extension failed
        line = self._corpus_line(self._page(monkeypatch, tmp_path,
                                            self._prov(from_cache=True, stale=True)))
        assert ("(served as an earlier day&#x27;s cache: it could not be brought up "
                "to date this run (see the log); --no-cache re-assembles it in full)"
                ) in line
        assert "earlier today" not in line

    @pytest.mark.parametrize("from_cache, source", [
        (False, "extended through today by this run"),
        (True, "served from an earlier run&#x27;s cache, assembled earlier today"),
    ])
    def test_an_extended_corpus_names_its_full_assembly(
            self, monkeypatch, tmp_path, from_cache, source):
        line = self._corpus_line(self._page(monkeypatch, tmp_path, self._prov(
            from_cache=from_cache, full_assembly_at=datetime(2026, 9, 20, 8, 15, tzinfo=UTC))))
        assert ("assembled 2026-09-24 12:37 UTC — it holds no market settled after "
                "that (extended day by day since a full assembly of 2026-09-20 "
                f"08:15 UTC) ({source}") in line

    @pytest.mark.parametrize("from_cache", [False, True])
    def test_a_post_cutoff_window_gets_no_banner(self, monkeypatch, tmp_path, from_cache):
        # A cutoff after the window's start date used to draw a red
        # "no trade could be entered" banner (or an amber stale-verdict line).
        # Post-cutoff markets are priced from the live candlestick endpoint
        # now, so the cutoff is reported and nothing more.
        page = self._page(monkeypatch, tmp_path, self._prov(
            archive_cutoff=datetime(2030, 1, 1, tzinfo=UTC), from_cache=from_cache),
            n_trades=2)
        assert "archive cutoff at assembly: 2030-01-01" in self._corpus_line(page)
        for gone in ("This window starts at or after", "no trade could be entered",
                     "verdict is stale", "structural"):
            assert gone not in page

    def test_an_unrecorded_cutoff_says_so(self, monkeypatch, tmp_path):
        page = self._page(monkeypatch, tmp_path, self._prov(
            from_cache=True, archive_cutoff=None))
        assert self._corpus_line(page).endswith("archive cutoff at assembly: not recorded")

    def test_an_unrecorded_assembly_time_says_so(self, monkeypatch, tmp_path):
        line = self._corpus_line(self._page(monkeypatch, tmp_path,
                                            self._prov(assembled_at=None)))
        assert "assembly time not recorded" in line

    @pytest.mark.parametrize("sweep", [True, False])
    def test_no_provenance_reads_not_recorded(self, monkeypatch, tmp_path, sweep):
        # An infeasible window, a stubbed corpus, or no sweep at all: said,
        # never silently absent.
        page = self._page(monkeypatch, tmp_path, None, sweep=sweep)
        assert ("Settled-market corpus: assembly time and archive cutoff not "
                "recorded for this run (no sweep was passed to the report") in page


class TestFigHtmlDivId:
    """_fig_html's div_id keyword: opt-in, backward compatible."""

    def test_default_lets_plotly_generate_the_id(self):
        import plotly.graph_objects as go
        out = dashboard._fig_html(go.Figure())
        assert re.search(r'<div id="[0-9a-f-]{36}"', out)

    def test_explicit_div_id_is_passed_through(self):
        import plotly.graph_objects as go
        assert 'id="my-fixed-id"' in dashboard._fig_html(go.Figure(), div_id="my-fixed-id")


class TestScenarioExplorerPageSize:
    """A full page for the real grid (36 bands x 13 ks) over a V3-length window
    (2,459 days), with realistic float values, every population point, a
    calibration per band and a top event per cell. Every cell trades ONE list,
    so the filter bar packs one chunk per k: 13. The scenario explorer ships
    its base block and one block per size cap (one here: no size-cap sweep),
    gzip-packed, each cell's curve as change points on its weekly axis.
    Measured 2026-09-26 (plotly 6.9.0, pandas 3.0.3, numpy 2.4.6): a
    1,724,773-byte page, the explorer's two blocks 95,920 bytes of base64 —
    against 3,902,183 bytes for the same fixture before the explorer's
    blocks were packed (C3), when its data was one JSON block with every
    curve dense. The budget is the measurement + 20% (the curve runs to
    today, so the page grows by a few bytes a day) — well inside the 5 MB
    the explorer first set. Re-measured at the merge of #68 (2026-09-26, same
    environment): 1,729,930 bytes without a tier-floors-off family (the
    bar's Tier floors select, its title and note, and each band's option
    text add about 5 KB). The same page with a family for its 18 binding
    bands, every one at every k in every population (another 936 points, on
    a second random walk), each tier-off cell trading ONE list of its own at
    each k — a list, not only a curve, of its own, since a chunk is keyed on
    k and trades (_list_key) and a family trading the tier-on list would
    share the tier-on chunks and measure nothing of itself — packs 13 more
    chunks for the bar's off grid: 2,559,454 bytes. Then the explorer's own
    tier-floors-off view (the merge's stage C: its Tier floors select, a
    hidden tier-off k-hat table, a longer script, and — with the family — a
    packed "scn-off-cap-0" block holding the 18 binding bands' rows and that
    grid's banner, matrices and titles), re-measured the same day in the
    same environment: 1,736,090 bytes without the family (the explorer's
    blocks 3,448 + 92,684 bytes of base64, unchanged but for the base
    block's new keys, null without a family) and 2,632,657 with it (base
    3,848, cap 92,684 and off block 56,340 bytes of base64). Each variant's
    budget is its measurement + 20%. (Before this branch's packed chunks,
    main measured this construction — its family trading the tier-on list,
    with the explorer's unpacked tier-off view — at 2,909,152 bytes without
    the family and 4,122,568 with it.)"""

    PAGE_BYTES_MEASURED = 1_736_090
    BUDGET = int(PAGE_BYTES_MEASURED * 1.2)
    PAGE_BYTES_MEASURED_FAMILY = 2_632_657
    BUDGET_FAMILY = int(PAGE_BYTES_MEASURED_FAMILY * 1.2)

    @pytest.mark.parametrize("family", [False, True], ids=["tier-on-only", "tier-off-family"])
    def test_page_size_under_5mb_for_a_468_cell_sweep(self, monkeypatch, tmp_path, family):
        import numpy as np
        monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(dashboard.yf, "download",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))

        bands = [(lo, hi) for lo in config.SPREAD_BAND_SWEEP_FLOORS
                 for hi in config.SPREAD_BAND_SWEEP_CEILINGS]
        ks = list(config.INTERVAL_DISCOUNT_SWEEP)
        assert len(bands) * len(ks) == 468

        # A seeded random walk: full-precision dollar values, as a real curve
        # carries, not short round numbers that would understate the payload.
        rng = np.random.default_rng(7)
        walk = 10_000.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, 2459))
        shared_equity = make_equity(list(walk), start=date(2019, 12, 31))
        trades = [_scn_trade(profit=float(p), event_ticker=f"KXEVENT-26SEP{i:02d}-ABCDEF")
                  for i, p in enumerate(rng.normal(0.0, 20.0, 88))]

        def points(band, k, equity, cell_trades=trades, **stamp) -> list[SweepPoint]:
            # PB7: the "all" AND the "time_series" point each carry both
            # checks, and the time-series one carries the per-cell curve
            out = [SweepPoint(k=k, trades=cell_trades, equity_df=equity, spread_band=band,
                              population=population,
                              halves=HalfSplit(float(rng.normal()), float(rng.normal()), 40, 48,
                                               120, 130),
                              ex_top_event=("KXEVENT-26SEP03-ABCDEF", float(rng.normal())),
                              **stamp)
                   for population in ("all", "time_series")]
            out += [SweepPoint(k=k, trades=cell_trades[:40], equity_df=equity,
                               spread_band=band, population=population, **stamp)
                    for population in ("ladder", "cross")]
            return out

        scenarios = [pt for band in bands for k in ks
                     for pt in points(band, k, shared_equity)]
        # Drawn after every tier-on point, so the tier-on half is the same
        # either way
        off, off_cals = [], {}
        if family:
            walk_off = 10_000.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, 2459))
            off_equity = make_equity(list(walk_off), start=date(2019, 12, 31))
            # A list of its own (the curve alone would not do: a chunk is
            # keyed on k and trades, so a family trading the tier-on list
            # would share the tier-on chunks and measure nothing of itself)
            off_trades = [_scn_trade(profit=float(p),
                                     event_ticker=f"KXEVENT-26OCT{i:02d}-ABCDEF")
                          for i, p in enumerate(rng.normal(0.0, 20.0, 88))]
            for band in bands:
                if backtester._tier_floors_bind(band):
                    off += [pt for k in ks
                            for pt in points(band, k, off_equity, off_trades,
                                             tier_floors=False)]
                    off_cals[band] = _scn_calibration("16-30d", 7)
            assert len(off_cals) == 18 and len(off) == 936
        sweep = BacktestSweep(
            primary=scenarios[0], points=[scenarios[0]], calibration=None,
            label_coverage=_scn_coverage(), scenarios=scenarios,
            same_title_point=SweepPoint(k=ks[0], trades=trades[:10],
                                        equity_df=shared_equity, population="same_title"),
            calibrations_by_band={b: _scn_calibration() for b in bands},
            same_event_ladders=True, split_date=date(2023, 5, 1),
            tier_off_scenarios=off, tier_off_calibrations_by_band=off_cals,
        )

        out_path = dashboard.generate_dashboard(
            trades, shared_equity, date(2020, 1, 1), 10_000.0, sweep=sweep)
        page = out_path.read_text(encoding="utf-8")
        # The family reaches the bar only when it exists: its off grid in the
        # base block, and a chunk per k of its own list (13 more)
        assert len(TestFilterPage._chunks(page)) == (26 if family else 13)
        # ... and the explorer's own tier-floors-off view exactly when the
        # family exists: its select, its hidden k-hat table and its off block
        # at the run's own (only) cap
        assert list(_scn_blocks(page)) == ["scn-data", "scn-cap-0"] + (
            ["scn-off-cap-0"] if family else [])
        for piece in ('id="scn-tier-select"', 'id="scn-khat-off"'):
            assert (piece in page) is family, piece
        if family:
            off = _scn_blocks(page)["scn-off-cap-0"]
            assert sum(row is not None for row in off["cells"]) == 18
        data = TestFilterPage._data(page)
        assert (data["grid_off"] is not None) is family
        if family:
            # A binding band's off row is its own chunks, every other band's
            # its tier-on row
            assert sum(data["tier_binds"]) == 18
            for bi, own in enumerate(data["tier_binds"]):
                assert (data["grid_off"][bi] != data["grid"][bi]) is own, bi
        size = out_path.stat().st_size
        budget = self.BUDGET_FAMILY if family else self.BUDGET
        assert size <= budget, f"page was {size} bytes"


class TestMedianReturns:
    """Median per-trade return beside the mean (performance cards, the explorer's
    KPI table and heatmap), and the median CALENDAR-MONTH return card."""

    @staticmethod
    def _month_curve(values_by_day: dict[date, float]) -> pd.DataFrame:
        df = pd.DataFrame({"date": list(values_by_day),
                           "portfolio_value": list(values_by_day.values())})
        df["daily_return"] = df["portfolio_value"].pct_change().fillna(0.0)
        return df

    def test_months_chain_from_the_opening_row(self):
        # Opening 1000 on Jan 31, so January's own return is 0.0; then
        # +10% / -5% / +10% month over month -> median of [0, .10, -.05, .10]
        curve = self._month_curve({
            date(2026, 1, 31): 1000.0,
            date(2026, 2, 14): 1300.0,          # intra-month values are ignored
            date(2026, 2, 28): 1100.0,
            date(2026, 3, 31): 1045.0,
            date(2026, 4, 30): 1149.5,
        })
        assert dashboard._median_monthly_return(curve) == pytest.approx(0.05)

    def test_flat_months_count_as_zero(self):
        # One active month out of three: the median describes the months lived
        curve = self._month_curve({date(2026, 1, 1): 1000.0, date(2026, 2, 1): 1200.0,
                                   date(2026, 3, 1): 1200.0})
        assert dashboard._median_monthly_return(curve) == 0.0

    def test_undefined_curves_are_none_not_zero(self):
        assert dashboard._median_monthly_return(None) is None
        assert dashboard._median_monthly_return(make_equity([])) is None
        assert dashboard._median_monthly_return(make_equity([0.0, 10.0])) is None

    def test_performance_cards_show_both_medians(self):
        # Skewed on purpose: one big win pulls the mean far from the median
        trades = [make_trade(profit=p) for p in (-1.0, 0.5, 0.6, 40.0)]
        # Jan 30 -> Jan 31 -> Feb 1: January +2.0%, February +1.97%, total +4.0%
        curve = make_equity([1000.0, 1020.0, 1040.1], start=date(2026, 1, 30))
        section = dashboard._section_performance(curve, trades, date(2026, 1, 30), 1000.0)
        ratios = sorted(t.profit_ratio for t in trades)
        median = (ratios[1] + ratios[2]) / 2

        def card(label: str) -> str:
            # The value div follows the label div inside one card (_KPI_TEMPLATE)
            match = re.search(re.escape(label) + r"</div>\s*<div[^>]*>([^<]*)</div>", section)
            return match.group(1).strip()

        assert card("Median Return/Trade") == f"{median:.1%}"
        assert card("Avg Return/Trade") != card("Median Return/Trade")
        assert card("Median Monthly Return") == "+2.0%"
        assert card("Total Return") == "+4.0%"

    def test_point_kpis_carry_the_median(self):
        trades = [make_trade(profit=p) for p in (-1.0, 0.5, 40.0)]
        kpis = dashboard._point_kpis(_scn_point((0.0, 1.0), 0.75, trades=trades))
        assert kpis["median_per_trade"] == pytest.approx(
            sorted(t.profit_ratio for t in trades)[1])
        assert kpis["median_per_trade"] != pytest.approx(kpis["mean_per_trade"])
        empty = dashboard._point_kpis(_scn_point((0.0, 1.0), 0.75, trades=[]))
        assert empty["median_per_trade"] is None

    def test_the_explorer_table_and_heatmap_carry_the_median(self):
        section = _Grid.section()
        assert "<th style=\"padding:8px 16px;\">Median/Trade</th>" in section
        assert "fmtPct(k.median_per_trade)" in section
        # The primary cell (band 1, k index 1) is cell 3, whose trades differ
        data, cap = _scn_data(section), _scn_cap(section)
        ts = cap["cells"][1][1][data["populations"].index("time_series")]
        ratios = sorted(t.profit_ratio for t in _Grid.all_trades(3))
        assert ts["median_per_trade"] == pytest.approx((ratios[1] + ratios[2]) / 2)
        metric = next(m for m in data["metrics"] if m["label"] == "Median per trade (equal stake)")
        assert cap["matrices"][metric["key"]][1][1] == pytest.approx(ts["median_per_trade"])
        assert metric["zmid"] == 0
        # The figure as rendered: the default metric (the mean), not the median
        heat, _ = _first_figure(section)
        assert heat[0]["z"] == cap["matrices"]["mean_per_trade"]


def _typed_trade(pair_type: str, ladder: bool, entry: date, exit_: date,
                 profit: float) -> BacktestTrade:
    """make_trade with a chosen pair type, ladder flag, dates and profit, its
    payoff set so profit = actual_payoff - total_cost - fees holds."""
    t = make_trade(profit=profit)
    return dataclasses.replace(
        t, pair_type=pair_type, same_event_ladder=ladder, entry_date=entry,
        exit_date=exit_, actual_payoff=profit + t.total_cost + t.fees)


class TestReturnByTradeType:
    """The performance section's first chart: total return plus one line per
    trade type, booked exactly as _build_equity_curve books each trade."""

    def test_type_lines_sum_to_the_total_return(self):
        start = date(2026, 1, 5)
        trades = [
            _typed_trade("same_title", False, date(2026, 1, 6), date(2026, 1, 8), 30.0),
            _typed_trade("time_series", True, date(2026, 1, 6), date(2026, 1, 9), -12.0),
            _typed_trade("time_series", False, date(2026, 1, 7), date(2026, 1, 9), 5.0),
            _typed_trade("same_title", False, date(2026, 1, 8), date(2026, 1, 9), -4.0),
        ]
        curve = backtester._build_equity_curve(trades, start, 1000.0)
        lines = dashboard._return_by_trade_type(trades, curve, 1000.0)
        assert [label for label, _, _ in lines] == [
            "Same-title", "Time-series: ladder", "Time-series: cross-event"]
        total = (curve["portfolio_value"] / 1000.0 - 1.0) * 100
        summed = [sum(values) for values in zip(*(s for _, _, s in lines), strict=True)]
        assert summed == pytest.approx(list(total), abs=1e-9)
        # Each type ends at its own profit
        ends = {label: s[-1] for label, _, s in lines}
        assert ends["Same-title"] == pytest.approx((30.0 - 4.0) / 10)
        assert ends["Time-series: ladder"] == pytest.approx(-1.2)
        assert ends["Time-series: cross-event"] == pytest.approx(0.5)

    def test_only_types_with_trades_get_a_line(self):
        trades = [_typed_trade("same_title", False, date(2026, 1, 6), date(2026, 1, 8), 3.0)]
        curve = backtester._build_equity_curve(trades, date(2026, 1, 5), 1000.0)
        assert [x for x, _, _ in dashboard._return_by_trade_type(trades, curve, 1000.0)] == [
            "Same-title"]
        assert dashboard._return_by_trade_type([], curve, 1000.0) == []

    def test_the_first_chart_carries_the_total_and_the_type_lines(self):
        trades = [
            _typed_trade("same_title", False, date(2026, 1, 6), date(2026, 1, 8), 3.0),
            _typed_trade("time_series", False, date(2026, 1, 6), date(2026, 1, 8), -1.0),
        ]
        curve = backtester._build_equity_curve(trades, date(2026, 1, 5), 1000.0)
        section = dashboard._section_performance(curve, trades, date(2026, 1, 5), 1000.0)
        data, layout = _first_figure(section)
        assert [tr["name"] for tr in data] == [
            "Total return", "Same-title", "Time-series: cross-event"]
        assert layout["title"]["text"].startswith("Cumulative Return by Trade Type")


class TestSectionDataHelpers:
    """The per-section data helpers every trade-derived section renders from:
    the one definition of each figure, which a filtered view of the page reads
    too, so they must describe exactly what the section shows."""

    def _trades(self):
        return [
            _typed_trade("same_title", False, date(2026, 1, 6), date(2026, 1, 8), 30.0),
            _typed_trade("time_series", True, date(2026, 1, 6), date(2026, 1, 9), -12.0),
            _typed_trade("time_series", False, date(2026, 2, 7), date(2026, 2, 9), 5.0),
        ]

    def test_every_performance_card_renders_its_label_and_value(self):
        trades = self._trades()
        curve = backtester._build_equity_curve(trades, date(2026, 1, 5), 1000.0)
        kpis = dashboard._performance_kpis(curve, trades, 1000.0)
        keys = [key for key, *_ in kpis]
        assert len(keys) == len(set(keys)) == 9
        section = dashboard._section_performance(curve, trades, date(2026, 1, 5), 1000.0)
        for key, label, value, color in kpis:
            assert dashboard._kpi(label, value, color, key=key) in section

    def test_the_decomposition_aggregates_leave_the_frame_untouched(self):
        df = dashboard._decomposition_frame(self._trades(), None)
        columns = list(df.columns)
        agg = dashboard._decomposition_aggregates(df)
        assert list(df.columns) == columns          # no price_bucket column added
        assert list(agg["monthly"]["month"]) == ["2026-01", "2026-02"]
        assert agg["monthly"]["profit"].sum() == pytest.approx(23.0)
        assert agg["price"].sum() == pytest.approx(23.0)

    def test_best_and_worst_keep_tied_trades_in_their_order(self):
        tied = [dataclasses.replace(make_trade(profit=1.0), ticker_a=f"T{i}") for i in range(7)]
        best, worst = dashboard._best_and_worst(tied)
        assert [t.ticker_a for t in best] == ["T0", "T1", "T2", "T3", "T4"]
        assert [t.ticker_a for t in worst] == ["T2", "T3", "T4", "T5", "T6"]

    def test_the_risk_helpers_match_the_scatter_and_the_deployment_trace(self):
        trades = self._trades()
        curve = backtester._build_equity_curve(trades, date(2026, 1, 5), 1000.0)
        kelly, actual = dashboard._kelly_points(trades, 0.75)
        section = dashboard._section_risk(trades, curve, 1000.0, k=0.75)
        data, _ = _nth_figure(section, 0)
        assert data[0]["x"] == pytest.approx(kelly)
        assert data[0]["y"] == pytest.approx(actual)
        assert data[1]["x"] == pytest.approx([0, dashboard._one_to_one_extent(kelly)])
        deployed, _ = _nth_figure(section, 1)
        assert deployed[0]["y"] == pytest.approx(dashboard._capital_deployed(trades, curve))


class TestBestWorstTradeRows:
    """Each best/worst row spells out both legs: the YES and NO prices paid,
    what each leg bought and when its market closed, and how each settled."""

    @staticmethod
    def _row(**overrides) -> str:
        t = dataclasses.replace(
            make_trade(profit=-10.0), title_a="Game <1>", title_b="Game <1>",
            subtitle_a="Western Illinois", subtitle_b="Western Illinois",
            ticker_a="KXNCAAWBGAME-X-WIU", ticker_b="KXNCAAMBGAME-X-WIU",
            close_date_a=date(2026, 1, 13), close_date_b=date(2026, 1, 14),
            settled_date_a=date(2026, 1, 14), settled_date_b=date(2026, 1, 15),
            **overrides)
        return dashboard._trade_row(t, "#FFF")

    @staticmethod
    def _cells(row: str) -> list[str]:
        return re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.S)

    def test_a_same_title_row_pays_no_on_a_and_yes_on_b(self):
        row = self._row(pair_type="same_title", entry_nA=0.32, entry_pB=0.35,
                        outcome_a="yes", outcome_b="no")
        cells = self._cells(row)
        assert cells[2:4] == ["$0.35", "$0.32"]          # YES paid, NO paid
        details, outcome = cells[4], cells[5]
        assert details.index("<b>NO</b> on Game &lt;1&gt; — Western Illinois") < details.index(
            "<b>YES</b> on")
        assert "(closes Jan 13, 2026) at $0.32" in details
        assert "(closes Jan 14, 2026) at $0.35" in details
        assert "KXNCAAWBGAME-X-WIU" in details and "KXNCAAMBGAME-X-WIU" in details
        assert "A: settled <b>YES</b> on Jan 14, 2026 (lost)" in outcome
        assert "B: settled <b>NO</b> on Jan 15, 2026 (lost)" in outcome

    def test_a_time_series_row_pays_yes_on_a_and_no_on_b(self):
        cells = self._cells(self._row(pair_type="time_series", entry_pA=0.30,
                                      entry_nB=0.40, outcome_a="no", outcome_b="no"))
        assert cells[2:4] == ["$0.30", "$0.40"]
        assert "A: settled <b>NO</b> on Jan 14, 2026 (lost)" in cells[5]
        assert "B: settled <b>NO</b> on Jan 15, 2026 (won)" in cells[5]

    def test_missing_dates_and_labels_degrade_readably(self):
        t = dataclasses.replace(make_trade(), subtitle_a="", subtitle_b="")
        cells = self._cells(dashboard._trade_row(t, "#FFF"))
        assert "(closes date unknown)" in cells[4]
        assert "on date unknown" in cells[5]
        assert " — " not in cells[4]


class TestFillsOnRowsAndScatter:
    """What a trade paid (backtester._paid_prices) is what the best/worst rows
    show and what the Kelly scatter prices at; the entry-price buckets and the
    spread calibration stay on the entry quotes; a trade built without fills
    reads its quotes, as before."""

    # The make_trade quotes: pA 0.30, nA 0.70, pB 0.60, nB 0.40
    @staticmethod
    def _with_fills(a: float, b: float, **overrides) -> BacktestTrade:
        return dataclasses.replace(make_trade(), fill_price_a=a, fill_price_b=b,
                                   book_walked=True, **overrides)

    def test_a_time_series_row_shows_the_fills_not_the_quotes(self):
        row = TestBestWorstTradeRows._row(
            pair_type="time_series", fill_price_a=0.33, fill_price_b=0.43,
            outcome_a="no", outcome_b="no", book_walked=True)
        cells = TestBestWorstTradeRows._cells(row)
        assert cells[2:4] == ["$0.33", "$0.43"]          # YES on A, NO on B
        assert "(closes Jan 13, 2026) at $0.33" in cells[4]
        assert "(closes Jan 14, 2026) at $0.43" in cells[4]
        assert "$0.30" not in row and "$0.40" not in row

    def test_a_same_title_row_shows_the_fills_not_the_quotes(self):
        row = TestBestWorstTradeRows._row(
            pair_type="same_title", entry_nA=0.32, entry_pB=0.35,
            fill_price_a=0.34, fill_price_b=0.37, outcome_a="yes", outcome_b="no",
            book_walked=True)
        cells = TestBestWorstTradeRows._cells(row)
        assert cells[2:4] == ["$0.37", "$0.34"]          # YES on B, NO on A
        assert "at $0.34" in cells[4] and "at $0.37" in cells[4]
        assert "$0.32" not in row and "$0.35" not in row

    def test_a_trade_with_no_fills_still_shows_its_quotes(self):
        cells = TestBestWorstTradeRows._cells(dashboard._trade_row(make_trade(), "#FFF"))
        assert cells[2:4] == ["$0.30", "$0.40"]

    def test_a_head_changes_with_the_fills_and_not_with_the_size(self):
        walked = self._with_fills(0.33, 0.43)
        assert (dashboard._trade_row_head(walked, "#FFF")
                != dashboard._trade_row_head(make_trade(), "#FFF"))
        resized = dataclasses.replace(walked, n=9, total_cost=3.0, fees=0.2)
        assert (dashboard._trade_row_head(resized, "#FFF")
                == dashboard._trade_row_head(walked, "#FFF"))

    def test_quotes_at_fills_swap_in_only_the_legs_bought(self):
        ts = self._with_fills(0.33, 0.43)
        # (YES fill on A, A's NO quote, B's YES quote stays the reference, NO fill on B)
        assert dashboard._quotes_at_fills(ts) == (0.33, 0.70, 0.60, 0.43)
        st = dataclasses.replace(ts, pair_type="same_title")
        # (A's YES quote, NO fill on A, YES fill on B, B's NO quote)
        assert dashboard._quotes_at_fills(st) == (0.30, 0.33, 0.43, 0.40)
        assert dashboard._quotes_at_fills(make_trade()) == (0.30, 0.70, 0.60, 0.40)

    def test_the_scatter_prices_a_time_series_trade_at_its_fills(self):
        k = 0.75
        x, _ = dashboard._kelly_points([self._with_fills(0.31, 0.41)], k)
        # Hand-worked: the YES fill and the NO fill are the leg prices, and the
        # forecast reads the entry quotes' mid spread, 0.60 - 0.30 on books
        # with no width (never the fills), as the live sizer prices its legs
        # at the fills and its forecast at the book's tops
        p = 1.0 - k * 0.30
        fee = fee_per_pair_approx(0.31, 0.41)
        b = (1.0 - 0.31 - 0.41 - fee) / (0.31 + 0.41 + fee)
        assert p - (1.0 - p) / b > 0
        assert x == [pytest.approx(p - (1.0 - p) / b)]
        assert x[0] != pytest.approx(dashboard._kelly_points([make_trade()], k)[0][0])

    def test_the_scatter_forecasts_at_the_entry_mid_spread_not_the_fills(self):
        # Market A's entry book is 0.05 wide (YES ask 0.30, NO ask 0.75), so
        # the entry mid spread is 0.325: the forecast reads that, while the
        # four quotes with the fills swapped in would read 0.315
        k = 0.75
        t = self._with_fills(0.31, 0.41, entry_nA=0.75)
        assert dashboard._entry_mid_spread(t) == pytest.approx(0.325)
        assert dashboard._entry_mid_spread(t) == time_series_mid_spread(0.30, 0.75, 0.60, 0.40)
        x, _ = dashboard._kelly_points([t], k)
        assert x == [_kelly_fraction(0.31, 0.75, 0.60, 0.41, "time_series", k=k,
                                     spread=dashboard._entry_mid_spread(t))]
        p = 1.0 - k * 0.325
        fee = fee_per_pair_approx(0.31, 0.41)
        b = (1.0 - 0.31 - 0.41 - fee) / (0.31 + 0.41 + fee)
        assert p - (1.0 - p) / b > 0
        assert x == [pytest.approx(p - (1.0 - p) / b)]
        # Not the mid spread of the four with the fills swapped in
        assert time_series_mid_spread(*dashboard._quotes_at_fills(t)) == pytest.approx(0.315)
        assert x[0] != pytest.approx(
            _kelly_fraction(*dashboard._quotes_at_fills(t), "time_series", k=k))

    def test_a_same_title_trade_has_no_entry_mid_spread(self):
        st = dataclasses.replace(make_trade(), pair_type="same_title")
        assert dashboard._entry_mid_spread(st) is None
        # ... and the spread keyword never reaches the same-title prior
        assert (_kelly_fraction(0.70, 0.20, 0.30, 0.65, "same_title", spread=0.99)
                == _kelly_fraction(0.70, 0.20, 0.30, 0.65, "same_title"))

    def test_the_scatter_prices_a_same_title_trade_at_its_fills(self):
        t = dataclasses.replace(self._with_fills(0.62, 0.30), pair_type="same_title",
                                entry_nA=0.60, entry_pB=0.28)
        x, _ = dashboard._kelly_points([t], 0.75)
        p = SAME_TITLE_CO_RESOLVE_PROB
        fee = fee_per_pair_approx(0.62, 0.30)
        b = (1.0 - 0.62 - 0.30 - fee) / (0.62 + 0.30 + fee)
        assert x == [pytest.approx(max(0.0, p - (1.0 - p) / b))]

    @pytest.mark.parametrize("pair_type", ["time_series", "same_title"])
    def test_a_trade_with_no_fills_plots_its_quotes_exactly(self, pair_type):
        t = dataclasses.replace(make_trade(), pair_type=pair_type)
        x, _ = dashboard._kelly_points([t], 0.75)
        assert x == [_kelly_fraction(t.entry_pA, t.entry_nA, t.entry_pB, t.entry_nB,
                                     pair_type, k=0.75)]

    def test_a_walked_trade_at_the_quotes_plots_the_same_point(self):
        # Fills equal to the top-of-book quotes change nothing
        x_walked, _ = dashboard._kelly_points([self._with_fills(0.30, 0.40)], 0.75)
        x_quotes, _ = dashboard._kelly_points([make_trade()], 0.75)
        assert x_walked == x_quotes

    def test_the_risk_section_scatter_reads_the_fills(self):
        t = self._with_fills(0.33, 0.43)
        curve = backtester._build_equity_curve([t], date(2026, 1, 5), 1000.0)
        data, _ = _nth_figure(dashboard._section_risk([t], curve, 1000.0, k=0.75), 0)
        assert data[0]["x"] == pytest.approx(dashboard._kelly_points([t], 0.75)[0])

    def test_the_entry_price_buckets_and_spread_stay_on_the_quotes(self):
        t = self._with_fills(0.33, 0.43, outcome_a="no", outcome_b="yes")
        frame = dashboard._decomposition_frame([t], None)
        assert list(frame["entry_pA"]) == [0.30]
        # The entry quotes' mid spread (0.30 on these books with no width),
        # never the fills', and the in-between cell as the outcome
        assert dashboard._spread_observations([t]) == [(pytest.approx(0.30), 1)]

    def test_chunk_keys_separate_trade_lists_that_differ_only_in_fills(self):
        base = make_trade()
        a = [dataclasses.replace(base, fill_price_a=0.31, fill_price_b=0.41)]
        b = [dataclasses.replace(base, fill_price_a=0.32, fill_price_b=0.41)]
        c = [dataclasses.replace(base, fill_price_a=0.31, fill_price_b=0.41, book_walked=True)]
        keys = {dashboard._list_key(0.75, lst) for lst in (a, b, c, [base])}
        assert len(keys) == 4
        # Equal lists still share one key
        twin = [dataclasses.replace(base, fill_price_a=0.31, fill_price_b=0.41)]
        assert dashboard._list_key(0.75, twin) == dashboard._list_key(0.75, a)
        # The three new fields are among the compared ones the key reads
        assert {"fill_price_a", "fill_price_b", "book_walked"} <= set(dashboard._TRADE_FIELDS)


class TestReturnsByCategory:
    """Returns Decomposition files every trade under Kalshi's official series
    category and first tag, with a per-category·tag table."""

    SERIES = {
        "KXNCAAMBGAME": ("Sports", ("Basketball",)),
        "KXUCLGAME": ("Sports", ("Soccer", "Europe")),
        "KXFISAEXTEND": ("Politics", ("Congress",)),
        "KXSCOTUSLAST": ("Politics", ()),
        "KXMVECROSSCATEGORY": ("", ()),
    }

    @staticmethod
    def _t(event_ticker: str, profit: float) -> BacktestTrade:
        return dataclasses.replace(make_trade(profit=profit), event_ticker=event_ticker)

    @pytest.mark.parametrize("event_ticker,expected", [
        ("KXNCAAMBGAME-26JAN13WIUEIU", ("Sports", "Sports · Basketball")),
        ("KXUCLGAME-26APR14ATMBAR", ("Sports", "Sports · Soccer")),       # FIRST tag
        ("KXSCOTUSLAST-26", ("Politics", "Politics · General")),         # no tags
        ("KXMVECROSSCATEGORY-S2026X", ("Uncategorised", "Uncategorised · General")),
    ])
    def test_a_trade_is_filed_under_its_series(self, event_ticker, expected):
        assert dashboard._trade_category(self._t(event_ticker, 1.0), self.SERIES) == expected

    def test_an_unknown_series_or_no_map_falls_back_to_the_prefix_label(self):
        t = dataclasses.replace(self._t("KXNOTLISTED-1", 1.0), category="Crypto")
        assert dashboard._trade_category(t, self.SERIES) == ("Crypto", "Crypto · General")
        assert dashboard._trade_category(t, None) == ("Crypto", "Crypto · General")

    def test_combo_series_are_looked_up_literally(self):
        # Not scanner.event_series, which folds every KXMVE* series together
        assert historical.series_ticker("KXMVECROSSCATEGORY0-S1") == "KXMVECROSSCATEGORY0"
        assert historical.series_ticker("") == ""

    def test_the_page_files_by_the_one_rule_the_live_filter_shares(self):
        # The one filing rule, shared with main._filter_by_category
        assert dashboard._series_labels is historical.series_labels

    def test_the_section_charts_and_tabulates_each_category_tag(self):
        trades = [self._t("KXNCAAMBGAME-A", 30.0), self._t("KXNCAAMBGAME-B", -10.0),
                  self._t("KXFISAEXTEND-C", 50.0), self._t("KXUCLGAME-D", -5.0)]
        section = dashboard._section_decomposition(trades, self.SERIES)
        # Decoded as JSON strings, not read raw: without orjson (as in CI)
        # plotly serialises through the stdlib json module, which writes the
        # "·" as ·
        titles = [json.loads(t) for t in re.findall(
            r'"title":\s*\{"text":\s*("(?:[^"\\]|\\.)*")', section)]
        assert "P&L by Category ($)" in titles
        assert "P&L by Category · Tag ($)" in titles
        table = section[section.index("Returns by category · tag"):]
        table = table[:table.index("</table>")]
        assert "(3 groups" in table
        rows = [re.findall(r"<td[^>]*>(.*?)</td>", r)
                for r in re.findall(r"<tr style='border-bottom[^>]*>(.*?)</tr>", table)]
        # Largest P&L first; win rate, P&L and share of the 65.00 total
        assert [r[0] for r in rows] == ["Politics · Congress", "Sports · Basketball",
                                         "Sports · Soccer"]
        assert rows[1][1:5] == ["2", "50%", "$+20.00", "31%"]
        assert rows[2][2:4] == ["0%", "$-5.00"]


class TestDashboardFileIsOverwritten:
    """Every run writes the one backtest_dashboard.html, replacing the last."""

    def test_a_second_run_replaces_the_first(self, monkeypatch, tmp_path):
        monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(dashboard.yf, "download",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
        first = dashboard.generate_dashboard([], make_equity([1000.0, 1000.0]),
                                             date(2026, 1, 5), 1000.0)
        (tmp_path / first.name).write_text("stale")
        second = dashboard.generate_dashboard([make_trade()], make_equity([1000.0, 1010.0]),
                                              date(2026, 1, 5), 1000.0)
        assert first == second == tmp_path / "backtest_dashboard.html"
        assert "stale" not in second.read_text(encoding="utf-8")
        # Nothing else is left behind: no timestamped page, no temporary file
        assert sorted(p.name for p in tmp_path.iterdir()) == ["backtest_dashboard.html"]


class TestEmpiricalKHatByBand:
    """k-hat per spread band on the explorer's grid: a heatmap metric that
    repeats each band's POOLED k-hat across every k column (k-hat does not
    depend on k), with the entries behind it, plus the same figures as a table."""

    KHAT = "Empirical k̂ (pooled per band)"
    DELTA = "k̂ − k (pooled per band, minus the column's k)"

    @staticmethod
    def _sweep() -> BacktestSweep:
        sweep = _Grid.sweep()
        base = _scn_calibration()
        wide = dataclasses.replace(base, pooled=dataclasses.replace(
            base.pooled, n=4, realised_rate=0.27, mean_implied=0.30, empirical_k=0.90))
        undefined = dataclasses.replace(base, pooled=dataclasses.replace(
            base.pooled, n=0, realised_rate=0.0, mean_implied=0.0, empirical_k=None))
        return dataclasses.replace(sweep, calibrations_by_band={
            _Grid.BANDS[0]: base, _Grid.BANDS[1]: wide, _Grid.BANDS[2]: undefined})

    @staticmethod
    def _metrics(section: str) -> dict[str, dict]:
        """Each heatmap metric's spec, by label, with its z and customdata
        matrices resolved as the script resolves them (the cap's block, else
        the base block)."""
        data, cap = _scn_data(section), _scn_cap(section)
        matrices = cap["matrices"] | data["matrices"]
        return {m["label"]: dict(m, z=matrices[m["key"]], customdata=matrices[m["custom"]])
                for m in data["metrics"]}

    def test_each_band_row_repeats_its_pooled_khat(self):
        section = _section_scenario_explorer(self._sweep())
        khat = self._metrics(section)[self.KHAT]
        assert khat["z"] == [[0.60, 0.60], [0.90, 0.90], [None, None]]
        # Its hover reads the entries pooled, not the trade count
        assert khat["customdata"] == [[10, 10], [4, 4], [0, 0]]
        # Centred on the run's primary k (the primary cell sits at k = 0.75)
        assert khat["zmid"] == 0.75
        assert "entries pooled" in khat["hover"]
        # k-hat is measured before the Kelly gate: it ships once, not per cap
        assert "empirical_k" in _scn_data(section)["matrices"]
        assert "empirical_k" not in _scn_cap(section)["matrices"]
        # The figure as rendered: the default metric's z, not k-hat's
        heat, _ = _first_figure(section)
        assert heat[0]["z"] == _scn_cap(section)["matrices"]["mean_per_trade"]

    def test_every_other_metric_restores_the_trade_counts(self):
        # customdata travels with every update, so switching back from k-hat
        # must put the trade counts back under the other metrics' hovers
        section = _section_scenario_explorer(self._sweep())
        metrics = self._metrics(section)
        trades = metrics["Trade count"]["z"]
        for label, spec in metrics.items():
            if label in (self.KHAT, self.DELTA):
                assert spec["custom"] == "empirical_k_n", label
            else:
                assert spec["customdata"] == trades, label
        heat, _ = _first_figure(section)
        assert heat[0]["customdata"] == trades

    def test_a_band_without_a_calibration_is_blank(self):
        sweep = dataclasses.replace(self._sweep(), calibrations_by_band={
            _Grid.BANDS[0]: _scn_calibration()})
        section = _section_scenario_explorer(sweep)
        metrics = self._metrics(section)
        assert metrics[self.KHAT]["z"] == [[0.60, 0.60], [None, None], [None, None]]
        # ... and so is its k-hat − k
        assert metrics[self.DELTA]["z"] == [[-0.05, -0.15], [None, None], [None, None]]
        heat, _ = _first_figure(section)
        assert heat[0]["z"] == _scn_cap(section)["matrices"]["mean_per_trade"]

    def test_the_table_lists_every_band(self):
        section = _section_scenario_explorer(self._sweep())
        assert "Empirical k&#770; by spread band" in section
        table = section[section.index("Empirical k&#770; by spread band"):]
        table = table[:table.index("</table>")]
        rows = re.findall(r"<tr style='border-bottom[^>]*>(.*?)</tr>", table)
        cells = [re.findall(r"<td[^>]*>(.*?)</td>", r) for r in rows]
        labels = [html_label for html_label, *_ in cells]
        assert labels == [dashboard._row_label(b).replace("&", "&amp;")
                          for b in _Grid.BANDS]
        assert cells[0][1:] == ["10", "0.1200", "0.2000", "0.600"]
        assert cells[1][1:] == ["4", "0.2700", "0.3000", "0.900"]
        assert cells[2][1:] == ["0", "0.0000", "0.0000", "—"]


def _khat_table_rows(section_html: str, title: str) -> list[list[str]]:
    """The cells of the "Empirical k̂ by spread band" table whose bold title
    is `title` — each row's band label first."""
    table = section_html[section_html.index(f"<b>{title}</b>"):]
    table = table[:table.index("</table>")]
    return [re.findall(r"<td[^>]*>(.*?)</td>", r)
            for r in re.findall(r"<tr style='border-bottom[^>]*>(.*?)</tr>", table)]


class TestScenarioExplorerTierFloors:
    """The explorer's tier-floors-off view: its own Tier floors select, a
    hidden tier-off k-hat table and a packed block per size cap the run
    simulated with the tiers off ("scn-off-cap-<i>") — offered only when the
    run carries a complete tier-off family; every figure in it over the grid
    with the tier floors off — a binding band's own tier-off points, every
    other band's tier-on ones — and each off cell shipped once."""

    OFF_TITLE = "Empirical k&#770; by spread band — tier floors off"
    CURVE = f"Equity curve — selected band, k and size cap — {_TS_LABEL}"

    def test_without_a_family_there_is_no_off_view(self):
        # (#68's test of the same name, on one heatmap and packed blocks.)
        section = _Grid.section()
        assert '<div id="scn-khat-on">' in section
        for absent in ('id="scn-tier-select"', 'id="scn-khat-off"', 'id="scn-off-cap-',
                       "Tier floors off:", "tier floors off"):
            assert absent not in section, absent
        assert list(_scn_blocks(section)) == ["scn-data", "scn-cap-0"]
        data = _scn_data(section)
        for key in ("tier_binds", "off_caps", "tier_labels", "band_labels_off",
                    "band_options_off", "calibration_by_band_off", "matrices_off",
                    "tier_phrases", "off_missing"):
            assert data[key] is None, key
        # The band select's texts, exactly as rendered
        assert data["band_options"] == [text for _, _, text in _options(section,
                                                                         "scn-band-select")]
        assert data["band_options"] == [
            "max(tier,0)-1", "max(tier,0.3)-0.6 (primary)", "max(tier,0.35)-0.8"]
        # Two charts: the heatmap and the curve, whose title the base block
        # carries for the one setting it has
        charts = _page_elements(section)["charts"]
        assert list(charts) == ["scn-heatmap", "scn-equity"]
        assert data["curve_titles"] == [charts["scn-equity"]["layout"]["title"]["text"], None]
        assert data["curve_titles"][0] == self.CURVE

    def test_an_incomplete_family_is_no_off_view(self):
        # A band the tiers bind at that the family does not name: never shown
        # as a tier-off view, and never its tier-on cells relabelled as one
        sweep = dataclasses.replace(_Grid.sweep_tiers(), tier_off_calibrations_by_band={})
        section = _section_scenario_explorer(sweep)
        assert 'id="scn-tier-select"' not in section and 'id="scn-off-cap-0"' not in section
        assert _scn_data(section)["tier_binds"] is None

    def test_the_off_view_needs_no_bar(self):
        # (Replaces #68's test_no_off_view_unless_the_page_offers_one: #68's
        # explorer followed the bar's select alone, so it withheld its off
        # view from a page whose bar had none; this explorer has a Tier floors
        # select of its own, so the view is reachable without the bar, and
        # the direct call draws it.) Without the family it is byte for byte
        # the section it always was
        section = _section_scenario_explorer(_Grid.sweep_tiers())
        assert 'id="scn-tier-select"' in section
        assert list(_scn_blocks(section)) == ["scn-data", "scn-cap-0", "scn-off-cap-0"]
        stripped = dataclasses.replace(_Grid.sweep_tiers(), tier_off_scenarios=[],
                                       tier_off_calibrations_by_band={})
        assert _section_scenario_explorer(stripped) == _Grid.section()

    def test_the_off_view_is_a_select_a_hidden_table_and_a_packed_block(self):
        # (Replaces #68's test_the_off_block_is_drawn_hidden_after_the_tier_on_one:
        # one heatmap, redrawn for the setting, instead of a second hidden
        # block of banner, heatmap and table.)
        section = _section_scenario_explorer(_Grid.sweep_tiers())
        # The Tier floors select sits after the band's, as on the filter bar,
        # rendered disabled with the bar's options, title and values
        band_at = section.index('id="scn-band-select"')
        tier_at = section.index('id="scn-tier-select"')
        assert band_at < tier_at < section.index('id="scn-k-select"')
        tier = re.search(r'<select id="scn-tier-select"([^>]*)>(.*?)</select>', section)
        assert tier.group(1) == (' disabled autocomplete="off" title="'
                                 f'{html.escape(dashboard._TIER_SELECT_TITLE)}"')
        assert tier.group(2) == (
            f'<option value="on" selected>{html.escape(dashboard._TIER_OPTION_ON)}</option>'
            f'<option value="off">{html.escape(dashboard._TIER_OPTION_OFF)}</option>')
        # The two k-hat tables: the tier-on one shown, the off one hidden after it
        on_at = section.index('<div id="scn-khat-on">')
        off_at = section.index('<div id="scn-khat-off" style="display:none">')
        assert on_at < off_at < band_at
        assert section.index(self.OFF_TITLE) > off_at
        # One heatmap and the curve; every "first" pin keeps reading the heatmap
        assert list(_page_elements(section)["charts"]) == ["scn-heatmap", "scn-equity"]
        heat, _ = _first_figure(section)
        assert heat[0]["y"] == [dashboard._row_label(b) for b in _Grid.BANDS]
        # The off block after the cap's, both before the script that unpacks them
        script = section.index("function unpack(el)")
        assert section.index('id="scn-cap-0"') < section.index('id="scn-off-cap-0"') < script

    def test_the_off_heatmap_reads_the_tier_off_grid(self):
        section = _section_scenario_explorer(_Grid.sweep_tiers())
        data, on, off = _scn_data(section), _scn_cap(section), _scn_blocks(section)[
            "scn-off-cap-0"]
        # With the tiers off the floor is the only floor: the bare labels
        assert data["band_labels_off"] == ["0-1", "0.3-0.6", "0.35-0.8"]
        assert data["band_labels"] == [dashboard._row_label(b) for b in _Grid.BANDS]
        # Every title the off heatmap shows is the tier-on one naming the setting
        assert off["titles"] == [t + " — tier floors off" for t in on["titles"]]
        assert off["titles"][0] == (
            "Mean per trade (equal stake) by spread band x k, cap not recorded — "
            f"{_TS_LABEL} — tier floors off")
        # The bands the tiers never bind at are their tier-on rows
        for key, z in off["matrices"].items():
            assert z[1:] == on["matrices"][key][1:], key
        z = off["matrices"]
        assert z["trades"][0] == [30, 31]
        assert z["total_return"][0] == pytest.approx([-0.05, 0.10])
        assert z["h1_return"][0] == [0.07, -0.02]
        assert z["h2_return"][0] == [0.01, 0.04]
        ts_off = [pt for pt in _Grid.off_points() if pt.population == "time_series"]
        assert z["mean_per_trade"][0] == pytest.approx(
            [dashboard._point_kpis(pt)["mean_per_trade"] for pt in ts_off])
        # Band 0's k-hat is its tier-off calibration's; the rest the tier-on ones
        assert data["matrices_off"]["empirical_k"] == [[0.50, 0.50], [0.60, 0.60],
                                                       [None, None]]
        assert data["matrices_off"]["empirical_k_n"] == [[12, 12], [10, 10], [None, None]]
        assert data["matrices_off"]["khat_minus_k"] == [[-0.15, -0.25], [-0.05, -0.15],
                                                        [None, None]]
        # ... and the tier-on heatmap is untouched by the family
        assert on["matrices"]["trades"] == [[1, 2], [3, 4], [5, 6]]
        assert data["matrices"]["empirical_k"][0] == [0.60, 0.60]

    def test_the_off_banner_measures_the_off_grid(self):
        from scipy.stats import spearmanr
        section = _section_scenario_explorer(_Grid.sweep_tiers())
        on_banner = _scn_cap(section)["banner"]
        off_banner = _scn_blocks(section)["scn-off-cap-0"]["banner"]
        assert off_banner.index(dashboard._SCENARIO_TIER_OFF_LEAD) < off_banner.index(
            "band x k cells computed")
        # 6 cells: band 0's tier-off points (8, every population at both k)
        # and bands 1-2's tier-on ones (4 + 4 + 4 + 3, cell 5 has no ladder)
        assert "<b>6 band x k cells computed</b> (23 scenario points" in off_banner
        # Time-series returns -5%, +10% (band 0, tiers off), +1%, +2%, +3%, -1%
        assert "66.7% of the cells with a measurable return" in off_banner
        h1 = [*_Grid.OFF_H1, *_Grid.H1[2:]]
        h2 = [*_Grid.OFF_H2, *_Grid.H2[2:]]
        rho = spearmanr(h1, h2).statistic
        assert f"{rho:+.3f}" == "-0.580"
        assert "H1 vs H2: -0.580." in off_banner
        assert "The best of 6 correlated cells overstates what you should expect." in off_banner
        # The tier-on banner is today's, and says nothing of the tiers
        assert "<b>6 band x k cells computed</b> (22 scenario points" in on_banner
        assert "60.0% of the cells with a measurable return" in on_banner
        assert "H1 vs H2: -0.116." in on_banner
        assert "Tier floors off" not in on_banner
        # The page as rendered shows the tier-on banner
        assert f'<div id="scn-banner">{on_banner}</div>' in section

    def test_the_off_banner_says_what_off_replaces_and_what_still_applies(self):
        # Off replaces the entry threshold max(tier, floor) with the floor —
        # nothing else: the band's ceiling, the gap cap, the positive-spread
        # rule, the fee check and the Kelly gate all still apply. The tiers
        # and the gap cap are named from config, never as literals
        lead = dashboard._SCENARIO_TIER_OFF_LEAD
        assert ("each band's own floor replaces max(tier, floor) as the time-series entry "
                f"threshold, without the {dashboard._TIER_FLOORS} deadline-gap tier floors: "
                "pB − pA of at least the floor, and pA + nB of at most 1 − the floor.") in lead
        assert (f"The band's ceiling, the {config.MAX_DEADLINE_GAP_DAYS}-day deadline-gap cap, "
                "the positive-spread rule (pB above pA), the fee check and the Kelly gate "
                "still apply.") in lead
        assert dashboard._TIER_FLOORS == (
            f"{config.MIN_PRICE_DIFF_SHORT_GAP:.2f}/{config.MIN_PRICE_DIFF_LONG_GAP:.2f}")
        assert ("Live trading follows the saved live defaults' tier floors (main.py "
                "--tier-floors / --no-tier-floors overrides them for one run).") in lead

    def test_the_data_blocks_ship_each_off_cell_once(self):
        # (#68's test_the_data_block_ships_each_off_cell_once, over the base
        # block and the packed off block.)
        section = _section_scenario_explorer(_Grid.sweep_tiers())
        data, cap, off = (_scn_data(section), _scn_cap(section),
                          _scn_blocks(section)["scn-off-cap-0"])
        pop = {name: i for i, name in enumerate(data["populations"])}
        assert data["tier_binds"] == [True, False, False]
        assert data["off_caps"] == [True] and data["tier_labels"] == ["on", "off"]
        # A band the tiers never bind at ships no off cells or calibration:
        # the script reads its tier-on ones there
        assert off["cells"][1] is None and off["cells"][2] is None
        assert data["calibration_by_band_off"][1:] == [None, None]
        # ... and the off block carries no same-title row: no tier floor
        # reaches a same-title pair, so the cap's own row is read
        assert "same_title" not in off and cap["same_title"]["trades"] == 7
        row = off["cells"][0]
        for name, first in (("all", 40), ("time_series", 30), ("ladder", 50), ("cross", 60)):
            assert [cell[pop[name]]["trades"] for cell in row] == [first, first + 1], name
        assert [cell[pop["time_series"]]["total_return"] for cell in row] == pytest.approx(
            [-0.05, 0.10])
        assert [cell[pop["time_series"]]["h1_return"] for cell in row] == _Grid.OFF_H1
        assert row[1][pop["time_series"]]["top_event"] == "EVT-OFF"
        # The curve travels with the time-series cell, as on the tier-on grid
        curve = _expand(row[0][pop["time_series"]]["equity"], len(data["dates"]))
        assert [v for v in curve if v is not None][-1] == _Grid.OFF_TS_FINALS[0]
        assert "equity" not in row[0][pop["all"]]
        # The tier-on cells are untouched by the family
        assert cap["cells"][0][0][pop["all"]]["trades"] == 1
        cal = data["calibration_by_band_off"][0]
        assert [r["label"] for r in cal] == ["16-30d", "POOLED"]
        assert [r["n"] for r in cal] == [7, 12]
        # Labelled with the floor alone, a floor-0 calibration's buckets carry
        # tier 0.0, which ships as null ("—", no floor to print) — like the
        # pooled row, never 0.00
        assert [r["tier"] for r in cal] == [None, None]
        assert data["band_options"] == [
            "max(tier,0)-1", "max(tier,0.3)-0.6 (primary)", "max(tier,0.35)-0.8"]
        assert data["band_options_off"] == ["0-1", "0.3-0.6 (primary)", "0.35-0.8"]
        # The curve's title under each setting: the one Python draws, and
        # the same naming the setting, as the off heatmap's titles do
        assert data["curve_titles"] == [self.CURVE, self.CURVE + " — tier floors off"]
        # The words the status line fills in, Python's
        assert data["tier_phrases"] == {"on": "", "off": " with the tier floors off"}
        assert data["off_missing"] == dashboard._SCENARIO_TIER_OFF_MISSING

    def test_the_off_khat_table_repeats_the_inert_bands_rows(self):
        section = _section_scenario_explorer(_Grid.sweep_tiers())
        on = _khat_table_rows(section, "Empirical k&#770; by spread band")
        off = _khat_table_rows(section, self.OFF_TITLE)
        assert [r[0] for r in on] == [dashboard._row_label(b) for b in _Grid.BANDS]
        assert [r[0] for r in off] == ["0-1", "0.3-0.6", "0.35-0.8"]
        assert off[0][1:] == ["12", "0.1500", "0.3000", "0.500"]
        assert on[0][1:] == ["10", "0.1200", "0.2000", "0.600"]
        assert off[1:] == [["0.3-0.6", *on[1][1:]], ["0.35-0.8", *on[2][1:]]]
        # The first such table on the page is the tier-on one
        assert section.index("Empirical k&#770; by spread band</b>") < section.index(
            self.OFF_TITLE)

    def test_the_script_words_nothing_of_the_off_view(self):
        # Every word the off view shows is Python's — the banner's lead, the
        # titles' suffix, the band options, the curve's titles and the status
        # line's phrase and template in the base block — and the script only
        # draws the blocks' and the base block's texts
        js = dashboard._SCENARIO_EXPLORER_JS
        for phrase in ("Tier floors off", "tier floors off", "floor alone", "never bind",
                       "(primary)", "max(tier", "Equity curve", "not simulated"):
            assert phrase not in js, phrase
        assert "off ? D.band_options_off : D.band_options" in js
        assert "D.curve_titles[offShown() ? 1 : 0]" in js
        assert "offShown() ? D.band_labels_off : D.band_labels" in js
        assert "D.tier_phrases[tier]" in js and "D.off_missing" in js

    def test_a_point_stamped_tier_floors_off_is_never_a_tier_on_cell(self):
        # Points stamped tier-off among the tier-on scenarios — at a gridded
        # band and k, listed after that cell's own points (so a grid that
        # ignored the stamp would overwrite them), at a band no tier-on point
        # has and at a k no tier-on point has — change nothing
        sweep = _Grid.sweep()
        strays = [_scn_point(_Grid.BANDS[1], 0.75, population, [_scn_trade()] * 99,
                             tier_floors=False)
                  for population in dashboard._SCENARIO_POPULATIONS]
        strays += [_scn_point((0.2, 0.6), 0.75, tier_floors=False),
                   _scn_point(_Grid.BANDS[0], 0.40, tier_floors=False)]
        mixed = dataclasses.replace(sweep, scenarios=[*sweep.scenarios, *strays])
        assert _section_scenario_explorer(mixed) == _Grid.section()

    def test_only_a_binding_bands_tier_off_points_fill_the_off_grid(self):
        # A point in the family not stamped tier-off is never read as one,
        # and a tier-off point at a band the tiers never bind at (whose off
        # row IS its tier-on row) is never written into the off grid.
        # _with_tier_off files points first-wins, so the unstamped stray at
        # the binding BANDS[0] is listed BEFORE the real points: a grid that
        # ignored the stamp rule would file it in place of the real point.
        # The explorer gates on the binding rule three times over
        # (_with_tier_off, the walk and _ExplorerVisitor._assemble — each
        # pinned alone by TestTierOffGrid), so the non-binding stray at
        # BANDS[1] is also checked at the first of them directly: it files no
        # off cell. Measured on a section that DRAWS an off view (its block
        # and its select are there), so the comparison is not vacuous
        sweep = _Grid.sweep_tiers()
        unstamped = _scn_point(_Grid.BANDS[0], 0.75, "time_series", [_scn_trade()] * 98)
        unbound = _scn_point(_Grid.BANDS[1], 0.75, "time_series", [_scn_trade()] * 99,
                             tier_floors=False)
        mixed = dataclasses.replace(
            sweep, tier_off_scenarios=[unstamped, *sweep.tier_off_scenarios, unbound])
        section = _section_scenario_explorer(sweep)
        assert 'id="scn-tier-select"' in section and 'id="scn-off-cap-0"' in section
        assert _section_scenario_explorer(mixed) == section
        off = _scn_blocks(section)["scn-off-cap-0"]
        assert off["matrices"]["trades"][:2] == [[30, 31], [3, 4]]
        grid = dashboard._with_tier_off(dashboard._eager_source(mixed), mixed)
        assert grid.tier_binds[:2] == (True, False)
        assert grid.off_cell(_Grid.BANDS[1], 0.75) == {}
        [real] = grid.off_cell(_Grid.BANDS[0], 0.75).values()
        assert real["time_series"] is not unstamped
        assert real["time_series"] in sweep.tier_off_scenarios

    def test_a_real_floor_0_tier_off_calibration_ships_its_tiers_as_null(self, monkeypatch):
        # run_backtest_sweep itself, over TestPrepareEntriesGolden's fixture
        # (each band at the primary k): with the tiers off, the band at floor
        # 0 labels its buckets 0.0 — shipped as null, like the pooled row —
        # while a binding band at floor 0.2 labels them 0.2
        from . import test_backtester as tb
        golden = tb.TestPrepareEntriesGolden()
        golden._patch(monkeypatch)
        sweep = backtester.run_backtest_sweep(
            hist_client=MagicMock(), live_client=MagicMock(), start_date=golden._START,
            initial_balance=10_000.0, sweep=False, same_event_ladders=True,
            band_sweep=True, tier_off_sweep=True)
        data = _scn_data(_section_scenario_explorer(sweep))
        bands = [tuple(b) for b in data["bands"]]
        for band, shipped_tier in (((0.0, 1.0), None), ((0.2, 1.0), 0.2)):
            cal = sweep.tier_off_calibrations_by_band[band]
            assert cal is not None and cal.buckets, band
            assert {b.tier for b in cal.buckets} == {band[0]}, band
            rows = data["calibration_by_band_off"][bands.index(band)]
            assert [r["tier"] for r in rows[:-1]] == [shipped_tier] * len(cal.buckets), band
            assert rows[-1]["tier"] is None                 # the pooled row
        assert data["tier_binds"] == [backtester._tier_floors_bind(b) for b in bands]


# ═══ The page-wide filter: spread band x tier floors x k x size cap x category x tag ═══

_FLT_START = date(2026, 1, 5)
_FLT_SERIES = {
    "KXNCAAMBGAME": ("Sports", ("Basketball",)),
    "KXNHLHART": ("Sports", ("Hockey",)),
    "KXBRENTW": ("Commodities", ("Oil & Gas", "Energy")),
}


def _ftrade(event_ticker: str, pair_type: str, entry: date, exit_: date,
            profit: float, *, ladder: bool = False, title: str = "Q?") -> BacktestTrade:
    """_typed_trade with an event ticker (its series decides the category and
    tag), a title and the fallback category "Other" (what infer_category gives
    these made-up tickers), its payoff set so profit = payoff - cost - fees."""
    return dataclasses.replace(
        _typed_trade(pair_type, ladder, entry, exit_, profit),
        event_ticker=event_ticker, title_a=title, ticker_a=f"{event_ticker}-A",
        category="Other")


def _flt_trades() -> list[BacktestTrade]:
    return [
        _ftrade("KXNCAAMBGAME-1", "same_title", date(2026, 1, 6), date(2026, 1, 8), 30.0),
        _ftrade("KXBRENTW-1", "same_title", date(2026, 1, 6), date(2026, 1, 9), -12.0),
        _ftrade("KXNHLHART-27", "time_series", date(2026, 1, 12), date(2026, 1, 20), 8.0,
                ladder=True),
        _ftrade("KXNCAAMBGAME-2", "same_title", date(2026, 1, 13), date(2026, 1, 14), -5.0),
        _ftrade("KXOTHER-1", "time_series", date(2026, 1, 13), date(2026, 1, 16), 4.0),
    ]


# Implied gaps whose float sum depends on the order they are added in: the
# carried tuple's pooled row cannot be matched by a reordered copy of it
_FLT_IMPLIED = (0.1, 0.2, 0.3, 0.35)


def _flt_observations(filed: list[tuple[str, str]]) -> tuple:
    """One observation per (event ticker, fallback category)."""
    return tuple(backtester.CalibrationObservation(
        5, _FLT_IMPLIED[i % len(_FLT_IMPLIED)], i % 2 == 0, ticker, category)
        for i, (ticker, category) in enumerate(filed))


def _flt_calibration(filed: list[tuple[str, str]]) -> IntervalCalibration:
    obs = _flt_observations(filed)
    return IntervalCalibration(pooled=backtester._calibration_bucket("POOLED", 0.0, obs),
                               buckets=[], excluded_premise_violations=0, observations=obs)


def _flt_sweep(size_cap: float | None = 0.2) -> BacktestSweep:
    """Three bands at k 0.75 — the primary, 0-1, holding every trade; 0.3-0.6
    holding only the same-title ones; 0.3-1 an EQUAL copy of 0.3-0.6's list
    — plus one more k at the primary band (0.60) and at 0.3-0.6 (1.00). The
    bar's k axis is therefore 0.60 / 0.75 / 1.00 and the grid is ragged: (0-1,
    1.00), (0.3-0.6, 0.60) and 0.3-1 at 0.60 and 1.00 were never simulated.
    The k 0.60 and 1.00 points trade EQUAL lists (one same-title trade each),
    which the bar still keeps apart: a list's Kelly scatter is priced at its
    k. Two points the bar must never read: another population at 0.3-0.6 and
    k 0.75, and a point with no band. Every point is stamped `size_cap` — the
    run's own cap, 0.2 as production stamps it (the summary then reads "20%
    cap per trade"; None pins "cap not recorded"). Only the primary band
    carries a k-hat population; KXSPACEX has no trade and no map entry, so its
    observation files under its fallback category, "Science", which no trade
    carries."""
    trades = _flt_trades()
    st = [t for t in trades if t.pair_type == "same_title"]
    curve = backtester._build_equity_curve

    def point(k, listed, band, population="all"):
        return SweepPoint(k=k, trades=listed, equity_df=curve(listed, _FLT_START, 1000.0),
                          spread_band=band, population=population, size_cap=size_cap)

    primary = point(0.75, trades, (0.0, 1.0))
    others = [
        point(0.75, st, (0.3, 0.6)),
        point(0.75, list(st), (0.3, 1.0)),
        point(0.60, st[:1], (0.0, 1.0)),
        point(1.00, st[:1], (0.3, 0.6)),
        point(0.75, st[:1], (0.3, 0.6), population="time_series"),
        point(0.75, st[:1], None),
    ]
    cals = {(0.0, 1.0): _flt_calibration([("KXNHLHART-27", "Other"),
                                          ("KXSPACEX-14", "Science"),
                                          ("KXOTHER-1", "Other")]),
            (0.3, 0.6): None, (0.3, 1.0): None}
    return BacktestSweep(primary=primary, points=[primary], calibration=cals[(0.0, 1.0)],
                         label_coverage=_scn_coverage(), scenarios=[primary, *others],
                         calibrations_by_band=cals, same_event_ladders=True)


# The flt fixture's primary scenario: band 0-1, k 0.75, the one cap
_FLT_PRIMARY = (0, 1, 0)

# One packed block of the page: <script type="text/plain" id=... data-encoding=...>
_PACKED = re.compile(
    r'<script type="text/plain" id="([^"]+)" data-encoding="gzip\+base64">([^<]*)</script>')


def _decode_block(body: str) -> dict:
    """A packed block's body, inflated as the script inflates it (base64, then
    gzip) and parsed strictly (a NaN/Infinity token fails the test)."""
    raw = gzip.decompress(base64.b64decode(body, validate=True))
    return json.loads(raw.decode("utf-8"), parse_constant=_fail_on_constant)


def _unpack(block: str) -> dict:
    """One packed <script> element (_packed_json_script's output), decoded."""
    return _decode_block(_PACKED.fullmatch(block).group(2))


def _packed_blocks(page: str) -> dict[str, dict]:
    """Every packed block on a page, decoded, by element id."""
    return {element_id: _decode_block(body) for element_id, body in _PACKED.findall(page)}


def _flt_walk(source, trades, curve, k_used=0.75, series=_FLT_SERIES, pooled_k=None):
    """Walk a grid as generate_dashboard does: (the grid walked, the base
    block, {chunk id: chunk}). The base block carries the interval-discount
    section's data from the same walk ("kd") when the grid records a band,
    as the page's does."""
    walked, chunker, kd = dashboard._build_filter_grid(
        source, trades, curve, k_used, _FLT_START, 1000.0, series, pooled_k=pooled_k)
    base = dashboard._filter_payload(
        walked, chunker, _FLT_START, 1000.0, series,
        kd=kd.payload() if walked.bands != (None,) else None)
    return walked, base, {cid: _unpack(block) for cid, block in enumerate(chunker.chunks)}


# _FLT_SERIES plus the series only the tier-off run trades
_FLT_SERIES_TIERS = {**_FLT_SERIES, "KXRAIN": ("Climate", ("Rain",))}


def _flt_sweep_tiers():
    """_flt_sweep() plus a tier-floors-off family for the one band a tier
    binds at: the primary (0.0, 1.0), whose floor 0 sits below both tiers —
    the other two bands' 0.3 floor sits at or above both, so they were never
    simulated again. The family's "all" point at the primary k holds the
    primary's trades plus one time-series trade the tiers held back
    (KXRAIN-1, "Climate" under _FLT_SERIES_TIERS); its "all" point at k 0.60
    holds that k's own one trade, the same list its tier-on point at 0.60
    trades (so the two share a chunk). After them come points the bar must
    never read as a band's "all" run — another population at the primary k,
    and an "all" point at the primary k NOT stamped tier-off — each listed
    later, so a filter that ignored the population or the stamp would keep
    it. Every point carries the run's own size cap (the one cap the family
    is simulated at). Its tier-off calibration files KXRAIN and KXNHLHART,
    where the tier-on one files KXNHLHART, KXSPACEX and KXOTHER."""
    sweep = _flt_sweep()
    rain = _ftrade("KXRAIN-1", "time_series", date(2026, 1, 12), date(2026, 1, 15), 6.0)
    trades = sorted([*_flt_trades(), rain], key=lambda t: t.entry_date)
    curve = backtester._build_equity_curve
    cap = sweep.primary.size_cap
    off = SweepPoint(k=0.75, trades=trades, equity_df=curve(trades, _FLT_START, 1000.0),
                     spread_band=(0.0, 1.0), population="all", tier_floors=False, size_cap=cap)
    at_060 = [p for p in sweep.scenarios if p.k == 0.60 and p.spread_band == (0.0, 1.0)][0]
    one = curve(trades[:1], _FLT_START, 1000.0)
    others = [
        SweepPoint(k=0.60, trades=list(at_060.trades), equity_df=at_060.equity_df,
                   spread_band=(0.0, 1.0), population="all", tier_floors=False, size_cap=cap),
        SweepPoint(k=0.75, trades=trades[:1], equity_df=one, spread_band=(0.0, 1.0),
                   population="time_series", tier_floors=False, size_cap=cap),
        SweepPoint(k=0.75, trades=trades[:1], equity_df=one, spread_band=(0.0, 1.0),
                   population="all", size_cap=cap),
    ]
    cal = _flt_calibration([("KXRAIN-1", "Other"), ("KXNHLHART-27", "Other")])
    return dataclasses.replace(sweep, tier_off_scenarios=[off, *others],
                               tier_off_calibrations_by_band={(0.0, 1.0): cal})


def _flt_payload(sweep=None, series=_FLT_SERIES):
    """(sweep, the grid walked, the base block, {chunk id: chunk}) — the
    page-wide filter as generate_dashboard builds it, at k 0.75, its trades
    filed by `series` (_FLT_SERIES_TIERS for a tier-off family's)."""
    sweep = sweep or _flt_sweep()
    trades, curve = sweep.primary.trades, sweep.primary.equity_df
    source = dashboard._grid_source(sweep, trades, curve, 0.75)
    return (sweep, *_flt_walk(source, trades, curve, series=series,
                              pooled_k=dashboard._pooled_k(sweep)))


def _expand(sparse: list, n: int) -> list:
    """The filter script's expand(), in Python: each value holds until the
    next change point."""
    out, j, cur = [], 0, None
    for i in range(n):
        while j < len(sparse) and sparse[j][0] <= i:
            cur = sparse[j][1]
            j += 1
        out.append(cur)
    return out


def _chunk_at(base: dict, chunks: dict, bi: int, ki: int, ci: int) -> dict:
    """The chunk behind one scenario (grid[band][k][cap])."""
    return chunks[base["grid"][bi][ki][ci]]


def _view(base: dict, chunks: dict, bi: int, ki: int, ci: int, key: str) -> dict:
    """One view of one scenario's list."""
    return _chunk_at(base, chunks, bi, ki, ci)["list"]["views"][key]


def _key(base: dict, category: str, tag: str | None = None) -> str:
    ci = base["categories"].index(category)
    if tag is None:
        return f"c{ci}"
    return f"s{base['subcats'].index([ci, tag])}"


def _phrase(base: dict, bi: int, ki: int, ci: int) -> str:
    """A scenario in the summary's words, from the base block, as the script
    builds it (D.text.scenario)."""
    return dashboard._scenario_phrase(base["bands"][bi]["where"], base["ks"][ki]["text"],
                                      base["caps"][ci]["text"])


def _phrase_off(base: dict, bi: int, ki: int, ci: int) -> str:
    """A scenario with the tier floors off, in the summary's words: the
    template the script fills, its band named by the base block's
    "bands_off" (_tier_off_where)."""
    return dashboard._scenario_phrase(base["bands_off"][bi]["where"],
                                      base["ks"][ki]["text"], base["caps"][ci]["text"])


def _row_html(base: dict, chunk: dict, pairs: list) -> list[str]:
    """Best/worst rows as the script joins them: head from the base block's
    shared table, tail from the chunk's own strings."""
    return [base["rows"][h] + chunk["strings"][t] for h, t in pairs]


class TestFilterGrid:
    """Which scenario each band x k x size cap the bar offers shows, and which
    chunk carries it."""

    def test_the_axes_are_every_band_k_and_cap_the_sweep_simulated(self):
        _, source, base, _ = _flt_payload()
        assert source.bands == ((0.0, 1.0), (0.3, 0.6), (0.3, 1.0))
        assert source.ks == (0.6, 0.75, 1.0)
        assert source.caps == (0.2,)
        assert source.primary == _FLT_PRIMARY and base["primary"] == list(_FLT_PRIMARY)
        assert [b["label"] for b in base["bands"]] == [
            dashboard._row_label(b) for b in source.bands]
        assert [(k["label"], k["text"], k["value"]) for k in base["ks"]] == [
            ("k = 0.60", "k = 0.60", 0.6), ("k = 0.75", "k = 0.75", 0.75),
            ("k = 1.00", "k = 1.00", 1.0)]
        assert base["caps"] == [{"label": "20%", "text": "20% cap per trade", "value": 0.2}]

    def test_a_scenario_the_sweep_never_simulated_is_null(self):
        _, _, base, chunks = _flt_payload()
        grid = base["grid"]
        assert [[grid[b][k][0] is None for k in range(3)] for b in range(3)] == [
            [False, False, True], [True, False, False], [True, False, True]]
        # Another population never makes a cell: 0.3-0.6 at k 0.75 is its
        # "all" point (3 trades), not the one-trade "time_series" point
        assert _view(base, chunks, 1, 1, 0, "all")["n"] == 3
        # Each extra k is its own run
        assert _view(base, chunks, 0, 0, 0, "all")["n"] == 1

    def test_the_primary_scenario_is_what_the_page_renders(self):
        # The page's own trades and curve (not the sweep's point looked up
        # again) build the primary chunk — chunk 0, the first built. Both
        # differ in CONTENT from the sweep's primary here (one trade fewer,
        # another with a different profit, the curve shifted), so a chunk
        # built from the sweep's trades or curve cannot pass
        sweep = _flt_sweep()
        page_trades = [dataclasses.replace(t, profit=t.profit + 50.0,
                                           profit_ratio=t.profit_ratio + 0.5) if i == 0 else t
                       for i, t in enumerate(sweep.primary.trades[:4])]
        page_curve = sweep.primary.equity_df.assign(
            portfolio_value=sweep.primary.equity_df["portfolio_value"] + 1.0)
        source = dashboard._grid_source(sweep, page_trades, page_curve, 0.75)
        _, base, chunks = _flt_walk(source, page_trades, page_curve)
        assert base["grid"][0][1][0] == 0
        chunk = _chunk_at(base, chunks, *_FLT_PRIMARY)
        view = chunk["list"]["views"]["all"]
        assert view["n"] == len(page_trades) == 4
        assert chunk["list"]["ret"] == pytest.approx(
            [t.profit_ratio * 100 for t in page_trades])
        best, worst = dashboard._best_and_worst(page_trades)
        assert _row_html(base, chunk, view["best"]) == [
            dashboard._trade_row(t, dashboard._BEST_ROW_COLOR) for t in best]
        assert _row_html(base, chunk, view["worst"]) == [
            dashboard._trade_row(t, dashboard._WORST_ROW_COLOR) for t in worst]
        assert _expand(view["eq"], len(base["dates"])) == pytest.approx(
            list(page_curve["portfolio_value"]))

    def test_the_primary_band_s_calibration_falls_back_to_the_sweep_s(self):
        sweep, source, _, _ = _flt_payload()
        assert source.calibrations[(0.0, 1.0)] is sweep.calibration
        assert source.calibrations[(0.3, 0.6)] is source.calibrations[(0.3, 1.0)] is None
        # A sweep whose per-band map lacks the primary band still has one
        bare = dataclasses.replace(sweep, calibrations_by_band={})
        source = dashboard._grid_source(bare, bare.primary.trades, bare.primary.equity_df, 0.75)
        assert source.calibrations[(0.0, 1.0)] is sweep.calibration
        assert source.calibrations[(0.3, 0.6)] is None

    def test_equal_lists_share_a_chunk_only_at_the_same_k(self):
        _, _, base, chunks = _flt_payload()
        grid = base["grid"]
        # 0.3-0.6 and 0.3-1 traded equal lists at k 0.75: one chunk
        assert grid[1][1][0] == grid[2][1][0]
        # (0-1, 0.60) and (0.3-0.6, 1.00) traded equal lists at two ks: two
        assert grid[0][0][0] != grid[1][2][0]
        assert len(chunks) == 4
        assert sorted({c for band in grid for ks in band for c in ks if c is not None}) \
            == [0, 1, 2, 3]

    def test_a_point_stamped_tier_floors_off_is_never_a_tier_on_cell(self):
        # (#68's test_a_point_stamped_tier_floors_off_is_never_a_tier_on_run,
        # on the grid.) Tier-off "all" points that land among the tier-on
        # scenarios — one at a band the bar offers, listed after that band's
        # own point (so a grid that ignored the stamp would keep it), one at a
        # band no tier-on point has, one at a k no tier-on point has — are
        # never a tier-on cell: the axes, every cell and every chunk are the
        # ones the sweep without them yields
        sweep = _flt_sweep()
        narrow = sweep.scenarios[1]
        stray = [SweepPoint(k=0.75, trades=narrow.trades[:1], equity_df=narrow.equity_df,
                            spread_band=band, population="all", tier_floors=False,
                            size_cap=0.2)
                 for band in ((0.3, 0.6), (0.2, 0.6))]
        stray.append(SweepPoint(k=0.40, trades=narrow.trades[:1], equity_df=narrow.equity_df,
                                spread_band=(0.0, 1.0), population="all", tier_floors=False,
                                size_cap=0.2))
        mixed = dataclasses.replace(sweep, scenarios=[*sweep.scenarios, *stray])
        source = dashboard._eager_source(mixed)
        assert source.bands == ((0.0, 1.0), (0.3, 0.6), (0.3, 1.0))
        assert source.ks == (0.6, 0.75, 1.0)
        assert source.cell((0.3, 0.6), 0.75)[0.2]["all"] is narrow
        _, _, base, chunks = _flt_payload(mixed)
        _, _, clean_base, clean_chunks = _flt_payload()
        assert base["grid"] == clean_base["grid"] and chunks == clean_chunks
        # ... and without a tier-off family on the sweep they are no off view
        assert base["grid_off"] is None and base["bands_off"] is None

    def test_the_primary_chunk_is_built_first(self):
        # The primary band sorts SECOND here, and a band before it trades the
        # same list at the same k: the shared chunk must keep the primary's
        # own curve (the page's), not the other band's (a curve built later can
        # run a day on)
        st = [t for t in _flt_trades() if t.pair_type == "same_title"]
        page_curve = backtester._build_equity_curve(st, _FLT_START, 1000.0)
        other_curve = page_curve.assign(portfolio_value=page_curve["portfolio_value"] + 1.0)
        primary = SweepPoint(k=0.75, trades=st, equity_df=page_curve,
                             spread_band=(0.3, 0.6), population="all", size_cap=0.2)
        other = SweepPoint(k=0.75, trades=list(st), equity_df=other_curve,
                           spread_band=(0.0, 1.0), population="all", size_cap=0.2)
        sweep = BacktestSweep(primary=primary, points=[primary], calibration=None,
                              scenarios=[primary, other])
        _, source, base, chunks = _flt_payload(sweep)
        assert source.primary == (1, 0, 0) and base["grid"] == [[[0]], [[0]]]
        assert _expand(_view(base, chunks, 0, 0, 0, "all")["eq"], len(base["dates"])) \
            == pytest.approx(list(page_curve["portfolio_value"]))

    def test_no_sweep_or_no_recorded_band_is_one_unlabelled_scenario(self):
        trades = _flt_trades()
        curve = backtester._build_equity_curve(trades, _FLT_START, 1000.0)
        source = dashboard._grid_source(None, trades, curve, None)
        assert (source.bands, source.ks, source.caps, source.primary) == (
            (None,), (None,), (None,), (0, 0, 0))
        _, base, chunks = _flt_walk(source, trades, curve, k_used=None)
        assert base["bands"] == [{"label": "not recorded",
                                  "option": "not recorded (primary)",
                                  "where": "the primary spread band (not recorded)",
                                  "value": None}]
        # No band recorded, so no tier-floors-off view either
        assert base["grid_off"] is base["bands_off"] is base["tier_binds"] is None
        assert base["ks"] == [{"label": "not recorded", "text": "k not recorded",
                               "value": None}]
        assert base["caps"] == [{"label": "not recorded", "text": "cap not recorded",
                                 "value": None}]
        assert base["grid"] == [[[0]]] and len(chunks) == 1
        hand_built = BacktestSweep(primary=SweepPoint(k=0.75, trades=trades, equity_df=curve),
                                   points=[], calibration=None)
        source = dashboard._grid_source(hand_built, trades, curve, 0.75)
        assert (source.bands, source.ks, source.caps) == ((None,), (0.75,), (None,))

    def test_a_no_band_sweep_keeps_every_k(self):
        # A hand-built multi-k sweep with no band: one "not recorded" band,
        # and a scenario per swept k
        trades = _flt_trades()
        points = [SweepPoint(k=k, trades=trades[:n],
                             equity_df=backtester._build_equity_curve(trades[:n], _FLT_START,
                                                                      1000.0))
                  for k, n in ((0.9, 3), (0.6, 1), (0.75, 5))]
        sweep = BacktestSweep(primary=points[2], points=points, calibration=None)
        _, source, base, chunks = _flt_payload(sweep)
        assert (source.bands, source.ks, source.primary) == ((None,), (0.6, 0.75, 0.9),
                                                            (0, 1, 0))
        assert [_view(base, chunks, 0, ki, 0, "all")["n"] for ki in range(3)] == [1, 5, 3]


class TestFilterViews:
    """Every view's figures, computed by the helpers the sections render with."""

    def test_the_default_view_is_exactly_the_rendered_page(self):
        sweep, _, base, chunks = _flt_payload()
        trades, curve = sweep.primary.trades, sweep.primary.equity_df
        view = _view(base, chunks, *base["primary"], "all")
        kpis = dashboard._performance_kpis(curve, trades, 1000.0)
        assert view["kpi"] == {key: value for key, _, value, _ in kpis}
        section = dashboard._section_performance(curve, trades, _FLT_START, 1000.0)
        for key, value in view["kpi"].items():
            assert f'id="kpi-{key}" ' in section and f">{value}</div>" in section
        assert view["bench"] == dashboard._strategy_row(curve, 1000.0)
        rel = dashboard._reliability(trades)
        assert view["cal"]["brier"] == f"{rel['brier']:.4f}"
        # The sparse series expand back to the curve the section draws
        n = len(base["dates"])
        assert _expand(view["eq"], n) == pytest.approx(list(curve["portfolio_value"]))
        total, _, drawdown = dashboard._performance_series(curve, trades, 1000.0)
        assert _expand(view["total"], n) == pytest.approx(list(total), abs=1e-4)
        assert _expand(view["dd"], n) == pytest.approx(list(drawdown), abs=1e-4)
        deployed = dashboard._capital_deployed(trades, curve)
        assert _expand(view["dep"], n) == pytest.approx(deployed, abs=0.01)

    def test_categories_and_tags_partition_the_band(self):
        _, _, base, chunks = _flt_payload()
        lst = _chunk_at(base, chunks, *_FLT_PRIMARY)["list"]
        assert base["categories"] == ["Commodities", "Other", "Science", "Sports"]
        assert base["subcats"] == [[0, "Oil & Gas"], [1, "General"], [2, "General"],
                                   [3, "Basketball"], [3, "Hockey"]]
        n_all = lst["views"]["all"]["n"]
        cats = {ci: lst["views"][f"c{ci}"]["n"] for ci in range(4) if f"c{ci}" in lst["views"]}
        assert sum(cats.values()) == n_all == 5
        for ci, n_cat in cats.items():
            subs = [lst["views"][f"s{si}"]["n"] for si, (c, _) in enumerate(base["subcats"])
                    if c == ci and f"s{si}" in lst["views"]]
            assert sum(subs) == n_cat
        # FIRST tag only: KXBRENTW's second tag ("Energy") files nothing
        assert [0, "Energy"] not in base["subcats"]

    def test_a_slice_shows_its_trades_contribution(self):
        _, _, base, chunks = _flt_payload()
        view = _view(base, chunks, *_FLT_PRIMARY, _key(base, "Sports"))
        sports = [t for t in _flt_trades() if t.event_ticker.startswith(("KXNCAA", "KXNHL"))]
        assert view["n"] == len(sports) == 3
        n = len(base["dates"])
        pnl = sum(t.profit for t in sports)
        assert _expand(view["eq"], n)[-1] == pytest.approx(1000.0 + pnl, abs=0.01)
        assert _expand(view["total"], n)[-1] == pytest.approx(pnl / 1000.0 * 100, abs=1e-4)
        assert view["kpi"]["trades"] == "3"
        assert view["kpi"]["total_return"] == f"{pnl / 1000.0:+.1%}"
        # Its type lines are only the types it holds
        assert [label for label, _ in view["types"]] == ["Same-title", "Time-series: ladder"]

    def test_best_and_worst_rows_are_the_slices_own(self):
        _, _, base, chunks = _flt_payload()
        key = _key(base, "Sports", "Basketball")
        chunk = _chunk_at(base, chunks, *_FLT_PRIMARY)
        view = chunk["list"]["views"][key]
        basketball = [t for t in _flt_trades() if t.event_ticker.startswith("KXNCAA")]
        best, worst = dashboard._best_and_worst(basketball)
        # Each row is its shared head and its chunk's own tail, joined: the
        # exact row the diagnostics section renders
        assert _row_html(base, chunk, view["best"]) == [
            dashboard._trade_row(t, dashboard._BEST_ROW_COLOR) for t in best]
        assert _row_html(base, chunk, view["worst"]) == [
            dashboard._trade_row(t, dashboard._WORST_ROW_COLOR) for t in worst]
        for t in best:
            assert dashboard._trade_row(t, "#FFF") == (
                dashboard._trade_row_head(t, "#FFF") + dashboard._trade_row_tail(t))
            # The head does not depend on the size: another n, the same head
            resized = dataclasses.replace(t, n=t.n + 7, profit=t.profit * 2)
            assert dashboard._trade_row_head(resized, "#FFF") == \
                dashboard._trade_row_head(t, "#FFF")
            assert dashboard._trade_row_tail(resized) != dashboard._trade_row_tail(t)
        lst = chunk["list"]
        assert [lst["ret"][i] for i in view["idx"]] == pytest.approx(
            [t.profit_ratio * 100 for t in basketball])

    def test_a_band_without_a_category_has_no_view_for_it(self):
        _, _, base, chunks = _flt_payload()
        narrow = _chunk_at(base, chunks, 1, 1, 0)["list"]["views"]
        assert _key(base, "Sports", "Hockey") not in narrow
        assert base["empty"]["n"] == 0
        n = len(base["dates"])
        assert set(_expand(base["empty"]["eq"], n)) == {1000.0}

    def test_a_category_seen_only_in_khat_observations_is_offered(self):
        _, _, base, chunks = _flt_payload()
        # KXSPACEX has no trade and no map entry: its observation still files
        # it under its fallback category, which the bar must offer for the
        # k-hat breakdown (with no trade view behind it — the trade sections
        # show the empty view)
        assert "Science" in base["categories"]
        assert _key(base, "Science") not in _chunk_at(base, chunks, *_FLT_PRIMARY)["list"]["views"]

    def test_an_empty_view_reports_no_drawdown_date(self):
        # A flat curve never falls, so there is no trough to date
        _, _, base, _ = _flt_payload()
        assert base["empty"]["kpi"]["max_drawdown"] == "0.0%"

    def test_a_slice_curve_is_cut_at_the_pages_axis(self):
        # A page whose own curve stops early (built before midnight UTC, say):
        # a slice's curve, built later and running on to today, is cut at
        # the page's last date for its figures AND its metrics
        trades = _flt_trades()
        full = backtester._build_equity_curve(trades, _FLT_START, 1000.0)
        cut = full.iloc[:-40].reset_index(drop=True)
        sweep = BacktestSweep(
            primary=SweepPoint(k=0.75, trades=trades, equity_df=cut, spread_band=(0.0, 1.0)),
            points=[], calibration=None)
        _, _, base, chunks = _flt_payload(sweep)
        sports = [t for t in trades if t.event_ticker.startswith(("KXNCAA", "KXNHL"))]
        slice_curve = backtester._build_equity_curve(sports, _FLT_START, 1000.0)
        on_axis = slice_curve[pd.to_datetime(slice_curve["date"])
                              <= pd.Timestamp(base["dates"][-1])]
        assert len(on_axis) == len(cut) < len(slice_curve)
        kpis = {k: v for k, _, v, _ in dashboard._performance_kpis(on_axis, sports, 1000.0)}
        uncut = {k: v for k, _, v, _ in dashboard._performance_kpis(slice_curve, sports, 1000.0)}
        view = _view(base, chunks, 0, 0, 0, _key(base, "Sports"))
        assert view["kpi"] == kpis != uncut

    def test_the_sparse_encoding_round_trips_and_gaps_are_none(self):
        axis = pd.DatetimeIndex(pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03",
                                                "2026-01-04", "2026-01-05"]))
        dates = [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 4), date(2026, 1, 5)]
        sparse = dashboard._sparse_on_axis(dates, [1.0, 1.0, 2.5, float("nan")], axis, 2)
        assert sparse == [[0, 1.0], [2, None], [3, 2.5], [4, None]]
        assert _expand(sparse, 5) == [1.0, 1.0, None, 2.5, None]
        assert dashboard._sparse_on_axis(dates, [1.0] * 4, pd.DatetimeIndex([]), 2) == []


# ─── A mix of categories and tags: the data the page works one out from ──────

# Seven series in three categories: Cat0 holds Tag0, Tag3 and Tag6, Cat1 Tag1
# and Tag4, Cat2 Tag2 and Tag5
_MIX_SERIES = {f"KXM{i}": (f"Cat{i % 3}", (f"Tag{i}",)) for i in range(7)}
_MIX_CATEGORIES = sorted({category for category, _ in _MIX_SERIES.values()})
_MIX_SUBCATS = sorted((category, tags[0]) for category, tags in _MIX_SERIES.values())
_MIX_START = date(2026, 1, 5)
_MIX_BALANCE = 10_000.0


def _mix_axes(series: dict) -> tuple[list[str], list[tuple[str, str]]]:
    """A series map's categories and (category, first tag) pairs, each
    sorted: the order a page filed by it lists them in, so a test can tick
    them by these indexes."""
    return (sorted({category for category, _ in series.values()}),
            sorted({(category, tags[0]) for category, tags in series.values()}))


def _mix_trades(seed: int, n: int = 36, *, quoted: bool = False, spread: int = 60,
                series: dict = _MIX_SERIES) -> list[BacktestTrade]:
    """
    Seeded trades over a series map's tags (_MIX_SERIES' seven by default):
    both pair types and every
    trade type, entry quotes on and between the price-bucket edges and in
    several calibration bins, every settlement cell, trades entered and paid
    out on one day, and profits that sit exactly on a rounding tie (12.5,
    0.0). `quoted` gives three trades in four quotes that move while they are
    held (a run valued at market); the fourth stays at cost. Entries fall in
    the first `spread` days from _MIX_START.
    """
    rng = random.Random(seed)
    mark_rng = random.Random(seed + 1000)
    names = sorted(series)
    trades = []
    for i in range(n):
        ticker = rng.choice(names)
        entry = _MIX_START + timedelta(days=rng.randint(0, spread))
        held = rng.choice([0, 1, 3, 8, 21])
        time_series = rng.random() < 0.6
        profit = rng.choice([round(rng.gauss(0.0, 20.0), 2), rng.gauss(0.0, 5.0), 12.5, -12.5, 0.0])
        pA = rng.choice([0.05, 0.2, 0.25, 0.4, 0.55, 0.8, 0.9])
        if time_series:
            pB = min(0.95, round(pA + rng.choice([0.1, 0.15, 0.3]), 2))
            outcomes = rng.choice([("yes", "yes"), ("no", "no"), ("no", "yes")])
        else:
            pB = round(max(0.03, pA - 0.1), 2)
            outcomes = rng.choice([("yes", "yes"), ("no", "no")])
        base = _typed_trade("time_series" if time_series else "same_title", rng.random() < 0.5,
                            entry, entry + timedelta(days=held), profit)
        trades.append(dataclasses.replace(
            base, entry_pA=pA, entry_pB=pB,
            entry_nA=round(1.0 - pA + rng.choice([0.0, 0.02]), 2),
            entry_nB=round(1.0 - pB + rng.choice([0.0, 0.03]), 2),
            outcome_a=outcomes[0], outcome_b=outcomes[1], event_ticker=f"{ticker}-{i}",
            ticker_a=f"{ticker}-{i}A", ticker_b=f"{ticker}-{i}B", title_a=f"Question {i}?",
            category="Other", holding_days=max(1, held), balance_at_entry=_MIX_BALANCE))
    trades.sort(key=lambda t: t.entry_date)
    if quoted:
        trades = [_random_marks(t, mark_rng) if i % 4 else t for i, t in enumerate(trades)]
    return trades


def _mix_list(trades: list[BacktestTrade], *, risk_free=None, curve=None,
              gaps: list | None = None) -> tuple[dict, pd.DatetimeIndex, list[str]]:
    """(_list_payload of `trades` filed by _MIX_SERIES, the axis, the list's
    strings) — the axis is `curve`'s dates, the trades' own curve by default."""
    if curve is None:
        curve = backtester._build_equity_curve(trades, _MIX_START, _MIX_BALANCE)
    axis = pd.DatetimeIndex(pd.to_datetime(list(curve["date"])))
    strings = dashboard._StringTable()
    lst = dashboard._list_payload(
        trades, curve, axis, _MIX_START, _MIX_BALANCE, _MIX_SERIES, 0.75,
        {c: i for i, c in enumerate(_MIX_CATEGORIES)},
        {pair: i for i, pair in enumerate(_MIX_SUBCATS)}, strings,
        heads=dashboard._StringTable(), risk_free=risk_free, mix_gaps=gaps)
    return lst, axis, strings.items


def _mix_curve_of(view: dict, n: int) -> np.ndarray:
    """A tag view's curve from its "m" parts: the starting balance plus every
    trade type's running dollars, as the page's script adds them."""
    total = np.full(n, _MIX_BALANCE)
    for _, sparse in view["m"]:
        total = total + np.array(_expand(sparse, n), dtype=float)
    return total


def _mix_tag_trades(trades: list[BacktestTrade], si: int) -> list[BacktestTrade]:
    """The trades filed under one of _MIX_SUBCATS."""
    return [t for t in trades
            if dashboard._series_labels(t.event_ticker, t.category, _MIX_SERIES)
            == _MIX_SUBCATS[si]]


class TestMixData:
    """What a chunk and the base block carry so the page can work out a mix
    of several categories and tags itself: per-trade arrays, each category ·
    tag view's curve in dollars ("m", in place of the four series that are
    the same curve: "eq", "total", "types" and "dd") and its table row, and
    the constants and markup Python renders with."""

    @pytest.mark.parametrize("quoted", [False, True], ids=["at-cost", "at-market"])
    def test_the_per_trade_arrays_are_the_trades_own_figures(self, quoted):
        trades = _mix_trades(1, quoted=quoted)
        lst, axis, _ = _mix_list(trades)
        days = [d.date() for d in axis]
        assert lst["pf"] == [t.profit for t in trades]
        assert lst["pr"] == [t.profit_ratio for t in trades]
        assert lst["st"] == [t.total_cost + t.fees for t in trades]
        assert [days[i] for i in lst["en"]] == [t.entry_date for t in trades]
        assert [days[i] for i in lst["ex"]] == [t.exit_date for t in trades]
        # The entry month, as the decomposition frame names it
        frame = dashboard._decomposition_frame(trades, _MIX_SERIES)
        assert [f"{m // 12:04d}-{m % 12 + 1:02d}" for m in lst["mo"]] == list(frame["month"])
        # The price bucket, as the decomposition's own cut files each trade
        cut = pd.cut(frame["entry_pA"], bins=dashboard._PRICE_BUCKET_BINS,
                     labels=dashboard._PRICE_BUCKET_LABELS)
        assert [dashboard._PRICE_BUCKET_LABELS[b] for b in lst["pb"]] == list(cut)
        assert {0, 1, 2, 3, 4} == set(lst["pb"])
        # A time-series trade's calibration pair; none for a same-title one
        obs = iter(dashboard._spread_observations(trades))
        for t, cp, ca in zip(trades, lst["cp"], lst["ca"], strict=True):
            assert ((cp, ca) == next(obs)) if t.pair_type == "time_series" \
                else (cp is None and ca is None)
        # Strict JSON, like everything a chunk holds
        json.loads(dashboard._strict_json(lst), parse_constant=_fail_on_constant)

    def test_a_date_off_the_axis_and_a_price_outside_every_bucket_read_minus_one(self):
        trades = _mix_trades(2, n=6)
        late = dataclasses.replace(trades[0], exit_date=date(2099, 1, 1), entry_pA=0.0)
        lst, _, _ = _mix_list([late, *trades[1:]],
                              curve=backtester._build_equity_curve(trades, _MIX_START,
                                                                   _MIX_BALANCE))
        assert lst["ex"][0] == -1 and lst["en"][0] >= 0
        assert lst["pb"][0] == -1

    @pytest.mark.parametrize("quoted", [False, True], ids=["at-cost", "at-market"])
    def test_the_tags_curves_add_up_to_each_category_and_to_the_whole(self, quoted):
        trades = _mix_trades(3, n=60, quoted=quoted)
        curve = backtester._build_equity_curve(trades, _MIX_START, _MIX_BALANCE)
        lst, axis, _ = _mix_list(trades, curve=curve)
        n, views = len(axis), lst["views"]
        tags = sorted(key for key in views if key.startswith("s"))
        assert len(tags) == len(_MIX_SUBCATS)
        whole = np.zeros(n)
        by_category = {ci: np.zeros(n) for ci in range(len(_MIX_CATEGORIES))}
        for key in tags:
            si = int(key[1:])
            own = _mix_tag_trades(trades, si)
            mine = _mix_curve_of(views[key], n)
            # Each tag's "m" is its own curve
            assert mine == pytest.approx(list(backtester._build_equity_curve(
                own, _MIX_START, _MIX_BALANCE)["portfolio_value"]), abs=1e-6)
            # ... split by trade type: each part is that type's line (the
            # view ships no "types" of its own beside "m")
            assert "types" not in views[key]
            lines = {label: series for label, _, series in dashboard._return_by_trade_type(
                own, backtester._build_equity_curve(own, _MIX_START, _MIX_BALANCE),
                _MIX_BALANCE)}
            assert [label for label, _ in views[key]["m"]] == list(lines)
            for label, sparse in views[key]["m"]:
                assert np.array(_expand(sparse, n)) / _MIX_BALANCE * 100 == pytest.approx(
                    lines[label], abs=1e-9)
            whole += mine - _MIX_BALANCE
            by_category[_MIX_CATEGORIES.index(_MIX_SUBCATS[si][0])] += mine - _MIX_BALANCE
        assert whole + _MIX_BALANCE == pytest.approx(list(curve["portfolio_value"]), abs=1e-6)
        for ci, pnl in by_category.items():
            own = [t for t in trades if dashboard._series_labels(
                t.event_ticker, t.category, _MIX_SERIES)[0] == _MIX_CATEGORIES[ci]]
            assert pnl + _MIX_BALANCE == pytest.approx(list(backtester._build_equity_curve(
                own, _MIX_START, _MIX_BALANCE)["portfolio_value"]), abs=1e-6)
            # ... which is the category view's own curve, to its cents
            assert np.round(pnl + _MIX_BALANCE, 2) == pytest.approx(
                _expand(views[f"c{ci}"]["eq"], n), abs=0.0051)

    def test_only_a_tag_view_changes_and_its_curve_replaces_its_series(self):
        trades = _mix_trades(4, quoted=True)
        lst, axis, _ = _mix_list(trades)
        n = len(axis)
        kelly_x, _ = dashboard._kelly_points(trades, 0.75)
        for key, view in lst["views"].items():
            idx = view["idx"]
            sel = [trades[i] for i in idx]
            curve = (backtester._build_equity_curve(trades, _MIX_START, _MIX_BALANCE)
                     if key == "all"
                     else backtester._build_equity_curve(sel, _MIX_START, _MIX_BALANCE))
            # The view as it is built with no mix parts, into tables of its own
            plain = dashboard._view_payload(
                sel, idx, curve, axis, _MIX_BALANCE, _MIX_SERIES, kelly_x,
                lambda t, which: [0, 0], dashboard._StringTable())
            # The four series that are the view's curve once more
            curve_series = ("eq", "total", "types", "dd")
            assert dashboard._MIX_CURVE_SERIES == curve_series
            independent = [k for k in plain
                           if k not in ("table", "best", "worst", *curve_series)]
            assert {k: view[k] for k in independent} == {k: plain[k] for k in independent}
            if key.startswith("s"):
                assert set(view) - set(plain) == {"m", "row"}
                assert set(plain) - set(view) == set(curve_series)
                # Each is its "m" curve again, rounded as Python rounded it:
                # the curve in cents, the return and the drawdown in percent
                # to four decimals, and one line per trade type
                value = _mix_curve_of(view, n)
                assert list(np.round(value, 2)) == _expand(plain["eq"], n)
                assert np.round((value / _MIX_BALANCE - 1.0) * 100, 4) == pytest.approx(
                    _expand(plain["total"], n), abs=1e-9)
                peak = np.maximum.accumulate(value)
                assert np.round((value - peak) / peak * 100, 4) == pytest.approx(
                    _expand(plain["dd"], n), abs=1e-9)
                assert [label for label, _ in view["m"]] == [label for label, _ in plain["types"]]
                for (_, dollars), (_, line) in zip(view["m"], plain["types"], strict=True):
                    assert np.round(np.array(_expand(dollars, n)) / _MIX_BALANCE * 100, 4) \
                        == pytest.approx(_expand(line, n), abs=1e-9)
            else:
                assert set(view) == set(plain)
                assert all(view[k] == plain[k] for k in curve_series)

    def test_a_curve_series_is_left_out_only_where_the_script_rebuilds_it_exactly(self):
        # Half-cent profits from $10,000 put many values on a rounding tie.
        # Whatever a tag view leaves out, the script's arithmetic
        # (_curve_series_from_m, its twin here) gives exactly; whatever it
        # keeps is the view's own series, and differs from that arithmetic
        rng = random.Random(83)
        trades = []
        for t in _mix_trades(83, n=60):
            profit = rng.randint(-4000, 4000) * 5 / 1000
            trades.append(dataclasses.replace(
                t, profit=profit, actual_payoff=profit + t.total_cost + t.fees))
        lst, axis, _ = _mix_list(trades)
        n = len(axis)
        kelly_x, _ = dashboard._kelly_points(trades, 0.75)
        kept = dropped = 0
        for key, view in lst["views"].items():
            if not key.startswith("s"):
                continue
            sel = [trades[i] for i in view["idx"]]
            plain = dashboard._view_payload(
                sel, view["idx"], backtester._build_equity_curve(sel, _MIX_START, _MIX_BALANCE),
                axis, _MIX_BALANCE, _MIX_SERIES, kelly_x, lambda t, which: [0, 0],
                dashboard._StringTable())
            rebuilt = dashboard._curve_series_from_m(view["m"], n, _MIX_BALANCE)
            for name in dashboard._MIX_CURVE_SERIES:
                same = dashboard._same_series(plain[name], rebuilt[name], n)
                assert (name not in view) is same, (key, name)
                if name in view:
                    assert view[name] == plain[name]
                kept += name in view
                dropped += name not in view
        assert kept and dropped
        # The twin's pieces: a sparse series back on every day, gaps as NaN
        expanded = dashboard._sparse_as_floats([[0, 1.5], [2, None], [3, -2.0]], 5)
        assert list(expanded[:2]) == [1.5, 1.5] and np.isnan(expanded[2])
        assert list(expanded[3:]) == [-2.0, -2.0]
        assert not dashboard._same_series([[0, 1.5], [2, None]], expanded, 5)
        assert dashboard._same_series([[0, 1.5]], np.full(5, 1.5), 5)
        assert not dashboard._same_series([["A", [[0, 1.5]]]], [("B", np.full(5, 1.5))], 5)
        assert not dashboard._same_series([["A", [[0, 1.5]]]], [], 5)

    def test_a_tags_row_is_its_line_of_the_category_table(self):
        trades = _mix_trades(5)
        lst, _, _ = _mix_list(trades)
        cells = dashboard._MIX_ROW_CELLS
        assert cells == ("name", "win", "pnl", "mean", "median")
        for si, (category, tag) in enumerate(_MIX_SUBCATS):
            view = lst["views"][f"s{si}"]
            row = dict(zip(cells, view["row"], strict=False))
            pnl = view["row"][len(cells)]
            assert len(view["row"]) == len(cells) + 1
            own = _mix_tag_trades(trades, si)
            assert pnl == pytest.approx(sum(t.profit for t in own))
            assert row["name"] == html.escape(f"{category} · {tag}")
            # The table of this tag alone: its one row, as the page fills the
            # row template — the count from the view, the colour from the sign
            # of the P&L, the share worked out (here the whole: 100%, or none)
            table = dashboard._category_table(dashboard._decomposition_frame(own, _MIX_SERIES))
            filled = dashboard._CATEGORY_ROW.format(
                trades=view["n"], share="—" if pnl == 0 else "100%",
                color=dashboard._COLORS["profit"] if pnl >= 0 else dashboard._COLORS["loss"],
                **row)
            assert table == (dashboard._CATEGORY_TABLE_HEAD.format(groups=1) + filled
                             + dashboard._CATEGORY_TABLE_FOOT)

    def test_a_curve_that_does_not_cover_the_axis_is_not_shipped(self):
        trades = _mix_trades(6)
        # The page's axis opens three days before the slices' own curves do
        long_curve = backtester._build_equity_curve(
            trades, _MIX_START - timedelta(days=3), _MIX_BALANCE)
        gaps: list = []
        lst, _, _ = _mix_list(trades, curve=long_curve, gaps=gaps)
        tags = [view for key, view in lst["views"].items() if key.startswith("s")]
        assert tags and all("m" not in view and "eq" in view and "row" in view for view in tags)
        # Without "m" a tag view keeps every series of its curve, like any other view
        assert all(name in view for view in tags for name in dashboard._MIX_CURVE_SERIES)
        # Counted once for the list, however many tags it holds
        assert gaps == [1]
        # A list that can be mixed counts nothing
        fine: list = []
        _mix_list(trades, gaps=fine)
        assert fine == []

    def test_a_figure_that_is_not_a_number_keeps_every_tag_out_of_a_mix(self):
        trades = _mix_trades(7)
        for bad in (dataclasses.replace(trades[0], profit_ratio=float("nan")),
                    dataclasses.replace(trades[0], profit_ratio=float("inf"))):
            gaps: list = []
            # The axis is the unchanged trades' own curve
            lst, _, _ = _mix_list([bad, *trades[1:]], gaps=gaps,
                                  curve=backtester._build_equity_curve(trades, _MIX_START,
                                                                       _MIX_BALANCE))
            assert gaps == [1]
            assert not any("m" in view for view in lst["views"].values())

    @pytest.mark.parametrize("quoted", [False, True], ids=["at-cost", "at-market"])
    def test_open_capital_from_the_curve_is_what_the_hurdle_is_charged_on(self, quoted):
        trades = _mix_trades(8, quoted=quoted)
        lst, axis, _ = _mix_list(trades, risk_free=_rates())
        days = day_numbers(axis)
        n = len(axis)
        assert quoted is any(t.marks is not None for t in trades)
        for key, view in lst["views"].items():
            if not key.startswith("s"):
                continue
            sel = [trades[i] for i in view["idx"]]
            # From the shipped pieces alone, as the script works it out: the
            # curve less the starting balance, plus the stakes of the trades
            # open, less the profits of the trades paid out
            value = _mix_curve_of(view, n)
            stake, paid = np.zeros(n), np.zeros(n)
            for i in view["idx"]:
                stake[lst["en"][i]] += lst["st"][i]
                stake[lst["ex"][i]] -= lst["st"][i]
                paid[lst["ex"][i]] += lst["pf"][i]
            derived = np.maximum(value - _MIX_BALANCE + np.cumsum(stake) - np.cumsum(paid), 0.0)
            assert derived == pytest.approx(list(dashboard._carried_on_days(sel, days)), abs=1e-6)
            # ... and for several tags at once it adds up the same way
        mixed = [t for t in trades if dashboard._series_labels(
            t.event_ticker, t.category, _MIX_SERIES)[0] != "Cat1"]
        curve = backtester._build_equity_curve(mixed, _MIX_START, _MIX_BALANCE)
        rows = {d.date(): i for i, d in enumerate(axis)}
        derived = dashboard._open_capital_from_curve(
            curve["portfolio_value"].to_numpy(dtype=float), _MIX_BALANCE, mixed,
            [rows[t.entry_date] for t in mixed], [rows[t.exit_date] for t in mixed])
        assert derived == pytest.approx(list(dashboard._carried_on_days(mixed, days)), abs=1e-6)

    def test_with_rates_a_curve_whose_open_capital_cannot_be_rebuilt_is_not_shipped(self):
        trades = _mix_trades(9)
        # A trade whose profit is not its payoff less its cost and fees: the
        # curve books the payoff, so open capital worked out from the profit
        # is off by the difference from its pay-out day on
        held = next(i for i, t in enumerate(trades) if t.exit_date > t.entry_date)
        odd = dataclasses.replace(trades[held], profit=trades[held].profit + 3.0)
        listed = [*trades[:held], odd, *trades[held + 1:]]
        curve = backtester._build_equity_curve(listed, _MIX_START, _MIX_BALANCE)
        with_rates: list = []
        lst, _, _ = _mix_list(listed, risk_free=_rates(), curve=curve, gaps=with_rates)
        assert with_rates == [1]
        assert not any("m" in view for view in lst["views"].values())
        # Without rates no hurdle is charged, so nothing reads that figure
        without: list = []
        _mix_list(listed, curve=curve, gaps=without)
        assert without == []
        # A trade paid out on a day the page does not have: with rates, no mix
        late = dataclasses.replace(trades[held], exit_date=date(2099, 1, 1))
        gone: list = []
        _mix_list([*trades[:held], late, *trades[held + 1:]], risk_free=_rates(), gaps=gone,
                  curve=backtester._build_equity_curve(trades, _MIX_START, _MIX_BALANCE))
        assert gone == [1]

    def test_with_rates_a_tag_whose_open_capital_dips_below_zero_is_not_mixed(self):
        # Tag0 holds a trade at cost dated to pay out twenty days BEFORE it
        # enters (not a shape the backtester produces); Tag1 a quoted trade
        # open over those days. What the hurdle is charged on floors the
        # trades at cost and the quoted ones at zero separately; the page
        # floors the sum of a mix's tags — so Tag0's dip below zero would be
        # taken off Tag1's open capital in a mix of the two.
        def filed(ticker: str, entry: int, held: int, profit: float) -> BacktestTrade:
            day = _MIX_START + timedelta(days=entry)
            return dataclasses.replace(
                _typed_trade("time_series", False, day, day + timedelta(days=held), profit),
                event_ticker=f"{ticker}-{entry}", ticker_a=f"{ticker}-{entry}A",
                ticker_b=f"{ticker}-{entry}B")

        backwards = filed("KXM0", 30, -20, 1.0)
        forwards = filed("KXM0", 10, 20, 1.0)
        quoted = _random_marks(filed("KXM1", 5, 40, -1.0), random.Random(3))
        rest = [filed("KXM2", 60, 3, 0.5), filed("KXM3", 70, 3, 0.5)]
        curve = backtester._build_equity_curve([forwards, quoted, *rest], _MIX_START, 40.0)
        axis = pd.DatetimeIndex(pd.to_datetime(list(curve["date"])))
        rows = {d.date(): i for i, d in enumerate(axis)}

        def listed(first: BacktestTrade, *, risk_free) -> tuple[dict, list]:
            trades = sorted([first, quoted, *rest], key=lambda t: t.entry_date)
            gaps: list = []
            lst = dashboard._list_payload(
                trades, backtester._build_equity_curve(trades, _MIX_START, 40.0), axis,
                _MIX_START, 40.0, _MIX_SERIES, 0.75,
                {c: i for i, c in enumerate(_MIX_CATEGORIES)},
                {pair: i for i, pair in enumerate(_MIX_SUBCATS)}, dashboard._StringTable(),
                heads=dashboard._StringTable(), risk_free=risk_free, mix_gaps=gaps)
            return lst, gaps

        # The two ways disagree on the two tags' trades together
        both = [quoted, backwards]
        value = backtester._build_equity_curve(both, _MIX_START, 40.0)[
            "portfolio_value"].to_numpy(dtype=float)
        rows_of = ([rows[t.entry_date] for t in both], [rows[t.exit_date] for t in both])
        page_way = dashboard._open_capital_from_curve(value, 40.0, both, *rows_of)
        assert not np.allclose(page_way, dashboard._carried_on_days(both, day_numbers(axis)),
                               atol=1e-6)
        # So with rates no tag of that list is offered for mixing
        lst, gaps = listed(backwards, risk_free=_steep_rates())
        assert gaps == [1]
        assert not any("m" in view for view in lst["views"].values())
        # The same list with that trade the right way round is
        lst, gaps = listed(forwards, risk_free=_steep_rates())
        assert gaps == []
        assert all("m" in view for key, view in lst["views"].items() if key.startswith("s"))
        # Without rates no hurdle is charged, so nothing reads that figure
        lst, gaps = listed(backwards, risk_free=None)
        assert gaps == []
        # What gives the tag away: the backwards trade's own figure is below
        # zero while it is "paid out but not yet entered", before any floor
        alone = dashboard._open_capital_from_curve(
            backtester._build_equity_curve([backwards], _MIX_START, 40.0)[
                "portfolio_value"].to_numpy(dtype=float), 40.0, [backwards],
            [rows[backwards.entry_date]], [rows[backwards.exit_date]], floored=False)
        assert alone.min() == pytest.approx(-backwards.total_cost)

    def test_names_that_differ_only_in_letter_case_are_grouped(self):
        # By the live filter's own folding (str.casefold), within one category
        assert dashboard._case_twins([]) == []
        assert dashboard._case_twins(["a", "b", "c"]) == []
        keys = [("Health", name.casefold()) for name in ("COVID", "Covid", "Flu", "covid")]
        keys += [("Sports", "covid"), ("Sports", "STRASSE".casefold()),
                 ("Sports", "straße".casefold())]
        assert dashboard._case_twins(keys) == [[0, 1, 3], [5, 6]]

    def test_the_type_lines_dollars_are_the_lines_before_the_percent(self):
        trades = _mix_trades(10, quoted=True)
        curve = backtester._build_equity_curve(trades, _MIX_START, _MIX_BALANCE)
        dollars: dict = {}
        lines = dashboard._return_by_trade_type(trades, curve, _MIX_BALANCE, dollars_out=dollars)
        assert list(dollars) == [label for label, _, _ in lines]
        for label, _, series in lines:
            assert list(dollars[label] / _MIX_BALANCE * 100) == series
        # The same lines with nothing asked for, and nothing written for no line
        assert dashboard._return_by_trade_type(trades, curve, _MIX_BALANCE) == lines
        untouched = {"kept": 1}
        assert dashboard._return_by_trade_type([], curve, _MIX_BALANCE,
                                               dollars_out=untouched) == []
        assert untouched == {"kept": 1}

    def test_the_base_block_carries_what_python_renders_with(self):
        trades = _mix_trades(11)
        curve = backtester._build_equity_curve(trades, _MIX_START, _MIX_BALANCE)
        axis = pd.DatetimeIndex(pd.to_datetime(list(curve["date"])))
        mix = dashboard._mix_base(axis, _MIX_BALANCE, None)
        assert mix == {
            "start": _MIX_BALANCE, "rf": None,
            "flat": config.FLAT_RETURN_TOLERANCE, "year": config.CALENDAR_DAYS_PER_YEAR,
            "profit": dashboard._COLORS["profit"], "loss": dashboard._COLORS["loss"],
            "khat_delta": ["#F44336", "#4CAF50"],
            "types": [label for label, _ in dashboard._TRADE_TYPE_LINES],
            "buckets": dashboard._PRICE_BUCKET_LABELS,
            "bins": [float(edge) for edge in np.linspace(0, 1, 11)],
            "marker": [6, 2], "log_clip": 1e-7, "top": 5,
            "sub_height": [350, 28, 120], "k11": [0.01, 1.1],
            "row": dashboard._CATEGORY_ROW, "table_head": dashboard._CATEGORY_TABLE_HEAD,
            "table_foot": dashboard._CATEGORY_TABLE_FOOT,
            "cells": list(dashboard._MIX_ROW_CELLS), "none": "—",
            "sep": config.TAG_SCOPE_SEPARATOR, "names": 3,
        }
        # The helpers read the same constants
        assert dashboard._subcategory_chart_height(3) == 350
        assert dashboard._subcategory_chart_height(20) == 28 * 20 + 120
        assert dashboard._one_to_one_extent([]) == pytest.approx(0.011)
        assert dashboard._khat_delta(0.9, 0.75) == ("+0.150", "#F44336")
        assert dashboard._khat_delta(0.6, 0.75) == ("-0.150", "#4CAF50")
        # With rates: the yield in force on each day of the axis, where it changes
        rates = _rates()
        rf = dashboard._mix_base(axis, _MIX_BALANCE, rates)["rf"]
        assert _expand(rf, len(axis)) == list(rates.annual_on(axis))
        assert [i for i, _ in rf][0] == 0 and len(rf) == 2
        # Unavailable rates charge nothing, on every day
        none = RiskFreeRates((), SOURCE_UNAVAILABLE, None)
        assert dashboard._mix_base(axis, _MIX_BALANCE, none)["rf"] == [[0, 0.0]]

    def test_the_page_ships_it_and_the_words_the_script_fills(self):
        _, _, base, chunks = _flt_payload()
        assert base["mix"] == dashboard._mix_base(
            pd.DatetimeIndex(pd.to_datetime(base["dates"])), 1000.0, None)
        for key in ("cal_title", "cal_title_none", "cal_bin", "khat_bar"):
            assert key in base["text"]
        assert base["text"]["cal_title"].format(brier="0.1000", log_loss="0.2000") == \
            dashboard._calibration_title(0.1, 0.2)
        assert base["text"]["cal_title_none"] == dashboard._calibration_title(None, None)
        assert base["text"]["khat_bar"].format(n=12, events=3) == dashboard._khat_bar_text(
            {"n": 12, "events": 3})
        # Every chunk on the page carries the mix parts
        for chunk in chunks.values():
            lst = chunk["list"]
            assert all(len(lst[name]) == len(lst["ret"]) for name in (
                "pf", "pr", "st", "en", "ex", "mo", "pb", "cp", "ca"))
            for key, view in lst["views"].items():
                assert ("m" in view and "row" in view and "eq" not in view) \
                    is key.startswith("s")

    def test_a_tag_groups_raw_khat_counts_are_its_observations(self):
        sweep, _, base, _ = _flt_payload()
        observations = sweep.calibrations_by_band[(0.0, 1.0)].observations
        groups = base["khat"][0]["groups"]
        assert any(key.startswith("s") for key in groups)
        for key, stat in groups.items():
            if not key.startswith("s"):
                assert "raw" not in stat
                continue
            category, tag = base["subcats"][int(key[1:])]
            members = [o for o in observations if dashboard._series_labels(
                o.event_ticker, o.category, _FLT_SERIES)
                == (base["categories"][category], tag)]
            assert stat["raw"] == [sum(1 for o in members if o.in_between),
                                   sum(o.implied for o in members)]
            # k-hat is their ratio, as the bucket's own arithmetic has it
            assert stat["k"] == pytest.approx(
                (stat["raw"][0] / stat["n"]) / (stat["raw"][1] / stat["n"]))

    def test_one_warning_for_the_whole_page_when_lists_cannot_be_mixed(
            self, monkeypatch, tmp_path, caplog):
        # The slices' curves open two days after the page's own: no list of
        # the page can be mixed, and the log says so once
        sweep = _flt_sweep()
        monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(dashboard.yf, "download",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
        with caplog.at_level(logging.WARNING):
            page = dashboard.generate_dashboard(
                sweep.primary.trades, sweep.primary.equity_df, _FLT_START + timedelta(days=2),
                1000.0, sweep=sweep, interval_discount=0.75,
                series_categories=_FLT_SERIES).read_text(encoding="utf-8")
        said = [r for r in caplog.records if "cannot combine categories and tags" in r.getMessage()]
        assert len(said) == 1
        chunks = TestFilterPage._chunks(page)
        assert len(chunks) > 1 and said[0].args[0] == len(chunks)
        assert not any("m" in view for chunk in chunks.values()
                       for view in chunk["list"]["views"].values())
        # A page whose lists can all be mixed says nothing
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            TestFilterPage()._page(monkeypatch, tmp_path)
        assert not [r for r in caplog.records
                    if "cannot combine categories and tags" in r.getMessage()]

    def test_a_sidecar_chunk_carries_the_same_parts(self, tmp_path):
        _, chunker, grid = _sl_build(_kc_sweep_sell(), tmp_path / "f")
        assert grid.sidecars and grid.mix_gaps == 0 and chunker.mix_gaps == []
        inline = _unpack(chunker.chunks[0])["list"]
        for chunk_id in range(len(chunker.chunks), len(chunker.chunks) + grid.sidecars):
            lst = _sl_file(tmp_path / "f", chunk_id)["list"]
            assert set(lst) == set(inline)
            tags = [view for key, view in lst["views"].items() if key.startswith("s")]
            assert tags and all("m" in view and "row" in view and "eq" not in view
                                for view in tags)


class TestTierOffGrid:
    """Which scenario each band's "Tier floors: off" choice shows (the grid's
    tier-off family: _tier_off_binds, _with_tier_off, the walk's off cells):
    a binding band's own tier-off simulation at every k, a band the tiers
    never bind at its tier-on chunks, and no off view at all unless the
    family is complete. #68's TestTierOffRuns, on the grid."""

    def test_a_binding_band_shows_its_own_tier_off_points(self):
        sweep = _flt_sweep_tiers()
        _, source, base, chunks = _flt_payload(sweep, _FLT_SERIES_TIERS)
        assert source.tier_binds == (True, False, False) == tuple(base["tier_binds"])
        assert dashboard._tier_off_binds(sweep, list(source.bands)) == [True, False, False]
        off, at_060 = sweep.tier_off_scenarios[0], sweep.tier_off_scenarios[1]
        # Filed under the run's own cap, every population the family holds
        cell = source.off_cell((0.0, 1.0), 0.75)
        assert list(cell) == [0.2]
        assert cell[0.2]["all"] is off and cell[0.2]["time_series"].population == "time_series"
        assert source.off_cell((0.0, 1.0), 0.60)[0.2]["all"] is at_060
        # A band the tiers never bind at has no off cell of its own
        assert source.off_cell((0.3, 0.6), 0.75) == {}
        assert source.off_calibrations == {
            (0.0, 1.0): sweep.tier_off_calibrations_by_band[(0.0, 1.0)]}
        # Its own chunk at the primary k: the tier-off run's six trades,
        # never the page's own list
        grid, grid_off = base["grid"], base["grid_off"]
        assert grid_off[0][1][0] != grid[0][1][0]
        assert chunks[grid_off[0][1][0]]["list"]["views"]["all"]["n"] == 6
        # ... and at k 0.60 the same list its tier-on point traded: one chunk
        assert grid_off[0][0][0] == grid[0][0][0]
        # No off point at k 1.00 (nor a tier-on one): never simulated
        assert grid_off[0][2][0] is None

    def test_a_point_not_stamped_tier_off_is_never_an_off_cell(self):
        # The family's "all" point at the primary k replaced by one that is
        # not stamped tier_floors False (listed FIRST, so a grid that ignored
        # the stamp would file it): that scenario is simply never simulated
        sweep = _flt_sweep_tiers()
        off = sweep.tier_off_scenarios[0]
        unstamped = dataclasses.replace(off, tier_floors=True)
        sweep = dataclasses.replace(
            sweep, tier_off_scenarios=[unstamped, *sweep.tier_off_scenarios[1:]])
        _, source, base, _ = _flt_payload(sweep, _FLT_SERIES_TIERS)
        assert "all" not in source.off_cell((0.0, 1.0), 0.75)[0.2]
        assert base["grid_off"][0][1][0] is None
        # ... while the stamped point at k 0.60 still is one
        assert base["grid_off"][0][0][0] == base["grid"][0][0][0]

    def test_a_stray_point_at_a_band_the_tiers_never_bind_is_never_an_off_cell(self):
        # (The bar-side twin of #68's
        # TestScenarioExplorerTierFloors::test_only_a_binding_bands_tier_off_
        # points_fill_the_off_grid.) A point stamped tier-off at a band the
        # tiers never bind at — listed after every real point, trading a
        # series filed under a category nothing else trades — is no off
        # cell: that band's off view is its tier-on row, and the bar never
        # offers a category no scenario it can show ever trades
        sweep = _flt_sweep_tiers()
        snow = _ftrade("KXSNOW-1", "time_series", date(2026, 1, 12), date(2026, 1, 15), 4.0)
        stray = SweepPoint(k=0.75, trades=[snow],
                           equity_df=backtester._build_equity_curve([snow], _FLT_START, 1000.0),
                           spread_band=(0.3, 0.6), population="all", tier_floors=False,
                           size_cap=sweep.primary.size_cap)
        sweep = dataclasses.replace(sweep,
                                    tier_off_scenarios=[*sweep.tier_off_scenarios, stray])
        series = {**_FLT_SERIES_TIERS, "KXSNOW": ("Snowland", ("Snow",))}
        _, source, base, _ = _flt_payload(sweep, series)
        assert source.tier_binds == (True, False, False)
        assert source.off_cell((0.3, 0.6), 0.75) == {}
        assert base["grid_off"][1] == base["grid"][1]
        assert "Snowland" not in base["categories"]
        # ... while the binding band's own tier-off trades are still offered
        assert "Climate" in base["categories"]

    def test_a_band_the_tiers_never_bind_reuses_its_chunks(self):
        sweep = _flt_sweep_tiers()
        _, _, base, _ = _flt_payload(sweep, _FLT_SERIES_TIERS)
        assert base["grid_off"][1:] == base["grid"][1:]
        assert [b["label"] for b in base["bands_off"]] == ["0-1", "0.3-0.6", "0.3-1"]
        # Their k-hat breakdown is their own (none recorded on this fixture)
        assert base["khat_off"][1:] == base["khat"][1:] == [None, None]

    def test_a_band_the_tiers_never_bind_reuses_its_calibration(self):
        # The fixture's two non-binding bands carry no calibration, so the
        # test above cannot tell a reused calibration from a dropped one:
        # with real ones, each off view regroups its band's own
        sweep = _flt_sweep_tiers()
        own = {(0.3, 0.6): _flt_calibration([("KXOTHER-1", "Other")]),
               (0.3, 1.0): _flt_calibration([("KXSPACEX-14", "Science")])}
        sweep = dataclasses.replace(
            sweep, calibrations_by_band={**sweep.calibrations_by_band, **own})
        _, source, base, _ = _flt_payload(sweep, _FLT_SERIES_TIERS)
        assert source.calibrations[(0.3, 0.6)] is own[(0.3, 0.6)]
        assert base["khat_off"][1:] == base["khat"][1:]
        assert None not in base["khat_off"][1:]

    def test_a_non_binding_primary_reuses_the_pages_own_chunk(self):
        # The primary band's floor (0.3) sits at or above both tiers: its off
        # view is the very chunk the page renders from (chunk 0, the page's
        # own trades), while the binding band beside it shows its own
        # tier-off simulation
        sweep = _flt_sweep_tiers()
        narrow = sweep.scenarios[1]
        sweep = dataclasses.replace(sweep, primary=narrow, points=[narrow],
                                    calibration=None)
        _, source, base, chunks = _flt_payload(sweep, _FLT_SERIES_TIERS)
        pb, pk, pc = source.primary
        assert source.bands[pb] == (0.3, 0.6)
        assert base["grid"][pb][pk][pc] == base["grid_off"][pb][pk][pc] == 0
        assert base["grid_off"][0][1][0] != base["grid"][0][1][0]
        assert chunks[base["grid_off"][0][1][0]]["list"]["views"]["all"]["n"] == 6

    def test_no_off_view_unless_the_family_is_complete(self):
        sweep = _flt_sweep_tiers()
        incomplete = {
            "no family": dataclasses.replace(sweep, tier_off_scenarios=[],
                                             tier_off_calibrations_by_band={}),
            # A binding band the family does not name: never shown tier-on
            "binding band missing": dataclasses.replace(sweep,
                                                        tier_off_calibrations_by_band={}),
        }
        for why, case in incomplete.items():
            _, source, base, _ = _flt_payload(case, _FLT_SERIES_TIERS)
            assert source.tier_binds is None, why
            assert base["grid_off"] is base["bands_off"] is base["khat_off"] is None, why
            assert base["tier_binds"] is None, why
        # No sweep, or an unrecorded band: nothing to show either
        assert dashboard._tier_off_binds(None, [(0.0, 1.0)]) is None
        assert dashboard._tier_off_binds(sweep, [None]) is None
        trades = sweep.primary.trades
        unbanded = dashboard._grid_source(None, trades, sweep.primary.equity_df, 0.75)
        assert unbanded.tier_binds is None

    def test_no_family_means_no_off_view_even_where_no_tier_binds(self):
        # Every band the run offers sits at or above both tiers, so each one's
        # off view WOULD be its own tier-on run — yet a run that simulated no
        # tier-off family has no off view at all
        sweep = _flt_sweep()
        narrow = sweep.scenarios[1]            # (0.3, 0.6): no tier binds there
        sweep = dataclasses.replace(sweep, primary=narrow, points=[narrow], scenarios=[],
                                    calibration=None)
        assert dashboard._tier_off_binds(sweep, [(0.3, 0.6)]) is None
        _, source, base, _ = _flt_payload(sweep)
        assert source.bands == ((0.3, 0.6),) and base["grid_off"] is None
        # The family check alone decides that: beside a non-empty family that
        # names no offered band, the same band reads as one the tiers never
        # bind at — and its off view is its tier-on chunks
        family = dataclasses.replace(
            sweep, tier_off_scenarios=[_flt_sweep_tiers().tier_off_scenarios[0]])
        assert dashboard._tier_off_binds(family, [(0.3, 0.6)]) == [False]
        _, source, base, _ = _flt_payload(family)
        assert source.tier_binds == (False,) and base["grid_off"] == base["grid"]

    def test_the_fallback_grid_keeps_the_family(self):
        # A size-cap sweep that cannot be used: the eager grid the page falls
        # back to carries the same tier-off family
        sweep = dataclasses.replace(_flt_sweep_tiers(), cap_sweep=object())
        trades, curve = sweep.primary.trades, sweep.primary.equity_df
        source = dashboard._grid_source(sweep, trades, curve, 0.75)
        assert source.cap_sweep is None and source.tier_binds == (True, False, False)


class TestFilterPayloadTierOff:
    """The base block's tier-off half: every band's tier-off name, phrase,
    chunks and k-hat breakdown beside its own, parallel to them, sharing a
    chunk wherever the two traded alike."""

    @staticmethod
    def _payload(sweep=None, off: bool = True, series=_FLT_SERIES_TIERS) -> dict:
        """The base block of _flt_sweep_tiers (or `sweep`); off=False drops
        its tier-off family first."""
        sweep = sweep or _flt_sweep_tiers()
        if not off:
            sweep = dataclasses.replace(sweep, tier_off_scenarios=[],
                                        tier_off_calibrations_by_band={})
        return _flt_payload(sweep, series)[2]

    def test_the_tier_floors_are_named_from_config(self):
        short, long_ = config.MIN_PRICE_DIFF_SHORT_GAP, config.MIN_PRICE_DIFF_LONG_GAP
        near, cap = config.SHORT_DEADLINE_GAP_DAYS, config.MAX_DEADLINE_GAP_DAYS
        assert dashboard._TIER_FLOORS == f"{short:.2f}/{long_:.2f}"
        assert dashboard._TIER_OPTION_ON == f"on ({short:.2f} / {long_:.2f} by deadline gap)"
        assert dashboard._TIER_SELECT_TITLE == (
            f"on — a time-series pair needs pB − pA of at least {short:.2f} when its "
            f"deadlines are up to {near} days apart and {long_:.2f} for {near + 1}–{cap} "
            "days, and at least the band's floor; off — the band's floor alone (the spread "
            "must still be positive); live trading follows the saved live defaults' tier "
            "floors (main.py --tier-floors / --no-tier-floors overrides them for one run).")
        assert dashboard._KHAT_TEXT["khat_scope_tier_off"] == (
            f"{{scope}}, with the {short:.2f}/{long_:.2f} tier floors off")

    def test_bands_off_parallel_to_bands(self):
        payload = self._payload()
        on, off = payload["bands"], payload["bands_off"]
        assert [b["label"] for b in off] == ["0-1", "0.3-0.6", "0.3-1"]
        assert [b["option"] for b in off] == ["0-1 (primary)", "0.3-0.6", "0.3-1"]
        assert [b["option"] for b in on] == [
            "max(tier,0)-1 (primary)", "max(tier,0.3)-0.6", "max(tier,0.3)-1"]
        assert len(payload["tier_binds"]) == len(payload["grid_off"]) == len(on) == 3
        floors = dashboard._TIER_FLOORS
        never = " (they never bind at this band: its run with them on)"
        assert [b["where"] for b in off] == [
            f"the primary spread band 0-1 with the {floors} tier floors off",
            f"spread band 0.3-0.6 with the {floors} tier floors off{never}",
            f"spread band 0.3-1 with the {floors} tier floors off{never}"]
        # The scenario phrase is the tier-on one's template, its band named
        # as the tiers-off setting names it
        assert dashboard._scenario_phrase(off[0]["where"], payload["ks"][1]["text"],
                                          payload["caps"][0]["text"]) == (
            f"the primary spread band 0-1 with the {floors} tier floors off, k = 0.75, "
            "20% cap per trade")

    def test_khat_off_parallel_too(self):
        payload = self._payload()
        sweep = _flt_sweep_tiers()
        cat_index = {c: i for i, c in enumerate(payload["categories"])}
        sub_index = {(payload["categories"][ci], tag): i
                     for i, (ci, tag) in enumerate(payload["subcats"])}
        cal = sweep.tier_off_calibrations_by_band[(0.0, 1.0)]
        assert payload["khat_off"] == [
            dashboard._khat_band(cal, _FLT_SERIES_TIERS, cat_index, sub_index,
                                 ks=(0.6, 0.75, 1.0)), None, None]
        assert payload["khat_off"][0] != payload["khat"][0]

    def test_categories_include_the_tier_off_runs(self):
        sweep, _, payload, chunks = _flt_payload(_flt_sweep_tiers(), _FLT_SERIES_TIERS)
        assert payload["categories"] == ["Climate", "Commodities", "Other", "Science", "Sports"]
        climate = payload["categories"].index("Climate")
        assert [climate, "Rain"] in payload["subcats"]
        # Only the tier-off list trades in it
        off_views = chunks[payload["grid_off"][0][1][0]]["list"]["views"]
        assert off_views[f"c{climate}"]["n"] == 1
        assert f"c{climate}" not in chunks[0]["list"]["views"]

    def test_each_half_of_the_category_union(self):
        # A category only a tier-off run's TRADES carry (Climate: KXRAIN-1)
        # and one only its k-hat OBSERVATIONS carry (Weather: KXSNOW-1) are
        # both offered — the fixture's own tier-off calibration files KXRAIN
        # as well, so the test above cannot tell the two halves apart
        series = {**_FLT_SERIES_TIERS, "KXSNOW": ("Weather", ("Snow",))}
        cal = _flt_calibration([("KXSNOW-1", "Other"), ("KXNHLHART-27", "Other")])
        sweep = dataclasses.replace(_flt_sweep_tiers(),
                                    tier_off_calibrations_by_band={(0.0, 1.0): cal})
        payload = self._payload(sweep, series=series)
        cats = payload["categories"]
        assert cats == ["Climate", "Commodities", "Other", "Science", "Sports", "Weather"]
        assert [cats.index("Climate"), "Rain"] in payload["subcats"]
        assert [cats.index("Weather"), "Snow"] in payload["subcats"]

    def test_every_tier_on_entry_is_unchanged(self):
        # The family adds its own keys and nothing else: every tier-on entry,
        # the tier-on grid and every tier-on chunk's whole-run view read as
        # without it (the category list itself gains the tier-off runs'
        # categories, so category-keyed views and k-hat groups renumber)
        _, _, with_off, chunks = _flt_payload(_flt_sweep_tiers(), _FLT_SERIES_TIERS)
        without_sweep = dataclasses.replace(_flt_sweep_tiers(), tier_off_scenarios=[],
                                            tier_off_calibrations_by_band={})
        _, _, without, plain = _flt_payload(without_sweep, _FLT_SERIES_TIERS)
        for key in ("bands", "ks", "caps", "primary", "grid", "kd"):
            assert with_off[key] == without[key], key
        assert [b and b["groups"]["all"] for b in with_off["khat"]] == \
            [b and b["groups"]["all"] for b in without["khat"]]
        for cid in {c for band in without["grid"] for ks in band for c in ks if c is not None}:
            assert chunks[cid]["list"]["views"]["all"] == plain[cid]["list"]["views"]["all"], cid

    def test_without_a_family_there_is_no_off_view(self):
        _, _, payload, _ = _flt_payload()
        for key in ("bands_off", "khat_off", "grid_off", "tier_binds"):
            assert payload[key] is None, key
        # Each band's option text, the primary marked
        assert [b["option"] for b in payload["bands"]] == [
            "max(tier,0)-1 (primary)", "max(tier,0.3)-0.6", "max(tier,0.3)-1"]

    def test_the_primary_is_marked_where_it_sorts(self):
        # The primary (0.3-0.6, whose floor the tiers never bind at) sorts
        # SECOND: its off option and phrase are the primary's, and its off
        # view is its own tier-on chunk
        sweep = _flt_sweep_tiers()
        narrow = sweep.scenarios[1]
        sweep = dataclasses.replace(sweep, primary=narrow, points=[narrow],
                                    calibration=None)
        payload = self._payload(sweep)
        assert payload["primary"][0] == 1
        assert payload["bands_off"][1]["option"] == "0.3-0.6 (primary)"
        assert payload["bands_off"][0]["option"] == "0-1"
        assert payload["bands_off"][1]["where"].startswith("the primary spread band 0.3-0.6 ")
        assert payload["grid_off"][1] == payload["grid"][1]


class TestFilterPage:
    """The bar, the data blocks and the script, on a whole rendered page."""

    def _page(self, monkeypatch, tmp_path, sweep=None, series=_FLT_SERIES) -> str:
        monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(dashboard.yf, "download",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
        sweep = sweep or _flt_sweep()
        out = dashboard.generate_dashboard(sweep.primary.trades, sweep.primary.equity_df,
                                           _FLT_START, 1000.0, sweep=sweep,
                                           interval_discount=0.75, series_categories=series)
        return out.read_text(encoding="utf-8")

    @staticmethod
    def _data(page: str) -> dict:
        """The base block (id="dash-data"), inflated as the script inflates it
        (base64, then gzip) and parsed strictly (a NaN/Infinity token fails the
        test)."""
        opening = '<script type="text/plain" id="dash-data" data-encoding="gzip+base64">'
        start = page.index(opening) + len(opening)
        end = page.index("</script>", start)
        return _decode_block(page[start:end])

    @staticmethod
    def _chunks(page: str) -> dict[int, dict]:
        """Every scenario chunk on the page (id="dash-chunk-<i>"), by chunk
        id, inflated and parsed strictly."""
        return {int(element_id[len("dash-chunk-"):]): data
                for element_id, data in _packed_blocks(page).items()
                if element_id.startswith("dash-chunk-")}

    def test_the_bar_preselects_the_primary_and_counts_its_trades(self, monkeypatch, tmp_path):
        page = self._page(monkeypatch, tmp_path)
        options = r'<option value="(\d+)"( selected)?>(.*?)</option>'
        band = re.search(r'<select id="flt-band"[^>]*>(.*?)</select>', page).group(1)
        assert re.findall(options, band) == [
            ("0", " selected", "max(tier,0)-1 (primary)"),
            ("1", "", "max(tier,0.3)-0.6"), ("2", "", "max(tier,0.3)-1")]
        ks = re.search(r'<select id="flt-k"[^>]*>(.*?)</select>', page).group(1)
        assert re.findall(options, ks) == [
            ("0", "", "k = 0.60"), ("1", " selected", "k = 0.75 (primary)"),
            ("2", "", "k = 1.00")]
        caps = re.search(r'<select id="flt-cap"[^>]*>(.*?)</select>', page).group(1)
        assert re.findall(options, caps) == [("0", " selected", "20% (primary)")]
        # The Category menu: a check-box dropdown, its clearing box first and
        # ticked, no category ticked
        cats = _page_elements(page)["menus"]["flt-cat"]
        assert [cats["all"]["text"], *(row["text"] for row in cats["rows"])] == [
            "All categories", "Commodities (1)", "Other (1)", "Science (0)", "Sports (3)"]
        assert cats["label"] == "All categories" and cats["all"]["checked"]
        assert not any(row["checked"] or row["hidden"] for row in cats["rows"])
        # The Tag menu lists every tag with its category in front
        tags = _page_elements(page)["menus"]["flt-tag"]
        assert [tags["all"]["text"], *(row["text"] for row in tags["rows"])] == [
            "All tags", "Commodities · Oil & Gas (1)", "Other · General (1)",
            "Science · General (0)", "Sports · Basketball (2)", "Sports · Hockey (1)"]
        assert tags["label"] == "All tags" and tags["all"]["checked"]
        assert not any(row["checked"] or row["hidden"] for row in tags["rows"])
        # The summary is escaped into the page (the bar's reach names "the
        # bar's k"), and ends with what the bar reaches beyond the sections
        assert html.escape(
            "Showing every trade of the run at the primary spread band max(tier,0)-1, "
            "k = 0.75, 20% cap per trade: 5 trades. "
            f"{dashboard._BAR_REACH_NO_EXPLORER_TIERS[(True, True)]}") in page
        assert re.search(r'Trades found: <span id="hdr-trades">5</span>', page)

    def test_the_bar_says_what_it_reaches_and_never_the_interval_discount_tiers(
            self, monkeypatch, tmp_path):
        # (#68's test_the_bar_names_what_it_does_not_filter, on _bar_reach.)
        # Pinned by literal on the page: the Interval Discount section never
        # follows the Tier floors choice, with the family or without it. The
        # explorer follows that choice exactly when it renders a Tier floors
        # select of its own — with the family (_BAR_REACH), never without it,
        # where the line says it does not (never _BAR_REACH naming a select
        # the page lacks)
        no_tiers = ("The Interval Discount section follows only the bar's k and size cap (at "
                    "the primary spread band, tier floors on — never its Tier floors choice), "
                    "and the Scenario Explorer's selects follow its band, k and size cap but "
                    "not its Tier floors choice; category and tag reach neither.")
        tiers = ("The Interval Discount section follows only the bar's k and size cap (at the "
                 "primary spread band, tier floors on — never its Tier floors choice), and the "
                 "Scenario Explorer's selects follow its band, Tier floors choice, k and size "
                 "cap; category and tag reach neither.")
        assert dashboard._BAR_REACH == tiers
        plain = self._page(monkeypatch, tmp_path)
        family = self._page(monkeypatch, tmp_path, _flt_sweep_tiers(), _FLT_SERIES_TIERS)
        for page, reach, not_reach, select in ((plain, no_tiers, tiers, False),
                                               (family, tiers, no_tiers, True)):
            assert ('id="scn-tier-select"' in page) is select
            assert self._data(page)["text"]["unfiltered"] == f" {reach}"
            assert html.escape(reach) in page
            assert html.escape(not_reach) not in page

    def test_the_selects_wait_for_the_script_and_are_never_restored(
            self, monkeypatch, tmp_path):
        # Disabled until the script has inflated its data; autocomplete off,
        # so a reload cannot restore a choice the rendered page is not showing
        page = self._page(monkeypatch, tmp_path)
        for sel in ("flt-band", "flt-k", "flt-cap"):
            assert f'<select id="{sel}" disabled autocomplete="off">' in page
        # The two check-box menus likewise: every box disabled and never
        # restored, the menu greyed until the script has its data, and the
        # rule a set of ticks follows on both as their hover text
        title = html.escape(dashboard._MENU_TITLE)
        for menu in ("flt-cat", "flt-tag"):
            assert f'<details id="{menu}" class="flt-multi flt-off" title="{title}">' in page
            body = re.search(rf'<details id="{menu}".*?</details>', page, re.S).group(0)
            boxes = re.findall(r"<input [^>]*>", body)
            assert len(boxes) > 1
            assert all(box.startswith('<input type="checkbox" id="')
                       and box.endswith(' disabled autocomplete="off">') for box in boxes)
            parsed = _page_elements(page)["menus"][menu]
            assert parsed["all"]["disabled"] and all(r["disabled"] for r in parsed["rows"])
        assert dashboard._MENU_TITLE == (
            "Tick as many as you like. A ticked category counts in full unless some of its "
            "tags are ticked; then only those count.")

    @pytest.mark.parametrize("family", [True, False])
    def test_the_tier_floors_select_offers_on_and_off(self, monkeypatch, tmp_path, family):
        # Right after the band select, "on" preselected, disabled until the
        # script has its data; a run with no tier-off view says so beside it
        sweep, series = ((_flt_sweep_tiers(), _FLT_SERIES_TIERS) if family
                         else (_flt_sweep(), _FLT_SERIES))
        page = self._page(monkeypatch, tmp_path, sweep, series)
        # Its title spells both rules out, escaped whole: nothing in it can
        # end the attribute (or the tag) early
        title = html.escape(dashboard._TIER_SELECT_TITLE)
        assert '"' not in title and ">" not in title
        assert (f'<select id="flt-tier" disabled autocomplete="off" title="{title}">'
                in page)
        assert page.index('id="flt-band"') < page.index('id="flt-tier"') \
            < page.index('id="flt-cat"')
        tier = re.search(r'<select id="flt-tier"[^>]*>(.*?)</select>', page).group(1)
        assert [(value, chosen, html.unescape(text)) for value, chosen, text in re.findall(
            r'<option value="([^"]*)"( selected)?>(.*?)</option>', tier)] == [
            ("on", " selected", dashboard._TIER_OPTION_ON),
            ("off", "", dashboard._TIER_OPTION_OFF)]
        # The page parser the script tests run on still reads the select
        parsed = _page_elements(page)["selects"]["flt-tier"]
        assert parsed["disabled"] and [o["value"] for o in parsed["options"]] == ["on", "off"]
        assert ('id="flt-tier-note"' in page) is not family
        assert ("(not simulated for this run)</span>" in page) is not family
        assert (self._data(page)["bands_off"] is None) is not family

    def test_the_block_is_strict_json_and_escapes_kalshi_text(self, monkeypatch, tmp_path):
        trades = [dataclasses.replace(t, title_a="</script><b>x</b>") for t in _flt_trades()]
        curve = backtester._build_equity_curve(trades, _FLT_START, 1000.0)
        sweep = BacktestSweep(
            primary=SweepPoint(k=0.75, trades=trades, equity_df=curve, spread_band=(0.0, 1.0)),
            points=[], calibration=None)
        series = {"KXNCAAMBGAME": ("<i>Sports</i>", ("B</script>",))}
        page = self._page(monkeypatch, tmp_path, sweep, series)
        data = self._data(page)            # strict: a NaN/Infinity token fails
        chunks = self._chunks(page)        # every chunk, strictly too
        assert "<i>Sports</i>" in data["categories"]
        assert all("<b>" not in text for text in chunks[0]["list"]["kt"])
        bar = page[page.index('<div id="flt-bar"'):
                   page.index("</details>", page.index('id="flt-tag"'))]
        assert "<i>" not in bar and "&lt;i&gt;Sports&lt;/i&gt;" in bar
        # Base64 has no "<", so nothing inside a block can close it early:
        # each block ends exactly where its encoded bytes end
        blocks = re.findall(r'id="(dash-[a-z0-9-]+)" data-encoding="gzip\+base64">', page)
        assert blocks[-1] == "dash-data" and len(blocks) == len(chunks) + 1
        for element_id in blocks:
            opening = f'id="{element_id}" data-encoding="gzip+base64">'
            body = page[page.index(opening) + len(opening):]
            assert re.fullmatch(r"[A-Za-z0-9+/=]+", body[:body.index("</script>")])

    def test_the_block_encodes_deterministically(self):
        payload = {"a": [1.5, float("nan")], "b": "</script>"}
        once = dashboard._packed_json_script("x", payload)
        assert once == dashboard._packed_json_script("x", payload)

    def test_every_element_the_script_reaches_exists(self, monkeypatch, tmp_path):
        page = self._page(monkeypatch, tmp_path)
        js = dashboard._FILTER_JS
        literal = {i for i in re.findall(r"(?:byId|getElementById|setText|traceOf|markerOf|"
                                         r"redraw|bars)\('([a-z][a-z0-9_-]*)'", js)
                   if not i.endswith("-")}          # 'kpi-' + key is checked below
        dynamic = {f"kpi-{key}" for key, *_ in dashboard._performance_kpis(
            _flt_sweep().primary.equity_df, _flt_trades(), 1000.0)}
        dynamic |= {f"{p}-{part}" for p in ("dec", "cal", "diag", "risk", "khat")
                    for part in ("empty", "body")}
        # 'dash-chunk-' + id: every scenario's chunk the grid names
        grid = self._data(page)["grid"]
        chunk_ids = {c for band in grid for ks in band for c in ks if c is not None}
        dynamic |= {f"dash-chunk-{c}" for c in chunk_ids}
        missing = sorted(i for i in literal | dynamic if f'id="{i}"' not in page)
        assert literal and chunk_ids and not missing
        assert "'dash-chunk-' + id" in js

    def test_the_script_comes_after_every_section(self, monkeypatch, tmp_path):
        page = self._page(monkeypatch, tmp_path)
        assert page.index('id="flt-bar"') < page.index("Portfolio Performance")
        script = page.index("function inflate(el)")
        chunk_at = [page.index(f'id="dash-chunk-{c}"') for c in sorted(self._chunks(page))]
        # Every chunk and the base block sit after the sections and before the
        # script: an element a script reaches exists when it runs
        assert page.index("Benchmark Comparison") < chunk_at[0]
        assert chunk_at == sorted(chunk_at)
        assert chunk_at[-1] < page.index('id="dash-data"') < script

    def test_every_redrawn_chart_is_captured_as_python_drew_it(self):
        # redraw() starts from the layout captured on load (CHARTS), never
        # the live one a zoom has changed — a chart left out would not redraw
        js = dashboard._FILTER_JS
        captured = set(re.findall(r"'([a-z-]+)'",
                                  re.search(r"var CHARTS = \[(.*?)\];", js, re.S).group(1)))
        redrawn = set(re.findall(r"(?:redraw|bars)\('([a-z][a-z0-9-]*)'", js))
        assert redrawn and redrawn == captured

    def test_a_filter_that_cannot_be_built_costs_only_the_bar(
            self, monkeypatch, tmp_path, caplog):
        def broken(*_a, **_k):
            raise ValueError("boom")
        monkeypatch.setattr(dashboard, "_filter_payload", broken)
        with caplog.at_level(logging.WARNING):
            page = self._page(monkeypatch, tmp_path)
        assert 'id="flt-bar"' not in page and 'id="dash-data"' not in page
        assert 'id="dash-chunk-' not in page
        assert 'id="flt-unavailable"' in page and "Portfolio Performance" in page
        assert "function inflate(el)" not in page
        assert any("could not be built" in r.getMessage() for r in caplog.records)

    def test_one_k_prices_the_whole_page(self, monkeypatch, tmp_path):
        # No override: the sweep's primary k (0.60 here) prices the Risk
        # section's Kelly scatter AND the primary chunk's copy of it, and is
        # the k the summary names — never the config constant beside it
        sweep = _flt_sweep()
        sweep = dataclasses.replace(
            sweep, primary=dataclasses.replace(sweep.primary, k=0.60),
            scenarios=[dataclasses.replace(p, k=0.60) if p.k == 0.75 else p
                       for p in sweep.scenarios])
        monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(dashboard.yf, "download",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
        page = dashboard.generate_dashboard(
            sweep.primary.trades, sweep.primary.equity_df, _FLT_START, 1000.0, sweep=sweep,
            series_categories=_FLT_SERIES).read_text(encoding="utf-8")
        kx, _ = dashboard._kelly_points(sweep.primary.trades, 0.60)
        data = self._data(page)
        assert self._chunks(page)[0]["list"]["kx"] == pytest.approx(kx)
        assert data["ks"][data["primary"][1]]["value"] == 0.60
        assert kx != pytest.approx(dashboard._kelly_points(sweep.primary.trades, None)[0])
        assert "max(tier,0)-1, k = 0.60, 20% cap per trade: 5 trades." in page

    def test_an_override_unlike_the_sweep_s_k_is_warned_about(
            self, monkeypatch, tmp_path, caplog):
        # Unsupported (backtest.py always passes the primary's own k): the
        # override still prices the Kelly scatter and the primary chunk, but
        # the bar's k axis is the sweep's — its summary names 0.75 — and the
        # log says the page names two ks rather than letting it pass silently
        sweep = _flt_sweep()
        monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(dashboard.yf, "download",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
        with caplog.at_level(logging.WARNING):
            page = dashboard.generate_dashboard(
                sweep.primary.trades, sweep.primary.equity_df, _FLT_START, 1000.0,
                sweep=sweep, interval_discount=0.6,
                series_categories=_FLT_SERIES).read_text(encoding="utf-8")
        warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
                  and "differs from the sweep's primary k" in r.getMessage()]
        assert len(warned) == 1 and "0.6" in warned[0] and "0.75" in warned[0]
        data = self._data(page)
        assert data["ks"][data["primary"][1]]["value"] == 0.75
        kx, _ = dashboard._kelly_points(sweep.primary.trades, 0.6)
        assert self._chunks(page)[0]["list"]["kx"] == pytest.approx(kx)
        assert "max(tier,0)-1, k = 0.75, 20% cap per trade: 5 trades." in page
        # An override equal to the sweep's k is no conflict
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            self._page(monkeypatch, tmp_path)
        assert not any("differs from the sweep's primary k" in r.getMessage()
                       for r in caplog.records)


class TestFilterSummary:
    """The line under the bar says what the page shows, from templates the
    script fills too (D.text) — so the script holds no sentence of its own."""

    T = dashboard._SUMMARY_TEMPLATES

    def test_the_whole_band_and_a_slice(self):
        where = dashboard._scenario_phrase("the primary spread band max(tier,0)-1",
                                           "k = 0.75", "20% cap per trade")
        assert where == "the primary spread band max(tier,0)-1, k = 0.75, 20% cap per trade"
        assert dashboard._filter_summary_text(self.T, where, True, None, 5, 5) == (
            "Showing every trade of the run at the primary spread band max(tier,0)-1, "
            f"k = 0.75, 20% cap per trade: 5 trades. {dashboard._BAR_REACH}")
        # What the bar reaches: the interval-discount section by k and size
        # cap only — at the primary band with the tier floors on, never the
        # Tier floors choice — the scenario explorer's selects by band, Tier
        # floors choice, k and size cap, category and tag neither
        assert dashboard._BAR_REACH == (
            "The Interval Discount section follows only the bar's k and size cap (at the "
            "primary spread band, tier floors on — never its Tier floors choice), and the "
            "Scenario Explorer's selects follow its band, Tier floors choice, k and size "
            "cap; category and tag reach neither.")
        # ...and on a page that lacks either — the interval-discount section
        # cannot follow the bar (no sweep, or its data could not be built),
        # or there is no explorer grid (no band sweep, or its data could not
        # be built) — the sentence says so
        assert dashboard._BAR_REACH_NO_EXPLORER == (
            "The Interval Discount section follows only the bar's k and size cap (at the "
            "primary spread band, tier floors on — never its Tier floors choice), not its "
            "category or tag; this page has no Scenario Explorer grid for the bar to reach.")
        assert dashboard._BAR_REACH_EXPLORER_ONLY == (
            "The Scenario Explorer's selects follow the bar's band, Tier floors choice, k and "
            "size cap (not its category or tag); the bar does not reach the Interval Discount "
            "section on this page.")
        assert dashboard._BAR_REACH_STATIC == (
            "The bar reaches neither the Interval Discount section on this page nor a "
            "Scenario Explorer grid.")
        # An explorer whose cap axis is not the bar's (rebuilt at the run's
        # own cap) follows the bar's band, Tier floors choice and k only, and
        # the line says so
        assert dashboard._BAR_REACH_OWN_CAP == (
            "The Interval Discount section follows only the bar's k and size cap (at the "
            "primary spread band, tier floors on — never its Tier floors choice), and the "
            "Scenario Explorer's selects follow its band, Tier floors choice and k but not "
            "its size cap — the explorer shows the run's own size cap only on this page; "
            "category and tag reach neither.")
        assert dashboard._BAR_REACH_EXPLORER_ONLY_OWN_CAP == (
            "The Scenario Explorer's selects follow the bar's band, Tier floors choice and k "
            "(not its size cap — the explorer shows the run's own size cap only on this page "
            "— nor its category or tag); the bar does not reach the Interval Discount section "
            "on this page.")
        assert [dashboard._bar_reach(kd, explorer) for kd, explorer in
                ((True, True), (True, False), (False, True), (False, False))] == [
            dashboard._BAR_REACH, dashboard._BAR_REACH_NO_EXPLORER,
            dashboard._BAR_REACH_EXPLORER_ONLY, dashboard._BAR_REACH_STATIC]
        assert [dashboard._bar_reach(kd, explorer, explorer_caps=False) for kd, explorer in
                ((True, True), (True, False), (False, True), (False, False))] == [
            dashboard._BAR_REACH_OWN_CAP, dashboard._BAR_REACH_NO_EXPLORER,
            dashboard._BAR_REACH_EXPLORER_ONLY_OWN_CAP, dashboard._BAR_REACH_STATIC]

        other = dashboard._filter_summary_text(
            self.T, dashboard._scenario_phrase("spread band x", "k = 0.60", "5% cap per trade"),
            False, None, 3, 3)
        assert ("k = 0.60, 5% cap per trade: 3 trades. This spread band, k and size cap "
                "is its own simulation, not a slice of the primary run.") in other
        sliced = dashboard._filter_summary_text(self.T, where, True, "Sports · Hockey", 1, 5)
        assert sliced.startswith("Showing Sports · Hockey within the run at the primary "
                                 "spread band max(tier,0)-1, k = 0.75, 20% cap per trade: "
                                 "1 of its 5 trades. ")
        # Every figure a slice draws from an equity curve is its contribution
        for figure in ("return", "drawdown", "Sharpe", "Sortino", "median monthly return",
                       "benchmark's strategy row"):
            assert figure in sliced

    def test_every_reach_says_the_interval_discount_ignores_the_tier_choice(self):
        # An explorer with no Tier floors axis while the bar offers a
        # tier-floors-off view (explorer_tiers False): the explorer's clause
        # says it does not follow that choice, pinned by literal
        no_tiers = {
            (True, True): (
                "The Interval Discount section follows only the bar's k and size cap (at "
                "the primary spread band, tier floors on — never its Tier floors choice), "
                "and the Scenario Explorer's selects follow its band, k and size cap but not "
                "its Tier floors choice; category and tag reach neither."),
            (True, False): (
                "The Interval Discount section follows only the bar's k and size cap (at "
                "the primary spread band, tier floors on — never its Tier floors choice), "
                "and the Scenario Explorer's selects follow its band and k but neither its "
                "size cap nor its Tier floors choice — the explorer shows the run's own "
                "size cap only on this page; category and tag reach neither."),
            (False, True): (
                "The Scenario Explorer's selects follow the bar's band, k and size cap (not "
                "its Tier floors choice, category or tag); the bar does not reach the "
                "Interval Discount section on this page."),
            (False, False): (
                "The Scenario Explorer's selects follow the bar's band and k (not its size "
                "cap — the explorer shows the run's own size cap only on this page — nor its "
                "Tier floors choice, category or tag); the bar does not reach the Interval "
                "Discount section on this page."),
        }
        for (kd, caps), sentence in no_tiers.items():
            assert dashboard._bar_reach(kd, True, explorer_caps=caps,
                                        explorer_tiers=False) == sentence, (kd, caps)
        # Without an explorer the tier setting changes nothing
        for kd in (True, False):
            assert dashboard._bar_reach(kd, False, explorer_tiers=False) == \
                dashboard._bar_reach(kd, False)
        # Every sentence either follows the bar with the tier floors on —
        # never its Tier floors choice — or says the bar does not reach the
        # Interval Discount section at all
        every = [dashboard._bar_reach(kd, ex, explorer_caps=caps, explorer_tiers=tiers)
                 for kd in (True, False) for ex in (True, False)
                 for caps in (True, False) for tiers in (True, False)]
        for sentence in every:
            assert ("tier floors on — never its Tier floors choice" in sentence
                    or "not reach the Interval Discount section" in sentence
                    or "reaches neither the Interval Discount section" in sentence), sentence

    def test_a_view_names_its_own_closing_note(self):
        # The closing note the script picks for the view shown: the primary
        # band, k and cap with the tier floors off where they bind is its own
        # simulation (other_run); any other scenario other_scenario; the
        # primary itself none
        where = dashboard._scenario_phrase(
            f"the primary spread band 0-1 with the {dashboard._TIER_FLOORS} tier floors off",
            "k = 0.75", "20% cap per trade")
        assert dashboard._filter_summary_text(self.T, where, True, None, 6, 6,
                                              note="other_run") == (
            f"Showing every trade of the run at {where}: 6 trades. This is its own "
            f"simulation, not a slice of the primary run. {dashboard._BAR_REACH}")
        # No note named: the tier-on rule, unchanged
        assert dashboard._filter_summary_text(self.T, where, False, None, 6, 6) == \
            dashboard._filter_summary_text(self.T, where, True, None, 6, 6,
                                           note="other_scenario")
        assert dashboard._filter_summary_text(self.T, where, True, None, 6, 6) == (
            f"Showing every trade of the run at {where}: 6 trades. {dashboard._BAR_REACH}")
        # A slice never carries one
        assert "own simulation" not in dashboard._filter_summary_text(
            self.T, where, True, "Sports", 1, 6, note="other_run")

    def test_counts_are_worded(self):
        assert dashboard._trade_count(1) == "1 trade"
        assert dashboard._trade_count(0) == "0 trades"
        one = dashboard._filter_summary_text(self.T, "x", True, None, 1, 1)
        assert "x: 1 trade." in one

    def test_a_run_without_a_band_says_so(self, monkeypatch, tmp_path):
        trades = _flt_trades()
        curve = backtester._build_equity_curve(trades, _FLT_START, 1000.0)
        monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(dashboard.yf, "download",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
        page = dashboard.generate_dashboard(trades, curve, _FLT_START, 1000.0) \
            .read_text(encoding="utf-8")
        assert ("Showing every trade of the run at the primary spread band (not recorded), "
                "k not recorded, cap not recorded: 5 trades.") in page

    def test_every_cap_is_named_for_the_select_and_the_summary(self):
        assert [dashboard._cap_option(c) for c in (0.05, 0.2, 0.55, 1.0, None)] == [
            "5%", "20%", "55%", "off (full Kelly)", "not recorded"]
        assert [dashboard._cap_text(c) for c in (0.05, 0.2, 1.0, None)] == [
            "5% cap per trade", "20% cap per trade", "no per-trade cap", "cap not recorded"]
        # Injective, like the completion lines' labels: a double beside 0.2 is
        # never printed as 20%
        assert dashboard._cap_option(0.19999999999999998) != dashboard._cap_option(0.2)
        assert dashboard._cap_text(0.19999999999999998) != dashboard._cap_text(0.2)

    def test_the_script_fills_the_templates_and_writes_none_of_its_own(self):
        js = dashboard._FILTER_JS
        for template in ("T.all", "T.slice", "T.unfiltered", "T.missing", "T.other_scenario",
                         "T.other_run", "D.text.scenario", "D.text.loading",
                         "D.text.unavailable", "D.text.khat_sized_at", "T.sell_note",
                         "D.text.sidecar_missing", "D.text.sidecar_empty",
                         "D.sell_levels[level].phrase", "D.sell_days[parseInt(d, 10)].phrase",
                         # The menus' and a mix's words
                         "T.mix_note", "D.text.mix_unavailable", "D.text.names_join",
                         "D.text.names_more", "T.menu_cat_all", "T.menu_cat_n",
                         "T.menu_tag_all", "T.menu_tag_n", "T.cal_title", "T.cal_title_none",
                         "T.cal_bin", "D.text.khat_bar"):
            assert template in js
        # Nor the tier-floors-off views' words, nor the Sell and Min. days
        # selects': Python's scenario, note, sell-level, minimum-days and k-hat
        # title templates carry every one
        for phrase in ("Showing", "contribution", "Not filtered", "its own simulation",
                       "Loading", "not simulated", "cap per trade", "sized at",
                       "never bind", "floor alone", "tier floors off", "selling",
                       "potential profit", "held no data", "sold on", "keep the folder",
                       "last market closes", "at least",
                       # Nor the check-box menus' words or a mix's: the menu
                       # labels, the note and the refusal are Python's too
                       "All categories", "All tags", "Combined", "cannot be combined",
                       "Calibration Curve", "Brier", "LogLoss"):
            assert phrase not in js, phrase


# ─── The filter script itself, run outside a browser ─────────────────────────

_JS_HARNESS = Path(__file__).parent / "js" / "filter_harness.js"
_JSC = Path("/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc")


def _js_runtime() -> str | None:
    """node when installed (CI's runners have it), else macOS's JavaScriptCore
    shell; None when neither is present."""
    node = shutil.which("node")
    if node:
        return node
    return str(_JSC) if _JSC.exists() else None


def _page_elements(page: str) -> dict:
    """The selects (options, "selected", "disabled"), the buttons ("disabled")
    and their hover texts ("titles": button id -> its title attribute),
    every chart (data and layout, typed arrays decoded) and every element id
    of a rendered page — the ids are the only elements a strict-mode run lets
    the script reach. A select's or button's id may be double-quoted (the
    filter bar's) or single-quoted (the scenario explorer's); a button is
    disabled when its tag carries the bare "disabled" attribute (attribute
    values, such as its title, are not read for it)."""
    selects = {}
    for m in re.finditer(r"""<select id=(["'])([^"']+)\1([^>]*)>(.*?)</select>""", page, re.S):
        options = [{"value": value, "text": html.unescape(text), "selected": bool(chosen)}
                   for value, chosen, text in re.findall(
                       r'<option value="([^"]*)"( selected)?>(.*?)</option>', m.group(4))]
        selects[m.group(2)] = {"options": options, "disabled": " disabled" in m.group(3)}
    # The check-box menus (<details>): the button's text, the clearing box and
    # every row — its text, and whether Python rendered it ticked, disabled
    # or hidden. Each is also listed among the selects, its rows as options
    # (the clearing box first, value ""), so a test reads a menu's options as
    # it reads a select's; "menu" tells the harness to build the real thing
    menus = {}
    for m in re.finditer(r'<details id="([^"]+)" class="([^"]*)"[^>]*>(.*?)</details>', page, re.S):
        menu_id, body = m.group(1), m.group(3)
        label = re.search(rf'<summary id="{menu_id}-label">(.*?)</summary>', body).group(1)
        clear = re.search(rf'<label><input type="checkbox" id="{menu_id}-all"([^>]*)> '
                          r"(.*?)</label>", body)
        rows = [{"text": html.unescape(text), "checked": " checked" in attrs,
                 "disabled": " disabled" in attrs, "hidden": bool(hidden)}
                for hidden, attrs, text in re.findall(
                    rf'<label id="{menu_id}-\d+-row"( hidden)?><input type="checkbox" '
                    rf'id="{menu_id}-\d+"([^>]*)> <span id="{menu_id}-\d+-text">(.*?)</span>'
                    r"</label>", body)]
        menus[menu_id] = {
            "label": html.unescape(label), "className": m.group(2), "rows": rows,
            "all": {"text": html.unescape(clear.group(2)),
                    "checked": " checked" in clear.group(1),
                    "disabled": " disabled" in clear.group(1)}}
        selects[menu_id] = {
            "menu": True, "disabled": menus[menu_id]["all"]["disabled"],
            "options": [{"value": "", "text": menus[menu_id]["all"]["text"],
                         "selected": menus[menu_id]["all"]["checked"]}]
            + [{"value": str(i), "text": row["text"], "selected": row["checked"]}
               for i, row in enumerate(rows)]}
    buttons, titles = {}, {}
    for m in re.finditer(r"""<button id=(["'])([^"']+)\1([^>]*)>""", page):
        # The attribute values blanked, so a title's words never read as the attribute
        attributes = re.sub(r'"[^"]*"', '""', m.group(3))
        buttons[m.group(2)] = {"disabled": bool(re.search(r"\sdisabled(?=[\s/]|$)", attributes))}
        title = re.search(r'\stitle="([^"]*)"', m.group(3))
        titles[m.group(2)] = html.unescape(title.group(1)) if title else ""
    charts, decoder = {}, json.JSONDecoder()
    for m in re.finditer(r'Plotly\.newPlot\(\s*"([^"]+)",', page):
        i, args = m.end(), []
        while len(args) < 2:
            i = dashboard_golden._skip_whitespace(page, i)
            if page[i] == ",":
                i += 1
                continue
            value, i = decoder.raw_decode(page, i)
            args.append(dashboard_golden._decode_typed_arrays(value))
        charts[m.group(1)] = {"data": args[0], "layout": args[1]}
    ids = sorted(set(re.findall(r"""\bid=["']([^"']+)["']""", page)))
    return {"selects": selects, "menus": menus, "buttons": buttons, "titles": titles,
            "charts": charts, "ids": ids}


def _script_body() -> str:
    """dashboard._FILTER_JS without its <script> tags, its inflate(el) handing
    back the already-inflated block of that element (the harness's
    __inflate, over every packed block of the page; the blocks' encoding is
    pinned by TestFilterPage's strict decodes) and its inflateText(text)
    handing back the chunk a sidecar file passed over, already inflated (the
    harness's __inflateText; the files' encoding is pinned by
    _sidecar_files' strict decodes)."""
    body = dashboard._FILTER_JS.strip()
    body = body[len("<script>"):-len("</script>")]
    for opening, stand_in in (
            ("  function inflate(el) {", "  function inflate(el) { return __inflate(el); }\n"),
            ("  function inflateText(text) {",
             "  function inflateText(text) { return __inflateText(text); }\n")):
        start = body.index(opening)
        end = body.index("\n  }\n", start) + len("\n  }\n")
        body = body[:start] + stand_in + body[end:]
    return body


# One sidecar chunk file's text: the call that hands its packed block over
_SIDECAR_CALL = re.compile(r'window\.__dashChunk\(document\.currentScript,"([A-Za-z0-9+/=]*)"\);\n')


def _sidecar_files(page: str, root: Path) -> dict[str, dict]:
    """Every sidecar chunk file of a page — in the folder its base block names
    (sidecar_dir), under `root` (the folder the page was written to) — decoded
    strictly, by the src the page's script asks for."""
    data = _packed_blocks(page).get("dash-data") or {}
    folder = data.get("sidecar_dir")
    if not folder:
        return {}
    out = {}
    for path in sorted((root / folder).glob("chunk-*.js")):
        match = _SIDECAR_CALL.fullmatch(path.read_text(encoding="utf-8"))
        assert match, path
        out[f"{folder}/{path.name}"] = _decode_block(match.group(1))
    return out


def _explorer_body() -> str:
    """dashboard._SCENARIO_EXPLORER_JS without its <script> tags, its
    unpack(el) handing back the already-inflated block of that element (the
    harness's __inflate, as for the filter's inflate)."""
    body = dashboard._SCENARIO_EXPLORER_JS.strip()
    body = body[len("<script>"):-len("</script>")]
    start = body.index("  function unpack(el) {")
    end = body.index("\n  }\n", start) + len("\n  }\n")
    return body[:start] + "  function unpack(el) { return __inflate(el); }\n" + body[end:]


def _run_script(tmp_path: Path, page: str, steps: list, pre: tuple = (),
                no_decompression: bool = False, *, strict: bool = False,
                damaged: tuple = (), keep: int | None = None,
                deferred: tuple = (), explorer: bool = False,
                strict_ids: bool = False, setup_js: str = "",
                files_root: Path | None = None, files_missing: tuple = (),
                files_empty: tuple = (), files_deferred: tuple = ()) -> dict:
    """
    Run the page's scripts — the filter script, and with explorer=True the
    scenario explorer's before it — over a rendered page under
    tests/js/filter_harness.js.

    The filter script runs only when the page carries it (its base block): a
    page whose filter could not be built is written without the bar and its
    script, so it runs without them here too.

    Args:
        tmp_path (Path): Where the assembled program is written.
        page (str): The rendered page.
        steps (list): The harness's steps (["wait"], ["settle"], ["set", id,
            value], ["fire", id], ["zoom", id], ["hide", id], ["select", id],
            ["repair", id], ["resolve", id], ["reject", id], ["call", name,
            args] — a page script's window function called with args, as
            another script would call it — ["click", id] — a reader clicking
            that element, its click listeners run unless it is disabled —
            ["tick", id] — a reader clicking a check box of the Category or
            Tag menu (_tick_category, _tick_tag) — ["open", id], ["outside",
            id] and ["key", name] — a reader opening a menu, clicking
            elsewhere on the page and pressing a key — and ["snap", name];
            each snapshot also carries "buttons" (id -> disabled), "titles"
            (id -> a button's hover text as it stands), "menus"
            (each check-box menu's button text, ticks and rows) and "opened",
            the window.open calls since the last snapshot as {url, target,
            features}).
        pre (tuple): (select id, value) pairs set BEFORE the scripts run — a
            browser restoring a reader's last choice (for a check-box menu,
            the ticked row indexes joined by commas).
        no_decompression (bool): Run as a browser without DecompressionStream.
        strict (bool): Keyword-only. getElementById returns null for an id
            the page does not hold, as a browser's does, so a script that
            reaches a missing element fails the run.
        damaged (tuple): Keyword-only. Block ids whose inflate throws (until a
            ["repair", id] step), as a damaged block's would.
        keep (int | None): Keyword-only. Replaces the script's KEEP (drawn
            chunks kept besides the primary's).
        deferred (tuple): Keyword-only. Block ids whose inflate waits for a
            ["resolve", id] or ["reject", id] step, so a test can choose the
            order in which chunks arrive.
        explorer (bool): Keyword-only. Also run the scenario explorer's
            script, before the filter's, as the page orders them — when the
            page carries it (a page without the band sweep does not, and
            then only the filter's runs, as in a browser); "wait" then also
            waits for the explorer's selects. False (default) runs the
            filter's alone.
        strict_ids (bool): Keyword-only. #68's name for strict mode: the same
            as strict=True (either one switches it on).
        setup_js (str): Keyword-only. JavaScript run after the page is set up
            and before the scripts — to take something off the page Python
            drew.
        files_root (Path | None): Keyword-only. The folder the page was
            written to: its sidecar chunk files (the Sell select's, in the
            folder the base block names) are handed to the harness, which
            loads one when the script appends its <script src> element
            (["resolve_file", src] / ["reject_file", src] for a deferred one;
            each snapshot's "files" lists every src loaded). None (default)
            hands none over, so every sidecar load fails as a missing file's.
        files_missing (tuple): Keyword-only. Srcs to leave out, as files that
            cannot be loaded.
        files_empty (tuple): Keyword-only. Srcs that run without handing
            anything over.
        files_deferred (tuple): Keyword-only. Srcs whose load waits for a
            ["resolve_file", src] or ["reject_file", src] step.

    Returns:
        dict: The snapshots the steps took, by name.
    """
    runtime = _js_runtime()
    if runtime is None:
        pytest.skip("no JavaScript runtime (node or jsc) to run the filter script with")
    elements = _page_elements(page)
    elements["strict"] = strict or strict_ids
    body = _script_body()
    if keep is not None:
        assert body.count("var KEEP = 16;") == 1
        body = body.replace("var KEEP = 16;", f"var KEEP = {int(keep)};")
    with_explorer = explorer and "function unpack(el)" in page
    has_filter = 'id="dash-data"' in page
    sidecars = {} if files_root is None else _sidecar_files(page, files_root)
    for src in files_missing:
        sidecars.pop(src, None)
    program = "\n".join([
        _JS_HARNESS.read_text(encoding="utf-8"),
        f"var __PAGE = {json.dumps(elements)};",
        f"__BLOCKS = {json.dumps(_packed_blocks(page))};",
        f"__SIDECARS = {json.dumps(sidecars)};",
        *(f"__SIDECAR_EMPTY[{json.dumps(src)}] = true;" for src in files_empty),
        *(f"__SIDECAR_DEFERRED[{json.dumps(src)}] = true;" for src in files_deferred),
        *(f"__DAMAGED[{json.dumps(i)}] = true;" for i in damaged),
        *(f"__DEFERRED[{json.dumps(i)}] = true;" for i in deferred),
        "__setup(__PAGE);",
        "__WAIT_EXPLORER = true;" if with_explorer else "",
        *(f"document.getElementById({json.dumps(i)}).value = {json.dumps(v)};" for i, v in pre),
        setup_js,
        "window.DecompressionStream = undefined;" if no_decompression else "",
        _explorer_body() if with_explorer else "",
        body if has_filter else "",
        f"__step({json.dumps(steps)}, 0);",
    ])
    path = tmp_path / "filter_run.js"
    path.write_text(program, encoding="utf-8")
    done = subprocess.run([runtime, str(path)], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def _last_react(snap: dict, chart: str) -> dict:
    """The last redraw of a chart in a snapshot (earlier steps may redraw it too)."""
    return [r for r in snap["reacts"] if r["id"] == chart][-1]


def _charts_redrawn() -> set[str]:
    """The charts _FILTER_JS captures on load, which are those it redraws."""
    listed = re.search(r"var CHARTS = \[(.*?)\];", dashboard._FILTER_JS, re.S).group(1)
    return set(re.findall(r"'([a-z-]+)'", listed))


# The bar's selects and its two check-box menus, which a snapshot also reads
# as selects: "value" the ticked row indexes, comma-joined ("" for none),
# "options" the clearing box and every row not hidden, "disabled" the boxes'
_FLT_SELECTS = ("flt-band", "flt-k", "flt-cap", "flt-cat", "flt-tag")


def _tick_category(index) -> list:
    """The step of a reader clicking one box of the Category menu: that
    category's (its index, a number or its text), or with "" the box that
    clears the menu ("All categories")."""
    return ["tick", "flt-cat-all" if index == "" else f"flt-cat-{index}"]


def _tick_tag(index) -> list:
    """The step of a reader clicking one box of the Tag menu: that tag's (its
    index in the base block's subcats), or with "" the box that clears the
    menu ("All tags")."""
    return ["tick", "flt-tag-all" if index == "" else f"flt-tag-{index}"]


class TestFilterScript:
    """The page-wide filter script, run over the page Python rendered: what it
    draws, writes and enables for each choice. Skipped without a runtime."""

    def _page(self, monkeypatch, tmp_path, sweep=None, series=_FLT_SERIES) -> str:
        return TestFilterPage()._page(monkeypatch, tmp_path, sweep, series)

    def test_on_load_the_bar_is_reset_disabled_and_nothing_is_drawn(
            self, monkeypatch, tmp_path):
        # A browser restored a reader's last choice (band 1, k 0, category 1,
        # tier floors off)
        page = self._page(monkeypatch, tmp_path)
        data = TestFilterPage._data(page)
        sports = str(data["categories"].index("Sports"))
        snaps = _run_script(tmp_path, page, [
            ["snap", "loaded"], ["wait"], ["snap", "ready"],
            # The first choice after the bar is enabled, with no time to load
            # anything: the primary scenario's chunk is already there
            _tick_category(sports), ["snap", "at_once"]],
            pre=(("flt-band", "1"), ("flt-k", "0"), ("flt-cat", "1"), ("flt-tier", "off")))
        loaded, ready, at_once = snaps["loaded"], snaps["ready"], snaps["at_once"]
        assert {i: loaded["selects"][i]["value"] for i in _FLT_SELECTS} == {
            "flt-band": "0", "flt-k": "1", "flt-cap": "0", "flt-cat": "", "flt-tag": ""}
        assert all(loaded["selects"][i]["disabled"] for i in _FLT_SELECTS)
        assert not any(ready["selects"][i]["disabled"] for i in _FLT_SELECTS)
        # The tier select is set back to "on" too; this run has no tier-off
        # view, so it stays disabled once the data is in
        assert loaded["selects"]["flt-tier"]["value"] == "on"
        assert loaded["selects"]["flt-tier"]["disabled"] and ready["selects"]["flt-tier"]["disabled"]
        assert loaded["reacts"] == ready["reacts"] == []
        # Enabled only once the base block AND the primary scenario's chunk
        # were inflated — and nothing else was
        assert loaded["inflated"] == []
        assert ready["inflated"] == ["dash-data", "dash-chunk-0"]
        # ... so a category chosen at once is drawn at once, never "Loading"
        assert at_once["text"]["flt-summary"] == dashboard._filter_summary_text(
            data["text"], _phrase(data, *_FLT_PRIMARY), True, "Sports", 3, 5)
        assert at_once["text"]["hdr-trades"] == "3"
        assert at_once["inflated"] == ["dash-data", "dash-chunk-0"]

    def test_every_redraw_is_the_chart_python_drew_for_that_view(
            self, monkeypatch, tmp_path):
        # Away to another band and back: the primary's unfiltered view, drawn
        # by the script, must be the page Python rendered
        page = self._page(monkeypatch, tmp_path)
        snap = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-band", "1"], ["fire", "flt-band"], ["settle"],
            ["set", "flt-band", "0"], ["fire", "flt-band"], ["settle"], ["snap", "back"]])["back"]
        charts = _page_elements(page)["charts"]
        drawn = {r["id"]: r for r in snap["reacts"]}
        assert set(drawn) == _charts_redrawn()
        for cid, react in drawn.items():
            python = charts[cid]
            assert react["layout"] == python["layout"], cid
            for i, trace in enumerate(python["data"]):
                got = react["data"][i]
                for key in ("x", "y", "text", "name"):
                    if isinstance(trace.get(key), list) and trace[key] \
                            and isinstance(trace[key][0], (int, float)):
                        assert got[key] == pytest.approx(trace[key], abs=0.006), (cid, i, key)
                    elif key in trace:
                        assert got[key] == trace[key], (cid, i, key)
        kpis = dict(re.findall(r'<div id="kpi-([a-z_]+)" style="[^"]*">(.*?)</div>', page))
        assert {k[4:]: v for k, v in snap["text"].items() if k.startswith("kpi-")} == kpis
        assert snap["text"]["hdr-trades"] == "5"
        assert html.escape(snap["text"]["flt-summary"]) in page
        assert f'<div id="dec-table">{snap["html"]["dec-table"]}</div>' in page
        assert f'<tbody id="diag-best">{snap["html"]["diag-best"]}</tbody>' in page

    def test_a_zoom_or_a_hidden_line_is_not_carried_into_the_next_view(
            self, monkeypatch, tmp_path):
        page = self._page(monkeypatch, tmp_path)
        snap = _run_script(tmp_path, page, [
            ["wait"], ["zoom", "perf-cum"], ["hide", "perf-cum"],
            ["select", "risk-kelly"],
            ["set", "flt-band", "1"], ["fire", "flt-band"], ["settle"],
            ["snap", "after"]])["after"]
        react = _last_react(snap, "perf-cum")
        python = _page_elements(page)["charts"]["perf-cum"]["layout"]
        assert react["layout"].get("xaxis") == python.get("xaxis")
        assert (react["layout"].get("xaxis") or {}).get("autorange") is not False
        assert "visible" not in react["data"][0]
        assert "selectedpoints" not in _last_react(snap, "risk-kelly")["data"][0]
        # Another band's whole run, in the same words Python would use
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        n = _view(data, chunks, 1, 1, 0, "all")["n"]
        assert snap["text"]["flt-summary"] == dashboard._filter_summary_text(
            data["text"], _phrase(data, 1, 1, 0), False, None, n, n)

    def test_a_tag_picked_under_all_categories_selects_its_category(
            self, monkeypatch, tmp_path):
        page = self._page(monkeypatch, tmp_path)
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        sports = data["categories"].index("Sports")
        hockey = data["subcats"].index([sports, "Hockey"])
        snap = _run_script(tmp_path, page, [
            ["wait"], _tick_tag(str(hockey)), ["settle"],
            ["snap", "s"]])["s"]
        assert snap["selects"]["flt-cat"]["value"] == str(sports)
        assert snap["selects"]["flt-tag"]["value"] == str(hockey)
        assert [text for _, text in snap["selects"]["flt-tag"]["options"]] == [
            "All tags", "Basketball (2)", "Hockey (1)"]
        assert snap["text"]["flt-summary"] == dashboard._filter_summary_text(
            data["text"], _phrase(data, *_FLT_PRIMARY), True, "Sports · Hockey", 1, 5)
        assert snap["text"]["hdr-trades"] == snap["text"]["kpi-trades"] == "1"
        # The row-dependent chart's height lands on both boxes, whichever one
        # the page's plotly.py wrote it on
        h = _view(data, chunks, *_FLT_PRIMARY, f"s{hockey}")["sub"]["h"]
        assert snap["heights"]["dec-sub"] == snap["ownHeights"]["dec-sub"] == f"{h}px"

    def test_a_band_without_the_selection_shows_no_trades(self, monkeypatch, tmp_path):
        # Band 1 holds only same-title trades: none of them is in "Other"
        page = self._page(monkeypatch, tmp_path)
        data = TestFilterPage._data(page)
        other = str(data["categories"].index("Other"))
        snap = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-band", "1"], ["fire", "flt-band"], ["settle"],
            _tick_category(other), ["settle"], ["snap", "s"]])["s"]
        assert [text for _, text in snap["selects"]["flt-cat"]["options"]] == [
            "All categories", "Commodities (1)", "Other (0)", "Science (0)", "Sports (2)"]
        for prefix in ("dec", "cal", "diag", "risk"):
            assert (snap["display"][f"{prefix}-empty"], snap["display"][f"{prefix}-body"]) \
                == ("", "none")
        assert snap["text"]["hdr-trades"] == "0"
        assert snap["text"]["kpi-max_drawdown"] == "0.0%"
        # A slice never claims to be a whole simulation
        assert "This spread band, k and size cap is its own simulation" \
            not in snap["text"]["flt-summary"]

    def test_the_khat_table_redraws_as_python_rendered_it(self, monkeypatch, tmp_path):
        page = self._page(monkeypatch, tmp_path)
        snap = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-band", "1"], ["fire", "flt-band"], ["settle"],
            ["set", "flt-band", "0"], ["fire", "flt-band"], ["settle"], ["snap", "back"]])["back"]
        body = re.search(r'<tbody id="khat-rows">(.*?)</tbody>', page, re.S).group(1)
        python_rows = [[html.unescape(c) for c in re.findall(r"<td[^>]*>(.*?)</td>", tr)]
                       for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", body)]
        assert snap["rows"]["khat-rows"] == python_rows
        assert snap["display"]["khat-body"] == "" and snap["display"]["khat-empty"] == "none"

    def test_group_by_stays_usable_when_a_selection_has_no_khat(self, monkeypatch, tmp_path):
        # Commodities has trades but no k-hat entry: grouped by tag it draws
        # nothing, and the reader must be able to group by category again
        page = self._page(monkeypatch, tmp_path)
        data = TestFilterPage._data(page)
        commodities = str(data["categories"].index("Commodities"))
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["set", "khat-group", "tag"], ["fire", "khat-group"],
            _tick_category(commodities), ["settle"], ["snap", "empty"],
            ["set", "khat-group", "category"], ["fire", "khat-group"], ["snap", "back"]])
        empty, back = snaps["empty"], snaps["back"]
        assert (empty["display"]["khat-body"], empty["display"]["khat-empty"]) == ("none", "")
        assert empty["text"]["khat-empty"] == data["text"]["khat_none"]
        assert not empty["selects"]["khat-group"]["disabled"]
        assert back["display"]["khat-body"] == ""
        react = _last_react(back, "khat-fig")
        assert react["data"][0]["y"] == ["All categories", "Other", "Science", "Sports"]

    def test_grouping_by_band_highlights_the_selected_band(self, monkeypatch, tmp_path):
        page = self._page(monkeypatch, tmp_path)
        data = TestFilterPage._data(page)
        sports = str(data["categories"].index("Sports"))
        snap = _run_script(tmp_path, page, [
            ["wait"], _tick_category(sports), ["settle"],
            ["set", "khat-group", "band"], ["fire", "khat-group"], ["snap", "s"]])["s"]
        react = _last_react(snap, "khat-fig")
        colors = data["styles"]["khat"]
        assert react["data"][0]["y"] == [b["label"] for b in data["bands"]]
        assert react["data"][0]["marker"]["color"] == [colors["selected"], colors["bar"],
                                                        colors["bar"]]
        assert react["layout"]["title"]["text"] == "Empirical k̂ by spread band — Sports"
        assert react["layout"]["height"] == dashboard._khat_chart_height(3)
        assert snap["heights"]["khat-fig"] == snap["ownHeights"]["khat-fig"] \
            == f"{dashboard._khat_chart_height(3)}px"
        # The two bands with no calibration: a row each, every cell blank
        assert snap["rows"]["khat-rows"][1:] == [[b["label"], *data["khat_blank"]]
                                                 for b in data["bands"][1:]]

    def test_grouping_by_tag_lists_the_selected_categorys_tags(self, monkeypatch, tmp_path):
        page = self._page(monkeypatch, tmp_path)
        data = TestFilterPage._data(page)
        sports = data["categories"].index("Sports")
        hockey = str(data["subcats"].index([sports, "Hockey"]))
        snap = _run_script(tmp_path, page, [
            ["wait"], _tick_tag(hockey), ["settle"],
            ["set", "khat-group", "tag"], ["fire", "khat-group"], ["snap", "s"]])["s"]
        react = _last_react(snap, "khat-fig")
        colors = data["styles"]["khat"]
        assert react["data"][0]["y"] == ["All Sports", "Hockey"]
        assert react["data"][0]["marker"]["color"] == [colors["all"], colors["selected"]]
        assert react["layout"]["title"]["text"] == (
            "Empirical k̂ by tag — the primary spread band max(tier,0)-1")

    def test_a_band_without_a_calibration_says_so(self, monkeypatch, tmp_path):
        page = self._page(monkeypatch, tmp_path)
        data = TestFilterPage._data(page)
        snap = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-band", "1"], ["fire", "flt-band"], ["settle"],
            ["snap", "s"]])["s"]
        assert (snap["display"]["khat-body"], snap["display"]["khat-empty"]) == ("none", "")
        assert snap["text"]["khat-empty"] == data["text"]["khat_not_recorded"]

    def test_tier_floors_off_switches_every_filtered_section_and_back(
            self, monkeypatch, tmp_path):
        page = self._page(monkeypatch, tmp_path, _flt_sweep_tiers(), _FLT_SERIES_TIERS)
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        python = _page_elements(page)
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["snap", "ready"],
            ["set", "flt-tier", "off"], ["fire", "flt-tier"], ["settle"], ["snap", "off"],
            ["set", "flt-tier", "on"], ["fire", "flt-tier"], ["settle"], ["snap", "on"]])
        ready, off, on = snaps["ready"], snaps["off"], snaps["on"]
        assert not ready["selects"]["flt-tier"]["disabled"]
        n = len(data["dates"])
        off_list = chunks[data["grid_off"][0][1][0]]["list"]
        n_off = off_list["views"]["all"]["n"]
        assert n_off == 6
        # The primary band, k and cap with the tiers off: its own simulation,
        # in Python's words
        assert off["text"]["flt-summary"] == dashboard._filter_summary_text(
            data["text"], _phrase_off(data, *_FLT_PRIMARY), True, None, n_off, n_off,
            note="other_run")
        assert off["text"]["hdr-trades"] == off["text"]["kpi-trades"] == "6"
        assert [text for _, text in off["selects"]["flt-band"]["options"]] == [
            "0-1 (primary)", "0.3-0.6", "0.3-1"]
        # Category counts come from the tier-off list
        cats = [text for _, text in off["selects"]["flt-cat"]["options"]]
        assert cats == ["All categories"] + [
            f"{c} ({off_list['views'][f'c{i}']['n'] if f'c{i}' in off_list['views'] else 0})"
            for i, c in enumerate(data["categories"])]
        assert "Climate (1)" in cats
        assert _last_react(off, "perf-cum")["data"][0]["y"] == pytest.approx(
            _expand(off_list["views"]["all"]["total"], n))
        # Back on: the page exactly as Python rendered it
        assert on["text"]["flt-summary"] == dashboard._filter_summary_text(
            data["text"], _phrase(data, *_FLT_PRIMARY), True, None, 5, 5)
        assert html.escape(on["text"]["flt-summary"]) in page
        assert on["text"]["hdr-trades"] == "5"
        for sel in ("flt-band", "flt-cat"):
            assert [text for _, text in on["selects"][sel]["options"]] == [
                o["text"] for o in python["selects"][sel]["options"]], sel
        assert _last_react(on, "perf-cum")["data"][0]["y"] == pytest.approx(
            python["charts"]["perf-cum"]["data"][0]["y"], abs=0.006)

    def test_without_a_family_the_tier_select_stays_shut(self, monkeypatch, tmp_path):
        page = self._page(monkeypatch, tmp_path)
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["snap", "ready"],
            ["set", "flt-tier", "off"], ["fire", "flt-tier"], ["settle"], ["snap", "fired"]])
        ready, fired = snaps["ready"], snaps["fired"]
        assert ready["selects"]["flt-tier"]["disabled"]
        assert not ready["selects"]["flt-band"]["disabled"]
        # Firing it anyway changes nothing: nothing redrawn, rewritten,
        # relabelled or loaded
        assert fired["reacts"] == [] and fired["text"] == ready["text"]
        assert fired["selects"]["flt-band"]["options"] == ready["selects"]["flt-band"]["options"]
        assert fired["inflated"] == ready["inflated"]

    def test_the_khat_chart_follows_the_tier_floors_choice(self, monkeypatch, tmp_path):
        page = self._page(monkeypatch, tmp_path, _flt_sweep_tiers(), _FLT_SERIES_TIERS)
        data = TestFilterPage._data(page)
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-tier", "off"], ["fire", "flt-tier"], ["settle"],
            ["snap", "by_category"],
            ["set", "khat-group", "band"], ["fire", "khat-group"], ["snap", "by_band"],
            ["set", "flt-tier", "on"], ["fire", "flt-tier"], ["settle"], ["snap", "by_band_on"]])
        # By category: the primary band's tier-off calibration, named so
        react = _last_react(snaps["by_category"], "khat-fig")
        assert react["layout"]["title"]["text"] == (
            f"Empirical k̂ by category — {data['bands_off'][0]['where']}")
        assert react["data"][0]["y"] == ["All categories", "Climate", "Sports"]
        # By band: every band as the tiers-off setting names it, the binding
        # band's bar from its tier-off calibration
        react = _last_react(snaps["by_band"], "khat-fig")
        first = data["khat_off"][0]["groups"]["all"]
        assert first != data["khat"][0]["groups"]["all"]
        assert react["data"][0]["y"] == ["0-1", "0.3-0.6", "0.3-1"]
        assert react["data"][0]["x"][0] == first["k"]
        assert (react["data"][0]["text"][0], react["data"][0]["customdata"][0]) == (
            first["text"], first["cells"])
        assert snaps["by_band"]["rows"]["khat-rows"][0] == ["0-1", *first["cells"]]
        # ... and its title names the setting, in Python's words — which a
        # band grouping's scope ("all categories") would not otherwise do
        T = data["text"]
        assert react["layout"]["title"]["text"] == T["khat_title"].format(
            group=T["khat_group_words"]["band"],
            scope=T["khat_scope_tier_off"].format(scope=T["khat_every_category"]))
        assert react["layout"]["title"]["text"] == (
            "Empirical k̂ by spread band — all categories, with the "
            f"{dashboard._TIER_FLOORS} tier floors off")
        # Back on, still by band: the title the tier-on bar has always drawn
        react = _last_react(snaps["by_band_on"], "khat-fig")
        assert react["layout"]["title"]["text"] == "Empirical k̂ by spread band — all categories"
        assert react["data"][0]["y"] == [b["label"] for b in data["bands"]]

    def test_the_khat_cards_follow_the_tier_floors_choice(self, monkeypatch, tmp_path):
        # The performance section's k-hat cards read the k-hat breakdown's
        # group for the view shown — with the tiers off, the tier-off one (a
        # four-entry tier-off population, whose k-hat differs from the
        # tier-on one's)
        cal = _flt_calibration([("KXRAIN-1", "Other"), ("KXNHLHART-27", "Other"),
                                ("KXRAIN-2", "Other"), ("KXNHLHART-28", "Other")])
        sweep = dataclasses.replace(_flt_sweep_tiers(),
                                    tier_off_calibrations_by_band={(0.0, 1.0): cal})
        page = self._page(monkeypatch, tmp_path, sweep, _FLT_SERIES_TIERS)
        data = TestFilterPage._data(page)
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["snap", "ready"],
            ["set", "flt-tier", "off"], ["fire", "flt-tier"], ["settle"], ["snap", "off"]])
        off_all = data["khat_off"][0]["groups"]["all"]
        on_all = data["khat"][0]["groups"]["all"]
        assert off_all["cells"][4] != on_all["cells"][4]
        assert snaps["off"]["text"]["kpi-khat"] == off_all["cells"][4]
        assert snaps["off"]["text"]["kpi-khat_delta"] == off_all["delta"][1][0]

    def test_with_the_tiers_off_a_slice_and_another_band_read_as_python_words_them(
            self, monkeypatch, tmp_path):
        page = self._page(monkeypatch, tmp_path, _flt_sweep_tiers(), _FLT_SERIES_TIERS)
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        climate = str(data["categories"].index("Climate"))
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-tier", "off"], ["fire", "flt-tier"], ["settle"],
            _tick_category(climate), ["settle"], ["snap", "slice"],
            ["set", "flt-band", "1"], ["fire", "flt-band"], ["settle"],
            _tick_category(""), ["settle"], ["snap", "band1"]])
        # A slice of the primary band's tier-off run: that run's scenario
        n_off = chunks[data["grid_off"][0][1][0]]["list"]["views"]["all"]["n"]
        assert snaps["slice"]["text"]["flt-summary"] == dashboard._filter_summary_text(
            data["text"], _phrase_off(data, *_FLT_PRIMARY), True, "Climate", 1, n_off)
        assert snaps["slice"]["text"]["hdr-trades"] == "1"
        # Another band, one the tiers never bind at: its own run (its tier-on
        # chunk), named as the tiers-off setting names it, closing on the
        # other-scenario note
        n1 = chunks[data["grid_off"][1][1][0]]["list"]["views"]["all"]["n"]
        assert data["grid_off"][1][1][0] == data["grid"][1][1][0]
        assert snaps["band1"]["text"]["flt-summary"] == dashboard._filter_summary_text(
            data["text"], _phrase_off(data, 1, 1, 0), False, None, n1, n1)
        assert snaps["band1"]["text"]["hdr-trades"] == str(n1)

    def test_a_non_binding_primary_with_the_tiers_off_is_the_primary(
            self, monkeypatch, tmp_path):
        # The primary band's floor sits at or above both tiers: with the
        # tiers off it shows the very run the page renders, and says so as
        # the primary (no closing note), naming the setting
        sweep = _flt_sweep_tiers()
        narrow = sweep.scenarios[1]
        sweep = dataclasses.replace(sweep, primary=narrow, points=[narrow], calibration=None)
        page = self._page(monkeypatch, tmp_path, sweep, _FLT_SERIES_TIERS)
        data = TestFilterPage._data(page)
        p = tuple(data["primary"])
        assert data["tier_binds"][p[0]] is False
        snap = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-tier", "off"], ["fire", "flt-tier"], ["settle"],
            ["snap", "off"]])["off"]
        n = len(narrow.trades)
        assert snap["text"]["flt-summary"] == dashboard._filter_summary_text(
            data["text"], _phrase_off(data, *p), True, None, n, n)
        assert "own simulation" not in snap["text"]["flt-summary"]
        assert "they never bind at this band" in snap["text"]["flt-summary"]

    def test_a_binding_band_off_at_another_cap_was_never_simulated(
            self, monkeypatch, tmp_path):
        # The family is simulated at the run's own cap only: a binding band's
        # tier-off view at any other cap is a scenario the run never
        # simulated, and the line says so; a band the tiers never bind at
        # shows its tier-on run there
        page = _kc_page(monkeypatch, tmp_path, _kc_sweep_tiers())
        data = TestFilterPage._data(page)
        other = next(ci for ci in range(len(data["caps"])) if ci != data["primary"][2])
        assert data["grid_off"][0][data["primary"][1]][other] is None
        snap = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-cap", str(other)], ["fire", "flt-cap"], ["settle"],
            ["set", "flt-tier", "off"], ["fire", "flt-tier"], ["settle"],
            ["snap", "off"]])["off"]
        pb, pk, _ = data["primary"]
        assert snap["text"]["flt-summary"] == (
            data["text"]["missing"].format(scenario=_phrase_off(data, pb, pk, other))
            + data["text"]["unfiltered"])
        # ... while the band the tiers never bind at shows its tier-on run there
        assert data["grid_off"][1] == data["grid"][1]

    def test_the_khat_notice_follows_the_tier_floors_choice(self, monkeypatch, tmp_path):
        # The primary band has a tier-off calibration but no tier-on one: an
        # empty selection with the tiers off was measured and found nothing
        # ("none"); with them on the band has no calibration ("not recorded")
        sweep = _flt_sweep_tiers()
        sweep = dataclasses.replace(sweep, calibration=None, calibrations_by_band={
            **sweep.calibrations_by_band, (0.0, 1.0): None})
        page = self._page(monkeypatch, tmp_path, sweep, _FLT_SERIES_TIERS)
        data = TestFilterPage._data(page)
        assert data["khat"][0] is None and data["khat_off"][0] is not None
        commodities = data["categories"].index("Commodities")
        assert f"c{commodities}" not in data["khat_off"][0]["groups"]
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["set", "khat-group", "tag"], ["fire", "khat-group"],
            ["set", "flt-tier", "off"], ["fire", "flt-tier"], ["settle"],
            _tick_category(str(commodities)), ["settle"], ["snap", "off"],
            ["set", "flt-tier", "on"], ["fire", "flt-tier"], ["settle"], ["snap", "on"]])
        for name, notice in (("off", "khat_none"), ("on", "khat_not_recorded")):
            snap = snaps[name]
            assert (snap["display"]["khat-body"], snap["display"]["khat-empty"]) \
                == ("none", ""), name
            assert snap["text"]["khat-empty"] == data["text"][notice], name

    def test_the_off_view_is_the_page_python_draws_for_the_tier_off_run(
            self, monkeypatch, tmp_path):
        # The primary band's tier-off view, drawn by the script, against the
        # page Python renders when that tier-off run IS the run it is given:
        # every redrawn chart but the k-hat one (which regroups a calibration,
        # not trades) and the interval-discount one (tier-on whatever the
        # choice), every card but the two k-hat ones (the tier-off group's,
        # below) and every table Python writes
        sweep = _flt_sweep_tiers()
        page = self._page(monkeypatch, tmp_path, sweep, _FLT_SERIES_TIERS)
        data = TestFilterPage._data(page)
        off = sweep.tier_off_scenarios[0]
        python_page = dashboard.generate_dashboard(
            off.trades, off.equity_df, _FLT_START, 1000.0, sweep=sweep,
            interval_discount=0.75, series_categories=_FLT_SERIES_TIERS,
        ).read_text(encoding="utf-8")
        snap = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-tier", "off"], ["fire", "flt-tier"], ["settle"],
            ["snap", "off"]])["off"]
        charts = _page_elements(python_page)["charts"]
        drawn = {r["id"]: r for r in snap["reacts"]}
        assert set(drawn) == _charts_redrawn()
        for cid in sorted(_charts_redrawn() - {"khat-fig", "kd-equity"}):
            react = drawn[cid]
            assert react["layout"] == charts[cid]["layout"], cid
            for i, trace in enumerate(charts[cid]["data"]):
                got = react["data"][i]
                for key in ("x", "y", "text", "name"):
                    if isinstance(trace.get(key), list) and trace[key] \
                            and isinstance(trace[key][0], (int, float)):
                        assert got[key] == pytest.approx(trace[key], abs=0.006), (cid, i, key)
                    elif key in trace:
                        assert got[key] == trace[key], (cid, i, key)
        kpis = dict(re.findall(r'<div id="kpi-([a-z_]+)" style="[^"]*">(.*?)</div>',
                               python_page))
        khat = {"khat", "khat_delta"}
        shown = {k[4:]: v for k, v in snap["text"].items() if k.startswith("kpi-")}
        assert {k: v for k, v in shown.items() if k not in khat and not k.startswith("kd_")} \
            == {k: v for k, v in kpis.items() if k not in khat and not k.startswith("kd_")}
        off_all = data["khat_off"][0]["groups"]["all"]
        assert (shown["khat"], shown["khat_delta"]) == (off_all["cells"][4],
                                                        off_all["delta"][1][0])
        assert snap["text"]["hdr-trades"] == str(len(off.trades)) == "6"
        assert f'<div id="dec-table">{snap["html"]["dec-table"]}</div>' in python_page
        for part in ("diag-best", "diag-worst"):
            assert f'<tbody id="{part}">{snap["html"][part]}</tbody>' in python_page, part

    def test_a_real_tier_off_sweep_offers_a_different_off_view(self, monkeypatch, tmp_path):
        # run_backtest_sweep itself, over TestPrepareEntriesGolden's fixture,
        # each band at the primary k only (sweep=False): with the tiers off
        # its ladder enters on 2026-01-05 instead of 2026-01-12, so the
        # primary band's off run trades differently from its on run
        from . import test_backtester
        golden = test_backtester.TestPrepareEntriesGolden()
        golden._patch(monkeypatch)
        sweep = backtester.run_backtest_sweep(
            hist_client=MagicMock(), live_client=MagicMock(), start_date=golden._START,
            initial_balance=10_000.0, sweep=False, same_event_ladders=True,
            band_sweep=True, tier_off_sweep=True)
        monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(dashboard.yf, "download",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
        page = dashboard.generate_dashboard(
            sweep.primary.trades, sweep.primary.equity_df, golden._START, 10_000.0,
            sweep=sweep).read_text(encoding="utf-8")
        data = TestFilterPage._data(page)
        bands = len(config.SPREAD_BAND_SWEEP_FLOORS) * len(config.SPREAD_BAND_SWEEP_CEILINGS)
        assert len(data["bands"]) == len(data["bands_off"]) == bands
        grid_bands = sorted({p.spread_band for p in sweep.scenarios if p.population == "all"})
        assert data["tier_binds"] == [backtester._tier_floors_bind(b) for b in grid_bands]
        assert 'id="flt-tier-note"' not in page
        pb, pk, pc = data["primary"]
        assert data["grid_off"][pb][pk][pc] != data["grid"][pb][pk][pc]
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["snap", "ready"],
            ["set", "flt-tier", "off"], ["fire", "flt-tier"], ["settle"], ["snap", "off"]])
        assert not snaps["ready"]["selects"]["flt-tier"]["disabled"]
        on_y = _page_elements(page)["charts"]["perf-cum"]["data"][0]["y"]
        off_y = _last_react(snaps["off"], "perf-cum")["data"][0]["y"]
        assert off_y != pytest.approx(on_y, abs=0.006)
        assert snaps["off"]["text"]["flt-summary"].startswith(
            "Showing every trade of the run at the primary spread band 0-1 with the "
            f"{dashboard._TIER_FLOORS} tier floors off, ")

    def test_a_failed_off_chunk_puts_the_tier_select_back(self, monkeypatch, tmp_path):
        # The tier-off chunk cannot be loaded: every select goes back to the
        # scenario still shown — the Tier floors choice too, with the band
        # options named for it — and the line names the scenario that failed
        page = self._page(monkeypatch, tmp_path, _flt_sweep_tiers(), _FLT_SERIES_TIERS)
        data = TestFilterPage._data(page)
        off_id = data["grid_off"][0][1][0]
        snap = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-tier", "off"], ["fire", "flt-tier"], ["settle"],
            ["snap", "failed"]], damaged=(f"dash-chunk-{off_id}",))["failed"]
        assert snap["selects"]["flt-tier"]["value"] == "on"
        assert [text for _, text in snap["selects"]["flt-band"]["options"]] == [
            "max(tier,0)-1 (primary)", "max(tier,0.3)-0.6", "max(tier,0.3)-1"]
        assert snap["text"]["flt-summary"] == data["text"]["unavailable"].format(
            failed=_phrase_off(data, *_FLT_PRIMARY),
            reason=f"Error: damaged block dash-chunk-{off_id}",
            scenario=_phrase(data, *_FLT_PRIMARY))
        # Nothing was drawn: the sections still show the tier-on primary
        assert snap["reacts"] == []

    @pytest.mark.parametrize("family", [False, True])
    def test_a_browser_that_cannot_inflate_keeps_the_bar_disabled(
            self, monkeypatch, tmp_path, family):
        # With a tier-off view in the data or without, nothing is enabled
        page = (self._page(monkeypatch, tmp_path, _flt_sweep_tiers(), _FLT_SERIES_TIERS)
                if family else self._page(monkeypatch, tmp_path))
        snap = _run_script(tmp_path, page, [["wait"], ["snap", "s"]],
                           no_decompression=True)["s"]
        assert all(snap["selects"][i]["disabled"]
                   for i in (*_FLT_SELECTS, "flt-tier", "flt-add", "flt-trim", "khat-group"))
        assert snap["text"]["flt-summary"].startswith(
            "The filter could not load its data (this browser cannot decompress it)")
        assert snap["reacts"] == [] and snap["inflated"] == []


# ─── The k and size-cap axes ──────────────────────────────────────────────────

_KC_B0, _KC_B1 = (0.0, 1.0), (0.3, 0.6)


def _kc_resized(t: BacktestTrade, n: int) -> BacktestTrade:
    """A trade of n contracts where it had t.n: cost, fees and payoffs scaled
    with it, so it still books profit = payoff - cost - fees."""
    s = n / t.n
    return dataclasses.replace(t, n=n, total_cost=t.total_cost * s, fees=t.fees * s,
                               profit=t.profit * s, actual_payoff=t.actual_payoff * s,
                               expected_payoff=t.expected_payoff * s, slippage=t.slippage * s)


def _kc_trades(n: int, count: int = 3) -> list[BacktestTrade]:
    """`count` time-series trades of n contracts each — a time-series trade's
    Kelly scatter depends on k, so the same list reads differently at two ks."""
    trades = [
        _ftrade("KXNHLHART-27", "time_series", date(2026, 1, 12), date(2026, 1, 20), 8.0,
                ladder=True),
        _ftrade("KXOTHER-1", "time_series", date(2026, 1, 13), date(2026, 1, 16), 4.0),
        _ftrade("KXNCAAMBGAME-1", "time_series", date(2026, 1, 6), date(2026, 1, 8), -3.0),
    ][:count]
    return [_kc_resized(t, n) for t in trades]


class _FakeCapSweep:
    """
    A backtester.CapSweep stand-in carrying exactly what the dashboard reads
    of one: bands, ks, caps, primary_cap, checks, simulated / reused, cell(),
    same_title() and entry_events().

    By default 2 bands x 2 ks x 3 caps — 5%, 20% (the run's own) and 1.0 (no
    cap). As CapSweep shares one simulation between every cap at or above a
    cell's peak Kelly fraction, caps above the peak here share ONE trade list
    object: at the primary band, 20% and no cap share a list and 5% trades a
    smaller one; at 0.3-0.6 every cap shares one list. (0.3-0.6, k 0.60) was
    never simulated. `raise_on` makes that cell's read raise, as a failed
    simulation would; `reads` records every cell read.
    """

    checks = False

    def __init__(self, points: dict, *, bands=(_KC_B0, _KC_B1), ks=(0.6, 0.75),
                 caps=(0.05, 0.2, 1.0), raise_on=None):
        self.points = points                    # (band, k, cap) -> SweepPoint
        self.bands, self.ks, self.caps, self.primary_cap = bands, ks, caps, 0.2
        self.raise_on = raise_on
        self.reads: list = []
        self.simulated = self.reused = 0

    def cell(self, band, k):
        self.reads.append((band, k))
        if (band, k) == self.raise_on:
            raise RuntimeError("the cap simulation failed")
        return {cap: {"all": self.points[(band, k, cap)]}
                for cap in self.caps if (band, k, cap) in self.points}

    def same_title(self):
        return {}

    def entry_events(self):
        return {(t.event_ticker, t.category) for p in self.points.values() for t in p.trades}


def _kc_sweep(*, raise_on=None) -> BacktestSweep:
    """_FakeCapSweep's default grid, with the eager points a run carries (every
    cell at the run's own 20% cap) and a k-hat population at the primary band."""
    full, small, other = _kc_trades(5), _kc_trades(2), _kc_trades(5, count=2)
    curves: dict = {}

    def point(band, k, cap, trades):
        if id(trades) not in curves:
            curves[id(trades)] = backtester._build_equity_curve(trades, _FLT_START, 1000.0)
        return SweepPoint(k=k, trades=trades, equity_df=curves[id(trades)],
                          spread_band=band, size_cap=cap)

    points = {}
    for k in (0.6, 0.75):
        for cap, trades in ((0.05, small), (0.2, full), (1.0, full)):
            points[(_KC_B0, k, cap)] = point(_KC_B0, k, cap, trades)
    for cap in (0.05, 0.2, 1.0):
        points[(_KC_B1, 0.75, cap)] = point(_KC_B1, 0.75, cap, other)
    primary = points[(_KC_B0, 0.75, 0.2)]
    eager = [primary, points[(_KC_B0, 0.6, 0.2)], points[(_KC_B1, 0.75, 0.2)]]
    cal = _flt_calibration([("KXNHLHART-27", "Other"), ("KXOTHER-1", "Other")])
    return BacktestSweep(primary=primary, points=eager[:2], calibration=cal,
                         label_coverage=_scn_coverage(), scenarios=eager,
                         calibrations_by_band={_KC_B0: cal, _KC_B1: None},
                         same_event_ladders=True,
                         cap_sweep=_FakeCapSweep(points, raise_on=raise_on))


def _kc_page(monkeypatch, tmp_path, sweep=None) -> str:
    sweep = sweep or _kc_sweep()
    return TestFilterPage()._page(monkeypatch, tmp_path, sweep)


def _kc_sweep_tiers() -> BacktestSweep:
    """_kc_sweep() plus a tier-floors-off family for its one binding band
    (_KC_B0, floor 0), simulated at the run's own 20% cap only (as the
    backtester's family is): at each k a two-trade list no tier-on cell
    trades, with its tier-off calibration. _KC_B1 (floor 0.3) sits at or
    above both tiers and was never simulated again."""
    sweep = _kc_sweep()
    off_trades = _kc_trades(4, count=2)
    curve = backtester._build_equity_curve(off_trades, _FLT_START, 1000.0)
    off = [SweepPoint(k=k, trades=off_trades, equity_df=curve, spread_band=_KC_B0,
                      size_cap=0.2, tier_floors=False) for k in (0.6, 0.75)]
    cal = _flt_calibration([("KXOTHER-1", "Other")])
    return dataclasses.replace(sweep, tier_off_scenarios=off,
                               tier_off_calibrations_by_band={_KC_B0: cal})


# The _kc grid's chunks, in walk order (primary cell first, primary cap first)
_KC_GRID = [[[3, 2, 2], [1, 0, 0]], [[None, None, None], [4, 4, 4]]]

# The header's cap clause when the run carried a size-cap sweep but the bar
# fell back to the run's own cap
_KC_UNUSED_CLAUSE = ("per-trade cap: 20% (size-cap sweep on, but it could not be used — "
                     "the filter bar offers the run's own cap only; the log names why)</p>")


class TestFilterKAndCap:
    """The bar's k and Size cap axes over a size-cap sweep: every scenario's
    chunk, the options, the summary, and what a failure costs."""

    def test_the_grid_maps_every_scenario_to_its_chunk(self):
        sweep = _kc_sweep()
        _, source, base, chunks = _flt_payload(sweep)
        assert (source.bands, source.ks, source.caps, source.primary) == (
            (_KC_B0, _KC_B1), (0.6, 0.75), (0.05, 0.2, 1.0), (0, 1, 1))
        assert source.cap_sweep is sweep.cap_sweep and source.fallback is not None
        # Caps above the peak share a list, and so a chunk; equal lists at
        # another k do not; (0.3-0.6, 0.60) was never simulated
        assert base["grid"] == _KC_GRID and len(chunks) == 5
        assert [_view(base, chunks, 0, 1, ci, "all")["n"] for ci in range(3)] == [3, 3, 3]
        tail = _chunk_at(base, chunks, 0, 1, 0)["strings"]
        assert any(">2</td>" in s for s in tail)          # 5%: two contracts a trade
        assert _chunk_at(base, chunks, 0, 0, 1)["list"]["kx"] != pytest.approx(
            _chunk_at(base, chunks, 0, 1, 1)["list"]["kx"])
        # The primary cell was read first, and every cell exactly once
        assert sweep.cap_sweep.reads == [(_KC_B0, 0.75), (_KC_B0, 0.6), (_KC_B1, 0.6),
                                         (_KC_B1, 0.75)]

    def test_the_options_mark_the_run_s_own(self, monkeypatch, tmp_path):
        page = _kc_page(monkeypatch, tmp_path)
        options = r'<option value="(\d+)"( selected)?>(.*?)</option>'
        ks = re.search(r'<select id="flt-k"[^>]*>(.*?)</select>', page).group(1)
        assert re.findall(options, ks) == [("0", "", "k = 0.60"),
                                           ("1", " selected", "k = 0.75 (primary)")]
        caps = re.search(r'<select id="flt-cap"[^>]*>(.*?)</select>', page).group(1)
        assert re.findall(options, caps) == [
            ("0", "", "5%"), ("1", " selected", "20% (primary)"), ("2", "", "off (full Kelly)")]
        assert ("Showing every trade of the run at the primary spread band max(tier,0)-1, "
                "k = 0.75, 20% cap per trade: 3 trades.") in page
        assert "| per-trade cap: 20% (size-cap sweep on)</p>" in page

    def test_the_summary_names_every_scenario(self):
        _, _, base, _ = _flt_payload(_kc_sweep())
        assert _phrase(base, 1, 1, 2) == (
            "spread band max(tier,0.3)-0.6, k = 0.75, no per-trade cap")
        assert _phrase(base, 0, 0, 0) == (
            "the primary spread band max(tier,0)-1, k = 0.60, 5% cap per trade")
        # A scenario the run never simulated says so, in Python's words
        assert base["text"]["missing"].format(scenario=_phrase(base, 1, 0, 1)) == (
            "Showing nothing: spread band max(tier,0.3)-0.6, k = 0.60, 20% cap per trade "
            "was not simulated by this run.")

    def test_a_chunk_that_cannot_be_built_costs_only_the_bar(
            self, monkeypatch, tmp_path, caplog):
        real, calls = dashboard._list_payload, []

        def second_fails(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise ValueError("boom")
            return real(*args, **kwargs)
        monkeypatch.setattr(dashboard, "_list_payload", second_fails)
        with caplog.at_level(logging.WARNING):
            page = _kc_page(monkeypatch, tmp_path)
        assert 'id="flt-unavailable"' in page and 'id="flt-bar"' not in page
        assert 'id="dash-chunk-' not in page and 'id="dash-data"' not in page
        assert "function inflate(el)" not in page
        for heading in ("Portfolio Performance", "Scenario Explorer", "Benchmark Comparison"):
            assert heading in page
        assert "could not be built for this run" in page          # the k-hat notice
        warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert sum("chunk could not be built" in m for m in warned) == 1

    def test_a_cap_cell_that_cannot_be_simulated_falls_back_to_the_run_s_own_cap(
            self, monkeypatch, tmp_path, caplog):
        # The LAST cell walked raises: every chunk built so far is dropped and
        # the eager points are walked instead — the page is complete, with the
        # run's own cap as the only option
        sweep = _kc_sweep(raise_on=(_KC_B1, 0.75))
        with caplog.at_level(logging.WARNING):
            page = _kc_page(monkeypatch, tmp_path, sweep)
        warned = [r for r in caplog.records if r.levelno == logging.WARNING
                  and "size-cap sweep could not be simulated" in r.getMessage()]
        assert len(warned) == 1 and warned[0].exc_info is not None
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        assert data["caps"] == [{"label": "20%", "text": "20% cap per trade", "value": 0.2}]
        assert data["grid"] == [[[1], [0]], [[None], [2]]] and len(chunks) == 3
        caps = re.search(r'<select id="flt-cap"[^>]*>(.*?)</select>', page).group(1)
        assert re.findall(r"<option[^>]*>(.*?)</option>", caps) == ["20% (primary)"]
        for piece in ('id="flt-bar"', "function inflate(el)", "Portfolio Performance",
                      "Benchmark Comparison"):
            assert piece in page
        # The one-option select's reason is on the page, not only in the log
        assert _KC_UNUSED_CLAUSE in page
        assert "(size-cap sweep on)" not in page

    def test_a_cap_sweep_whose_events_cannot_be_read_costs_only_the_cap_axis(
            self, monkeypatch, tmp_path, caplog):
        # A failure outside cell() — reading the entries' events up front —
        # costs what a cell failure costs: the cap axis, not the bar
        sweep = _kc_sweep()

        def broken():
            raise ZeroDivisionError("entry events")
        sweep.cap_sweep.entry_events = broken
        with caplog.at_level(logging.WARNING):
            page = _kc_page(monkeypatch, tmp_path, sweep)
        warned = [r for r in caplog.records if r.levelno == logging.WARNING
                  and "size-cap sweep could not be simulated" in r.getMessage()]
        assert len(warned) == 1 and warned[0].exc_info is not None
        assert not any("filter could not be built" in r.getMessage() for r in caplog.records)
        assert 'id="flt-bar"' in page and 'id="flt-unavailable"' not in page
        caps = re.search(r'<select id="flt-cap"[^>]*>(.*?)</select>', page).group(1)
        assert re.findall(r"<option[^>]*>(.*?)</option>", caps) == ["20% (primary)"]
        assert TestFilterPage._data(page)["grid"] == [[[1], [0]], [[None], [2]]]
        assert _KC_UNUSED_CLAUSE in page
        # Nothing was simulated from the sweep that could not be read
        assert sweep.cap_sweep.reads == []

    def test_a_cap_sweep_on_a_run_without_a_band_is_set_aside_and_said(
            self, monkeypatch, tmp_path, caplog):
        # A hand-built sweep whose primary records no band cannot place the
        # cap sweep's cells on a band axis: one "not recorded" band at the
        # run's own cap, a WARNING, and the header saying the sweep went unused
        sweep = _kc_sweep()
        primary = dataclasses.replace(sweep.primary, spread_band=None)
        sweep = dataclasses.replace(sweep, primary=primary, points=[primary], scenarios=[])
        with caplog.at_level(logging.WARNING):
            page = _kc_page(monkeypatch, tmp_path, sweep)
        assert any("cannot be placed on a run whose primary records no spread band"
                   in r.getMessage() for r in caplog.records)
        data = TestFilterPage._data(page)
        assert [c["label"] for c in data["caps"]] == ["20%"]
        assert [b["label"] for b in data["bands"]] == ["not recorded"]
        assert _KC_UNUSED_CLAUSE in page
        assert sweep.cap_sweep.reads == []

    def test_a_curve_built_after_midnight_is_cut_to_the_page_s_last_date(self, monkeypatch):
        # A cell simulated after UTC midnight ends its curve a day past the
        # page's; the walk hands every visitor a copy ending on the page's
        # last date — and never modifies the point itself
        class Clock(datetime):
            moment = datetime(2026, 3, 1, 23, 58, tzinfo=UTC)

            @classmethod
            def now(cls, tz=None):
                return cls.moment if tz is None else cls.moment.astimezone(tz)
        monkeypatch.setattr(backtester, "datetime", Clock)
        trades = _kc_trades(5)
        page_curve = backtester._build_equity_curve(trades, _FLT_START, 1000.0)
        Clock.moment = datetime(2026, 3, 2, 0, 2, tzinfo=UTC)
        late_curve = backtester._build_equity_curve(trades, _FLT_START, 1000.0)
        assert late_curve["date"].iloc[-1] == page_curve["date"].iloc[-1] + timedelta(days=1)
        late = SweepPoint(k=0.6, trades=trades, equity_df=late_curve, spread_band=_KC_B0,
                          size_cap=0.2)
        on_time = SweepPoint(k=0.75, trades=trades, equity_df=page_curve,
                             spread_band=_KC_B0, size_cap=0.2)
        cells = {0.6: {0.2: {"all": late, "time_series": late}},
                 0.75: {0.2: {"all": on_time}}}
        source = dashboard._GridSource(bands=(_KC_B0,), ks=(0.6, 0.75), caps=(0.2,),
                                       primary=(0, 1, 0), cell=lambda b, k: cells[k],
                                       calibrations={_KC_B0: None})
        seen = []

        class Spy:
            def reset(self, _source):
                pass

            def __call__(self, bi, ki, ci, pops):
                seen.append((ki, dict(pops)))
        axis_end = pd.Timestamp(page_curve["date"].iloc[-1])
        dashboard._walk_grid(source, [Spy()], axis_end)
        assert [ki for ki, _ in seen] == [1, 0]
        cut = seen[1][1]
        assert cut["all"] is not late and cut["all"].trades is late.trades
        # The two populations sharing one curve share its cut
        assert cut["all"].equity_df is cut["time_series"].equity_df
        assert list(cut["all"].equity_df["date"]) == list(page_curve["date"])
        assert len(late.equity_df) == len(page_curve) + 1          # untouched
        assert seen[0][1]["all"] is on_time                        # nothing to cut

    @pytest.mark.usefixtures("pre_toggle_defaults")
    def test_a_real_cap_sweep_pages_every_cap_at_its_own_simulation(
            self, monkeypatch, tmp_path):
        # A real run_backtest_sweep over the backtester's golden fixture,
        # narrowed to one band and one k: every cap the page offers is the
        # scenario a fresh simulation at that cap produces, figure for figure.
        # The run's own cap is pre_toggle_defaults' 20%, which the primary index uses
        golden = _tb.TestPrepareEntriesGolden()
        golden._patch(monkeypatch)
        monkeypatch.setattr(backtester, "SPREAD_BAND_SWEEP_FLOORS", (0.0,))
        monkeypatch.setattr(backtester, "SPREAD_BAND_SWEEP_CEILINGS", (1.0,))
        monkeypatch.setattr(backtester, "INTERVAL_DISCOUNT_SWEEP", (0.75,))
        run = backtester.run_backtest_sweep(MagicMock(), MagicMock(), golden._START, 10_000.0,
                                            same_event_ladders=True, band_sweep=True,
                                            cap_sweep=True)
        monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(dashboard.yf, "download",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
        page = dashboard.generate_dashboard(
            run.primary.trades, run.primary.equity_df, golden._START, 10_000.0,
            sweep=run).read_text(encoding="utf-8")
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        caps = run.cap_sweep.caps
        assert [c["value"] for c in data["caps"]] == list(caps) and len(caps) == 20
        assert data["primary"] == [0, 0, caps.index(0.2)]
        band = (0.0, 1.0)
        end = run.primary.equity_df["date"].iloc[-1]
        # The scenario explorer rides the same walk: a block per cap
        explorer = _scn_blocks(page)
        assert len(explorer) == 1 + len(caps)
        for ci, cap in enumerate(caps):
            fresh = backtester._simulate_at_discount(
                run.cap_sweep.entries_by_band[band], golden._START, 10_000.0, k=0.75,
                spread_band=band, size_cap=cap, quiet=True, end_date=end)
            view = _view(data, chunks, 0, 0, ci, "all")
            kpis = dashboard._performance_kpis(fresh.equity_df, fresh.trades, 10_000.0)
            assert view["kpi"] == {key: value for key, _, value, _ in kpis}, cap
            # ... and its "all" row there is that cap's own simulation too
            row = explorer[f"scn-cap-{ci}"]["cells"][0][0][0]
            assert row["trades"] == len(fresh.trades), cap
            assert row["final_balance"] == pytest.approx(
                float(fresh.equity_df["portfolio_value"].iloc[-1])), cap
        # Not vacuous: the caps below the fixture's peak size differently
        assert len({c for c in data["grid"][0][0] if c is not None}) > 1

    # ─── The script over the k and cap axes ───────────────────────────────────

    def test_a_k_change_loads_its_chunk_and_redraws_every_chart(self, monkeypatch, tmp_path):
        page = _kc_page(monkeypatch, tmp_path)
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-k", "0"], ["fire", "flt-k"], ["snap", "loading"],
            ["settle"], ["snap", "k"]], strict=True)
        # While its chunk inflates, the line says so, in Python's words
        assert snaps["loading"]["text"]["flt-summary"] == data["text"]["loading"].format(
            scenario=_phrase(data, 0, 0, 1))
        snap = snaps["k"]
        assert {r["id"] for r in snap["reacts"]} == _charts_redrawn()
        # The k's own Kelly scatter — not the primary's
        kx = _chunk_at(data, chunks, 0, 0, 1)["list"]["kx"]
        assert _last_react(snap, "risk-kelly")["data"][0]["x"] == pytest.approx(kx)
        assert kx != pytest.approx(_chunk_at(data, chunks, 0, 1, 1)["list"]["kx"])
        # The k-hat chart's line moves to the chosen k
        layout = _last_react(snap, "khat-fig")["layout"]
        assert (layout["shapes"][0]["x0"], layout["shapes"][0]["x1"]) == (0.6, 0.6)
        assert layout["annotations"][0]["text"] == "sized at k = 0.60"
        assert snap["text"]["flt-summary"] == dashboard._filter_summary_text(
            data["text"], _phrase(data, 0, 0, 1), False, None, 3, 3)
        assert snap["inflated"] == ["dash-data", "dash-chunk-0", "dash-chunk-2"]

    def test_a_cap_change_shows_that_cap_s_sizes(self, monkeypatch, tmp_path):
        page = _kc_page(monkeypatch, tmp_path)
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        snap = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-cap", "0"], ["fire", "flt-cap"], ["settle"],
            ["snap", "cap"]], strict=True)["cap"]
        assert {r["id"] for r in snap["reacts"]} == _charts_redrawn()
        small = _kc_trades(2)
        best, _ = dashboard._best_and_worst(small)
        assert snap["html"]["diag-best"] == "".join(
            dashboard._trade_row(t, dashboard._BEST_ROW_COLOR) for t in best)
        view = _view(data, chunks, 0, 1, 0, "all")
        assert {key: snap["text"][f"kpi-{key}"] for key in view["kpi"]} == view["kpi"]
        assert snap["text"]["kpi-brier"] == view["cal"]["brier"]
        assert "k = 0.75, 5% cap per trade: 3 trades. This spread band" \
            in snap["text"]["flt-summary"]

    def test_a_later_choice_supersedes_a_chunk_still_loading(self, monkeypatch, tmp_path):
        # k, then the cap, before either chunk has loaded: only the LAST
        # choice is drawn — the k's chunk, arriving after the cap was chosen,
        # is never drawn over it
        page = _kc_page(monkeypatch, tmp_path)
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        snap = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-k", "0"], ["fire", "flt-k"],
            ["set", "flt-cap", "0"], ["fire", "flt-cap"], ["settle"], ["snap", "s"]])["s"]
        assert [r["id"] for r in snap["reacts"]].count("perf-cum") == 1
        view = _view(data, chunks, 0, 0, 0, "all")
        assert _last_react(snap, "perf-cum")["data"][0]["y"] == pytest.approx(
            _expand(view["total"], len(data["dates"])))
        assert snap["text"]["flt-summary"].startswith(
            "Showing every trade of the run at "
            "the primary spread band max(tier,0)-1, k = 0.60, 5% cap per trade: 3 trades.")
        assert snap["inflated"] == ["dash-data", "dash-chunk-0", "dash-chunk-2",
                                    "dash-chunk-3"]

    @pytest.mark.parametrize("keep, inflated", [
        (None, ["dash-data", "dash-chunk-0", "dash-chunk-2", "dash-chunk-3"]),
        (1, ["dash-data", "dash-chunk-0", "dash-chunk-2", "dash-chunk-3", "dash-chunk-2"]),
    ])
    def test_loaded_chunks_are_kept_least_recently_used_first(
            self, monkeypatch, tmp_path, keep, inflated):
        # k 0.60 (chunk 2), then 5% (chunk 3), then back to 20% (chunk 2), then
        # the primary: with room for one chunk besides the primary's, chunk 2
        # was dropped for chunk 3 and is inflated again; the primary's never is
        page = _kc_page(monkeypatch, tmp_path)
        snap = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-k", "0"], ["fire", "flt-k"], ["settle"],
            ["set", "flt-cap", "0"], ["fire", "flt-cap"], ["settle"],
            ["set", "flt-cap", "1"], ["fire", "flt-cap"], ["settle"],
            ["set", "flt-k", "1"], ["fire", "flt-k"], ["settle"], ["snap", "s"]],
            keep=keep)["s"]
        assert snap["inflated"] == inflated
        assert snap["text"]["hdr-trades"] == "3"

    def test_a_scenario_never_simulated_shows_nothing_and_says_so(self, monkeypatch, tmp_path):
        page = _kc_page(monkeypatch, tmp_path)
        data = TestFilterPage._data(page)
        snap = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-band", "1"], ["fire", "flt-band"], ["settle"],
            ["set", "flt-k", "0"], ["fire", "flt-k"], ["settle"], ["snap", "s"]],
            strict=True)["s"]
        assert snap["text"]["flt-summary"] == (
            data["text"]["missing"].format(scenario=_phrase(data, 1, 0, 1))
            + data["text"]["unfiltered"])
        assert snap["text"]["hdr-trades"] == "0"
        for prefix in ("dec", "cal", "diag", "risk"):
            assert snap["display"][f"{prefix}-body"] == "none"
        # Nothing to inflate for it
        assert snap["inflated"] == ["dash-data", "dash-chunk-0", "dash-chunk-4"]

    def test_a_chunk_that_cannot_be_loaded_is_named_and_can_be_retried(
            self, monkeypatch, tmp_path):
        page = _kc_page(monkeypatch, tmp_path)
        data = TestFilterPage._data(page)
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-k", "0"], ["fire", "flt-k"], ["settle"], ["snap", "failed"],
            ["repair", "dash-chunk-2"],
            ["set", "flt-k", "0"], ["fire", "flt-k"], ["settle"], ["snap", "retried"]],
            damaged=("dash-chunk-2",))
        failed, retried = snaps["failed"], snaps["retried"]
        # The line names the scenario that failed and the one still shown; the
        # selects go back to the one shown, and stay usable
        assert failed["text"]["flt-summary"] == data["text"]["unavailable"].format(
            failed=_phrase(data, 0, 0, 1), reason="Error: damaged block dash-chunk-2",
            scenario=_phrase(data, 0, 1, 1))
        assert failed["selects"]["flt-k"]["value"] == "1"
        assert not any(failed["selects"][i]["disabled"] for i in _FLT_SELECTS)
        assert failed["reacts"] == []
        # Chosen again, it is loaded again
        assert retried["inflated"].count("dash-chunk-2") == 2
        assert retried["selects"]["flt-k"]["value"] == "0"
        assert "k = 0.60, 20% cap per trade: 3 trades." in retried["text"]["flt-summary"]

    def test_a_failed_load_puts_the_category_and_tag_back_too(self, monkeypatch, tmp_path):
        # k 0.60's chunk is still inflating when a category is chosen; then
        # the chunk fails. Nothing was drawn for either choice, so EVERY
        # select goes back to what the sections show — the category (and the
        # tag list it rebuilds) as well as the scenario
        page = _kc_page(monkeypatch, tmp_path)
        data = TestFilterPage._data(page)
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["snap", "loaded"],
            ["set", "flt-k", "0"], ["fire", "flt-k"],
            _tick_category("0"), ["snap", "pending"],
            ["reject", "dash-chunk-2"], ["snap", "failed"]],
            strict=True, deferred=("dash-chunk-2",))
        loaded, pending, failed = snaps["loaded"], snaps["pending"], snaps["failed"]
        assert pending["selects"]["flt-cat"]["value"] == "0"
        for sid in _FLT_SELECTS:
            assert failed["selects"][sid]["value"] == loaded["selects"][sid]["value"], sid
            assert failed["selects"][sid]["options"] == loaded["selects"][sid]["options"], sid
        assert not any(failed["selects"][sid]["disabled"] for sid in _FLT_SELECTS)
        # Nothing is redrawn: the sections still show the unfiltered primary
        assert pending["reacts"] == failed["reacts"] == []
        assert "hdr-trades" not in failed["text"]
        # ... the k-hat cards and the interval-discount section included: both
        # still describe the scenario drawn, as Python rendered it
        for element in ("kpi-khat", "kpi-khat_delta", "kpi-kd_k", "kpi-kd_delta"):
            assert element not in failed["text"]
        assert "kd-rows" not in failed["rows"]
        assert failed["text"]["flt-summary"] == data["text"]["unavailable"].format(
            failed=_phrase(data, 0, 0, 1), reason="Error: damaged block dash-chunk-2",
            scenario=_phrase(data, 0, 1, 1))

    def test_the_khat_chart_describes_the_scenario_on_screen_while_one_loads(
            self, monkeypatch, tmp_path):
        # A Group-by change while another band's chunk is inflating draws the
        # band the sections still show (the primary, which has a k-hat
        # population) — not the band still loading (which has none) — so a
        # failure of that chunk leaves the chart describing the page
        page = _kc_page(monkeypatch, tmp_path)
        data = TestFilterPage._data(page)
        snaps = _run_script(tmp_path, page, [
            ["wait"],
            ["set", "flt-band", "1"], ["fire", "flt-band"],
            ["set", "khat-group", "tag"], ["fire", "khat-group"], ["snap", "pending"],
            ["reject", "dash-chunk-4"], ["snap", "failed"]],
            strict=True, deferred=("dash-chunk-4",))
        pending, failed = snaps["pending"], snaps["failed"]
        assert [r["id"] for r in pending["reacts"]] == ["khat-fig"]
        title = _last_react(pending, "khat-fig")["layout"]["title"]["text"]
        assert data["bands"][0]["where"] in title and data["bands"][1]["where"] not in title
        assert pending["display"]["khat-body"] == ""
        # The failure redraws nothing: the chart already shows the band shown
        assert failed["reacts"] == []
        assert failed["selects"]["flt-band"]["value"] == "0"
        assert failed["display"]["khat-body"] == ""

    def test_a_load_the_reader_moved_past_never_pushes_out_the_chunk_on_screen(
            self, monkeypatch, tmp_path):
        # KEEP = 1. The 5% chunk is still inflating when the reader goes back
        # to 20% at k 0.60 (drawn at once, from the chunk already loaded); the
        # 5% chunk then arrives, superseded. It was never drawn, so it is not
        # kept — and it does not push the chunk on screen out: a category
        # change draws at once, with no second inflate and no "Loading" line.
        # Chosen again, the 5% chunk is inflated again
        page = _kc_page(monkeypatch, tmp_path)
        snaps = _run_script(tmp_path, page, [
            ["wait"],
            ["set", "flt-k", "0"], ["fire", "flt-k"], ["resolve", "dash-chunk-2"],
            ["set", "flt-cap", "0"], ["fire", "flt-cap"],
            ["set", "flt-cap", "1"], ["fire", "flt-cap"], ["snap", "back"],
            ["resolve", "dash-chunk-3"],
            _tick_category("0"), ["snap", "cat"],
            ["set", "flt-cap", "0"], ["fire", "flt-cap"], ["resolve", "dash-chunk-3"],
            ["snap", "again"]],
            keep=1, deferred=("dash-chunk-2", "dash-chunk-3"))
        back, cat, again = snaps["back"], snaps["cat"], snaps["again"]
        assert back["text"]["flt-summary"].startswith("Showing every trade")
        assert cat["text"]["flt-summary"].startswith("Showing ")
        assert cat["inflated"].count("dash-chunk-2") == 1
        assert {r["id"] for r in cat["reacts"]} >= {"perf-cum", "khat-fig"}
        assert again["inflated"].count("dash-chunk-3") == 2
        assert "5% cap per trade" in again["text"]["flt-summary"]

    def test_at_the_shipped_keep_superseded_loads_never_evict_the_chunk_on_screen(
            self, monkeypatch, tmp_path):
        # KEEP = 16, 20 caps each trading its own list: step through 16 other
        # caps while their chunks inflate, come back to the cap on screen, and
        # let all 16 arrive. None was drawn, so none is kept or counted, and
        # the chunk on screen stays loaded: the next category change draws at
        # once, without inflating it again
        caps = tuple(round(0.05 * i, 2) for i in range(1, 20)) + (1.0,)
        points = {}
        for i, cap in enumerate(caps):
            listed = _kc_trades(i + 1)
            points[(_KC_B0, 0.75, cap)] = SweepPoint(
                k=0.75, trades=listed, spread_band=_KC_B0, size_cap=cap,
                equity_df=backtester._build_equity_curve(listed, _FLT_START, 1000.0))
        primary = points[(_KC_B0, 0.75, 0.2)]
        sweep = BacktestSweep(primary=primary, points=[primary], calibration=None,
                              label_coverage=_scn_coverage(),
                              cap_sweep=_FakeCapSweep(points, bands=(_KC_B0,), ks=(0.75,),
                                                      caps=caps))
        page = _kc_page(monkeypatch, tmp_path, sweep)
        grid = TestFilterPage._data(page)["grid"][0][0]
        assert len(set(grid)) == len(caps)
        shown = 0
        others = [ci for ci in range(len(caps)) if ci not in (shown, caps.index(0.2))][:16]
        steps = [["wait"], ["set", "flt-cap", str(shown)], ["fire", "flt-cap"],
                 ["resolve", f"dash-chunk-{grid[shown]}"]]
        for ci in others:
            steps += [["set", "flt-cap", str(ci)], ["fire", "flt-cap"]]
        steps += [["set", "flt-cap", str(shown)], ["fire", "flt-cap"], ["snap", "back"]]
        steps += [["resolve", f"dash-chunk-{grid[ci]}"] for ci in others]
        steps += [_tick_category("0"), ["snap", "cat"]]
        deferred = tuple(f"dash-chunk-{c}" for c in grid if c != grid[caps.index(0.2)])
        snaps = _run_script(tmp_path, page, steps, deferred=deferred)
        assert snaps["back"]["text"]["flt-summary"].startswith("Showing every trade")
        assert snaps["cat"]["text"]["flt-summary"].startswith("Showing ")
        assert snaps["cat"]["inflated"].count(f"dash-chunk-{grid[shown]}") == 1
        assert snaps["cat"]["pending"] == []

    def test_a_primary_chunk_that_cannot_be_loaded_disables_the_bar(
            self, monkeypatch, tmp_path):
        page = _kc_page(monkeypatch, tmp_path)
        data = TestFilterPage._data(page)
        snap = _run_script(tmp_path, page, [["wait"], ["snap", "s"]],
                           damaged=("dash-chunk-0",))["s"]
        assert all(snap["selects"][i]["disabled"] for i in (*_FLT_SELECTS, "khat-group"))
        primary = _phrase(data, 0, 1, 1)
        assert snap["text"]["flt-summary"] == data["text"]["unavailable"].format(
            failed=primary, reason="Error: damaged block dash-chunk-0", scenario=primary)

    def test_a_page_without_a_sweep_runs_strictly(self, monkeypatch, tmp_path):
        # Placeholders for the Interval Discount and Scenario Explorer
        # sections, one scenario: the script reaches no element the page lacks
        trades = _flt_trades()
        curve = backtester._build_equity_curve(trades, _FLT_START, 1000.0)
        monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(dashboard.yf, "download",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
        page = dashboard.generate_dashboard(
            trades, curve, _FLT_START, 1000.0, interval_discount=0.75,
            series_categories=_FLT_SERIES).read_text(encoding="utf-8")
        data = TestFilterPage._data(page)
        sports = data["categories"].index("Sports")
        hockey = str(data["subcats"].index([sports, "Hockey"]))
        snap = _run_script(tmp_path, page, [
            ["wait"], _tick_tag(hockey), ["settle"],
            ["set", "khat-group", "band"], ["fire", "khat-group"],
            ["set", "flt-k", "0"], ["fire", "flt-k"], ["settle"], ["snap", "s"]],
            strict=True)["s"]
        assert not any(snap["selects"][i]["disabled"] for i in _FLT_SELECTS)
        assert snap["text"]["hdr-trades"] == "1"
        assert snap["text"]["flt-summary"].startswith(
            "Showing Sports · Hockey within the run at the primary spread band (not "
            "recorded), k = 0.75, cap not recorded: 1 of its 5 trades.")


# ─── The filter bar's "Save as live defaults…" button ────────────────────────

# The defaults server's confirmation page, as the button's address names it
_SAVE_URL = f"http://{config.DEFAULTS_SERVER_HOST}:{config.DEFAULTS_SERVER_PORT}/confirm"
# The Host header a browser sends to that address
_SAVE_HOST = f"{config.DEFAULTS_SERVER_HOST}:{config.DEFAULTS_SERVER_PORT}"


class TestSaveLiveDefaultsButton:
    """The filter bar's "Save as live defaults…" button: rendered disabled,
    enabled by the script only while the bar's scenario on screen can become
    the live settings, and opening the defaults server's confirmation page for
    exactly that scenario — never the selects' choice while a chunk loads, and
    never the Scenario Explorer's own selects. The address it opens is checked
    against the server's own parser. Its script tests are skipped without a
    JavaScript runtime; its render and _save_target tests are not."""

    @staticmethod
    def _page(monkeypatch, tmp_path, sweep=None, series=_FLT_SERIES, *,
              same_title_size_cap: float | None = 0.5) -> str:
        """
        Render a whole page of the flt fixture, or of another sweep.

        Args:
            monkeypatch (pytest.MonkeyPatch): The test's monkeypatch, which
                points the page at tmp_path and keeps the benchmark offline.
            tmp_path (Path): Where the page is written.
            sweep (BacktestSweep | None): The sweep to render; None renders
                _flt_sweep().
            series (dict | None): The series listing the page files trades
                by; None files them by ticker prefix.
            same_title_size_cap (float | None): Keyword-only. The run's
                same-title cap set on the sweep; None means not recorded.

        Returns:
            str: The page's HTML.
        """
        sweep = dataclasses.replace(sweep or _flt_sweep(),
                                    same_title_size_cap=same_title_size_cap)
        return TestFilterPage()._page(monkeypatch, tmp_path, sweep, series)

    @staticmethod
    def _kc(monkeypatch, tmp_path, sweep=None) -> str:
        """
        Render a whole page of the k and size-cap fixture, same-title cap 0.5.

        Args:
            monkeypatch (pytest.MonkeyPatch): The test's monkeypatch.
            tmp_path (Path): Where the page is written.
            sweep (BacktestSweep | None): The sweep to render; None renders
                _kc_sweep().

        Returns:
            str: The page's HTML.
        """
        sweep = dataclasses.replace(sweep or _kc_sweep(), same_title_size_cap=0.5)
        return _kc_page(monkeypatch, tmp_path, sweep)

    @staticmethod
    def _query(data: dict, opened: dict) -> dict[str, str]:
        """
        Read one window.open call's address, checked against the page's save target.

        The call must open a new tab with "noopener", the address before its
        query must be the save target's (the defaults server's confirmation
        page), and every field of the query must appear exactly once.

        Args:
            data (dict): The page's base block.
            opened (dict): One recorded window.open call: url, target, features.

        Returns:
            dict[str, str]: The query's fields and their one value each.

        Raises:
            AssertionError: If the target, the features or the address differ
                from those above, the query cannot be parsed strictly, or a
                field repeats.
        """
        assert opened["target"] == "_blank" and opened["features"] == "noopener"
        url = opened["url"]
        base, _, query = url.partition("?")
        assert base == data["save"]["url"] == _SAVE_URL
        fields = parse_qs(query, keep_blank_values=True, strict_parsing=True)
        assert all(len(values) == 1 for values in fields.values()), fields
        return {name: values[0] for name, values in fields.items()}

    @staticmethod
    def _expected(data: dict, bi: int, ki: int, ci: int, tier: str = "on") -> dict:
        """
        Give the fields a click on scenario [bi, ki, ci] must send, from the base block.

        Args:
            data (dict): The page's base block.
            bi (int): The band's index.
            ki (int): The k's index.
            ci (int): The size cap's index.
            tier (str): The Tier floors choice on screen, "on" or "off".

        Returns:
            dict: tier_floors (the choice), spread_min and spread_max (the
                band's bounds), k and size_cap (the axes' values).
        """
        band = data["bands"][bi]["value"]
        return {"tier_floors": tier, "spread_min": band[0], "spread_max": band[1],
                "k": data["ks"][ki]["value"], "size_cap": data["caps"][ci]["value"]}

    def _assert_sends(self, data: dict, query: dict, bi: int, ki: int, ci: int,
                      tier: str = "on", *, category: str | None = None,
                      tag: str | None = None) -> None:
        """
        Check that a query names scenario [bi, ki, ci] exactly, and nothing else.

        Each number must read back as the base block's own value, and the
        query must carry the run's same-title cap, the source note and the
        category and tag given, and no other field. add_to_held_pairs,
        sell_at and sell_min_days may be present (a page with the Add to held
        pairs or Sell view sends them) and are not checked here: their values
        are the caller's to check.

        Args:
            data (dict): The page's base block.
            query (dict): The query, as _query read it.
            bi (int): The band's index.
            ki (int): The k's index.
            ci (int): The size cap's index.
            tier (str): The Tier floors choice on screen, "on" or "off".
            category (str | None): Keyword-only. The category sent, or None
                for none.
            tag (str | None): Keyword-only. The tag sent, or None for none.

        Raises:
            AssertionError: If any field differs from the expected value, or
                the query carries a field not listed above.
        """
        expected = self._expected(data, bi, ki, ci, tier)
        assert query["tier_floors"] == expected.pop("tier_floors")
        for name, value in expected.items():
            assert float(query[name]) == value, name
        assert float(query["same_title_size_cap"]) == data["save"]["same_title_size_cap"]
        assert query["source"] == data["save"]["source"]
        assert query.get("category") == category and query.get("tag") == tag
        known = {"tier_floors", "spread_min", "spread_max", "k", "size_cap",
                 "same_title_size_cap", "source", "category", "tag", "add_to_held_pairs",
                 "sell_at", "sell_min_days"}
        assert set(query) <= known

    def test_the_button_is_rendered_disabled_with_its_words(self, monkeypatch, tmp_path):
        page = self._page(monkeypatch, tmp_path)
        button = re.search(r'<button id="flt-save"([^>]*)>(.*?)</button>', page)
        assert button is not None
        assert button.group(2) == html.escape(dashboard._SAVE_LABEL)
        assert f'title="{html.escape(dashboard._SAVE_TITLE)}"' in button.group(1)
        assert _page_elements(page)["buttons"] == {"flt-save": {"disabled": True}}
        assert html.escape(dashboard._SAVE_NOTE) in page
        # After the Tag select, before the summary line
        assert page.index('id="flt-tag"') < page.index('id="flt-save"') \
            < page.index('id="flt-summary"')
        data = TestFilterPage._data(page)
        # Each band carries its bounds for the button's spread_min / spread_max
        assert [b["value"] for b in data["bands"]] == [[0.0, 1.0], [0.3, 0.6], [0.3, 1.0]]
        save = data["save"]
        assert save["url"] == _SAVE_URL
        assert save["same_title_size_cap"] == 0.5 and save["filed_by_listing"] is True
        assert re.fullmatch(config.LIVE_DEFAULTS_SOURCE_PATTERN, save["source"], re.ASCII)
        assert save["source"].startswith(f"backtest dashboard for {_FLT_START.isoformat()} to ")
        # Every word about the button is Python's: the script writes none
        for words in (dashboard._SAVE_LABEL, dashboard._SAVE_TITLE,
                      dashboard._SAVE_TITLE_UNFILED, dashboard._SAVE_NOTE):
            assert words not in dashboard._FILTER_JS
        # ... and the address comes from config, never spelled on the page's side
        assert config.DEFAULTS_SERVER_HOST not in dashboard._FILTER_JS
        assert str(config.DEFAULTS_SERVER_PORT) not in dashboard._FILTER_JS

    def test_once_loaded_a_click_opens_the_primary_scenario(self, monkeypatch, tmp_path):
        page = self._page(monkeypatch, tmp_path)
        data = TestFilterPage._data(page)
        snaps = _run_script(tmp_path, page, [
            ["snap", "loaded"], ["wait"], ["snap", "ready"],
            ["click", "flt-save"], ["snap", "clicked"]], strict=True)
        assert snaps["loaded"]["buttons"]["flt-save"] is True
        assert snaps["ready"]["buttons"]["flt-save"] is False
        assert snaps["loaded"]["opened"] == snaps["ready"]["opened"] == []
        [opened] = snaps["clicked"]["opened"]
        self._assert_sends(data, self._query(data, opened), *data["primary"])
        # A click draws nothing and changes no select
        assert snaps["clicked"]["reacts"] == [] and snaps["clicked"]["updates"] == []
        assert snaps["clicked"]["selects"] == snaps["ready"]["selects"]

    def test_it_follows_the_band_k_and_size_cap_on_screen(self, monkeypatch, tmp_path):
        page = self._kc(monkeypatch, tmp_path)
        data = TestFilterPage._data(page)
        snaps = _run_script(tmp_path, page, [
            ["wait"],
            ["set", "flt-cap", "2"], ["fire", "flt-cap"], ["settle"],
            ["click", "flt-save"], ["snap", "cap"],
            ["set", "flt-k", "0"], ["fire", "flt-k"], ["settle"],
            ["click", "flt-save"], ["snap", "k"],
            ["set", "flt-k", "1"], ["fire", "flt-k"], ["settle"],
            ["set", "flt-band", "1"], ["fire", "flt-band"], ["settle"],
            ["click", "flt-save"], ["snap", "band"]], strict=True)
        for name, (bi, ki, ci) in (("cap", (0, 1, 2)), ("k", (0, 0, 2)), ("band", (1, 1, 2))):
            [opened] = snaps[name]["opened"]
            self._assert_sends(data, self._query(data, opened), bi, ki, ci)
        # The size cap sent is the cap on screen: "off (full Kelly)" is 1.0
        [opened] = snaps["cap"]["opened"]
        assert float(self._query(data, opened)["size_cap"]) == 1.0

    def test_it_follows_a_category_and_a_tag_with_its_category(self, monkeypatch, tmp_path):
        page = self._page(monkeypatch, tmp_path)
        data = TestFilterPage._data(page)
        sports = data["categories"].index("Sports")
        basketball = str(data["subcats"].index([sports, "Basketball"]))
        snaps = _run_script(tmp_path, page, [
            ["wait"],
            _tick_category(str(sports)), ["settle"],
            ["click", "flt-save"], ["snap", "category"],
            _tick_category(""), ["settle"],
            # A "Category · Tag" chosen under "All" sets its category too
            _tick_tag(basketball), ["settle"],
            ["click", "flt-save"], ["snap", "tag"],
            _tick_category(""), ["settle"],
            ["click", "flt-save"], ["snap", "all"]], strict=True)
        primary = data["primary"]
        [opened] = snaps["category"]["opened"]
        self._assert_sends(data, self._query(data, opened), *primary, category="Sports")
        [opened] = snaps["tag"]["opened"]
        self._assert_sends(data, self._query(data, opened), *primary, category="Sports",
                           tag="Sports · Basketball")
        [opened] = snaps["all"]["opened"]
        self._assert_sends(data, self._query(data, opened), *primary)

    def test_it_follows_the_tier_floors_choice(self, monkeypatch, tmp_path):
        page = self._kc(monkeypatch, tmp_path, _kc_sweep_tiers())
        data = TestFilterPage._data(page)
        snaps = _run_script(tmp_path, page, [
            ["wait"],
            ["set", "flt-tier", "off"], ["fire", "flt-tier"], ["settle"],
            ["click", "flt-save"], ["snap", "off"],
            ["set", "flt-tier", "on"], ["fire", "flt-tier"], ["settle"],
            ["click", "flt-save"], ["snap", "on"]], strict=True)
        [opened] = snaps["off"]["opened"]
        self._assert_sends(data, self._query(data, opened), *data["primary"], tier="off")
        [opened] = snaps["on"]["opened"]
        self._assert_sends(data, self._query(data, opened), *data["primary"], tier="on")

    def test_a_scenario_the_run_never_simulated_cannot_be_saved(self, monkeypatch, tmp_path):
        # (0.3-0.6, k 0.60) was never simulated: the button is disabled and a
        # click opens nothing; back on a simulated scenario it is enabled again
        page = self._kc(monkeypatch, tmp_path)
        data = TestFilterPage._data(page)
        assert _KC_GRID[1][0] == [None, None, None]
        snaps = _run_script(tmp_path, page, [
            ["wait"],
            ["set", "flt-k", "0"], ["fire", "flt-k"], ["settle"],
            ["set", "flt-band", "1"], ["fire", "flt-band"], ["settle"],
            ["click", "flt-save"], ["snap", "missing"],
            ["set", "flt-k", "1"], ["fire", "flt-k"], ["settle"],
            ["click", "flt-save"], ["snap", "back"]], strict=True)
        missing, back = snaps["missing"], snaps["back"]
        assert missing["text"]["flt-summary"].startswith(
            data["text"]["missing"].format(scenario=_phrase(data, 1, 0, 1)))
        assert missing["buttons"]["flt-save"] is True and missing["opened"] == []
        assert back["buttons"]["flt-save"] is False
        [opened] = back["opened"]
        self._assert_sends(data, self._query(data, opened), 1, 1, 1)

    def test_while_a_chunk_loads_it_saves_nothing_then_the_scenario_on_screen(
            self, monkeypatch, tmp_path):
        page = self._kc(monkeypatch, tmp_path)
        data = TestFilterPage._data(page)
        assert _KC_GRID[1][1][1] == 4
        steps = [["wait"], ["set", "flt-band", "1"], ["fire", "flt-band"], ["settle"],
                 ["click", "flt-save"], ["snap", "loading"]]
        resolved = _run_script(tmp_path, page, steps + [
            ["resolve", "dash-chunk-4"], ["snap", "drawn"],
            ["click", "flt-save"], ["snap", "clicked"]],
            strict=True, deferred=("dash-chunk-4",))
        loading = resolved["loading"]
        assert loading["pending"] == ["dash-chunk-4"]
        assert loading["buttons"]["flt-save"] is True and loading["opened"] == []
        # Once drawn: the new scenario
        assert resolved["drawn"]["buttons"]["flt-save"] is False
        [opened] = resolved["clicked"]["opened"]
        self._assert_sends(data, self._query(data, opened), 1, 1, 1)
        # A chunk that fails: the scenario still shown, which the selects go back to
        failed = _run_script(tmp_path, page, steps + [
            ["reject", "dash-chunk-4"], ["snap", "failed"],
            ["click", "flt-save"], ["snap", "clicked"]],
            strict=True, deferred=("dash-chunk-4",))
        assert failed["loading"]["buttons"]["flt-save"] is True
        assert failed["failed"]["selects"]["flt-band"]["value"] == "0"
        assert failed["failed"]["buttons"]["flt-save"] is False
        [opened] = failed["clicked"]["opened"]
        self._assert_sends(data, self._query(data, opened), *data["primary"])

    @pytest.mark.parametrize("damaged", ["dash-data", "dash-chunk-0"])
    def test_a_block_that_cannot_be_loaded_keeps_it_disabled(
            self, monkeypatch, tmp_path, damaged):
        # The base block, or the primary scenario's chunk: the bar cannot
        # work, and nothing can be saved from it
        page = self._kc(monkeypatch, tmp_path)
        snap = _run_script(tmp_path, page, [
            ["wait"], ["click", "flt-save"], ["snap", "s"]], damaged=(damaged,))["s"]
        assert snap["buttons"]["flt-save"] is True and snap["opened"] == []
        assert snap["selects"]["flt-band"]["disabled"]

    def test_a_browser_that_cannot_inflate_keeps_it_disabled(self, monkeypatch, tmp_path):
        page = self._page(monkeypatch, tmp_path)
        snap = _run_script(tmp_path, page, [
            ["wait"], ["click", "flt-save"], ["snap", "s"]], no_decompression=True)["s"]
        assert snap["buttons"]["flt-save"] is True and snap["opened"] == []

    def test_without_a_series_listing_no_category_or_tag_is_saved(self, monkeypatch, tmp_path):
        # Trades filed by ticker prefix: the live category filter files by
        # Kalshi's listing, so a slice of this page is not one it can trade
        page = self._page(monkeypatch, tmp_path, series=None)
        data = TestFilterPage._data(page)
        assert data["save"]["filed_by_listing"] is False and data["categories"]
        title = re.search(r'<button id="flt-save"[^>]*title="([^"]*)"', page).group(1)
        assert html.unescape(title) == dashboard._SAVE_TITLE + dashboard._SAVE_TITLE_UNFILED
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["click", "flt-save"], ["snap", "all"],
            _tick_category("0"), ["settle"],
            ["click", "flt-save"], ["snap", "category"],
            _tick_category(""), ["settle"], ["snap", "back"]],
            strict=True)
        [opened] = snaps["all"]["opened"]
        self._assert_sends(data, self._query(data, opened), *data["primary"])
        assert snaps["category"]["buttons"]["flt-save"] is True
        assert snaps["category"]["opened"] == []
        assert snaps["back"]["buttons"]["flt-save"] is False

    def test_a_listing_page_s_title_has_no_unfiled_sentence(self, monkeypatch, tmp_path):
        page = self._page(monkeypatch, tmp_path)
        title = re.search(r'<button id="flt-save"[^>]*title="([^"]*)"', page).group(1)
        assert html.unescape(title) == dashboard._SAVE_TITLE

    def test_a_k_that_is_not_positive_cannot_be_saved(self, monkeypatch, tmp_path):
        # The live settings need k in (0, 1]: a run's k of 0.0 is not one
        sweep = _flt_sweep()
        scenarios = [dataclasses.replace(p, k=0.0) if p.k == 0.60 else p
                     for p in sweep.scenarios]
        sweep = dataclasses.replace(sweep, scenarios=scenarios)
        page = self._page(monkeypatch, tmp_path, sweep)
        data = TestFilterPage._data(page)
        zero = next(i for i, k in enumerate(data["ks"]) if k["value"] == 0.0)
        assert data["grid"][0][zero][0] is not None
        snap = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-k", str(zero)], ["fire", "flt-k"], ["settle"],
            ["click", "flt-save"], ["snap", "s"]], strict=True)["s"]
        assert snap["buttons"]["flt-save"] is True and snap["opened"] == []

    def test_a_k_of_many_decimals_is_sent_exactly(self, monkeypatch, tmp_path):
        # A k such as backtest.py --interval-discount can name: the address
        # carries the number that reads back as that very k (a two-decimal
        # 0.33 would be another k), and the server reads it back as that k
        third = 1 / 3
        sweep = _flt_sweep()
        scenarios = [dataclasses.replace(p, k=third) if p.k == 0.60 else p
                     for p in sweep.scenarios]
        sweep = dataclasses.replace(sweep, scenarios=scenarios)
        page = self._page(monkeypatch, tmp_path, sweep)
        data = TestFilterPage._data(page)
        ki = next(i for i, k in enumerate(data["ks"]) if k["value"] == third)
        assert data["grid"][0][ki][0] is not None
        snap = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-k", str(ki)], ["fire", "flt-k"], ["settle"],
            ["click", "flt-save"], ["snap", "s"]], strict=True)["s"]
        [opened] = snap["opened"]
        query = self._query(data, opened)
        self._assert_sends(data, query, 0, ki, 0)
        assert float(query["k"]) == third
        settings, _ = defaults_server._proposal(
            defaults_server._params(opened["url"].partition("?")[2]), None)
        assert settings.interval_discount == third

    def test_a_page_without_a_sweep_cannot_be_saved(self, monkeypatch, tmp_path):
        # No sweep: one "not recorded" band and cap (k is the page's own
        # 0.75), so nothing to save
        trades = _flt_trades()
        curve = backtester._build_equity_curve(trades, _FLT_START, 1000.0)
        monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(dashboard.yf, "download",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
        page = dashboard.generate_dashboard(
            trades, curve, _FLT_START, 1000.0, interval_discount=0.75,
            series_categories=_FLT_SERIES).read_text(encoding="utf-8")
        data = TestFilterPage._data(page)
        pb, pk, pc = data["primary"]
        assert data["bands"][pb]["value"] is None and data["caps"][pc]["value"] is None
        assert data["ks"][pk]["value"] == 0.75
        assert data["save"]["same_title_size_cap"] is None
        snap = _run_script(tmp_path, page, [
            ["wait"], ["click", "flt-save"], ["snap", "s"]], strict=True)["s"]
        assert not snap["selects"]["flt-band"]["disabled"]
        assert snap["buttons"]["flt-save"] is True and snap["opened"] == []

    def test_a_band_the_run_did_not_record_cannot_be_saved(self, monkeypatch, tmp_path):
        # A sweep whose primary records no band but does record its k and
        # cap: the band alone is what keeps the button disabled, on load and
        # on every redraw (a category change redraws at once, its chunk
        # being the one on screen)
        base = _flt_sweep()
        primary = dataclasses.replace(base.primary, spread_band=None)
        sweep = dataclasses.replace(base, primary=primary, points=[primary], scenarios=[])
        page = self._page(monkeypatch, tmp_path, sweep)
        data = TestFilterPage._data(page)
        pb, pk, pc = data["primary"]
        assert data["bands"][pb]["value"] is None
        assert data["ks"][pk]["value"] == 0.75 and data["caps"][pc]["value"] == 0.2
        assert data["save"]["same_title_size_cap"] == 0.5
        sports = str(data["categories"].index("Sports"))
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["click", "flt-save"], ["snap", "ready"],
            _tick_category(sports), ["settle"],
            ["click", "flt-save"], ["snap", "category"]], strict=True)
        ready, category = snaps["ready"], snaps["category"]
        assert not ready["selects"]["flt-band"]["disabled"]
        assert ready["buttons"]["flt-save"] is True and ready["opened"] == []
        # The category was drawn, and the button stayed disabled through it
        assert category["selects"]["flt-cat"]["value"] == sports and category["reacts"]
        assert category["buttons"]["flt-save"] is True and category["opened"] == []

    def test_a_size_cap_the_run_did_not_record_cannot_be_saved(self, monkeypatch, tmp_path):
        # A sweep whose points record their band and k but no size cap: the
        # cap alone is what keeps the button disabled, on load and on every
        # redraw (the server would refuse a size_cap of null, so the page
        # must not offer it)
        page = self._page(monkeypatch, tmp_path, _flt_sweep(size_cap=None))
        data = TestFilterPage._data(page)
        pb, pk, pc = data["primary"]
        assert data["bands"][pb]["value"] == [0.0, 1.0]
        assert data["ks"][pk]["value"] == 0.75 and data["caps"][pc]["value"] is None
        assert data["save"]["same_title_size_cap"] == 0.5
        sports = str(data["categories"].index("Sports"))
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["click", "flt-save"], ["snap", "ready"],
            _tick_category(sports), ["settle"],
            ["click", "flt-save"], ["snap", "category"]], strict=True)
        ready, category = snaps["ready"], snaps["category"]
        assert not ready["selects"]["flt-band"]["disabled"]
        assert ready["buttons"]["flt-save"] is True and ready["opened"] == []
        # The category was drawn, and the button stayed disabled through it
        assert category["selects"]["flt-cat"]["value"] == sports and category["reacts"]
        assert category["buttons"]["flt-save"] is True and category["opened"] == []

    def test_a_ladder_setting_that_departs_is_named_in_the_source(self, monkeypatch, tmp_path):
        sweep = dataclasses.replace(_flt_sweep(), same_event_ladders=False,
                                    config_same_event_ladders=True)
        page = self._page(monkeypatch, tmp_path, sweep)
        source = TestFilterPage._data(page)["save"]["source"]
        assert source.endswith(" (same-event ladders off, config.py on: its pairs are not "
                               "the live bot's)")
        assert re.fullmatch(config.LIVE_DEFAULTS_SOURCE_PATTERN, source, re.ASCII)

    def test_the_save_target_s_source_note(self):
        today = date(2026, 9, 28)
        sweep = _flt_sweep()
        target = dashboard._save_target(sweep, _FLT_START, today, _FLT_SERIES)
        assert target == {"url": _SAVE_URL, "same_title_size_cap": None,
                          "filed_by_listing": True,
                          "source": "backtest dashboard for 2026-01-05 to 2026-09-28",
                          "live_adds_to_held_pairs": False, "live_sells": False,
                          # The most categories, and tags, one address may name
                          "max_names": config.DEFAULTS_SERVER_MAX_FILTER_NAMES}
        # ... which is True only for a run that recorded saved defaults adding
        for saved, said in ((True, True), (False, False), (None, False)):
            swept = dataclasses.replace(sweep, live_add_to_held_pairs=saved)
            assert dashboard._save_target(swept, _FLT_START, today, _FLT_SERIES)[
                "live_adds_to_held_pairs"] is said
        # ... and "live_sells" only for a recorded sell level (a minimum alone is no sale)
        for level, days, said in ((0.85, 3, True), (0.85, None, True), (None, None, False),
                                  (None, 3, False)):
            swept = dataclasses.replace(sweep, live_sell_at=level, live_sell_min_days=days)
            assert dashboard._save_target(swept, _FLT_START, today, _FLT_SERIES)[
                "live_sells"] is said
        # Only a recorded setting that differs from a recorded switch is named
        for ladders, configured in ((True, True), (False, None), (None, True)):
            swept = dataclasses.replace(sweep, same_event_ladders=ladders,
                                        config_same_event_ladders=configured)
            assert dashboard._save_target(swept, _FLT_START, today, None)["source"] == \
                "backtest dashboard for 2026-01-05 to 2026-09-28"
        swept = dataclasses.replace(sweep, same_event_ladders=True,
                                    config_same_event_ladders=False)
        assert dashboard._save_target(swept, _FLT_START, today, {})["source"].endswith(
            " (same-event ladders on, config.py off: its pairs are not the live bot's)")
        # No sweep: nothing recorded; an empty listing files by ticker prefix
        bare = dashboard._save_target(None, datetime(2026, 1, 5, 12, tzinfo=UTC), today, {})
        assert bare["same_title_size_cap"] is None and bare["filed_by_listing"] is False
        assert bare["source"] == "backtest dashboard for 2026-01-05 to 2026-09-28"

    def test_a_note_the_server_would_refuse_is_left_out(self, monkeypatch, tmp_path):
        # Should the note ever not fit the server's pattern, the button sends
        # none (the save then keeps no note) rather than a refused one
        monkeypatch.setattr(dashboard, "LIVE_DEFAULTS_SOURCE_PATTERN", "seed only")
        assert dashboard._save_target(_flt_sweep(), _FLT_START, date(2026, 9, 28),
                                      _FLT_SERIES)["source"] is None
        page = self._page(monkeypatch, tmp_path)
        data = TestFilterPage._data(page)
        snap = _run_script(tmp_path, page, [
            ["wait"], ["click", "flt-save"], ["snap", "s"]], strict=True)["s"]
        [opened] = snap["opened"]
        assert "source" not in self._query(data, opened)

    def test_an_unrecorded_same_title_cap_is_left_out(self, monkeypatch, tmp_path):
        # The server then keeps the saved same-title cap, or the seed's when
        # none is saved
        page = self._page(monkeypatch, tmp_path, same_title_size_cap=None)
        data = TestFilterPage._data(page)
        assert data["save"]["same_title_size_cap"] is None
        snap = _run_script(tmp_path, page, [
            ["wait"], ["click", "flt-save"], ["snap", "s"]], strict=True)["s"]
        [opened] = snap["opened"]
        assert "same_title_size_cap" not in self._query(data, opened)

    def test_a_filter_that_cannot_be_built_leaves_no_button(self, monkeypatch, tmp_path):
        def broken(*_a, **_k):
            """
            Stand in for _filter_payload: the filter's data cannot be built.

            Args:
                *_a: Ignored.
                **_k: Ignored.

            Raises:
                ValueError: Always.
            """
            raise ValueError("boom")
        monkeypatch.setattr(dashboard, "_filter_payload", broken)
        page = self._page(monkeypatch, tmp_path)
        assert 'id="flt-unavailable"' in page
        assert 'id="flt-save"' not in page and "<button" not in page
        assert html.escape(dashboard._SAVE_LABEL) not in page
        assert _page_elements(page)["buttons"] == {}

    def test_the_scenario_explorer_s_own_selects_are_not_saved(self, monkeypatch, tmp_path):
        # The explorer's band select moves only the explorer: the button still
        # sends the bar's scenario on screen
        base = _Grid.sweep()
        # Every point stamped with a size cap, so the bar's scenario can be saved
        scenarios = [dataclasses.replace(p, size_cap=0.2) for p in base.scenarios]
        primary = next(p for p in scenarios if p.population == "all"
                       and (p.spread_band, p.k) == (base.primary.spread_band, base.primary.k))
        sweep = dataclasses.replace(
            base, primary=primary, points=[primary], scenarios=scenarios,
            same_title_point=dataclasses.replace(base.same_title_point, size_cap=0.2),
            same_title_size_cap=0.5)
        page = TestScenarioExplorerScript._page(monkeypatch, tmp_path, sweep)
        data = TestFilterPage._data(page)
        pb = data["primary"][0]
        other = str(next(i for i in range(len(data["bands"])) if i != pb))
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["set", "scn-band-select", other], ["fire", "scn-band-select"],
            ["click", "flt-save"], ["snap", "s"]], explorer=True)
        snap = snaps["s"]
        assert snap["selects"]["scn-band-select"]["value"] == other
        assert snap["selects"]["flt-band"]["value"] == str(pb)
        [opened] = snap["opened"]
        query = self._query(data, opened)
        assert [float(query["spread_min"]), float(query["spread_max"])] == \
            data["bands"][pb]["value"]
        assert query["source"] == data["save"]["source"]

    @pytest.mark.parametrize("saved", [False, True])
    def test_the_address_is_what_the_defaults_server_accepts(
            self, monkeypatch, tmp_path, saved):
        # The clicked address, handed to the server's own application as a
        # browser would send it: it answers with the confirmation page, and its
        # parser reads exactly the scenario on screen — tier floors off, k
        # 0.60, a 20% cap, Sports · Hockey, the run's 0.5 same-title cap
        page = self._kc(monkeypatch, tmp_path, _kc_sweep_tiers())
        data = TestFilterPage._data(page)
        sports = data["categories"].index("Sports")
        hockey = str(data["subcats"].index([sports, "Hockey"]))
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["click", "flt-save"], ["snap", "primary"],
            ["set", "flt-tier", "off"], ["fire", "flt-tier"], ["settle"],
            ["set", "flt-k", "0"], ["fire", "flt-k"], ["settle"],
            _tick_tag(hockey), ["settle"],
            ["click", "flt-save"], ["snap", "chosen"]], strict=True)
        if saved:
            save_config_live_defaults()
        current = config.read_saved_live_defaults()
        assert (current is not None) is saved
        app = defaults_server._App(config.DEFAULTS_SERVER_PORT)
        pb, pk, pc = data["primary"]
        # The page names no add-to-held-pairs choice, so the server keeps the
        # saved one, or the seed's with none saved
        add_on = (current.add_to_held_pairs if current is not None
                  else config.LIVE_DEFAULTS_SEED.add_to_held_pairs)
        cases = (
            ("primary", config.LiveSettings(
                tier_floors=True, spread_band=tuple(data["bands"][pb]["value"]),
                interval_discount=data["ks"][pk]["value"],
                size_cap=data["caps"][pc]["value"], same_title_size_cap=0.5,
                add_to_held_pairs=add_on)),
            ("chosen", config.LiveSettings(
                tier_floors=False, spread_band=tuple(data["bands"][pb]["value"]),
                interval_discount=data["ks"][0]["value"],
                size_cap=data["caps"][pc]["value"], same_title_size_cap=0.5,
                categories=("Sports",), tags=("Sports · Hockey",), add_to_held_pairs=add_on)),
        )
        for name, expected in cases:
            [opened] = snaps[name]["opened"]
            self._query(data, opened)
            query = opened["url"].partition("?")[2]
            response = app.handle(defaults_server._Request("GET", f"/confirm?{query}",
                                                           _SAVE_HOST))
            assert response.status == 200, response.body
            settings, source = defaults_server._proposal(defaults_server._params(query),
                                                         current)
            assert settings == expected, name
            # Every field, one by one (equality skips none of the toggles)
            for field in config.LIVE_TOGGLE_FIELDS:
                assert getattr(settings, field) == getattr(expected, field), (name, field)
            assert source == data["save"]["source"]
        # Nothing was written by showing the page
        assert config.read_saved_live_defaults() == current

    def test_a_page_without_the_sell_view_sends_neither_sell_field(self, monkeypatch, tmp_path):
        # The server then keeps the saved sell level and minimum of days, as it
        # keeps the saved add-to-held choice from a page without that view
        config.save_live_defaults(
            dataclasses.replace(config.live_settings(), sell_at=0.9, sell_min_days=5),
            source="")
        current = config.read_saved_live_defaults()
        assert (current.sell_at, current.sell_min_days) == (0.9, 5)
        page = self._kc(monkeypatch, tmp_path)
        data = TestFilterPage._data(page)
        assert data["sell_blocks"] is None
        snap = _run_script(tmp_path, page, [
            ["wait"], ["click", "flt-save"], ["snap", "s"]], strict=True)["s"]
        [opened] = snap["opened"]
        query = self._query(data, opened)
        self._assert_sends(data, query, *data["primary"])
        assert "sell_at" not in query and "sell_min_days" not in query
        settings, _ = defaults_server._proposal(
            defaults_server._params(opened["url"].partition("?")[2]), current)
        assert (settings.sell_at, settings.sell_min_days) == (0.9, 5)

    @pytest.mark.parametrize("saved", ["nothing", "not selling", "selling"])
    def test_the_sell_scenario_on_screen_is_what_the_defaults_server_reads(
            self, monkeypatch, tmp_path, saved):
        # A page with the Sell view names its choice every time: "off" twice
        # under no selling (so saved defaults that sell are turned off, and
        # the seed is not leaned on), the level and the minimum of days when a
        # level is shown. The clicked address goes to the server's own
        # application and parser, whatever is saved.
        page = self._kc(monkeypatch, tmp_path, _kc_sweep_sell())
        data = TestFilterPage._data(page)
        assert data["sell_blocks"] is not None
        three = TestSellScript._di(3)
        snaps = TestSellScript._run(tmp_path, page, [
            ["wait"], ["click", "flt-save"], ["snap", "none"],
            ["set", "flt-sell", "0"], ["fire", "flt-sell"], ["settle"],
            ["click", "flt-save"], ["snap", "level"],
            ["set", "flt-days", three], ["fire", "flt-days"], ["settle"],
            ["click", "flt-save"], ["snap", "days"],
            ["set", "flt-sell", "1"], ["fire", "flt-sell"], ["settle"],
            ["click", "flt-save"], ["snap", "other"],
            ["set", "flt-sell", "none"], ["fire", "flt-sell"], ["settle"],
            ["click", "flt-save"], ["snap", "back"]], strict=True)
        if saved != "nothing":
            save_config_live_defaults()
        if saved == "selling":
            config.save_live_defaults(
                dataclasses.replace(config.live_settings(), sell_at=0.9, sell_min_days=5),
                source="")
        current = config.read_saved_live_defaults()
        assert (current is not None) is (saved != "nothing")
        app = defaults_server._App(config.DEFAULTS_SERVER_PORT)
        pb, pk, pc = data["primary"]
        # The page names no add-to-held-pairs choice, so the server keeps the
        # saved one, or the seed's with none saved
        add_on = (current.add_to_held_pairs if current is not None
                  else config.LIVE_DEFAULTS_SEED.add_to_held_pairs)
        levels, days = data["sell_levels"], data["sell_days"]
        assert levels[0]["value"] == 0.25 and levels[1]["value"] == 0.5
        assert [d["value"] for d in days][:3] == [1, 2, 3]
        # Another level keeps the minimum of days chosen; no selling sends none
        cases = (("none", None, None), ("level", 0.25, 1), ("days", 0.25, 3),
                 ("other", 0.5, 3), ("back", None, None))
        for name, level, minimum in cases:
            [opened] = snaps[name]["opened"]
            query = self._query(data, opened)
            self._assert_sends(data, query, pb, pk, pc)
            assert query["sell_at"] == ("off" if level is None else str(level)), name
            assert query["sell_min_days"] == ("off" if minimum is None else str(minimum)), name
            raw = opened["url"].partition("?")[2]
            response = app.handle(defaults_server._Request("GET", f"/confirm?{raw}",
                                                           _SAVE_HOST))
            assert response.status == 200, response.body
            settings, source = defaults_server._proposal(defaults_server._params(raw), current)
            expected = config.LiveSettings(
                tier_floors=True, spread_band=tuple(data["bands"][pb]["value"]),
                interval_discount=data["ks"][pk]["value"],
                size_cap=data["caps"][pc]["value"], same_title_size_cap=0.5,
                add_to_held_pairs=add_on, sell_at=level, sell_min_days=minimum)
            assert settings == expected, name
            for field in config.LIVE_TOGGLE_FIELDS:
                assert getattr(settings, field) == getattr(expected, field), (name, field)
            assert source == data["save"]["source"]
        # Nothing was written by showing the page
        assert config.read_saved_live_defaults() == current


# The defaults server's trade page, as the dashboard's link must name it
_TRADE_URL = f"http://{config.DEFAULTS_SERVER_HOST}:{config.DEFAULTS_SERVER_PORT}/trade"


class TestTradeUsingDefaultsLink:
    """The filter bar's "Trade using defaults…" link: a plain link to the
    defaults server's trade page, opened in a new tab, that needs no script;
    it stays on a page whose filter bar could not be built; and its id is
    what the defaults server looks for, early enough in the page for the
    server's scan to find it."""

    @staticmethod
    def _link(page: str) -> tuple[dict, str]:
        """
        Read the one Trade link on a page.

        Args:
            page (str): The page's HTML.

        Returns:
            tuple[dict, str]: The link's attributes (unescaped) and its text.

        Raises:
            AssertionError: If the page does not hold exactly one such link.
        """
        links = re.findall(r'<a id="flt-trade" ([^>]*)>(.*?)</a>', page)
        assert len(links) == 1, links
        attributes, text = links[0]
        return ({name: html.unescape(value)
                 for name, value in re.findall(r'(\w+)="([^"]*)"', attributes)},
                html.unescape(text))

    def test_it_opens_the_trade_page_in_a_new_tab(self, monkeypatch, tmp_path):
        page = TestSaveLiveDefaultsButton._page(monkeypatch, tmp_path)
        attributes, text = self._link(page)
        assert attributes["href"] == _TRADE_URL == dashboard._trade_url()
        assert attributes["target"] == "_blank" and attributes["rel"] == "noopener"
        assert attributes["title"] == dashboard._TRADE_TITLE
        assert attributes["style"] == dashboard._BUTTON_LINK_STYLE
        assert text == dashboard._TRADE_LABEL == "Trade using defaults…"
        # Right after the save button, before its note and the summary line
        assert page.index('id="flt-save"') < page.index('id="flt-trade"') \
            < page.index('id="flt-save-note"') < page.index('id="flt-summary"')
        # A plain link: the script neither names it nor words it, and it is no button
        assert "flt-trade" not in dashboard._FILTER_JS
        for words in (dashboard._TRADE_LABEL, dashboard._TRADE_TITLE):
            assert words not in dashboard._FILTER_JS
        assert _page_elements(page)["buttons"] == {"flt-save": {"disabled": True}}

    def test_its_address_is_read_from_config(self, monkeypatch):
        monkeypatch.setattr(dashboard, "DEFAULTS_SERVER_HOST", "127.0.0.9")
        monkeypatch.setattr(dashboard, "DEFAULTS_SERVER_PORT", 9123)
        assert dashboard._trade_url() == "http://127.0.0.9:9123/trade"
        assert 'href="http://127.0.0.9:9123/trade"' in dashboard._trade_link_html()

    def test_it_stays_when_the_filter_bar_cannot_be_built(self, monkeypatch, tmp_path):
        def broken(*_a, **_k):
            """
            Stand in for _filter_payload: the filter's data cannot be built.

            Args:
                *_a: Ignored.
                **_k: Ignored.

            Raises:
                ValueError: Always.
            """
            raise ValueError("boom")
        monkeypatch.setattr(dashboard, "_filter_payload", broken)
        page = TestSaveLiveDefaultsButton._page(monkeypatch, tmp_path)
        attributes, text = self._link(page)
        assert attributes["href"] == _TRADE_URL and text == dashboard._TRADE_LABEL
        # Under the notice that replaces the bar, in a paragraph of its own
        notice = page.index(dashboard._FILTER_UNAVAILABLE_HTML)
        paragraph = page.index('<p id="live-trade"')
        assert notice + len(dashboard._FILTER_UNAVAILABLE_HTML) < paragraph \
            < page.index('id="flt-trade"')
        assert html.escape(dashboard._SAVE_NOTE) in page[paragraph:]
        # Still no button: the save button needs the bar's data
        assert 'id="flt-save"' not in page and "<button" not in page

    def test_the_note_and_the_save_button_s_words(self):
        assert dashboard._SAVE_NOTE == "(needs ./start_dashboard.sh running)"
        # The confirmation page's buttons, named as that page names them
        for words in ("Confirm and save saves it", "Confirm and trade saves it and then "
                      "runs the live bot with it, placing real orders", "Dry run runs the "
                      "live bot with it without placing orders and saves nothing"):
            assert words in dashboard._SAVE_TITLE, words
        assert "when Confirm is clicked" not in dashboard._SAVE_TITLE
        assert dashboard._TRADE_TITLE == (
            "Open a page, in a new tab, that shows the saved live defaults and runs the live "
            "bot with them when you confirm there (real orders, or a dry run).")

    @pytest.mark.parametrize("bar", [True, False], ids=["bar", "no-bar"])
    def test_the_defaults_server_finds_the_buttons_early_in_the_page(
            self, monkeypatch, tmp_path, bar):
        if not bar:
            def broken(*_a, **_k):
                """
                Stand in for _filter_payload: the filter's data cannot be built.

                Args:
                    *_a: Ignored.
                    **_k: Ignored.

                Raises:
                    ValueError: Always.
                """
                raise ValueError("boom")
            monkeypatch.setattr(dashboard, "_filter_payload", broken)
        TestSaveLiveDefaultsButton._page(monkeypatch, tmp_path)
        path = tmp_path / config.DASHBOARD_FILENAME
        data = path.read_bytes()
        marker = defaults_server._DASHBOARD_MARKER
        # In the page's opening part, before the first section (and so before
        # every data block), well inside what the server reads
        first_section = dashboard._SECTION_STYLE.format(title="Portfolio Performance")
        assert data.count(marker) == 1
        assert data.index(marker) < data.index(first_section.encode("utf-8"))
        assert data.index(marker) + len(marker) <= config.DASHBOARD_MARKER_SCAN_BYTES
        assert defaults_server._dashboard_state(path) == (defaults_server._DASHBOARD_READY,
                                                          None)


def _kpi_trades(snap: dict) -> list[tuple[str, str]]:
    """(population label, trade count) of every row of the explorer's KPI
    table as the script last wrote it (its sub-rows carry no count)."""
    cell = '<td style="padding:6px 16px;">'
    return re.findall(rf'<tr style="border-bottom:1px solid #E0E0E0">{cell}([^<]*)</td>'
                      rf"{cell}([^<]*)</td>", snap["html"]["scn-kpi-body"])


def _explorer_calls(snap: dict, fn: str | None = None, chart: str | None = None) -> list[dict]:
    """The Plotly calls a snapshot recorded on the explorer's charts (its
    ids start "scn-"), in order — optionally only one function's, and only
    one chart's. The harness records them as snap["updates"] ({id, kind,
    data, layout, traces}); each is handed back with its kind as "fn" too
    (#68's name)."""
    return [dict(c, fn=c["kind"]) for c in snap["updates"]
            if c["id"].startswith("scn-") and (fn is None or c["kind"] == fn)
            and (chart is None or c["id"] == chart)]


def _heatmap_calls(snap: dict) -> list[tuple[str, str]]:
    """(function, chart) of every Plotly call a snapshot recorded on the
    explorer's heatmaps — every explorer call but the curve's."""
    return [(c["fn"], c["id"]) for c in _explorer_calls(snap) if c["id"] != "scn-equity"]


class TestScenarioExplorerScript:
    """The scenario explorer's script, run with the page-wide filter script
    over the page Python rendered: its own Tier floors select — which the
    bar's Tier floors choice moves too, through window.dashScenarioSelect —
    redraws the heatmap (its band labels included), the banner, the k-hat
    table shown, the tables and the curve for the setting chosen. Skipped
    without a runtime."""

    TS = _TS_LABEL
    # A KPI / calibration table cell, as the script writes one
    TD = '<td style="padding:6px 16px;">'

    @staticmethod
    def _page(monkeypatch, tmp_path, sweep) -> str:
        monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(dashboard.yf, "download",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
        return dashboard.generate_dashboard(
            sweep.primary.trades, sweep.primary.equity_df, date(2026, 1, 5), 1000.0,
            sweep=sweep).read_text(encoding="utf-8")

    def _rows(self, ts: str, ladder: str, cross: str, all_: str) -> list[tuple[str, str]]:
        # Same-title's row is the same under both Tier floors settings
        return [(self.TS, ts), ("Ladders (same-event)", ladder), ("Cross-event", cross),
                ("All (time-series + same-title)", all_),
                ("Same-title (independent of band, k and the tier floors)", "7")]

    @staticmethod
    def _blocks(page: str) -> tuple[dict, dict, dict]:
        """The explorer's base block, its one cap block and its off block."""
        blocks = _scn_blocks(_ex_section(page))
        return blocks["scn-data"], blocks["scn-cap-0"], blocks.get("scn-off-cap-0")

    @staticmethod
    def _curve(snap: dict) -> dict:
        """The explorer curve's last redraw in a snapshot."""
        return _explorer_calls(snap, "update", chart="scn-equity")[-1]

    @staticmethod
    def _heat(snap: dict) -> list[dict]:
        """The heatmap's redraws in a snapshot."""
        return _explorer_calls(snap, "update", chart="scn-heatmap")

    @staticmethod
    def _assert_no_off_view(page: str) -> None:
        # No Tier floors select, no hidden off table, no off block: an
        # explorer tier-on whatever the bar's Tier floors select reads
        assert 'id="scn-tier-select"' not in page and 'id="scn-khat-off"' not in page
        assert 'id="scn-off-cap-' not in page
        data = _scn_data(_ex_section(page))
        for key in ("tier_binds", "off_caps", "tier_labels", "band_options_off",
                    "calibration_by_band_off"):
            assert data[key] is None, key
        assert data["curve_titles"][1] is None

    def test_the_explorer_follows_the_tier_floors_select_and_back(self, monkeypatch, tmp_path):
        # The BAR's select: its change reaches the explorer through
        # window.dashScenarioSelect, which moves the explorer's own select
        page = self._page(monkeypatch, tmp_path, _Grid.sweep_tiers())
        data, cap, off_block = self._blocks(page)
        ts = data["populations"].index("time_series")
        n = len(data["dates"])
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["set", "scn-band-select", "0"], ["fire", "scn-band-select"],
            ["snap", "on0"],
            ["set", "flt-tier", "off"], ["fire", "flt-tier"], ["settle"], ["snap", "off"],
            ["set", "scn-band-select", "1"], ["fire", "scn-band-select"], ["snap", "off1"],
            ["set", "scn-band-select", "0"], ["fire", "scn-band-select"],
            ["set", "flt-tier", "on"], ["fire", "flt-tier"], ["settle"], ["snap", "on"]],
            explorer=True)
        on0, off, off1, on = snaps["on0"], snaps["off"], snaps["off1"], snaps["on"]
        assert not on0["selects"]["flt-tier"]["disabled"]
        assert not on0["selects"]["scn-tier-select"]["disabled"]
        # Band 0 (the one a tier binds at), k 0.75: its tier-on cell ...
        assert _kpi_trades(on0) == self._rows("2", "11", "21", "2")
        assert "0-7d" in on0["html"]["scn-cal-body"]
        # ... then, with the tiers off, the explorer's own select moved ...
        assert off["selects"]["scn-tier-select"]["value"] == "off"
        assert "scn-off-cap-0" in off["inflated"]
        # ... the band options named for the setting, the band kept ...
        assert [text for _, text in off["selects"]["scn-band-select"]["options"]] == \
            data["band_options_off"] == ["0-1", "0.3-0.6 (primary)", "0.35-0.8"]
        assert off["selects"]["scn-band-select"]["value"] == "0"
        # ... the k-hat table Python drew for it shown, the banner its own ...
        assert (off["display"]["scn-khat-on"], off["display"]["scn-khat-off"]) == ("none", "")
        assert off["html"]["scn-banner"] == off_block["banner"]
        # ... its tier-off cell in the KPI table, calibration and curve ...
        assert _kpi_trades(off) == self._rows("31", "51", "61", "41")
        assert "16-30d" in off["html"]["scn-cal-body"]
        assert "0-7d" not in off["html"]["scn-cal-body"]
        curve = self._curve(off)
        assert curve["data"]["y"] == [_expand(off_block["cells"][0][1][ts]["equity"], n)]
        assert curve["layout"]["title.text"] == data["curve_titles"][1]
        # ... and the heatmap redrawn from the off block, its rows relabelled
        (heat,) = self._heat(off)
        assert heat["data"]["z"] == [off_block["matrices"]["mean_per_trade"]]
        assert heat["data"]["y"] == [data["band_labels_off"]]
        assert heat["layout"] == {"title.text": off_block["titles"][0]}
        # A band the tiers never bind at reads its tier-on cell and calibration
        assert _kpi_trades(off1) == self._rows("4", "13", "23", "4")
        assert "8-15d" in off1["html"]["scn-cal-body"]
        assert self._heat(off1) == []
        # Back on: the page as Python rendered it
        assert on["selects"]["scn-tier-select"]["value"] == "on"
        assert (on["display"]["scn-khat-on"], on["display"]["scn-khat-off"]) == ("", "none")
        assert [text for _, text in on["selects"]["scn-band-select"]["options"]] == \
            data["band_options"]
        assert _kpi_trades(on) == _kpi_trades(on0)
        assert on["html"]["scn-cal-body"] == on0["html"]["scn-cal-body"]
        assert on["html"]["scn-banner"] == cap["banner"]
        (heat,) = self._heat(on)
        assert heat["data"]["z"] == [cap["matrices"]["mean_per_trade"]]
        assert heat["data"]["y"] == [data["band_labels"]]
        assert heat["layout"] == {"title.text": cap["titles"][0]}
        assert on["layouts"]["scn-heatmap"]["title"]["text"] == cap["titles"][0]
        assert self._curve(on)["data"]["y"] == [_expand(cap["cells"][0][1][ts]["equity"], n)]
        assert self._curve(on)["layout"]["title.text"] == data["curve_titles"][0]

    def test_the_bar_s_tier_choice_moves_the_explorer_s_select(self, monkeypatch, tmp_path):
        # (Replaces #68's test_the_heatmap_shown_is_resized_even_without_a_menu_to_read
        # — both params: one heatmap, never drawn hidden, so there is no
        # resize or menu to replay; what the setting must change on it is its
        # rows.) The bar's call names its Tier floors choice as the fourth
        # label: it alone moves the explorer's Tier floors select, which the
        # heatmap's band labels then follow
        page = self._page(monkeypatch, tmp_path, _Grid.sweep_tiers())
        data, cap, off_block = self._blocks(page)
        band, k, size = (data["band_labels"][1], data["k_labels"][1], data["cap_labels"][0])
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["call", "dashScenarioSelect", [band, k, size, "off"]], ["settle"],
            ["snap", "off"],
            ["call", "dashScenarioSelect", [band, k, size, "off"]], ["settle"],
            ["snap", "again"],
            ["call", "dashScenarioSelect", [band, k, size, "on"]], ["settle"],
            ["snap", "on"]], strict=True, explorer=True)
        off, again, on = snaps["off"], snaps["again"], snaps["on"]
        assert _scn_values(off) + [off["selects"]["scn-tier-select"]["value"]] == [
            "1", "1", "0", "off"]
        assert [h["data"]["y"] for h in self._heat(off)] == [[data["band_labels_off"]]]
        assert off["layouts"]["scn-heatmap"]["title"]["text"] == off_block["titles"][0]
        # The same labels again: nothing moved, nothing is drawn
        assert again["updates"] == []
        assert [h["data"]["y"] for h in self._heat(on)] == [[data["band_labels"]]]
        assert on["selects"]["scn-tier-select"]["value"] == "on"
        assert on["layouts"]["scn-heatmap"]["title"]["text"] == cap["titles"][0]

    def test_the_metric_carries_across_a_toggle(self, monkeypatch, tmp_path):
        # (#68's test, on the scripted metric select rather than a menu: the
        # metric chosen stays chosen, and each setting draws it from its own
        # block, titled for it.)
        page = self._page(monkeypatch, tmp_path, _Grid.sweep_tiers())
        data, cap, off_block = self._blocks(page)
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["set", "scn-metric", "2"], ["fire", "scn-metric"],
            ["set", "scn-tier-select", "off"], ["fire", "scn-tier-select"], ["settle"],
            ["snap", "off"],
            ["set", "scn-metric", "4"], ["fire", "scn-metric"], ["snap", "metric"],
            ["set", "scn-tier-select", "on"], ["fire", "scn-tier-select"], ["snap", "on"]],
            strict=True, explorer=True)
        # The metric change (tier-on), then the toggle (the off block's)
        metric2, off = self._heat(snaps["off"])
        assert metric2["data"]["z"] == [cap["matrices"]["total_return"]]
        assert metric2["layout"] == {"title.text": cap["titles"][2]}
        assert off["data"]["z"] == [off_block["matrices"]["total_return"]]
        assert off["data"]["y"] == [data["band_labels_off"]]
        assert off["layout"] == {"title.text": off_block["titles"][2]}
        (metric,) = self._heat(snaps["metric"])
        assert metric["data"]["z"] == [off_block["matrices"]["h1_return"]]
        assert metric["layout"] == {"title.text": off_block["titles"][4]}
        (on,) = self._heat(snaps["on"])
        assert on["data"]["z"] == [cap["matrices"]["h1_return"]]
        assert on["data"]["y"] == [data["band_labels"]]
        assert on["layout"] == {"title.text": cap["titles"][4]}
        assert snaps["on"]["selects"]["scn-metric"]["value"] == "4"

    @pytest.mark.parametrize("family", [True, False], ids=["tier-off-family", "no-family"])
    def test_a_bar_that_is_not_live_changes_nothing(self, monkeypatch, tmp_path, family):
        # With the family, a bar that cannot inflate its data never enables
        # its Tier floors select (here its base block is damaged: the
        # explorer's own blocks, packed as well, still load — #68's run
        # without DecompressionStream would now stop the explorer too); without
        # it, the bar keeps that select shut. Either way a change event
        # reaching it moves nothing in the explorer.
        page = self._page(monkeypatch, tmp_path,
                          _Grid.sweep_tiers() if family else _Grid.sweep())
        data, cap, _ = self._blocks(page)
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["snap", "ready"],
            ["set", "flt-tier", "off"], ["fire", "flt-tier"], ["settle"], ["snap", "fired"]],
            explorer=True, damaged=("dash-data",) if family else ())
        ready, fired = snaps["ready"], snaps["fired"]
        assert ready["selects"]["flt-tier"]["disabled"]
        # The explorer drew its primary cell on load, with the floors on ...
        ts = data["populations"].index("time_series")
        assert self._curve(ready)["data"]["y"] == [
            _expand(cap["cells"][1][1][ts]["equity"], len(data["dates"]))]
        # ... and the event moves nothing: no call, no table shown or hidden,
        # no option renamed, no table rewritten
        assert _explorer_calls(fired) == []
        assert "scn-khat-on" not in fired["display"] and "scn-khat-off" not in fired["display"]
        assert fired["selects"]["scn-band-select"] == ready["selects"]["scn-band-select"]
        assert fired["html"]["scn-kpi-body"] == ready["html"]["scn-kpi-body"]
        assert fired["html"]["scn-cal-body"] == ready["html"]["scn-cal-body"]
        if family:
            assert fired["selects"]["scn-tier-select"]["value"] == "on"

    def test_a_family_missing_a_binding_band_offers_no_off_view_anywhere(
            self, monkeypatch, tmp_path):
        # (Replaces #68's test_an_explorer_without_a_complete_family_stays_on:
        # there the explorer's grid could hold a band the bar's did not, so
        # the bar could be off while the explorer stayed on. Here the bar and
        # the explorer walk ONE grid, so a family that does not name a band
        # the tiers bind at withholds the off view from both.)
        sweep = _Grid.sweep_tiers()
        extra = _scn_point((0.2, 0.6), 0.65, "all")
        sweep = dataclasses.replace(sweep, scenarios=[*sweep.scenarios, extra])
        page = self._page(monkeypatch, tmp_path, sweep)
        self._assert_no_off_view(page)
        assert 'id="flt-tier-note"' in page and TestFilterPage._data(page)["grid_off"] is None
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["snap", "ready"],
            ["set", "flt-tier", "off"], ["fire", "flt-tier"], ["settle"], ["snap", "off"]],
            explorer=True)
        ready, off = snaps["ready"], snaps["off"]
        assert ready["selects"]["flt-tier"]["disabled"]
        assert _explorer_calls(off) == []
        assert off["html"]["scn-kpi-body"] == ready["html"]["scn-kpi-body"]

    def test_a_page_whose_bar_could_not_be_built_keeps_the_explorer_s_off_view(
            self, monkeypatch, tmp_path):
        # (Replaces #68's test_a_page_whose_bar_could_not_be_built_draws_no_off_view,
        # reversed by design: #68's explorer followed only the bar's select, so
        # without the bar its off view was unreachable and not drawn; this
        # explorer's own select reaches it.) The filter's payload fails: the
        # page is written without the bar and its script — and, under strict
        # ids (no flt-tier on this page reads null, as in a browser), the
        # explorer loads without an error and switches on its own select
        def broken(*_a, **_k):
            raise ValueError("boom")
        monkeypatch.setattr(dashboard, "_filter_payload", broken)
        page = self._page(monkeypatch, tmp_path, _Grid.sweep_tiers())
        assert 'id="flt-unavailable"' in page and 'id="flt-tier"' not in page
        assert 'id="scn-tier-select"' in page and 'id="scn-off-cap-0"' in page
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["snap", "ready"],
            ["set", "scn-band-select", "0"], ["fire", "scn-band-select"], ["snap", "band0"],
            ["set", "scn-tier-select", "off"], ["fire", "scn-tier-select"], ["settle"],
            ["snap", "off"]],
            explorer=True, strict_ids=True)
        ready, band0, off = snaps["ready"], snaps["band0"], snaps["off"]
        assert "flt-tier" not in ready["selects"]
        assert _kpi_trades(ready) == self._rows("4", "13", "23", "4")
        assert _kpi_trades(band0) == self._rows("2", "11", "21", "2")
        assert self._heat(band0) == []
        assert _kpi_trades(off) == self._rows("31", "51", "61", "41")
        data = _scn_data(_ex_section(page))
        assert [h["data"]["y"] for h in self._heat(off)] == [[data["band_labels_off"]]]

    def test_a_cell_the_family_lacks_reads_as_missing(self, monkeypatch, tmp_path):
        # (Replaces #68's test_a_bar_with_no_off_view_draws_none_in_the_explorer:
        # #68 withheld the whole off view when the family lacked band 0's
        # tier-off "all" point at the primary k; on this grid band 0 is still
        # a binding band, so that ONE scenario is "missing" — the bar's
        # grid_off cell, and the explorer's All row — and every other one is
        # shown.)
        sweep = _Grid.sweep_tiers()
        sweep = dataclasses.replace(sweep, tier_off_scenarios=[
            pt for pt in sweep.tier_off_scenarios
            if not (pt.population == "all" and pt.k == sweep.primary.k)])
        assert dashboard._tier_off_binds(sweep, _Grid.BANDS) == [True, False, False]
        page = self._page(monkeypatch, tmp_path, sweep)
        base = TestFilterPage._data(page)
        assert base["grid_off"][0][1] == [None] and base["grid_off"][0][0] != [None]
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["set", "scn-band-select", "0"], ["fire", "scn-band-select"],
            ["set", "scn-tier-select", "off"], ["fire", "scn-tier-select"], ["settle"],
            ["snap", "off"],
            ["set", "scn-k-select", "0"], ["fire", "scn-k-select"], ["snap", "k0"]],
            strict_ids=True, explorer=True)
        assert _kpi_trades(snaps["off"]) == self._rows("31", "51", "61", "—")
        assert _kpi_trades(snaps["k0"]) == self._rows("30", "50", "60", "40")

    def test_a_zoom_never_carries_into_another_view(self, monkeypatch, tmp_path):
        # A reader zooms the curve, then changes the Tier floors setting, the
        # band or k: every redraw autoranges both axes (a Plotly.update), so
        # the next curve is never drawn inside the last one's zoom
        page = self._page(monkeypatch, tmp_path, _Grid.sweep_tiers())
        data, _, _ = self._blocks(page)
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["zoom", "scn-equity"], ["snap", "zoomed"],
            ["set", "scn-tier-select", "off"], ["fire", "scn-tier-select"], ["settle"],
            ["snap", "tier"],
            ["zoom", "scn-equity"],
            ["set", "scn-band-select", "0"], ["fire", "scn-band-select"], ["snap", "band"],
            ["zoom", "scn-equity"],
            ["set", "scn-k-select", "0"], ["fire", "scn-k-select"], ["snap", "k"]],
            explorer=True)
        # The zoom took: the reader's range, autorange off ...
        assert snaps["zoomed"]["layouts"]["scn-equity"]["xaxis"]["autorange"] is False
        # ... and each change redraws the curve with both axes autoranged
        for name in ("tier", "band", "k"):
            layout = snaps[name]["layouts"]["scn-equity"]
            assert layout["xaxis"]["autorange"] is True, name
            assert layout["yaxis"]["autorange"] is True, name
            assert self._curve(snaps[name])["layout"] == {
                "xaxis.autorange": True, "yaxis.autorange": True,
                "title.text": data["curve_titles"][1]}, name
        # The curve is never restyled (a restyle would leave the zoom in place)
        assert not any(_explorer_calls(s, "restyle") for s in snaps.values())

    def test_the_curve_is_titled_for_the_setting_shown(self, monkeypatch, tmp_path):
        # The off title names the setting, as the off heatmap's does; both
        # titles are Python's, from the base block — the script words none
        page = self._page(monkeypatch, tmp_path, _Grid.sweep_tiers())
        data, _, _ = self._blocks(page)
        rendered = _page_elements(page)["charts"]["scn-equity"]["layout"]["title"]["text"]
        assert data["curve_titles"] == [rendered, rendered + " — tier floors off"]
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["snap", "ready"],
            ["set", "scn-tier-select", "off"], ["fire", "scn-tier-select"], ["settle"],
            ["snap", "off"],
            ["set", "scn-tier-select", "on"], ["fire", "scn-tier-select"], ["snap", "on"]],
            explorer=True)
        assert snaps["ready"]["layouts"]["scn-equity"]["title"]["text"] == rendered
        for name, title in (("ready", rendered), ("off", data["curve_titles"][1]),
                            ("on", rendered)):
            assert [c["layout"]["title.text"] for c in _explorer_calls(
                snaps[name], "update", chart="scn-equity")] == [title], name
            assert snaps[name]["layouts"]["scn-equity"]["title"]["text"] == title, name

    def test_every_part_describes_the_setting_drawn(self, monkeypatch, tmp_path):
        # (Replaces #68's test_the_tables_and_curve_read_the_setting_shown.
        # #68 read a cell under the setting its blocks showed even when the
        # bar's select had changed without its event; here the explorer's
        # selects are all its own, so a band or k change draws what EVERY
        # select names — its Tier floors select included — and the heatmap,
        # banner, k-hat table, KPI table, calibration table and curve all
        # move together, never describing two settings at once.)
        page = self._page(monkeypatch, tmp_path, _Grid.sweep_tiers())
        data, cap, off_block = self._blocks(page)
        ts = data["populations"].index("time_series")
        n = len(data["dates"])
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["set", "scn-tier-select", "off"],
            ["set", "scn-band-select", "0"], ["fire", "scn-band-select"], ["settle"],
            ["snap", "off"],
            ["set", "scn-tier-select", "on"],
            ["set", "scn-k-select", "0"], ["fire", "scn-k-select"], ["snap", "on"]],
            explorer=True)
        off, on = snaps["off"], snaps["on"]
        assert (off["display"]["scn-khat-on"], off["display"]["scn-khat-off"]) == ("none", "")
        assert off["html"]["scn-banner"] == off_block["banner"]
        assert [h["data"]["y"] for h in self._heat(off)] == [[data["band_labels_off"]]]
        assert _kpi_trades(off) == self._rows("31", "51", "61", "41")
        assert "16-30d" in off["html"]["scn-cal-body"]
        assert self._curve(off)["data"]["y"] == [
            _expand(off_block["cells"][0][1][ts]["equity"], n)]
        # Band 0, k 0.65, the Tier floors select back on: every part tier-on
        assert (on["display"]["scn-khat-on"], on["display"]["scn-khat-off"]) == ("", "none")
        assert on["html"]["scn-banner"] == cap["banner"]
        assert [h["data"]["y"] for h in self._heat(on)] == [[data["band_labels"]]]
        assert _kpi_trades(on) == self._rows("1", "10", "—", "1")
        assert "0-7d" in on["html"]["scn-cal-body"]
        assert self._curve(on)["data"]["y"] == [_expand(cap["cells"][0][0][ts]["equity"], n)]

    def test_a_floor_0_tier_reads_as_a_dash(self, monkeypatch, tmp_path):
        # A tier-off calibration at a floor of 0 labels its buckets with that
        # floor alone, 0.0, which the base block ships as null: the table the
        # script draws prints "—" there, as for the pooled row — never 0.00
        page = self._page(monkeypatch, tmp_path, _Grid.sweep_tiers())
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["set", "scn-band-select", "0"], ["fire", "scn-band-select"],
            ["snap", "on"], ["set", "scn-tier-select", "off"], ["fire", "scn-tier-select"],
            ["settle"], ["snap", "off"]],
            explorer=True)
        td = self.TD
        on, off = snaps["on"]["html"]["scn-cal-body"], snaps["off"]["html"]["scn-cal-body"]
        assert f"{td}0-7d</td>{td}0.15</td>{td}6</td>" in on
        assert f"{td}16-30d</td>{td}—</td>{td}7</td>" in off
        assert f"{td}POOLED</td>{td}—</td>{td}12</td>" in off
        assert f"{td}0.00</td>" not in off

    def test_the_top_event_is_escaped(self, monkeypatch, tmp_path):
        # The top event's ticker is Kalshi-controlled: the script escapes it
        # (esc) before it reaches the KPI table's HTML
        page = self._page(monkeypatch, tmp_path, _Grid.sweep(top_event="EVT-</script><b>x"))
        kpi = _run_script(tmp_path, page, [["wait"], ["snap", "ready"]],
                          explorer=True)["ready"]["html"]["scn-kpi-body"]
        assert "Top event: EVT-&lt;/script&gt;&lt;b&gt;x (" in kpi
        assert "<b>x" not in kpi and "</script>" not in kpi


class TestKhatBreakdown:
    """The k-hat chart: each band's carried k-hat population, regrouped by
    category and tag through backtester._calibration_bucket."""

    FILED = [("KXNHLHART-27", "Other"), ("KXSPACEX-14", "Science"),
             ("KXOTHER-1", "Other"), ("KXNHLHART-27", "Other")]

    @staticmethod
    def _band(cal):
        return dashboard._khat_band(cal, _FLT_SERIES, {"Other": 0, "Science": 1, "Sports": 2},
                                    {("Other", "General"): 0, ("Science", "General"): 1,
                                     ("Sports", "Hockey"): 2})

    @staticmethod
    def _one_band_base(calibration) -> dict:
        """The base block of a one-scenario grid (band 0-1, k 0.75, 20%) whose
        band carries `calibration`."""
        trades = _flt_trades()
        curve = backtester._build_equity_curve(trades, _FLT_START, 1000.0)
        point = SweepPoint(k=0.75, trades=trades, equity_df=curve, spread_band=(0.0, 1.0),
                           size_cap=0.2)
        source = dashboard._GridSource(
            bands=((0.0, 1.0),), ks=(0.75,), caps=(0.2,), primary=(0, 0, 0),
            cell=lambda band, k: {0.2: {"all": point}},
            calibrations={(0.0, 1.0): calibration})
        return _flt_walk(source, trades, curve)[1]

    def test_the_all_group_is_the_pooled_row_exactly(self):
        cal = _flt_calibration(self.FILED)
        # The fixture has teeth: summed in another order its implied gaps
        # differ, so only the carried tuple itself matches the pooled row
        flipped = backtester._calibration_bucket("", 0.0, cal.observations[::-1])
        assert flipped.mean_implied != cal.pooled.mean_implied
        band = self._band(cal)
        assert band["carried"] is True
        whole = band["groups"]["all"]
        assert (whole["n"], whole["rate"], whole["implied"], whole["k"]) == (
            cal.pooled.n, cal.pooled.realised_rate, cal.pooled.mean_implied,
            cal.pooled.empirical_k)
        # Two entries of one event count as one event
        assert whole["events"] == 3
        # Categories partition the population, each reduced by the one
        # k-hat definition
        assert sum(band["groups"][f"c{i}"]["n"] for i in range(3)) == whole["n"]
        hockey = [o for o in cal.observations if o.event_ticker.startswith("KXNHL")]
        bucket = backtester._calibration_bucket("", 0.0, hockey)
        assert band["groups"]["s2"]["k"] == bucket.empirical_k
        assert band["groups"]["c2"]["n"] == len(hockey) == 2
        assert whole["text"] == "n=4 · 3 ev" and whole["cells"] == dashboard._khat_cells(whole)

    def test_cells_are_formatted_once_in_python(self):
        # 1/32 is a binary tie at 4 dp: Python rounds it to even (0.0312)
        # where a browser's toFixed gives 0.0313 — the script shows these
        # cells and formats none of its own
        stat = dashboard._khat_finish({"n": 32, "events": 5, "rate": 1 / 32,
                                       "implied": 0.5, "k": 0.0625})
        assert stat["cells"] == ["32", "5", "0.0312", "0.5000", "0.062"]
        assert stat["text"] == "n=32 · 5 ev"
        assert dashboard._khat_cells(None) == ["—"] * 5

    def test_a_calibration_without_its_population_keeps_only_the_pooled_row(self):
        cal = IntervalCalibration(
            pooled=IntervalCalibrationBucket("POOLED", 0.0, 12, 0.25, 0.5, 0.5),
            buckets=[], excluded_premise_violations=0)          # hand-built: () carried
        band = dashboard._khat_band(cal, None, {}, {})
        assert band == {"carried": False, "groups": {"all": {
            "n": 12, "events": None, "rate": 0.25, "implied": 0.5, "k": 0.5,
            "text": "n=12 · ? ev", "cells": ["12", "—", "0.2500", "0.5000", "0.500"]}}}
        assert dashboard._khat_band(None, None, {}, {}) is None

    def test_the_payload_carries_every_band_in_band_order(self):
        _, source, base, _ = _flt_payload()
        assert len(base["khat"]) == len(source.bands) == 3
        assert base["khat"][1] is None and base["khat"][2] is None
        groups = base["khat"][0]["groups"]
        # Science (seen only in an observation) is a group; Commodities (seen
        # only in trades) is not
        assert _key(base, "Science") in groups
        assert _key(base, "Commodities") not in groups
        assert base["khat_blank"] == ["—"] * 5
        assert base["styles"]["khat_height"] == list(dashboard._KHAT_HEIGHT)
        least, per_row, axes = dashboard._KHAT_HEIGHT
        assert dashboard._khat_chart_height(9) == max(least, per_row * 9 + axes)
        # The reference line the script redraws at any k: no x, no text
        ref = base["styles"]["khat_ref"]
        assert "x0" not in ref["shape"] and "x" not in ref["annotation"]
        assert "text" not in ref["annotation"]

    def test_the_default_render_is_by_category_at_the_primary_band(self):
        _, _, base, _ = _flt_payload()
        section = dashboard._section_khat(base, 0.625)
        data, layout = _nth_figure(section, 0)
        groups = base["khat"][0]["groups"]
        assert data[0]["y"] == ["All categories", "Other", "Science", "Sports"]
        assert data[0]["x"] == pytest.approx(
            [groups["all"]["k"], groups["c1"]["k"], groups["c2"]["k"], groups["c3"]["k"]])
        assert data[0]["marker"]["color"] == ["#9E9E9E", "#2196F3", "#2196F3", "#2196F3"]
        assert data[0]["text"][0] == groups["all"]["text"]
        # The hover shows the table's own cells, with no number format of its own
        assert data[0]["customdata"][0] == groups["all"]["cells"]
        assert ":." not in data[0]["hovertemplate"]
        assert (data[0]["textposition"], data[0]["cliponaxis"]) == ("outside", False)
        assert layout["title"]["text"] == (
            "Empirical k̂ by category — the primary spread band max(tier,0)-1")
        # The line marks the k the run was sized at, printed exactly — drawn
        # from the base block's line, so the script redraws the same one
        assert layout["shapes"][0] == {**base["styles"]["khat_ref"]["shape"],
                                       "x0": 0.625, "x1": 0.625}
        assert layout["annotations"][0] == {**base["styles"]["khat_ref"]["annotation"],
                                            "x": 0.625, "text": "sized at k = 0.625"}
        # The table repeats the bars, one row each, with Python's cells
        assert section.count("<tr style='border-bottom:1px solid #E0E0E0'>") == 4
        assert "".join(f"<td style='padding:4px 12px;'>{c}</td>"
                       for c in groups["all"]["cells"]) in section

    def test_the_section_reads_the_primary_scenario_s_band(self):
        # The primary band sorts second: the section reads primary[0]
        sweep = _flt_sweep()
        _, _, base, _ = _flt_payload(dataclasses.replace(sweep, primary=sweep.scenarios[1]))
        assert base["primary"][0] == 1 and base["khat"][1] is None
        section = dashboard._section_khat(base, 0.75)
        assert html.escape(dashboard._KHAT_TEXT["khat_not_recorded"]) in section
        assert "the primary spread band max(tier,0.3)-0.6" in section

    def test_group_by_stays_outside_the_body_it_hides(self):
        # A selection with nothing to draw hides the body; the grouping that
        # could draw something must stay reachable, so the select sits
        # before the notice and outside the body — disabled until the
        # script has its data, and never restored by a reload
        _, _, base, _ = _flt_payload()
        section = dashboard._section_khat(base, 0.75)
        select = section.index('<select id="khat-group" disabled autocomplete="off">')
        assert select < section.index('<p id="khat-empty"') < section.index('<div id="khat-body"')
        assert '<option value="band">' in section

    def test_a_band_without_a_calibration_says_it_was_not_recorded(self):
        section = dashboard._section_khat(self._one_band_base(None), None)
        notice = html.escape(dashboard._KHAT_TEXT["khat_not_recorded"])
        assert f'color:#616161;">{notice}</p>' in section
        assert '<div id="khat-body" style="display:none">' in section
        assert "shapes" not in json.dumps(_nth_figure(section, 0)[1])   # no k to mark

    def test_an_empty_population_is_not_called_unrecorded(self):
        # Every candidate a premise violation: measured, and nothing counted
        empty = IntervalCalibration(
            pooled=backtester._calibration_bucket("POOLED", 0.0, ()), buckets=[],
            excluded_premise_violations=3, observations=())
        section = dashboard._section_khat(self._one_band_base(empty), 0.75)
        assert html.escape(dashboard._KHAT_TEXT["khat_none"]) in section
        assert '<div id="khat-body" style="display:none">' in section

    def test_without_the_filter_data_the_section_says_so(self):
        section = dashboard._section_khat(None, 0.75)
        assert "could not be built" in section and "khat-group" not in section

    def test_the_intro_names_the_midpoint_basis(self):
        # DR-78: the intro states k-hat's basis, the mean mid spread, in the
        # same words as the Portfolio Performance cards' caption
        section = dashboard._section_khat(None, 0.75)
        assert ("k&#770; = realised in-between rate ÷ mean market-implied gap at the "
                "midpoints, over every time-series candidate entry at the band") in section
        assert "pB − pA" not in section

    def test_group_names_are_escaped_in_the_table(self):
        stat = dashboard._khat_finish({"n": 2, "events": 1, "rate": 0.5, "implied": 0.4,
                                       "k": 1.25})
        row = dashboard._khat_row_html("<b>Sports</b>", stat)
        assert "<b>" not in row and "&lt;b&gt;Sports&lt;/b&gt;" in row
        assert row.count("—") == 0 and "1.250" in row
        assert dashboard._khat_row_html("x", None).count("—") == 5

    def test_the_section_sits_after_the_interval_discount_section(self, monkeypatch, tmp_path):
        page = TestFilterPage()._page(monkeypatch, tmp_path)

        def heading(title: str) -> int:
            # Section titles, not the filter bar's summary, which names two of them
            return page.index(f"\n{title}\n</div>")
        assert (heading("Interval Discount (k) Calibration")
                < heading("Empirical k̂ by Category, Tag and Spread Band")
                < heading("Scenario Explorer"))

    def test_a_filter_that_cannot_be_built_leaves_the_section_a_notice(
            self, monkeypatch, tmp_path):
        def broken(*_a, **_k):
            raise ValueError("boom")
        monkeypatch.setattr(dashboard, "_filter_payload", broken)
        page = TestFilterPage()._page(monkeypatch, tmp_path)
        assert "Empirical k̂ by Category, Tag and Spread Band" in page
        assert "could not be built for this run" in page and 'id="khat-group"' not in page


def _page_kpi(page: str, key: str) -> tuple[str, str]:
    """(value, colour) of a keyed KPI card as Python rendered it."""
    m = re.search(rf'<div id="kpi-{key}" style="font-size:26px; font-weight:700; '
                  rf'color:([^;]+);">(.*?)</div>', page)
    assert m, key
    return m.group(2), m.group(1)


def _kd_table(page: str) -> tuple[list, list]:
    """The interval-discount section's per-k table as Python rendered it:
    (each row's cells, unescaped; each row's label weight)."""
    body = re.search(r'<tbody id="kd-rows">(.*?)</tbody>', page, re.S).group(1)
    rows = [[html.unescape(c) for c in re.findall(r"<td[^>]*>(.*?)</td>", tr)]
            for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", body)]
    weights = re.findall(r"font-weight:(\d+)'>", body)
    return rows, weights


class TestKhatCards:
    """The Portfolio Performance section's two k-hat cards: the empirical
    k-hat of the band and category or tag selected, and it minus the k
    selected — rendered for the primary view from the primary calibration
    itself, rewritten by the script from the k-hat breakdown's own groups."""

    def test_the_delta_is_red_when_khat_is_above_k(self):
        # p = 1 − k·(the mid spread): a k-hat above k means the sizer sized too big
        assert dashboard._khat_delta(0.9, 0.75) == ("+0.150", "#F44336")
        assert dashboard._khat_delta(0.6, 0.75) == ("-0.150", "#4CAF50")
        assert dashboard._khat_delta(0.75, 0.75) == ("+0.000", "#4CAF50")
        # Rounded to the digits shown BEFORE the sign and colour are read: a
        # difference a float's width below zero is not "-0.000", and one a
        # hair above it is not a red zero (sized too big, for a zero shown)
        assert dashboard._khat_delta(0.7999999999999999, 0.80) == ("+0.000", "#4CAF50")
        assert dashboard._khat_delta(0.45045, 0.45) == ("+0.000", "#4CAF50")
        assert dashboard._khat_delta(0.4512, 0.45) == ("+0.001", "#F44336")
        assert dashboard._khat_delta(0.4488, 0.45) == ("-0.001", "#4CAF50")
        for khat, k in ((None, 0.75), (0.6, None), (None, None)):
            assert dashboard._khat_delta(khat, k) == ("—", dashboard._KPI_DEFAULT_COLOR)

    def test_every_group_carries_its_delta_per_k_only_when_asked(self):
        cal = _flt_calibration(TestKhatBreakdown.FILED)
        plain = TestKhatBreakdown._band(cal)
        assert all("delta" not in st for st in plain["groups"].values())
        ks = (0.6, 0.75, None)
        band = dashboard._khat_band(cal, _FLT_SERIES, {"Other": 0, "Science": 1, "Sports": 2},
                                    {("Other", "General"): 0, ("Science", "General"): 1,
                                     ("Sports", "Hockey"): 2}, ks=ks)
        assert band["groups"].keys() == plain["groups"].keys()
        for key, st in band["groups"].items():
            assert st["delta"] == [list(dashboard._khat_delta(st["k"], k)) for k in ks]
            assert {f: v for f, v in st.items() if f != "delta"} == plain["groups"][key]
        # A hand-built calibration's pooled row gains it too
        pooled = IntervalCalibration(
            pooled=IntervalCalibrationBucket("POOLED", 0.0, 12, 0.25, 0.5, 0.5),
            buckets=[], excluded_premise_violations=0)
        assert dashboard._khat_band(pooled, None, {}, {}, ks=(0.75,))["groups"]["all"][
            "delta"] == [["-0.250", "#4CAF50"]]

    def test_the_caption_names_the_midpoint_basis(self):
        # DR-78: k-hat divides by the mean mid spread, the quantity the
        # forecast's k multiplies, and the caption says so in plain words
        caption = dashboard._KHAT_CARDS_CAPTION
        assert ("Empirical k&#770; = realised in-between rate ÷ mean market-implied gap "
                "at the midpoints (a market's midpoint is halfway between its YES ask and "
                "its YES bid), over every time-series candidate entry") in caption
        assert "pB − pA" not in caption

    def test_the_section_gains_the_cards_only_when_given(self):
        trades, curve = _flt_trades(), _flt_sweep().primary.equity_df
        plain = dashboard._section_performance(curve, trades, _FLT_START, 1000.0)
        cards = [("khat", "Empirical k̂", "3.333", "#2196F3"),
                 ("khat_delta", "k̂ − k (this selection)", "+2.583", "#F44336")]
        with_cards = dashboard._section_performance(curve, trades, _FLT_START, 1000.0,
                                                    extra_kpis=cards)
        added = "".join(dashboard._kpi(label, value, color, key=key)
                        for key, label, value, color in cards) + dashboard._KHAT_CARDS_CAPTION
        assert added in with_cards and with_cards.replace(added, "") == plain
        assert 'id="kpi-khat"' not in plain and "k&#770; − k" not in plain

    def test_the_cards_show_the_primary_band_against_the_page_s_k(self, monkeypatch, tmp_path):
        sweep = _flt_sweep()
        page = TestFilterPage()._page(monkeypatch, tmp_path, sweep)
        stat = dashboard._khat_stat(sweep.calibration.observations)
        assert _page_kpi(page, "khat") == (stat["cells"][4], "#2196F3")
        assert _page_kpi(page, "khat_delta") == dashboard._khat_delta(stat["k"], 0.75)
        assert "Empirical k̂</div>" in page and "k̂ − k (this selection)</div>" in page
        assert dashboard._KHAT_CARDS_CAPTION in page
        # The very figure the script shows for the primary view: the k-hat
        # breakdown's "all" group of the primary band, at the primary k
        data = TestFilterPage._data(page)
        pb, pk, _ = data["primary"]
        whole = data["khat"][pb]["groups"]["all"]
        assert (whole["cells"][4], *whole["delta"][pk]) == (
            _page_kpi(page, "khat")[0], *_page_kpi(page, "khat_delta"))
        assert data["styles"]["khat_card"] == "#2196F3"

    def test_the_cards_render_even_when_the_filter_cannot_be_built(self, monkeypatch, tmp_path):
        # Read off the primary calibration itself, never the filter's data
        def broken(*_a, **_k):
            raise ValueError("boom")
        monkeypatch.setattr(dashboard, "_filter_payload", broken)
        sweep = _flt_sweep()
        page = TestFilterPage()._page(monkeypatch, tmp_path, sweep)
        assert 'id="flt-unavailable"' in page
        stat = dashboard._khat_stat(sweep.calibration.observations)
        assert _page_kpi(page, "khat") == (stat["cells"][4], "#2196F3")
        assert _page_kpi(page, "khat_delta") == dashboard._khat_delta(stat["k"], 0.75)

    def test_a_tainted_corpus_carries_its_caveat_onto_the_card(self, monkeypatch, tmp_path):
        sweep = dataclasses.replace(_flt_sweep(), label_coverage=_coverage(2))
        page = TestFilterPage()._page(monkeypatch, tmp_path, sweep)
        # Suffixed, never replaced; recoloured, as the section's own card is
        assert "Empirical k̂ (see caveat in Interval Discount)</div>" in page
        assert _page_kpi(page, "khat")[1] == "#F44336"
        assert TestFilterPage._data(page)["styles"]["khat_card"] == "#F44336"
        kpis = dashboard._khat_kpis(None, 0, True)
        assert kpis == [("khat", "Empirical k̂ (see caveat in Interval Discount)", "—",
                         dashboard._KPI_DEFAULT_COLOR),
                        ("khat_delta", "k̂ − k (this selection)", "—",
                         dashboard._KPI_DEFAULT_COLOR)]

    def test_an_unsupported_override_prices_the_card_at_the_bar_s_k(
            self, monkeypatch, tmp_path, caplog):
        # interval_discount 0.62 against the sweep's primary k 0.75 (never in
        # production): the bar names 0.75 for the primary scenario, so the
        # card as rendered is priced at 0.75 — the figure the script writes
        # for that same view, and the interval-discount section's own card
        monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(dashboard.yf, "download",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
        sweep = _flt_sweep()
        with caplog.at_level(logging.WARNING):
            page = dashboard.generate_dashboard(
                sweep.primary.trades, sweep.primary.equity_df, _FLT_START, 1000.0,
                sweep=sweep, interval_discount=0.62,
                series_categories=_FLT_SERIES).read_text(encoding="utf-8")
        assert any("differs from the sweep's primary k" in r.getMessage()
                   for r in caplog.records)
        data = TestFilterPage._data(page)
        pb, pk, _ = data["primary"]
        stat = dashboard._khat_stat(sweep.calibration.observations)
        rendered = _page_kpi(page, "khat_delta")
        assert rendered == dashboard._khat_delta(stat["k"], 0.75) \
            == tuple(data["khat"][pb]["groups"]["all"]["delta"][pk]) \
            == _page_kpi(page, "kd_delta")
        assert rendered != dashboard._khat_delta(stat["k"], 0.62)
        # Away and back to the primary view: the script writes the same card
        snap = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-k", "0"], ["fire", "flt-k"], ["settle"],
            ["set", "flt-k", str(pk)], ["fire", "flt-k"], ["settle"], ["snap", "back"]],
            strict=True)["back"]
        assert (snap["text"]["kpi-khat_delta"], snap["colors"]["kpi-khat_delta"]) == rendered

    def test_no_calibration_shows_dashes_and_no_sweep_shows_no_cards(
            self, monkeypatch, tmp_path):
        sweep = dataclasses.replace(_flt_sweep(), calibration=None, calibrations_by_band={})
        page = TestFilterPage()._page(monkeypatch, tmp_path, sweep)
        assert _page_kpi(page, "khat") == ("—", dashboard._KPI_DEFAULT_COLOR)
        assert _page_kpi(page, "khat_delta") == ("—", dashboard._KPI_DEFAULT_COLOR)
        trades = _flt_trades()
        curve = backtester._build_equity_curve(trades, _FLT_START, 1000.0)
        monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
        bare = dashboard.generate_dashboard(trades, curve, _FLT_START, 1000.0) \
            .read_text(encoding="utf-8")
        assert 'id="kpi-khat"' not in bare and dashboard._KHAT_CARDS_CAPTION not in bare

    # ─── The script over the cards ───────────────────────────────────────────

    def test_the_cards_follow_the_band_category_and_k(self, monkeypatch, tmp_path):
        page = _kc_page(monkeypatch, tmp_path)
        data = TestFilterPage._data(page)
        sports = data["categories"].index("Sports")
        snaps = _run_script(tmp_path, page, [
            ["wait"], _tick_category(str(sports)), ["settle"],
            ["snap", "cat"],
            ["set", "flt-k", "0"], ["fire", "flt-k"], ["settle"], ["snap", "k"],
            ["set", "flt-band", "1"], ["fire", "flt-band"], ["settle"], ["snap", "band"],
            ["set", "flt-band", "0"], ["fire", "flt-band"], ["settle"],
            _tick_category(""),
            ["set", "flt-k", "1"], ["fire", "flt-k"], ["settle"], ["snap", "back"]],
            strict=True)
        group = data["khat"][0]["groups"][f"c{sports}"]
        card = data["styles"]["khat_card"]

        def cards(snap):
            return ((snap["text"]["kpi-khat"], snap["colors"]["kpi-khat"]),
                    (snap["text"]["kpi-khat_delta"], snap["colors"]["kpi-khat_delta"]))
        # The selected category's own k-hat, against the k shown
        assert cards(snaps["cat"]) == ((group["cells"][4], card), tuple(group["delta"][1]))
        assert cards(snaps["k"]) == ((group["cells"][4], card), tuple(group["delta"][0]))
        # A band with no k-hat population: blanks in the default colour
        blank = (data["khat_blank"][4], data["styles"]["kpi_default"])
        assert data["khat"][1] is None and cards(snaps["band"]) == (blank, blank)
        # Back at the primary view: exactly the cards Python rendered
        assert cards(snaps["back"]) == (_page_kpi(page, "khat"), _page_kpi(page, "khat_delta"))

    def test_the_script_keeps_the_caveat_colour_on_a_tainted_corpus(
            self, monkeypatch, tmp_path):
        # DR-66b: on a strike-blind corpus the k-hat card is recoloured, and a
        # filter change must not turn it back to the healthy blue
        page = _kc_page(monkeypatch, tmp_path,
                        dataclasses.replace(_kc_sweep(), label_coverage=_coverage(2)))
        data = TestFilterPage._data(page)
        assert data["styles"]["khat_card"] == "#F44336" == _page_kpi(page, "khat")[1]
        sports = data["categories"].index("Sports")
        snaps = _run_script(tmp_path, page, [
            ["wait"], _tick_category(str(sports)), ["settle"],
            ["snap", "cat"],
            ["set", "flt-k", "0"], ["fire", "flt-k"], ["settle"], ["snap", "k"],
            ["set", "flt-band", "1"], ["fire", "flt-band"], ["settle"], ["snap", "band"]],
            strict=True)
        group = data["khat"][0]["groups"][f"c{sports}"]
        for name in ("cat", "k"):
            assert (snaps[name]["text"]["kpi-khat"], snaps[name]["colors"]["kpi-khat"]) \
                == (group["cells"][4], "#F44336")
        # A band with no k-hat population: the blank, in the default colour
        assert data["khat"][1] is None
        assert (snaps["band"]["text"]["kpi-khat"], snaps["band"]["colors"]["kpi-khat"]) \
            == (data["khat_blank"][4], data["styles"]["kpi_default"])

    def test_a_category_with_no_khat_entry_shows_blanks(self, monkeypatch, tmp_path):
        # Commodities has trades on the _flt page but no k-hat observation
        page = TestFilterPage()._page(monkeypatch, tmp_path)
        data = TestFilterPage._data(page)
        commodities = str(data["categories"].index("Commodities"))
        snap = _run_script(tmp_path, page, [
            ["wait"], _tick_category(commodities), ["settle"],
            ["snap", "s"]])["s"]
        blank = data["khat_blank"][4]
        assert snap["text"]["kpi-khat"] == snap["text"]["kpi-khat_delta"] == blank
        assert snap["colors"]["kpi-khat"] == data["styles"]["kpi_default"]


class TestIntervalDiscountFollowsTheBar:
    """The interval-discount section at the primary spread band, for the k and
    size cap the page-wide filter bar shows: its data comes from the one walk
    (_KdVisitor), the page renders it for the primary scenario, and the
    script redraws it for every other k and cap (renderKd)."""

    def test_the_walk_collects_every_k_and_cap_of_the_primary_band(self):
        sweep = _kc_sweep()
        _, source, base, _ = _flt_payload(sweep)
        kd = base["kd"]
        points = sweep.cap_sweep.points
        axis = pd.DatetimeIndex(pd.to_datetime(base["dates"]))
        # rows[cap][k], curves[k][cap]: every (k, cap) of the primary band
        for ci, cap in enumerate(source.caps):
            for ki, k in enumerate(source.ks):
                point = points[(_KC_B0, k, cap)]
                assert kd["rows"][ci][ki] == dashboard._kd_cells(point, ki == 1)
                assert kd["curves"][ki][ci] == dashboard._kd_curve(point, axis)
        assert kd["rows"][1][1][0] == "k = 0.75 (primary)"
        assert kd["titles"][0][0] == "Equity Curve at interval discount k = 0.60, 5% cap per trade"
        assert kd["titles"][2][1] == "Equity Curve at interval discount k = 0.75, no per-trade cap"
        assert kd["k_text"] == ["0.600", "0.750"]
        pooled = sweep.calibration.pooled.empirical_k
        assert kd["delta"] == [list(dashboard._khat_delta(pooled, k)) for k in (0.6, 0.75)]
        assert kd["primary"] == [1, 1] and "dates" not in kd
        # The same one walk the chunks came from: every cell read once
        assert sweep.cap_sweep.reads == [(_KC_B0, 0.75), (_KC_B0, 0.6), (_KC_B1, 0.6),
                                         (_KC_B1, 0.75)]

    def test_the_page_renders_the_primary_scenario_from_the_walk(self, monkeypatch, tmp_path):
        sweep = _kc_sweep()
        page = _kc_page(monkeypatch, tmp_path, sweep)
        kd = TestFilterPage._data(page)["kd"]
        rows, weights = _kd_table(page)
        assert rows == kd["rows"][1] and weights == ["400", "700"]
        assert _page_kpi(page, "kd_k") == ("0.750", "#2196F3")
        assert _page_kpi(page, "kd_delta") == tuple(kd["delta"][1])
        section = page[page.index("\nInterval Discount (k) Calibration\n"):]
        data, layout = _nth_figure(section, 0)
        assert layout["title"]["text"] == kd["titles"][1][1]
        assert data[0]["y"] == pytest.approx(
            _expand(kd["curves"][1][1], len(TestFilterPage._data(page)["dates"])))
        assert "updatemenus" not in section[:section.index("Empirical k̂ by")]
        # A section that follows the bar carries neither static notice, and
        # the summary line says the bar reaches it — and, the _kc size-cap
        # sweep having run without the band sweep, that there is no explorer
        # grid for it to reach
        for text in ("no_bar", "failed"):
            assert html.escape(dashboard._KD_TEXT[text]) not in page
        assert html.escape(dashboard._BAR_REACH_NO_EXPLORER) in page
        assert html.escape(dashboard._BAR_REACH_STATIC) not in page
        assert html.escape(dashboard._BAR_REACH) not in page

    def test_a_filter_that_cannot_be_built_leaves_the_section_static(
            self, monkeypatch, tmp_path):
        def broken(*_a, **_k):
            raise ValueError("boom")
        monkeypatch.setattr(dashboard, "_filter_payload", broken)
        sweep = _kc_sweep()
        page = _kc_page(monkeypatch, tmp_path, sweep)
        assert html.escape(dashboard._KD_TEXT["no_bar"]) in page
        # The sweep's own points at the run's own cap
        rows, weights = _kd_table(page)
        assert rows == [dashboard._kd_cells(p, p is sweep.primary) for p in sweep.points]
        assert weights == ["700", "400"]

    def test_bar_false_renders_statically_with_its_notice(self):
        sweep = _kc_sweep()
        _, _, base, _ = _flt_payload(sweep)
        kd = dict(base["kd"], dates=base["dates"])
        live = _section_interval_discount(sweep, kd)
        static = _section_interval_discount(sweep, kd, bar=False)
        notice = html.escape(dashboard._KD_TEXT["no_bar"])
        assert notice in static and notice not in live
        assert static.replace(f"<p style='font-family:sans-serif;font-size:13px;"
                              f"color:#616161;'>{notice}</p>", "") \
            == _section_interval_discount(sweep)
        # The walk's data is ignored: the static table is the sweep's points
        assert _kd_table(live)[0] == kd["rows"][1] != _kd_table(static)[0]

    def test_a_run_without_a_band_follows_the_bar_at_its_lone_band(
            self, monkeypatch, tmp_path):
        # A hand-built multi-k sweep with no band: its one "not recorded" band
        # is its primary band, so the section follows the bar's k there as it
        # does on a banded run — no notice, and a summary that says so
        trades = _flt_trades()
        points = [SweepPoint(k=k, trades=trades[:n],
                             equity_df=backtester._build_equity_curve(trades[:n], _FLT_START,
                                                                      1000.0))
                  for k, n in ((0.6, 1), (0.75, 5), (0.9, 3))]
        sweep = BacktestSweep(primary=points[1], points=points, calibration=None)
        page = TestFilterPage()._page(monkeypatch, tmp_path, sweep)
        data = TestFilterPage._data(page)
        kd = data["kd"]
        assert kd is not None and len(data["ks"]) == 3 and kd["primary"] == [1, 0]
        # As rendered: the sweep's points, the run's own k in bold
        rows, weights = _kd_table(page)
        assert rows == kd["rows"][0] == [dashboard._kd_cells(p, i == 1)
                                         for i, p in enumerate(points)]
        assert weights == ["400", "700", "400"]
        for text in ("no_bar", "failed"):
            assert html.escape(dashboard._KD_TEXT[text]) not in page
        # No scenarios: no explorer grid for the bar to reach
        assert html.escape(dashboard._BAR_REACH_NO_EXPLORER) in page
        # The script moves it to the bar's k
        snap = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-k", "0"], ["fire", "flt-k"], ["settle"], ["snap", "s"]],
            strict=True)["s"]
        assert snap["text"]["kpi-kd_k"] == kd["k_text"][0] == "0.600"
        assert _last_react(snap, "kd-equity")["layout"]["title"]["text"] == kd["titles"][0][0]
        assert snap["rows"]["kd-rows"] == kd["rows"][0]
        assert [w[0] for w in snap["weights"]["kd-rows"]] == ["700", "400", "400"]
        assert snap["text"]["hdr-trades"] == "1"
        assert snap["text"]["flt-summary"].endswith(dashboard._BAR_REACH_NO_EXPLORER)

    def test_a_section_whose_data_fails_says_so_and_costs_nothing_else(
            self, monkeypatch, tmp_path, caplog):
        real, calls = dashboard._kd_curve, []

        def first_fails(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise ValueError("boom")
            return real(*args, **kwargs)
        monkeypatch.setattr(dashboard, "_kd_curve", first_fails)
        with caplog.at_level(logging.WARNING):
            page = _kc_page(monkeypatch, tmp_path)
        warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert sum("Interval Discount section's k and size-cap figures" in m
                   for m in warned) == 1
        assert html.escape(dashboard._KD_TEXT["failed"]) in page
        data = TestFilterPage._data(page)
        assert data["kd"] is None and 'id="flt-bar"' in page and len(data["caps"]) == 3
        # The summary line does not claim a reach the section lost — as
        # rendered, and as the script fills it for another scenario
        assert data["text"]["unfiltered"] == f" {dashboard._BAR_REACH_STATIC}"
        assert html.escape(dashboard._BAR_REACH_STATIC) in page
        assert html.escape(dashboard._BAR_REACH) not in page
        snap = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-k", "0"], ["fire", "flt-k"], ["settle"], ["snap", "s"]],
            strict=True)["s"]
        assert snap["text"]["flt-summary"].endswith(dashboard._BAR_REACH_STATIC)
        # ...and the section stays as Python drew it
        assert "kd-equity" not in {r["id"] for r in snap["reacts"]}
        assert "kpi-kd_k" not in snap["text"] and "kd-rows" not in snap["rows"]

    def test_a_page_without_a_sweep_does_not_claim_the_section_follows_the_bar(
            self, monkeypatch, tmp_path):
        # No sweep: the section is its placeholder, so the summary line says
        # the bar does not reach it
        monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(dashboard.yf, "download",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
        trades = _flt_trades()
        curve = backtester._build_equity_curve(trades, _FLT_START, 1000.0)
        page = dashboard.generate_dashboard(trades, curve, _FLT_START, 1000.0,
                                            series_categories=_FLT_SERIES) \
            .read_text(encoding="utf-8")
        assert "No interval-discount sweep for this run." in page and 'id="flt-bar"' in page
        data = TestFilterPage._data(page)
        assert data["kd"] is None
        assert data["text"]["unfiltered"] == f" {dashboard._BAR_REACH_STATIC}"
        assert html.escape(dashboard._BAR_REACH_STATIC) in page
        assert html.escape(dashboard._BAR_REACH) not in page

    # ─── The script over the section ─────────────────────────────────────────

    def test_the_section_follows_the_k_and_the_cap(self, monkeypatch, tmp_path):
        page = _kc_page(monkeypatch, tmp_path)
        data = TestFilterPage._data(page)
        kd, n = data["kd"], len(data["dates"])
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-k", "0"], ["fire", "flt-k"], ["settle"], ["snap", "k"],
            ["set", "flt-cap", "0"], ["fire", "flt-cap"], ["settle"], ["snap", "cap"]],
            strict=True)
        for name, ki, ci in (("k", 0, 1), ("cap", 0, 0)):
            snap = snaps[name]
            assert snap["text"]["kpi-kd_k"] == kd["k_text"][ki]
            assert [snap["text"]["kpi-kd_delta"], snap["colors"]["kpi-kd_delta"]] \
                == kd["delta"][ki]
            react = _last_react(snap, "kd-equity")
            assert react["layout"]["title"]["text"] == kd["titles"][ci][ki]
            assert react["data"][0]["x"] == data["dates"]
            assert react["data"][0]["y"] == pytest.approx(_expand(kd["curves"][ki][ci], n))
            assert react["data"][0]["name"] == "Portfolio value"
            # That cap's rows, the k shown in bold
            assert snap["rows"]["kd-rows"] == kd["rows"][ci]
            assert [w[0] for w in snap["weights"]["kd-rows"]] == ["700", "400"]
        # The caps trade different sizes, so the tables differ
        assert kd["rows"][0] != kd["rows"][1]

    def test_a_band_change_redraws_the_section_as_python_drew_it(self, monkeypatch, tmp_path):
        # The section follows only the k and cap: another band shows the same
        # (primary band) scenario, drawn exactly as Python rendered it
        page = _kc_page(monkeypatch, tmp_path)
        snap = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-band", "1"], ["fire", "flt-band"], ["settle"],
            ["snap", "band"]], strict=True)["band"]
        python = _page_elements(page)["charts"]["kd-equity"]
        react = _last_react(snap, "kd-equity")
        assert react["layout"] == python["layout"]
        # Exactly: both expand the same change points (_expand_sparse / expand)
        assert react["data"][0]["y"] == python["data"][0]["y"]
        assert react["data"][0]["x"] == python["data"][0]["x"]
        rows, weights = _kd_table(page)
        assert snap["rows"]["kd-rows"] == rows
        assert [w[0] for w in snap["weights"]["kd-rows"]] == weights
        for key in ("kd_k", "kd_delta"):
            assert (snap["text"][f"kpi-{key}"], snap["colors"].get(f"kpi-{key}",
                                                                    "#2196F3")) \
                == _page_kpi(page, key)

    def test_a_k_the_primary_band_never_simulated_says_so(self, monkeypatch, tmp_path):
        # The _flt primary band was swept at k 0.60 and 0.75, never 1.00
        page = TestFilterPage()._page(monkeypatch, tmp_path)
        data = TestFilterPage._data(page)
        kd = data["kd"]
        snap = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-k", "2"], ["fire", "flt-k"], ["settle"], ["snap", "s"]],
            strict=True)["s"]
        react = _last_react(snap, "kd-equity")
        assert kd["curves"][2][0] is None and react["data"][0]["y"] == []
        assert react["layout"]["title"]["text"] == kd["titles"][0][2] == (
            "Equity Curve at interval discount k = 1.00, 20% cap per trade: not simulated "
            "by this run")
        assert [r[0] for r in snap["rows"]["kd-rows"]] == ["k = 0.60", "k = 0.75 (primary)"]
        assert [w[0] for w in snap["weights"]["kd-rows"]] == ["400", "400"]
        assert snap["text"]["kpi-kd_k"] == "1.000"


# ─── The scenario explorer: size-cap axis, Sharpe and k-hat − k, bar sync ────

_EX_CAPS = (0.05, 0.2, 1.0)
_EX_METRIC_LABELS = [
    "Mean per trade (equal stake)", "Median per trade (equal stake)", "Total return",
    "Sharpe ratio (365-day base)", "H1 return", "H2 return", "Trade count",
    "Empirical k̂ (pooled per band)", "k̂ − k (pooled per band, minus the column's k)"]


class _ExplorerCapSweep(_FakeCapSweep):
    """
    _FakeCapSweep with the band sweep's checks, as a production CapSweep has
    them whenever the band sweep ran: every scenario carries the "all" and
    "time_series" populations (the primary band's also "ladder"), and the
    same-title population has a point per cap. `points` maps (band, k, cap)
    -> {population: SweepPoint}; `same_title` maps cap -> SweepPoint.
    """

    checks = True

    def __init__(self, points: dict, *, same_title: dict | None = None, **kwargs):
        super().__init__(points, **kwargs)
        self._same_title = same_title or {}

    def cell(self, band, k):
        self.reads.append((band, k))
        if (band, k) == self.raise_on:
            raise RuntimeError("the cap simulation failed")
        return {cap: dict(self.points[(band, k, cap)]) for cap in self.caps
                if (band, k, cap) in self.points}

    def same_title(self):
        return dict(self._same_title)

    def entry_events(self):
        return {(t.event_ticker, t.category) for pops in self.points.values()
                for p in pops.values() for t in p.trades}


def _ex_sweep(*, raise_on=None) -> BacktestSweep:
    """
    A band sweep with a size-cap sweep over 2 bands x 2 ks x 3 caps (5%, 20%
    — the run's own — and no cap), primary (0-1, k 0.75, 20%).

    The primary band loses at 5% (a losing list) and wins at 20% and no cap
    (the full list); at no cap and k 0.75 the full list shares its TRADES
    with 20% but not its curve (booked on a 1,100 balance), so a row cached
    on the trades alone would be the 20% row. Band 0.3-0.6 traded nothing at
    k 0.60 (a flat curve: Sharpe 0.0, which the heatmap must not show) and
    the same winning list at every cap at k 0.75. Same-title: one trade at
    5%, two at 20% and no cap.
    """
    def resized(t, n):
        return _kc_resized(t, n)

    full = _kc_trades(5)                               # +8, +4, -3
    losing = [resized(_ftrade("KXNHLHART-27", "time_series", date(2026, 1, 12),
                              date(2026, 1, 20), -8.0, ladder=True), 5),
              resized(_ftrade("KXOTHER-1", "time_series", date(2026, 1, 13),
                              date(2026, 1, 16), -4.0), 5)]
    other = _kc_trades(5, count=2)                     # +8, +4
    empty: list = []
    curves, halves = {}, {id(full): HalfSplit(0.02, 0.03, 1, 2),
                          id(losing): HalfSplit(-0.01, -0.02, 1, 1),
                          id(other): HalfSplit(0.01, 0.0, 1, 1),
                          id(empty): HalfSplit(0.0, 0.0, 0, 0)}

    def curve(trades):
        if id(trades) not in curves:
            curves[id(trades)] = backtester._build_equity_curve(trades, _FLT_START, 1000.0)
        return curves[id(trades)]

    def pops(band, k, cap, trades, eq=None, ladder=False):
        point = SweepPoint(k=k, trades=trades, equity_df=curve(trades) if eq is None else eq,
                           spread_band=band, size_cap=cap, halves=halves.get(id(trades)),
                           ex_top_event=("KXNHLHART-27", 0.01) if trades else None)
        out = {"all": point, "time_series": dataclasses.replace(point, population="time_series")}
        if ladder:
            out["ladder"] = dataclasses.replace(point, population="ladder", halves=None,
                                                ex_top_event=None)
        return out

    rich = backtester._build_equity_curve(full, _FLT_START, 1100.0)
    points = {}
    for k in (0.6, 0.75):
        points[(_KC_B0, k, 0.05)] = pops(_KC_B0, k, 0.05, losing, ladder=True)
        points[(_KC_B0, k, 0.2)] = pops(_KC_B0, k, 0.2, full, ladder=True)
        points[(_KC_B0, k, 1.0)] = pops(_KC_B0, k, 1.0, full, ladder=True,
                                        eq=rich if k == 0.75 else None)
        for cap in _EX_CAPS:
            points[(_KC_B1, k, cap)] = pops(_KC_B1, k, cap, empty if k == 0.6 else other)
    st_lists = {0.05: [_flt_trades()[0]], 0.2: _flt_trades()[:2], 1.0: _flt_trades()[:2]}
    st = {cap: SweepPoint(k=0.75, trades=listed, equity_df=curve(listed),
                          population="same_title", size_cap=cap)
          for cap, listed in st_lists.items()}
    eager = [p for (_, _, cap), by_pop in points.items() if cap == 0.2 for p in by_pop.values()]
    primary = points[(_KC_B0, 0.75, 0.2)]["all"]
    cal = _flt_calibration([("KXNHLHART-27", "Other"), ("KXOTHER-1", "Other")])
    return BacktestSweep(
        primary=primary, points=[points[(_KC_B0, k, 0.2)]["all"] for k in (0.6, 0.75)],
        calibration=cal, label_coverage=_scn_coverage(), scenarios=eager,
        same_title_point=st[0.2], calibrations_by_band={_KC_B0: cal, _KC_B1: None},
        same_event_ladders=True, split_date=date(2026, 1, 13),
        cap_sweep=_ExplorerCapSweep(points, same_title=st, raise_on=raise_on, caps=_EX_CAPS))


def _ex_sweep_tiers() -> BacktestSweep:
    """
    _ex_sweep() plus a tier-floors-off family for its one binding band
    (_KC_B0, floor 0), simulated at the run's own 20% cap only (as the
    backtester's family is): at each k an "all" and a "time_series" point
    trading one two-trade list no tier-on cell trades (a winning then a
    losing trade), with its tier-off calibration. _KC_B1 (floor 0.3) sits at
    or above both tiers and was never simulated again. So the explorer has
    an off block at the 20% cap alone.
    """
    sweep = _ex_sweep()
    off_trades = [_kc_resized(_ftrade("KXRAIN-1", "time_series", date(2026, 1, 12),
                                      date(2026, 1, 15), 6.0), 5),
                  _kc_resized(_ftrade("KXOTHER-1", "time_series", date(2026, 1, 13),
                                      date(2026, 1, 16), -2.0), 5)]
    curve = backtester._build_equity_curve(off_trades, _FLT_START, 1000.0)
    off = []
    for k in (0.6, 0.75):
        point = SweepPoint(k=k, trades=off_trades, equity_df=curve, spread_band=_KC_B0,
                           size_cap=0.2, tier_floors=False,
                           halves=HalfSplit(0.004, -0.002, 1, 1),
                           ex_top_event=("KXRAIN-1", -0.002))
        off += [point, dataclasses.replace(point, population="time_series")]
    cal = _flt_calibration([("KXOTHER-1", "Other")])
    return dataclasses.replace(sweep, tier_off_scenarios=off,
                               tier_off_calibrations_by_band={_KC_B0: cal})


def _ex_section(page: str) -> str:
    """The rendered page's Scenario Explorer section (to the next section)."""
    start = page.index("\nScenario Explorer\n")
    return page[start:page.index("Trade-Level Diagnostics", start)]


_SCN_SELECTS = ("scn-band-select", "scn-k-select", "scn-cap-select")


def _scn_values(snap: dict) -> list[str]:
    """The explorer's band, k and size-cap selects' values in a snapshot."""
    return [snap["selects"][s]["value"] for s in _SCN_SELECTS]


class TestExplorerCapAndMetrics:
    """The scenario explorer walks the page's grid — every band x k x size cap
    — beside the filter bar (one walk), ships a block per cap, and gains the
    Sharpe and k-hat − k metrics."""

    def test_the_explorer_rides_the_page_s_one_walk(self, monkeypatch, tmp_path):
        sweep = _ex_sweep()
        page = _kc_page(monkeypatch, tmp_path, sweep)
        section = _ex_section(page)
        blocks = _scn_blocks(section)
        # Every cap's block, the base block first; every cell read once for
        # the whole page (chunks, interval-discount section and explorer)
        assert list(blocks) == ["scn-data", "scn-cap-0", "scn-cap-1", "scn-cap-2"]
        assert sweep.cap_sweep.reads == [(_KC_B0, 0.75), (_KC_B0, 0.6), (_KC_B1, 0.6),
                                         (_KC_B1, 0.75)]
        data = blocks["scn-data"]
        assert data["caps"] == list(_EX_CAPS) and data["primary"] == [0, 1, 1]
        # The bar's own labels, so its calls match
        bar = TestFilterPage._data(page)
        assert data["band_labels"] == [b["label"] for b in bar["bands"]]
        assert data["k_labels"] == [k["label"] for k in bar["ks"]]
        assert data["cap_labels"] == [c["label"] for c in bar["caps"]]
        assert _options(section, "scn-cap-select") == [
            ("0", False, "5%"), ("1", True, "20% (primary)"), ("2", False, "off (full Kelly)")]
        # Each cap's own simulation: the primary cell trades 2 contracts' worth
        # of losers at 5%, the full list at 20% and no cap
        ts = data["populations"].index("time_series")
        rows = [blocks[f"scn-cap-{ci}"]["cells"][0][1][ts] for ci in range(3)]
        assert [r["trades"] for r in rows] == [2, 3, 3]
        assert rows[0]["total_return"] < 0 < rows[1]["total_return"]
        # A curve shared with no other cap: its own row (a cache on the
        # trades alone would repeat the 20% row here)
        assert rows[2]["final_balance"] == pytest.approx(rows[1]["final_balance"] + 100.0)
        assert rows[2]["total_return"] != pytest.approx(rows[1]["total_return"])
        matrices = [blocks[f"scn-cap-{ci}"]["matrices"] for ci in range(3)]
        assert matrices[2]["total_return"][0][1] == pytest.approx(rows[2]["total_return"])
        assert matrices[1]["total_return"][0][1] == pytest.approx(rows[1]["total_return"])
        # ... and the population is part of what a row is: the headline row
        # carries its curve, the "all" row sharing its trades does not
        assert "equity" in rows[1] and "equity" not in blocks["scn-cap-1"]["cells"][0][1][0]
        # The page as rendered shows the primary cap
        heat, layout = _first_figure(section)
        assert heat[0]["z"] == matrices[1]["mean_per_trade"]
        assert layout["title"]["text"] == blocks["scn-cap-1"]["titles"][0]

    def test_nine_metrics_in_order_each_on_its_block(self, monkeypatch, tmp_path):
        section = _ex_section(_kc_page(monkeypatch, tmp_path, _ex_sweep()))
        data, cap = _scn_data(section), _scn_cap(section)
        assert [m["label"] for m in data["metrics"]] == _EX_METRIC_LABELS
        assert [html.unescape(o[2]) for o in _options(section, "scn-metric")] \
            == _EX_METRIC_LABELS
        keys = [m["key"] for m in data["metrics"]]
        # The figures of the trades ship per cap, k-hat once
        assert set(cap["matrices"]) == set(keys[:7])
        assert set(data["matrices"]) == {"empirical_k", "khat_minus_k", "empirical_k_n"}
        assert len(cap["titles"]) == 9
        assert cap["titles"][3] == ("Sharpe ratio (365-day base) by spread band x k, 20% cap "
                                    f"per trade — {_TS_LABEL}")
        assert cap["titles"][8] == ("k̂ − k (pooled per band, minus the column's k) by spread "
                                    f"band x k — {_TS_LABEL}")
        assert "updatemenus" not in _first_figure(section)[1]

    def test_sharpe_is_blank_on_a_cell_with_no_trade(self, monkeypatch, tmp_path):
        section = _ex_section(_kc_page(monkeypatch, tmp_path, _ex_sweep()))
        data, cap = _scn_data(section), _scn_cap(section)
        ts = data["populations"].index("time_series")
        sharpe = cap["matrices"]["sharpe"]
        # Band 0.3-0.6 at k 0.60 traded nothing: its flat curve's 0.0 is not
        # a Sharpe ratio — the matrix blanks it ...
        assert cap["matrices"]["trades"][1][0] == 0 and sharpe[1][0] is None
        # ... while the KPI row keeps _point_kpis' own figure
        assert cap["cells"][1][0][ts]["sharpe"] == 0.0
        # A cell that traded shows its own curve's Sharpe
        assert sharpe[0][1] == pytest.approx(cap["cells"][0][1][ts]["sharpe"])
        assert sharpe[0][1] not in (None, 0.0)
        spec = data["metrics"][3]
        assert (spec["key"], spec["zmid"], spec["custom"]) == ("sharpe", 0, "trades")
        assert "Sharpe=%{z:.2f}" in spec["hover"]

    def test_khat_minus_k_is_the_cards_figure_and_red_when_positive(
            self, monkeypatch, tmp_path):
        sweep = _ex_sweep()
        section = _ex_section(_kc_page(monkeypatch, tmp_path, sweep))
        data = _scn_data(section)
        khat = sweep.calibration.pooled.empirical_k
        delta = data["matrices"]["khat_minus_k"]
        # Pooled k-hat of the band minus the column's k, rounded as the cards
        # round it: its text is the card's
        assert delta[0] == [dashboard._khat_delta_value(khat, k) for k in (0.6, 0.75)]
        assert [f"{v:+.3f}" for v in delta[0]] == [dashboard._khat_delta(khat, k)[0]
                                                   for k in (0.6, 0.75)]
        # A band without a calibration has none
        assert delta[1] == [None, None]
        assert data["matrices"]["empirical_k_n"][0] == [sweep.calibration.pooled.n] * 2
        spec = data["metrics"][8]
        assert (spec["key"], spec["zmid"], spec["custom"]) == ("khat_minus_k", 0,
                                                               "empirical_k_n")
        # Positive (the sizer sized too big) is the red end, as on the cards;
        # the return metrics put the blue end on top
        import plotly.graph_objects as go
        assert spec["colorscale"] == [list(c) for c in go.Heatmap(colorscale="RdBu_r").colorscale]
        red, blue = "rgb(103,0,31)", "rgb(5,48,97)"
        assert spec["colorscale"][-1][1] == red and spec["colorscale"][0][1] == blue
        assert data["metrics"][0]["colorscale"][-1][1] == blue
        assert "k̂ − k=%{z:+.3f}" in spec["hover"]
        # The k-hat view shows the same fact (k-hat above the run's k) on the
        # same scale, centred on that k: one cell, one colour, in both views
        khat_spec = data["metrics"][7]
        assert (khat_spec["key"], khat_spec["zmid"]) == ("empirical_k", sweep.primary.k)
        assert khat_spec["colorscale"] == spec["colorscale"]
        # (the primary cell: k-hat above k, so above both centres — the red end)
        cell_khat, cell_delta = data["matrices"]["empirical_k"][0][1], delta[0][1]
        assert cell_khat > khat_spec["zmid"] and cell_delta > spec["zmid"]

    def test_the_banner_and_same_title_row_are_each_cap_s_own(self, monkeypatch, tmp_path):
        section = _ex_section(_kc_page(monkeypatch, tmp_path, _ex_sweep()))
        blocks = _scn_blocks(section)
        caps = [blocks[f"scn-cap-{ci}"] for ci in range(3)]
        # 5%: the primary band lost at both ks, 0.3-0.6 traded nothing at
        # 0.60 and won at 0.75 — 1 of 4 positive; 20% and no cap: 3 of 4
        for cap, label, share in zip(caps, ("5%", "20%", "off (full Kelly)"),
                                     ("25.0%", "75.0%", "75.0%"), strict=True):
            assert f"at size cap {label}" in cap["banner"]
            assert "<b>4 band x k cells computed</b>" in cap["banner"]
            assert f"{share} of the cells with a measurable return" in cap["banner"]
        # 4 cells x 2 populations + the primary band's 2 ladder points
        assert "(10 scenario points" in caps[0]["banner"]
        # The page renders the primary cap's banner
        assert f'<div id="scn-banner">{caps[1]["banner"]}</div>' in section
        assert [cap["same_title"]["trades"] for cap in caps] == [1, 2, 2]

    def test_without_the_band_sweep_the_explorer_is_its_placeholder(
            self, monkeypatch, tmp_path):
        # The _kc size-cap sweep ran without the band sweep (checks False):
        # whatever the walk read, the explorer is today's placeholder, with
        # no grid, blocks or script — and the bar does not claim to reach it
        page = _kc_page(monkeypatch, tmp_path)
        section = _ex_section(page)
        assert ("No scenarios were computed: band sweep off (--no-band-sweep / "
                "band_sweep=False).") in section
        assert 'id="scn-data"' not in page and "function unpack(el)" not in page
        assert html.escape(dashboard._BAR_REACH_NO_EXPLORER) in page

    def test_an_explorer_that_cannot_be_built_from_the_walk_uses_the_eager_points(
            self, monkeypatch, tmp_path, caplog):
        real, calls = dashboard._explorer_curve, []

        def first_fails(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise ValueError("boom")
            return real(*args, **kwargs)
        monkeypatch.setattr(dashboard, "_explorer_curve", first_fails)
        with caplog.at_level(logging.WARNING):
            page = _kc_page(monkeypatch, tmp_path, _ex_sweep())
        warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert sum("Scenario Explorer's figures could not be built" in m for m in warned) == 1
        section = _ex_section(page)
        # The sweep's own points at the run's own cap; the bar keeps every cap
        assert _options(section, "scn-cap-select") == [("0", True, "20% (primary)")]
        assert list(_scn_blocks(section)) == ["scn-data", "scn-cap-0"]
        assert len(TestFilterPage._data(page)["caps"]) == 3
        # ... so the page says the explorer follows the bar's band and k but
        # not its size cap — in the bar's summary and in the section itself
        # (the explorer has no Tier floors select of its own, so the line
        # says it does not follow that choice either)
        own_cap = dashboard._BAR_REACH_NO_EXPLORER_TIERS[(True, False)]
        assert html.escape(own_cap) in page
        assert html.escape(dashboard._BAR_REACH_NO_EXPLORER_TIERS[(True, True)]) not in page
        assert html.escape(dashboard._BAR_REACH_OWN_CAP) not in page
        assert html.escape(dashboard._BAR_REACH) not in page
        assert dashboard._EXPLORER_OWN_CAP_HTML in section
        assert section.index('id="scn-own-cap"') < section.index('id="scn-banner"')
        # The script agrees: the bar at 5% names the explorer's reach so,
        # and the explorer keeps the one cap it has
        snap = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-cap", "0"], ["fire", "flt-cap"], ["settle"], ["snap", "s"]],
            strict=True, explorer=True)["s"]
        assert "5% cap per trade" in snap["text"]["flt-summary"]
        assert snap["text"]["flt-summary"].endswith(own_cap)
        assert snap["selects"]["scn-cap-select"]["value"] == "0"

    def test_an_explorer_rebuilt_without_a_size_cap_sweep_claims_its_whole_reach(
            self, monkeypatch, tmp_path, caplog):
        # With no size-cap sweep the rebuilt explorer's one cap IS the bar's
        # one cap: nothing was lost, and nothing is said to have been
        real, calls = dashboard._explorer_curve, []

        def first_fails(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise ValueError("boom")
            return real(*args, **kwargs)
        monkeypatch.setattr(dashboard, "_explorer_curve", first_fails)
        sweep = dataclasses.replace(_ex_sweep(), cap_sweep=None)
        with caplog.at_level(logging.WARNING):
            page = _kc_page(monkeypatch, tmp_path, sweep)
        assert len(calls) > 1                  # the walk's visitor failed, then rebuilt
        section = _ex_section(page)
        assert _options(section, "scn-cap-select") == [("0", True, "20% (primary)")]
        assert [c["label"] for c in TestFilterPage._data(page)["caps"]] == ["20%"]
        assert html.escape(dashboard._BAR_REACH_NO_EXPLORER_TIERS[(True, True)]) in page
        assert 'id="scn-own-cap"' not in page

    def test_an_explorer_that_cannot_be_built_at_all_is_a_notice(
            self, monkeypatch, tmp_path, caplog):
        def broken(*_a, **_k):
            raise ValueError("boom")
        monkeypatch.setattr(dashboard, "_explorer_curve", broken)
        with caplog.at_level(logging.WARNING):
            page = _kc_page(monkeypatch, tmp_path, _ex_sweep())
        assert 'id="scn-unavailable"' in page and 'id="scn-data"' not in page
        assert any("could not be built for this run" in r.getMessage()
                   for r in caplog.records)
        # The page is written, the bar intact, and it does not claim the
        # explorer follows it
        for piece in ('id="flt-bar"', "Portfolio Performance", "Benchmark Comparison"):
            assert piece in page
        assert html.escape(dashboard._BAR_REACH_NO_EXPLORER) in page

    def test_a_same_title_simulation_failure_costs_only_the_same_title_rows(
            self, monkeypatch, tmp_path, caplog):
        # The size-cap sweep's same-title population cannot be simulated,
        # while every cell can: the bar and the explorer keep their cap axis,
        # and the explorer's same-title row is the run's own, at its own cap
        sweep = _ex_sweep()

        def broken():
            raise RuntimeError("same-title cap simulation failed")
        sweep.cap_sweep.same_title = broken
        with caplog.at_level(logging.WARNING):
            page = _kc_page(monkeypatch, tmp_path, sweep)
        warned = [r for r in caplog.records if r.levelno == logging.WARNING
                  and "same-title" in r.getMessage()]
        assert [r.getMessage() for r in warned] == [
            "The same-title population could not be simulated at every size cap; the "
            "Scenario Explorer shows it at the run's own cap only"]
        assert warned[0].exc_info is not None
        assert not any("size-cap sweep could not be simulated" in r.getMessage()
                       for r in caplog.records)
        assert [c["label"] for c in TestFilterPage._data(page)["caps"]] == [
            "5%", "20%", "off (full Kelly)"]
        assert sweep.cap_sweep.reads == [(_KC_B0, 0.75), (_KC_B0, 0.6), (_KC_B1, 0.6),
                                         (_KC_B1, 0.75)]
        section = _ex_section(page)
        blocks = _scn_blocks(section)
        assert [blocks[f"scn-cap-{ci}"]["same_title"] for ci in (0, 2)] == [None, None]
        assert blocks["scn-cap-1"]["same_title"]["trades"] == len(
            sweep.same_title_point.trades)
        assert html.escape(dashboard._BAR_REACH_NO_EXPLORER_TIERS[(True, True)]) in page
        assert 'id="scn-own-cap"' not in page

    def test_a_late_cell_is_shown_over_the_page_s_span_as_the_bar_shows_it(
            self, monkeypatch):
        # A cell simulated after UTC midnight carries one flat row past the
        # page's last date. The explorer reads it through the walk, cut to the
        # page's span: its Sharpe is the cut curve's — the filter bar's figure
        # for the same scenario — and not the over-long curve's
        class Clock(datetime):
            moment = datetime(2026, 3, 1, 23, 58, tzinfo=UTC)

            @classmethod
            def now(cls, tz=None):
                return cls.moment if tz is None else cls.moment.astimezone(tz)
        monkeypatch.setattr(backtester, "datetime", Clock)
        trades = _kc_trades(5)
        page_curve = backtester._build_equity_curve(trades, _FLT_START, 1000.0)
        Clock.moment = datetime(2026, 3, 2, 0, 2, tzinfo=UTC)
        late_curve = backtester._build_equity_curve(trades, _FLT_START, 1000.0)
        assert len(late_curve) == len(page_curve) + 1

        def pops(k, curve):
            point = SweepPoint(k=k, trades=trades, equity_df=curve, spread_band=_KC_B0,
                               size_cap=0.2)
            return {"all": point, "time_series": dataclasses.replace(point,
                                                                     population="time_series")}
        cells = {0.6: {0.2: pops(0.6, late_curve)}, 0.75: {0.2: pops(0.75, page_curve)}}
        source = dashboard._GridSource(bands=(_KC_B0,), ks=(0.6, 0.75), caps=(0.2,),
                                       primary=(0, 1, 0), cell=lambda b, k: cells[k],
                                       calibrations={_KC_B0: None}, checks=True)
        primary = cells[0.75][0.2]["all"]
        sweep = BacktestSweep(primary=primary, points=[primary], calibration=None,
                              label_coverage=_scn_coverage(),
                              scenarios=list(cells[0.75][0.2].values()))
        explorer = dashboard._ExplorerVisitor(source, sweep)
        _, chunker, _ = dashboard._build_filter_grid(
            source, trades, page_curve, 0.75, _FLT_START, 1000.0, _FLT_SERIES,
            explorer=explorer)
        row = _unpack(explorer.payload().blocks[0])["cells"][0][0][0]     # the late "all"
        cut = dashboard._sharpe(page_curve["daily_return"])
        assert row["sharpe"] == pytest.approx(cut)
        assert row["sharpe"] != pytest.approx(dashboard._sharpe(late_curve["daily_return"]))
        chunk = _unpack(chunker.chunks[chunker.grid[0][0][0]])
        assert chunk["list"]["views"]["all"]["kpi"]["sharpe"] == f"{row['sharpe']:.2f}"

    # ─── The script: the explorer and the filter bar in one program ───────────

    @staticmethod
    def _page(monkeypatch, tmp_path):
        page = _kc_page(monkeypatch, tmp_path, _ex_sweep())
        section = _ex_section(page)
        return page, _scn_data(section), _scn_blocks(section)

    @staticmethod
    def _heat_updates(snap):
        return [u for u in snap["updates"] if u["id"] == "scn-heatmap"]

    def test_a_metric_change_updates_the_heatmap_as_python_built_it(
            self, monkeypatch, tmp_path):
        page, data, blocks = self._page(monkeypatch, tmp_path)
        cap = blocks["scn-cap-1"]
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["snap", "ready"],
            ["set", "scn-metric", "3"], ["fire", "scn-metric"], ["snap", "sharpe"],
            ["set", "scn-metric", "8"], ["fire", "scn-metric"], ["snap", "delta"]],
            strict=True, explorer=True)
        ready = snaps["ready"]
        assert not any(ready["selects"][s]["disabled"] for s in
                       ("scn-band-select", "scn-k-select", "scn-cap-select", "scn-metric"))
        # Loaded: the base block and the primary cap's, nothing else
        assert {i for i in ready["inflated"] if i.startswith("scn-")} == {"scn-data",
                                                                         "scn-cap-1"}
        assert self._heat_updates(ready) == []
        for name, mi, z, custom in (
                ("sharpe", 3, cap["matrices"]["sharpe"], cap["matrices"]["trades"]),
                ("delta", 8, data["matrices"]["khat_minus_k"],
                 data["matrices"]["empirical_k_n"])):
            (update,) = self._heat_updates(snaps[name])
            spec = data["metrics"][mi]
            assert update["data"] == {"z": [z], "colorscale": [spec["colorscale"]],
                                      "zmid": [spec["zmid"]], "hovertemplate": [spec["hover"]],
                                      "customdata": [custom]}
            assert update["layout"] == {"title.text": cap["titles"][mi]}

    def test_a_cap_change_loads_that_cap_s_block_and_redraws(self, monkeypatch, tmp_path):
        page, data, blocks = self._page(monkeypatch, tmp_path)
        small = blocks["scn-cap-0"]
        snap = _run_script(tmp_path, page, [
            ["wait"], ["set", "scn-metric", "2"], ["fire", "scn-metric"],
            ["set", "scn-cap-select", "0"], ["fire", "scn-cap-select"], ["snap", "loading"],
            ["settle"], ["snap", "cap"]], strict=True, explorer=True)
        assert snap["loading"]["text"]["scn-status"] == "Loading 5% cap per trade…"
        cap = snap["cap"]
        assert "scn-cap-0" in cap["inflated"] and "scn-cap-2" not in cap["inflated"]
        # The metric chosen, at the cap chosen, with its title and banner
        update = self._heat_updates(cap)[-1]
        assert update["data"]["z"] == [small["matrices"]["total_return"]]
        assert update["layout"] == {"title.text": small["titles"][2]}
        assert cap["html"]["scn-banner"] == small["banner"]
        assert cap["text"].get("scn-status", "") == ""          # cleared once drawn
        # The KPI table and the curve: the primary cell at 5%
        ts = data["populations"].index("time_series")
        cell = small["cells"][0][1]
        first = re.findall(r"<td[^>]*>(.*?)</td>", cap["html"]["scn-kpi-body"])
        assert first[0] == html.escape(_TS_LABEL) and first[1] == str(cell[ts]["trades"])
        restyle = [u for u in cap["updates"] if u["id"] == "scn-equity"][-1]
        assert restyle["data"]["y"] == [_expand(cell[ts]["equity"], len(data["dates"]))]
        assert restyle["data"]["x"] == [data["dates"]]

    def test_the_explorer_follows_the_bar_even_before_its_data_is_unpacked(
            self, monkeypatch, tmp_path):
        page, data, blocks = self._page(monkeypatch, tmp_path)
        snaps = _run_script(tmp_path, page, [
            ["wait"],
            # The bar moves to 5% before the explorer's base block is unpacked
            ["set", "flt-cap", "0"], ["fire", "flt-cap"], ["settle"], ["snap", "before"],
            ["resolve", "scn-data"], ["settle"], ["snap", "after"],
            # ... then to another band and k: the explorer follows at once
            ["set", "flt-band", "1"], ["fire", "flt-band"], ["settle"],
            ["set", "flt-k", "0"], ["fire", "flt-k"], ["settle"], ["snap", "moved"]],
            strict=True, explorer=True, deferred=("scn-data",))
        before, after, moved = snaps["before"], snaps["after"], snaps["moved"]
        assert before["selects"]["scn-cap-select"]["value"] == "1"
        assert before["selects"]["scn-cap-select"]["disabled"]
        # Applied once the explorer's data is there, and drawn
        assert after["selects"]["scn-cap-select"]["value"] == "0"
        assert not after["selects"]["scn-cap-select"]["disabled"]
        assert "scn-cap-0" in after["inflated"]
        assert self._heat_updates(after)[-1]["data"]["z"] == [
            blocks["scn-cap-0"]["matrices"]["mean_per_trade"]]
        assert after["html"]["scn-banner"] == blocks["scn-cap-0"]["banner"]
        assert [moved["selects"][s]["value"] for s in
                ("scn-band-select", "scn-k-select", "scn-cap-select")] == ["1", "0", "0"]
        # The band's own calibration table (0.3-0.6 has none)
        assert "No time-series candidate was measurable" in moved["html"]["scn-cal-body"]

    def test_an_unknown_label_leaves_its_select_as_it_is(self, monkeypatch, tmp_path):
        page, data, _ = self._page(monkeypatch, tmp_path)
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["snap", "ready"],
            ["call", "dashScenarioSelect", ["no such band", data["k_labels"][0], "no such cap"]],
            ["settle"], ["snap", "k"],
            ["call", "dashScenarioSelect", ["no such band", "no such k", "no such cap"]],
            ["settle"], ["snap", "none"]], strict=True, explorer=True)
        ready, k, none = snaps["ready"], snaps["k"], snaps["none"]
        # On load the explorer drew its tables and the primary curve only
        assert [u["id"] for u in ready["updates"]] == ["scn-equity"]
        assert [k["selects"][s]["value"] for s in
                ("scn-band-select", "scn-k-select", "scn-cap-select")] == ["0", "0", "1"]
        assert [u["id"] for u in k["updates"]] == ["scn-equity"]
        # Nothing it names exists: nothing moves and nothing is drawn
        assert none["updates"] == []
        assert [none["selects"][s]["value"] for s in
                ("scn-band-select", "scn-k-select", "scn-cap-select")] == ["0", "0", "1"]

    def test_a_category_change_leaves_the_explorer_s_own_choice(self, monkeypatch, tmp_path):
        page, _, _ = self._page(monkeypatch, tmp_path)
        other = str(TestFilterPage._data(page)["categories"].index("Other"))
        snaps = _run_script(tmp_path, page, [
            ["wait"],
            # The reader moves the explorer on its own ...
            ["set", "scn-band-select", "1"], ["fire", "scn-band-select"], ["settle"],
            ["set", "scn-k-select", "0"], ["fire", "scn-k-select"], ["settle"],
            ["snap", "own"],
            # ... then filters the page by a category: the bar's band, k and
            # cap did not change, so neither does the explorer
            _tick_category(other), ["settle"], ["snap", "cat"]],
            strict=True, explorer=True)
        own, cat = snaps["own"], snaps["cat"]
        assert _scn_values(own) == ["1", "0", "1"]
        assert cat["text"]["flt-summary"].startswith("Showing Other within the run at ")
        assert _scn_values(cat) == ["1", "0", "1"]
        assert [u for u in cat["updates"] if u["id"].startswith("scn-")] == []

    def test_a_bar_change_moves_only_the_explorer_s_axis_that_changed(
            self, monkeypatch, tmp_path):
        page, _, blocks = self._page(monkeypatch, tmp_path)
        snaps = _run_script(tmp_path, page, [
            ["wait"],
            # The reader picks k 0.60 and the 5% cap in the explorer itself
            ["set", "scn-k-select", "0"], ["fire", "scn-k-select"], ["settle"],
            ["set", "scn-cap-select", "0"], ["fire", "scn-cap-select"], ["settle"],
            ["snap", "own"],
            # The bar moves to the other band: the explorer's band follows,
            # its k and cap stay the reader's
            ["set", "flt-band", "1"], ["fire", "flt-band"], ["settle"], ["snap", "band"],
            # The bar's cap then moves: the explorer's cap alone follows
            ["set", "flt-cap", "2"], ["fire", "flt-cap"], ["settle"], ["snap", "cap"]],
            strict=True, explorer=True)
        assert _scn_values(snaps["own"]) == ["0", "0", "0"]
        assert _scn_values(snaps["band"]) == ["1", "0", "0"]
        assert _scn_values(snaps["cap"]) == ["1", "0", "2"]
        assert snaps["cap"]["html"]["scn-banner"] == blocks["scn-cap-2"]["banner"]

    def test_a_bar_change_while_an_explorer_cap_loads_keeps_the_reader_s_cap(
            self, monkeypatch, tmp_path):
        page, _, blocks = self._page(monkeypatch, tmp_path)
        snaps = _run_script(tmp_path, page, [
            ["wait"],
            # The reader picks 5% in the explorer; its block is still unpacking
            ["set", "scn-cap-select", "0"], ["fire", "scn-cap-select"], ["snap", "loading"],
            # ... when the bar moves to the other band alone
            ["set", "flt-band", "1"], ["fire", "flt-band"], ["settle"], ["snap", "bar"],
            ["resolve", "scn-cap-0"], ["snap", "arrived"]],
            strict=True, explorer=True, deferred=("scn-cap-0",))
        assert snaps["loading"]["text"]["scn-status"] == "Loading 5% cap per trade…"
        assert _scn_values(snaps["bar"]) == ["1", "1", "0"]
        arrived = snaps["arrived"]
        # The reader's cap, at the bar's band, drawn once its block is there
        assert _scn_values(arrived) == ["1", "1", "0"]
        assert arrived["html"]["scn-banner"] == blocks["scn-cap-0"]["banner"]
        assert arrived["text"].get("scn-status", "") == ""

    def test_a_superseded_cap_load_is_never_drawn(self, monkeypatch, tmp_path):
        page, _, blocks = self._page(monkeypatch, tmp_path)
        snaps = _run_script(tmp_path, page, [
            ["wait"],
            # Two caps chosen while neither block has arrived
            ["set", "scn-cap-select", "0"], ["fire", "scn-cap-select"],
            ["set", "scn-cap-select", "2"], ["fire", "scn-cap-select"], ["settle"],
            # The first arrives: the reader has moved past it, so it is not drawn
            ["resolve", "scn-cap-0"], ["snap", "first"],
            # A metric change meanwhile redraws the cap still on screen
            ["set", "scn-metric", "2"], ["fire", "scn-metric"], ["snap", "metric"],
            ["resolve", "scn-cap-2"], ["snap", "second"]],
            strict=True, explorer=True, deferred=("scn-cap-0", "scn-cap-2"))
        first, metric, second = snaps["first"], snaps["metric"], snaps["second"]
        assert first["text"]["scn-status"] == "Loading no per-trade cap…"
        assert self._heat_updates(first) == [] and "scn-banner" not in first["html"]
        (update,) = self._heat_updates(metric)
        assert update["layout"] == {"title.text": blocks["scn-cap-1"]["titles"][2]}
        assert update["data"]["z"] == [blocks["scn-cap-1"]["matrices"]["total_return"]]
        # The second is the one drawn, at the metric chosen while it loaded
        assert _scn_values(second) == ["0", "1", "2"]
        assert second["html"]["scn-banner"] == blocks["scn-cap-2"]["banner"]
        assert self._heat_updates(second)[-1]["data"]["z"] == [
            blocks["scn-cap-2"]["matrices"]["total_return"]]
        assert second["text"].get("scn-status", "") == ""

    def test_a_cap_block_that_cannot_be_unpacked_is_named_and_can_be_retried(
            self, monkeypatch, tmp_path):
        page, _, blocks = self._page(monkeypatch, tmp_path)
        snaps = _run_script(tmp_path, page, [
            ["wait"],
            # A band and the 5% cap chosen, whose block cannot be unpacked
            ["set", "scn-band-select", "1"], ["set", "scn-cap-select", "0"],
            ["fire", "scn-cap-select"], ["settle"], ["snap", "failed"],
            # Once it reads, choosing it again loads and draws it
            ["repair", "scn-cap-0"],
            ["set", "scn-cap-select", "0"], ["fire", "scn-cap-select"], ["settle"],
            ["snap", "retried"]],
            strict=True, explorer=True, damaged=("scn-cap-0",))
        failed, retried = snaps["failed"], snaps["retried"]
        # Every select back on the scenario still shown, and the line says why
        assert _scn_values(failed) == ["0", "1", "1"]
        status = failed["text"]["scn-status"]
        assert status.startswith("The explorer could not load 5% cap per trade (")
        assert status.endswith("); it still shows 20% cap per trade.")
        assert self._heat_updates(failed) == []
        assert _scn_values(retried) == ["0", "1", "0"]
        assert retried["html"]["scn-banner"] == blocks["scn-cap-0"]["banner"]
        assert retried["text"].get("scn-status", "") == ""

    def test_explorer_data_that_cannot_be_unpacked_says_so(self, monkeypatch, tmp_path):
        page, _, _ = self._page(monkeypatch, tmp_path)
        snap = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-cap", "0"], ["fire", "flt-cap"], ["settle"], ["snap", "s"]],
            strict=True, explorer=True, damaged=("scn-data",))["s"]
        assert all(snap["selects"][s]["disabled"] for s in
                   ("scn-band-select", "scn-k-select", "scn-cap-select", "scn-metric"))
        assert "The Scenario Explorer could not load its data" in snap["html"]["scn-kpi-body"]
        # The bar is unaffected, and its call reaches no half-built explorer
        assert not any(snap["selects"][i]["disabled"] for i in _FLT_SELECTS)
        assert "5% cap per trade" in snap["text"]["flt-summary"]
        assert self._heat_updates(snap) == []

    def test_a_page_without_the_explorer_s_grid_still_redraws_on_k_and_cap(
            self, monkeypatch, tmp_path):
        # No band sweep: no explorer script, so no dashScenarioSelect — the
        # bar redraws every section on k and cap, and reaches no missing element
        page = _kc_page(monkeypatch, tmp_path)
        assert "function unpack(el)" not in page
        data = TestFilterPage._data(page)
        other = str(data["categories"].index("Other"))
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-k", "0"], ["fire", "flt-k"], ["settle"],
            ["set", "flt-cap", "0"], ["fire", "flt-cap"], ["settle"], ["snap", "s"],
            # From a chunk already loaded the redraw is synchronous, inside
            # the change event itself: a call to a function the page does
            # not define would throw out of it and end the run
            _tick_category(other), ["snap", "cat"]],
            strict=True, explorer=True)
        snap, cat = snaps["s"], snaps["cat"]
        assert {r["id"] for r in snap["reacts"]} == _charts_redrawn()
        assert "k = 0.60, 5% cap per trade" in snap["text"]["flt-summary"]
        assert snap["text"]["flt-summary"].endswith(dashboard._BAR_REACH_NO_EXPLORER)
        assert cat["text"]["flt-summary"].startswith("Showing Other within the run at ")
        assert {r["id"] for r in cat["reacts"]} >= {"perf-cum", "kd-equity"}

    # The ids the explorer's script reaches only once the base block says the
    # page has a tier-floors-off view (D.tier_labels) — its Tier floors
    # select, and the hidden k-hat table a setting change shows
    _OFF_ONLY = {"scn-tier-select", "scn-khat-off"}

    @pytest.mark.parametrize("family", [False, True], ids=["tier-on-only", "tier-off-family"])
    def test_every_element_the_explorer_script_reaches_exists(self, monkeypatch, tmp_path,
                                                               family):
        # Without a family, every id but the two it reaches only with an off
        # view is on the page (strict runs of such pages pin that it never
        # reaches for those two); with one, every id, off blocks included
        page = _kc_page(monkeypatch, tmp_path, _ex_sweep_tiers() if family else _ex_sweep())
        data = _scn_data(_ex_section(page))
        js = dashboard._SCENARIO_EXPLORER_JS
        literal = {i for i in re.findall(r"(?:byId|getElementById|restyle|update)\('"
                                         r"([a-z][a-z0-9_-]*)'", js) if not i.endswith("-")}
        dynamic = {f"scn-cap-{ci}" for ci in range(len(data["caps"]))}
        if family:
            dynamic |= {f"scn-off-cap-{ci}" for ci, own in enumerate(data["off_caps"]) if own}
        missing = sorted(i for i in literal | dynamic if f'id="{i}"' not in page
                         and f"id='{i}'" not in page)
        assert literal >= {"scn-data", "scn-kpi-body", "scn-heatmap", "scn-equity",
                           "scn-banner", "scn-status", "scn-khat-on", *self._OFF_ONLY}
        assert missing == ([] if family else sorted(self._OFF_ONLY))
        assert "'scn-cap-' + ci" not in js and "prefix + ci" in js
        assert "'scn-cap-'" in js and "'scn-off-cap-'" in js
        # Every block precedes the script that unpacks it
        section = _ex_section(page)
        script = section.index("function unpack(el)")
        assert all(section.index(f'id="{i}"') < script for i in ("scn-data", *dynamic))

    # ─── The tier-floors-off view at a size cap ───────────────────────────────

    def test_an_off_block_ships_only_at_a_cap_the_family_was_simulated_at(
            self, monkeypatch, tmp_path):
        page = _kc_page(monkeypatch, tmp_path, _ex_sweep_tiers())
        blocks = _scn_blocks(_ex_section(page))
        assert list(blocks) == ["scn-data", "scn-cap-0", "scn-cap-1", "scn-cap-2",
                                "scn-off-cap-1"]
        data = blocks["scn-data"]
        assert data["tier_binds"] == [True, False] and data["off_caps"] == [False, True, False]
        off = blocks["scn-off-cap-1"]
        pop = {name: i for i, name in enumerate(data["populations"])}
        assert off["cells"][1] is None
        assert [cell[pop["all"]]["trades"] for cell in off["cells"][0]] == [2, 2]
        assert off["titles"][0] == blocks["scn-cap-1"]["titles"][0] + " — tier floors off"
        # The bar's off grid agrees: the binding band's own chunk at 20%, and
        # at the other caps a scenario never simulated
        base = TestFilterPage._data(page)
        assert [row[0] is None and row[2] is None and row[1] is not None
                for row in base["grid_off"][0]] == [True, True]
        # The page's summary names the explorer's Tier floors select
        assert html.escape(dashboard._BAR_REACH) in page

    def test_a_cap_not_simulated_with_the_tiers_off_is_refused(self, monkeypatch, tmp_path):
        page = _kc_page(monkeypatch, tmp_path, _ex_sweep_tiers())
        blocks = _scn_blocks(_ex_section(page))
        data = blocks["scn-data"]
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["set", "scn-tier-select", "off"], ["fire", "scn-tier-select"],
            ["settle"], ["snap", "off"],
            # The explorer's own cap select: 5% was never simulated with the
            # tiers off
            ["set", "scn-cap-select", "0"], ["fire", "scn-cap-select"], ["settle"],
            ["snap", "refused"],
            # The bar's: at no cap, tiers off (its own view reads "missing")
            ["set", "flt-tier", "off"], ["fire", "flt-tier"], ["settle"],
            ["set", "flt-cap", "2"], ["fire", "flt-cap"], ["settle"], ["snap", "bar"],
            # The tiers back on at 5%: drawn
            ["set", "scn-cap-select", "0"], ["set", "scn-tier-select", "on"],
            ["fire", "scn-tier-select"], ["settle"], ["snap", "on"]],
            strict=True, explorer=True)
        off, refused, bar, on = snaps["off"], snaps["refused"], snaps["bar"], snaps["on"]
        assert off["html"]["scn-banner"] == blocks["scn-off-cap-1"]["banner"]
        # Every select back on the scenario shown, the line in Python's words,
        # nothing drawn and no block loaded for it
        assert _scn_values(refused) + [refused["selects"]["scn-tier-select"]["value"]] == [
            "0", "1", "1", "off"]
        assert refused["text"]["scn-status"] == data["off_missing"].format(
            failed="5% cap per trade with the tier floors off",
            shown="20% cap per trade with the tier floors off")
        assert _explorer_calls(refused) == []
        assert not any(i.startswith("scn-off-cap-0") for i in refused["inflated"])
        assert bar["text"]["flt-summary"].startswith("Showing nothing: ")
        assert _scn_values(bar) == ["0", "1", "1"]
        assert bar["text"]["scn-status"] == data["off_missing"].format(
            failed="no per-trade cap with the tier floors off",
            shown="20% cap per trade with the tier floors off")
        assert _scn_values(on) == ["0", "1", "0"]
        assert on["html"]["scn-banner"] == blocks["scn-cap-0"]["banner"]
        assert on["text"].get("scn-status", "") == ""

    def test_a_bar_cap_the_explorer_refused_is_applied_on_the_bar_s_next_move(
            self, monkeypatch, tmp_path):
        # The bar turns the tier floors off (the explorer follows, at 20%),
        # then picks 5% — never simulated with the tiers off, so the explorer
        # refuses it — then turns the tiers back on: the explorer follows the
        # bar to 5% with the tiers on, the cap it refused included, rather
        # than staying at 20% under a bar that shows 5%
        page = _kc_page(monkeypatch, tmp_path, _ex_sweep_tiers())
        blocks = _scn_blocks(_ex_section(page))
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-tier", "off"], ["fire", "flt-tier"], ["settle"],
            ["set", "flt-cap", "0"], ["fire", "flt-cap"], ["settle"], ["snap", "refused"],
            ["set", "flt-tier", "on"], ["fire", "flt-tier"], ["settle"], ["snap", "back"]],
            strict=True, explorer=True)
        refused, back = snaps["refused"], snaps["back"]
        assert _scn_values(refused) + [refused["selects"]["scn-tier-select"]["value"]] == [
            "0", "1", "1", "off"]
        assert refused["text"]["scn-status"] == blocks["scn-data"]["off_missing"].format(
            failed="5% cap per trade with the tier floors off",
            shown="20% cap per trade with the tier floors off")
        bar = [back["selects"][s]["value"] for s in ("flt-cap", "flt-tier")]
        assert bar == ["0", "on"]
        assert [back["selects"][s]["value"] for s in ("scn-cap-select", "scn-tier-select")] == bar
        assert back["html"]["scn-banner"] == blocks["scn-cap-0"]["banner"]
        assert back["text"].get("scn-status", "") == ""

    def test_a_bar_setting_the_explorer_refused_is_applied_on_the_bar_s_next_move(
            self, monkeypatch, tmp_path):
        # The bar picks 5% (drawn), then turns the tier floors off — refused
        # at 5%. The reader then picks a k in the explorer itself (drawn,
        # clearing the refusal line), and a bar category change — no axis of
        # the bar's changed — re-attempts nothing: the reader's k and the
        # cleared line stay. Then the bar picks 20%: the explorer follows it
        # to 20% with the tiers OFF, as the bar shows, rather than drawing
        # the tiers on under a tiers-off bar, and keeps the reader's k
        page = _kc_page(monkeypatch, tmp_path, _ex_sweep_tiers())
        blocks = _scn_blocks(_ex_section(page))
        data = TestFilterPage._data(page)
        other = str(len(data["categories"]) - 1)
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-cap", "0"], ["fire", "flt-cap"], ["settle"],
            ["set", "flt-tier", "off"], ["fire", "flt-tier"], ["settle"], ["snap", "refused"],
            ["set", "scn-k-select", "0"], ["fire", "scn-k-select"], ["settle"],
            ["snap", "reader"],
            _tick_category(other), ["settle"], ["snap", "cat"],
            ["set", "flt-cap", "1"], ["fire", "flt-cap"], ["settle"], ["snap", "back"]],
            strict=True, explorer=True)
        refused, reader, cat, back = (snaps["refused"], snaps["reader"], snaps["cat"],
                                      snaps["back"])
        assert _scn_values(refused) + [refused["selects"]["scn-tier-select"]["value"]] == [
            "0", "1", "0", "on"]
        assert refused["text"]["scn-status"] == blocks["scn-data"]["off_missing"].format(
            failed="5% cap per trade with the tier floors off",
            shown="5% cap per trade")
        assert reader["text"].get("scn-status", "") == ""
        # The bar redrew (its missing-scenario line, as before) and called
        # the explorer — a call that changed none of its axes
        assert cat["selects"]["flt-cat"]["value"] == other
        assert cat["text"]["flt-summary"].startswith("Showing nothing: ")
        assert _scn_values(cat) + [cat["selects"]["scn-tier-select"]["value"]] == [
            "0", "0", "0", "on"]
        assert _explorer_calls(cat) == []
        assert cat["text"].get("scn-status", "") == ""
        bar = [back["selects"][s]["value"] for s in ("flt-cap", "flt-tier")]
        assert bar == ["1", "off"]
        assert [back["selects"][s]["value"] for s in ("scn-cap-select", "scn-tier-select")] == bar
        assert back["selects"]["scn-k-select"]["value"] == "0"
        assert back["html"]["scn-banner"] == blocks["scn-off-cap-1"]["banner"]
        assert back["text"].get("scn-status", "") == ""

    def test_an_off_block_that_cannot_be_unpacked_is_named_and_puts_the_select_back(
            self, monkeypatch, tmp_path):
        page = _kc_page(monkeypatch, tmp_path, _ex_sweep_tiers())
        blocks = _scn_blocks(_ex_section(page))
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["snap", "ready"],
            ["set", "scn-tier-select", "off"], ["fire", "scn-tier-select"],
            ["settle"], ["snap", "failed"],
            ["repair", "scn-off-cap-1"],
            ["set", "scn-tier-select", "off"], ["fire", "scn-tier-select"], ["settle"],
            ["snap", "retried"]],
            strict=True, explorer=True, damaged=("scn-off-cap-1",))
        failed, retried = snaps["failed"], snaps["retried"]
        assert failed["selects"]["scn-tier-select"]["value"] == "on"
        status = failed["text"]["scn-status"]
        assert status.startswith("The explorer could not load 20% cap per trade with the "
                                 "tier floors off (")
        assert status.endswith("); it still shows 20% cap per trade.")
        assert _explorer_calls(failed) == []
        assert retried["selects"]["scn-tier-select"]["value"] == "off"
        assert retried["html"]["scn-banner"] == blocks["scn-off-cap-1"]["banner"]

    def test_an_off_block_the_bar_s_move_could_not_load_is_loaded_on_its_next_move(
            self, monkeypatch, tmp_path):
        # The bar turns the tier floors off, but the off block cannot be
        # unpacked: the explorer puts its Tier floors select back on. Once
        # the block can be read, the bar's next move — a k change — brings
        # the setting along with it, rather than drawing the tiers on under
        # a tiers-off bar
        page = _kc_page(monkeypatch, tmp_path, _ex_sweep_tiers())
        blocks = _scn_blocks(_ex_section(page))
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-tier", "off"], ["fire", "flt-tier"], ["settle"],
            ["snap", "failed"], ["repair", "scn-off-cap-1"],
            ["set", "flt-k", "0"], ["fire", "flt-k"], ["settle"], ["snap", "back"]],
            strict=True, explorer=True, damaged=("scn-off-cap-1",))
        failed, back = snaps["failed"], snaps["back"]
        assert failed["selects"]["scn-tier-select"]["value"] == "on"
        assert failed["text"]["scn-status"].startswith(
            "The explorer could not load 20% cap per trade with the tier floors off (")
        bar = [back["selects"][s]["value"] for s in ("flt-k", "flt-tier")]
        assert bar == ["0", "off"]
        assert [back["selects"][s]["value"] for s in ("scn-k-select", "scn-tier-select")] == bar
        assert back["html"]["scn-banner"] == blocks["scn-off-cap-1"]["banner"]
        assert back["text"].get("scn-status", "") == ""

    def test_an_explorer_rebuilt_with_a_family_keeps_its_tier_floors_select(
            self, monkeypatch, tmp_path, caplog):
        # The walk's explorer visitor fails: rebuilt from the sweep's own
        # eager points (_explorer_from_sweep), the explorer keeps its
        # tier-floors-off view at the run's own cap — so the line says it
        # follows the bar's band, Tier floors choice and k but not its cap
        real, calls = dashboard._explorer_curve, []

        def first_fails(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise ValueError("boom")
            return real(*args, **kwargs)
        monkeypatch.setattr(dashboard, "_explorer_curve", first_fails)
        with caplog.at_level(logging.WARNING):
            page = _kc_page(monkeypatch, tmp_path, _ex_sweep_tiers())
        section = _ex_section(page)
        assert list(_scn_blocks(section)) == ["scn-data", "scn-cap-0", "scn-off-cap-0"]
        assert 'id="scn-tier-select"' in section
        assert html.escape(dashboard._BAR_REACH_OWN_CAP) in page
        assert html.escape(dashboard._BAR_REACH) not in page
        assert dashboard._EXPLORER_OWN_CAP_HTML in section

    def test_the_eager_fallback_walks_the_off_cells(self):
        # _explorer_from_sweep (the direct call's path) hands the family to
        # the visitor's off(): the section drawn from it has the off view
        data = dashboard._explorer_from_sweep(_ex_sweep_tiers())
        assert data.tier_select and data.base["off_caps"] == [True]
        assert len(data.off_blocks) == 1
        assert data.base["band_labels_off"] == ["0-1", "0.3-0.6"]
        assert [None if p is None else p.n for p in data.pooled_off] == [1, None]


# ─── Tier floors off at every size cap (C5) ──────────────────────────────────


def _off_small(event: str = "KXOTHER-1") -> list[BacktestTrade]:
    """The one-trade list a binding band's tier-off run trades at the 5% cap
    — a list no other scenario trades."""
    return [_kc_resized(_ftrade(event, "time_series", date(2026, 1, 13), date(2026, 1, 16),
                                3.0), 2)]


def _kc_sweep_tiers_capped(*, raise_on=None, small_event: str = "KXOTHER-1",
                           caps=(0.05, 0.2, 1.0)) -> BacktestSweep:
    """_kc_sweep_tiers() plus a tier-floors-off size-cap sweep over its one
    binding band (_KC_B0), as run_backtest_sweep builds one: at the run's own
    20% cap the family's own eager points (the same objects), at no cap a
    copy sharing their trades (the 20% cap already reached their peak), and
    at 5% a smaller one-trade list. `raise_on` makes that off cell's read
    raise, as a failed simulation would; `caps` sets its caps (a mismatch
    with the tier-on grid's makes it unusable)."""
    sweep = _kc_sweep_tiers()
    small = _off_small(small_event)
    curve = backtester._build_equity_curve(small, _FLT_START, 1000.0)
    points = {}
    for eager in sweep.tier_off_scenarios:
        points[(_KC_B0, eager.k, 0.05)] = SweepPoint(
            k=eager.k, trades=small, equity_df=curve, spread_band=_KC_B0, size_cap=0.05,
            tier_floors=False)
        points[(_KC_B0, eager.k, 0.2)] = eager
        points[(_KC_B0, eager.k, 1.0)] = dataclasses.replace(eager, size_cap=1.0)
    return dataclasses.replace(sweep, tier_off_cap_sweep=_FakeCapSweep(
        points, bands=(_KC_B0,), caps=caps, raise_on=raise_on))


def _ex_sweep_tiers_capped(*, raise_on=None) -> BacktestSweep:
    """_ex_sweep_tiers() plus a tier-floors-off size-cap sweep over its one
    binding band (_KC_B0) with the band sweep's checks, as
    _kc_sweep_tiers_capped builds for the bar: the family's own "all" and
    "time_series" points at 20%, copies at no cap, and a one-trade list at
    5%."""
    sweep = _ex_sweep_tiers()
    small = _off_small("KXRAIN-1")
    curve = backtester._build_equity_curve(small, _FLT_START, 1000.0)
    by_k: dict = {}
    for eager in sweep.tier_off_scenarios:
        by_k.setdefault(eager.k, {})[eager.population] = eager
    points = {}
    for k, pops in by_k.items():
        low = SweepPoint(k=k, trades=small, equity_df=curve, spread_band=_KC_B0,
                         size_cap=0.05, tier_floors=False,
                         halves=HalfSplit(0.001, 0.0, 1, 1), ex_top_event=("KXRAIN-1", 0.0))
        points[(_KC_B0, k, 0.05)] = {"all": low,
                                     "time_series": dataclasses.replace(
                                         low, population="time_series")}
        points[(_KC_B0, k, 0.2)] = dict(pops)
        points[(_KC_B0, k, 1.0)] = {pop: dataclasses.replace(p, size_cap=1.0)
                                    for pop, p in pops.items()}
    return dataclasses.replace(sweep, tier_off_cap_sweep=_ExplorerCapSweep(
        points, bands=(_KC_B0,), caps=_EX_CAPS, raise_on=raise_on))


_OFF_UNUSED_CLAUSE = ("per-trade cap: 20% (size-cap sweep on, but its tier-floors-off half "
                      "could not be used — with the tier floors off the filter bar offers the "
                      "run's own cap only; the log names why)</p>")
_OFF_WARNING = ("The tier-floors-off size-cap sweep could not be simulated; with the tier "
                "floors off the page offers the run's own cap only")
_ON_WARNING = "The size-cap sweep could not be simulated; the page offers the run's own cap only"


def _referenced_heads(chunks: dict[int, dict]) -> set[int]:
    """Every row-head index (into the base block's "rows" table) some
    shipped chunk's best / worst table refers to (a view with no trade has
    neither table)."""
    return {head for chunk in chunks.values() for view in chunk["list"]["views"].values()
            for which in ("best", "worst") for head, _tail in view.get(which, [])}


def _resolved_chunk(data: dict, chunk: dict) -> dict:
    """One chunk with every best / worst row resolved to its HTML (the page's
    shared row head plus the chunk's own tail), so two pages' chunks of one
    list compare equal whatever order their head tables were filled in."""
    out = json.loads(json.dumps(chunk))
    for view in out["list"]["views"].values():
        for which in ("best", "worst"):
            if which in view:
                view[which] = [data["rows"][head] + chunk["strings"][tail]
                               for head, tail in view[which]]
    return out


class TestTierOffAtEveryCap:
    """With a tier-floors-off size-cap sweep (BacktestSweep.tier_off_cap_sweep)
    the page's Tier floors off view covers every size cap: the grid reads a
    binding band's off cells from it, the bar ships their chunks, the
    explorer ships an off block per cap, and a failure costs the off view's
    other caps only — never the page, never the tier-on cap axis."""

    def test_the_grid_reads_its_off_cells_from_the_tier_off_cap_sweep(self):
        sweep = _kc_sweep_tiers_capped()
        source = dashboard._grid_source(sweep, sweep.primary.trades, sweep.primary.equity_df,
                                        0.75)
        assert source.off_cap_sweep is sweep.tier_off_cap_sweep
        cell = source.off_cell(_KC_B0, 0.75)
        assert sorted(cell) == [0.05, 0.2, 1.0]
        # The run's own cap is the family's own point, the object itself
        assert cell[0.2]["all"] is sweep.tier_off_scenarios[1]
        assert cell[0.05]["all"].trades == _off_small()
        # A band the tiers never bind at has no off cell: its tier-on run stands in
        assert source.off_cell(_KC_B1, 0.75) == {}
        # The fallback reads the family's eager points at the run's own cap
        # alone, keeping the tier-on cap axis and the bar's labels
        fallback = source.off_fallback()
        assert fallback.off_cap_sweep is None and fallback.off_fallback is None
        assert sorted(fallback.off_cell(_KC_B0, 0.75)) == [0.2]
        assert fallback.off_events == source.off_events
        assert fallback.cap_sweep is source.cap_sweep and fallback.caps == source.caps
        # Without the tier-on cap axis the off view stays at the run's own cap
        eager = dashboard._grid_source(sweep, sweep.primary.trades, sweep.primary.equity_df,
                                       0.75, use_cap_sweep=False)
        assert eager.off_cap_sweep is None and sorted(eager.off_cell(_KC_B0, 0.75)) == [0.2]

    def test_a_tier_off_cap_sweep_that_does_not_match_the_grid_is_set_aside(self, caplog):
        sweep = _kc_sweep_tiers_capped(caps=(0.05, 0.2))
        with caplog.at_level(logging.WARNING):
            source = dashboard._grid_source(sweep, sweep.primary.trades,
                                            sweep.primary.equity_df, 0.75)
        assert source.off_cap_sweep is None and source.off_fallback is None
        assert sorted(source.off_cell(_KC_B0, 0.75)) == [0.2]
        assert [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING] == [
            "The tier-floors-off size-cap sweep does not match the page's grid (its caps, "
            "ks, binding bands or primary cap); with the tier floors off the page offers "
            "the run's own cap only"]

    def test_the_bar_fills_every_cap_of_a_binding_band_with_the_tiers_off(
            self, monkeypatch, tmp_path):
        page = _kc_page(monkeypatch, tmp_path, _kc_sweep_tiers_capped())
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        grid, grid_off = data["grid"], data["grid_off"]
        pc = data["primary"][2]
        for ki in range(len(data["ks"])):
            row = grid_off[0][ki]
            assert None not in row, ki
            # 5%: the one-trade list; no cap shares the 20% chunk (one list)
            assert chunks[row[0]]["list"]["views"]["all"]["n"] == 1
            assert row[2] == row[pc] and row[0] != row[pc]
            assert row[0] not in grid[0][ki]
        # The band the tiers never bind at: its tier-on run at every cap
        assert grid_off[1] == grid[1]
        # Every chunk shipped is one some scenario shows
        shown = {cid for g in (grid, grid_off) for band in g for row in band for cid in row
                 if cid is not None}
        assert shown == set(chunks)
        # A healthy run's header line is unchanged
        assert "| per-trade cap: 20% (size-cap sweep on)</p>" in page

    def test_a_tier_off_choice_at_another_cap_draws_that_scenario(self, monkeypatch, tmp_path):
        page = _kc_page(monkeypatch, tmp_path, _kc_sweep_tiers_capped())
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        pb, pk, _ = data["primary"]
        cid = data["grid_off"][pb][pk][0]
        snap = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-cap", "0"], ["fire", "flt-cap"], ["settle"],
            ["set", "flt-tier", "off"], ["fire", "flt-tier"], ["settle"], ["snap", "off"]],
            strict=True)["off"]
        n = chunks[cid]["list"]["views"]["all"]["n"]
        assert snap["text"]["flt-summary"] == dashboard._filter_summary_text(
            data["text"], _phrase_off(data, pb, pk, 0), False, None, n, n)
        assert not snap["text"]["flt-summary"].startswith("Showing nothing")
        assert snap["text"]["hdr-trades"] == str(n)
        assert f"dash-chunk-{cid}" in snap["inflated"]

    def test_the_explorer_ships_an_off_block_per_cap_and_follows_the_bar_there(
            self, monkeypatch, tmp_path):
        page = _kc_page(monkeypatch, tmp_path, _ex_sweep_tiers_capped())
        blocks = _scn_blocks(_ex_section(page))
        data = blocks["scn-data"]
        assert data["off_caps"] == [True, True, True]
        assert [f"scn-off-cap-{ci}" in blocks for ci in range(3)] == [True, True, True]
        low = blocks["scn-off-cap-0"]
        # A binding band's row at 5%: the one-trade list; the other band's row
        # is read from the tier-on block
        assert low["cells"][1] is None
        assert low["cells"][0][1][0]["trades"] == 1                      # "all"
        assert blocks["scn-off-cap-1"]["cells"][0][1][0]["trades"] == 2
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-tier", "off"], ["fire", "flt-tier"], ["settle"],
            ["set", "flt-cap", "0"], ["fire", "flt-cap"], ["settle"], ["snap", "bar"],
            # The explorer's own cap select, with the tiers off: no refusal
            ["set", "scn-cap-select", "2"], ["fire", "scn-cap-select"], ["settle"],
            ["snap", "own"]], strict=True, explorer=True)
        bar, own = snaps["bar"], snaps["own"]
        assert _scn_values(bar) + [bar["selects"]["scn-tier-select"]["value"]] == [
            "0", "1", "0", "off"]
        assert bar["text"].get("scn-status", "") == ""
        assert bar["html"]["scn-banner"] == low["banner"]
        assert "scn-off-cap-0" in bar["inflated"]
        assert _scn_values(own) == ["0", "1", "2"]
        assert own["text"].get("scn-status", "") == ""
        assert own["html"]["scn-banner"] == blocks["scn-off-cap-2"]["banner"]

    def test_a_raising_off_cell_keeps_the_tier_on_cap_axis(
            self, monkeypatch, tmp_path, caplog):
        # The second off cell raises, after the first was packed at every cap:
        # the page falls back to the family's eager points at the run's own
        # cap for the off view alone — one WARNING, the tier-on caps kept, no
        # chunk shipped that nothing shows, and the header says why
        sweep = _ex_sweep_tiers_capped(raise_on=(_KC_B0, 0.75))
        with caplog.at_level(logging.WARNING):
            page = _kc_page(monkeypatch, tmp_path, sweep)
        warned = [r for r in caplog.records if r.levelno == logging.WARNING
                  and "size-cap sweep could not be simulated" in r.getMessage()]
        assert [r.getMessage() for r in warned] == [_OFF_WARNING]
        assert warned[0].exc_info is not None
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        pc = data["primary"][2]
        assert len(data["caps"]) == 3
        assert all(cid is not None for row in data["grid"][0] for cid in row)
        for row in data["grid_off"][0]:
            assert row[pc] is not None
            assert [row[ci] for ci in range(3) if ci != pc] == [None, None]
        shown = {cid for g in (data["grid"], data["grid_off"]) for band in g for row in band
                 for cid in row if cid is not None}
        assert shown == set(chunks)
        # ... nor a row head: the heads the dropped off cells packed went with them
        assert _referenced_heads(chunks) == set(range(len(data["rows"])))
        blocks = _scn_blocks(_ex_section(page))
        assert blocks["scn-data"]["off_caps"] == [False, True, False]
        assert sorted(b for b in blocks if b.startswith("scn-off-cap-")) == ["scn-off-cap-1"]
        assert [f"scn-cap-{ci}" in blocks for ci in range(3)] == [True, True, True]
        assert _OFF_UNUSED_CLAUSE in page and _KC_UNUSED_CLAUSE not in page
        # The page is whole
        assert "Trade-Level Diagnostics" in page and page.rstrip().endswith("</html>")

    def test_the_off_fallback_ships_the_eager_off_chunks_and_nothing_else(
            self, monkeypatch, tmp_path):
        # The first off cell (k 0.60) is packed at every cap — its 20% list, a
        # list no tier-on scenario trades, in a chunk packed AFTER the off
        # mark — and the second (k 0.75) raises. The re-walk from the eager
        # points must pack that 20% list again rather than reuse the id of
        # the chunk reset_off dropped: each k's off chunk at the run's own cap
        # is exactly the one a page without the tier-off size-cap sweep ships
        # there, every id the grids name is shipped and every shipped chunk
        # and row head is one some scenario shows
        page = _kc_page(monkeypatch, tmp_path, _ex_sweep_tiers_capped(raise_on=(_KC_B0, 0.75)))
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        eager_page = _kc_page(monkeypatch, tmp_path, _ex_sweep_tiers())
        eager_data, eager_chunks = (TestFilterPage._data(eager_page),
                                    TestFilterPage._chunks(eager_page))
        pc = data["primary"][2]
        assert eager_data["primary"][2] == pc
        tier_on = {cid for band in data["grid"] for row in band for cid in row
                   if cid is not None}
        off_ids = [data["grid_off"][0][ki][pc] for ki in range(len(data["ks"]))]
        # One chunk per k, neither of them a tier-on scenario's
        assert len(set(off_ids)) == len(off_ids) and not set(off_ids) & tier_on
        for ki, cid in enumerate(off_ids):
            assert cid in chunks, ki
            eager_cid = eager_data["grid_off"][0][ki][pc]
            assert _resolved_chunk(data, chunks[cid]) == _resolved_chunk(
                eager_data, eager_chunks[eager_cid]), ki
        shown = {cid for g in (data["grid"], data["grid_off"]) for band in g for row in band
                 for cid in row if cid is not None}
        assert shown <= set(chunks)
        assert set(chunks) <= shown
        assert _referenced_heads(chunks) == set(range(len(data["rows"])))

    def test_a_raising_tier_on_cell_still_falls_back_as_before(
            self, monkeypatch, tmp_path, caplog):
        sweep = _ex_sweep_tiers_capped()
        sweep.cap_sweep.raise_on = (_KC_B1, 0.75)
        with caplog.at_level(logging.WARNING):
            page = _kc_page(monkeypatch, tmp_path, sweep)
        warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
                  and "size-cap sweep could not be simulated" in r.getMessage()]
        assert warned == [_ON_WARNING]
        data = TestFilterPage._data(page)
        assert len(data["caps"]) == 1
        assert all(len(row) == 1 and row[0] is not None for row in data["grid_off"][0])
        assert _KC_UNUSED_CLAUSE in page and _OFF_UNUSED_CLAUSE not in page

    def test_an_off_fallback_keeps_the_tier_on_cap_axis(self):
        # The first off cell (k 0.6) holds 9 trades at no cap; the second
        # raises. The walk then re-reads the off view from the family's eager
        # points, which the page shows instead, and keeps the tier-on caps
        sweep = _kc_sweep_tiers_capped(raise_on=(_KC_B0, 0.75))
        big = [_kc_resized(_ftrade(f"KXBIG-{i}", "time_series", date(2026, 1, 13),
                                   date(2026, 1, 16), 1.0), 1) for i in range(9)]
        sweep.tier_off_cap_sweep.points[(_KC_B0, 0.6, 1.0)] = SweepPoint(
            k=0.6, trades=big, equity_df=backtester._build_equity_curve(big, _FLT_START, 1000.0),
            spread_band=_KC_B0, size_cap=1.0, tier_floors=False)
        source = dashboard._grid_source(sweep, sweep.primary.trades, sweep.primary.equity_df,
                                        0.75)
        walked, _, _ = dashboard._build_filter_grid(
            source, sweep.primary.trades, sweep.primary.equity_df, 0.75, _FLT_START, 1000.0,
            _FLT_SERIES)
        assert walked.off_cap_sweep is None and walked.cap_sweep is source.cap_sweep

    def test_the_labels_include_every_tier_off_cap_event(self):
        series = {**_FLT_SERIES, "KXSNOW": ("Climate", ("Snow",))}
        sweep = _kc_sweep_tiers_capped(small_event="KXSNOW-1")
        trades, curve = sweep.primary.trades, sweep.primary.equity_df
        capped = dashboard._grid_source(sweep, trades, curve, 0.75)
        eager = dashboard._grid_source(sweep, trades, curve, 0.75, use_cap_sweep=False)
        cats, subs = dashboard._filter_labels(capped, trades, series)
        assert "Climate" in cats and ("Climate", "Snow") in subs
        # Only a tier-off cap point trades it: the eager grid never lists it
        assert "Climate" not in dashboard._filter_labels(eager, trades, series)[0]

    def test_the_run_settings_line_names_a_tier_off_cap_sweep_it_could_not_use(self):
        pt = dataclasses.replace(_scn_point((0.3, 0.6), 0.75), size_cap=0.2)
        sweep = BacktestSweep(primary=pt, points=[pt], calibration=None,
                              label_coverage=_scn_coverage(), cap_sweep=object(),
                              tier_off_cap_sweep=object(), same_event_ladders=True)
        assert dashboard._run_settings_html(
            sweep, tier_off_cap_sweep_unused=True).endswith(_OFF_UNUSED_CLAUSE)
        # Every cap lost says so once, as before
        assert dashboard._run_settings_html(
            sweep, cap_sweep_unused=True, tier_off_cap_sweep_unused=True).endswith(
            _KC_UNUSED_CLAUSE)
        # Healthy, or without a tier-off cap sweep: the line is unchanged
        healthy = dashboard._run_settings_html(sweep)
        assert healthy.endswith("per-trade cap: 20% (size-cap sweep on)</p>")
        without = dataclasses.replace(sweep, tier_off_cap_sweep=None)
        assert dashboard._run_settings_html(without, tier_off_cap_sweep_unused=True) == healthy


# ─── Add to held pairs (C6) ──────────────────────────────────────────────────

_ADD_WARNING = ("An Add to held pairs simulation failed; the filter bar's Add to held pairs "
                "select stays disabled")
_ADD_MISMATCH = "The Add to held pairs simulations do not match the page's grid"
_ADD_MISMATCH_OFF = ("The Add to held pairs simulations with the tier floors off do not match "
                     "the page's grid; with the tier floors off, adding is not shown")
_ADD_UNREAD = ("The Add to held pairs simulations could not be read; the filter bar's Add to "
               "held pairs select stays disabled")
_ADD_NO_BAND = ("The Add to held pairs simulations cannot be placed on a run whose primary "
                "records no spread band; the filter bar's Add to held pairs select stays "
                "disabled")
_ADD_LOST_WITH_CAP_SWEEP = ("The Add to held pairs simulations are not shown, because the "
                            "size-cap sweep they are read with could not be used; the filter "
                            "bar's Add to held pairs select stays disabled")
_ADD_NOT_SIMULATED = "(not simulated in this backtest)"
_ADD_UNAVAILABLE = "(not available on this page; see the log)"


def _ao_trade(t: BacktestTrade, n: int = 2, event: str | None = None) -> BacktestTrade:
    """A trade that adds to a pair the run still holds: t at n contracts, marked
    add_on (and, with event, filed under that event and its own ticker)."""
    added = dataclasses.replace(_kc_resized(t, n), add_on=True)
    if event is not None:
        added = dataclasses.replace(added, event_ticker=event, ticker_a=f"{event}-A")
    return added


def _kc_sweep_add_on(base: BacktestSweep | None = None, *, raise_on=None, off_raise_on=None,
                     event: str | None = None) -> BacktestSweep:
    """
    A size-cap sweep (_kc_sweep() unless `base`) plus the Add to held pairs family.

    The family is a fake CapSweep over the base grid's own bands, ks and caps:
    at k 0.60 no pair is held twice, so its points are the base points' own
    lists (its chunks are the base ones); at k 0.75 every list gains one add-on
    trade — the caps at or above the peak sharing one list, as CapSweep shares
    them; and (0.3-0.6, 0.60) is never simulated, as in the base. With a
    tier-floors-off family on `base` (_kc_sweep_tiers()), a tier-floors-off twin
    over its one binding band holds that family's list plus an add-on trade.
    `raise_on` and `off_raise_on` make one cell of the family's tier-on or tier-off
    sweep raise when it is read; `event` files the (0.3-0.6, 0.75) add-on trade
    under another event (a series no other scenario trades).
    """
    sweep = base or _kc_sweep()
    # The base sweep's points: one "all" point per cell (a size-cap sweep with
    # the band sweep's checks holds a dict of populations there)
    points = {key: (p["all"] if isinstance(p, dict) else p)
              for key, p in sweep.cap_sweep.points.items()}
    curves: dict = {}

    def added(band, k, cap, trades, *, tier_floors=True) -> SweepPoint:
        if id(trades) not in curves:
            curves[id(trades)] = backtester._build_equity_curve(trades, _FLT_START, 1000.0)
        return SweepPoint(k=k, trades=trades, equity_df=curves[id(trades)], spread_band=band,
                          size_cap=cap, tier_floors=tier_floors, add_to_held=True)

    b0 = [points[(_KC_B0, 0.75, cap)].trades for cap in (0.05, 0.2)]
    small = [*b0[0], _ao_trade(b0[0][0])]
    full = [*b0[1], _ao_trade(b0[1][0])]
    b1 = points[(_KC_B1, 0.75, 0.2)].trades
    other = [*b1, _ao_trade(b1[0], event=event)]
    family = {}
    for cap in (0.05, 0.2, 1.0):
        family[(_KC_B0, 0.6, cap)] = dataclasses.replace(points[(_KC_B0, 0.6, cap)],
                                                         add_to_held=True)
        family[(_KC_B0, 0.75, cap)] = added(_KC_B0, 0.75, cap, small if cap == 0.05 else full)
        family[(_KC_B1, 0.75, cap)] = added(_KC_B1, 0.75, cap, other)
    twin = None
    if sweep.tier_off_scenarios:
        off = sweep.tier_off_scenarios[0].trades
        off_added = [*off, _ao_trade(off[0])]
        twin = _FakeCapSweep(
            {(_KC_B0, k, cap): added(_KC_B0, k, cap, off_added, tier_floors=False)
             for k in (0.6, 0.75) for cap in (0.05, 0.2, 1.0)},
            bands=(_KC_B0,), raise_on=off_raise_on)
    return dataclasses.replace(sweep, add_on_cap_sweep=_FakeCapSweep(family, raise_on=raise_on),
                               add_on_tier_off_cap_sweep=twin)


def _ao_page(monkeypatch, tmp_path, sweep, series=_FLT_SERIES_TIERS) -> str:
    """A whole page of a sweep, its trades filed by `series`."""
    return TestFilterPage()._page(monkeypatch, tmp_path, sweep, series)


def _ao_payload(sweep: BacktestSweep, state: str = "not simulated") -> tuple:
    """(the grid walked, the chunk visitor, the base block) as generate_dashboard
    builds them for a sweep, with the add-on state the page would pass."""
    trades, curve = sweep.primary.trades, sweep.primary.equity_df
    source = dashboard._grid_source(sweep, trades, curve, 0.75)
    walked, chunker, kd = dashboard._build_filter_grid(
        source, trades, curve, 0.75, _FLT_START, 1000.0, _FLT_SERIES_TIERS)
    base = dashboard._filter_payload(walked, chunker, _FLT_START, 1000.0, _FLT_SERIES_TIERS,
                                     kd=kd.payload(), add_on_state=state)
    return walked, chunker, base


def _ao_ids(grid) -> set[int]:
    """Every chunk id a [band][k][cap] grid names."""
    return {c for band in grid for ks in band for c in ks if c is not None}


def _ao_chunks(chunker) -> list:
    """The chunks a chunk visitor packed, inflated, in id order."""
    return [_unpack(block) for block in chunker.chunks]


class TestAddOnView:
    """The bar's "Add to held pairs" view: a lazy family of add-on runs joins the
    page's one walk, every scenario of it is its own run (or the run as
    simulated, when adding traded nothing), and a family that does not fit, or
    cannot be simulated, costs that choice alone."""

    def test_the_family_joins_a_grid_it_fits(self):
        sweep = _kc_sweep_add_on()
        source = dashboard._grid_source(sweep, sweep.primary.trades, sweep.primary.equity_df,
                                        0.75)
        family = sweep.add_on_cap_sweep
        assert source.add_cell == family.cell and source.add_cap_sweep is family
        assert source.add_off_cell is None and source.add_off_cap_sweep is None
        # The categories and tags of what it can trade are known before any
        # cell is read, and attaching it simulates nothing
        assert source.events >= frozenset(family.entry_events())
        assert family.reads == []

    def test_every_add_on_scenario_is_its_own_chunk_or_the_base_one(self):
        sweep = _kc_sweep_add_on()
        _, source, base, chunks = _flt_payload(sweep)
        assert base["grid"] == _KC_GRID
        # k 0.60: nothing to add to, so its lists are the run as simulated
        # and its chunks the base ones; k 0.75 gains a trade, so a list of its
        # own — the caps above the peak sharing one; (0.3-0.6, 0.60) was never
        # simulated
        assert base["grid_add"] == [[[3, 2, 2], [6, 5, 5]], [[None, None, None], [7, 7, 7]]]
        assert base["add_state"] == "shown" and base["grid_add_off"] is None
        assert len(chunks) == 8
        assert [chunks[c]["list"]["views"]["all"]["n"] for c in (5, 6, 7)] == [4, 4, 3]
        # Each is priced at its own k, on its own trades — never the page's
        added = sweep.add_on_cap_sweep.points[(_KC_B0, 0.75, 0.2)].trades
        assert chunks[5]["list"]["kx"] == pytest.approx(dashboard._kelly_points(added, 0.75)[0])
        assert len(chunks[5]["list"]["kx"]) == 4 != len(chunks[1]["list"]["kx"])
        # The family's cells were each read once, the page's own first
        assert sweep.add_on_cap_sweep.reads == [(_KC_B0, 0.6), (_KC_B0, 0.75), (_KC_B1, 0.6),
                                                (_KC_B1, 0.75)]

    def test_the_grids_the_page_ships_are_the_base_ones_beside_the_add_on(self):
        # Adding the family changes no chunk or grid the page had without it:
        # the add-on chunks are appended after the base ones
        _, clean_chunker, clean_base = _ao_payload(_kc_sweep())
        _, chunker, base = _ao_payload(_kc_sweep_add_on())
        assert base["grid"] == clean_base["grid"]
        assert base["rows"][:len(clean_base["rows"])] == clean_base["rows"]
        assert _ao_chunks(chunker)[:5] == _ao_chunks(clean_chunker)

    def test_without_the_family_the_payload_says_so(self):
        _, _, base = _ao_payload(_kc_sweep())
        assert base["grid_add"] is None and base["grid_add_off"] is None
        assert base["add_state"] == "not simulated"
        # ... and a page that had one it could not use says that instead
        _, _, base = _ao_payload(_kc_sweep(), "unavailable")
        assert base["grid_add"] is None and base["add_state"] == "unavailable"
        # A page with the view is "shown", whatever state it was handed
        _, _, base = _ao_payload(_kc_sweep_add_on(), "unavailable")
        assert base["add_state"] == "shown" and base["grid_add"] is not None

    def test_a_grid_of_nothing_is_never_shown_as_a_view(self):
        # A family none of whose cells holds a point ships no grid at all
        sweep = dataclasses.replace(_kc_sweep(), add_on_cap_sweep=_FakeCapSweep({}))
        walked, chunker, base = _ao_payload(sweep, "unavailable")
        assert walked.add_cell is not None and chunker.add_grid() is None
        assert base["grid_add"] is None and base["add_state"] == "unavailable"

    def test_the_tier_off_twin_pairs_with_the_tier_off_view_only(self):
        sweep = _kc_sweep_add_on(_kc_sweep_tiers())
        walked, chunker, base = _ao_payload(sweep)
        assert walked.tier_binds == (True, False) and walked.add_off_cap_sweep is not None
        add, off = base["grid_add"], base["grid_add_off"]
        assert base["add_state"] == "shown" and add is not None
        # A binding band's row is its own add-on-with-the-tiers-off chunks ...
        assert off[0] != add[0] and all(c is not None for row in off[0] for c in row)
        assert _ao_ids(off[:1]).isdisjoint(_ao_ids(add))
        # ... and a band the tiers never bind at reads its add-on tier-on row
        assert off[1] == add[1]
        # A tier-off twin with no tier-off view beside it ships no off grid
        _, _, no_view = _ao_payload(dataclasses.replace(
            sweep, tier_off_scenarios=[], tier_off_calibrations_by_band={}))
        assert no_view["grid_add_off"] is None and no_view["grid_add"] is not None

    def test_a_tier_off_view_without_its_twin_reads_missing_only_at_the_binding_band(self):
        sweep = dataclasses.replace(_kc_sweep_add_on(_kc_sweep_tiers()),
                                    add_on_tier_off_cap_sweep=None)
        walked, chunker, base = _ao_payload(sweep)
        assert walked.add_off_cell is None and base["add_state"] == "shown"
        off = base["grid_add_off"]
        assert off[0] == [[None] * 3, [None] * 3] and off[1] == base["grid_add"][1]

    # ─── A family that does not fit ───────────────────────────────────────────

    @pytest.mark.parametrize("axis, value", [("bands", (_KC_B0,)), ("ks", (0.75,)),
                                             ("caps", (0.05, 0.2))])
    def test_a_family_on_other_axes_gives_no_view(
            self, monkeypatch, tmp_path, caplog, axis, value):
        sweep = _kc_sweep_add_on()
        setattr(sweep.add_on_cap_sweep, axis, value)
        with caplog.at_level(logging.WARNING):
            source = dashboard._grid_source(sweep, sweep.primary.trades,
                                            sweep.primary.equity_df, 0.75)
        assert source.add_cell is None and source.add_cap_sweep is None
        assert [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
                and "Add to held" in r.getMessage()] == [
                    f"{_ADD_MISMATCH} (its bands, ks or caps); the filter bar's Add to held "
                    "pairs select stays disabled"]
        # The page keeps every other choice and says why the select is shut
        page = _ao_page(monkeypatch, tmp_path, sweep)
        data = TestFilterPage._data(page)
        assert data["grid_add"] is None and data["add_state"] == "unavailable"
        assert 'id="flt-add-note"' in page and 'id="flt-bar"' in page
        assert data["grid"] == _KC_GRID
        assert sweep.add_on_cap_sweep.reads == []

    def test_a_family_that_cannot_be_read_gives_no_view(self, caplog):
        sweep = _kc_sweep_add_on()

        def broken():
            raise ZeroDivisionError("entry events")
        sweep.add_on_cap_sweep.entry_events = broken
        with caplog.at_level(logging.WARNING):
            source = dashboard._grid_source(sweep, sweep.primary.trades,
                                            sweep.primary.equity_df, 0.75)
        assert source.add_cell is None and source.cap_sweep is sweep.cap_sweep
        [warned] = [r for r in caplog.records if "Add to held" in r.getMessage()]
        assert warned.exc_info is not None and warned.getMessage() == _ADD_UNREAD

    def test_a_tier_off_twin_on_other_axes_costs_the_tier_off_choice_only(self, caplog):
        sweep = _kc_sweep_add_on(_kc_sweep_tiers())
        sweep.add_on_tier_off_cap_sweep.caps = (0.05, 0.2)
        with caplog.at_level(logging.WARNING):
            walked, _, base = _ao_payload(sweep)
        assert [r.getMessage() for r in caplog.records if "Add to held" in r.getMessage()] == [
            _ADD_MISMATCH_OFF]
        assert walked.add_cell is not None and walked.add_off_cell is None
        assert base["add_state"] == "shown" and base["grid_add"] is not None
        # With the tiers off, adding reads "missing" at the band they bind at
        assert base["grid_add_off"][0] == [[None] * 3, [None] * 3]

    def test_the_cap_axis_fallback_grid_has_no_view_and_no_warning(self, caplog):
        sweep = _kc_sweep_add_on()
        with caplog.at_level(logging.WARNING):
            fallback = dashboard._grid_source(sweep, sweep.primary.trades,
                                              sweep.primary.equity_df, 0.75,
                                              use_cap_sweep=False)
        assert fallback.add_cell is None and fallback.cap_sweep is None
        assert not [r for r in caplog.records if "Add to held" in r.getMessage()]
        assert sweep.add_on_cap_sweep.reads == []

    def test_a_lost_cap_axis_costs_the_add_on_view_too(self, monkeypatch, tmp_path, caplog):
        # The size-cap sweep's last cell raises: the walk falls back to the
        # eager grid, which never carries the family (its caps are the size-cap
        # grid's); the page says the choice could not be built, and no add-on
        # cell was ever simulated
        sweep = _kc_sweep_add_on(_kc_sweep(raise_on=(_KC_B1, 0.75)))
        with caplog.at_level(logging.WARNING):
            page = _ao_page(monkeypatch, tmp_path, sweep)
        data = TestFilterPage._data(page)
        assert data["grid_add"] is None and data["add_state"] == "unavailable"
        assert _ADD_UNAVAILABLE in page and 'id="flt-bar"' in page
        assert sweep.add_on_cap_sweep.reads == []
        assert sum("size-cap sweep could not be simulated" in r.getMessage()
                   for r in caplog.records) == 1
        # ... and the log says what that cost the select, once
        assert [r.getMessage() for r in caplog.records
                if "Add to held" in r.getMessage()] == [_ADD_LOST_WITH_CAP_SWEEP]

    def test_a_run_without_a_size_cap_sweep_offers_the_view_at_its_own_cap(self):
        # The eager grid's caps are the run's own cap alone, and so are the
        # family's: it fits, and only the cells it simulated hold a chunk
        sweep = _flt_sweep()
        primary = sweep.primary
        added = dataclasses.replace(
            primary, trades=[*primary.trades, _ao_trade(primary.trades[0])], add_to_held=True)
        family = _FakeCapSweep({((0.0, 1.0), 0.75, 0.2): added},
                               bands=((0.0, 1.0), (0.3, 0.6), (0.3, 1.0)),
                               ks=(0.6, 0.75, 1.0), caps=(0.2,))
        _, source, base, chunks = _flt_payload(dataclasses.replace(
            sweep, add_on_cap_sweep=family))
        assert source.cap_sweep is None and source.add_cell == family.cell
        assert base["add_state"] == "shown"
        assert base["grid_add"] == [[[None], [4], [None]], [[None]] * 3, [[None]] * 3]
        assert chunks[4]["list"]["views"]["all"]["n"] == len(primary.trades) + 1

    def test_a_family_on_a_run_without_a_band_is_set_aside_and_said(
            self, monkeypatch, tmp_path, caplog):
        # A hand-built sweep whose primary records no band cannot place the
        # family's cells on a band axis either: its own WARNING (beside the
        # size-cap sweep's), a shut select and no simulation
        sweep = _kc_sweep_add_on()
        primary = dataclasses.replace(sweep.primary, spread_band=None)
        sweep = dataclasses.replace(sweep, primary=primary, points=[primary], scenarios=[])
        with caplog.at_level(logging.WARNING):
            page = _ao_page(monkeypatch, tmp_path, sweep)
        assert [r.getMessage() for r in caplog.records
                if "Add to held" in r.getMessage()] == [_ADD_NO_BAND]
        data = TestFilterPage._data(page)
        assert data["grid_add"] is None and data["add_state"] == "unavailable"
        assert _ADD_UNAVAILABLE in page
        assert sweep.add_on_cap_sweep.reads == []

    @pytest.mark.parametrize("how", ["unreadable", "without the primary"])
    def test_a_size_cap_sweep_set_aside_says_what_it_costs_the_add_on_view(
            self, monkeypatch, tmp_path, caplog, how):
        # The size-cap sweep the family is read with cannot be used (its events
        # cannot be read, or it holds no scenario of the run's primary): its own
        # WARNING names that, and one more says the select stays shut — never a
        # complaint that the family's grid does not match the eager one
        sweep = _kc_sweep_add_on()
        if how == "unreadable":
            def broken():
                raise ZeroDivisionError("entry events")
            sweep.cap_sweep.entry_events = broken
        else:
            sweep.cap_sweep.bands = (_KC_B1,)
        with caplog.at_level(logging.WARNING):
            source = dashboard._grid_source(sweep, sweep.primary.trades,
                                            sweep.primary.equity_df, 0.75)
        assert source.cap_sweep is None and source.add_cell is None
        assert [r.getMessage() for r in caplog.records
                if "Add to held" in r.getMessage()] == [_ADD_LOST_WITH_CAP_SWEEP]
        assert sweep.add_on_cap_sweep.reads == []
        # ... and the page says the choice is not available, pointing at the log
        page = _ao_page(monkeypatch, tmp_path, sweep)
        data = TestFilterPage._data(page)
        assert data["grid_add"] is None and data["add_state"] == "unavailable"
        assert _ADD_UNAVAILABLE in page

    def test_a_run_without_the_family_says_nothing_when_the_cap_axis_is_lost(
            self, monkeypatch, tmp_path, caplog):
        with caplog.at_level(logging.WARNING):
            _ao_page(monkeypatch, tmp_path, _kc_sweep(raise_on=(_KC_B1, 0.75)))
        assert any("size-cap sweep could not be simulated" in r.getMessage()
                   for r in caplog.records)
        assert not [r for r in caplog.records if "Add to held" in r.getMessage()]

    # ─── A cell that cannot be simulated ──────────────────────────────────────

    def test_an_add_cell_that_raises_costs_only_the_add_on_view(
            self, monkeypatch, tmp_path, caplog):
        # The LAST add-on cell raises, after the earlier ones were packed: every
        # add-on chunk and row head is dropped, and the page is the page it was
        # without the family — chunk for chunk — beside the choice's own note
        clean = _ao_page(monkeypatch, tmp_path, _kc_sweep())
        sweep = _kc_sweep_add_on(raise_on=(_KC_B1, 0.75))
        with caplog.at_level(logging.WARNING):
            page = _ao_page(monkeypatch, tmp_path, sweep)
        warned = [r for r in caplog.records if r.levelno == logging.WARNING
                  and _ADD_WARNING in r.getMessage()]
        assert len(warned) == 1 and warned[0].exc_info is not None
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        clean_data, clean_chunks = TestFilterPage._data(clean), TestFilterPage._chunks(clean)
        assert data["grid_add"] is None and data["grid_add_off"] is None
        assert data["add_state"] == "unavailable"
        assert data["grid"] == clean_data["grid"] == _KC_GRID
        assert data["rows"] == clean_data["rows"] and chunks == clean_chunks
        assert data["text"] == clean_data["text"] and data["kd"] == clean_data["kd"]
        assert 'id="flt-bar"' in page and _ADD_UNAVAILABLE in page
        assert sweep.add_on_cap_sweep.reads == [
            (_KC_B0, 0.6), (_KC_B0, 0.75), (_KC_B1, 0.6), (_KC_B1, 0.75)]

    def test_a_tier_off_add_cell_that_raises_costs_only_the_add_on_view(
            self, monkeypatch, tmp_path, caplog):
        # The tier-floors-off twin's cell raises after the whole tier-on family
        # was packed: both add-on grids go, the tier-off view stays
        clean = _ao_page(monkeypatch, tmp_path, _kc_sweep_tiers())
        sweep = _kc_sweep_add_on(_kc_sweep_tiers(), off_raise_on=(_KC_B0, 0.75))
        with caplog.at_level(logging.WARNING):
            page = _ao_page(monkeypatch, tmp_path, sweep)
        assert len([r for r in caplog.records if _ADD_WARNING in r.getMessage()]) == 1
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        clean_data, clean_chunks = TestFilterPage._data(clean), TestFilterPage._chunks(clean)
        assert data["grid_add"] is data["grid_add_off"] is None
        assert data["add_state"] == "unavailable"
        assert data["grid"] == clean_data["grid"] and data["grid_off"] == clean_data["grid_off"]
        assert data["grid_off"] is not None
        assert data["rows"] == clean_data["rows"] and chunks == clean_chunks

    def test_a_failing_tier_off_size_cap_cell_keeps_the_add_on_view(
            self, monkeypatch, tmp_path, caplog):
        # The tier-floors-off SIZE-CAP sweep cannot simulate its cell: the walk
        # swaps in the grid's eager tier-off lookup, which was built before the
        # add-on family was attached — and must carry it, or the add-on phase
        # would be skipped and no view shipped
        sweep = _kc_sweep_add_on(_kc_sweep_tiers_capped(raise_on=(_KC_B0, 0.6)))
        with caplog.at_level(logging.WARNING):
            walked, chunker, base = _ao_payload(sweep)
        assert any(_OFF_WARNING in r.getMessage() for r in caplog.records)
        assert not any(_ADD_WARNING in r.getMessage() for r in caplog.records)
        assert walked.off_cap_sweep is None            # the eager lookup was walked
        assert walked.add_cell is not None and walked.add_off_cell is not None
        assert base["add_state"] == "shown"
        assert base["grid_add"] is not None and base["grid_add_off"] is not None
        assert base["grid_off"] is not None

    def test_a_walk_over_a_grid_with_a_missing_family_walks_nothing_extra(self, caplog):
        sweep = _kc_sweep_add_on()
        source = dashboard._grid_source(sweep, sweep.primary.trades, sweep.primary.equity_df,
                                        0.75)
        stripped = dataclasses.replace(source, add_cell=None, add_off_cell=None,
                                       add_cap_sweep=None, add_off_cap_sweep=None)
        with caplog.at_level(logging.WARNING):
            _, chunker, _ = dashboard._build_filter_grid(
                stripped, sweep.primary.trades, sweep.primary.equity_df, 0.75, _FLT_START,
                1000.0, _FLT_SERIES_TIERS)
        assert sweep.add_on_cap_sweep.reads == []
        # No add-on phase ran: nothing was warned or packed for it
        assert not [r for r in caplog.records if "Add to held" in r.getMessage()]
        assert chunker.grid_add is None

    # ─── What it never reaches ────────────────────────────────────────────────

    def test_only_the_bar_s_visitors_take_the_add_on_cells(self):
        assert hasattr(dashboard._ChunkVisitor, "add")
        assert not hasattr(dashboard._ExplorerVisitor, "add")
        assert not hasattr(dashboard._KdVisitor, "add")

    def test_the_explorer_and_the_interval_discount_are_the_same_with_the_family(
            self, monkeypatch, tmp_path):
        # The scenario explorer's blocks and the interval-discount section's
        # data are those of the page without the family: neither follows the choice
        clean = _ao_page(monkeypatch, tmp_path, _ex_sweep_tiers())
        page = _ao_page(monkeypatch, tmp_path, _kc_sweep_add_on(_ex_sweep_tiers()))
        assert _scn_blocks(_ex_section(page)) == _scn_blocks(_ex_section(clean))
        assert len(_scn_blocks(_ex_section(page))) > 3
        assert TestFilterPage._data(page)["kd"] == TestFilterPage._data(clean)["kd"]
        # ... and so are the k-hat figures, which are measured before sizing
        assert TestFilterPage._data(page)["khat"] == TestFilterPage._data(clean)["khat"]
        assert TestFilterPage._data(page)["khat_off"] == TestFilterPage._data(clean)["khat_off"]

    def test_the_bar_lists_a_series_only_an_add_on_run_traded(self, monkeypatch, tmp_path):
        # Kelly on a bigger position can trade a pair no other scenario traded;
        # the bar must list its category and tag before any cell is simulated,
        # or that chunk fails and costs the whole bar
        sweep = _kc_sweep_add_on(event="KXRAIN-1")
        page = _ao_page(monkeypatch, tmp_path, sweep)
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        assert 'id="flt-bar"' in page and data["grid_add"] is not None
        assert "Climate" in data["categories"]
        # Its chunk holds that category's view
        climate = _key(data, "Climate")
        assert chunks[data["grid_add"][1][1][1]]["list"]["views"][climate]["n"] == 1
        # The page without the family never offers it
        clean = TestFilterPage._data(_ao_page(monkeypatch, tmp_path, _kc_sweep()))
        assert "Climate" not in clean["categories"]

    # ─── The bar, the words ───────────────────────────────────────────────────

    def test_the_words_are_short_and_plain(self):
        assert dashboard._ADD_ON_OPTION_OFF == "off"
        assert dashboard._ADD_ON_OPTION_ON == "on (up to the size cap)"
        assert dashboard._ADD_ON_NOTES == {"not simulated": _ADD_NOT_SIMULATED,
                                           "unavailable": _ADD_UNAVAILABLE}
        assert dashboard._ADD_ON_PHRASE == ", adding to held pairs"
        assert dashboard._SUMMARY_TEMPLATES["add_on"] == dashboard._ADD_ON_PHRASE
        # The detail lives in the tooltip
        assert "Live trading follows the saved live defaults" in dashboard._ADD_ON_SELECT_TITLE
        assert "sits below the 1:1 line" in dashboard._ADD_ON_SELECT_TITLE
        # The Tier floors note keeps its own words, which a test finds by them
        assert "not simulated for this run" not in " ".join(dashboard._ADD_ON_NOTES.values())

    def test_the_select_follows_the_size_cap_and_is_rendered_shut(self, monkeypatch, tmp_path):
        page = _ao_page(monkeypatch, tmp_path, _kc_sweep_add_on())
        bar = page[page.index('id="flt-bar"'):page.index('id="flt-summary"')]
        assert bar.index('id="flt-cap"') < bar.index('id="flt-add"') < bar.index('id="flt-cat"')
        assert "<label>Add to held pairs: <select" in bar
        select = _page_elements(page)["selects"]["flt-add"]
        assert select == {"disabled": True, "options": [
            {"value": "off", "text": "off", "selected": True},
            {"value": "on", "text": "on (up to the size cap)", "selected": False}]}
        title = re.search(r'<select id="flt-add"[^>]*title="([^"]*)"', bar).group(1)
        assert html.unescape(title) == dashboard._ADD_ON_SELECT_TITLE
        # A page with the view has no note beside it
        assert 'id="flt-add-note"' not in page

    @pytest.mark.parametrize("state, note", [
        ("not simulated", _ADD_NOT_SIMULATED),
        ("unavailable", _ADD_UNAVAILABLE),
        ("something else", _ADD_NOT_SIMULATED)])
    def test_a_shut_select_says_why(self, state, note):
        _, chunker, base = _ao_payload(_kc_sweep())
        bar = dashboard._filter_bar_html({**base, "add_state": state}, {"all": {"n": 3}})
        assert f'id="flt-add-note" style="color:#9E9E9E; font-size:13px;">{note}</span>' in bar
        assert 'id="flt-add" disabled' in bar
        shown = dashboard._filter_bar_html({**base, "add_state": "shown"}, {"all": {"n": 3}})
        assert 'id="flt-add-note"' not in shown

    def test_the_tier_floors_note_keeps_its_own_words(self, monkeypatch, tmp_path):
        page = _ao_page(monkeypatch, tmp_path, _kc_sweep_add_on())
        assert re.search(r'id="flt-tier-note"[^>]*>\(not simulated for this run\)</span>', page)
        assert 'id="flt-add-note"' not in page
        bare = _ao_page(monkeypatch, tmp_path, _kc_sweep())
        assert re.search(r'id="flt-add-note"[^>]*>\(not simulated in this backtest\)</span>',
                         bare)

    def test_the_reach_sentence_names_the_page_it_is_on(self, monkeypatch, tmp_path):
        # Without an explorer grid (the payload built here has none) ...
        _, _, base = _ao_payload(_kc_sweep_add_on())
        assert base["text"]["unfiltered"].endswith(" " + dashboard._ADD_ON_REACH_NO_EXPLORER)
        assert dashboard._ADD_ON_REACH not in base["text"]["unfiltered"]
        # ... with one (a band sweep's populations, so the page carries its grid) ...
        page = _ao_page(monkeypatch, tmp_path, _kc_sweep_add_on(_ex_sweep()))
        data = TestFilterPage._data(page)
        assert data["text"]["unfiltered"].endswith(" " + dashboard._ADD_ON_REACH)
        assert html.escape(dashboard._ADD_ON_REACH) in page
        # ... and a page without the view never claims it
        _, _, clean = _ao_payload(_kc_sweep())
        assert "Add to held pairs" not in clean["text"]["unfiltered"]
        assert "Add to held pairs" not in TestFilterPage._data(
            _ao_page(monkeypatch, tmp_path, _kc_sweep()))["text"]["unfiltered"]

    def test_the_save_button_names_the_choice(self):
        assert "Add to held pairs choice (when this page simulated it)" in dashboard._SAVE_TITLE

    def test_every_chunk_the_add_on_grids_name_is_on_the_page(self, monkeypatch, tmp_path):
        # The chunk-id check of TestFilterPage's, over the add-on grids too
        page = _ao_page(monkeypatch, tmp_path, _kc_sweep_add_on(_kc_sweep_tiers()))
        data = TestFilterPage._data(page)
        ids = (_ao_ids(data["grid"]) | _ao_ids(data["grid_off"]) | _ao_ids(data["grid_add"])
               | _ao_ids(data["grid_add_off"]))
        assert _ao_ids(data["grid_add"]) - _ao_ids(data["grid"])
        assert _ao_ids(data["grid_add_off"]) - _ao_ids(data["grid_add"])
        missing = sorted(i for i in ids if f'id="dash-chunk-{i}"' not in page)
        assert ids and not missing
        # Every chunk on the page is named by some grid: none is left over
        assert set(TestFilterPage._chunks(page)) == ids

    # ─── The header line ──────────────────────────────────────────────────────

    def test_the_header_promises_a_view_only_where_the_bar_holds_one(self):
        run = TestLiveRuleHeader
        promise = " — choose Add to held pairs on in the filter bar to see it"
        nothing = " — this page has no Add to held pairs view"
        # Tiers on, at the primary band: its row of the add-on grid must hold a chunk
        sweep = dataclasses.replace(run._sweep(), live_add_to_held_pairs=True)
        for grid, tail in (([[[7]]], promise), ([[[None]]], nothing)):
            assert run()._text(sweep, {**run._bar(), "grid_add": grid}).endswith(tail)
        # Tiers off, at a band they bind at: the row of the tier-floors-off twin,
        # not the tier-on row beside it
        band = (0.0, 0.5)
        off_sweep = dataclasses.replace(
            run._sweep(live_tier_floors=False, live_spread_band=band,
                       calibrations_by_band={(0.0, 1.0): None, band: None},
                       tier_off_calibrations_by_band={(0.0, 1.0): None, band: None},
                       tier_off_scenarios=[object()]),
            live_add_to_held_pairs=True)
        bar = {**run._bar(bands=[band, (0.0, 1.0)]), "grid_add": [[[3]], [[4]]]}
        for twin, tail in (([[[7]], [[8]]], promise), ([[[None]], [[8]]], nothing),
                           (None, nothing)):
            assert run()._text(off_sweep, {**bar, "grid_add_off": twin}).endswith(tail)

    def test_the_save_button_warns_that_saving_with_adding_off_turns_it_off(
            self, monkeypatch, tmp_path):
        note = ("The live defaults add to held pairs; saving with Add to held pairs off "
                "here turns that off.")
        assert dashboard._ADD_ON_SAVE_NOTE == note

        def between(page):
            bar = page[page.index('id="flt-bar"'):page.index('id="flt-summary"')]
            return bar[bar.index('id="flt-save"'):bar.index('id="flt-trade"')]

        with_view = _kc_sweep_add_on()
        page = _ao_page(monkeypatch, tmp_path, dataclasses.replace(
            with_view, live_add_to_held_pairs=True))
        # Right after the button, before the trade link, in Python's words
        assert re.search(r'</button>&nbsp;<span id="flt-add-save-note"[^>]*>' + re.escape(note)
                         + '</span>', between(page))
        # No note when the saved defaults do not add (or the run recorded none) ...
        for saved in (False, None):
            page = _ao_page(monkeypatch, tmp_path, dataclasses.replace(
                with_view, live_add_to_held_pairs=saved))
            assert 'id="flt-add-save-note"' not in page
        # ... nor on a page with no add-on view, whose Save sends no choice
        page = _ao_page(monkeypatch, tmp_path, dataclasses.replace(
            _kc_sweep(), live_add_to_held_pairs=True))
        assert 'id="flt-add-save-note"' not in page and 'id="flt-save"' in page

    def test_the_live_rule_header_names_adding_where_the_page_shows_the_rule(self):
        run = TestLiveRuleHeader
        rule = config.describe_time_series_rule(True, (0.0, 1.0))
        head = f"Live rule (saved live defaults): {rule}; {run._CAP} — "
        note = "; the live defaults add to held pairs, which this run's primary does not"
        sweep = dataclasses.replace(run._sweep(), live_add_to_held_pairs=True)
        shown = {**run._bar(), "grid_add": [[[1]]]}
        # The rule is the page's primary and the bar has the view: choose it
        assert run()._text(sweep, shown) == (
            head + "this run's primary" + note
            + " — choose Add to held pairs on in the filter bar to see it")
        # ... a bar without the view, or none at all: say so
        for bar in (run._bar(), {**run._bar(), "grid_add": None}, None):
            assert run()._text(sweep, bar) == (
                head + "this run's primary" + note + " — this page has no Add to held pairs view")
        # ... a rule the page does not show: the note alone, no promise of a view
        elsewhere = dataclasses.replace(sweep, live_spread_band=(0.3, 0.6),
                                        calibrations_by_band={(0.0, 1.0): None})
        assert run()._text(elsewhere, shown).endswith("— not simulated by this run" + note)
        # Not adding (or not recorded): no note at all
        for saved in (False, None):
            plain = dataclasses.replace(sweep, live_add_to_held_pairs=saved)
            assert "add to held pairs" not in run()._text(plain, shown)

    def test_the_header_of_a_whole_page_points_at_the_select(self, monkeypatch, tmp_path):
        sweep = dataclasses.replace(_kc_sweep_add_on(), live_add_to_held_pairs=True,
                                    live_tier_floors=True, live_spread_band=_KC_B0)
        page = _ao_page(monkeypatch, tmp_path, sweep)
        assert ("the live defaults add to held pairs, which this run's primary does not — "
                "choose Add to held pairs on in the filter bar to see it</p>") in page
        page = _ao_page(monkeypatch, tmp_path, dataclasses.replace(sweep, add_on_cap_sweep=None))
        assert ("the live defaults add to held pairs, which this run's primary does not — "
                "this page has no Add to held pairs view</p>") in page


_TRIM_MISMATCH = ("The Trim to Kelly simulations do not match the page's grid (its bands, ks "
                  "or caps); the filter bar's Trim to Kelly select stays disabled")
_TRIM_MISMATCH_OFF = ("The Trim to Kelly simulations with the tier floors off do not match the "
                      "page's grid; with the tier floors off, trimming is not shown")
_TRIM_UNREAD = ("The Trim to Kelly simulations could not be read; the filter bar's Trim to "
                "Kelly select stays disabled")
_TRIM_WARNING = "A Trim to Kelly simulation failed"
_TRIM_NO_BAND = ("The Trim to Kelly simulations cannot be placed on a run whose primary records "
                 "no spread band; the filter bar's Trim to Kelly select stays disabled")
_TRIM_LOST_WITH_CAP_SWEEP = ("The Trim to Kelly simulations are not shown, because the size-cap "
                             "sweep they are read with could not be used; the filter bar's Trim "
                             "to Kelly select stays disabled")


class _FakeTrimSweep:
    """
    A backtester.TrimSweep stand-in carrying what the dashboard reads of one:
    bands, off_bands, ks, caps, sweep() and entry_events().

    `sweeps` maps (adding to held pairs, tier floors off) to the _FakeCapSweep
    sweep() hands back for that setting; `asked` records each sweep() call.
    """

    def __init__(self, sweeps: dict, *, bands=(_KC_B0, _KC_B1), off_bands=(),
                 ks=(0.6, 0.75), caps=(0.05, 0.2, 1.0)):
        self.sweeps = sweeps
        self.bands, self.off_bands, self.ks, self.caps = bands, off_bands, ks, caps
        self.asked: list = []

    def sweep(self, *, tier_floors=True, add_to_held=False):
        self.asked.append((add_to_held, not tier_floors))
        return self.sweeps[(add_to_held, not tier_floors)]

    def entry_events(self):
        return {event for capped in self.sweeps.values() for event in capped.entry_events()}


def _tr_split(trades: list, event: str | None = None) -> list:
    """A list whose first pair was trimmed: that trade as two records of one
    contract pair each (the part sold and the part kept), then the rest."""
    part = _kc_resized(trades[0], 1)
    if event is not None:
        part = dataclasses.replace(part, event_ticker=event, ticker_a=f"{event}-A")
    return [part, _kc_resized(trades[0], 1), *trades[1:]]


def _kc_sweep_trim(base: BacktestSweep | None = None, *, raise_on: dict | None = None,
                   event: str | None = None) -> BacktestSweep:
    """
    A size-cap sweep (_kc_sweep() unless `base`) plus the Trim to Kelly family.

    Trimming alone, tier floors on: at k 0.60 nothing is trimmed, so each
    point holds the base point's own list; at k 0.75 the primary band's 5%
    and 20% caps each trim their first pair (a list of their own) while no
    cap trims nothing, and the other band trims nothing at any cap. With an
    Add to held pairs family on `base`, the adding-on half does the same over
    that family's lists. With a tier-floors-off family on `base`, the binding
    band's twins trim that family's list at every cap. `raise_on` maps a
    setting (adding, tier floors off) to the cell whose read raises; `event`
    files the 20%-cap part sold under another event.
    """
    sweep = base or _kc_sweep()
    raise_on = raise_on or {}
    curves: dict = {}

    def stamped(point, trades=None, **more) -> SweepPoint:
        if trades is None:
            return dataclasses.replace(point, trim_to_kelly=True, **more)
        if id(trades) not in curves:
            curves[id(trades)] = backtester._build_equity_curve(trades, _FLT_START, 1000.0)
        return dataclasses.replace(point, trades=trades, equity_df=curves[id(trades)],
                                   trim_to_kelly=True, **more)

    def half(points: dict, add: bool) -> _FakeCapSweep:
        small = _tr_split(points[(_KC_B0, 0.75, 0.05)].trades)
        full = _tr_split(points[(_KC_B0, 0.75, 0.2)].trades, event)
        family = {}
        for cap in (0.05, 0.2, 1.0):
            family[(_KC_B0, 0.6, cap)] = stamped(points[(_KC_B0, 0.6, cap)])
            family[(_KC_B1, 0.75, cap)] = stamped(points[(_KC_B1, 0.75, cap)])
        family[(_KC_B0, 0.75, 0.05)] = stamped(points[(_KC_B0, 0.75, 0.05)], small)
        family[(_KC_B0, 0.75, 0.2)] = stamped(points[(_KC_B0, 0.75, 0.2)], full)
        family[(_KC_B0, 0.75, 1.0)] = stamped(points[(_KC_B0, 0.75, 1.0)])
        return _FakeCapSweep(family, raise_on=raise_on.get((add, False)))

    own = {key: (p["all"] if isinstance(p, dict) else p)
           for key, p in sweep.cap_sweep.points.items()}
    sweeps = {(False, False): half(own, False)}
    adding = getattr(sweep, "add_on_cap_sweep", None)
    sweeps[(True, False)] = half(adding.points if adding is not None else own, True)
    off_bands = ()
    if sweep.tier_off_scenarios:
        off_bands = (_KC_B0,)
        off = sweep.tier_off_scenarios[0]
        for add in (False, True):
            listed = _tr_split([*off.trades, _ao_trade(off.trades[0])] if add else off.trades)
            sweeps[(add, True)] = _FakeCapSweep(
                {(_KC_B0, k, cap): stamped(off, listed, k=k, size_cap=cap, add_to_held=add)
                 for k in (0.6, 0.75) for cap in (0.05, 0.2, 1.0)},
                bands=(_KC_B0,), raise_on=raise_on.get((add, True)))
    return dataclasses.replace(sweep, trim_sweep=_FakeTrimSweep(sweeps, off_bands=off_bands))


def _tr_payload(sweep: BacktestSweep, state: str = "not simulated") -> tuple:
    """(the grid walked, the chunk visitor, the base block) as generate_dashboard
    builds them for a sweep, with the trim state the page would pass."""
    trades, curve = sweep.primary.trades, sweep.primary.equity_df
    source = dashboard._grid_source(sweep, trades, curve, 0.75)
    walked, chunker, kd = dashboard._build_filter_grid(
        source, trades, curve, 0.75, _FLT_START, 1000.0, _FLT_SERIES_TIERS)
    base = dashboard._filter_payload(walked, chunker, _FLT_START, 1000.0, _FLT_SERIES_TIERS,
                                     kd=kd.payload(), trim_state=state)
    return walked, chunker, base


def _tr_warnings(caplog) -> list[str]:
    """Every WARNING that names the Trim to Kelly view."""
    return [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and "Trim to Kelly" in r.getMessage()]


class TestTrimView:
    """The page's Trim to Kelly data: a lazy family of trimming runs joins the
    page's one walk after the add-on cells, every scenario of it is its own
    run (or the scenario's own chunk, when nothing was trimmed), and a family
    that does not fit, or cannot be simulated, costs that view alone."""

    def test_the_family_joins_a_grid_it_fits(self):
        sweep = _kc_sweep_trim()
        source = dashboard._grid_source(sweep, sweep.primary.trades, sweep.primary.equity_df,
                                        0.75)
        family = sweep.trim_sweep
        assert source.trim_cell is not None and source.trim_off_cell is None
        assert set(source.trim_sweeps) == {(False, False), (True, False)}
        assert source.trim_sweeps[(False, False)] is family.sweeps[(False, False)]
        assert source.events >= frozenset(family.entry_events())
        # Attaching it simulates nothing
        assert all(capped.reads == [] for capped in family.sweeps.values())

    def test_an_event_only_a_trimming_run_trades_is_listed_up_front(self):
        # The part sold is filed under an event no other scenario trades: the
        # grid knows it before any cell is read
        clean, sweep = _kc_sweep(), _kc_sweep_trim(event="KXONLYTRIM-1")
        before = dashboard._grid_source(clean, clean.primary.trades, clean.primary.equity_df,
                                        0.75)
        source = dashboard._grid_source(sweep, sweep.primary.trades, sweep.primary.equity_df,
                                        0.75)
        assert {event for event, _ in source.events - before.events} == {"KXONLYTRIM-1"}
        assert all(capped.reads == [] for capped in sweep.trim_sweep.sweeps.values())

    def test_every_trim_scenario_is_its_own_chunk_or_the_scenario_s_own(self):
        sweep = _kc_sweep_trim()
        _, source, base, chunks = _flt_payload(sweep)
        assert base["grid"] == _KC_GRID
        # k 0.60 and the other band trim nothing: the scenario's own chunk. At
        # (0-1, 0.75) the 5% and 20% caps each sold part of a pair, so each has
        # a list of its own; with no cap nothing is trimmed
        assert base["grid_trim"] == [[[3, 2, 2], [6, 5, 0]], [[None, None, None], [4, 4, 4]]]
        assert base["trim_state"] == "shown"
        assert base["grid_trim_off"] is None
        # The page has no Add to held pairs view, so no adding-on half either,
        # and those cells were never simulated
        assert base["grid_trim_add"] is None and base["grid_trim_add_off"] is None
        assert sweep.trim_sweep.sweeps[(True, False)].reads == []
        assert len(chunks) == 7
        assert [chunks[c]["list"]["views"]["all"]["n"] for c in (5, 6)] == [4, 4]
        # Each is priced at its own k, on its own trades
        trimmed = sweep.trim_sweep.sweeps[(False, False)].points[(_KC_B0, 0.75, 0.2)].trades
        assert chunks[5]["list"]["kx"] == pytest.approx(
            dashboard._kelly_points(trimmed, 0.75)[0])
        # Every cell read once, the page's own first
        assert sweep.trim_sweep.sweeps[(False, False)].reads == [
            (_KC_B0, 0.6), (_KC_B0, 0.75), (_KC_B1, 0.6), (_KC_B1, 0.75)]

    def test_the_reach_sentence_says_what_the_choice_changes(self):
        _, _, base = _tr_payload(_kc_sweep_trim())
        assert base["text"]["unfiltered"].endswith(dashboard._TRIM_REACH_NO_EXPLORER)
        assert base["text"]["trim"] == ", trimming to Kelly"
        assert base["text"]["trim_note"] == dashboard._TRIM_SUMMARY_NOTE
        _, _, clean = _tr_payload(_kc_sweep())
        assert "Trim to Kelly" not in clean["text"]["unfiltered"]

    def test_the_page_s_other_grids_and_chunks_are_unchanged_beside_it(self):
        # The trim chunks come after every other chunk, so a page with the
        # family holds the page without it, chunk for chunk, then its own
        _, clean_chunker, clean_base = _ao_payload(_kc_sweep_add_on())
        _, chunker, base = _tr_payload(_kc_sweep_trim(_kc_sweep_add_on()))
        for key in ("grid", "grid_add", "grid_add_off", "add_state"):
            assert base[key] == clean_base[key]
        assert base["rows"][:len(clean_base["rows"])] == clean_base["rows"]
        assert _ao_chunks(chunker)[:len(clean_chunker.chunks)] == _ao_chunks(clean_chunker)
        assert len(chunker.chunks) > len(clean_chunker.chunks)

    def test_with_the_add_on_view_the_adding_half_is_built_too(self):
        sweep = _kc_sweep_trim(_kc_sweep_add_on())
        walked, chunker, base = _tr_payload(sweep)
        add, trim, both = base["grid_add"], base["grid_trim"], base["grid_trim_add"]
        assert base["trim_state"] == "shown" and both is not None
        # Nothing trimmed: the cell is the Add to held pairs scenario's own chunk
        assert both[0][0] == add[0][0] and both[1] == add[1]
        assert both[0][1][2] == add[0][1][2]
        # Trimmed: a list neither the add-on scenario nor the trim-alone one traded
        for ci in (0, 1):
            assert both[0][1][ci] not in _ao_ids(add) | _ao_ids(trim) | _ao_ids(base["grid"])
        assert sweep.trim_sweep.sweeps[(True, False)].reads == [
            (_KC_B0, 0.6), (_KC_B0, 0.75), (_KC_B1, 0.6), (_KC_B1, 0.75)]

    def test_without_the_family_the_payload_says_so(self):
        _, _, base = _tr_payload(_kc_sweep())
        assert [base[key] for key in ("grid_trim", "grid_trim_off", "grid_trim_add",
                                      "grid_trim_add_off")] == [None] * 4
        assert base["trim_state"] == "not simulated"
        _, _, base = _tr_payload(_kc_sweep(), "unavailable")
        assert base["grid_trim"] is None and base["trim_state"] == "unavailable"
        _, _, base = _tr_payload(_kc_sweep_trim(), "unavailable")
        assert base["trim_state"] == "shown" and base["grid_trim"] is not None

    def test_a_grid_of_nothing_is_never_shown_as_a_view(self):
        empty = {key: _FakeCapSweep({}) for key in ((False, False), (True, False))}
        sweep = dataclasses.replace(_kc_sweep(), trim_sweep=_FakeTrimSweep(empty))
        walked, chunker, base = _tr_payload(sweep, "unavailable")
        assert walked.trim_cell is not None and chunker.trim_grid() is None
        assert base["grid_trim"] is None and base["trim_state"] == "unavailable"

    def test_the_tier_off_twins_pair_with_the_tier_off_view_only(self):
        sweep = _kc_sweep_trim(_kc_sweep_add_on(_kc_sweep_tiers()))
        walked, chunker, base = _tr_payload(sweep)
        assert walked.tier_binds == (True, False) and walked.trim_off_cell is not None
        for on_key, off_key in (("grid_trim", "grid_trim_off"),
                                ("grid_trim_add", "grid_trim_add_off")):
            on, off = base[on_key], base[off_key]
            # A binding band's row is its own tier-floors-off chunks ...
            assert all(c is not None for row in off[0] for c in row)
            assert _ao_ids(off[:1]).isdisjoint(_ao_ids(on))
            # ... and a band the tiers never bind at reads its tier-on row
            assert off[1] == on[1]
        assert _ao_ids(base["grid_trim_off"][:1]).isdisjoint(
            _ao_ids(base["grid_trim_add_off"][:1]))
        # Without a tier-off view beside them, no off grid is shipped
        _, _, no_view = _tr_payload(dataclasses.replace(
            sweep, tier_off_scenarios=[], tier_off_calibrations_by_band={},
            add_on_tier_off_cap_sweep=None))
        assert no_view["grid_trim_off"] is None and no_view["grid_trim_add_off"] is None
        assert no_view["grid_trim"] is not None

    def test_trimming_with_adding_needs_the_add_on_runs_with_the_tiers_off_too(self):
        # The page has no add-on run with the tier floors off at the band the
        # tiers bind at, so it shows no trimming-with-adding run there either,
        # and simulates none
        sweep = dataclasses.replace(
            _kc_sweep_trim(_kc_sweep_add_on(_kc_sweep_tiers())), add_on_tier_off_cap_sweep=None)
        _, _, base = _tr_payload(sweep)
        assert base["grid_add_off"][0] == [[None] * 3, [None] * 3]
        assert base["grid_trim_add_off"][0] == [[None] * 3, [None] * 3]
        assert base["grid_trim_add_off"][1] == base["grid_trim_add"][1]
        assert sweep.trim_sweep.sweeps[(True, True)].reads == []
        # Trimming alone with the tier floors off is still shown there
        assert all(c is not None for row in base["grid_trim_off"][0] for c in row)

    def test_a_tier_off_half_that_cannot_be_read_says_so(self, caplog):
        sweep = _kc_sweep_trim(_kc_sweep_tiers())
        real = sweep.trim_sweep.sweep

        def broken(*, tier_floors=True, add_to_held=False):
            if not tier_floors:
                raise ZeroDivisionError("off sweep")
            return real(tier_floors=tier_floors, add_to_held=add_to_held)
        sweep.trim_sweep.sweep = broken
        with caplog.at_level(logging.WARNING):
            walked, _, base = _tr_payload(sweep)
        [warned] = [r for r in caplog.records if "Trim to Kelly" in r.getMessage()]
        assert warned.exc_info is not None and warned.getMessage() == (
            "The Trim to Kelly simulations with the tier floors off could not be read; with "
            "the tier floors off, trimming is not shown")
        assert walked.trim_cell is not None and walked.trim_off_cell is None
        assert base["trim_state"] == "shown" and base["grid_trim"] is not None

    def test_a_chunk_visitor_that_has_failed_simulates_no_trim_cell(self, monkeypatch, caplog):
        # The bar is already lost (a chunk could not be built), so the page is
        # written without it: the family's cells are not simulated for nothing
        sweep = _kc_sweep_trim()
        real = dashboard._ChunkVisitor.__call__

        def failing(self, bi, ki, ci, pops):
            real(self, bi, ki, ci, pops)
            self.failed = True
        monkeypatch.setattr(dashboard._ChunkVisitor, "__call__", failing)
        trades, curve = sweep.primary.trades, sweep.primary.equity_df
        source = dashboard._grid_source(sweep, trades, curve, 0.75)
        dashboard._build_filter_grid(source, trades, curve, 0.75, _FLT_START, 1000.0,
                                     _FLT_SERIES_TIERS)
        assert all(capped.reads == [] for capped in sweep.trim_sweep.sweeps.values())

    def test_a_family_without_the_binding_bands_costs_the_tier_off_half_only(self, caplog):
        sweep = _kc_sweep_trim(_kc_sweep_tiers())
        sweep.trim_sweep.off_bands = ()
        with caplog.at_level(logging.WARNING):
            walked, _, base = _tr_payload(sweep)
        assert _tr_warnings(caplog) == [_TRIM_MISMATCH_OFF]
        assert walked.trim_cell is not None and walked.trim_off_cell is None
        assert base["trim_state"] == "shown"
        # With the tiers off, trimming reads "missing" at the band they bind at
        assert base["grid_trim_off"][0] == [[None] * 3, [None] * 3]
        assert base["grid_trim_off"][1] == base["grid_trim"][1]
        assert sweep.trim_sweep.sweeps[(False, True)].reads == []

    # ─── A family that does not fit ───────────────────────────────────────────

    @pytest.mark.parametrize("axis, value", [("bands", (_KC_B0,)), ("ks", (0.75,)),
                                             ("caps", (0.05, 0.2))])
    def test_a_family_on_other_axes_gives_no_view(self, caplog, axis, value):
        sweep = _kc_sweep_trim()
        setattr(sweep.trim_sweep, axis, value)
        with caplog.at_level(logging.WARNING):
            walked, _, base = _tr_payload(sweep, "unavailable")
        assert _tr_warnings(caplog) == [_TRIM_MISMATCH]
        assert walked.trim_cell is None and walked.trim_sweeps == {}
        assert base["grid_trim"] is None and base["trim_state"] == "unavailable"
        assert base["grid"] == _KC_GRID
        assert sweep.trim_sweep.asked == []

    def test_a_family_that_cannot_be_read_gives_no_view(self, caplog):
        sweep = _kc_sweep_trim()

        def broken():
            raise ZeroDivisionError("entry events")
        sweep.trim_sweep.entry_events = broken
        with caplog.at_level(logging.WARNING):
            source = dashboard._grid_source(sweep, sweep.primary.trades,
                                            sweep.primary.equity_df, 0.75)
        assert source.trim_cell is None and source.cap_sweep is sweep.cap_sweep
        [warned] = [r for r in caplog.records if "Trim to Kelly" in r.getMessage()]
        assert warned.exc_info is not None and warned.getMessage() == _TRIM_UNREAD

    def test_the_cap_axis_fallback_grid_has_no_view_and_no_warning(self, caplog):
        sweep = _kc_sweep_trim()
        with caplog.at_level(logging.WARNING):
            fallback = dashboard._grid_source(sweep, sweep.primary.trades,
                                              sweep.primary.equity_df, 0.75,
                                              use_cap_sweep=False)
        assert fallback.trim_cell is None and fallback.cap_sweep is None
        assert _tr_warnings(caplog) == []
        assert sweep.trim_sweep.asked == []

    def test_a_lost_cap_axis_costs_the_trim_view_too(self, caplog):
        sweep = _kc_sweep_trim(_kc_sweep(raise_on=(_KC_B1, 0.75)))
        with caplog.at_level(logging.WARNING):
            walked, _, base = _tr_payload(sweep, "unavailable")
        assert walked.cap_sweep is None and walked.trim_cell is None
        assert base["grid_trim"] is None and base["trim_state"] == "unavailable"
        assert _tr_warnings(caplog) == [_TRIM_LOST_WITH_CAP_SWEEP]
        assert all(capped.reads == [] for capped in sweep.trim_sweep.sweeps.values())

    @pytest.mark.parametrize("how", ["unreadable", "without the primary"])
    def test_a_size_cap_sweep_set_aside_says_what_it_costs_the_trim_view(self, caplog, how):
        sweep = _kc_sweep_trim()
        if how == "unreadable":
            def broken():
                raise ZeroDivisionError("entry events")
            sweep.cap_sweep.entry_events = broken
        else:
            sweep.cap_sweep.bands = (_KC_B1,)
        with caplog.at_level(logging.WARNING):
            source = dashboard._grid_source(sweep, sweep.primary.trades,
                                            sweep.primary.equity_df, 0.75)
        assert source.cap_sweep is None and source.trim_cell is None
        assert _tr_warnings(caplog) == [_TRIM_LOST_WITH_CAP_SWEEP]
        assert sweep.trim_sweep.asked == []

    def test_a_family_on_a_run_without_a_band_is_set_aside_and_said(self, caplog):
        sweep = _kc_sweep_trim()
        primary = dataclasses.replace(sweep.primary, spread_band=None)
        sweep = dataclasses.replace(sweep, primary=primary, points=[primary], scenarios=[])
        with caplog.at_level(logging.WARNING):
            source = dashboard._grid_source(sweep, sweep.primary.trades,
                                            sweep.primary.equity_df, 0.75)
        assert source.trim_cell is None and _tr_warnings(caplog) == [_TRIM_NO_BAND]
        assert sweep.trim_sweep.asked == []

    def test_a_run_without_a_size_cap_sweep_offers_the_view_at_its_own_cap(self):
        sweep = _flt_sweep()
        primary = sweep.primary
        trimmed = dataclasses.replace(primary, trades=_tr_split(primary.trades),
                                      trim_to_kelly=True)
        axes = {"bands": ((0.0, 1.0), (0.3, 0.6), (0.3, 1.0)), "ks": (0.6, 0.75, 1.0),
                "caps": (0.2,)}
        halves = {(False, False): _FakeCapSweep({((0.0, 1.0), 0.75, 0.2): trimmed}, **axes),
                  (True, False): _FakeCapSweep({}, **axes)}
        _, source, base, chunks = _flt_payload(dataclasses.replace(
            sweep, trim_sweep=_FakeTrimSweep(halves, **axes)))
        assert source.cap_sweep is None and source.trim_cell is not None
        assert base["trim_state"] == "shown"
        assert base["grid_trim"] == [[[None], [4], [None]], [[None]] * 3, [[None]] * 3]
        assert chunks[4]["list"]["views"]["all"]["n"] == len(primary.trades) + 1

    # ─── A cell that cannot be simulated ──────────────────────────────────────

    @pytest.mark.parametrize("setting", [(False, False), (True, False), (False, True),
                                         (True, True)])
    def test_a_trim_cell_that_raises_costs_only_the_trim_view(
            self, monkeypatch, tmp_path, caplog, setting):
        # One cell of one setting raises after earlier trim cells were packed:
        # every trim chunk and row head is dropped, and the page is the page it
        # was without the family, chunk for chunk
        base_sweep = _kc_sweep_add_on(_kc_sweep_tiers())
        clean = _ao_page(monkeypatch, tmp_path, base_sweep)
        sweep = _kc_sweep_trim(_kc_sweep_add_on(_kc_sweep_tiers()),
                               raise_on={setting: (_KC_B0, 0.75)})
        with caplog.at_level(logging.WARNING):
            page = _ao_page(monkeypatch, tmp_path, sweep)
        warned = [r for r in caplog.records if r.levelno == logging.WARNING
                  and _TRIM_WARNING in r.getMessage()]
        assert len(warned) == 1 and warned[0].exc_info is not None
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        clean_data, clean_chunks = TestFilterPage._data(clean), TestFilterPage._chunks(clean)
        assert [data[key] for key in ("grid_trim", "grid_trim_off", "grid_trim_add",
                                      "grid_trim_add_off")] == [None] * 4
        assert data["trim_state"] == "unavailable"
        for key in ("grid", "grid_off", "grid_add", "grid_add_off", "add_state", "rows",
                    "text", "kd"):
            assert data[key] == clean_data[key]
        assert data["grid_add"] is not None and chunks == clean_chunks
        assert 'id="flt-bar"' in page
        # The walk had begun on the family before the cell that raised
        assert sweep.trim_sweep.sweeps[(False, False)].reads[0] == (_KC_B0, 0.6)

    def test_a_failing_add_on_cell_keeps_the_trim_view_without_its_adding_half(self, caplog):
        # The Add to held pairs family cannot be simulated: its view goes, and
        # the trim view is built for adding off alone — the adding-on cells are
        # never read, since the page cannot show adding on
        sweep = _kc_sweep_trim(_kc_sweep_add_on(raise_on=(_KC_B1, 0.75)))
        with caplog.at_level(logging.WARNING):
            walked, chunker, base = _tr_payload(sweep)
        assert any(_ADD_WARNING in r.getMessage() for r in caplog.records)
        assert _tr_warnings(caplog) == []
        assert base["grid_add"] is None and base["trim_state"] == "shown"
        assert base["grid_trim"] is not None and base["grid_trim_add"] is None
        assert sweep.trim_sweep.sweeps[(True, False)].reads == []
        # ... and the trim chunks sit right after the page's own
        assert min(_ao_ids(base["grid_trim"]) - _ao_ids(base["grid"])) == 5

    def test_a_failing_tier_off_size_cap_cell_keeps_the_trim_view(self, caplog):
        # The walk swaps in the grid's eager tier-off lookup, which was built
        # before the family was attached, and must still carry it
        sweep = _kc_sweep_trim(_kc_sweep_tiers_capped(raise_on=(_KC_B0, 0.6)))
        with caplog.at_level(logging.WARNING):
            walked, chunker, base = _tr_payload(sweep)
        assert any(_OFF_WARNING in r.getMessage() for r in caplog.records)
        assert _tr_warnings(caplog) == []
        assert walked.off_cap_sweep is None
        assert walked.trim_cell is not None and walked.trim_off_cell is not None
        assert base["trim_state"] == "shown"
        assert base["grid_trim"] is not None and base["grid_trim_off"] is not None

    def test_the_walk_reads_the_trim_cells_after_every_other_kind(self):
        # The order the families are simulated in is what lets one family's
        # chunks be dropped without cutting into another's
        sweep = _kc_sweep_trim(_kc_sweep_add_on(_kc_sweep_tiers()))
        order: list[str] = []
        named = {"cap": sweep.cap_sweep, "add": sweep.add_on_cap_sweep,
                 "add off": sweep.add_on_tier_off_cap_sweep,
                 **{f"trim {key}": capped for key, capped in sweep.trim_sweep.sweeps.items()}}
        for name, capped in named.items():
            read = capped.cell

            def cell(band, k, read=read, name=name):
                if not order or order[-1] != name:
                    order.append(name)
                return read(band, k)
            capped.cell = cell
        _tr_payload(sweep)
        assert order == ["cap", "add", "add off", "trim (False, False)", "trim (False, True)",
                         "trim (True, False)", "trim (True, True)"]


class TestAddOnScript:
    """The bar's Add to held pairs select under the page script (run outside a
    browser): enabled only with the view, "on" loads the add-on scenario's chunk
    and words it with Python's phrase, a chunk that cannot be loaded puts the
    choice back, and the save address names the choice only where the page
    simulated it. The explorer and the interval-discount section never follow
    it. The harness's "wait" does not wait on flt-add (a page without the view
    keeps it shut), so these tests ask for it: setup_js adds it to the wait
    list. Skipped without a JavaScript runtime."""

    WAIT = "__BAR.push('flt-add');"

    @classmethod
    def _run(cls, tmp_path, page, steps, **kwargs) -> dict:
        """_run_script over a page, waiting on the add-on select too."""
        return _run_script(tmp_path, page, steps, setup_js=cls.WAIT + kwargs.pop("js", ""),
                           **kwargs)

    def test_the_select_is_enabled_only_with_the_view(self, monkeypatch, tmp_path):
        page = _ao_page(monkeypatch, tmp_path, _kc_sweep_add_on())
        snaps = self._run(tmp_path, page, [["snap", "loaded"], ["wait"], ["snap", "ready"]],
                          strict=True)
        assert snaps["loaded"]["selects"]["flt-add"]["disabled"] is True
        ready = snaps["ready"]["selects"]["flt-add"]
        assert ready["disabled"] is False and ready["value"] == "off"
        assert ready["options"] == [["off", "off"], ["on", "on (up to the size cap)"]]
        # A page without the view keeps it shut, whatever else is chosen —
        # and reaches no element the page lacks
        bare = _ao_page(monkeypatch, tmp_path, _kc_sweep())
        snap = _run_script(tmp_path, bare, [
            ["wait"], ["set", "flt-k", "0"], ["fire", "flt-k"], ["settle"],
            ["set", "flt-add", "on"], ["fire", "flt-add"], ["settle"], ["snap", "s"]],
            strict=True)["s"]
        assert snap["selects"]["flt-add"]["disabled"] is True
        # The change event on the shut select drew nothing more than the k's did
        assert snap["inflated"] == ["dash-data", "dash-chunk-0", "dash-chunk-2"]

    def test_on_loads_the_add_on_chunk_and_words_the_scenario(self, monkeypatch, tmp_path):
        page = _ao_page(monkeypatch, tmp_path, _kc_sweep_add_on())
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        cid = data["grid_add"][0][1][1]
        scenario = _phrase(data, 0, 1, 1) + data["text"]["add_on"]
        assert scenario.endswith("20% cap per trade, adding to held pairs")
        steps = [["wait"], ["set", "flt-add", "on"], ["fire", "flt-add"], ["snap", "loading"],
                 ["settle"], ["snap", "on"], ["set", "flt-add", "off"], ["fire", "flt-add"],
                 ["settle"], ["snap", "off"]]
        snaps = self._run(tmp_path, page, steps, strict=True)
        # While its chunk inflates, the line says so, in Python's words
        assert snaps["loading"]["text"]["flt-summary"] == data["text"]["loading"].format(
            scenario=scenario)
        snap = snaps["on"]
        assert snap["inflated"] == ["dash-data", "dash-chunk-0", f"dash-chunk-{cid}"]
        assert {r["id"] for r in snap["reacts"]} == _charts_redrawn()
        view = chunks[cid]["list"]["views"]["all"]
        # The primary cell with adding on is its own simulation, not a slice
        assert snap["text"]["flt-summary"] == dashboard._filter_summary_text(
            data["text"], scenario, False, None, view["n"], view["n"], note="other_run",
            add_on=True)
        # ... which, while adding is on, says what an added purchase counts as,
        # in Python's words, just before the closing reach sentence
        note = dashboard._ADD_ON_SUMMARY_NOTE
        assert note.strip() == ("Each purchase that adds to a held pair counts as a trade of "
                                "its own; in the Kelly chart it shows only what it added.")
        assert data["text"]["add_on_note"] == note
        assert snap["text"]["flt-summary"].endswith(note + data["text"]["unfiltered"])
        assert snap["text"]["hdr-trades"] == str(view["n"]) == "4"
        # The add-on list's own Kelly scatter, never the page's
        assert _last_react(snap, "risk-kelly")["data"][0]["x"] == pytest.approx(
            chunks[cid]["list"]["kx"])
        assert snap["selects"]["flt-add"]["value"] == "on"
        # A slice of it is worded with the phrase too, its counts the add-on's own
        sports = _key(data, "Sports")
        snap = self._run(tmp_path, page, [
            ["wait"], ["set", "flt-add", "on"], ["fire", "flt-add"], ["settle"],
            _tick_category(sports[1:]), ["settle"],
            ["snap", "sports"]], strict=True)["sports"]
        views = chunks[cid]["list"]["views"]
        assert snap["text"]["flt-summary"] == dashboard._filter_summary_text(
            data["text"], scenario, False, "Sports", views[sports]["n"], views["all"]["n"],
            add_on=True)
        # Off again is the page as rendered: the primary's chunk, kept, loads nothing
        off = snaps["off"]
        assert off["inflated"] == snaps["on"]["inflated"]
        assert off["text"]["flt-summary"] == dashboard._filter_summary_text(
            data["text"], _phrase(data, 0, 1, 1), True, None, 3, 3)
        assert note not in off["text"]["flt-summary"]
        assert off["text"]["hdr-trades"] == "3"

    def test_a_restored_on_is_set_back_to_off_and_loads_no_add_on_chunk(
            self, monkeypatch, tmp_path):
        # A browser restored a reader's last choice, "on": the bar goes back to
        # the page as rendered, and no chunk of the add-on view is inflated
        page = _ao_page(monkeypatch, tmp_path, _kc_sweep_add_on())
        snaps = self._run(tmp_path, page, [["snap", "loaded"], ["wait"], ["snap", "ready"]],
                          pre=(("flt-add", "on"),), strict=True)
        assert snaps["loaded"]["selects"]["flt-add"]["value"] == "off"
        ready = snaps["ready"]
        assert ready["selects"]["flt-add"] == {**ready["selects"]["flt-add"], "value": "off",
                                               "disabled": False}
        assert ready["inflated"] == ["dash-data", "dash-chunk-0"]
        assert ready["reacts"] == []

    def test_tier_floors_off_with_adding_on_reads_the_add_on_off_grid(
            self, monkeypatch, tmp_path):
        page = _ao_page(monkeypatch, tmp_path, _kc_sweep_add_on(_kc_sweep_tiers()))
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        binding, plain = data["grid_add_off"][0][1][1], data["grid_add_off"][1][1][1]
        assert binding not in _ao_ids(data["grid_add"]) and plain in _ao_ids(data["grid_add"])
        snaps = self._run(tmp_path, page, [
            ["wait"], ["set", "flt-tier", "off"], ["fire", "flt-tier"], ["settle"],
            ["set", "flt-add", "on"], ["fire", "flt-add"], ["settle"], ["snap", "binding"],
            # A band the tiers never bind at: its off view is its add-on tier-on run
            ["set", "flt-band", "1"], ["fire", "flt-band"], ["settle"], ["snap", "plain"]],
            strict=True)
        add = data["text"]["add_on"]
        view = chunks[binding]["list"]["views"]["all"]["n"]
        assert f"dash-chunk-{binding}" in snaps["binding"]["inflated"]
        assert snaps["binding"]["text"]["flt-summary"] == dashboard._filter_summary_text(
            data["text"], _phrase_off(data, 0, 1, 1) + add, False, None, view, view,
            note="other_run", add_on=True)
        view = chunks[plain]["list"]["views"]["all"]["n"]
        assert snaps["plain"]["text"]["flt-summary"] == dashboard._filter_summary_text(
            data["text"], _phrase_off(data, 1, 1, 1) + add, False, None, view, view,
            add_on=True)
        assert f"dash-chunk-{plain}" in snaps["plain"]["inflated"]

    def test_a_scenario_the_family_never_simulated_reads_missing_and_cannot_be_saved(
            self, monkeypatch, tmp_path):
        # A tier-floors-off view with no add-on twin: adding "on" at a band the
        # tiers bind at is a scenario the run never simulated
        sweep = dataclasses.replace(_kc_sweep_add_on(_kc_sweep_tiers()),
                                    add_on_tier_off_cap_sweep=None)
        page = TestSaveLiveDefaultsButton._kc(monkeypatch, tmp_path, sweep)
        data = TestFilterPage._data(page)
        snaps = self._run(tmp_path, page, [
            ["wait"], ["set", "flt-tier", "off"], ["fire", "flt-tier"], ["settle"],
            ["set", "flt-add", "on"], ["fire", "flt-add"], ["settle"], ["snap", "missing"]],
            strict=True)
        snap = snaps["missing"]
        scenario = _phrase_off(data, 0, 1, 1) + data["text"]["add_on"]
        # Nothing is shown, so nothing is said about what an added purchase counts as
        assert snap["text"]["flt-summary"] == data["text"]["missing"].format(
            scenario=scenario) + data["text"]["unfiltered"]
        assert snap["buttons"]["flt-save"] is True and snap["text"]["hdr-trades"] == "0"

    def test_a_chunk_that_cannot_be_loaded_puts_the_choice_back(self, monkeypatch, tmp_path):
        page = _ao_page(monkeypatch, tmp_path, _kc_sweep_add_on())
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        cid = data["grid_add"][0][1][1]
        add = data["text"]["add_on"]
        snaps = self._run(tmp_path, page, [
            ["wait"], ["set", "flt-add", "on"], ["fire", "flt-add"], ["settle"], ["snap", "bad"],
            ["repair", f"dash-chunk-{cid}"],
            ["set", "flt-add", "on"], ["fire", "flt-add"], ["settle"], ["snap", "good"]],
            strict=True, damaged=(f"dash-chunk-{cid}",))
        bad = snaps["bad"]
        assert bad["selects"]["flt-add"]["value"] == "off" and bad["reacts"] == []
        # The line names the scenario that failed, in the phrase adding uses,
        # and the one still shown
        assert bad["text"]["flt-summary"] == data["text"]["unavailable"].format(
            failed=_phrase(data, 0, 1, 1) + add, reason=f"Error: damaged block dash-chunk-{cid}",
            scenario=_phrase(data, 0, 1, 1))
        # The scenario still shown can be saved again
        assert bad["buttons"]["flt-save"] is False
        # Repaired, choosing it again draws it
        good = snaps["good"]
        assert good["selects"]["flt-add"]["value"] == "on"
        assert good["text"]["hdr-trades"] == str(
            chunks[cid]["list"]["views"]["all"]["n"])

    def test_the_bar_s_call_to_the_explorer_keeps_its_four_arguments(
            self, monkeypatch, tmp_path):
        # The explorer never follows the choice: the label call carries the band,
        # k, size cap and Tier floors choice, whatever adding says
        page = _ao_page(monkeypatch, tmp_path, _kc_sweep_add_on())
        data = TestFilterPage._data(page)
        spy = ("window.dashScenarioSelect = function() { var el = document.getElementById("
               "'__spy'); var seen = el.textContent ? JSON.parse(el.textContent) : []; "
               "seen.push(Array.prototype.slice.call(arguments)); "
               "el.textContent = JSON.stringify(seen); };")
        snaps = self._run(tmp_path, page, [
            ["wait"], ["set", "flt-add", "on"], ["fire", "flt-add"], ["settle"],
            ["set", "flt-add", "off"], ["fire", "flt-add"], ["settle"], ["snap", "s"]], js=spy)
        calls = json.loads(snaps["s"]["text"]["__spy"])
        labels = [data["bands"][0]["label"], data["ks"][1]["label"], data["caps"][1]["label"],
                  "on"]
        assert calls == [labels, labels]
        js = dashboard._FILTER_JS
        [call] = re.findall(r"window\.dashScenarioSelect\(([^;]*)\);", js)
        assert len(re.split(r",\s*(?![^\[]*\])", call)) == 4 and "SHOWN[6]" not in call

    def test_the_explorer_s_selects_never_move_on_an_add_on_change(self, monkeypatch, tmp_path):
        page = _ao_page(monkeypatch, tmp_path, _kc_sweep_add_on(_ex_sweep_tiers()))
        snaps = self._run(tmp_path, page, [
            ["wait"], ["snap", "before"], ["set", "flt-add", "on"], ["fire", "flt-add"],
            ["settle"], ["snap", "on"], ["set", "flt-add", "off"], ["fire", "flt-add"],
            ["settle"], ["snap", "off"]], strict=True, explorer=True)
        before, on, off = snaps["before"], snaps["on"], snaps["off"]
        assert _scn_values(on) == _scn_values(off) == _scn_values(before)
        for snap in (on, off):
            assert snap["selects"]["scn-tier-select"]["value"] == "on"
            # ... and nothing of the explorer was redrawn
            assert _explorer_calls(snap) == []
        assert on["selects"]["flt-add"]["value"] == "on"

    # ─── The save address ─────────────────────────────────────────────────────

    def test_the_address_names_the_choice_only_from_a_page_that_simulated_it(
            self, monkeypatch, tmp_path):
        save = TestSaveLiveDefaultsButton()
        page = save._kc(monkeypatch, tmp_path, _kc_sweep_add_on())
        data = TestFilterPage._data(page)
        snaps = self._run(tmp_path, page, [
            ["wait"], ["click", "flt-save"], ["snap", "off"], ["set", "flt-add", "on"],
            ["fire", "flt-add"], ["settle"], ["click", "flt-save"], ["snap", "on"]],
            strict=True)
        for name, chosen in (("off", "off"), ("on", "on")):
            [opened] = snaps[name]["opened"]
            query = save._query(data, opened)
            save._assert_sends(data, query, 0, 1, 1)
            assert query["add_to_held_pairs"] == chosen
        # A page without the view sends none: the server keeps the saved value
        bare = save._kc(monkeypatch, tmp_path)
        bare_data = TestFilterPage._data(bare)
        [opened] = _run_script(tmp_path, bare, [["wait"], ["click", "flt-save"], ["snap", "s"]],
                               strict=True)["s"]["opened"]
        query = save._query(bare_data, opened)
        save._assert_sends(bare_data, query, *bare_data["primary"])
        assert "add_to_held_pairs" not in query

    @pytest.mark.parametrize("saved", [False, True])
    def test_the_address_is_what_the_defaults_server_reads(self, monkeypatch, tmp_path, saved):
        # The clicked address, handed to the server's own application and parser:
        # the settings it reads carry the choice on screen — off explicitly at
        # the page as rendered (so the seed's "on" does not leak in), on when
        # chosen — whatever is saved
        save = TestSaveLiveDefaultsButton()
        page = save._kc(monkeypatch, tmp_path, _kc_sweep_add_on(_kc_sweep_tiers()))
        data = TestFilterPage._data(page)
        sports = data["categories"].index("Sports")
        hockey = str(data["subcats"].index([sports, "Hockey"]))
        snaps = self._run(tmp_path, page, [
            ["wait"], ["click", "flt-save"], ["snap", "primary"],
            ["set", "flt-tier", "off"], ["fire", "flt-tier"], ["settle"],
            ["set", "flt-k", "0"], ["fire", "flt-k"], ["settle"],
            ["set", "flt-add", "on"], ["fire", "flt-add"], ["settle"],
            _tick_tag(hockey), ["settle"],
            ["click", "flt-save"], ["snap", "chosen"]], strict=True)
        if saved:
            save_config_live_defaults()
        current = config.read_saved_live_defaults()
        app = defaults_server._App(config.DEFAULTS_SERVER_PORT)
        pb, pk, pc = data["primary"]
        cases = (
            ("primary", config.LiveSettings(
                tier_floors=True, spread_band=tuple(data["bands"][pb]["value"]),
                interval_discount=data["ks"][pk]["value"],
                size_cap=data["caps"][pc]["value"], same_title_size_cap=0.5,
                add_to_held_pairs=False)),
            ("chosen", config.LiveSettings(
                tier_floors=False, spread_band=tuple(data["bands"][pb]["value"]),
                interval_discount=data["ks"][0]["value"],
                size_cap=data["caps"][pc]["value"], same_title_size_cap=0.5,
                categories=("Sports",), tags=("Sports · Hockey",), add_to_held_pairs=True)),
        )
        for name, expected in cases:
            [opened] = snaps[name]["opened"]
            save._query(data, opened)
            query = opened["url"].partition("?")[2]
            response = app.handle(defaults_server._Request("GET", f"/confirm?{query}",
                                                           _SAVE_HOST))
            assert response.status == 200, response.body
            settings, source = defaults_server._proposal(defaults_server._params(query),
                                                         current)
            assert settings == expected, name
            for field in config.LIVE_TOGGLE_FIELDS:
                assert getattr(settings, field) == getattr(expected, field), (name, field)
            assert source == data["save"]["source"]


class TestAddOnEndToEnd:
    """A real run_backtest_sweep with the add-on family, over the backtester's
    golden fixture narrowed to one band and one k, rendered through
    generate_dashboard: every cap's add-on chunk is the scenario a fresh
    simulation with add_to_held at that cap produces, figure for figure."""

    @pytest.mark.usefixtures("pre_toggle_defaults")
    def test_a_real_add_on_sweep_pages_every_cap_at_its_own_simulation(
            self, monkeypatch, tmp_path):
        golden = _tb.TestPrepareEntriesGolden()
        golden._patch(monkeypatch)
        monkeypatch.setattr(backtester, "SPREAD_BAND_SWEEP_FLOORS", (0.0,))
        monkeypatch.setattr(backtester, "SPREAD_BAND_SWEEP_CEILINGS", (1.0,))
        monkeypatch.setattr(backtester, "INTERVAL_DISCOUNT_SWEEP", (0.5,))
        run = backtester.run_backtest_sweep(
            MagicMock(), MagicMock(), golden._START, 10_000.0, same_event_ladders=True,
            interval_discount=0.5, band_sweep=True, cap_sweep=True, add_on_sweep=True)
        monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(dashboard.yf, "download",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
        page = dashboard.generate_dashboard(
            run.primary.trades, run.primary.equity_df, golden._START, 10_000.0,
            sweep=run).read_text(encoding="utf-8")
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        caps = run.add_on_cap_sweep.caps
        assert data["add_state"] == "shown" and data["grid_add"] is not None
        assert [c["value"] for c in data["caps"]] == list(caps) and len(caps) == 20
        band, end = (0.0, 1.0), run.primary.equity_df["date"].iloc[-1]
        added = 0
        for ci, cap in enumerate(caps):
            fresh = backtester._simulate_at_discount(
                run.add_on_cap_sweep.entries_by_band[band], golden._START, 10_000.0, k=0.5,
                spread_band=band, size_cap=cap, quiet=True, end_date=end, add_to_held=True)
            view = chunks[data["grid_add"][0][0][ci]]["list"]["views"]["all"]
            kpis = dashboard._performance_kpis(fresh.equity_df, fresh.trades, 10_000.0)
            assert view["kpi"] == {key: value for key, _, value, _ in kpis}, cap
            assert view["n"] == len(fresh.trades), cap
            added += sum(t.add_on for t in fresh.trades)
        # Not vacuous: adding made trades of its own, and the run as simulated differs
        assert added > 0
        assert data["grid_add"] != data["grid"]


_TRIM_NOT_SIMULATED = "(not simulated in this backtest)"
_TRIM_UNAVAILABLE = "(not available on this page; see the log)"


class TestTrimSelect:
    """The bar's "Trim to Kelly" select as Python renders it: after Add to held
    pairs and before Sell, shut, with its two options, its rule in the hover
    text and, on a page without the view, the reason beside it."""

    def test_the_select_follows_add_to_held_pairs_and_is_rendered_shut(
            self, monkeypatch, tmp_path):
        page = _ao_page(monkeypatch, tmp_path, _kc_sweep_trim())
        ids = re.findall(r'<select id="(flt-[a-z]+)"', page)
        assert ids == ["flt-band", "flt-tier", "flt-k", "flt-cap", "flt-add", "flt-trim",
                       "flt-sell", "flt-days"]
        # ... and before the Category and Tag menus
        bar = page[page.index('id="flt-bar"'):page.index('id="flt-summary"')]
        assert bar.index('id="flt-trim"') < bar.index('id="flt-cat"') < bar.index('id="flt-tag"')
        assert "<label>Trim to Kelly: <select" in page
        select = _page_elements(page)["selects"]["flt-trim"]
        assert select == {"disabled": True, "options": [
            {"value": "off", "text": "off", "selected": True},
            {"value": "on", "text": "on (sell down to the Kelly size)", "selected": False}]}
        title = re.search(r'<select id="flt-trim"[^>]*title="([^"]*)"', page).group(1)
        assert html.unescape(title) == dashboard._TRIM_SELECT_TITLE
        # The hover text says what the choice shuts, and why
        for words in ("Sell is set to no selling", "live trading does not trim",
                      "Each part sold counts as a trade of its own"):
            assert words in dashboard._TRIM_SELECT_TITLE
        # A page with the view has no note beside it
        assert 'id="flt-trim-note"' not in page

    @pytest.mark.parametrize("state, note", [
        ("not simulated", _TRIM_NOT_SIMULATED),
        ("unavailable", _TRIM_UNAVAILABLE),
        ("something else", _TRIM_NOT_SIMULATED)])
    def test_a_shut_select_says_why(self, state, note):
        _, chunker, base = _tr_payload(_kc_sweep())
        bar = dashboard._filter_bar_html({**base, "trim_state": state}, {"all": {"n": 3}})
        assert f'id="flt-trim-note" style="color:#9E9E9E; font-size:13px;">{note}</span>' in bar
        assert 'id="flt-trim" disabled' in bar
        shown = dashboard._filter_bar_html({**base, "trim_state": "shown"}, {"all": {"n": 3}})
        assert 'id="flt-trim-note"' not in shown

    def test_a_page_says_which_of_the_two_reasons_applies(self, monkeypatch, tmp_path):
        # No family: not simulated. A family the page could not use: see the log
        bare = _ao_page(monkeypatch, tmp_path, _kc_sweep())
        assert re.search(r'id="flt-trim-note"[^>]*>\(not simulated in this backtest\)</span>',
                         bare)
        sweep = _kc_sweep_trim()
        sweep.trim_sweep.caps = (0.05, 0.2)
        lost = _ao_page(monkeypatch, tmp_path, sweep)
        assert re.search(
            r'id="flt-trim-note"[^>]*>\(not available on this page; see the log\)</span>', lost)
        assert TestFilterPage._data(lost)["trim_state"] == "unavailable"

    def test_the_summary_text_gains_the_trim_note_after_the_add_on_one(self):
        _, _, base = _tr_payload(_kc_sweep_trim(_kc_sweep_add_on()))
        text = base["text"]
        plain = dashboard._filter_summary_text(text, "S", False, None, 4, 4)
        trimmed = dashboard._filter_summary_text(text, "S", False, None, 4, 4, trim=True)
        both = dashboard._filter_summary_text(text, "S", False, None, 4, 4, add_on=True,
                                              trim=True)
        note = dashboard._TRIM_SUMMARY_NOTE
        assert trimmed == plain[:-len(text["unfiltered"])] + note + text["unfiltered"]
        assert both.endswith(dashboard._ADD_ON_SUMMARY_NOTE + note + text["unfiltered"])


class TestTrimScript:
    """The bar's Trim to Kelly select under the page script (run outside a
    browser): enabled only with the view, "on" loads the trimming scenario's
    chunk and words it with Python's phrase and note, with Add to held pairs
    on it reads the grid of runs that do both, a chunk that cannot be loaded
    puts the choice back, and while it is on Sell is shut at no selling and
    nothing can be saved. The explorer never follows it. The harness's "wait"
    does not wait on flt-trim (a page without the view keeps it shut), so
    setup_js adds it. Skipped without a JavaScript runtime."""

    WAIT = "__BAR.push('flt-trim');"
    ON = [["set", "flt-trim", "on"], ["fire", "flt-trim"], ["settle"]]
    OFF = [["set", "flt-trim", "off"], ["fire", "flt-trim"], ["settle"]]

    @classmethod
    def _run(cls, tmp_path, page, steps, **kwargs) -> dict:
        """_run_script over a page, waiting on the trim select too."""
        return _run_script(tmp_path, page, steps, setup_js=cls.WAIT + kwargs.pop("js", ""),
                           **kwargs)

    def test_the_select_is_enabled_only_with_the_view(self, monkeypatch, tmp_path):
        page = _ao_page(monkeypatch, tmp_path, _kc_sweep_trim())
        snaps = self._run(tmp_path, page, [["snap", "loaded"], ["wait"], ["snap", "ready"]],
                          strict=True)
        assert snaps["loaded"]["selects"]["flt-trim"]["disabled"] is True
        ready = snaps["ready"]["selects"]["flt-trim"]
        assert ready["disabled"] is False and ready["value"] == "off"
        assert ready["options"] == [["off", "off"], ["on", "on (sell down to the Kelly size)"]]
        # A page without the view keeps it shut, whatever else is chosen
        bare = _ao_page(monkeypatch, tmp_path, _kc_sweep())
        snap = _run_script(tmp_path, bare, [
            ["wait"], ["set", "flt-k", "0"], ["fire", "flt-k"], ["settle"], *self.ON,
            ["snap", "s"]], strict=True)["s"]
        assert snap["selects"]["flt-trim"]["disabled"] is True
        # The change event on the shut select drew nothing more than the k's did
        assert snap["inflated"] == ["dash-data", "dash-chunk-0", "dash-chunk-2"]
        assert "trimming to Kelly" not in snap["text"]["flt-summary"]

    def test_on_loads_the_trim_chunk_and_words_the_scenario(self, monkeypatch, tmp_path):
        page = _ao_page(monkeypatch, tmp_path, _kc_sweep_trim())
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        cid = data["grid_trim"][0][1][1]
        assert cid not in _ao_ids(data["grid"])
        scenario = _phrase(data, 0, 1, 1) + data["text"]["trim"]
        assert scenario.endswith("20% cap per trade, trimming to Kelly")
        snaps = self._run(tmp_path, page, [
            ["wait"], ["set", "flt-trim", "on"], ["fire", "flt-trim"], ["snap", "loading"],
            ["settle"], ["snap", "on"], *self.OFF, ["snap", "off"]], strict=True)
        assert snaps["loading"]["text"]["flt-summary"] == data["text"]["loading"].format(
            scenario=scenario)
        snap = snaps["on"]
        assert snap["inflated"] == ["dash-data", "dash-chunk-0", f"dash-chunk-{cid}"]
        assert {r["id"] for r in snap["reacts"]} == _charts_redrawn()
        view = chunks[cid]["list"]["views"]["all"]
        # The primary cell with trimming on is its own simulation, and the line
        # says what a part sold counts as, in Python's words
        assert snap["text"]["flt-summary"] == dashboard._filter_summary_text(
            data["text"], scenario, False, None, view["n"], view["n"], note="other_run",
            trim=True)
        assert snap["text"]["flt-summary"].endswith(
            dashboard._TRIM_SUMMARY_NOTE + data["text"]["unfiltered"])
        assert snap["text"]["hdr-trades"] == str(view["n"]) == "4"
        assert _last_react(snap, "risk-kelly")["data"][0]["x"] == pytest.approx(
            chunks[cid]["list"]["kx"])
        # Off again is the page as rendered: the primary's chunk, kept
        off = snaps["off"]
        assert off["inflated"] == snap["inflated"]
        assert off["text"]["flt-summary"] == dashboard._filter_summary_text(
            data["text"], _phrase(data, 0, 1, 1), True, None, 3, 3)
        assert off["text"]["hdr-trades"] == "3"

    def test_ticked_categories_and_tags_are_read_from_the_trim_scenario(
            self, monkeypatch, tmp_path):
        # One ticked category is a view of the trimming run's own chunk; the
        # line words it with the trim phrase and note, and its counts are that
        # run's. Ticks that are a mix the script works out itself say so too.
        page = _ao_page(monkeypatch, tmp_path, _kc_sweep_trim())
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        cid = data["grid_trim"][0][1][1]
        views = chunks[cid]["list"]["views"]
        scenario = _phrase(data, 0, 1, 1) + data["text"]["trim"]
        sports = _key(data, "Sports")
        snaps = self._run(tmp_path, page, [
            ["wait"], *self.ON, _tick_category(sports[1:]), ["settle"], ["snap", "sports"],
            *self.OFF, ["snap", "off"]], strict=True)
        snap = snaps["sports"]
        assert snap["text"]["flt-summary"] == dashboard._filter_summary_text(
            data["text"], scenario, False, "Sports", views[sports]["n"], views["all"]["n"],
            trim=True)
        assert snap["text"]["hdr-trades"] == str(views[sports]["n"])
        # Trimming off keeps the tick, on the page's own run
        off = snaps["off"]
        assert off["selects"]["flt-cat"]["value"] == sports[1:]
        assert data["text"]["trim"] not in off["text"]["flt-summary"]
        assert "Sports" in off["text"]["flt-summary"]

    def test_a_scenario_that_trimmed_nothing_shows_the_scenario_s_own_chunk(
            self, monkeypatch, tmp_path):
        # With no cap nothing is trimmed: the trimming run traded the list of
        # the run without it, so no other chunk is loaded — and the line still
        # names the choice
        page = _ao_page(monkeypatch, tmp_path, _kc_sweep_trim())
        data = TestFilterPage._data(page)
        assert data["grid_trim"][0][1][2] == data["grid"][0][1][2]
        snap = self._run(tmp_path, page, [
            ["wait"], ["set", "flt-cap", "2"], ["fire", "flt-cap"], ["settle"],
            ["snap", "plain"], *self.ON, ["snap", "on"]], strict=True)
        assert snap["on"]["inflated"] == snap["plain"]["inflated"]
        assert snap["on"]["text"]["hdr-trades"] == snap["plain"]["text"]["hdr-trades"]
        assert data["text"]["trim"] in snap["on"]["text"]["flt-summary"]

    def test_a_restored_on_is_set_back_to_off_and_loads_no_trim_chunk(
            self, monkeypatch, tmp_path):
        page = _ao_page(monkeypatch, tmp_path, _kc_sweep_trim())
        snaps = self._run(tmp_path, page, [["snap", "loaded"], ["wait"], ["snap", "ready"]],
                          pre=(("flt-trim", "on"),), strict=True)
        assert snaps["loaded"]["selects"]["flt-trim"]["value"] == "off"
        ready = snaps["ready"]
        assert ready["selects"]["flt-trim"]["value"] == "off"
        assert ready["selects"]["flt-trim"]["disabled"] is False
        assert ready["inflated"] == ["dash-data", "dash-chunk-0"] and ready["reacts"] == []

    def test_with_adding_on_it_reads_the_grid_of_runs_that_do_both(self, monkeypatch, tmp_path):
        page = _ao_page(monkeypatch, tmp_path, _kc_sweep_trim(_kc_sweep_add_on()))
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        both = data["grid_trim_add"][0][1][1]
        assert both not in (_ao_ids(data["grid"]) | _ao_ids(data["grid_add"])
                            | _ao_ids(data["grid_trim"]))
        snaps = self._run(tmp_path, page, [
            ["wait"], ["set", "flt-add", "on"], ["fire", "flt-add"], ["settle"], *self.ON,
            ["snap", "both"], ["set", "flt-add", "off"], ["fire", "flt-add"], ["settle"],
            ["snap", "trim"]], js="__BAR.push('flt-add');", strict=True)
        snap = snaps["both"]
        assert f"dash-chunk-{both}" in snap["inflated"]
        scenario = _phrase(data, 0, 1, 1) + data["text"]["add_on"] + data["text"]["trim"]
        assert scenario.endswith(", adding to held pairs, trimming to Kelly")
        view = chunks[both]["list"]["views"]["all"]["n"]
        assert snap["text"]["flt-summary"] == dashboard._filter_summary_text(
            data["text"], scenario, False, None, view, view, note="other_run", add_on=True,
            trim=True)
        # Adding off again: the trim-alone scenario's chunk
        alone = data["grid_trim"][0][1][1]
        assert f"dash-chunk-{alone}" in snaps["trim"]["inflated"]
        assert snaps["trim"]["text"]["hdr-trades"] == str(
            chunks[alone]["list"]["views"]["all"]["n"])

    def test_tier_floors_off_reads_the_trim_off_grid(self, monkeypatch, tmp_path):
        page = _ao_page(monkeypatch, tmp_path, _kc_sweep_trim(_kc_sweep_tiers()))
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        binding, plain = data["grid_trim_off"][0][1][1], data["grid_trim_off"][1][1][1]
        assert binding not in _ao_ids(data["grid_trim"]) and plain in _ao_ids(data["grid_trim"])
        snaps = self._run(tmp_path, page, [
            ["wait"], ["set", "flt-tier", "off"], ["fire", "flt-tier"], ["settle"], *self.ON,
            ["snap", "binding"],
            ["set", "flt-band", "1"], ["fire", "flt-band"], ["settle"], ["snap", "plain"]],
            strict=True)
        trim = data["text"]["trim"]
        view = chunks[binding]["list"]["views"]["all"]["n"]
        assert f"dash-chunk-{binding}" in snaps["binding"]["inflated"]
        assert snaps["binding"]["text"]["flt-summary"] == dashboard._filter_summary_text(
            data["text"], _phrase_off(data, 0, 1, 1) + trim, False, None, view, view,
            note="other_run", trim=True)
        view = chunks[plain]["list"]["views"]["all"]["n"]
        assert snaps["plain"]["text"]["flt-summary"] == dashboard._filter_summary_text(
            data["text"], _phrase_off(data, 1, 1, 1) + trim, False, None, view, view,
            trim=True)

    def test_a_scenario_the_family_never_simulated_reads_missing(self, monkeypatch, tmp_path):
        # The family holds no run with the tier floors off: trimming at a band
        # the tiers bind at, with them off, was never simulated
        sweep = _kc_sweep_trim(_kc_sweep_tiers())
        sweep.trim_sweep.off_bands = ()
        page = TestSaveLiveDefaultsButton._kc(monkeypatch, tmp_path, sweep)
        data = TestFilterPage._data(page)
        snap = self._run(tmp_path, page, [
            ["wait"], ["set", "flt-tier", "off"], ["fire", "flt-tier"], ["settle"], *self.ON,
            ["snap", "missing"]], strict=True)["missing"]
        scenario = _phrase_off(data, 0, 1, 1) + data["text"]["trim"]
        # Nothing is shown, so nothing is said about what a part sold counts as
        assert snap["text"]["flt-summary"] == data["text"]["missing"].format(
            scenario=scenario) + data["text"]["unfiltered"]
        assert snap["buttons"]["flt-save"] is True and snap["text"]["hdr-trades"] == "0"

    def test_a_chunk_that_cannot_be_loaded_puts_the_choice_back(self, monkeypatch, tmp_path):
        page = TestSaveLiveDefaultsButton._kc(monkeypatch, tmp_path, _kc_sweep_trim())
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        cid = data["grid_trim"][0][1][1]
        snaps = self._run(tmp_path, page, [
            ["wait"], *self.ON, ["snap", "bad"], ["repair", f"dash-chunk-{cid}"], *self.ON,
            ["snap", "good"]], strict=True, damaged=(f"dash-chunk-{cid}",))
        bad = snaps["bad"]
        assert bad["selects"]["flt-trim"]["value"] == "off" and bad["reacts"] == []
        assert bad["text"]["flt-summary"] == data["text"]["unavailable"].format(
            failed=_phrase(data, 0, 1, 1) + data["text"]["trim"],
            reason=f"Error: damaged block dash-chunk-{cid}", scenario=_phrase(data, 0, 1, 1))
        # The scenario still shown, which does not trim, can be saved again
        assert bad["buttons"]["flt-save"] is False
        good = snaps["good"]
        assert good["selects"]["flt-trim"]["value"] == "on"
        assert good["text"]["hdr-trades"] == str(chunks[cid]["list"]["views"]["all"]["n"])

    # ─── Save and Sell while trimming ─────────────────────────────────────────

    def test_nothing_can_be_saved_while_trimming_is_on(self, monkeypatch, tmp_path):
        # Live trading does not trim, so a trimming scenario has no live
        # settings to save: the button is shut, a click opens nothing, and it
        # opens again with the choice off
        save = TestSaveLiveDefaultsButton()
        page = save._kc(monkeypatch, tmp_path, _kc_sweep_trim())
        data = TestFilterPage._data(page)
        snaps = self._run(tmp_path, page, [
            ["wait"], ["snap", "before"], *self.ON, ["click", "flt-save"], ["snap", "on"],
            # ... also where trimming changed nothing (no cap: the same chunk)
            ["set", "flt-cap", "2"], ["fire", "flt-cap"], ["settle"], ["click", "flt-save"],
            ["snap", "same"], *self.OFF, ["click", "flt-save"], ["snap", "off"]], strict=True)
        assert snaps["before"]["buttons"]["flt-save"] is False
        for name in ("on", "same"):
            assert snaps[name]["buttons"]["flt-save"] is True and snaps[name]["opened"] == []
        assert snaps["off"]["buttons"]["flt-save"] is False
        [opened] = snaps["off"]["opened"]
        query = save._query(data, opened)
        save._assert_sends(data, query, 0, 1, 2)
        assert not any("trim" in key for key in query)

    def test_sell_is_shut_at_no_selling_while_trimming_is_on(self, monkeypatch, tmp_path):
        page = _sl_page(monkeypatch, tmp_path, _kc_sweep_trim(_kc_sweep_sell()))
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        cid = data["grid_trim"][0][1][1]
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-sell", "0"], ["fire", "flt-sell"], ["settle"],
            ["snap", "level"], *self.ON, ["snap", "on"], *self.OFF, ["snap", "off"]],
            setup_js=self.WAIT + "__BAR.push('flt-sell');", files_root=tmp_path, strict=True)
        level = snaps["level"]["selects"]
        assert level["flt-sell"]["value"] == "0" and level["flt-days"]["disabled"] is False
        on = snaps["on"]
        # Sell went back to no selling and both its selects are shut ...
        assert on["selects"]["flt-sell"]["value"] == "none"
        assert on["selects"]["flt-sell"]["disabled"] is True
        assert on["selects"]["flt-days"]["disabled"] is True
        # ... and the scenario drawn is the trimming one, with no sale in its words
        assert f"dash-chunk-{cid}" in on["inflated"]
        view = chunks[cid]["list"]["views"]["all"]["n"]
        assert on["text"]["flt-summary"] == dashboard._filter_summary_text(
            data["text"], _phrase(data, 0, 1, 1) + data["text"]["trim"], False, None, view,
            view, note="other_run", trim=True)
        assert data["text"]["sell_note"] not in on["text"]["flt-summary"]
        # Trimming off: Sell opens again, still at no selling
        off = snaps["off"]["selects"]
        assert off["flt-sell"] == {**off["flt-sell"], "value": "none", "disabled": False}
        assert off["flt-days"]["disabled"] is True

    def test_a_failed_trim_chunk_gives_the_sell_level_back(self, monkeypatch, tmp_path):
        # A sell level was on screen when trimming was chosen and its chunk
        # could not be loaded: the level, its days and both selects come back
        page = _sl_page(monkeypatch, tmp_path, _kc_sweep_trim(_kc_sweep_sell()))
        data = TestFilterPage._data(page)
        cid = data["grid_trim"][0][1][1]
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-sell", "0"], ["fire", "flt-sell"], ["settle"],
            ["snap", "level"], *self.ON, ["snap", "back"]],
            setup_js=self.WAIT + "__BAR.push('flt-sell');", files_root=tmp_path, strict=True,
            damaged=(f"dash-chunk-{cid}",))
        back = snaps["back"]["selects"]
        assert back["flt-trim"]["value"] == "off"
        assert back["flt-sell"]["value"] == "0" and back["flt-sell"]["disabled"] is False
        assert back["flt-days"]["disabled"] is False
        assert snaps["back"]["text"]["hdr-trades"] == snaps["level"]["text"]["hdr-trades"]

    def test_a_page_without_the_sell_view_keeps_sell_shut_either_way(
            self, monkeypatch, tmp_path):
        page = _ao_page(monkeypatch, tmp_path, _kc_sweep_trim())
        snaps = self._run(tmp_path, page, [["wait"], *self.ON, ["snap", "on"], *self.OFF,
                                           ["snap", "off"]], strict=True)
        for name in ("on", "off"):
            assert snaps[name]["selects"]["flt-sell"]["disabled"] is True
            assert snaps[name]["selects"]["flt-days"]["disabled"] is True

    # ─── The explorer ─────────────────────────────────────────────────────────

    def test_the_bar_s_call_to_the_explorer_keeps_its_four_arguments(
            self, monkeypatch, tmp_path):
        page = _ao_page(monkeypatch, tmp_path, _kc_sweep_trim())
        data = TestFilterPage._data(page)
        spy = ("window.dashScenarioSelect = function() { var el = document.getElementById("
               "'__spy'); var seen = el.textContent ? JSON.parse(el.textContent) : []; "
               "seen.push(Array.prototype.slice.call(arguments)); "
               "el.textContent = JSON.stringify(seen); };")
        snaps = self._run(tmp_path, page, [["wait"], *self.ON, *self.OFF, ["snap", "s"]],
                          js=spy)
        calls = json.loads(snaps["s"]["text"]["__spy"])
        labels = [data["bands"][0]["label"], data["ks"][1]["label"], data["caps"][1]["label"],
                  "on"]
        assert calls == [labels, labels]
        [call] = re.findall(r"window\.dashScenarioSelect\(([^;]*)\);", dashboard._FILTER_JS)
        assert len(re.split(r",\s*(?![^\[]*\])", call)) == 4 and "SHOWN[9]" not in call

    def test_the_explorer_s_selects_never_move_on_a_trim_change(self, monkeypatch, tmp_path):
        page = _ao_page(monkeypatch, tmp_path, _kc_sweep_trim(_ex_sweep_tiers()))
        snaps = self._run(tmp_path, page, [
            ["wait"], ["snap", "before"], *self.ON, ["snap", "on"], *self.OFF, ["snap", "off"]],
            strict=True, explorer=True)
        before, on, off = snaps["before"], snaps["on"], snaps["off"]
        assert _scn_values(on) == _scn_values(off) == _scn_values(before)
        for snap in (on, off):
            assert _explorer_calls(snap) == []
        assert on["selects"]["flt-trim"]["value"] == "on"


class TestTrimEndToEnd:
    """Real size-cap, add-on and trim sweeps over the backtester's trimming
    fixture (TestTrimCapSweep's: a pair that outgrows its size cap, beside two
    others), rendered through generate_dashboard: every cap's chunk, with
    adding to held pairs off and on, is the scenario a fresh simulation that
    trims at that cap produces, figure for figure."""

    def test_real_trim_sweeps_page_every_cap_at_its_own_simulation(
            self, monkeypatch, tmp_path):
        fixture = _tb.TestTrimCapSweep()
        entries = fixture._entries()
        band, k, start, end = fixture._BAND, 0.75, fixture._START, fixture._END
        caps = backtester.SIZE_CAP_SWEEP

        def fresh(cap, **options):
            return backtester._simulate_at_discount(
                entries, start, 10_000.0, k=k, spread_band=band, size_cap=cap, quiet=True,
                end_date=end, **options)

        primary = fresh(0.20)
        axes = {"caps": caps, "primary_cap": 0.20, "bands": (band,), "ks": (k,),
                "primary_k": k, "start_date": start, "initial_balance": 10_000.0}
        lazy = {**axes, "split_date": None, "checks": False,
                "entries_by_band": {band: entries}, "st_entries": []}
        days = {(band, k, "all"): end}
        run = BacktestSweep(
            primary=primary, points=[primary], calibration=None, scenarios=[primary],
            calibrations_by_band={band: None}, same_event_ladders=True,
            cap_sweep=backtester.CapSweep(**lazy, eager={(band, k, "all"): primary}),
            add_on_cap_sweep=backtester.CapSweep(**lazy, eager={}, add_to_held=True,
                                                 end_dates=days),
            trim_sweep=backtester.TrimSweep(
                **axes, off_bands=(), entries_by_band={band: entries},
                off_entries_by_band={}, end_dates=days, off_end_dates={}))
        monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(dashboard.yf, "download",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
        page = dashboard.generate_dashboard(
            primary.trades, primary.equity_df, start, 10_000.0,
            sweep=run).read_text(encoding="utf-8")
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        assert data["trim_state"] == "shown" and data["add_state"] == "shown"
        assert [c["value"] for c in data["caps"]] == list(caps) and len(caps) == 20
        trimmed = {False: 0, True: 0}
        for add, key in ((False, "grid_trim"), (True, "grid_trim_add")):
            assert data[key] is not None
            for ci, cap in enumerate(caps):
                point = fresh(cap, add_to_held=add, trim_to_kelly=True)
                view = chunks[data[key][0][0][ci]]["list"]["views"]["all"]
                kpis = dashboard._performance_kpis(point.equity_df, point.trades, 10_000.0)
                assert view["kpi"] == {name: value for name, _, value, _ in kpis}, (add, cap)
                assert view["n"] == len(point.trades), (add, cap)
                trimmed[add] += any(t.sold for t in point.trades)
        # Not vacuous: at some caps a pair was trimmed, in both halves, and at
        # others nothing was, so those scenarios share the chunk of the run
        # without trimming
        assert 0 < trimmed[False] < len(caps) and 0 < trimmed[True] < len(caps)
        for with_trim, without in (("grid_trim", "grid"), ("grid_trim_add", "grid_add")):
            pairs = list(zip(data[with_trim][0][0], data[without][0][0], strict=True))
            assert any(a != b for a, b in pairs) and any(a == b for a, b in pairs)
        # The select is there to reach them
        assert 'id="flt-trim" disabled' in page and 'id="flt-trim-note"' not in page


# ─── Sell: every sell level of every scenario (sidecar chunks) ───────────────

_SELL_UNREAD = ("The selling simulations could not be read; the filter bar's Sell select "
                "stays disabled")
_SELL_MISMATCH = "The selling simulations do not match the page's grid"


def _sold(t: BacktestTrade, *, a: float | None = 0.81, b: float | None = 0.12,
          day: date | None = None) -> BacktestTrade:
    """t sold before it paid out: on `day` (the day before its exit, by
    default), market A's leg at `a` and B's at `b` (None: that leg's market
    paid out before the sale)."""
    return dataclasses.replace(t, sold=True, sale_price_a=a, sale_price_b=b, sale_fees=0.02,
                               exit_date=day or t.exit_date - timedelta(days=1))


class _FakeSellSweep:
    """
    A backtester.SellSweep stand-in carrying what the dashboard reads of one:
    levels, min_days, bands, off_bands, ks, caps, entry_events(), for_band()
    and sold_grid().

    sold_grid yields every (level, minimum of days), levels then each level's
    minimums ascending, and per cap one of what the real one yields, by these
    rules in order (rest = (band, k, tier_floors, add_to_held, cap)):
      * `same[(level, days, *rest)]`: that SameSale, as configured (it must
        name a cell yielded earlier as a point);
      * None (the run without selling) when `sold` lacks (level, *rest), or
        when days is above `reach[(level, *rest)]`;
      * `later[(level, days, *rest)]`: a point of its own (another run);
      * a SameSale naming the cell the last non-None cell of this level and
        cap was (a point names itself, a SameSale its target);
      * else `sold[(level, *rest)]`, the point at this level's first minimum.
    `raise_on` makes the cells of one (band, tier_floors) raise once the
    first (level, minimum) has been yielded, as a simulation failing part of
    the way through would; `reads` records every (band, k, tier_floors,
    add_to_held, caps) read, shared by every narrowed copy. The stats it is
    handed count one simulation per read and every SameSale and None cell.
    """

    def __init__(self, sold: dict, *, levels=(0.25, 0.5), min_days=None, bands=(_KC_B0, _KC_B1),
                 ks=(0.6, 0.75), caps=(0.05, 0.2, 1.0), off_bands=(), raise_on=None,
                 reads=None, events_raise=False, later=None, same=None, reach=None):
        # config's options (the page's) unless a test names its own
        self.min_days = tuple(config.TAKE_PROFIT_MIN_DAYS) if min_days is None else min_days
        self.sold, self.levels = sold, levels
        self.bands, self.ks, self.caps = bands, ks, caps
        self.off_bands, self.raise_on, self.events_raise = off_bands, raise_on, events_raise
        self.later, self.same, self.reach = later or {}, same or {}, reach or {}
        self.reads = [] if reads is None else reads

    def entry_events(self):
        if self.events_raise:
            raise RuntimeError("the entries could not be read")
        return {(t.event_ticker, t.category)
                for p in (*self.sold.values(), *self.later.values()) for t in p.trades}

    def for_band(self, band, *, tier_floors=True):
        return _FakeSellSweep(self.sold, levels=self.levels, min_days=self.min_days,
                              bands=(band,) if tier_floors else (), ks=self.ks, caps=self.caps,
                              off_bands=() if tier_floors else (band,),
                              raise_on=self.raise_on, reads=self.reads, later=self.later,
                              same=self.same, reach=self.reach)

    def _cell(self, level, days, rest, last):
        """One (level, minimum, cap) by the rules above; `last` is the call's
        (level, cap) -> the (level, minimum) its last non-None cell named."""
        cap = rest[-1]
        if (level, days, *rest) in self.same:
            target = self.same[(level, days, *rest)]
            last[(level, cap)] = (target.level, target.min_days)
            return target
        if (level, *rest) not in self.sold or days > self.reach.get((level, *rest), days):
            return None
        if (level, days, *rest) in self.later:
            last[(level, cap)] = (level, days)
            return self.later[(level, days, *rest)]
        if (level, cap) in last:
            return backtester.SameSale(*last[(level, cap)])
        last[(level, cap)] = (level, days)
        return self.sold[(level, *rest)]

    def sold_grid(self, band, k, *, tier_floors=True, add_to_held=False, caps=None,
                  stats=None):
        caps = tuple(self.caps if caps is None else caps)
        self.reads.append((band, k, tier_floors, add_to_held, caps))
        if stats is not None:
            stats["simulated"] += 1
        last: dict = {}
        for level in self.levels:
            for days in self.min_days:
                out = {cap: self._cell(level, days, (band, k, tier_floors, add_to_held, cap),
                                       last) for cap in caps}
                if stats is not None:
                    stats["same"] += sum(isinstance(c, backtester.SameSale)
                                         for c in out.values())
                    stats["pruned"] += sum(c is None for c in out.values())
                yield level, days, out
                if (band, tier_floors) == self.raise_on:
                    raise RuntimeError("the sell simulation failed")


def _sl_point(band, k, cap, trades, *, tier_floors=True, add_to_held=False,
              sell_at=0.25) -> SweepPoint:
    """A sell run's "all" point over `trades`."""
    return SweepPoint(k=k, trades=trades,
                      equity_df=backtester._build_equity_curve(trades, _FLT_START, 1000.0),
                      spread_band=band, size_cap=cap, tier_floors=tier_floors,
                      add_to_held=add_to_held, sell_at=sell_at)


def _kc_sweep_sell(base: BacktestSweep | None = None, *, raise_on=None,
                   events_raise=False, event: str | None = None) -> BacktestSweep:
    """
    A size-cap sweep (_kc_sweep() unless `base`) plus a Sell family at two
    levels, 25% and 50%, and config's minimum-days options (1 to 7, 14, 21).

    At 25%: (_KC_B0, k 0.60) sells at the 20% and no-cap caps (one list, as
    caps above a peak share one), its 5% cap is above every sale (the run
    without selling); (_KC_B0, 0.75) sells at every cap — 5% its own list
    with 1 day to spare at most (so from 2 days on it is the run without
    selling), 20% and no cap one shared list; (_KC_B1, 0.75) sells at every
    cap, one list (with `event`, filed under that event: a series no other
    scenario trades), and from 3 days on another list (a run of its own). At
    50%: only (_KC_B0, 0.75) at 20% sells, later and lower; at no cap it is
    exactly the 25% run (SameSale). Every other (level, minimum) of a cell
    that sells is its first minimum's run (SameSale). With an Add to held
    pairs family on `base`, (_KC_B0, 0.75)'s add-on cells sell at 25% too;
    with a tier-floors-off family, the binding band (_KC_B0)'s run with the
    tiers off sells at 25% at the run's own cap.
    """
    sweep = base or _kc_sweep()
    points = {key: (p["all"] if isinstance(p, dict) else p)
              for key, p in sweep.cap_sweep.points.items()}
    full = points[(_KC_B0, 0.75, 0.2)].trades
    small = points[(_KC_B0, 0.75, 0.05)].trades
    other = points[(_KC_B1, 0.75, 0.2)].trades
    at_060 = [_sold(points[(_KC_B0, 0.6, 0.2)].trades[0]),
              *points[(_KC_B0, 0.6, 0.2)].trades[1:]]
    sold_small = [_sold(small[0]), *small[1:]]
    sold_full = [_sold(full[0]), *full[1:]]
    late = [_sold(full[0], a=0.9, b=None), _sold(full[1], a=0.7, b=0.2), *full[2:]]
    sold_other = [_sold(other[0])] if event is None else [
        dataclasses.replace(_sold(other[0]), event_ticker=event, ticker_a=f"{event}-A"),
        *other[1:]]
    sold = {}
    for cap in (0.2, 1.0):
        sold[(0.25, _KC_B0, 0.6, True, False, cap)] = _sl_point(_KC_B0, 0.6, cap, at_060)
        sold[(0.25, _KC_B0, 0.75, True, False, cap)] = _sl_point(_KC_B0, 0.75, cap, sold_full)
        sold[(0.5, _KC_B0, 0.75, True, False, cap)] = _sl_point(_KC_B0, 0.75, cap, late,
                                                                sell_at=0.5)
    sold[(0.25, _KC_B0, 0.75, True, False, 0.05)] = _sl_point(_KC_B0, 0.75, 0.05, sold_small)
    for cap in (0.05, 0.2, 1.0):
        sold[(0.25, _KC_B1, 0.75, True, False, cap)] = _sl_point(_KC_B1, 0.75, cap, sold_other)
    if sweep.add_on_cap_sweep is not None:
        added = sweep.add_on_cap_sweep.points[(_KC_B0, 0.75, 0.2)].trades
        sold_added = [_sold(added[0]), *added[1:]]
        for cap in (0.05, 0.2, 1.0):
            sold[(0.25, _KC_B0, 0.75, True, True, cap)] = _sl_point(
                _KC_B0, 0.75, cap, sold_added, add_to_held=True)
    off_bands = ()
    if sweep.tier_off_scenarios:
        off_bands = (_KC_B0,)
        off = sweep.tier_off_scenarios[0].trades
        sold[(0.25, _KC_B0, 0.75, False, False, 0.2)] = _sl_point(
            _KC_B0, 0.75, 0.2, [_sold(off[0]), *off[1:]], tier_floors=False)
    other_later = [_sold(other[0], a=0.85)]
    later = {(0.25, 3, _KC_B1, 0.75, True, False, cap): _sl_point(_KC_B1, 0.75, cap, other_later)
             for cap in (0.05, 0.2, 1.0)}
    same = {(0.5, 1, _KC_B0, 0.75, True, False, 1.0): backtester.SameSale(0.25, 1)}
    reach = {(0.25, _KC_B0, 0.75, True, False, 0.05): 1}
    family = _FakeSellSweep(sold, off_bands=off_bands, raise_on=raise_on,
                            events_raise=events_raise, later=later, same=same, reach=reach)
    return dataclasses.replace(sweep, sell_sweep=family)


def _sl_page(monkeypatch, tmp_path, sweep, series=_FLT_SERIES_TIERS) -> str:
    """A whole page of a sweep, written to tmp_path with its sidecar folder."""
    return TestFilterPage()._page(monkeypatch, tmp_path, sweep, series)


def _sl_build(sweep: BacktestSweep, folder: Path, *, workers: int = 1) -> tuple:
    """(the grid walked, the chunk visitor, the Sell grid) as generate_dashboard
    builds them for a sweep, the sidecar files written to `folder`."""
    trades, curve = sweep.primary.trades, sweep.primary.equity_df
    source = dashboard._grid_source(sweep, trades, curve, 0.75)
    walked, chunker, _kd = dashboard._build_filter_grid(
        source, trades, curve, 0.75, _FLT_START, 1000.0, _FLT_SERIES_TIERS)
    folder.mkdir(parents=True, exist_ok=True)
    grid = dashboard._build_sell_grid(walked, chunker, start_date=_FLT_START,
                                      initial_balance=1000.0,
                                      series_categories=_FLT_SERIES_TIERS, risk_free=None,
                                      folder=folder, workers=workers)
    return walked, chunker, grid


def _sl_file(folder: Path, chunk_id: int) -> dict:
    """One sidecar chunk file, decoded strictly."""
    text = (folder / f"chunk-{chunk_id}.js").read_text(encoding="utf-8")
    return _decode_block(_SIDECAR_CALL.fullmatch(text).group(1))


def _sl_blocks(blocks) -> list:
    """The Sell blocks of a _SellGrid (or of a page's text), decoded strictly, by number."""
    decoded = _packed_blocks(blocks if isinstance(blocks, str) else "".join(blocks))
    found = sorted(int(name[len("dash-sell-"):]) for name in decoded
                   if name.startswith("dash-sell-"))
    assert found == list(range(len(found)))
    return [decoded[f"dash-sell-{i}"] for i in found]


def _sl_view(index, blocks: list, bases: dict, li: int, di: int, n_days: int) -> list:
    """
    The chunk ids one (level index li, minimum-days index di) shows, resolved
    as the page's script resolves them (chunkAt): [Tier floors 0 on / 1 off]
    [Add to held pairs 0 off / 1 on] -> [band][k][cap], None for a view the
    page lacks (`bases`: (t, a) -> its base grid, or None). A cell is None
    where its band has no block, the block has no such view or row, or the
    row is not covered; -1 is the scenario's own chunk, the base grid's id.
    """
    out = [[None, None], [None, None]]
    for t in (0, 1):
        for a in (0, 1):
            base = bases[(t, a)]
            if base is None:
                continue
            view = []
            for bi, band_rows in enumerate(base):
                bid = index[t][bi] if index is not None else None
                rows = blocks[bid][a] if bid is not None else None
                view.append([[None if rows is None or rows[ki][ci] is None
                              else (cid if (v := rows[ki][ci][li * n_days + di]) == -1 else v)
                              for ci, cid in enumerate(row)]
                             for ki, row in enumerate(band_rows)])
            out[t][a] = view
    return out


def _sl_grid_view(chunker, grid, li: int, di: int) -> list:
    """_sl_view over a _SellGrid built for a chunk visitor's grid."""
    add = chunker.add_grid()
    off = chunker.off_grid()
    bases = {(0, 0): chunker.grid, (0, 1): add, (1, 0): off,
             (1, 1): chunker.add_off_grid() if off is not None and add is not None else None}
    return _sl_view(grid.index, _sl_blocks(grid.blocks), bases, li, di, len(grid.days))


def _sl_page_view(page: str, li: int, di: int) -> list:
    """_sl_view over a whole page: its base block's grids and its Sell blocks."""
    data = TestFilterPage._data(page)
    bases = {(0, 0): data["grid"], (0, 1): data["grid_add"], (1, 0): data["grid_off"],
             (1, 1): data["grid_add_off"]}
    return _sl_view(data["sell_blocks"], _sl_blocks(page), bases, li, di,
                    len(data["sell_days"]))


# The _kc grid's Sell view (tier floors on, adding off) by (level index,
# minimum of days): the page's own chunks (_KC_GRID: 0-4) where a setting
# sells nothing, new sidecar chunks (5 on) where it sells, numbered in task
# order — band 0.0-1 first, its k 0.60 then 0.75, each level and minimum in
# turn, caps ascending. From 3 days on every minimum reads as 3 does.
_SL_VIEWS = {
    (0, 1): [[[3, 5, 5], [6, 7, 7]], [[None, None, None], [9, 9, 9]]],
    (0, 2): [[[3, 5, 5], [1, 7, 7]], [[None, None, None], [9, 9, 9]]],
    (0, 3): [[[3, 5, 5], [1, 7, 7]], [[None, None, None], [10, 10, 10]]],
    (1, 1): [[[3, 2, 2], [1, 8, 7]], [[None, None, None], [4, 4, 4]]],
    (1, 2): [[[3, 2, 2], [1, 8, 7]], [[None, None, None], [4, 4, 4]]],
    (1, 3): [[[3, 2, 2], [1, 8, 7]], [[None, None, None], [4, 4, 4]]],
}


def _sl_expected(li: int, days: int) -> list:
    """_SL_VIEWS at any minimum of days (every one past 3 reads as 3)."""
    return _SL_VIEWS[(li, min(days, 3))]


class TestSellGrid:
    """The bar's Sell view built after the walk: every (sell level, minimum of
    days) of every cell the page shows, a setting that sells nothing its
    scenario's own chunk (-1), one that repeats an earlier run that run's
    chunk, a new trade list a sidecar file numbered after the page's own
    chunks; one packed block per finished task, named by the base block's
    index; and a failure costing its band's cells alone."""

    def test_the_family_joins_a_grid_it_fits(self):
        sweep = _kc_sweep_sell(event="KXRAIN-9")
        source = dashboard._grid_source(sweep, sweep.primary.trades,
                                        sweep.primary.equity_df, 0.75)
        assert source.sell is sweep.sell_sweep
        # Every event a sell run trades under is listed before anything is simulated
        assert ("KXRAIN-9", "Other") in source.events
        # The cap-axis fallback keeps it: its one cap is among the family's
        assert source.fallback().sell is sweep.sell_sweep

    def test_a_family_that_does_not_fit_or_cannot_be_read_is_set_aside(self, caplog):
        sweep = _kc_sweep_sell()
        trades, curve = sweep.primary.trades, sweep.primary.equity_df
        other = dataclasses.replace(sweep, sell_sweep=_FakeSellSweep({}, ks=(0.75,)))
        with caplog.at_level(logging.WARNING):
            assert dashboard._grid_source(other, trades, curve, 0.75).sell is None
        assert any(_SELL_MISMATCH in r.getMessage() for r in caplog.records)
        caplog.clear()
        unread = dataclasses.replace(sweep, sell_sweep=_FakeSellSweep({}, events_raise=True))
        with caplog.at_level(logging.WARNING):
            assert dashboard._grid_source(unread, trades, curve, 0.75).sell is None
        assert any(_SELL_UNREAD in r.getMessage() for r in caplog.records)

    def test_every_setting_maps_each_cell_to_its_chunk(self, tmp_path, caplog):
        with caplog.at_level(logging.INFO):
            walked, chunker, grid = _sl_build(_kc_sweep_sell(), tmp_path / "f")
        days = tuple(config.TAKE_PROFIT_MIN_DAYS)
        assert chunker.grid == _KC_GRID and len(chunker.chunks) == 5
        # One block per band, each band's tier-on block named; no tier-off view
        assert grid.index == [[0, 1], [None, None]] and len(grid.blocks) == 2
        for li in range(2):
            for di, n in enumerate(days):
                view = _sl_grid_view(chunker, grid, li, di)
                assert view[0][0] == _sl_expected(li, n), (li, n)
                # Without the tier-floors-off or the add-on view, those are null
                assert view[0][1] is None and view[1] == [None, None]
        assert grid.sidecars == 6
        assert sorted(p.name for p in (tmp_path / "f").iterdir()) == sorted(
            f"chunk-{i}.js" for i in range(5, 11))
        assert [(e["label"], e["phrase"], e["value"]) for e in grid.levels] == [
            ("sell at 25% of potential profit",
             ", selling each position at 25% of its potential profit", 0.25),
            ("sell at 50% of potential profit",
             ", selling each position at 50% of its potential profit", 0.5)]
        assert [(e["label"], e["phrase"], e["value"]) for e in grid.days] == [
            (f"{n} day" if n == 1 else f"{n} days",
             f" and at least {n} day{'' if n == 1 else 's'} before its last market closes", n)
            for n in days]
        # Each band's cells were read once per k it shows, adding off, every cap
        # at once; band 0.3-0.6 shows no cell at k 0.60, so it was not read there
        assert walked.sell.reads == [(_KC_B0, 0.6, True, False, (0.05, 0.2, 1.0)),
                                     (_KC_B0, 0.75, True, False, (0.05, 0.2, 1.0)),
                                     (_KC_B1, 0.75, True, False, (0.05, 0.2, 1.0))]
        # The build's closing line counts the cells that repeat a run, and
        # those that sell nothing, as sold_grid reported them
        built = [r.getMessage() for r in caplog.records
                 if r.getMessage().startswith("Dashboard: Sell select built")]
        assert len(built) == 1
        assert len(days) == 9
        # Repeating: k 0.60's two selling caps at 25% past 1 day (2 x 8); k
        # 0.75's 20% at 25% and 50% (8 + 8), no cap at 25% (8) and at 50% (9:
        # its first minimum repeats the 25% run); band 0.3-0.6 at 25%, every
        # cap at every minimum but 1 and 3 days (3 x 7) — 70
        # Without a sale: k 0.60's 5% at 25% and every cap at 50% (9 + 27);
        # k 0.75's 5% at 25% past 1 day (8) and at 50% (9); band 0.3-0.6 at
        # 50% (27) — 80
        assert "70 cells repeating an earlier run, 80 cells without a sale" in built[0], \
            built[0]

    def test_every_covered_row_is_full_and_names_a_chunk_or_minus_one(self, tmp_path):
        _walked, chunker, grid = _sl_build(_kc_sweep_sell(), tmp_path / "f")
        width = len(grid.levels) * len(grid.days)
        ids = set(range(len(chunker.chunks) + grid.sidecars)) | {-1}
        for block in _sl_blocks(grid.blocks):
            assert block[1] is None
            for row in block[0]:
                for cell in row:
                    # A row is None (not covered) or holds every setting
                    assert cell is None or (len(cell) == width and set(cell) <= ids)
        # Band 0.3-0.6 shows no cell at k 0.60: those rows are not covered
        assert _sl_blocks(grid.blocks)[1][0][0] == [None, None, None]

    def test_a_repeated_run_takes_the_id_of_the_cell_it_names(self, tmp_path):
        _walked, chunker, grid = _sl_build(_kc_sweep_sell(), tmp_path / "f")
        days = tuple(config.TAKE_PROFIT_MIN_DAYS)
        (b0, _b1) = _sl_blocks(grid.blocks)
        cell = b0[0][1][2]                     # band 0.0-1, k 0.75, no cap
        # 50% at 1 day repeats the 25% run at 1 day (an earlier level)
        assert cell[len(days)] == cell[0] == 7
        # A setting that sells nothing is -1, not the own chunk's id
        assert b0[0][1][0][1] == -1 and b0[0][0][0] == [-1] * (2 * len(days))

    def test_a_sidecar_chunk_is_the_list_s_own_payload(self, tmp_path):
        sweep = _kc_sweep_sell()
        walked, chunker, _grid = _sl_build(sweep, tmp_path / "f")
        point = sweep.sell_sweep.sold[(0.25, _KC_B0, 0.75, True, False, 0.2)]
        strings, heads = dashboard._StringTable(), dashboard._StringTable()
        expected = dashboard._list_payload(
            point.trades, point.equity_df, chunker.axis, _FLT_START, 1000.0,
            _FLT_SERIES_TIERS, 0.75, chunker.cat_index, chunker.sub_index, strings,
            heads=heads)
        chunk = _sl_file(tmp_path / "f", 7)
        assert chunk == json.loads(dashboard._strict_json(
            {"list": expected, "strings": strings.items, "heads": heads.items}))
        # Its rows' heads are its own, and name the sale
        assert any("sold on" in head for head in chunk["heads"])

    def test_the_add_on_and_tier_off_views_are_covered(self, tmp_path):
        sweep = _kc_sweep_sell(_kc_sweep_add_on(_kc_sweep_tiers()))
        walked, chunker, grid = _sl_build(sweep, tmp_path / "f")
        off, add, add_off = chunker.off_grid(), chunker.add_grid(), chunker.add_off_grid()
        assert None not in (off, add, add_off)
        # Tasks: band 0.0-1 and 0.3-0.6 with the tiers on, then 0.0-1 (the one
        # the tiers bind at) off; 0.3-0.6 off reads its tier-on block
        assert grid.index == [[0, 1], [2, 1]]
        view = _sl_grid_view(chunker, grid, 0, 0)
        # Adding on: the add-on cells that sell are new chunks; the rest are the
        # add-on view's own chunks
        assert view[0][1][0][1] != add[0][1] and view[0][1][0][1][0] not in _ao_ids(add)
        assert view[0][1][0][0] == add[0][0]
        # Tier floors off: the binding band's own cell that sells is new, its
        # other cells (the family is at the run's own cap) stay the off view's;
        # the band the tiers never bind at shows its tier-on Sell rows
        assert view[1][0][0][1][1] not in _ao_ids(off)
        assert view[1][0][0][1][0] == off[0][1][0] and view[1][0][0][0] == off[0][0]
        assert view[1][0][1] == view[0][0][1]
        assert view[1][1][1] == view[0][1][1]
        # With both, the binding band's tier-off add-on cells never sell here:
        # each is the base grid's own chunk
        assert view[1][1][0] == add_off[0]
        assert (_KC_B0, 0.75, False, False, (0.05, 0.2, 1.0)) not in walked.sell.reads
        assert (_KC_B0, 0.75, False, False, (0.2,)) in walked.sell.reads

    def test_a_failed_band_costs_its_own_cells(self, tmp_path, caplog):
        # Band 0.3-0.6's cells raise once its first setting has been read and
        # its list written: that file is deleted, and its band has no block
        sweep = _kc_sweep_sell(raise_on=(_KC_B1, True))
        with caplog.at_level(logging.WARNING):
            _walked, chunker, grid = _sl_build(sweep, tmp_path / "f")
        warnings_ = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings_ == ["The Sell select's simulations at spread band 0.3-0.6 failed; "
                             "selling is not shown there"]
        assert grid.index == [[0, None], [None, None]] and len(grid.blocks) == 1
        for li in range(2):
            for di, n in enumerate(config.TAKE_PROFIT_MIN_DAYS):
                view = _sl_grid_view(chunker, grid, li, di)[0][0]
                assert view[0] == _sl_expected(li, n)[0]
                assert view[1] == [[None] * 3, [None] * 3]
        assert grid.sidecars == 4
        assert sorted(p.name for p in (tmp_path / "f").iterdir()) == [
            f"chunk-{i}.js" for i in range(5, 9)]

    def test_no_sale_anywhere_writes_no_file(self, tmp_path):
        sweep = dataclasses.replace(_kc_sweep(), sell_sweep=_FakeSellSweep({}))
        _walked, chunker, grid = _sl_build(sweep, tmp_path / "f")
        # Every setting sells nothing: every cell is the page's own chunk
        assert grid.index == [[0, 1], [None, None]]
        assert all(set(c) == {-1} for b in _sl_blocks(grid.blocks) for row in b[0] for c in row
                   if c is not None)
        assert _sl_grid_view(chunker, grid, 1, 4)[0][0] == chunker.grid
        assert grid.sidecars == 0 and list((tmp_path / "f").iterdir()) == []

    def test_the_base_block_carries_the_index_and_no_grid_of_ids(self, monkeypatch, tmp_path):
        page = _sl_page(monkeypatch, tmp_path, _kc_sweep_sell())
        data = TestFilterPage._data(page)
        assert "grid_sell" not in data
        assert data["sell_blocks"] == [[0, 1], [None, None]]
        assert [d["value"] for d in data["sell_days"]] == list(config.TAKE_PROFIT_MIN_DAYS)
        # The page's resolution of every setting is the grid's
        for li in range(2):
            for di, n in enumerate(config.TAKE_PROFIT_MIN_DAYS):
                assert _sl_page_view(page, li, di)[0][0] == _sl_expected(li, n)
        # The blocks sit after the page's own chunks and before the base block
        assert (page.index('id="dash-chunk-4"') < page.index('id="dash-sell-0"')
                < page.index('id="dash-sell-1"') < page.index('id="dash-data"'))

    def test_a_task_returns_one_compact_row_per_cell(self, tmp_path):
        # A task's result holds, per covered cell, one array of small numbers
        # (a place in its key list, or -1 for the run without selling), not
        # one entry per setting: the parent keeps every result until the
        # last task ends
        walked, chunker, _grid = _sl_build(_kc_sweep_sell(), tmp_path / "f")
        folder = tmp_path / "g"
        folder.mkdir()
        tasks = dashboard._sell_tasks(walked, chunker, start_date=_FLT_START,
                                      initial_balance=1000.0,
                                      series_categories=_FLT_SERIES_TIERS, risk_free=None,
                                      folder=folder)
        width = len(walked.sell.levels) * len(walked.sell.min_days)
        assert tasks
        for task in tasks:
            result = dashboard._run_sell_task(task)
            assert set(result.rows) == set(task.cells)
            assert len(set(result.keys)) == len(result.keys)
            for row in result.rows.values():
                assert isinstance(row, array) and row.typecode == "i" and len(row) == width
                assert all(-1 <= place < len(result.keys) for place in row)
            # Every key is named by a row, and is the page's own or one the
            # task wrote
            assert {result.keys[place] for row in result.rows.values()
                    for place in row if place >= 0} == set(result.keys)
            assert set(result.written) <= set(result.keys) <= (
                set(chunker.seen) | set(result.written))

    def test_a_worker_sells_by_the_parent_s_day_count(self, monkeypatch, tmp_path):
        # Each task carries the parent's TAKE_PROFIT_HOLD_DAYS beside its
        # same-title cap and schedule; a worker process, which imports
        # backtester afresh, installs all three before it simulates
        for name in ("SAME_TITLE_SIZE_CAP", "SCHEDULED_RUN"):
            monkeypatch.setattr(backtester, name, getattr(backtester, name))
        monkeypatch.setattr(backtester, "TAKE_PROFIT_HOLD_DAYS", 2)
        walked, chunker, _grid = _sl_build(_kc_sweep_sell(), tmp_path / "f")
        tasks = dashboard._sell_tasks(walked, chunker, start_date=_FLT_START,
                                      initial_balance=1000.0,
                                      series_categories=_FLT_SERIES_TIERS, risk_free=None,
                                      folder=tmp_path / "g")
        assert tasks and all(task.hold_days == 2 for task in tasks)
        # Run in this process as a worker would (nothing to cover, so no simulation)
        dashboard._run_sell_task(dataclasses.replace(tasks[0], cells=frozenset(),
                                                     install=True, hold_days=5))
        assert backtester.TAKE_PROFIT_HOLD_DAYS == 5

    @pytest.mark.usefixtures("pre_toggle_defaults")
    def test_worker_processes_build_what_the_parent_builds(self, monkeypatch, tmp_path):
        run = TestSellEndToEnd._run(monkeypatch)
        pages = {}
        for workers in (1, 2):
            root = tmp_path / f"w{workers}"
            root.mkdir()
            monkeypatch.setattr(dashboard, "PROJECT_ROOT", root)
            monkeypatch.setattr(dashboard.yf, "download",
                                lambda *a, **k: (_ for _ in ()).throw(RuntimeError("off")))
            page = dashboard.generate_dashboard(
                run.primary.trades, run.primary.equity_df, TestSellEndToEnd._START, 10_000.0,
                sweep=run, sell_workers=workers).read_text(encoding="utf-8")
            pages[workers] = (TestFilterPage._data(page), TestFilterPage._chunks(page),
                              _sidecar_files(page, root), _sl_blocks(page))
        (d1, c1, f1, _b1), (d2, c2, f2, _b2) = pages[1], pages[2]
        assert d1["sell_blocks"] == d2["sell_blocks"] and d1["sell_blocks"] is not None
        assert d1["inline_chunks"] == d2["inline_chunks"]
        assert c1 == c2
        assert pages[1][3] == pages[2][3] and len(pages[1][3]) > 0
        # The same files under the same names, whichever worker wrote each
        names = {src.rsplit("/", 1)[1]: chunk for src, chunk in f1.items()}
        assert names == {src.rsplit("/", 1)[1]: chunk for src, chunk in f2.items()}
        assert len(names) > 0


def _sl_flat_base(data: dict, t: int, a: int) -> list:
    """Every chunk id the base view of a Tier floors and Add to held pairs
    choice names, flattened."""
    grid = {(0, 0): data["grid"], (1, 0): data["grid_off"], (0, 1): data["grid_add"],
            (1, 1): data["grid_add_off"]}[(t, a)]
    return [c for band in grid for row in band for c in row]


class TestSellPage:
    """The Sell select on a whole page: rendered shut, with its levels and, on a
    page without the view, the note saying why; the base block's view; and
    the page's own header and chunks unchanged."""

    def test_the_title_says_how_many_days_a_level_must_hold(self, monkeypatch, tmp_path):
        # One day is the checkpoint alone; more names the count and how the
        # days are checked
        one, three = dashboard._sell_select_title(1), dashboard._sell_select_title(3)
        assert ("reaches that share of its potential profit (its contract pairs at $1.00 "
                "each, less what it cost). It is sold only while at least the days the "
                "Min. days to maturity select names remain before its last market stops "
                "trading. A position sold at a checkpoint") in one
        assert "days in a row" not in one
        assert ("has stayed at or above that share of its potential profit (its contract "
                "pairs at $1.00 each, less what it cost) for 3 days in a row: it is checked "
                "once a day, 24 hours apart, the last check at the checkpoint") in three
        # Saving a level makes live runs sell at it: the title says how, in the
        # same days
        assert one.endswith(
            "Save as live defaults… saves the level and minimum shown: a live run then "
            "sells at them, on real order books, once a position reaches its level. The "
            "Scenario Explorer and Interval Discount sections always show no selling.")
        assert three.endswith(
            "Save as live defaults… saves the level and minimum shown: a live run then "
            "sells at them, on real order books, once a position has held its level for 3 "
            "days in a row (read on Kalshi's hourly candles for the days before the run). "
            "The Scenario Explorer and Interval Discount sections always show no selling.")
        # Nothing on the page still says live trading never sells
        for words in (one, three, dashboard._SELL_SUMMARY_NOTE, dashboard._SELL_DAYS_TITLE,
                      dashboard._SELL_REACH, dashboard._SAVE_TITLE, dashboard._FILTER_JS):
            assert "never sells" not in words and "Live trading never" not in words
        assert dashboard._SELL_SUMMARY_NOTE.endswith(
            " Save as live defaults… saves this level and minimum, so a live run sells at "
            "them.")
        # A live run reads each market's scheduled close for the minimum of days
        assert "a live run reads each market's scheduled close instead" in \
            dashboard._SELL_DAYS_TITLE
        # The page names the count its sell runs read
        monkeypatch.setattr(backtester, "TAKE_PROFIT_HOLD_DAYS", 2)
        page = _sl_page(monkeypatch, tmp_path, _kc_sweep_sell())
        select = re.search(r'<select id="flt-sell"([^>]*)>', page)
        assert html.unescape(re.search(r'title="([^"]*)"', select.group(1)).group(1)) == \
            dashboard._sell_select_title(2)

    def test_the_select_lists_no_selling_then_every_level(self, monkeypatch, tmp_path):
        page = _sl_page(monkeypatch, tmp_path, _kc_sweep_sell())
        select = re.search(r'<select id="flt-sell"([^>]*)>(.*?)</select>', page)
        assert " disabled" in select.group(1)
        assert html.unescape(re.search(r'title="([^"]*)"', select.group(1)).group(1)) == \
            dashboard._sell_select_title(backtester.TAKE_PROFIT_HOLD_DAYS)
        assert re.findall(r'<option value="([^"]*)"( selected)?>(.*?)</option>',
                          select.group(2)) == [
            ("none", " selected", "no selling"), ("0", "", "sell at 25% of potential profit"),
            ("1", "", "sell at 50% of potential profit")]
        assert 'id="flt-sell-note"' not in page
        data = TestFilterPage._data(page)
        assert data["sell_state"] == "shown" and data["sell_blocks"] == [[0, 1], [None, None]]
        assert data["inline_chunks"] == 5
        assert data["sidecar_dir"].startswith(f"{config.DASHBOARD_FILES_DIRNAME}/")
        assert sorted(_sidecar_files(page, tmp_path)) == sorted(
            f"{data['sidecar_dir']}/chunk-{i}.js" for i in range(5, 11))
        # Each Sell block decodes strictly: [adding][k][cap] -> one id per setting
        blocks = _sl_blocks(page)
        assert len(blocks) == 2 and all(block[1] is None for block in blocks)
        # The summary's closing sentence says what the choice reaches
        assert data["text"]["unfiltered"].endswith(" " + dashboard._SELL_REACH_NO_EXPLORER)

    def test_the_days_select_follows_sell_with_every_option_and_its_rule(self, monkeypatch,
                                                                          tmp_path):
        page = _sl_page(monkeypatch, tmp_path, _kc_sweep_sell())
        # Right after the Sell select, and before Category
        # (the Category control is a check-box menu, a <details> element)
        ids = re.findall(r'<(?:select|details) id="(flt-[a-z]+)"', page)
        assert ids[ids.index("flt-sell") + 1:ids.index("flt-sell") + 3] == ["flt-days",
                                                                            "flt-cat"]
        assert '<label>Min. days to maturity: <select id="flt-days"' in page
        select = re.search(r'<select id="flt-days"([^>]*)>(.*?)</select>', page)
        assert " disabled" in select.group(1) and 'autocomplete="off"' in select.group(1)
        assert html.unescape(re.search(r'title="([^"]*)"', select.group(1)).group(1)) == \
            dashboard._SELL_DAYS_TITLE
        days = tuple(config.TAKE_PROFIT_MIN_DAYS)
        assert len(days) == 9
        assert re.findall(r'<option value="([^"]*)"( selected)?>(.*?)</option>',
                          select.group(2)) == [
            (str(i), " selected" if i == 0 else "", f"{n} day" if n == 1 else f"{n} days")
            for i, n in enumerate(days)]
        data = TestFilterPage._data(page)
        assert [d["label"] for d in data["sell_days"]] == [
            dashboard._sell_days_option(n) for n in days]
        assert data["sell_days"][2]["phrase"] == \
            " and at least 3 days before its last market closes"

    def test_a_page_without_the_view_says_why(self, monkeypatch, tmp_path):
        page = _sl_page(monkeypatch, tmp_path, _kc_sweep())
        assert re.search(r'<span id="flt-sell-note"[^>]*>([^<]*)</span>', page).group(1) == \
            "(not simulated in this backtest)"
        data = TestFilterPage._data(page)
        assert (data["sell_blocks"], data["sell_levels"], data["sell_days"], data["sell_state"],
                data["sidecar_dir"]) == (None, [], [], "not simulated", None)
        assert _sl_blocks(page) == []
        # The Min. days select is there, shut and empty
        days = re.search(r'<select id="flt-days"([^>]*)>(.*?)</select>', page)
        assert " disabled" in days.group(1) and days.group(2) == ""
        assert dashboard._SELL_REACH_NO_EXPLORER not in data["text"]["unfiltered"]
        assert not (tmp_path / config.DASHBOARD_FILES_DIRNAME).exists()
        unfit = dataclasses.replace(_kc_sweep(), sell_sweep=_FakeSellSweep({}, ks=(0.75,)))
        page = _sl_page(monkeypatch, tmp_path, unfit)
        assert re.search(r'<span id="flt-sell-note"[^>]*>([^<]*)</span>', page).group(1) == \
            "(not available on this page; see the log)"

    def test_the_page_s_own_data_is_unchanged_by_the_view(self, monkeypatch, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        with_view = TestFilterPage._chunks(_sl_page(monkeypatch, tmp_path / "a",
                                                    _kc_sweep_sell()))
        without = TestFilterPage._chunks(_sl_page(monkeypatch, tmp_path / "b", _kc_sweep()))
        assert with_view == without


class TestLiveSellRule:
    """The backtest report and the filter bar when the saved live defaults sell:
    a grey note beside the save button warning that saving with no selling
    turns live selling off (only on a page with the Sell view), and the
    page-header line naming the Sell options that show the sale rule at the
    live rule's band — or that the page cannot show it."""

    _LEVEL = {"label": "sell at 85% of potential profit", "value": 0.85}
    _DAYS = [{"label": "1 day", "value": 1}, {"label": "3 days", "value": 3}]
    _NOTE = ("; the live defaults sell at 85% of potential profit (at least 3 days before "
             "maturity), which this run's primary does not")

    @classmethod
    def _view(cls, blocks=((0,), (None,)), levels=None, days=None) -> dict:
        """The Sell keys of a filter bar's base block: a Sell block per band
        under each Tier floors setting (tiers on, then off), the levels
        offered and the minimums of days offered."""
        return {"sell_blocks": [list(row) for row in blocks],
                "sell_levels": [cls._LEVEL] if levels is None else levels,
                "sell_days": cls._DAYS if days is None else days}

    # ─── The save button's note ───────────────────────────────────────────────

    def test_the_save_button_warns_that_saving_with_no_selling_turns_selling_off(
            self, monkeypatch, tmp_path):
        note = ('The live defaults sell; saving with Sell set to "no selling" turns live '
                "selling off.")
        assert dashboard._SELL_SAVE_NOTE == note

        def between(page):
            bar = page[page.index('id="flt-bar"'):page.index('id="flt-summary"')]
            return bar[bar.index('id="flt-save"'):bar.index('id="flt-trade"')]

        with_view = _kc_sweep_sell()
        page = _sl_page(monkeypatch, tmp_path, dataclasses.replace(
            with_view, live_sell_at=0.85, live_sell_min_days=3))
        # Right after the button, before the trade link, in Python's words
        assert re.search(r'</button>&nbsp;<span id="flt-sell-save-note"[^>]*>'
                         + re.escape(html.escape(note)) + '</span>', between(page))
        assert note not in dashboard._FILTER_JS
        # A level with no minimum is selling too
        page = _sl_page(monkeypatch, tmp_path, dataclasses.replace(
            with_view, live_sell_at=0.85, live_sell_min_days=None))
        assert 'id="flt-sell-save-note"' in page
        # No note when the saved defaults do not sell (or the run recorded none) ...
        page = _sl_page(monkeypatch, tmp_path, with_view)
        assert 'id="flt-sell-save-note"' not in page
        # ... nor on a page with no Sell view, whose Save sends no choice
        page = _sl_page(monkeypatch, tmp_path, dataclasses.replace(
            _kc_sweep(), live_sell_at=0.85, live_sell_min_days=3))
        assert 'id="flt-sell-save-note"' not in page and 'id="flt-save"' in page

    def test_both_notes_follow_the_button_when_both_apply(self, monkeypatch, tmp_path):
        sweep = dataclasses.replace(_kc_sweep_sell(_kc_sweep_add_on()),
                                    live_add_to_held_pairs=True, live_sell_at=0.85)
        page = _sl_page(monkeypatch, tmp_path, sweep)
        bar = page[page.index('id="flt-bar"'):page.index('id="flt-summary"')]
        between = bar[bar.index('id="flt-save"'):bar.index('id="flt-trade"')]
        assert between.index('id="flt-add-save-note"') < between.index('id="flt-sell-save-note"')

    # ─── The page-header line ─────────────────────────────────────────────────

    def test_the_header_names_the_options_that_show_the_rule(self):
        run = TestLiveRuleHeader
        rule = config.describe_time_series_rule(True, (0.0, 1.0))
        head = f"Live rule (saved live defaults): {rule}; {run._CAP} — this run's primary"
        sweep = dataclasses.replace(run._sweep(), live_sell_at=0.85, live_sell_min_days=3)
        shown = {**run._bar(), **self._view()}
        # The rule is the page's primary and the bar has the view at that level
        # and minimum: choose them
        assert run()._text(sweep, shown) == (
            head + self._NOTE + ' — choose Sell "sell at 85% of potential profit" and Min. '
            'days to maturity "3 days" in the filter bar to see it')
        # No saved minimum: the smallest option is the nearest the bar has
        no_minimum = dataclasses.replace(sweep, live_sell_min_days=None)
        assert run()._text(no_minimum, shown) == (
            head + "; the live defaults sell at 85% of potential profit, which this run's "
            'primary does not — choose Sell "sell at 85% of potential profit" and Min. days '
            'to maturity "1 day" (the nearest to no minimum) in the filter bar to see it')
        # A level written with float noise still finds its option
        noisy = dataclasses.replace(sweep, live_sell_at=0.85 + 1e-12)
        assert run()._text(noisy, shown).endswith("in the filter bar to see it")

    def test_a_page_that_cannot_show_the_level_says_so(self):
        run = TestLiveRuleHeader
        rule = config.describe_time_series_rule(True, (0.0, 1.0))
        head = f"Live rule (saved live defaults): {rule}; {run._CAP} — this run's primary"
        cannot = " — this page cannot show that Sell level and minimum of days"
        sweep = dataclasses.replace(run._sweep(), live_sell_at=0.85, live_sell_min_days=3)
        other_level = [{"label": "sell at 90% of potential profit", "value": 0.9}]
        other_days = [{"label": "1 day", "value": 1}, {"label": "5 days", "value": 5}]
        for bar in (
                None, run._bar(), {**run._bar(), **self._view(), "sell_blocks": None},
                # no Sell block for the band under the tiers-on setting
                {**run._bar(), **self._view(blocks=((None,), (4,)))},
                {**run._bar(), **self._view(blocks=())},
                # the level, or the saved minimum of days, is not offered
                {**run._bar(), **self._view(levels=other_level)},
                {**run._bar(), **self._view(levels=[])},
                {**run._bar(), **self._view(days=other_days)},
                {**run._bar(), **self._view(days=[])}):
            assert run()._text(sweep, bar) == head + self._NOTE + cannot, bar
        # A rule the page does not show: the note alone, no promise of a view
        elsewhere = dataclasses.replace(sweep, live_spread_band=(0.3, 0.6),
                                        calibrations_by_band={(0.0, 1.0): None})
        assert run()._text(elsewhere, {**run._bar(), **self._view()}).endswith(
            "— not simulated by this run" + self._NOTE)
        # Not selling (or not recorded): no note at all
        for level, days in ((None, None), (None, 3)):
            plain = dataclasses.replace(sweep, live_sell_at=level, live_sell_min_days=days)
            text = run()._text(plain, {**run._bar(), **self._view()})
            assert "live defaults sell" not in text and "Sell" not in text

    def test_tiers_off_at_a_binding_band_reads_the_tier_off_blocks(self):
        run = TestLiveRuleHeader
        band = (0.0, 0.5)
        sweep = dataclasses.replace(
            run._sweep(live_tier_floors=False, live_spread_band=band,
                       calibrations_by_band={(0.0, 1.0): None, band: None},
                       tier_off_calibrations_by_band={(0.0, 1.0): None, band: None},
                       tier_off_scenarios=[object()]),
            live_sell_at=0.85, live_sell_min_days=3)
        for blocks, said in (
                (((0, 1), (2, 3)), "in the filter bar to see it"),
                # a block for the band under the tiers on, none under the tiers off
                (((0, 1), (None, 3)), "this page cannot show that Sell level and minimum of days")):
            bar = {**run._bar(bands=[band, (0.0, 1.0)]), **self._view(blocks=blocks)}
            assert run()._text(sweep, bar).endswith(said), blocks

    def test_the_header_of_a_whole_page_points_at_the_selects(self, monkeypatch, tmp_path):
        sweep = dataclasses.replace(_kc_sweep_sell(), live_sell_at=0.25, live_sell_min_days=3,
                                    live_tier_floors=True, live_spread_band=_KC_B0)
        page = _sl_page(monkeypatch, tmp_path, sweep)
        assert ("the live defaults sell at 25% of potential profit (at least 3 days before "
                "maturity), which this run's primary does not — choose Sell \"sell at 25% of "
                "potential profit\" and Min. days to maturity \"3 days\" in the filter bar "
                "to see it</p>") in page
        page = _sl_page(monkeypatch, tmp_path, dataclasses.replace(sweep, sell_sweep=None))
        assert ("the live defaults sell at 25% of potential profit (at least 3 days before "
                "maturity), which this run's primary does not — this page cannot show that Sell "
                "level and minimum of days</p>") in page

    def test_a_minimum_of_one_day_is_worded_in_the_singular(self):
        run = TestLiveRuleHeader
        sweep = dataclasses.replace(run._sweep(), live_sell_at=0.9, live_sell_min_days=1)
        assert run()._text(sweep, None).endswith(
            "; the live defaults sell at 90% of potential profit (at least 1 day before "
            "maturity), which this run's primary does not — this page cannot show that Sell "
            "level and minimum of days")


class TestSidecarFolders:
    """Each build's sidecar folder: made beside the page, published with it,
    and every earlier build's deleted — never one another build is still
    writing, and never the folder of the page in place."""

    @staticmethod
    def _folders(root: Path) -> list[str]:
        files = root / config.DASHBOARD_FILES_DIRNAME
        return sorted(p.name for p in files.iterdir() if p.is_dir()) if files.exists() else []

    def test_a_build_replaces_the_last_build_s_folder(self, monkeypatch, tmp_path):
        first = TestFilterPage._data(_sl_page(monkeypatch, tmp_path, _kc_sweep_sell()))
        second = TestFilterPage._data(_sl_page(monkeypatch, tmp_path, _kc_sweep_sell()))
        assert first["sidecar_dir"] != second["sidecar_dir"]
        assert self._folders(tmp_path) == [second["sidecar_dir"].split("/")[1]]
        # Published: no longer marked as being written
        assert not (tmp_path / second["sidecar_dir"] / ".writing").exists()

    def test_a_build_still_being_written_is_kept_and_a_stale_one_deleted(
            self, monkeypatch, tmp_path):
        files = tmp_path / config.DASHBOARD_FILES_DIRNAME
        (files / "other-build").mkdir(parents=True)
        (files / "other-build" / ".writing").write_text("", encoding="utf-8")
        (files / "crashed").mkdir()
        (files / "crashed" / ".writing").write_text("", encoding="utf-8")
        old = (datetime.now(UTC) - timedelta(seconds=config.DASHBOARD_BUILD_STALE_SECONDS + 60))
        os.utime(files / "crashed" / ".writing", (old.timestamp(), old.timestamp()))
        (files / "finished").mkdir()
        data = TestFilterPage._data(_sl_page(monkeypatch, tmp_path, _kc_sweep_sell()))
        assert self._folders(tmp_path) == sorted(["other-build",
                                                  data["sidecar_dir"].split("/")[1]])

    def test_a_page_without_the_view_deletes_the_folders_no_page_reads(
            self, monkeypatch, tmp_path):
        _sl_page(monkeypatch, tmp_path, _kc_sweep_sell())
        assert len(self._folders(tmp_path)) == 1
        _sl_page(monkeypatch, tmp_path, _kc_sweep())
        assert self._folders(tmp_path) == []

    def test_a_failed_page_keeps_the_last_page_and_its_folder(self, monkeypatch, tmp_path):
        page = _sl_page(monkeypatch, tmp_path, _kc_sweep_sell())
        kept = self._folders(tmp_path)

        def fail(*_a, **_k):
            raise OSError("disk full")

        monkeypatch.setattr(dashboard, "_publish_page", fail)
        with pytest.raises(OSError, match="disk full"):
            _sl_page(monkeypatch, tmp_path, _kc_sweep_sell())
        assert (tmp_path / config.DASHBOARD_FILENAME).read_text(encoding="utf-8") == page
        assert self._folders(tmp_path) == kept

    def test_a_cleanup_failure_after_the_rename_keeps_the_new_page_s_folder(
            self, monkeypatch, tmp_path, caplog):
        # Another build's folder whose mark cannot be read (made by another
        # user, say): the page is already in place when the cleanup meets it,
        # so the cleanup must neither raise nor cost the new page its folder
        files = tmp_path / config.DASHBOARD_FILES_DIRNAME
        (files / "foreign").mkdir(parents=True)
        (files / "foreign" / ".writing").write_text("", encoding="utf-8")
        real_stat = Path.stat

        def stat(self, *args, **kwargs):
            if self.name == ".writing" and self.parent.name == "foreign":
                raise PermissionError(13, "Permission denied", str(self))
            return real_stat(self, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", stat)
        with caplog.at_level(logging.WARNING):
            data = TestFilterPage._data(_sl_page(monkeypatch, tmp_path, _kc_sweep_sell()))
        assert (tmp_path / data["sidecar_dir"]).is_dir()
        assert list((tmp_path / data["sidecar_dir"]).glob("chunk-*.js"))
        # Its mark unread, the other build's folder is kept
        assert self._folders(tmp_path) == sorted(["foreign", data["sidecar_dir"].split("/")[1]])
        assert any("Could not check or delete an earlier dashboard build" in r.getMessage()
                   for r in caplog.records)

    def test_a_lock_that_cannot_be_taken_still_publishes_the_page(
            self, monkeypatch, tmp_path, caplog):
        # A run without the Sell family on a filesystem whose locks fail,
        # after an earlier build left its folder: the page is still replaced,
        # as it always was before these folders existed, and nothing is deleted
        _sl_page(monkeypatch, tmp_path, _kc_sweep_sell())
        kept = self._folders(tmp_path)
        assert kept

        def boom(*_args, **_kwargs):
            raise OSError(37, "No locks available")

        monkeypatch.setattr(dashboard.fcntl, "flock", boom)
        (tmp_path / config.DASHBOARD_FILENAME).write_text("old page", encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            page = _sl_page(monkeypatch, tmp_path, _kc_sweep())
        assert page != "old page" and "flt-bar" in page
        assert self._folders(tmp_path) == kept
        assert any("Could not take the dashboard builds' lock" in r.getMessage()
                   for r in caplog.records)

    def test_a_failed_sell_build_leaves_no_folder(self, monkeypatch, tmp_path, caplog):
        def fail(*_a, **_k):
            raise RuntimeError("the pool could not start")

        monkeypatch.setattr(dashboard, "_run_sell_tasks", fail)
        with caplog.at_level(logging.WARNING):
            page = _sl_page(monkeypatch, tmp_path, _kc_sweep_sell())
        assert any("The Sell select could not be built" in r.getMessage()
                   for r in caplog.records)
        assert TestFilterPage._data(page)["sell_state"] == "unavailable"
        assert self._folders(tmp_path) == []


class TestSoldTradeRow:
    """A trade sold before it paid out: its row names each leg's sale (or the
    payout of a leg whose market paid out first) beside how it settled; every
    other row is unchanged."""

    def test_a_sold_trade_names_its_sales(self):
        t = _flt_trades()[2]
        sold = _sold(t, a=0.81, b=None)
        head = dashboard._trade_row_head(sold, "#fff")
        assert (f"A: sold on {dashboard._fmt_day(sold.exit_date)} at $0.81 (settled "
                f"<b>{sold.outcome_a.upper()}</b> on {dashboard._fmt_day(sold.settled_date_a)})"
                in head)
        assert (f"B: settled <b>{sold.outcome_b.upper()}</b> on "
                f"{dashboard._fmt_day(sold.settled_date_b)}") in head
        assert "before the sale" in head
        # Not sold: the row is the one it always was
        assert "sold on" not in dashboard._trade_row_head(t, "#fff")
        assert dashboard._trade_row_head(dataclasses.replace(sold, sold=False), "#fff") == \
            dashboard._trade_row_head(dataclasses.replace(t, exit_date=sold.exit_date,
                                                          sale_price_a=0.81), "#fff")


class TestSellScript:
    """The bar's Sell and Min. days to maturity selects under the page script
    (run outside a browser): Sell enabled only with the view and set back to
    no selling on load; Min. days shut under no selling and open once a level
    is chosen; a level first inflates its band's Sell block, then loads its
    sidecar file (or the page's own chunk, where the setting sells nothing),
    words the scenario with Python's level and days phrases and note, shows
    the chunk's own row heads and keeps Save disabled until the chunk is drawn,
    after which its address carries the level and minimum shown; a block that
    cannot be inflated, or a file that cannot be loaded or hands nothing over,
    puts both choices back in Python's words. The harness's "wait" does not wait
    on flt-sell (a page without the view keeps it shut), so setup_js adds it.
    Skipped without a JavaScript runtime."""

    WAIT = "__BAR.push('flt-sell');"

    @classmethod
    def _run(cls, tmp_path, page, steps, **kwargs) -> dict:
        """_run_script over a page and its sidecar files, waiting on the Sell select."""
        return _run_script(tmp_path, page, steps, setup_js=cls.WAIT + kwargs.pop("js", ""),
                           files_root=tmp_path, **kwargs)

    @staticmethod
    def _scenario(data: dict, bi: int, ki: int, ci: int, li: int, di: int) -> str:
        """A selling scenario in the summary's words, as the script builds it."""
        return (_phrase(data, bi, ki, ci) + data["sell_levels"][li]["phrase"]
                + data["sell_days"][di]["phrase"])

    @staticmethod
    def _di(days: int) -> str:
        """The Min. days select's value for a number of days."""
        return str(tuple(config.TAKE_PROFIT_MIN_DAYS).index(days))

    def test_the_select_is_enabled_only_with_the_view(self, monkeypatch, tmp_path):
        page = _sl_page(monkeypatch, tmp_path, _kc_sweep_sell())
        snaps = self._run(tmp_path, page, [["snap", "loaded"], ["wait"], ["snap", "ready"]],
                          strict=True, pre=(("flt-sell", "1"), ("flt-days", "3")))
        assert snaps["loaded"]["selects"]["flt-sell"]["disabled"] is True
        assert snaps["loaded"]["selects"]["flt-days"]["disabled"] is True
        ready = snaps["ready"]["selects"]
        # A browser's restored choice is set back to the page as rendered,
        # and Min. days stays shut under no selling
        assert ready["flt-sell"]["disabled"] is False and ready["flt-sell"]["value"] == "none"
        assert ready["flt-days"]["disabled"] is True and ready["flt-days"]["value"] == "0"
        assert snaps["ready"]["files"] == []
        assert snaps["ready"]["inflated"] == ["dash-data", "dash-chunk-0"]
        bare = _sl_page(monkeypatch, tmp_path, _kc_sweep())
        snap = _run_script(tmp_path, bare, [
            ["wait"], ["set", "flt-sell", "0"], ["fire", "flt-sell"], ["settle"],
            ["snap", "s"]], strict=True, files_root=tmp_path)["s"]
        assert snap["selects"]["flt-sell"]["disabled"] is True
        assert snap["selects"]["flt-days"]["disabled"] is True
        assert snap["inflated"] == ["dash-data", "dash-chunk-0"] and snap["files"] == []

    def test_min_days_opens_with_a_level_and_shuts_without_one(self, monkeypatch, tmp_path):
        page = _sl_page(monkeypatch, tmp_path, _kc_sweep_sell())
        steps = [["wait"], ["set", "flt-days", self._di(3)], ["fire", "flt-days"], ["settle"],
                 ["snap", "ignored"], ["set", "flt-days", "0"],
                 ["set", "flt-sell", "0"], ["fire", "flt-sell"], ["settle"], ["snap", "sold"],
                 ["set", "flt-sell", "none"], ["fire", "flt-sell"], ["settle"],
                 ["snap", "none"]]
        snaps = self._run(tmp_path, page, steps, strict=True)
        # Under no selling a days change draws nothing and loads nothing
        assert snaps["ignored"]["reacts"] == [] and snaps["ignored"]["files"] == []
        assert snaps["sold"]["selects"]["flt-days"]["disabled"] is False
        assert snaps["none"]["selects"]["flt-days"]["disabled"] is True

    def test_a_level_inflates_its_band_s_block_then_loads_its_file(self, monkeypatch, tmp_path):
        page = _sl_page(monkeypatch, tmp_path, _kc_sweep_sell())
        data = TestFilterPage._data(page)
        files = _sidecar_files(page, tmp_path)
        cid = _sl_page_view(page, 0, 0)[0][0][0][1][1]
        assert cid == 7
        src = f"{data['sidecar_dir']}/chunk-{cid}.js"
        scenario = self._scenario(data, 0, 1, 1, 0, 0)
        steps = [["wait"], ["click", "flt-save"], ["snap", "before"],
                 ["set", "flt-sell", "0"], ["fire", "flt-sell"], ["snap", "loading"],
                 ["settle"], ["snap", "sold"], ["click", "flt-save"], ["snap", "clicked"],
                 ["set", "flt-sell", "none"], ["fire", "flt-sell"], ["settle"],
                 ["snap", "none"]]
        js = ("var __probe = document.getElementById('probe');"
              "window.dashScenarioSelect = function() {"
              " __probe.textContent = JSON.stringify(Array.prototype.slice.call(arguments)); };")
        snaps = self._run(tmp_path, page, steps, js=js)
        assert snaps["loading"]["text"]["flt-summary"] == data["text"]["loading"].format(
            scenario=scenario)
        snap = snaps["sold"]
        # The band's block first, then the file it names
        assert snap["inflated"] == ["dash-data", "dash-chunk-0", "dash-sell-0"]
        assert snap["files"] == [src]
        assert {r["id"] for r in snap["reacts"]} == _charts_redrawn()
        view = files[src]["list"]["views"]["all"]
        text = dashboard._filter_summary_text(data["text"], scenario, False, None, view["n"],
                                              view["n"], note="other_run")
        note = dashboard._SELL_SUMMARY_NOTE
        assert snap["text"]["flt-summary"] == text.replace(
            data["text"]["unfiltered"], note + data["text"]["unfiltered"])
        # The summary names the minimum of days after the level
        assert (data["sell_levels"][0]["phrase"] + " and at least 1 day before its last "
                "market closes") in snap["text"]["flt-summary"]
        # The trade tables join the chunk's own heads with its tails
        best = files[src]
        assert snap["html"]["diag-best"] == "".join(
            best["heads"][h] + best["strings"][t] for h, t in view["best"])
        assert "sold on" in snap["html"]["diag-best"]
        # While the band's block and the level's file load nothing can be saved;
        # once the chunk is drawn the button is enabled, and its address
        # carries the level and the minimum of days shown
        assert snaps["loading"]["buttons"]["flt-save"] is True
        assert snap["buttons"]["flt-save"] is False
        [opened] = snaps["clicked"]["opened"]
        query = TestSaveLiveDefaultsButton._query(data, opened)
        assert float(query["sell_at"]) == data["sell_levels"][0]["value"]
        assert query["sell_min_days"] == str(data["sell_days"][0]["value"])
        # The explorer is told the scenario with four labels, never the level
        assert len(json.loads(snap["text"]["probe"])) == 4
        # Back to no selling: the page's own chunk, nothing loaded again, Save on
        none = snaps["none"]
        assert none["files"] == [src] and none["buttons"]["flt-save"] is False
        assert none["inflated"] == ["dash-data", "dash-chunk-0", "dash-sell-0"]
        assert dashboard._SELL_SUMMARY_NOTE not in none["text"]["flt-summary"]

    def test_the_save_button_follows_the_level_and_minimum_on_screen(self, monkeypatch,
                                                                       tmp_path):
        page = _sl_page(monkeypatch, tmp_path,
                        dataclasses.replace(_kc_sweep_sell(), same_title_size_cap=0.5))
        data = TestFilterPage._data(page)
        src = f"{data['sidecar_dir']}/chunk-7.js"
        save = TestSaveLiveDefaultsButton()
        steps = [["wait"], ["set", "flt-sell", "0"], ["fire", "flt-sell"], ["settle"],
                 ["click", "flt-save"], ["snap", "loading"],
                 ["resolve_file", src], ["snap", "drawn"],
                 ["click", "flt-save"], ["snap", "clicked"]]
        snaps = self._run(tmp_path, page, steps, strict=True, files_deferred=(src,))
        loading = snaps["loading"]
        assert loading["pendingFiles"] == [src]
        assert loading["buttons"]["flt-save"] is True and loading["opened"] == []
        assert snaps["drawn"]["buttons"]["flt-save"] is False
        [opened] = snaps["clicked"]["opened"]
        query = save._query(data, opened)
        save._assert_sends(data, query, *data["primary"])
        assert (query["sell_at"], query["sell_min_days"]) == ("0.25", "1")
        # A file that cannot be loaded puts the choice back: the scenario still
        # shown is the one saved, with no selling
        failed = self._run(tmp_path, page, steps[:4] + [
            ["reject_file", src], ["snap", "failed"], ["click", "flt-save"],
            ["snap", "clicked"]], strict=True, files_deferred=(src,))
        assert failed["failed"]["buttons"]["flt-save"] is False
        [opened] = failed["clicked"]["opened"]
        query = save._query(data, opened)
        assert (query["sell_at"], query["sell_min_days"]) == ("off", "off")

    def test_a_sell_scenario_the_run_never_simulated_cannot_be_saved(self, monkeypatch,
                                                                       tmp_path):
        # (0.3-0.6, k 0.60) was never simulated, so it holds no sale either; a
        # simulated scenario at the same level can be saved again
        page = _sl_page(monkeypatch, tmp_path, _kc_sweep_sell())
        data = TestFilterPage._data(page)
        assert _KC_GRID[1][0] == [None, None, None]
        steps = [["wait"], ["set", "flt-k", "0"], ["set", "flt-band", "1"],
                 ["set", "flt-sell", "0"], ["fire", "flt-sell"], ["settle"],
                 ["click", "flt-save"], ["snap", "missing"],
                 ["set", "flt-k", "1"], ["fire", "flt-k"], ["settle"],
                 ["click", "flt-save"], ["snap", "back"]]
        snaps = self._run(tmp_path, page, steps, strict=True)
        assert snaps["missing"]["buttons"]["flt-save"] is True
        assert snaps["missing"]["opened"] == []
        assert snaps["back"]["buttons"]["flt-save"] is False
        [opened] = snaps["back"]["opened"]
        query = TestSaveLiveDefaultsButton._query(data, opened)
        assert (query["sell_at"], query["sell_min_days"]) == ("0.25", "1")

    def test_changing_the_days_loads_another_chunk(self, monkeypatch, tmp_path):
        page = _sl_page(monkeypatch, tmp_path, _kc_sweep_sell())
        data = TestFilterPage._data(page)
        folder = data["sidecar_dir"]
        # Band 0.3-0.6, k 0.75, 20%, selling at 25%: 1 and 2 days read chunk
        # 9, 3 days and more chunk 10
        assert [_sl_page_view(page, 0, int(self._di(n)))[0][0][1][1][1] for n in (1, 2, 3, 21)] \
            == [9, 9, 10, 10]
        steps = [["wait"], ["set", "flt-band", "1"], ["set", "flt-sell", "0"],
                 ["fire", "flt-sell"], ["settle"], ["snap", "one"],
                 ["set", "flt-days", self._di(3)], ["fire", "flt-days"], ["settle"],
                 ["snap", "three"], ["set", "flt-days", self._di(2)], ["fire", "flt-days"],
                 ["settle"], ["snap", "two"]]
        snaps = self._run(tmp_path, page, steps, strict=True)
        assert snaps["one"]["files"] == [f"{folder}/chunk-9.js"]
        three = snaps["three"]
        assert three["files"] == [f"{folder}/chunk-9.js", f"{folder}/chunk-10.js"]
        assert three["inflated"] == ["dash-data", "dash-chunk-0", "dash-sell-1"]
        assert self._scenario(data, 1, 1, 1, 0, 2) in three["text"]["flt-summary"]
        assert three["selects"]["flt-days"]["value"] == self._di(3)
        # Back to 2 days: chunk 9 again, kept, nothing loaded
        assert snaps["two"]["files"] == three["files"]
        assert self._scenario(data, 1, 1, 1, 0, 1) in snaps["two"]["text"]["flt-summary"]

    def test_a_setting_that_sells_nothing_shows_the_page_s_own_chunk(self, monkeypatch,
                                                                     tmp_path):
        page = _sl_page(monkeypatch, tmp_path, _kc_sweep_sell())
        data = TestFilterPage._data(page)
        # k 0.60 at the 5% cap: no sale at 25%, so the run without selling (chunk 3)
        assert _sl_blocks(page)[0][0][0][0][0] == -1
        assert _sl_page_view(page, 0, 0)[0][0][0][0][0] == data["grid"][0][0][0] == 3
        steps = [["wait"], ["set", "flt-k", "0"], ["set", "flt-cap", "0"],
                 ["set", "flt-sell", "0"], ["fire", "flt-sell"], ["settle"], ["snap", "s"]]
        snap = self._run(tmp_path, page, steps, strict=True)["s"]
        assert snap["files"] == []
        assert snap["inflated"] == ["dash-data", "dash-chunk-0", "dash-sell-0", "dash-chunk-3"]
        assert self._scenario(data, 0, 0, 0, 0, 0) in snap["text"]["flt-summary"]
        # At k 0.75 and 5%, 25% sells at 1 day and nothing from 2 days: the
        # page's own chunk 1, which the script inflates from the page
        steps = [["wait"], ["set", "flt-cap", "0"], ["set", "flt-sell", "0"],
                 ["fire", "flt-sell"], ["settle"], ["set", "flt-days", self._di(2)],
                 ["fire", "flt-days"], ["settle"], ["snap", "s"]]
        snap = self._run(tmp_path, page, steps, strict=True)["s"]
        assert snap["files"] == [f"{data['sidecar_dir']}/chunk-6.js"]
        assert snap["inflated"][-1] == "dash-chunk-1"

    def test_a_block_that_cannot_be_inflated_puts_both_choices_back(self, monkeypatch,
                                                                    tmp_path):
        page = _sl_page(monkeypatch, tmp_path, _kc_sweep_sell())
        data = TestFilterPage._data(page)
        steps = [["wait"], ["set", "flt-sell", "0"], ["fire", "flt-sell"], ["settle"],
                 ["snap", "s"], ["repair", "dash-sell-0"], ["set", "flt-sell", "0"],
                 ["fire", "flt-sell"], ["settle"], ["snap", "retried"]]
        snaps = self._run(tmp_path, page, steps, strict=True, damaged=("dash-sell-0",))
        snap = snaps["s"]
        assert snap["text"]["flt-summary"] == data["text"]["unavailable"].format(
            failed=self._scenario(data, 0, 1, 1, 0, 0),
            reason="Error: damaged block dash-sell-0", scenario=_phrase(data, 0, 1, 1))
        assert snap["selects"]["flt-sell"]["value"] == "none"
        assert snap["selects"]["flt-days"]["value"] == "0"
        assert snap["selects"]["flt-days"]["disabled"] is True
        assert snap["files"] == [] and snap["buttons"]["flt-save"] is False
        # A failed block is not kept: the next choice inflates it again
        assert snaps["retried"]["inflated"].count("dash-sell-0") == 2
        assert snaps["retried"]["files"] == [f"{data['sidecar_dir']}/chunk-7.js"]

    def test_a_file_that_cannot_be_loaded_puts_both_choices_back(self, monkeypatch, tmp_path):
        page = _sl_page(monkeypatch, tmp_path, _kc_sweep_sell())
        data = TestFilterPage._data(page)
        folder = data["sidecar_dir"]
        src = f"{folder}/chunk-7.js"
        steps = [["wait"], ["set", "flt-sell", "0"], ["fire", "flt-sell"], ["settle"],
                 ["snap", "s"]]
        failed = self._scenario(data, 0, 1, 1, 0, 0)
        for kwargs, template in (({"files_missing": (src,)}, "sidecar_missing"),
                                 ({"files_empty": (src,)}, "sidecar_empty")):
            snap = self._run(tmp_path, page, steps, strict=True, **kwargs)["s"]
            reason = "Error: " + data["text"][template].format(file=src)
            assert snap["text"]["flt-summary"] == data["text"]["unavailable"].format(
                failed=failed, reason=reason, scenario=_phrase(data, 0, 1, 1))
            assert snap["selects"]["flt-sell"]["value"] == "none"
            assert snap["selects"]["flt-days"]["disabled"] is True
            assert snap["files"] == [src]
            assert snap["buttons"]["flt-save"] is False
        # A days change whose file is missing puts the days back too, the level kept
        later = f"{folder}/chunk-10.js"
        steps = [["wait"], ["set", "flt-band", "1"], ["set", "flt-sell", "0"],
                 ["fire", "flt-sell"], ["settle"], ["set", "flt-days", self._di(3)],
                 ["fire", "flt-days"], ["settle"], ["snap", "s"]]
        snap = self._run(tmp_path, page, steps, strict=True, files_missing=(later,))["s"]
        assert snap["selects"]["flt-sell"]["value"] == "0"
        assert snap["selects"]["flt-days"]["value"] == "0"
        assert snap["selects"]["flt-days"]["disabled"] is False
        assert snap["text"]["flt-summary"] == data["text"]["unavailable"].format(
            failed=self._scenario(data, 1, 1, 1, 0, 2),
            reason="Error: " + data["text"]["sidecar_missing"].format(file=later),
            scenario=self._scenario(data, 1, 1, 1, 0, 0))

    def test_a_later_choice_supersedes_a_file_still_loading(self, monkeypatch, tmp_path):
        page = _sl_page(monkeypatch, tmp_path, _kc_sweep_sell())
        data = TestFilterPage._data(page)
        src = f"{data['sidecar_dir']}/chunk-7.js"
        steps = [["wait"], ["set", "flt-sell", "0"], ["fire", "flt-sell"], ["settle"],
                 ["snap", "loading"], ["set", "flt-sell", "1"], ["fire", "flt-sell"],
                 ["settle"], ["resolve_file", src], ["snap", "s"]]
        snaps = self._run(tmp_path, page, steps, strict=True, files_deferred=(src,))
        assert snaps["loading"]["pendingFiles"] == [src]
        snap = snaps["s"]
        # The later level (50%) is what is drawn
        later = f"{data['sidecar_dir']}/chunk-8.js"
        assert snap["files"] == [src, later]
        assert data["sell_levels"][1]["phrase"] in snap["text"]["flt-summary"]
        assert snap["selects"]["flt-sell"]["value"] == "1"

    def test_a_later_choice_supersedes_a_block_still_inflating(self, monkeypatch, tmp_path):
        page = _sl_page(monkeypatch, tmp_path, _kc_sweep_sell())
        data = TestFilterPage._data(page)
        folder = data["sidecar_dir"]
        # A level at band 0.0-1, whose block waits; then band 0.3-0.6, whose
        # block inflates at once: that is drawn, and the first block, once it
        # arrives, draws nothing over it (and is kept)
        steps = [["wait"], ["set", "flt-sell", "0"], ["fire", "flt-sell"], ["settle"],
                 ["snap", "waiting"], ["set", "flt-band", "1"], ["fire", "flt-band"],
                 ["settle"], ["snap", "other"], ["resolve", "dash-sell-0"], ["snap", "s"]]
        snaps = self._run(tmp_path, page, steps, strict=True, deferred=("dash-sell-0",))
        assert snaps["waiting"]["pending"] == ["dash-sell-0"]
        assert snaps["waiting"]["files"] == [] and snaps["waiting"]["reacts"] == []
        other = snaps["other"]
        assert other["files"] == [f"{folder}/chunk-9.js"]
        assert self._scenario(data, 1, 1, 1, 0, 0) in other["text"]["flt-summary"]
        snap = snaps["s"]
        assert snap["reacts"] == [] and snap["files"] == other["files"]
        assert snap["text"]["flt-summary"] == other["text"]["flt-summary"]
        assert snap["selects"]["flt-band"]["value"] == "1"

    def test_tiers_off_at_a_band_they_never_bind_reads_the_tier_on_block(self, monkeypatch,
                                                                         tmp_path):
        page = _sl_page(monkeypatch, tmp_path, _kc_sweep_sell(_kc_sweep_tiers()))
        data = TestFilterPage._data(page)
        folder = data["sidecar_dir"]
        # Blocks: band 0.0-1 and 0.3-0.6 with the tiers on, then 0.0-1 off;
        # 0.3-0.6 off names its tier-on block
        assert data["sell_blocks"] == [[0, 1], [2, 1]]
        steps = [["wait"], ["set", "flt-tier", "off"], ["fire", "flt-tier"], ["settle"],
                 ["set", "flt-band", "1"], ["fire", "flt-band"], ["settle"],
                 ["set", "flt-sell", "0"], ["fire", "flt-sell"], ["settle"], ["snap", "s"],
                 ["set", "flt-sell", "1"], ["fire", "flt-sell"], ["settle"], ["snap", "50"],
                 ["set", "flt-sell", "0"], ["set", "flt-band", "0"], ["fire", "flt-band"],
                 ["settle"], ["snap", "binding"]]
        snaps = self._run(tmp_path, page, steps, strict=True)
        snap = snaps["s"]
        assert "dash-sell-1" in snap["inflated"] and "dash-sell-2" not in snap["inflated"]
        # The tier-on run's sidecar at 25%
        cid = _sl_page_view(page, 0, 0)[1][0][1][1][1]
        assert cid == _sl_page_view(page, 0, 0)[0][0][1][1][1] >= data["inline_chunks"]
        assert snap["files"] == [f"{folder}/chunk-{cid}.js"]
        # At 50% it sells nothing: the tier-off view's own chunk — its tier-on
        # run's, drawn just before and kept, so nothing is loaded
        own = data["grid_off"][1][1][1]
        assert own == data["grid"][1][1][1] == 4
        fifty = snaps["50"]
        assert fifty["inflated"] == snap["inflated"] and fifty["files"] == snap["files"]
        view = TestFilterPage._chunks(page)[own]["list"]["views"]["all"]
        scenario = (_phrase_off(data, 1, 1, 1) + data["sell_levels"][1]["phrase"]
                    + data["sell_days"][0]["phrase"])
        text = dashboard._filter_summary_text(data["text"], scenario, False, None, view["n"],
                                              view["n"])
        assert fifty["text"]["flt-summary"] == text.replace(
            data["text"]["unfiltered"], dashboard._SELL_SUMMARY_NOTE + data["text"]["unfiltered"])
        # Band 0.0-1, where the tiers bind, reads its own tier-off block and
        # the sidecar of its run with the tiers off
        binding = snaps["binding"]
        assert binding["inflated"][-1] == "dash-sell-2"
        off_cid = _sl_page_view(page, 0, 0)[1][0][0][1][1]
        assert off_cid != _sl_page_view(page, 0, 0)[0][0][0][1][1]
        assert binding["files"][-1] == f"{folder}/chunk-{off_cid}.js"


class TestSellEndToEnd:
    """A real run_backtest_sweep with the Sell family, over the backtester's
    selling golden fixture narrowed to one band and one k, four sell levels
    and three minimum-days options, rendered through generate_dashboard:
    every (level, minimum, cap) chunk is what a fresh simulation selling at
    that level with that minimum produces — a sidecar file where it sells,
    the page's own chunk where it does not. The fixture's sales come 13, 20,
    26, 33, 53 and 60 days before maturity, so 1, 14 and 30 days give cells
    that are simulated, cells that repeat an earlier run and cells that sell
    nothing; the build's closing line says so."""

    _START = _tb.TestPrepareEntriesGolden._START
    _DAYS = (1, 14, 30)

    @classmethod
    def _run(cls, monkeypatch) -> BacktestSweep:
        """The selling golden fixture through run_backtest_sweep with every family on."""
        golden = _tb._SellingGolden()
        golden._patch(monkeypatch)
        monkeypatch.setattr(backtester, "SPREAD_BAND_SWEEP_FLOORS", (0.0,))
        monkeypatch.setattr(backtester, "SPREAD_BAND_SWEEP_CEILINGS", (1.0,))
        monkeypatch.setattr(backtester, "INTERVAL_DISCOUNT_SWEEP", (0.5,))
        monkeypatch.setattr(backtester, "TAKE_PROFIT_LEVELS", (0.05, 0.25, 0.5, 1.0))
        monkeypatch.setattr(backtester, "TAKE_PROFIT_MIN_DAYS", cls._DAYS)
        return backtester.run_backtest_sweep(
            MagicMock(), MagicMock(), golden._START, 10_000.0, same_event_ladders=True,
            interval_discount=0.5, band_sweep=True, tier_off_sweep=True, cap_sweep=True,
            add_on_sweep=True, sell_sweep=True)

    @pytest.mark.usefixtures("pre_toggle_defaults")
    def test_every_setting_pages_its_own_simulation(self, monkeypatch, tmp_path, caplog):
        run = self._run(monkeypatch)
        monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(dashboard.yf, "download",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
        with caplog.at_level(logging.INFO):
            page = dashboard.generate_dashboard(
                run.primary.trades, run.primary.equity_df, self._START, 10_000.0,
                sweep=run).read_text(encoding="utf-8")
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        files = {int(src.rsplit("-", 1)[1][:-3]): chunk
                 for src, chunk in _sidecar_files(page, tmp_path).items()}
        sell, band = run.sell_sweep, (0.0, 1.0)
        assert sell.min_days == self._DAYS
        assert [d["value"] for d in data["sell_days"]] == list(self._DAYS)
        end = run.primary.equity_df["date"].iloc[-1]
        sold = shared = 0
        for tier_floors in (True, False):
            entries = (sell.entries_by_band if tier_floors else sell.off_entries_by_band)[band]
            t = 0 if tier_floors else 1
            for add_to_held in (False, True):
                a = 1 if add_to_held else 0
                for li, level in enumerate(sell.levels):
                    for di, min_days in enumerate(sell.min_days):
                        view_ids = _sl_page_view(page, li, di)[t][a][0][0]
                        for ci, cap in enumerate(sell.caps):
                            cid = view_ids[ci]
                            assert cid is not None
                            fresh = backtester._simulate_at_discount(
                                entries, self._START, 10_000.0, k=0.5, spread_band=band,
                                size_cap=cap, quiet=True, end_date=end,
                                tier_floors=tier_floors, add_to_held=add_to_held,
                                sell_at=level, sell_min_days=min_days)
                            chunk = files[cid] if cid >= data["inline_chunks"] else chunks[cid]
                            view = chunk["list"]["views"]["all"]
                            kpis = dashboard._performance_kpis(fresh.equity_df, fresh.trades,
                                                               10_000.0)
                            assert view["kpi"] == {key: value for key, _, value, _ in kpis}, (
                                tier_floors, add_to_held, level, min_days, cap)
                            assert view["n"] == len(fresh.trades)
                            if any(tr.sold for tr in fresh.trades):
                                assert cid >= data["inline_chunks"]
                                sold += 1
                            else:
                                # Sells nothing: the scenario's own chunk
                                assert cid == _sl_flat_base(data, t, a)[ci]
                                shared += 1
        # Not vacuous either way, and every kind of cell occurred: simulated,
        # repeating an earlier run, and selling nothing
        assert sold > 0 and shared > 0
        built = [r.getMessage() for r in caplog.records
                 if r.getMessage().startswith("Dashboard: Sell select built")]
        assert len(built) == 1
        counts = re.search(r"(\d+) cap points simulated, \d+ shared across caps, (\d+) cells "
                           r"repeating an earlier run, (\d+) cells without a sale", built[0])
        assert counts and all(int(n) > 0 for n in counts.groups()), built[0]


class TestExplorerFullGrid:
    """
    A full size-cap grid — 36 bands x 13 ks x 20 caps (9,360 scenarios, the
    CLI's default grid) — whose cells share 4 trade lists and 4 curves
    through a fake CapSweep, each scenario carrying the "all" and
    "time_series" populations: the page's one walk packs 52 filter chunks (13
    ks x 4 lists) and 20 explorer cap blocks. Measured 2026-09-26 (plotly
    6.9.0, pandas 3.0.3, numpy 2.4.6, one run of the test body in a fresh
    process): the explorer's 21 blocks total 249,128 bytes of base64 (the
    base block 2,996, each cap block about 12.3 KB — four lists compress
    far better than a real grid's distinct cells will), in a 1.21 MB page
    built in 6.7 s, the process peaking at 348 MiB RSS (imports included).
    The budget is the plan's 8 MB for the explorer's blocks. Slow by the
    suite's standards; never run against a mutant.
    """

    BUDGET = 8_000_000
    # The quoted variant: every (band, k) cell's curve its own, from trades
    # whose quotes move on most days they are held (a run valued at market),
    # each about 270 rows, plotted daily. Measured 2026-09-30 (same
    # environment): the explorer's 21 blocks total 7,608,372 bytes (the
    # largest cap block 380,276), against 249,188 at cost, in an
    # 11,729,311-byte page (468 filter chunks, one per cell). Curves stay
    # daily by choice, so each budget is its measurement + 20%, with no ceiling.
    EXPLORER_BYTES_MEASURED_QUOTED = 7_608_372
    BUDGET_QUOTED = int(EXPLORER_BYTES_MEASURED_QUOTED * 1.2)
    PAGE_BYTES_MEASURED_QUOTED = 11_729_311
    PAGE_BUDGET_QUOTED = int(PAGE_BYTES_MEASURED_QUOTED * 1.2)

    def test_the_explorer_blocks_of_a_full_grid_stay_under_8_mb(self, monkeypatch, tmp_path):
        explorer, chunks, _page = self._explorer_blocks(monkeypatch, tmp_path, quoted=False)
        assert chunks == 52
        size = sum(len(body) for _, body in explorer)
        assert size <= self.BUDGET, f"the explorer's blocks were {size} bytes"

    def test_a_quoted_full_grid_s_explorer_blocks_stay_under_budget(self, monkeypatch,
                                                                     tmp_path):
        explorer, chunks, page = self._explorer_blocks(monkeypatch, tmp_path, quoted=True)
        # Each (band, k) cell's list has quotes of its own, so its curve is
        # its own: the filter ships one chunk per cell (13 ks x 36 bands),
        # never one curve for another
        assert chunks == 468
        size = sum(len(body) for _, body in explorer)
        assert size <= self.BUDGET_QUOTED, f"the explorer's blocks were {size} bytes"
        assert page <= self.PAGE_BUDGET_QUOTED, f"the page was {page} bytes"

    def _explorer_blocks(self, monkeypatch, tmp_path, *, quoted: bool) -> tuple[list, int, int]:
        """
        Build the full-grid page; return its explorer blocks, filter chunk count and size.

        With `quoted`, every (band, k) cell trades its list with quotes of its
        own (_random_marks), so each cell's curve is distinct and changes on
        most days a trade is open — the explorer's worst case — and the
        filter ships a chunk per cell. The page then files its trades by
        ticker prefix (no series listing), one category and tag, which the
        explorer never reads, so the filter's 468 lists build in seconds
        rather than as 13 views each.

        Returns:
            tuple[list, int, int]: [(block id, base64 body)] of the explorer's
                blocks in page order, the number of filter chunks, and the
                page's size in bytes.
        """
        import random
        rng = random.Random(3)
        series = [f"KXS{i:02d}" for i in range(8)]
        categories = {s: (f"Cat{i % 4}", (f"Tag{i}",)) for i, s in enumerate(series)}
        start = date(2026, 1, 5)

        def trade(i: int) -> BacktestTrade:
            s = rng.choice(series)
            entry = start + timedelta(days=rng.randint(0, 150))
            return dataclasses.replace(
                _typed_trade("time_series", rng.random() < 0.5, entry,
                             entry + timedelta(days=rng.randint(1, 30)), rng.gauss(0, 20)),
                event_ticker=f"{s}-{i}", ticker_a=f"{s}-{i}A", title_a=f"Question {i}?")

        bands = tuple((lo, hi) for lo in config.SPREAD_BAND_SWEEP_FLOORS
                      for hi in config.SPREAD_BAND_SWEEP_CEILINGS)
        ks = tuple(config.INTERVAL_DISCOUNT_SWEEP)
        caps = backtester.SIZE_CAP_SWEEP
        assert (len(bands), len(ks), len(caps)) == (36, 13, 20) and 0.2 in caps
        lists = []
        for j in range(4):
            trades = sorted((trade(100 * j + i) for i in range(40)), key=lambda t: t.entry_date)
            lists.append((trades, backtester._build_equity_curve(trades, start, 10_000.0),
                           HalfSplit(rng.gauss(0, 0.05), rng.gauss(0, 0.05), 20, 20)))
        mark_rng = random.Random(11)
        quoted_cells = {}
        if quoted:
            # One list per (band, k) cell at every cap, its quotes its own
            for bi in range(len(bands)):
                for ki in range(len(ks)):
                    base_trades, _, halves = lists[(bi + ki) % 4]
                    marked = [_random_marks(t, mark_rng) for t in base_trades]
                    quoted_cells[(bi, ki)] = (
                        marked, backtester._build_equity_curve(marked, start, 10_000.0), halves)
        points = {}
        for bi, band in enumerate(bands):
            for ki, k in enumerate(ks):
                for ci, cap in enumerate(caps):
                    trades, curve, halves = (quoted_cells[(bi, ki)] if quoted
                                             else lists[(bi + ki + ci) % 4])
                    point = SweepPoint(k=k, trades=trades, equity_df=curve, spread_band=band,
                                       size_cap=cap, halves=halves)
                    points[(band, k, cap)] = {
                        "all": point,
                        "time_series": dataclasses.replace(point, population="time_series")}
        obs = tuple(backtester.CalibrationObservation(
            rng.randint(1, 30), rng.uniform(0.15, 0.8), rng.random() < 0.4,
            f"{rng.choice(series)}-{j}", "Other") for j in range(300))
        cal = IntervalCalibration(pooled=backtester._calibration_bucket("POOLED", 0.0, obs),
                                  buckets=[], excluded_premise_violations=0, observations=obs)
        primary = points[(bands[0], 0.75, 0.2)]["all"]
        sweep = BacktestSweep(
            primary=primary, points=[points[(bands[0], k, 0.2)]["all"] for k in ks],
            calibration=cal, label_coverage=_scn_coverage(),
            scenarios=[p for (_, _, cap), by_pop in points.items() if cap == 0.2
                       for p in by_pop.values()],
            calibrations_by_band=dict.fromkeys(bands, cal), same_event_ladders=True,
            cap_sweep=_ExplorerCapSweep(points, bands=bands, ks=ks, caps=caps))
        monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(dashboard.yf, "download",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
        out = dashboard.generate_dashboard(
            primary.trades, primary.equity_df, start, 10_000.0, sweep=sweep,
            interval_discount=0.75, series_categories=None if quoted else categories)
        page = out.read_text(encoding="utf-8")
        explorer = [(i, body) for i, body in _PACKED.findall(page) if i.startswith("scn-")]
        assert [i for i, _ in explorer] == ["scn-data"] + [f"scn-cap-{ci}" for ci in range(20)]
        return explorer, len(TestFilterPage._chunks(page)), out.stat().st_size


class TestFilterPageSize:
    """A band sweep whose 36 bands each traded a DIFFERENT list — the worst
    case for the filter, which ships a chunk per distinct scenario list and a
    view per list x category x tag — stays well inside the page budget the
    scenario explorer set (5 MB), because every block is gzip-packed
    (_packed_json_script) and every distinct trade-row head is shipped once
    (the base block's shared table). Each band also carries a
    300-entry k-hat population, which the k-hat breakdown ships per band x
    category x tag. Measured 810,784 bytes on this fixture (2026-09-26,
    plotly 6.9.0 / pandas 3.0.3 / numpy 2.4.6), against 744,946 for the same
    fixture's one-block page before the chunks existed: packing each list on
    its own loses the compression one block shared across lists, and sharing
    the rows' heads wins back part of it. The k-hat cards (every k-hat
    group now carries its k-hat − k per k) and the interval-discount
    section's data at every k and cap ("kd") add about 7 KB: re-measured
    819,188 bytes (2026-09-26, same environment), against 812,186 for the
    page before them on the same day. Packing the scenario explorer's data
    (C4: a base block and a block per size cap in place of one JSON block;
    this fixture's 36 bands carry "all" points only) takes about 15 KB off:
    re-measured 804,179 bytes (2026-09-26, same environment), against
    819,347 for the page before it on the same day. The merge of #68 adds
    about 5 KB with no family at all (the Tier floors select, its title and
    note, and each band's option text in the base block): re-measured
    809,356 bytes (2026-09-26, same environment).

    With the tier-floors-off family (#68's worst case) the 18 bands a tier
    binds at each traded yet another list — 54 chunks in all (a band the
    tiers never bind at shares its tier-on chunk) — and each binding band's
    tier-off run carries its own 300-entry k-hat population, which the base
    block ships per band x category x tag beside the tier-on one. Measured
    1,077,022 bytes on this fixture (2026-09-26, the merge of #68, same
    environment). The scenario explorer's own tier-floors-off view (the
    merge's stage C — its select, hidden k-hat table and longer script, and
    with the family an "scn-off-cap-0" block of 3,712 bytes of base64)
    re-measured both the same day: 815,777 bytes without the family and
    1,098,598 with it. Each budget is its measurement + 20% (the curve runs
    to today, so the page grows by a few bytes a day).

    What the page needs to combine several categories and tags (per-trade
    arrays in every chunk; each category · tag view's curve in dollars, "m",
    and its table row; the base block's "mix" and each tag group's raw k-hat
    counts) first added 5% to 9%, with "m" shipped in place of the view's
    "eq" alone (with "eq" kept beside it, 11% to 15%). A tag view's return
    line, type lines and drawdown are that same curve again, so they are now
    left out too wherever the script rebuilds them exactly from "m": every
    variant re-measured 2026-10-10 (same environment), in the constants
    below — the pages at cost end 5.6% to 7.2% above the page before the mix
    data, and the pages valued at market, whose curves move daily, 4.9% and
    6.0% BELOW it. The check-box Category and Tag menus and the script that
    works a mix out are about 34 KB of every variant, whatever its data."""

    # 837,197 before the mix data; 922,701 with it, the menus and the mix
    # script ("m" in place of "eq" alone); 897,445 with a tag view's other
    # curve series left out as well (2026-10-10)
    PAGE_BYTES_MEASURED = 897_445
    BUDGET = int(PAGE_BYTES_MEASURED * 1.2)
    # 1,119,167 before; 1,233,927; 1,193,307
    PAGE_BYTES_MEASURED_FAMILY = 1_193_307
    BUDGET_FAMILY = int(PAGE_BYTES_MEASURED_FAMILY * 1.2)
    # The Add to held pairs view, where every band's add-on run trades yet
    # another list (a worst case: a real run's add-on lists mostly equal the
    # run as simulated and share its chunks): on its own it took the page
    # from 815,777 to 1,274,895 bytes, and beside the tier-floors-off family
    # (whose 18 binding bands each gain an add-on twin of their own) from
    # 1,098,598 to 1,784,261 (2026-09-30). On 2026-10-10: 1,289,102 and
    # 1,798,488 before the mix data; 1,424,198 and 1,987,940 with it, the
    # menus and the mix script; and, with a tag view's other curve series
    # left out as well, the constants below. Each budget is its measurement
    # + 20%.
    PAGE_BYTES_MEASURED_ADD_ON = 1_367_286
    BUDGET_ADD_ON = int(PAGE_BYTES_MEASURED_ADD_ON * 1.2)
    PAGE_BYTES_MEASURED_ADD_ON_FAMILY = 1_899_628
    BUDGET_ADD_ON_FAMILY = int(PAGE_BYTES_MEASURED_ADD_ON_FAMILY * 1.2)
    # Every trade with quotes that move on most days it is held (a run valued
    # at market), the trade lists unchanged: each curve then changes on most
    # days a trade is open, where at cost it changed only on entry and pay-out
    # days. Measured 2026-09-30 (plotly 6.9.0, pandas 3.0.3, numpy 2.4.6):
    # 1,397,952 bytes (823,634 at cost the same day, +70%), and with the
    # tier-floors-off family 1,969,146 (1,105,564 at cost, +78%). Curves stay
    # daily by choice, so each budget is its measurement + 20%, with no ceiling.
    # On 2026-10-10: 1,411,508 and 1,982,690 before the mix data; 1,518,228
    # and 2,130,430 with it, the menus and the mix script; and, with a tag
    # view's other curve series left out as well, the constants below (a
    # curve that moves daily was the costly part of each tag view).
    PAGE_BYTES_MEASURED_QUOTED = 1_342_624
    BUDGET_QUOTED = int(PAGE_BYTES_MEASURED_QUOTED * 1.2)
    PAGE_BYTES_MEASURED_QUOTED_FAMILY = 1_863_334
    BUDGET_QUOTED_FAMILY = int(PAGE_BYTES_MEASURED_QUOTED_FAMILY * 1.2)
    # The Trim to Kelly view, where every band's trimming run trades yet
    # another list (a worst case: a real trimming run mostly trades the list
    # of the run without it and shares its chunk). Measured 1,286,299 bytes
    # (2026-10-10, plotly 6.9.0 / pandas 3.0.3 / numpy 2.4.6); its budget is
    # that + 20%.
    PAGE_BYTES_MEASURED_TRIM = 1_286_299
    BUDGET_TRIM = int(PAGE_BYTES_MEASURED_TRIM * 1.2)

    @pytest.mark.parametrize(
        "family, add_on, quoted, trim",
        [(False, False, False, False), (True, False, False, False),
         (False, True, False, False), (True, True, False, False),
         (False, False, True, False), (True, False, True, False),
         (False, False, False, True)],
        ids=["tier-on-only", "tier-off-family", "add-on", "add-on-and-tier-off-family",
             "quoted", "quoted-and-tier-off-family", "trim"])
    def test_distinct_band_lists_stay_under_budget(self, monkeypatch, tmp_path, family,
                                                   add_on, quoted, trim):
        import random
        monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(dashboard.yf, "download",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
        rng = random.Random(3)
        # The quotes' own draws, so the trade lists are the unquoted variant's
        mark_rng = random.Random(11)
        # 4 categories x 2 tags: 13 views per list, 468 (702 with the family) in all
        series = [f"KXS{i:02d}" for i in range(8)]
        categories = {s: (f"Cat{i % 4}", (f"Tag{i}",)) for i, s in enumerate(series)}
        start = date(2026, 1, 5)

        def trade(i: int) -> BacktestTrade:
            s = rng.choice(series)
            entry = start + timedelta(days=rng.randint(0, 150))
            return dataclasses.replace(
                _typed_trade(rng.choice(["same_title", "time_series"]), rng.random() < 0.5,
                             entry, entry + timedelta(days=rng.randint(1, 30)),
                             rng.gauss(0, 20)),
                event_ticker=f"{s}-{i}", ticker_a=f"{s}-{i}A", title_a=f"Question {i}?")

        def point(first: int, band, **stamp) -> SweepPoint:
            # Stamped with the run's own cap, as production stamps it
            trades = sorted((trade(first + i) for i in range(40)), key=lambda t: t.entry_date)
            if quoted:
                trades = [_random_marks(t, mark_rng) for t in trades]
            return SweepPoint(k=0.75, trades=trades, spread_band=band, size_cap=0.2,
                              equity_df=backtester._build_equity_curve(trades, start, 10_000.0),
                              **stamp)

        def calibration() -> IntervalCalibration:
            obs = tuple(backtester.CalibrationObservation(
                rng.randint(1, 30), rng.uniform(0.15, 0.8), rng.random() < 0.4,
                f"{rng.choice(series)}-{j}", "Other") for j in range(300))
            return IntervalCalibration(
                pooled=backtester._calibration_bucket("POOLED", 0.0, obs), buckets=[],
                excluded_premise_violations=0, observations=obs)

        bands = [(lo, hi) for lo in config.SPREAD_BAND_SWEEP_FLOORS
                 for hi in config.SPREAD_BAND_SWEEP_CEILINGS]
        scenarios, cals = [], {}
        for band in bands:
            scenarios.append(point(0, band))
            cals[band] = calibration()
        # Drawn after every tier-on band, so the tier-on half is the same
        # either way; one run per band a tier binds at, each a new list
        off_scenarios, off_cals = [], {}
        binding = [band for band in bands if backtester._tier_floors_bind(band)] if family else []
        for band in binding:
            off_scenarios.append(point(1000, band, tier_floors=False))
            off_cals[band] = calibration()
        # The add-on family: one more list per band (and per binding band with
        # the tiers off), each a run that added a trade of its own to the
        # band's list, over the grid's one k and the run's own cap
        add_points, add_off_points = {}, {}
        if add_on:
            def added(base: SweepPoint) -> SweepPoint:
                extra = _ao_trade(base.trades[0], event=f"KXS00-{rng.randint(0, 10**9)}")
                trades = sorted([*base.trades, extra], key=lambda t: t.entry_date)
                return dataclasses.replace(
                    base, trades=trades, add_to_held=True,
                    equity_df=backtester._build_equity_curve(trades, start, 10_000.0))
            add_points = {(p.spread_band, 0.75, 0.2): added(p) for p in scenarios}
            add_off_points = {(p.spread_band, 0.75, 0.2): added(p) for p in off_scenarios}
        sweep = BacktestSweep(primary=scenarios[0], points=[scenarios[0]],
                              calibration=cals[bands[0]], label_coverage=_scn_coverage(),
                              scenarios=scenarios, calibrations_by_band=cals,
                              tier_off_scenarios=off_scenarios,
                              tier_off_calibrations_by_band=off_cals)
        if add_on:
            def fake_family(points: dict, band_list) -> _FakeCapSweep:
                return _FakeCapSweep(points, bands=tuple(band_list), ks=(0.75,), caps=(0.2,))
            sweep = dataclasses.replace(
                sweep, add_on_cap_sweep=fake_family(add_points, bands),
                add_on_tier_off_cap_sweep=(fake_family(add_off_points, binding)
                                           if binding else None))
        if trim:
            # One more list per band: the band's list with its first pair sold
            # in two parts, over the grid's one k and the run's own cap
            def trimmed(base: SweepPoint) -> SweepPoint:
                trades = sorted(_tr_split(base.trades), key=lambda t: t.entry_date)
                return dataclasses.replace(
                    base, trades=trades, trim_to_kelly=True,
                    equity_df=backtester._build_equity_curve(trades, start, 10_000.0))
            axes = {"bands": tuple(bands), "ks": (0.75,), "caps": (0.2,)}
            sweep = dataclasses.replace(sweep, trim_sweep=_FakeTrimSweep({
                (False, False): _FakeCapSweep(
                    {(p.spread_band, 0.75, 0.2): trimmed(p) for p in scenarios}, **axes),
                (True, False): _FakeCapSweep({}, **axes)}, **axes))
        out = dashboard.generate_dashboard(
            scenarios[0].trades, scenarios[0].equity_df, start, 10_000.0, sweep=sweep,
            interval_discount=0.75, series_categories=categories)
        page = out.read_text(encoding="utf-8")
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        # Nothing collapsed: every list is its own
        assert len(chunks) == (54 if family else 36) + (
            (36 + (18 if family else 0)) if add_on else 0) + (36 if trim else 0)
        assert all(band is not None for band in data["khat"])
        if family:
            assert len(binding) == 18 and len(data["bands_off"]) == 36
            assert None not in data["khat_off"]
            # A binding band's off view is its own chunk, every other band's
            # its tier-on one
            binds = data["tier_binds"]
            assert sum(binds) == 18
            for bi, own in enumerate(binds):
                assert (data["grid_off"][bi] != data["grid"][bi]) is own, bi
        else:
            assert data["bands_off"] is None and data["khat_off"] is None
            assert data["grid_off"] is None
        if add_on:
            # The view is on the page, and each add-on list is a chunk of its own
            assert data["add_state"] == "shown"
            assert len(_ao_ids(data["grid_add"])) == 36
            assert _ao_ids(data["grid_add"]).isdisjoint(_ao_ids(data["grid"]))
            assert (data["grid_add_off"] is not None) is family
        else:
            assert data["grid_add"] is None and data["grid_add_off"] is None
        if trim:
            # The view is on the page, and each trimming list is a chunk of its own
            assert data["trim_state"] == "shown"
            assert len(_ao_ids(data["grid_trim"])) == 36
            assert _ao_ids(data["grid_trim"]).isdisjoint(_ao_ids(data["grid"]))
            assert data["grid_trim_add"] is None
        else:
            assert data["grid_trim"] is None and data["trim_state"] == "not simulated"
        if trim:
            budget = self.BUDGET_TRIM
        elif quoted:
            budget = self.BUDGET_QUOTED_FAMILY if family else self.BUDGET_QUOTED
        elif add_on:
            budget = self.BUDGET_ADD_ON_FAMILY if family else self.BUDGET_ADD_ON
        else:
            budget = self.BUDGET_FAMILY if family else self.BUDGET
        assert out.stat().st_size <= budget, f"page was {out.stat().st_size} bytes"


def _cal_trade(pA: float, pB: float, outcome_a: str, outcome_b: str,
               pair_type: str = "time_series", *, nA: float | None = None,
               nB: float | None = None) -> BacktestTrade:
    """make_trade with the entry quotes, settlement sides and pair type the
    calibration section reads. The NO asks default to 1 minus their YES asks
    (books with no width), so the mid spread the section scores is the
    YES-ask gap the trade is written with."""
    return dataclasses.replace(make_trade(), entry_pA=pA, entry_pB=pB,
                               entry_nA=1.0 - pA if nA is None else nA,
                               entry_nB=1.0 - pB if nB is None else nB,
                               outcome_a=outcome_a, outcome_b=outcome_b,
                               pair_type=pair_type)


class TestSpreadCalibration:
    """The Calibration Analysis section scores each time-series trade's entry
    mid spread (the later market's midpoint minus the earlier one's) against
    whether it settled A = NO, B = YES."""

    # Spreads chosen off the 0.1-wide bin edges, which float noise can
    # straddle; books with no width, so each mid spread is the YES-ask gap
    TRADES = [
        _cal_trade(0.30, 0.65, "yes", "yes"),               # 0.35, event by A
        _cal_trade(0.20, 0.65, "no", "yes"),                # 0.45, in between
        _cal_trade(0.10, 0.35, "no", "no"),                 # 0.25, never
        _cal_trade(0.70, 0.60, "no", "yes", "same_title"),  # left out
    ]

    def test_observations_are_the_spread_and_the_in_between_cell(self):
        obs = dashboard._spread_observations(self.TRADES)
        assert [p for p, _ in obs] == pytest.approx([0.35, 0.45, 0.25])
        assert [a for _, a in obs] == [0, 1, 0]

    def test_the_prediction_is_the_entry_mid_spread_not_the_ask_gap(self):
        # DR-78: B's entry book is 0.20 wide (YES ask 0.65, NO ask 0.55) and
        # A's has none, so a 0.35 YES-ask gap reads 0.25 at the midpoints —
        # the quantity the forecast multiplies by k. It lands in the 0.2–0.3
        # bin, where the YES-ask gap would land in 0.3–0.4
        t = _cal_trade(0.30, 0.65, "no", "yes", nA=0.70, nB=0.55)
        assert dashboard._spread_observations([t]) == [
            (time_series_mid_spread(0.30, 0.70, 0.65, 0.55), 1)]
        assert dashboard._spread_observations([t]) == [(dashboard._entry_mid_spread(t), 1)]
        assert dashboard._spread_observations([t])[0][0] == pytest.approx(0.25)
        rel = dashboard._reliability([t])
        assert rel["labels"] == ["0.2–0.3"]
        assert rel["mean_pred"] == pytest.approx([0.25])
        assert rel["brier"] == pytest.approx((0.25 - 1) ** 2)
        # An earlier book with width moves it the other way: A 0.30/0.80
        # (0.10 wide) with B 0.65/0.35 reads 0.40 against a 0.35 ask gap
        wide_a = _cal_trade(0.30, 0.65, "no", "no", nA=0.80, nB=0.35)
        assert dashboard._spread_observations([wide_a])[0][0] == pytest.approx(0.40)

    def test_same_title_trades_contribute_nothing(self):
        assert dashboard._spread_observations(self.TRADES[3:]) == []

    def test_scores_match_a_hand_computation(self):
        rel = dashboard._reliability(self.TRADES)
        pairs = [(0.35, 0), (0.45, 1), (0.25, 0)]
        brier = sum((p - a) ** 2 for p, a in pairs) / 3
        ll = -sum(a * math.log(p) + (1 - a) * math.log(1 - p) for p, a in pairs) / 3
        assert rel["brier"] == pytest.approx(brier)
        assert rel["log_loss"] == pytest.approx(ll)
        assert rel["labels"] == ["0.2–0.3", "0.3–0.4", "0.4–0.5"]
        assert rel["mean_pred"] == pytest.approx([0.25, 0.35, 0.45])
        assert rel["mean_act"] == [0, 0, 1]
        assert rel["counts"] == [1, 1, 1]

    def test_no_time_series_trade_reads_dash(self):
        rel = dashboard._reliability(self.TRADES[3:])
        assert rel["brier"] is None and rel["log_loss"] is None
        assert rel["mean_pred"] == [] and rel["counts"] == []
        assert dashboard._calibration_title(None, None) == (
            "Calibration Curve (no time-series trades)")
        section = dashboard._section_calibration(self.TRADES[3:])
        assert 'id="kpi-brier"' in section and ">—</div>" in section
        assert "Calibration Curve (no time-series trades)" in section

    def test_the_section_names_its_axes(self):
        section = dashboard._section_calibration(self.TRADES)
        assert "Predicted probability: mid spread" in section
        assert "pB − pA" not in section and "pB \\u2212 pA" not in section
        assert "Time-series trades only" in section

    def test_the_caption_names_the_mid_spread(self):
        section = dashboard._section_calibration(self.TRADES)
        assert dashboard._CALIBRATION_CAPTION in section
        assert dashboard._CALIBRATION_CAPTION == (
            '<p style="color:#666;font-size:13px">Time-series trades only: the entry mid '
            "spread (B's midpoint minus A's; a midpoint is halfway between a market's YES "
            "ask and its YES bid) — the market-implied probability that the event lands "
            "between the two deadlines — against how often the pair settled A = NO, "
            "B = YES.</p>")


class TestFilterableSections:
    """A trade section renders its body whether or not it has trades — another
    band can have trades the primary does not — and swaps in "No trades."."""

    @pytest.mark.parametrize("prefix,render", [
        ("dec", lambda t, c: dashboard._section_decomposition(t)),
        ("cal", lambda t, c: dashboard._section_calibration(t)),
        ("diag", lambda t, c: dashboard._section_diagnostics(t)),
        ("risk", lambda t, c: dashboard._section_risk(t, c, 1000.0)),
    ])
    def test_the_body_is_always_rendered(self, prefix, render):
        curve = make_equity([1000.0, 1000.0])
        empty, full = render([], curve), render(_flt_trades(), curve)
        assert f'<p id="{prefix}-empty">No trades.</p>' in empty
        assert f'<div id="{prefix}-body" style="display:none">' in empty
        assert f'<p id="{prefix}-empty" style="display:none">' in full
        assert f'<div id="{prefix}-body">' in full
        assert empty.count("Plotly.newPlot(") == full.count("Plotly.newPlot(") > 0


class TestGoldenSections:
    """The seven sections that predate the scenario explorer render exactly as
    they did on main @ fe0a758. The digests were captured there, by running
    tests/dashboard_golden.py as a script against that tree (the recipe is in
    its module docstring); the same module replays them here, so the fixture,
    renderer and normaliser can never drift apart. A same-tree golden would be
    tautological.
    """

    _PATH = Path(__file__).parent / "fixtures" / "dashboard_golden_main.json"

    def _check(self) -> None:
        golden = json.loads(self._PATH.read_text(encoding="utf-8"))
        got = dashboard_golden.section_digests(dashboard_golden.render_sections())
        assert set(got) == set(golden["hashes"])
        diverged = sorted(name for name, digest in got.items()
                          if digest != golden["hashes"][name])
        env = dashboard_golden.rendering_environment()
        if diverged and env != golden["rendering_env"]:
            pytest.skip(
                f"sections {diverged} diverge from the golden, but it was captured "
                f"under {golden['rendering_env']} and this is {env}; re-capture it "
                "(tests/dashboard_golden.py's docstring) to compare in this environment")
        assert not diverged, (
            f"sections {diverged} diverged from main @ {golden['source_sha'][:7]}")

    def test_sections_match_main(self):
        self._check()

    def test_sections_match_main_under_the_stdlib_json_engine(self, monkeypatch):
        # CI installs no orjson, so plotly serialises through the stdlib json
        # module there; the golden must not depend on which one ran.
        import plotly.io as pio
        monkeypatch.setattr(pio.json.config, "default_engine", "json")
        self._check()

    def test_the_normaliser_still_sees_content(self):
        sections = dashboard_golden.render_sections()
        risk = sections["risk"]
        once = dashboard_golden.normalize(risk)
        assert once == dashboard_golden.normalize(risk)
        assert '"template":' not in once
        assert dashboard_golden.normalize(risk.replace("Kelly", "Kellx")) != once
        # Every golden section's charts carry fixed ids now (the page-wide
        # filter redraws them by id — the interval-discount section's too,
        # "kd-equity"), so the random UUID the normaliser must still replace
        # comes from a chart rendered without one, as Plotly renders it.
        assert all(not re.search(r'id="UUID"', dashboard_golden.normalize(html_))
                   for html_ in sections.values())
        unnamed = dashboard._fig_html(dashboard.go.Figure(dashboard.go.Scatter(x=[1], y=[2])))
        assert dashboard_golden._UUID_RE.search(unnamed)
        assert "UUID" in dashboard_golden.normalize(unnamed)
        assert not dashboard_golden._UUID_RE.search(dashboard_golden.normalize(unnamed))


# ─── The risk-free rate: the 8-week Treasury bill's yield, day by day ─────────

def _rates(source: str = SOURCE_API,
           fetched_at: datetime | None = datetime(2026, 9, 27, 6, 30, tzinfo=UTC)
           ) -> RiskFreeRates:
    """Two auctions — 3% on 2026-01-01, 5% on 2026-01-08 — so the rate changes
    inside every fixture window below (make_equity's opens on 2026-01-05, the
    _flt/_kc/_ex curves on 2026-01-04), and every rf-adjusted figure can be
    told from its rf = 0 value."""
    return RiskFreeRates(((date(2026, 1, 1), 0.03), (date(2026, 1, 8), 0.05)), source,
                         fetched_at)


def _steep_rates() -> RiskFreeRates:
    """_rates()'s two auctions at 60% and 95%, for whole-page tests: the page
    fixtures hold about 1% of their balance in open trades, so a real yield
    would round away at two decimals and a dropped pass-through could hide."""
    return RiskFreeRates(((date(2026, 1, 1), 0.60), (date(2026, 1, 8), 0.95)), SOURCE_API,
                         datetime(2026, 9, 27, 6, 30, tzinfo=UTC))


def _hand_ratios(returns: pd.Series, rf, periods: int) -> tuple[float, float]:
    """(Sharpe, Sortino) of a returns series against a per-period rf array,
    computed by hand from the textbook definitions — never through the
    helpers under test."""
    excess = returns.to_numpy(dtype=float) - np.asarray(rf, dtype=float) / periods
    sharpe = excess.mean() / excess.std(ddof=1) * math.sqrt(periods)
    downside = np.minimum(excess, 0.0)
    sortino = excess.mean() / math.sqrt(float(np.mean(downside ** 2))) * math.sqrt(periods)
    return float(sharpe), float(sortino)


def _hand_annual(rates: RiskFreeRates, day: date) -> float:
    """The yield in force on a day, by hand: the latest auction on or before
    it, or the first auction's before the first (0.0 with none)."""
    if not rates.auctions:
        return 0.0
    in_force = [rate for auction_day, rate in rates.auctions if auction_day <= day]
    return in_force[-1] if in_force else rates.auctions[0][1]


def _hand_hurdle(eq: pd.DataFrame, trades: list[BacktestTrade],
                 rates: RiskFreeRates) -> np.ndarray:
    """
    A strategy curve's per-row annual hurdle by hand, never through the
    helpers under test: row t is charged the yield in force on its date times
    the share of the previous row's portfolio in open trades — the cost,
    WITHOUT fees, of every trade with entry <= previous date < exit: the cost
    basis the portfolio value itself carries — and row 0 nothing.

    The rule a trade whose entry and exit fall on the curve's axis (or its
    exit after the axis's end) follows; every fixture's does.
    """
    days = [d.date() if isinstance(d, datetime) else d for d in eq["date"]]
    values = [float(v) for v in eq["portfolio_value"]]
    out = [0.0] if days else []
    for row in range(1, len(days)):
        prev = days[row - 1]
        open_cost = sum(t.total_cost for t in trades if t.entry_date <= prev < t.exit_date)
        out.append(_hand_annual(rates, days[row]) * open_cost / values[row - 1])
    return np.array(out, dtype=float)


class TestRiskFreeHurdle:
    """_sharpe / _sortino subtract rf POSITIONALLY — one scalar, or one annual
    yield per row (_rf_hurdle's) — and read 0.0 on a curve that never moves,
    whatever the rate. At rf = 0 they are the textbook expressions."""

    # Four days at 3%, then four at 5%, one per row of _RETURNS
    _VARYING = np.array([0.03] * 4 + [0.05] * 4)

    def test_a_constant_array_equals_the_scalar(self):
        full = np.full(len(_RETURNS), 0.05)
        for helper in (dashboard._sharpe, dashboard._sortino):
            assert helper(_RETURNS, rf=full) == pytest.approx(helper(_RETURNS, 0.05))
            assert helper(_RETURNS, 0.05) != pytest.approx(helper(_RETURNS))

    @pytest.mark.parametrize("periods", [365, 252])
    def test_a_time_varying_rate_matches_a_hand_computation(self, periods):
        sharpe, sortino = _hand_ratios(_RETURNS, self._VARYING, periods)
        assert dashboard._sharpe(_RETURNS, rf=self._VARYING,
                                 periods_per_year=periods) == pytest.approx(sharpe)
        assert dashboard._sortino(_RETURNS, rf=self._VARYING,
                                  periods_per_year=periods) == pytest.approx(sortino)
        # ... and neither is the flat 3% or 5% hurdle's figure
        for flat in (0.03, 0.05):
            assert dashboard._sharpe(_RETURNS, rf=self._VARYING, periods_per_year=periods) \
                != pytest.approx(dashboard._sharpe(_RETURNS, flat, periods_per_year=periods))

    def test_rf_zero_is_bit_identical_to_the_old_expression(self):
        # The textbook bodies at rf = 0: no rate (the default, None's 0.0
        # hurdle, or unavailable rates' zeros) reproduces them bit for bit
        excess = _RETURNS - 0.0 / 365
        old_sharpe = float(excess.mean() / excess.std() * np.sqrt(365))
        downside = np.minimum(excess, 0.0)
        old_sortino = float(excess.mean() / float(np.sqrt(np.mean(np.square(downside))))
                            * np.sqrt(365))
        zeros = RiskFreeRates((), SOURCE_UNAVAILABLE, None).annual_on(
            [date(2026, 1, 5) + timedelta(days=i) for i in range(len(_RETURNS))])
        # _rf_hurdle without rates reads neither the frame nor the trades
        for rf in (0.0, dashboard._rf_hurdle(None, None, None), zeros):
            assert dashboard._sharpe(_RETURNS, rf=rf) == old_sharpe
            assert dashboard._sortino(_RETURNS, rf=rf) == old_sortino

    @pytest.mark.parametrize("rf", [0.05, np.full(300, 0.05), np.linspace(0.01, 0.05, 300)],
                             ids=["scalar", "constant array", "varying array"])
    def test_a_flat_curve_reads_zero_at_any_rate(self, rf):
        # A curve with no trade. Without the guard a nonzero rate makes every
        # flat day a negative excess return (a float std of ~1e-20, not 0)
        flat = pd.Series([0.0] * 300)
        assert dashboard._sharpe(flat, rf=rf) == 0.0
        assert dashboard._sortino(flat, rf=rf) == 0.0
        # The Sortino hazard the guard removes, reproduced by hand
        excess = flat - np.asarray(rf, dtype=float) / 365
        downside = np.minimum(excess, 0.0)
        unguarded = excess.mean() / math.sqrt(float(np.mean(downside ** 2))) * math.sqrt(365)
        assert unguarded < -1.0

    def test_varies(self):
        assert dashboard._varies(_RETURNS)
        assert not dashboard._varies(pd.Series([0.0] * 5))
        assert not dashboard._varies(pd.Series([], dtype=float))
        assert not dashboard._varies(pd.Series([np.nan, np.nan]))
        assert not dashboard._varies(pd.Series([0.01, np.nan, 0.01]))

    def test_float_noise_alone_is_not_movement(self):
        # A range at or under config.FLAT_RETURN_TOLERANCE is noise, never a move
        tol = config.FLAT_RETURN_TOLERANCE
        assert not dashboard._varies(pd.Series([0.0, 1e-17, -1e-17, 0.0]))
        assert not dashboard._varies(pd.Series([0.0, tol]))
        # ...while a real move, however small, still counts
        assert dashboard._varies(pd.Series([0.0, 10 * tol]))
        assert dashboard._varies(pd.Series([0.0, 1e-9]))

    def test_a_noise_only_curve_reads_zero_not_an_absurd_ratio(self):
        # Returns that differ only by float noise, with a rate subtracted: an
        # exact max > min test would let them through and divide by a std of
        # ~1e-17
        noisy = pd.Series([0.0, 1e-17, -1e-17] * 100)
        excess = noisy - 0.05 / 365
        unguarded = excess.mean() / excess.std() * math.sqrt(365)
        assert abs(unguarded) > 1e6
        assert dashboard._sharpe(noisy, rf=0.05) == 0.0
        assert dashboard._sortino(noisy, rf=0.05) == 0.0

    def test_a_rate_of_the_wrong_length_raises(self):
        # Positional, so a misaligned array is an error, never a silent
        # broadcast or an index alignment into NaN
        short = np.full(len(_RETURNS) - 1, 0.05)
        with pytest.raises(ValueError):
            dashboard._sharpe(_RETURNS, rf=short)
        with pytest.raises(ValueError):
            dashboard._sortino(_RETURNS, rf=short)
        # ... but only on a series that varies (a flat one returns 0.0 before
        # rf is read), and a length-1 array broadcasts like the scalar
        flat = pd.Series([0.0] * 5)
        assert dashboard._sharpe(flat, rf=short) == dashboard._sortino(flat, rf=short) == 0.0
        one = np.array([0.05])
        assert dashboard._sharpe(_RETURNS, rf=one) == dashboard._sharpe(_RETURNS, 0.05)
        assert dashboard._sortino(_RETURNS, rf=one) == dashboard._sortino(_RETURNS, 0.05)

    def test_the_hurdle(self):
        dates = [date(2026, 1, 5) + timedelta(days=i) for i in range(6)]
        eq = make_equity([1000.0] * 6)
        assert dashboard._rf_hurdle(None, eq, [make_trade()]) == 0.0
        # Fully invested (the ^GSPC row): the whole yield on every date
        assert dashboard._rf_hurdle_invested(None, dates) == 0.0
        assert list(dashboard._rf_hurdle_invested(_rates(), dates)) == [
            0.03, 0.03, 0.03, 0.05, 0.05, 0.05]
        unavailable = RiskFreeRates((), SOURCE_UNAVAILABLE, None)
        assert list(dashboard._rf_hurdle_invested(unavailable, dates)) == [0.0] * 6
        assert list(dashboard._rf_hurdle(unavailable, eq, [make_trade()])) == [0.0] * 6
        # A strategy curve: the yield on the previous close's capital in open
        # trades — make_trade's $3.50 cost from 2026-01-05, its fees left out
        # (already spent), on $1,000
        t = make_trade()
        assert t.fees > 0
        share = t.total_cost / 1000.0
        assert list(dashboard._rf_hurdle(_rates(), eq, [t])) == pytest.approx(
            [0.0, 0.03 * share, 0.03 * share, 0.05 * share, 0.05 * share, 0.05 * share])

    def test_a_filtered_frame_is_read_by_position(self):
        # _view_payload cuts a curve to the page's axis without resetting its
        # index: the hurdle must follow the rows, not the index labels. The
        # trade is held 2026-01-08 to 01-11, inside the cut
        eq = make_equity([1000.0, 1000.0, 1010.0, 1004.0, 1020.0, 1015.0, 1030.0, 1025.0])
        cut = eq[eq.index >= 2]
        assert cut.index[0] == 2
        rates = _rates()
        held = [dataclasses.replace(make_trade(), entry_date=date(2026, 1, 8),
                                    exit_date=date(2026, 1, 11))]
        on_cut = dashboard._performance_kpis(cut, held, 1000.0, risk_free=rates)
        on_reset = dashboard._performance_kpis(cut.reset_index(drop=True), held,
                                               1000.0, risk_free=rates)
        assert on_cut == on_reset
        cards = {key: value for key, _, value, _ in on_cut}
        hurdle = _hand_hurdle(cut, held, rates)
        assert np.count_nonzero(hurdle) == 3
        sharpe, sortino = _hand_ratios(cut["daily_return"], hurdle, 365)
        assert (cards["sharpe"], cards["sortino"]) == (f"{sharpe:.2f}", f"{sortino:.2f}")
        assert cards["sharpe"] != "nan"


def _old_capital_deployed(trades: list[BacktestTrade], equity_df: pd.DataFrame) -> list[float]:
    """A per-row dict loop over the curve's dates: the reference
    _capital_deployed must match bit for bit."""
    entry_by_date: dict[date, float] = {}
    exit_by_date: dict[date, float] = {}
    for t in trades:
        cost = t.total_cost + t.fees
        entry_by_date[t.entry_date] = entry_by_date.get(t.entry_date, 0.0) + cost
        exit_by_date[t.exit_date] = exit_by_date.get(t.exit_date, 0.0) + cost
    invested_by_date: list[float] = []
    running_invested = 0.0
    for d in equity_df["date"]:
        d = d.date() if isinstance(d, datetime) else d
        running_invested += entry_by_date.get(d, 0.0) - exit_by_date.get(d, 0.0)
        invested_by_date.append(max(0.0, running_invested))
    return invested_by_date


def _float_bits(values: list[float]) -> list[str]:
    """Each value's exact IEEE-754 bits (float.hex keeps -0.0 apart from 0.0)."""
    return [float(v).hex() for v in values]


def _held(entry: date, exit_: date, cost: float, fees: float = 0.0) -> BacktestTrade:
    """make_trade held from `entry` to `exit_` at `cost` (+ `fees`)."""
    return dataclasses.replace(make_trade(), entry_date=entry, exit_date=exit_,
                               total_cost=cost, fees=fees)


class TestCapitalDeployedParity:
    """_capital_deployed (_deployed_on_days with fees) is bit-identical to the
    per-row loop reference for datetime.date trade dates (BacktestTrade's
    type), on every shape a curve and its trades can take. It departs from
    the loop only on a datetime or Timestamp trade date, which it matches to
    its calendar day; a None date matches nothing."""

    _D = date(2026, 1, 5)

    @classmethod
    def _curve(cls, rows: int, *, stamped: bool = False, tz: str | None = None) -> pd.DataFrame:
        eq = make_equity([1000.0 + 3.0 * (i % 5) for i in range(rows)], start=cls._D)
        if stamped:
            eq = eq.assign(date=pd.to_datetime(eq["date"]))
        if tz is not None:
            eq = eq.assign(date=pd.to_datetime(eq["date"]).dt.tz_localize(tz))
        return eq

    @classmethod
    def _fixtures(cls) -> dict[str, tuple[list[BacktestTrade], pd.DataFrame]]:
        d = cls._D
        day = timedelta(days=1)
        return {
            "no trade": ([], cls._curve(10)),
            "one trade": ([make_trade()], cls._curve(10)),
            "same-day entry and exit": ([_held(d + 2 * day, d + 2 * day, 50.0, 1.5)],
                                        cls._curve(10)),
            "overlapping": ([_held(d, d + 6 * day, 120.0, 2.1),
                             _held(d + 2 * day, d + 4 * day, 33.3, 0.7),
                             _held(d + 2 * day, d + 9 * day, 0.1, 0.07),
                             _held(d + 4 * day, d + 4 * day, 7.0, 0.3)], cls._curve(12)),
            # Before the axis: its exit subtracts with nothing added, and the
            # running sum stays below zero (floored per row, not in the sum)
            "entry before the axis": ([_held(d - 3 * day, d + 2 * day, 40.0, 1.0),
                                       _held(d + 1 * day, d + 5 * day, 25.0, 0.5)],
                                      cls._curve(8)),
            # After the axis: its exit never subtracts
            "exit after the axis": ([_held(d + 2 * day, d + 30 * day, 60.0, 1.2)],
                                    cls._curve(8)),
            "entirely off the axis": ([_held(d + 40 * day, d + 45 * day, 9.0, 0.1)],
                                      cls._curve(8)),
            "a Timestamp date column": ([make_trade(), _held(d + 1 * day, d + 3 * day, 5.0)],
                                        cls._curve(10, stamped=True)),
            "a tz-aware Timestamp column": ([make_trade()],
                                            cls._curve(10, tz="America/New_York")),
            "an empty curve": ([make_trade()], cls._curve(0)),
            "float noise": ([_held(d, d + 3 * day, 0.1, 0.2), _held(d, d + 5 * day, 0.7, 0.1),
                             _held(d + 1 * day, d + 3 * day, 1e-9, 0.3),
                             _held(d + 1 * day, d + 5 * day, 123.456, 0.0)], cls._curve(8)),
            # A NaN poisons the running sum from its entry on: the floor keeps
            # a value only when it is > 0.0, so every such row reads 0.0, as
            # max(0.0, nan) did (np.maximum would read NaN)
            "a NaN cost": ([_held(d + 1 * day, d + 4 * day, 5.0, 0.5),
                            _held(d + 3 * day, d + 6 * day, float("nan")),
                            _held(d + 5 * day, d + 7 * day, 2.0, 0.1)], cls._curve(10)),
            "a -0.0 cost": ([_held(d, d + 2 * day, -0.0), _held(d + 1 * day, d + 2 * day, -0.0),
                             _held(d + 3 * day, d + 5 * day, 4.0, -0.0)], cls._curve(8)),
            # Hand-built trades: a None date matches no row, never raising
            "a None entry or exit": ([dataclasses.replace(_held(d, d + 3 * day, 9.0, 0.4),
                                                          entry_date=None),
                                      dataclasses.replace(_held(d + 1 * day, d, 6.0, 0.2),
                                                          exit_date=None),
                                      _held(d + 2 * day, d + 5 * day, 3.0, 0.1)],
                                     cls._curve(8)),
        }

    @pytest.mark.parametrize("name", [
        "no trade", "one trade", "same-day entry and exit", "overlapping",
        "entry before the axis", "exit after the axis", "entirely off the axis",
        "a Timestamp date column", "a tz-aware Timestamp column", "an empty curve",
        "float noise", "a NaN cost", "a -0.0 cost", "a None entry or exit"])
    def test_bit_identical_to_the_loop(self, name):
        trades, eq = self._fixtures()[name]
        new = dashboard._capital_deployed(trades, eq)
        old = _old_capital_deployed(trades, eq)
        assert _float_bits(new) == _float_bits(old)
        assert all(type(v) is float for v in new)

    def test_bit_identical_on_randomized_trades(self):
        # 400 random curves and trade sets: entries and exits on, before and
        # after the axis, same-day trades, repeated dates, costs of every size
        rng = np.random.default_rng(20260927)
        for _ in range(400):
            rows = int(rng.integers(0, 40))
            eq = self._curve(rows, stamped=bool(rng.integers(0, 2)))
            trades = []
            for _ in range(int(rng.integers(0, 10))):
                entry = self._D + timedelta(days=int(rng.integers(-8, rows + 8)))
                exit_ = entry + timedelta(days=int(rng.integers(0, 15)))
                cost = float(rng.choice([rng.random() * 800, 3.5, 0.1, 1e-9]))
                trades.append(_held(entry, exit_, cost, float(rng.random() * 5)))
            assert _float_bits(dashboard._capital_deployed(trades, eq)) \
                == _float_bits(_old_capital_deployed(trades, eq))

    def test_a_timestamp_trade_date_matches_its_calendar_day(self):
        # A trade dated with a datetime or a Timestamp matches its calendar
        # day, as a row's date does (treasury.day_numbers); the loop's dict,
        # keyed on the Timestamp itself, matches no datetime.date row
        d = self._D
        eq = self._curve(6)
        plain = _held(d + timedelta(days=1), d + timedelta(days=4), 10.0, 0.5)
        want = dashboard._capital_deployed([plain], eq)
        assert want == [0.0, 10.5, 10.5, 10.5, 0.0, 0.0]
        for stamped in (pd.Timestamp(d + timedelta(days=1)) + pd.Timedelta(hours=9),
                        datetime(2026, 1, 6, 9, 0)):
            dated = dataclasses.replace(plain, entry_date=stamped)
            assert dashboard._capital_deployed([dated], eq) == want
            assert _old_capital_deployed([dated], eq) == [0.0] * 6

    def test_a_none_trade_date_never_raises(self):
        # A hand-built trade with no dates matches no row, in the Risk section
        # (never given rates) and in the hurdle alike
        eq = make_equity([1000.0, 1004.0, 998.0, 1010.0, 1007.0, 1015.0, 1012.0, 1020.0])
        dated = make_trade()
        undated = dataclasses.replace(dated, entry_date=None, exit_date=None)
        section = dashboard._section_risk([undated, dated], eq, 1000.0)
        assert "Capital Deployed Over Time" in section
        assert dashboard._capital_deployed([undated, dated], eq) \
            == dashboard._capital_deployed([dated], eq) \
            == _old_capital_deployed([undated, dated], eq)
        assert list(dashboard._rf_hurdle(_rates(), eq, [undated, dated])) \
            == list(dashboard._rf_hurdle(_rates(), eq, [dated]))
        assert not dashboard._rf_hurdle(_rates(), eq, [undated]).any()


class TestRiskFreeOnDeployedCapital:
    """A strategy curve is charged the bill's yield only on its capital in
    open trades (idle cash is taken to earn it): row t's hurdle is the yield
    in force on its date times open[t-1] / portfolio_value[t-1], open being
    the open trades' cost WITHOUT fees — the cost basis the portfolio value
    carries, so the share is at most 1 on a simulated curve. The ^GSPC row
    stays fully invested."""

    _START = date(2026, 1, 5)

    @pytest.mark.parametrize("days_held", [0, 1, 7])
    def test_a_trade_held_n_days_is_charged_exactly_n_days(self, days_held):
        entry = date(2026, 1, 9)
        trade = _held(entry, entry + timedelta(days=days_held), 2_000.0, 12.0)
        eq = backtester._build_equity_curve([trade], self._START, 10_000.0,
                                            end_date=date(2026, 1, 25))
        hurdle = dashboard._rf_hurdle(_rates(), eq, [trade])
        charged = [eq["date"].iloc[i] for i in np.flatnonzero(hurdle)]
        # The days after the entry, through the settlement day
        assert charged == [entry + timedelta(days=i) for i in range(1, days_held + 1)]
        assert hurdle == pytest.approx(_hand_hurdle(eq, [trade], _rates()))

    def test_an_idle_curve_equals_rf_zero_exactly(self):
        # No trade: a curve that moves but holds nothing overnight is charged
        # nothing, bit for bit — whatever the rate
        eq = make_equity([1000.0, 1000.0, 1010.0, 1004.0, 1020.0, 1015.0, 1030.0, 1025.0])
        for trades in ([], [_held(date(2026, 1, 7), date(2026, 1, 7), 400.0, 3.0)]):
            hurdle = dashboard._rf_hurdle(_rates(), eq, trades)
            assert not hurdle.any()
            for helper in (dashboard._sharpe, dashboard._sortino):
                assert helper(eq["daily_return"], rf=hurdle) == helper(eq["daily_return"])
            assert dashboard._performance_kpis(eq, trades, 1000.0, risk_free=_rates()) \
                == dashboard._performance_kpis(eq, trades, 1000.0)
        # A whole run's curve with one same-day trade: every ratio as at 0%
        same_day = [_held(date(2026, 1, 7), date(2026, 1, 7), 400.0, 3.0)]
        curve = backtester._build_equity_curve(same_day, self._START, 1000.0,
                                               end_date=date(2026, 1, 20))
        assert dashboard._varies(curve["daily_return"])
        assert dashboard._strategy_row(curve, 1000.0, risk_free=_rates(), trades=same_day) \
            == dashboard._strategy_row(curve, 1000.0)

    def test_a_fully_deployed_curve_equals_the_full_hurdle(self):
        # Each close's whole value is in a trade held to the next close: f = 1
        # on every row after the first, so the hurdle is the whole yield there
        # (row 0 has no previous close and is charged 0). The fees do not
        # count — already spent, they are no part of the value carried
        eq = make_equity([1000.0, 1004.0, 998.0, 1010.0, 1007.0, 1015.0, 1012.0, 1020.0])
        days = list(eq["date"])
        values = list(eq["portfolio_value"])
        trades = [_held(days[i], days[i + 1], values[i], 2.5) for i in range(len(days) - 1)]
        hurdle = dashboard._rf_hurdle(_rates(), eq, trades)
        full = dashboard._rf_hurdle_invested(_rates(), eq["date"])
        assert hurdle[0] == 0.0
        assert list(hurdle[1:]) == list(full[1:])
        full_but_row_0 = np.concatenate([[0.0], full[1:]])
        assert dashboard._sharpe(eq["daily_return"], rf=hurdle) \
            == dashboard._sharpe(eq["daily_return"], rf=full_but_row_0)
        assert dashboard._sharpe(eq["daily_return"], rf=hurdle) \
            < dashboard._sharpe(eq["daily_return"])

    def test_a_fully_deployed_row_s_share_is_its_cost_over_its_value(self):
        # A simulated curve whose one trade spends every dollar: cost + fees
        # = the whole $1,000, so the next close holds no cash and a value of
        # exactly the $990 cost basis. f = cost / V = 1, never above it; the
        # fee-inclusive capital deployed ($1,000, which the Risk chart still
        # shows) over V would have charged 1000 / 990 of the yield
        trade = _held(date(2026, 1, 7), date(2026, 1, 10), 990.0, 10.0)
        eq = backtester._build_equity_curve([trade], self._START, 1000.0,
                                            end_date=date(2026, 1, 14))
        # One auction at an exact binary fraction, so hurdle / yield is f exactly
        rates = RiskFreeRates(((date(2026, 1, 1), 0.5),), SOURCE_API, None)
        share = dashboard._rf_hurdle(rates, eq, [trade]) / 0.5
        dates, values = list(eq["date"]), list(eq["portfolio_value"])
        held = [i for i, d in enumerate(dates) if trade.entry_date <= d < trade.exit_date]
        assert [values[i] for i in held] == [990.0, 990.0, 990.0]
        for row in range(1, len(dates)):
            want = trade.total_cost / values[row - 1] if row - 1 in held else 0.0
            assert share[row] == want
        assert [share[i + 1] for i in held] == [1.0, 1.0, 1.0]
        assert share.max() <= 1.0
        # The chart's capital deployed stays fee-inclusive
        assert [dashboard._capital_deployed([trade], eq)[i] for i in held] == [1000.0] * 3

    def test_a_ruined_curve_is_charged_nothing_after_a_non_positive_close(self):
        # A hand-built curve whose value falls to 0 and below with a trade
        # still open: a row whose previous close is not positive is charged
        # nothing — no division, so no inf, no NaN, no sign flip, no warning
        eq = make_equity([1000.0, 0.0, -50.0, 20.0, 40.0])
        trade = _held(date(2026, 1, 5), date(2026, 1, 9), 100.0, 1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            hurdle = dashboard._rf_hurdle(_rates(), eq, [trade])
        # Held 01-05 to 01-09: 3% to 01-07, 5% from 01-08
        assert list(hurdle[:4]) == [0.0, 0.03 * (100.0 / 1000.0), 0.0, 0.0]
        # ... and a positive close charges again (a hand-built frame's cash is
        # not the backtester's, so its share can exceed 1)
        assert hurdle[4] == 0.05 * (100.0 / 20.0)

    def test_each_slice_is_charged_on_its_own_trades(self):
        # _view_payload's slices: each its own curve (_build_equity_curve over
        # its trades alone) and its own trades. The DOLLARS each is charged a
        # day — hurdle x previous value — sum to the whole run's, because
        # deployed capital adds up and a slice is never charged on another's
        trades = _flt_trades()
        slices = [trades[:2], trades[2:]]
        end = date(2026, 1, 25)

        def charged(listed):
            curve = backtester._build_equity_curve(listed, _FLT_START, 1000.0, end_date=end)
            hurdle = dashboard._rf_hurdle(_rates(), curve, listed)
            assert hurdle == pytest.approx(_hand_hurdle(curve, listed, _rates()))
            previous = np.concatenate([[0.0], curve["portfolio_value"].to_numpy()[:-1]])
            return hurdle * previous

        whole = charged(trades)
        assert whole.any()
        assert charged(slices[0]) + charged(slices[1]) == pytest.approx(whole)
        assert charged(slices[0]).any() and charged(slices[1]).any()

    def test_rates_without_trades_raise(self, monkeypatch):
        monkeypatch.setattr(dashboard.yf, "download", lambda *a, **k: pd.DataFrame())
        eq = make_equity([1000.0, 1010.0, 1004.0])
        with pytest.raises(TypeError, match="without the curve's trades"):
            dashboard._rf_hurdle(_rates(), eq, None)
        with pytest.raises(TypeError, match="without the curve's trades"):
            dashboard._strategy_row(eq, 1000.0, risk_free=_rates())
        with pytest.raises(TypeError, match="without the curve's trades"):
            dashboard._section_benchmark(eq, date(2026, 1, 5), 1000.0, risk_free=_rates())

    def test_without_rates_no_date_or_trade_is_read(self):
        # The strategy row on a frame with no date column: rf = None must not
        # touch it
        eq = make_equity([1000.0, 1010.0, 1004.0]).drop(columns="date")
        assert dashboard._strategy_row(eq, 1000.0)["sharpe"] \
            == f"{dashboard._sharpe(eq['daily_return']):.2f}"
        assert dashboard._rf_hurdle(None, eq, None) == 0.0

    def test_the_sp_row_is_still_charged_the_whole_yield(self, monkeypatch):
        monkeypatch.setattr(dashboard.yf, "download", lambda *a, **k: pd.DataFrame(
            {"Close": _RF_SP_CLOSE}, index=_RF_SP_INDEX))
        eq = backtester._build_equity_curve([], _FLT_START, 1000.0, end_date=date(2026, 1, 14))
        out = dashboard._section_benchmark(eq, _FLT_START, 1000.0, risk_free=_rates(),
                                           trades=[])
        sp_daily = pd.Series(_RF_SP_CLOSE, index=_RF_SP_INDEX).pct_change().dropna()
        whole = np.array([_hand_annual(_rates(), d.date()) for d in sp_daily.index])
        assert _RF_SP_ROW.search(out).group(2) == f"{_hand_ratios(sp_daily, whole, 252)[0]:.2f}"
        # ... while the strategy row, holding nothing, is charged nothing
        assert re.search(r'<td id="bench-sharpe"[^>]*>([^<]*)</td>', out).group(1) == "0.00"


# ^GSPC closes for the risk-free page: eight trading days around the second
# auction, so the S&P row's rate changes inside its window too
_RF_SP_INDEX = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08",
                               "2026-01-09", "2026-01-12", "2026-01-13", "2026-01-14"])
_RF_SP_CLOSE = [100.0, 101.0, 99.5, 102.0, 101.2, 103.0, 102.1, 104.0]

# The benchmark table's S&P row: (return, Sharpe)
_RF_SP_ROW = re.compile(r"font-weight:400'>S&P 500</td><td style='padding:8px 16px;'>"
                        r"([^<]*)</td><td style='padding:8px 16px;'>([^<]*)</td>")


def _rf_page(monkeypatch, tmp_path, sweep: BacktestSweep,
             risk_free: RiskFreeRates | None) -> str:
    """A whole page of `sweep` (TestFilterPage._page's call) with `risk_free`,
    and ^GSPC stubbed to _RF_SP_CLOSE so the benchmark's S&P row renders."""
    monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(dashboard.yf, "download", lambda *a, **k: pd.DataFrame(
        {"Close": _RF_SP_CLOSE}, index=_RF_SP_INDEX))
    out = dashboard.generate_dashboard(sweep.primary.trades, sweep.primary.equity_df,
                                       _FLT_START, 1000.0, sweep=sweep, interval_discount=0.75,
                                       series_categories=_FLT_SERIES, risk_free=risk_free)
    return out.read_text(encoding="utf-8")


@pytest.fixture(scope="class")
def rf_pages(tmp_path_factory):
    """(sweep, rates, {"rf": page with the rates, "zero": page without}) —
    one _ex_sweep (a band sweep with a size-cap sweep: every section that
    shows a ratio has data) rendered both ways, at _steep_rates()."""
    sweep, rates = _ex_sweep(), _steep_rates()
    pages = {}
    with pytest.MonkeyPatch.context() as mp:
        for name, rf in (("rf", rates), ("zero", None)):
            pages[name] = _rf_page(mp, tmp_path_factory.mktemp(name), sweep, rf)
    return sweep, rates, pages


class TestRiskFreeReachesEveryRatio:
    """generate_dashboard(risk_free=...) reaches every Sharpe and Sortino on
    the page: the performance cards, both benchmark rows, the per-k table
    (static and walked), every filter-bar chunk and the scenario explorer.
    Each figure is checked against its own curve's rf-adjusted value — the
    yield on that curve's own trades' open capital, computed by hand
    (_hand_hurdle) — AND against the same page built without rates, so a
    dropped pass-through — which would silently compute at 0% — fails here."""

    @staticmethod
    def _expected(eq: pd.DataFrame, trades: list[BacktestTrade],
                  rates: RiskFreeRates) -> tuple[str, str]:
        hurdle = _hand_hurdle(eq, trades, rates)
        return (f"{dashboard._sharpe(eq['daily_return'], rf=hurdle):.2f}",
                f"{dashboard._sortino(eq['daily_return'], rf=hurdle):.2f}")

    def test_the_performance_cards(self, rf_pages):
        sweep, rates, pages = rf_pages
        eq = sweep.primary.equity_df
        sharpe, sortino = self._expected(eq, sweep.primary.trades, rates)
        assert _page_kpi(pages["rf"], "sharpe")[0] == sharpe
        assert _page_kpi(pages["rf"], "sortino")[0] == sortino
        assert _page_kpi(pages["zero"], "sharpe")[0] == f"{dashboard._sharpe(eq['daily_return']):.2f}"
        assert _page_kpi(pages["zero"], "sortino")[0] \
            == f"{dashboard._sortino(eq['daily_return']):.2f}"
        assert _page_kpi(pages["rf"], "sharpe")[0] != _page_kpi(pages["zero"], "sharpe")[0]
        assert _page_kpi(pages["rf"], "sortino")[0] != _page_kpi(pages["zero"], "sortino")[0]

    def test_both_benchmark_rows(self, rf_pages):
        sweep, rates, pages = rf_pages
        strategy = {name: re.search(r'<td id="bench-sharpe"[^>]*>([^<]*)</td>', page).group(1)
                    for name, page in pages.items()}
        assert strategy["rf"] == self._expected(sweep.primary.equity_df,
                                                sweep.primary.trades, rates)[0]
        assert strategy["rf"] != strategy["zero"]
        # The S&P row: fully invested — the same bill's WHOLE yield on its own
        # TRADING days, per trading day (rf / 252), by hand
        sp_daily = pd.Series(_RF_SP_CLOSE, index=_RF_SP_INDEX).pct_change().dropna()
        hurdle = np.array([_hand_annual(rates, d.date()) for d in sp_daily.index])
        assert len(set(hurdle)) == 2
        sp = {name: _RF_SP_ROW.search(page).group(2) for name, page in pages.items()}
        assert sp["rf"] == f"{_hand_ratios(sp_daily, hurdle, 252)[0]:.2f}"
        assert sp["zero"] == f"{_hand_ratios(sp_daily, np.zeros(len(sp_daily)), 252)[0]:.2f}"
        assert sp["rf"] != sp["zero"]

    def test_the_per_k_table_static_and_walked(self, rf_pages):
        sweep, rates, pages = rf_pages
        expected = [self._expected(pt.equity_df, pt.trades, rates)[0] for pt in sweep.points]
        # Static: the section's own form, from the sweep's points
        static_rf, _ = _kd_table(_section_interval_discount(sweep, risk_free=rates))
        static_zero, _ = _kd_table(_section_interval_discount(sweep))
        assert [row[-1] for row in static_rf] == expected
        assert [row[-1] for row in static_zero] \
            == [f"{dashboard._sharpe(pt.equity_df['daily_return']):.2f}" for pt in sweep.points]
        assert [row[-1] for row in static_rf] != [row[-1] for row in static_zero]
        # Walked: the base block's rows at every k and cap, and the table the
        # page renders from them
        for name in ("rf", "zero"):
            kd = TestFilterPage._data(pages[name])["kd"]
            _, pc = kd["primary"]
            assert _kd_table(pages[name])[0] == kd["rows"][pc]
        kd = TestFilterPage._data(pages["rf"])["kd"]
        kd_zero = TestFilterPage._data(pages["zero"])["kd"]
        for ci, cap in enumerate(_EX_CAPS):
            for ki, k in enumerate((0.6, 0.75)):
                point = sweep.cap_sweep.points[(_KC_B0, k, cap)]["all"]
                assert kd["rows"][ci][ki][-1] == self._expected(point.equity_df, point.trades,
                                                                rates)[0]
                assert kd["rows"][ci][ki][-1] != kd_zero["rows"][ci][ki][-1]

    def test_every_chunk(self, rf_pages):
        sweep, rates, pages = rf_pages
        base = TestFilterPage._data(pages["rf"])
        chunks = TestFilterPage._chunks(pages["rf"])
        base_zero = TestFilterPage._data(pages["zero"])
        chunks_zero = TestFilterPage._chunks(pages["zero"])
        pb, pk, pc = base["primary"]
        # The primary chunk's view is the cards as rendered
        view = _view(base, chunks, pb, pk, pc, "all")
        assert view["kpi"]["sharpe"] == _page_kpi(pages["rf"], "sharpe")[0]
        assert view["kpi"]["sortino"] == _page_kpi(pages["rf"], "sortino")[0]
        assert view["bench"]["sharpe"] == self._expected(sweep.primary.equity_df,
                                                         sweep.primary.trades, rates)[0]
        # Every other scenario's chunk: its own curve's rf-adjusted figures.
        # Not (no cap, k 0.75): _ex_sweep books that cell's curve on a 1,100
        # balance under the 20% cell's very trades, which a chunk never sees —
        # chunks are shared by trade list (_list_key), so it shows the 20% one
        for ci, cap in enumerate(_EX_CAPS):
            for ki, k in enumerate((0.6, 0.75)):
                if (cap, k) == (1.0, 0.75):
                    continue
                point = sweep.cap_sweep.points[(_KC_B0, k, cap)]["all"]
                view = _view(base, chunks, 0, ki, ci, "all")
                assert (view["kpi"]["sharpe"], view["kpi"]["sortino"]) \
                    == self._expected(point.equity_df, point.trades, rates)
                assert view["kpi"]["sharpe"] \
                    != _view(base_zero, chunks_zero, 0, ki, ci, "all")["kpi"]["sharpe"]
        # Every category and tag slice: its attributed curve, on its own
        # dates, charged on its OWN trades' open capital (never the run's).
        # A slice holding little for a day or two can round to its rf = 0
        # figure even at the steep yield, so the by-hand expectation — never
        # the rf = 0 one — is what each view must equal, and most must move
        primary = sweep.primary
        none = RiskFreeRates((), SOURCE_UNAVAILABLE, None)
        moved = 0
        for key in _view_keys(chunks[base["grid"][pb][pk][pc]]):
            if key == "all":
                continue
            sliced = _view(base, chunks, pb, pk, pc, key)
            own = [primary.trades[i] for i in sliced["idx"]]
            curve = backtester._build_equity_curve(own, _FLT_START, 1000.0)
            curve = curve[pd.to_datetime(curve["date"])
                          <= pd.Timestamp(primary.equity_df["date"].iloc[-1])]
            assert (sliced["kpi"]["sharpe"], sliced["kpi"]["sortino"]) \
                == self._expected(curve, own, rates)
            assert sliced["bench"]["sharpe"] == sliced["kpi"]["sharpe"]
            sliced_zero = _view(base_zero, chunks_zero, pb, pk, pc, key)
            assert (sliced_zero["kpi"]["sharpe"], sliced_zero["kpi"]["sortino"]) \
                == self._expected(curve, own, none)
            moved += sliced["kpi"] != sliced_zero["kpi"]
        assert moved >= 3
        # The empty view (a selection with no trade) stays 0.00 at any rate
        assert base["empty"]["kpi"]["sharpe"] == base["empty"]["kpi"]["sortino"] == "0.00"

    def test_the_scenario_explorer(self, rf_pages):
        sweep, rates, pages = rf_pages
        rows = {}
        for name, page in pages.items():
            section = _ex_section(page)
            data, cap = _scn_data(section), _scn_cap(section)
            ts = data["populations"].index("time_series")
            pb, pk, _ = data["primary"]
            rows[name] = (cap["cells"][pb][pk][ts], cap["matrices"]["sharpe"][pb][pk],
                          cap["cells"][1][0][ts], cap["same_title"])
        point = sweep.cap_sweep.points[(_KC_B0, 0.75, 0.2)]["time_series"]
        expected = dashboard._point_kpis(point, risk_free=rates)
        # ... which is the point's own trades' open capital, by hand
        assert expected["sharpe"] == pytest.approx(dashboard._sharpe(
            point.equity_df["daily_return"],
            rf=_hand_hurdle(point.equity_df, point.trades, rates)))
        row, sharpe, flat, same_title = rows["rf"]
        assert row["sharpe"] == pytest.approx(expected["sharpe"])
        assert row["sortino"] == pytest.approx(expected["sortino"])
        assert sharpe == pytest.approx(expected["sharpe"])
        assert row["sharpe"] != pytest.approx(rows["zero"][0]["sharpe"])
        assert row["sortino"] != pytest.approx(rows["zero"][0]["sortino"])
        # The same-title row takes the rates too ...
        assert same_title["sharpe"] == pytest.approx(
            dashboard._point_kpis(sweep.cap_sweep.same_title()[0.2], risk_free=rates)["sharpe"])
        assert same_title["sharpe"] != pytest.approx(rows["zero"][3]["sharpe"])
        # ... and a cell that traded nothing still reads 0.0 (_varies)
        assert flat["trades"] == 0 and flat["sharpe"] == 0.0 and flat["sortino"] == 0.0

    @staticmethod
    def _page_without(monkeypatch, tmp_path, sweep, rates) -> dict[str, str]:
        """{"rf": the page with `rates`, "zero": without}, each in its own
        directory under tmp_path, with whatever the test broke still broken."""
        pages = {}
        for name, rf in (("rf", rates), ("zero", None)):
            (tmp_path / name).mkdir()
            pages[name] = _rf_page(monkeypatch, tmp_path / name, sweep, rf)
        return pages

    def test_the_explorer_fallback_through_the_page(self, monkeypatch, tmp_path):
        # The explorer's visitor cannot be made, so generate_dashboard rebuilds
        # the section from the sweep's own eager points (_explorer_fallback) —
        # with the rates
        monkeypatch.setattr(dashboard, "_new_explorer_visitor", lambda *a, **k: None)
        sweep, rates = _ex_sweep(), _steep_rates()
        pages = self._page_without(monkeypatch, tmp_path, sweep, rates)
        cells = {}
        for name, page in pages.items():
            section = _ex_section(page)
            assert dashboard._EXPLORER_OWN_CAP_HTML in section
            data, cap = _scn_data(section), _scn_cap(section)
            ts = data["populations"].index("time_series")
            pb, pk, _ = data["primary"]
            cells[name] = (cap["cells"][pb][pk][ts], cap["same_title"])
        point = next(p for p in sweep.scenarios if p.population == "time_series"
                     and p.spread_band == _KC_B0 and p.k == 0.75)
        daily = point.equity_df["daily_return"]
        hurdle = _hand_hurdle(point.equity_df, point.trades, rates)
        row, same_title = cells["rf"]
        assert row["sharpe"] == pytest.approx(dashboard._sharpe(daily, rf=hurdle))
        assert row["sortino"] == pytest.approx(dashboard._sortino(daily, rf=hurdle))
        assert row["sharpe"] != pytest.approx(cells["zero"][0]["sharpe"])
        assert row["sortino"] != pytest.approx(cells["zero"][0]["sortino"])
        # The same-title row, the run's own point, takes them too
        st = sweep.same_title_point
        assert same_title["sharpe"] == pytest.approx(dashboard._sharpe(
            st.equity_df["daily_return"], rf=_hand_hurdle(st.equity_df, st.trades, rates)))
        assert same_title["sharpe"] != pytest.approx(cells["zero"][1]["sharpe"])

    def test_the_static_interval_discount_through_the_page(self, monkeypatch, tmp_path):
        # The filter cannot be built, so the Interval Discount section renders
        # statically from the sweep's own points (_kd_from_points) — with the
        # rates, as the cards and the benchmark row beside it do
        def broken(*_a, **_k):
            raise ValueError("boom")
        monkeypatch.setattr(dashboard, "_filter_payload", broken)
        sweep, rates = _ex_sweep(), _steep_rates()
        pages = self._page_without(monkeypatch, tmp_path, sweep, rates)
        sharpes = {}
        for name, page in pages.items():
            assert html.escape(dashboard._KD_TEXT["no_bar"]) in page
            sharpes[name] = [row[-1] for row in _kd_table(page)[0]]
        assert sharpes["rf"] == [self._expected(pt.equity_df, pt.trades, rates)[0]
                                 for pt in sweep.points]
        assert sharpes["zero"] == [f"{dashboard._sharpe(pt.equity_df['daily_return']):.2f}"
                                   for pt in sweep.points]
        assert all(a != b for a, b in zip(sharpes["rf"], sharpes["zero"], strict=True))
        primary = sweep.primary
        assert _page_kpi(pages["rf"], "sharpe")[0] \
            == self._expected(primary.equity_df, primary.trades, rates)[0] \
            != _page_kpi(pages["zero"], "sharpe")[0]

    def test_the_explorer_built_from_the_sweep_takes_the_rates(self):
        # The section's own fallback (_explorer_from_sweep), as a direct call
        sweep, rates = _ex_sweep(), _rates()
        section = _section_scenario_explorer(sweep, risk_free=rates)
        data, cap = _scn_data(section), _scn_cap(section)
        ts = data["populations"].index("time_series")
        pb, pk, _ = data["primary"]
        point = next(p for p in sweep.scenarios if p.population == "time_series"
                     and p.spread_band == _KC_B0 and p.k == 0.75)
        assert cap["cells"][pb][pk][ts]["sharpe"] == pytest.approx(
            dashboard._point_kpis(point, risk_free=rates)["sharpe"])
        assert cap["cells"][pb][pk][ts]["sharpe"] != pytest.approx(
            dashboard._point_kpis(point)["sharpe"])


def _view_keys(chunk: dict) -> list[str]:
    """A chunk's view keys ("all", then its categories' and tags')."""
    return list(chunk["list"]["views"])


class TestRiskFreeHeader:
    """One header line names the rate every ratio subtracts — on every page,
    in four states (DR-66: absence is never the only signal)."""

    _EQ = make_equity([1000.0, 1000.0, 1010.0, 1004.0, 1020.0, 1015.0, 1030.0, 1025.0])

    def test_none_supplied(self):
        line = dashboard._risk_free_html(None, self._EQ)
        assert line == ('<p style="color:#616161; font-size:14px;">Risk-free rate: none '
                        "supplied — every Sharpe and Sortino ratio on this page subtracts "
                        "0%.</p>")

    def test_unavailable(self):
        line = dashboard._risk_free_html(RiskFreeRates((), SOURCE_UNAVAILABLE, None), self._EQ)
        assert line.startswith('<p style="color:#B71C1C; font-size:14px; font-weight:700;">'
                               "Risk-free rate unavailable:")
        assert ("the download from the Treasury's Fiscal Data API failed and no usable "
                "earlier download is saved, so every Sharpe and "
                "Sortino ratio on this page subtracts 0% instead of the 8-week bill's yield."
                ) in line

    def test_downloaded(self):
        rates = _rates()
        line = dashboard._risk_free_html(rates, self._EQ)
        average = float(np.mean(rates.annual_on(self._EQ["date"])))
        assert f"{average:.2%}" == "4.25%"    # 3 days at 3%, 5 at 5%
        assert line == (
            '<p style="color:#616161; font-size:14px;">Risk-free rate: the 8-week Treasury '
            "bill's auction yield (high_investment_rate, Treasury Fiscal Data) in force on "
            "each day — the latest auction on or before it — is subtracted in every Sharpe "
            "and Sortino ratio on this page, charged on the capital in open trades at each "
            "previous close: idle cash is taken to earn the same yield. The S&amp;P 500 row, "
            "when shown, is fully invested and is charged the whole yield. The yield averaged "
            "4.25% over this window. Latest auction 2026-01-08: 5.000% (downloaded "
            "2026-09-27 06:30 UTC).</p>")

    def test_cached(self):
        line = dashboard._risk_free_html(_rates(SOURCE_CACHE), self._EQ)
        assert line.startswith('<p style="color:#E65100; font-size:14px; font-weight:700;">'
                               "Risk-free rate: the 8-week Treasury bill's auction yield")
        assert "The yield averaged 4.25% over this window." in line
        assert ("charged on the capital in open trades at each previous close: idle cash is "
                "taken to earn the same yield. The S&amp;P 500 row, when shown, is fully "
                "invested") in line
        assert line.endswith(
            " The download failed: these are the yields downloaded 2026-09-27 06:30 "
            "UTC, whose latest auction (2026-01-08, 5.000%) stands for every day after it.</p>")

    def test_the_download_time_is_shown_in_utc(self):
        from datetime import timezone
        eastern = datetime(2026, 9, 27, 2, 30, tzinfo=timezone(timedelta(hours=-4)))
        line = dashboard._risk_free_html(_rates(fetched_at=eastern), self._EQ)
        assert "(downloaded 2026-09-27 06:30 UTC)" in line
        assert "at a time not recorded" in dashboard._risk_free_html(
            _rates(fetched_at=None), self._EQ)

    @pytest.mark.parametrize("stamp", ["9999-12-31T23:30:00-01:00", "0001-01-01T00:30:00+01:00"])
    def test_a_stamp_with_no_utc_instant_never_raises(self, stamp):
        # treasury._read_cache refuses such a stamp; a hand-built RiskFreeRates
        # may still carry one, and astimezone(UTC) overflows on it — the header
        # is built after the backtest has finished, so it must not raise
        rates = _rates(SOURCE_CACHE, fetched_at=datetime.fromisoformat(stamp))
        line = dashboard._risk_free_html(rates, self._EQ)
        assert "these are the yields downloaded at a time not recorded" in line

    def test_the_first_auction_note_only_before_it(self):
        note = "The 8-week bill was first auctioned on 2026-01-01; earlier days use"
        assert note not in dashboard._risk_free_html(_rates(), self._EQ)
        early = make_equity([1000.0, 1001.0, 999.0, 1003.0, 1002.0, 1004.0],
                            start=date(2025, 12, 29))
        line = dashboard._risk_free_html(_rates(), early)
        assert note in line
        # Before the first auction a day takes that auction's yield: 3% from
        # 2025-12-29 to 2026-01-03
        assert "The yield averaged 3.00% over this window." in line
        # An empty curve: no average, no note
        assert "averaged" not in dashboard._risk_free_html(_rates(), self._EQ.iloc[:0])

    @pytest.mark.parametrize("risk_free, words", [
        (None, "Risk-free rate: none supplied"),
        (_rates(), "Risk-free rate: the 8-week Treasury bill"),
        (RiskFreeRates((), SOURCE_UNAVAILABLE, None), "Risk-free rate unavailable"),
    ])
    def test_it_sits_under_the_run_settings_line(self, monkeypatch, tmp_path, risk_free,
                                                 words):
        kwargs = {} if risk_free is None else {"risk_free": risk_free}
        page = TestRunSettingsHeader._page(monkeypatch, tmp_path, **kwargs)
        assert (page.index("Period:") < page.index("Primary spread band:")
                < page.index('14px;">Live rule')
                < page.index(words) < page.index("Portfolio Performance"))
        assert page.count("Risk-free rate") == 1


class TestRiskFreeIsThreaded:
    """AST pin over dashboard.py, on VALUES, not just keywords: every
    _sharpe / _sortino call passes an rf that is a hurdle helper's result
    (_rf_hurdle, _rf_hurdle_invested) — called directly, or a name bound only
    from one in the same function — and every hurdle helper call is fed the
    page's risk_free; every `risk_free=` keyword passes a name or attribute
    spelled risk_free (so `risk_free=None` fails); every in-module call to a
    function (or class) taking a keyword-only `risk_free` passes it; and every
    call that passes risk_free to a taker that also takes a keyword-only
    `trades` passes trades (never None). Without this, a dropped pass-through
    computes that path at 0% silently and the page disagrees with itself.
    The pin checks how each pass-through is spelled, not what the name is
    bound to: a local rebinding, a walrus or a helper taking risk_free
    positionally passes it — the by-value page tests are what catch those."""

    # What the rules must find, so a refactor cannot make them vacuous
    _EXPECTED = {
        "_performance_kpis", "_section_performance", "_kd_cells", "_kd_from_points",
        "_section_interval_discount", "_point_kpis", "_section_scenario_explorer",
        "_strategy_row", "_section_benchmark", "_build_filter_grid", "_new_explorer_visitor",
        "_explorer_fallback", "_explorer_from_sweep", "_view_payload", "_list_payload",
        "generate_dashboard", "_KdVisitor", "_ExplorerVisitor", "_ChunkVisitor",
    }
    # The functions that compute a Sharpe or Sortino
    _RATIO_SITES = {"_performance_kpis", "_kd_cells", "_point_kpis", "_strategy_row",
                    "_section_benchmark"}
    _HURDLES = {"_rf_hurdle", "_rf_hurdle_invested"}

    @staticmethod
    def _kw_only(tree: ast.Module, param: str) -> set[str]:
        """Names of the functions — and classes, by their __init__ — with a
        keyword-only parameter named `param`."""
        def takes_it(fn) -> bool:
            return any(a.arg == param for a in fn.args.kwonlyargs)

        found = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__" \
                            and takes_it(item):
                        found.add(node.name)
            elif isinstance(node, ast.FunctionDef) and node.name != "__init__" \
                    and takes_it(node):
                found.add(node.name)
        return found

    @staticmethod
    def _name(call: ast.Call) -> str | None:
        fn = call.func
        return fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)

    @classmethod
    def _calls(cls, tree: ast.AST, names: set[str]) -> list[tuple[str, ast.Call]]:
        """Every call to one of `names`, by bare name or attribute."""
        return [(cls._name(node), node) for node in ast.walk(tree)
                if isinstance(node, ast.Call) and cls._name(node) in names]

    @staticmethod
    def _spelled_risk_free(node: ast.AST | None) -> bool:
        """A name or an attribute spelled risk_free (risk_free, self.risk_free,
        chunks.risk_free) — never a constant, never another name."""
        return (isinstance(node, ast.Name) and node.id == "risk_free") \
            or (isinstance(node, ast.Attribute) and node.attr == "risk_free")

    @staticmethod
    def _own_nodes(fn: ast.FunctionDef):
        """The nodes of a function's own body, not of functions or classes
        nested in it (a nested def's names are its own)."""
        stack = list(fn.body)
        while stack:
            node = stack.pop()
            yield node
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                stack.extend(ast.iter_child_nodes(node))

    def _is_fed_hurdle(self, node: ast.AST) -> bool:
        """A call to a hurdle helper whose first argument is the page's
        risk_free."""
        if not (isinstance(node, ast.Call) and self._name(node) in self._HURDLES):
            return False
        given = node.args[0] if node.args else next(
            (kw.value for kw in node.keywords if kw.arg == "risk_free"), None)
        return self._spelled_risk_free(given)

    def _rf_violations(self, source: str) -> tuple[set[str], list[str]]:
        """(the functions with a ratio call, each ratio call whose rf is not a
        fed hurdle — directly, or through a name bound only from one)."""
        tree = ast.parse(source)
        sites, bad = set(), []
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef):
                continue
            nodes = list(self._own_nodes(fn))
            # name -> whether EVERY binding of it in this function is a fed hurdle
            bound: dict[str, bool] = {}
            for node in nodes:
                if isinstance(node, ast.Assign | ast.AnnAssign | ast.AugAssign):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    ok = not isinstance(node, ast.AugAssign) and self._is_fed_hurdle(node.value)
                    for target in targets:
                        for leaf in ast.walk(target):
                            if isinstance(leaf, ast.Name):
                                bound[leaf.id] = bound.get(leaf.id, True) and ok
            for node in nodes:
                if not (isinstance(node, ast.Call)
                        and self._name(node) in {"_sharpe", "_sortino"}):
                    continue
                sites.add(fn.name)
                rf = next((kw.value for kw in node.keywords if kw.arg == "rf"),
                          node.args[1] if len(node.args) > 1 else None)
                if not (self._is_fed_hurdle(rf)
                        or (isinstance(rf, ast.Name) and bound.get(rf.id, False))):
                    bad.append(f"{self._name(node)} in {fn.name} (line {node.lineno})")
        return sites, bad

    def _violations(self, source: str) -> tuple[set[str], list[str]]:
        """(the risk_free takers, each call that drops risk_free, passes it a
        value not spelled risk_free, or passes it without trades where the
        taker takes trades)."""
        tree = ast.parse(source)
        takers = self._kw_only(tree, "risk_free")
        with_trades = takers & self._kw_only(tree, "trades")
        missing = []
        for name, call in self._calls(tree, takers):
            given = next((kw.value for kw in call.keywords if kw.arg == "risk_free"), None)
            if not self._spelled_risk_free(given):
                missing.append(f"{name} (line {call.lineno})")
            elif name in with_trades:
                trades = next((kw.value for kw in call.keywords if kw.arg == "trades"), None)
                if trades is None or (isinstance(trades, ast.Constant)
                                      and trades.value is None):
                    missing.append(f"{name} trades (line {call.lineno})")
        # Any other call passing a risk_free keyword passes the page's too
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and self._name(node) not in takers:
                for kw in node.keywords:
                    if kw.arg == "risk_free" and not self._spelled_risk_free(kw.value):
                        missing.append(f"{self._name(node)} (line {node.lineno})")
        return takers, missing

    def test_every_ratio_call_passes_a_fed_hurdle(self):
        source = inspect.getsource(dashboard)
        calls = self._calls(ast.parse(source), {"_sharpe", "_sortino"})
        assert sum(name == "_sharpe" for name, _ in calls) >= 5
        assert sum(name == "_sortino" for name, _ in calls) >= 2
        sites, bad = self._rf_violations(source)
        assert self._RATIO_SITES <= sites, sorted(self._RATIO_SITES - sites)
        assert bad == []
        # Both hurdle helpers are used, each fed the page's risk_free
        hurdles = self._calls(ast.parse(source), self._HURDLES)
        assert {name for name, _ in hurdles} == self._HURDLES
        assert all(self._is_fed_hurdle(call) for _, call in hurdles)

    def test_every_risk_free_taker_is_passed_it(self):
        source = inspect.getsource(dashboard)
        takers, missing = self._violations(source)
        assert self._EXPECTED <= takers, sorted(self._EXPECTED - takers)
        assert {"_strategy_row", "_section_benchmark"} <= self._kw_only(
            ast.parse(source), "trades")
        assert missing == []
        # Every taker but the entry point is actually called in the module,
        # so the rule above checked something for each
        called = {name for name, _ in self._calls(ast.parse(source), takers)}
        assert self._EXPECTED - {"generate_dashboard"} <= called

    def test_the_rule_catches_a_dropped_pass_through(self):
        source = ("class V:\n"
                  "    def __init__(self, *, risk_free=None):\n"
                  "        pass\n"
                  "def f(x, *, risk_free=None):\n"
                  "    return x\n"
                  "def g(risk_free, dates):\n"
                  "    return f(1, risk_free=risk_free), V(risk_free=risk_free), g(1, 2)\n"
                  "def h():\n"
                  "    return f(2), V()\n")
        takers, missing = self._violations(source)
        assert takers == {"V", "f"}
        assert missing == ["f (line 9)", "V (line 9)"]

    def test_the_rule_catches_a_risk_free_none_mutant(self):
        # risk_free passed, but as None (or as another name)
        source = ("def f(x, *, risk_free=None):\n"
                  "    return x\n"
                  "class V:\n"
                  "    def __init__(self, *, risk_free=None):\n"
                  "        self.risk_free = risk_free\n"
                  "    def go(self):\n"
                  "        return f(1, risk_free=self.risk_free), f(2, risk_free=None)\n"
                  "def g(risk_free, other):\n"
                  "    return V(risk_free=other), f(3, risk_free=risk_free)\n"
                  "def h(risk_free):\n"
                  "    return helper(risk_free=None)\n")
        _, missing = self._violations(source)
        assert sorted(missing) == sorted(["f (line 7)", "V (line 9)", "helper (line 11)"])

    def test_the_rule_catches_trades_dropped_beside_the_rates(self):
        source = ("def row(eq, *, risk_free=None, trades=None):\n"
                  "    return eq\n"
                  "def g(eq, risk_free, sel):\n"
                  "    return (row(eq, risk_free=risk_free, trades=sel),\n"
                  "            row(eq, risk_free=risk_free),\n"
                  "            row(eq, risk_free=risk_free, trades=None))\n")
        _, missing = self._violations(source)
        assert sorted(missing) == ["row trades (line 5)", "row trades (line 6)"]

    def test_the_rule_catches_an_rf_mutant(self):
        # rf passed as 0.0, a name bound from something else, a name rebound
        # after a hurdle, or a hurdle not fed the page's rates
        source = ("def a(d, eq, t, risk_free):\n"
                  "    return _sharpe(d, rf=_rf_hurdle(risk_free, eq, t))\n"
                  "def b(d, eq, t, risk_free):\n"
                  "    h = _rf_hurdle(risk_free, eq, t)\n"
                  "    return _sharpe(d, rf=h), _sortino(d, rf=h)\n"
                  "def c(d, risk_free):\n"
                  "    return _sharpe(d, rf=0.0)\n"
                  "def e(d, risk_free):\n"
                  "    h = 0.0\n"
                  "    return _sortino(d, rf=h)\n"
                  "def f(d, eq, t, risk_free):\n"
                  "    h = _rf_hurdle(risk_free, eq, t)\n"
                  "    h = 0.0\n"
                  "    return _sharpe(d, rf=h)\n"
                  "def g(d, eq, t, risk_free):\n"
                  "    return _sharpe(d, rf=_rf_hurdle(None, eq, t)), _sharpe(d)\n"
                  "def k(d, idx, risk_free):\n"
                  "    return _sharpe(d, rf=_rf_hurdle_invested(risk_free, idx))\n")
        sites, bad = self._rf_violations(source)
        assert sites == {"a", "b", "c", "e", "f", "g", "k"}
        assert sorted(bad) == ["_sharpe in c (line 7)", "_sharpe in f (line 14)",
                               "_sharpe in g (line 16)", "_sharpe in g (line 16)",
                               "_sortino in e (line 10)"]


# ─── Open trades valued at market (BacktestTrade.marks) ──────────────────────

def _marked(trade: BacktestTrade, held_a: list[float], held_b: list[float]) -> BacktestTrade:
    """
    `trade` with quotes (BacktestTrade.marks), built straight from day-end asks.

    held_a[i] and held_b[i] are the asks of the side each leg holds
    (scanner.leg_sides) at the end of day entry_date + i; a NaN is no usable
    quote, so that leg counts at its last usable ask that day (LegQuotes
    carries it forward), or at its entry price before it has had one. Past
    the last value a leg is worth its payout, read from the trade's own
    outcomes. The side a leg does not hold has no samples (the curve never
    reads it).
    """
    side_a, side_b = scanner.leg_sides(trade.pair_type)

    def quotes(ticker: str, side: str, outcome: str,
               held: list[float]) -> backtester.LegQuotes:
        """One market's LegQuotes holding `held` on `side`."""
        values = np.array(held, dtype=float)
        unused = np.full(len(values), np.nan)
        yes, no = (values, unused) if side == "yes" else (unused, values)
        paid_yes = 1.0 if outcome == "yes" else 0.0
        return backtester.LegQuotes(ticker, trade.entry_date, yes, no, trade.entry_date,
                                    np.empty(0), np.empty(0), paid_yes, 1.0 - paid_yes)

    return dataclasses.replace(trade, marks=(
        quotes(trade.ticker_a, side_a, trade.outcome_a, held_a),
        quotes(trade.ticker_b, side_b, trade.outcome_b, held_b)))


def _random_marks(trade: BacktestTrade, rng) -> BacktestTrade:
    """
    `trade` with quotes that move on most days it is held: each leg's
    held-side ask a random walk from its entry price on the cent grid, kept
    inside [0.01, 0.99] — the shape a run valued at market gives its curves.
    """
    held = (trade.exit_date - trade.entry_date).days
    prices = backtester._leg_prices_for(trade.pair_type, trade.entry_pA, trade.entry_nA,
                                        trade.entry_pB, trade.entry_nB)

    def walk(price: float) -> list[float]:
        """One leg's day-end asks, a cent-grid random walk from `price`."""
        out = []
        for _ in range(held):
            out.append(round(price, 2))
            price = min(0.99, max(0.01, price + rng.gauss(0.0, 0.02)))
        return out

    return _marked(trade, walk(prices[0]), walk(prices[1]))


class TestOpenTradesAtMarket:
    """
    A backtest trade with quotes (BacktestTrade.marks) is valued at market
    while it is open. The dashboard reads that value through the
    backtester's one day-end path — _value_steps for the per-type return
    lines, _carry_steps (through _carried_on_days) for what the risk-free
    hurdle charges — so every line, card and hurdle describes the curve
    _build_equity_curve draws. A trade without quotes is booked exactly as
    before, at cost.
    """

    _START = date(2026, 1, 5)
    _END = date(2026, 1, 15)

    @classmethod
    def _trades(cls) -> list[BacktestTrade]:
        """
        Four trades: a time-series ladder whose earlier leg dips while it is
        held, a cross-event pair that falls before it loses, a same-title pair
        with a day of no usable quote (it keeps the day before's ask), and a
        cross-event pair with no quotes (carried at cost).
        """
        base = make_trade()
        ladder = _marked(
            _typed_trade("time_series", True, date(2026, 1, 6), date(2026, 1, 10), base.profit),
            [0.30, 0.10, 0.25, 0.55], [0.40, 0.40, 0.45, 0.40])
        cross = _marked(
            dataclasses.replace(
                _typed_trade("time_series", False, date(2026, 1, 7), date(2026, 1, 9),
                             -(base.total_cost + base.fees)),
                ticker_a="CROSS-A", ticker_b="CROSS-B", outcome_a="no", outcome_b="yes"),
            [0.30, 0.15], [0.40, 0.20])
        # A same-title pair priced consistently: NO on A at 0.45, YES on B at 0.40
        n = base.n
        fees = fee_leg_exact(n, 0.45) + fee_leg_exact(n, 0.40)
        same_title = _marked(
            dataclasses.replace(
                _typed_trade("same_title", False, date(2026, 1, 6), date(2026, 1, 11), 0.0),
                ticker_a="ST-A", ticker_b="ST-B", entry_nA=0.45, entry_pB=0.40,
                total_cost=n * 0.85, fees=fees, outcome_a="no", outcome_b="no",
                actual_payoff=float(n), profit=n - n * 0.85 - fees),
            [0.45, 0.50, 0.60, 0.40, 0.48], [0.40, 0.42, float("nan"), 0.30, 0.35])
        plain = dataclasses.replace(
            _typed_trade("time_series", False, date(2026, 1, 8), date(2026, 1, 12), 2.0),
            ticker_a="PLAIN-A", ticker_b="PLAIN-B")
        return [ladder, cross, same_title, plain]

    @staticmethod
    def _cash(trades: list[BacktestTrade], day: date, initial: float) -> float:
        """The cash at a day's close, by hand: the start, less each entered
        trade's cost and fees, plus each paid-out trade's receipt."""
        return (initial - sum(t.total_cost + t.fees for t in trades if t.entry_date <= day)
                + sum(t.actual_payoff for t in trades if t.exit_date <= day))

    @staticmethod
    def _held_value(trade: BacktestTrade, day: date) -> float:
        """
        What an open trade holds at a day's close, by hand from its quotes: n x
        each leg's held-side ask that day (its entry price where the stored
        ask is NaN, i.e. before the leg's first usable ask; its payout past
        its last quote); its cost when it has no quotes.
        """
        if trade.marks is None:
            return trade.total_cost
        i = (day - trade.entry_date).days
        prices = backtester._leg_prices_for(trade.pair_type, trade.entry_pA, trade.entry_nA,
                                            trade.entry_pB, trade.entry_nB)
        total = 0.0
        for quotes, side, price in zip(trade.marks, scanner.leg_sides(trade.pair_type),
                                       prices, strict=True):
            samples = quotes.yes_days if side == "yes" else quotes.no_days
            paid = quotes.paid_yes if side == "yes" else quotes.paid_no
            value = float(samples[i]) if i < len(samples) else paid
            total += price if math.isnan(value) else value
        return trade.n * total

    def _curve(self, trades: list[BacktestTrade]) -> pd.DataFrame:
        """The trades' equity curve on $1,000, from the real builder."""
        return backtester._build_equity_curve(trades, self._START, 1000.0, end_date=self._END)

    def test_an_unquoted_trade_is_booked_exactly_at_cost_on_400_random_fixtures(self):
        # TestCapitalDeployedParity's random fixtures: with no quotes the steps
        # are the two at-cost steps, and what the curve carries is the cost
        # without fees, bit for bit
        rng = np.random.default_rng(20260927)
        d0 = TestCapitalDeployedParity._D
        for _ in range(400):
            rows = int(rng.integers(0, 40))
            eq = TestCapitalDeployedParity._curve(rows, stamped=bool(rng.integers(0, 2)))
            trades = []
            for _ in range(int(rng.integers(0, 10))):
                entry = d0 + timedelta(days=int(rng.integers(-8, rows + 8)))
                exit_ = entry + timedelta(days=int(rng.integers(0, 15)))
                cost = float(rng.choice([rng.random() * 800, 3.5, 0.1, 1e-9]))
                trades.append(_held(entry, exit_, cost, float(rng.random() * 5)))
            days = day_numbers(eq["date"])
            assert _float_bits(dashboard._carried_on_days(trades, days)) == _float_bits(
                dashboard._deployed_on_days(trades, days, include_fees=False))
            for t in trades:
                assert backtester._value_steps(t) == (
                    (t.entry_date, -t.fees), (t.exit_date, t.actual_payoff - t.total_cost))
                assert backtester._carry_steps(t) == (
                    (t.entry_date, t.total_cost), (t.exit_date, -t.total_cost))

    @pytest.mark.parametrize("name", [
        "no trade", "one trade", "same-day entry and exit", "overlapping",
        "entry before the axis", "exit after the axis", "entirely off the axis",
        "a Timestamp date column", "a tz-aware Timestamp column", "an empty curve",
        "float noise", "a NaN cost", "a -0.0 cost", "a None entry or exit"])
    def test_unquoted_carried_value_is_the_cost_without_fees(self, name):
        trades, eq = TestCapitalDeployedParity._fixtures()[name]
        days = day_numbers(eq["date"])
        assert _float_bits(dashboard._carried_on_days(trades, days)) == _float_bits(
            dashboard._deployed_on_days(trades, days, include_fees=False))

    def test_a_list_s_key_tells_apart_lists_whose_quotes_differ(self):
        # Two lists equal but for their quotes draw different curves, so they
        # never share a chunk; equal quotes built twice (the same prices) do
        trades = [make_trade(), _held(date(2026, 1, 6), date(2026, 1, 9), 2.0, 0.1)]
        marked = [_marked(trades[0], [0.2] * 7, [0.5] * 7), trades[1]]
        again = [_marked(trades[0], [0.2] * 7, [0.5] * 7), trades[1]]
        moved = [_marked(trades[0], [0.2] * 6 + [0.3], [0.5] * 7), trades[1]]
        assert marked[0].marks[0] is not again[0].marks[0]
        assert dashboard._list_key(0.75, marked) == dashboard._list_key(0.75, again)
        assert dashboard._list_key(0.75, marked) != dashboard._list_key(0.75, moved)
        assert dashboard._list_key(0.75, marked) != dashboard._list_key(0.75, trades)
        assert dashboard._list_key(0.75, marked) != dashboard._list_key(0.5, marked)
        # A list with no quotes keys on its compared fields alone, so its
        # chunks are shared exactly as they were before trades had quotes
        rows = [tuple(getattr(t, name) for name in dashboard._TRADE_FIELDS) for t in trades]
        assert dashboard._list_key(0.75, trades) == hashlib.sha256(
            repr((0.75, rows)).encode("utf-8")).hexdigest()
        # marks is the one BacktestTrade field not compared, and so not in
        # _TRADE_FIELDS: the key reads it through the quotes' fingerprints
        uncompared = [f.name for f in dataclasses.fields(BacktestTrade) if not f.compare]
        assert uncompared == ["marks"]
        assert dashboard._TRADE_FIELDS == tuple(
            f.name for f in dataclasses.fields(BacktestTrade) if f.name != "marks")

    def test_the_curve_and_its_type_lines_value_open_trades_at_market(self):
        trades = self._trades()
        ladder, cross, same_title, plain = trades
        curve = self._curve(trades)
        days = list(curve["date"])
        want = [self._cash(trades, d, 1000.0)
                + sum(self._held_value(t, d) for t in trades if t.entry_date <= d < t.exit_date)
                for d in days]
        assert list(curve["portfolio_value"]) == pytest.approx(want, abs=1e-9)
        # The leading row is the untouched start (DR-03), the last the start
        # plus every trade's profit, as at cost
        assert curve["portfolio_value"].iloc[0] == 1000.0
        assert curve["portfolio_value"].iloc[-1] == pytest.approx(
            1000.0 + sum(t.profit for t in trades), abs=1e-9)
        at_cost = self._curve([dataclasses.replace(t, marks=None) for t in trades])
        assert at_cost["portfolio_value"].iloc[-1] == pytest.approx(
            curve["portfolio_value"].iloc[-1], abs=1e-9)
        # The same-title pair's YES leg has no usable ask on 01-08: it keeps
        # 01-07's 0.42 (its last usable ask), not its 0.40 entry price
        assert self._held_value(same_title, date(2026, 1, 8)) == pytest.approx(
            same_title.n * (0.60 + 0.42))
        row = days.index(date(2026, 1, 7))
        assert curve["portfolio_value"].iloc[row] != pytest.approx(
            at_cost["portfolio_value"].iloc[row], abs=1e-6)

        lines = dashboard._return_by_trade_type(trades, curve, 1000.0)
        assert [label for label, _, _ in lines] == [
            "Same-title", "Time-series: ladder", "Time-series: cross-event"]
        total = (curve["portfolio_value"] / 1000.0 - 1.0) * 100
        summed = [sum(values) for values in zip(*(s for _, _, s in lines), strict=True)]
        assert summed == pytest.approx(list(total), abs=1e-9)
        by_label = {label: s for label, _, s in lines}
        # Each type ends at its own profit
        assert by_label["Time-series: ladder"][-1] == pytest.approx(ladder.profit / 10)
        assert by_label["Time-series: cross-event"][-1] == pytest.approx(
            (cross.profit + plain.profit) / 10)
        assert by_label["Same-title"][-1] == pytest.approx(same_title.profit / 10)
        # The ladder's line moves on a day it is held: its fees on entry, then
        # its earlier leg's fall from 0.30 to 0.10 on 01-07
        assert by_label["Time-series: ladder"][row] == pytest.approx(
            (-ladder.fees + 5 * (0.10 + 0.40) - 5 * (0.30 + 0.40)) / 10)

    def test_the_category_and_tag_slices_sum_to_the_whole(self):
        trades = [dataclasses.replace(t, event_ticker=event) for t, event in zip(
            self._trades(), ["KXAAA-1", "KXBBB-2", "KXAAA-3", "KXBBB-4"], strict=True)]
        curve = self._curve(trades)
        categories = {"KXAAA": ("CatA", ("TagA",)), "KXBBB": ("CatB", ("TagB",))}
        axis = pd.DatetimeIndex(pd.to_datetime(list(curve["date"])))
        payload = dashboard._list_payload(
            trades, curve, axis, self._START, 1000.0, categories, 0.75,
            {"CatA": 0, "CatB": 1}, {("CatA", "TagA"): 0, ("CatB", "TagB"): 1},
            dashboard._StringTable(), heads=dashboard._StringTable())
        views, n = payload["views"], len(axis)

        def total(key: str) -> list:
            """A view's return line in percent: shipped, or — for a tag view,
            which ships its curve in dollars per trade type instead ("m") —
            those dollars added up over the starting balance."""
            view = views[key]
            if "total" in view:
                return _expand(view["total"], n)
            return list(sum(np.array(_expand(part, n)) for _, part in view["m"]) / 1000.0 * 100)

        assert "total" not in views["s0"] and "m" in views["s0"]
        whole = _expand(views["all"]["total"], n)
        for keys in (("c0", "c1"), ("s0", "s1")):
            parts = [total(key) for key in keys]
            # Each total is rounded to 4 decimals of a percent
            assert [a + b for a, b in zip(*parts, strict=True)] == pytest.approx(whole, abs=2e-4)
        # Exactly, on the curves the views are drawn from (each slice's own)
        slices = [[t for t in trades if t.event_ticker.startswith(prefix)]
                  for prefix in ("KXAAA", "KXBBB")]
        pnl = [self._curve(s)["portfolio_value"] - 1000.0 for s in slices]
        assert list(pnl[0] + pnl[1]) == pytest.approx(
            list(curve["portfolio_value"] - 1000.0), abs=1e-9)

    def test_a_dip_while_a_trade_is_held_shows_in_max_drawdown(self):
        # TestDeploymentIsNotRenderedAsDrawdown's two winners on $10, the first
        # one's earlier leg falling from 0.30 to 0.10 on its second day and
        # back: $1.00 of value lost on 01-06 on top of the $0.34 of fees
        fixture = TestDeploymentIsNotRenderedAsDrawdown()
        first, second = fixture._trades()
        held = (first.exit_date - first.entry_date).days
        first = _marked(first, [0.30, 0.10] + [0.30] * (held - 2), [0.40] * held)
        trades = [first, second]
        curve = backtester._build_equity_curve(trades, fixture._START, fixture._INITIAL)
        out = dashboard._section_performance(curve, trades, fixture._START, fixture._INITIAL)
        assert "-13.4% (2026-01-06)" in out
        assert "-3.4% (2026-01-05)" not in out
        # The endpoint is the at-cost run's
        assert "+26.6%" in out
        point = SweepPoint(k=config.TIME_SERIES_INTERVAL_PROB_DISCOUNT, trades=trades,
                           equity_df=curve)
        row = _section_interval_discount(
            BacktestSweep(primary=point, points=[point], calibration=None))
        assert "-13.4%" in row and "+26.6%" in row

    def test_what_the_curve_carries_is_its_value_less_its_cash(self):
        trades = self._trades()
        curve = self._curve(trades)
        days = day_numbers(curve["date"])
        carried = dashboard._carried_on_days(trades, days)
        cash = [self._cash(trades, d, 1000.0) for d in curve["date"]]
        assert list(carried) == pytest.approx(
            list(curve["portfolio_value"].to_numpy() - np.array(cash)), abs=1e-9)
        # Not the cost: the ladder's dip and the cross-event pair's fall are in it
        at_cost = dashboard._deployed_on_days(trades, days, include_fees=False)
        assert not np.allclose(carried, at_cost)

    def test_a_fully_deployed_curve_at_market_is_charged_at_most_the_whole_yield(self):
        # One quoted trade spends every dollar, so each close holds no cash and
        # the value is the trade's value at market: the share in open trades is
        # exactly 1 while it is held, whether its value rose or fell
        trade = _marked(dataclasses.replace(make_trade(), entry_date=date(2026, 1, 7),
                                            exit_date=date(2026, 1, 10)),
                        [0.30, 0.50, 0.10], [0.40, 0.40, 0.40])
        eq = backtester._build_equity_curve([trade], self._START, trade.total_cost + trade.fees,
                                            end_date=date(2026, 1, 14))
        # One auction at an exact binary fraction, so hurdle / yield is the share exactly
        rates = RiskFreeRates(((date(2026, 1, 1), 0.5),), SOURCE_API, None)
        share = dashboard._rf_hurdle(rates, eq, [trade]) / 0.5
        dates, values = list(eq["date"]), list(eq["portfolio_value"])
        held = [i for i, d in enumerate(dates) if trade.entry_date <= d < trade.exit_date]
        assert [values[i] for i in held] == pytest.approx([3.5, 4.5, 2.5])
        assert [share[i + 1] for i in held] == [1.0, 1.0, 1.0]
        assert share.max() <= 1.0
        # Charging the cost instead would read 3.5 / 2.5 of the yield after the fall
        assert trade.total_cost / values[held[-1]] > 1.0


# ─── Several categories and tags ticked at once: the menus and the mix ────────

def _mix_observations(seed: int, n: int = 60) -> tuple:
    """Seeded k-hat observations over _MIX_SERIES' events, several per tag."""
    rng = random.Random(seed)
    names = sorted(_MIX_SERIES)
    return tuple(backtester.CalibrationObservation(
        rng.randint(1, 30), rng.uniform(0.15, 0.8), rng.random() < 0.4,
        f"{rng.choice(names)}-{rng.randint(0, 9)}", "Other") for _ in range(n))


def _mix_sweep(trades: list[BacktestTrade], observations: tuple = (),
               balance: float = _MIX_BALANCE, end: date | None = None) -> BacktestSweep:
    """A one-band sweep (0-1, k 0.75, a 20% cap) whose primary run is `trades`
    from a starting balance, with a k-hat population at that band when
    observations are given. Its curve, and so the page's dates, end on `end`
    (today when None)."""
    curve = backtester._build_equity_curve(trades, _MIX_START, balance, end_date=end)
    primary = SweepPoint(k=0.75, trades=trades, equity_df=curve, spread_band=(0.0, 1.0),
                         size_cap=0.2)
    cal = None
    if observations:
        cal = IntervalCalibration(
            pooled=backtester._calibration_bucket("POOLED", 0.0, observations), buckets=[],
            excluded_premise_violations=0, observations=observations)
    return BacktestSweep(primary=primary, points=[primary], calibration=cal,
                         label_coverage=_scn_coverage(), scenarios=[primary],
                         calibrations_by_band={(0.0, 1.0): cal}, same_event_ladders=True,
                         config_same_event_ladders=True, same_title_size_cap=0.5)


def _mix_page(monkeypatch, tmp_path, trades: list[BacktestTrade], *, risk_free=None,
              observations: tuple = (), start: date = _MIX_START,
              balance: float = _MIX_BALANCE, series: dict = _MIX_SERIES,
              end: date | None = None) -> str:
    """A whole page of _mix_sweep(trades), filed by a series map (_MIX_SERIES
    by default). The trades must cover every tag of the map, so the page's
    menus list the categories and tags in _mix_axes' order (_MIX_CATEGORIES'
    and _MIX_SUBCATS' by default) and a test can tick them by those indexes."""
    monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(dashboard.yf, "download",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    sweep = _mix_sweep(trades, observations, balance, end)
    page = dashboard.generate_dashboard(
        trades, sweep.primary.equity_df, start, balance, sweep=sweep,
        interval_discount=0.75, series_categories=series,
        risk_free=risk_free).read_text(encoding="utf-8")
    data = TestFilterPage._data(page)
    categories, subcats = _mix_axes(series)
    assert data["categories"] == categories
    assert [(data["categories"][c], tag) for c, tag in data["subcats"]] == subcats
    return page


def _mix_tick_sets(seed: int, count: int,
                   series: dict = _MIX_SERIES) -> list[tuple[list[int], list[int]]]:
    """Seeded sets of ticks over a series map: (category indexes, tag indexes),
    one to three categories, each whole or narrowed to some of its tags."""
    rng = random.Random(seed)
    categories, subcats = _mix_axes(series)
    out = []
    for _ in range(count):
        cats = sorted(rng.sample(range(len(categories)), rng.randint(1, 3)))
        tags: list[int] = []
        for c in cats:
            own = [i for i, (name, _) in enumerate(subcats) if name == categories[c]]
            if rng.random() < 0.6:
                tags += rng.sample(own, rng.randint(1, len(own)))
        out.append((cats, sorted(tags)))
    return out


def _mix_tick_steps(cats: list[int], tags: list[int], name: str) -> list:
    """The steps of a reader clearing the Category menu, ticking these
    categories and tags, and the snapshot taken once they are drawn."""
    return [_tick_category(""), *(_tick_category(c) for c in cats),
            *(_tick_tag(t) for t in tags), ["settle"], ["snap", name]]


def _mix_names(cats: list[int], tags: list[int],
               series: dict = _MIX_SERIES) -> tuple[list[str], list[str], list[str]]:
    """A set of ticks as (category names, tied tag names, the names the
    summary line spells: a category whole where none of its tags is ticked,
    else each of its ticked tags)."""
    names, subcats = _mix_axes(series)
    categories = [names[c] for c in cats]
    tied = [config.TAG_SCOPE_SEPARATOR.join(subcats[t]) for t in tags]
    spelled = []
    for c in cats:
        own = [config.TAG_SCOPE_SEPARATOR.join(subcats[t]) for t in tags
               if subcats[t][0] == names[c]]
        spelled += own or [names[c]]
    return categories, tied, spelled


def _mix_kept(cats: list[int], tags: list[int], series: dict = _MIX_SERIES):
    """The live filter's own test for a set of ticks saved as the page saves
    them: config.trade_filter of the ticked categories and tied tags."""
    categories, tied, _ = _mix_names(cats, tags, series)
    # The live rule itself, on settings holding exactly the page's ticks
    return config.trade_filter(dataclasses.replace(
        config.live_settings(), categories=tuple(categories) or None, tags=tuple(tied) or None))


def _mix_reference(data: dict, trades: list[BacktestTrade], cats: list[int],
                   tags: list[int], *, risk_free=None,
                   balance: float = _MIX_BALANCE, series: dict = _MIX_SERIES) -> dict:
    """
    What Python computes for the trades a set of ticks covers: the view
    _view_payload builds over exactly those trades and the curve
    _build_equity_curve draws from them, with its table and trade rows as
    HTML. The trades are the ones the live filter would keep
    (config.trade_filter), never the page's own resolution of the ticks.
    """
    keeps = _mix_kept(cats, tags, series)
    idx = [i for i, t in enumerate(trades)
           if keeps(*dashboard._series_labels(t.event_ticker, t.category, series))]
    sel = [trades[i] for i in idx]
    axis = pd.DatetimeIndex(pd.to_datetime(data["dates"]))
    strings = dashboard._StringTable()
    kelly_x, _ = dashboard._kelly_points(trades, 0.75)
    view = dashboard._view_payload(
        sel, idx, backtester._build_equity_curve(sel, _MIX_START, balance), axis,
        balance, series, kelly_x, lambda t, which: [0, 0], strings,
        risk_free=risk_free)
    best, worst = dashboard._best_and_worst(sel)

    def rows(listed, color) -> str:
        return "".join(dashboard._trade_row_head(t, color) + dashboard._trade_row_tail(t)
                       for t in listed)

    return {"view": view, "idx": idx, "kelly_x": kelly_x,
            "table": strings.items[view["table"]] if sel else "",
            "best": rows(best, dashboard._BEST_ROW_COLOR),
            "worst": rows(worst, dashboard._WORST_ROW_COLOR)}


def _assert_mix_is_pythons(snap: dict, ref: dict, data: dict, lst: dict) -> None:
    """Everything a snapshot shows for a selection is what Python computed for
    the same trades (_mix_reference): every card, the benchmark row, the
    category table and the trade rows as exact text, every series to 1e-6."""
    view, n = ref["view"], len(data["dates"])
    near = {"abs": 1e-6}
    assert snap["text"]["hdr-trades"] == str(view["n"])
    for key, value in view["kpi"].items():
        assert snap["text"][f"kpi-{key}"] == value, key
    for field, value in view["bench"].items():
        assert snap["text"][f"bench-{field}"] == value, field
    assert snap["html"]["dec-table"] == ref["table"]
    assert snap["html"]["diag-best"] == ref["best"]
    assert snap["html"]["diag-worst"] == ref["worst"]
    # The curves
    perf = _last_react(snap, "perf-cum")["data"]
    assert perf[0]["y"] == pytest.approx(_expand(view["total"], n), **near)
    assert [trace["name"] for trace in perf[1:]] == [label for label, _ in view["types"]]
    for trace, (_, series) in zip(perf[1:], view["types"], strict=True):
        assert trace["y"] == pytest.approx(_expand(series, n), **near)
    assert _last_react(snap, "perf-dd")["data"][0]["y"] == pytest.approx(
        _expand(view["dd"], n), **near)
    assert _last_react(snap, "bench-fig")["data"][0]["y"] == pytest.approx(
        _expand(view["eq"], n), **near)
    assert _last_react(snap, "risk-dep")["data"][0]["y"] == pytest.approx(
        _expand(view["dep"], n), **near)
    # The decomposition's bars: names and order exactly, sums to float noise
    for chart, key, values, labels in (("dec-monthly", "monthly", "y", "x"),
                                       ("dec-cat", "cat", "x", "y"),
                                       ("dec-sub", "sub", "x", "y"),
                                       ("dec-price", "price", "y", "x")):
        trace = _last_react(snap, chart)["data"][0]
        assert trace[labels] == view[key][labels], chart
        assert trace[values] == pytest.approx(view[key][values], abs=1e-9), chart
        assert trace["marker"]["color"] == view[key]["c"], chart
    assert _last_react(snap, "dec-sub")["layout"]["height"] == view["sub"]["h"]
    assert snap["heights"]["dec-sub"] == f"{view['sub']['h']}px"
    # Calibration: the scores as text, the diagram's points
    assert snap["text"]["kpi-brier"] == view["cal"]["brier"]
    assert snap["text"]["kpi-log_loss"] == view["cal"]["log_loss"]
    cal = _last_react(snap, "cal-curve")
    assert cal["layout"]["title"]["text"] == view["cal"]["title"]
    assert cal["data"][1]["x"] == pytest.approx(view["cal"]["x"], abs=1e-12)
    assert cal["data"][1]["y"] == pytest.approx(view["cal"]["y"], abs=1e-12)
    assert cal["data"][1]["text"] == view["cal"]["text"]
    assert cal["data"][1]["marker"]["size"] == view["cal"]["size"]
    # The per-trade charts read the list's arrays at exactly these trades
    idx = ref["idx"]
    assert _last_react(snap, "dec-hold")["data"][0]["x"] == [lst["hold"][i] for i in idx]
    assert _last_react(snap, "diag-ret")["data"][0]["x"] == [lst["ret"][i] for i in idx]
    assert _last_react(snap, "diag-slip")["data"][0]["x"] == [lst["slip"][i] for i in idx]
    kelly = _last_react(snap, "risk-kelly")["data"]
    assert kelly[0]["x"] == [lst["kx"][i] for i in idx]
    assert kelly[1]["x"] == pytest.approx([0, view["k11"]], abs=1e-12)


# What a snapshot draws for a selection, apart from the summary line
_MIX_TEXT_PREFIXES = ("kpi-", "bench-", "hdr-")
_MIX_HTML = ("dec-table", "diag-best", "diag-worst")


def _assert_same_drawing(a: dict, b: dict) -> None:
    """Two snapshots draw one selection alike: the same cards, table and
    trade rows as text, and every chart's last redraw the same — names, texts
    and layouts exactly, numbers to 1e-6."""
    for snap in (a, b):
        assert any(key.startswith("kpi-") for key in snap["text"])
    texts = [{k: v for k, v in snap["text"].items() if k.startswith(_MIX_TEXT_PREFIXES)}
             for snap in (a, b)]
    assert texts[0] == texts[1]
    for key in _MIX_HTML:
        assert a["html"].get(key) == b["html"].get(key), key
    charts = {r["id"] for r in a["reacts"]}
    assert charts == {r["id"] for r in b["reacts"]} and "perf-cum" in charts
    for chart in sorted(charts):
        one, two = _last_react(a, chart), _last_react(b, chart)
        assert one["layout"] == two["layout"], chart
        assert len(one["data"]) == len(two["data"]), chart
        for left, right in zip(one["data"], two["data"], strict=True):
            for key in ("x", "y", "text", "name"):
                mine, theirs = left.get(key), right.get(key)
                if isinstance(mine, list) and mine and isinstance(mine[0], (int, float)):
                    assert mine == pytest.approx(theirs, abs=1e-6), (chart, key)
                else:
                    assert mine == theirs, (chart, key)
            assert (left.get("marker") or {}).get("color") == \
                (right.get("marker") or {}).get("color"), chart


def _js_function(name: str) -> str:
    """One function of the filter script, by name, as source text (a
    one-line function, or one that closes on a line of its own)."""
    body = dashboard._FILTER_JS
    start = body.index(f"  function {name}(")
    line_end = body.index("\n", start)
    if body[start:line_end].rstrip().endswith("}"):
        return body[start:line_end + 1]
    return body[start:body.index("\n  }\n", start) + len("\n  }\n")]


def _run_js(tmp_path: Path, functions: tuple[str, ...], expression: str):
    """Run some of the filter script's own functions on their own under the
    JavaScript runtime and return the JSON an expression evaluates to."""
    runtime = _js_runtime()
    if runtime is None:
        pytest.skip("no JavaScript runtime (node or jsc) to run the filter script with")
    program = "\n".join([
        "var __emit = (typeof print === 'function') ? print : function(s) { console.log(s); };",
        *(_js_function(name) for name in functions),
        f"__emit(JSON.stringify({expression}));"])
    path = tmp_path / "functions.js"
    path.write_text(program, encoding="utf-8")
    done = subprocess.run([runtime, str(path)], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


# The page's own switch that sends every selection with a trade through the
# script's mix (never set by the page): a mix can then be compared with the
# view Python built for the same trades
_FORCE_MIX = "window.__dashForceMix = true;"

# Kalshi files a few series under tags that differ only in letter case within
# one category, and the live filter reads such tags as one. Health holds the
# twins COVID and Covid; Sports has a Covid of its own, which is no twin of theirs
_TWIN_SERIES = {"KXC0": ("Health", ("COVID",)), "KXC1": ("Health", ("Covid",)),
                "KXC2": ("Health", ("Flu",)), "KXC3": ("Sports", ("Covid",)),
                "KXC4": ("Sports", ("Golf",)), "KXC5": ("Economics", ("Fed",))}

# Twenty-four tags in five categories, for P&L sums that tie
_TIE_SERIES = {f"KXT{i:02d}": (f"Cat{i % 5}", (f"Tag{i:02d}",)) for i in range(24)}

# Run before the page's script: every block the script inflates is kept, and
# window.__mixProbe() writes, for each chunk among them, the mixes its list
# remembers (their keys, oldest first), how many it holds and how many
# strings the chunk holds
_MIX_PROBE = """
var __handed = [], __plainInflate = __inflate;
__inflate = function(el) {
  return Promise.resolve(__plainInflate(el)).then(function(d) { __handed.push(d); return d; });
};
__elements['mix-probe'] = __element('mix-probe');
window.__mixProbe = function() {
  var out = [];
  __handed.forEach(function(d) {
    if (!d || !d.list) { return; }
    var mix = d.list._mix;
    out.push([mix ? mix.keys : [], mix ? Object.keys(mix.at).length : 0, d.strings.length]);
  });
  __elements['mix-probe'].textContent = JSON.stringify(out);
};
"""


def _counted_trades(series: dict) -> list[BacktestTrade]:
    """Trades over a series map in which the tag at index i of _mix_axes'
    pairs holds exactly 2**i trades, so a trade count names the exact set of
    tags it covers."""
    _, subcats = _mix_axes(series)
    trades: list[BacktestTrade] = []
    for ticker, (category, tags) in sorted(series.items()):
        for _ in range(2 ** subcats.index((category, tags[0]))):
            n = len(trades)
            entry = _MIX_START + timedelta(days=n % 40)
            trades.append(dataclasses.replace(
                _typed_trade("time_series", False, entry, entry + timedelta(days=2),
                             1.0 + n % 3),
                event_ticker=f"{ticker}-{n}", ticker_a=f"{ticker}-{n}A",
                ticker_b=f"{ticker}-{n}B", title_a=f"Question {n}?", category="Other"))
    return sorted(trades, key=lambda t: t.entry_date)


def _mix_covered(cats: list[int], tags: list[int], series: dict = _MIX_SERIES) -> list[int]:
    """The tag indexes a set of ticks covers, by the rule the menus state: per
    ticked category its ticked tags, or every one of its tags."""
    categories, subcats = _mix_axes(series)
    return [si for si, (name, _) in enumerate(subcats)
            if categories.index(name) in cats
            and (si in tags or not any(subcats[t][0] == name for t in tags))]


class TestMixedSelection:
    """
    The Category and Tag menus are check-box dropdowns: a reader ticks as many
    as they like, and a ticked category counts in full unless some of its
    tags are ticked, then only those (the live filter's rule,
    config.trade_filter). A set of ticks that is one of Python's views draws
    that view; any other is a mix, which the page's script works out itself
    from the tags' views. These tests hold a mix to what Python computes for
    the same trades, part by part, on several fixtures — trades valued at cost,
    trades with quotes valued at market, pages with T-bill rates, pages whose
    every month holds trades, and one whose tags mostly tie on P&L — and
    pin the menus' behaviour (tags that differ only in letter case move
    together), the address the save button opens and why it will not open
    one, the k-hat figures of a mix and how many mixes a list remembers.
    Skipped without a JavaScript runtime.
    """

    # name -> (seed, quotes that move, T-bill rates on the page, starting
    # balance, the days entries are spread over, the page's last day or None
    # for today). With rates the balance is small and the yield steep (it
    # changes inside the window), so the trades hold a share of the balance
    # that the yield visibly moves each ratio by. The first four pages run
    # to today with every trade in their first three months, so most of
    # their months are flat and every Median Monthly Return reads +0.0%; the
    # "every-month" pages end on a fixed day with trades in every month (five
    # months on one, six on the other: an odd and an even count to take the
    # median of), so that card is a real figure there, different from one
    # mix to the next
    FIXTURES = {
        "at-cost": (21, False, False, _MIX_BALANCE, 60, None),
        "at-market": (22, True, False, _MIX_BALANCE, 60, None),
        "at-market-with-rates": (23, True, True, 400.0, 60, None),
        "at-cost-with-rates": (24, False, True, 150.0, 60, None),
        "every-month-at-cost": (25, False, False, 150.0, 120, date(2026, 5, 31)),
        "every-month-at-market-with-rates": (26, True, True, 400.0, 150, date(2026, 6, 30)),
    }
    # The fixtures whose months all hold trades
    EVERY_MONTH = ("every-month-at-cost", "every-month-at-market-with-rates")

    @staticmethod
    def _rates(with_rates):
        return _steep_rates() if with_rates else None

    @pytest.mark.parametrize("fixture", sorted(FIXTURES))
    def test_a_mix_is_what_python_computes_for_the_same_trades(
            self, monkeypatch, tmp_path, fixture):
        seed, quoted, kind, balance, spread, end = self.FIXTURES[fixture]
        rates = self._rates(kind)
        trades = _mix_trades(seed, n=60, quoted=quoted, spread=spread)
        page = _mix_page(monkeypatch, tmp_path, trades, risk_free=rates, balance=balance,
                         end=end)
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        lst = chunks[0]["list"]
        assert (data["mix"]["rf"] is None) is (rates is None)
        assert all("m" in view for key, view in lst["views"].items() if key.startswith("s"))
        sets = _mix_tick_sets(seed, 10)
        steps = [["wait"]]
        for i, (cats, tags) in enumerate(sets):
            steps += _mix_tick_steps(cats, tags, f"mix{i}")
        snaps = _run_script(tmp_path, page, steps, strict=True, setup_js=_FORCE_MIX)
        n_all = lst["views"]["all"]["n"]
        mixed = 0
        medians = set()
        for i, (cats, tags) in enumerate(sets):
            snap = snaps[f"mix{i}"]
            ref = _mix_reference(data, trades, cats, tags, risk_free=rates, balance=balance)
            assert ref["view"]["n"] > 0, (cats, tags)
            _assert_mix_is_pythons(snap, ref, data, lst)
            medians.add(ref["view"]["kpi"]["median_monthly"])
            # The line names the ticks and says the page combined them
            _, _, spelled = _mix_names(cats, tags)
            assert snap["text"]["flt-summary"] == dashboard._filter_summary_text(
                data["text"], _phrase(data, 0, 0, 0), True,
                dashboard._selection_name(spelled, data["text"]), ref["view"]["n"], n_all,
                mix=True)
            mixed += len(ref["view"]["types"]) > 1
        # The fixture exercises what it claims to: several trade types in one
        # mix, a ratio that the hurdle moves when rates are on the page, and
        # — where every month holds trades — a Median Monthly Return that is
        # a real figure, different from one mix to the next
        assert mixed
        if fixture in self.EVERY_MONTH:
            assert data["dates"][-1] == end.isoformat()
            assert len(medians - {"+0.0%", "-0.0%"}) >= 4, medians
        if rates is not None:
            charged = [_mix_reference(data, trades, cats, tags, risk_free=rates,
                                      balance=balance)["view"]["kpi"] for cats, tags in sets]
            free = [_mix_reference(data, trades, cats, tags, balance=balance)["view"]["kpi"]
                    for cats, tags in sets]
            assert [k["sharpe"] for k in charged] != [k["sharpe"] for k in free]
            assert [k["sortino"] for k in charged] != [k["sortino"] for k in free]

    @staticmethod
    def _each_tag_alone_is_pythons(monkeypatch, tmp_path, trades, *, rates=None,
                                   balance=_MIX_BALANCE, end=None) -> dict:
        """Tick each tag of a page of `trades` on its own and hold the series
        the script draws for it — the return line, each trade type's line,
        the drawdown and the curve in dollars — to the ones _view_payload
        computes for that tag, rounded as Python rounds them. Returns the
        list's tag views by tag index, as the page ships them."""
        page = _mix_page(monkeypatch, tmp_path, trades, risk_free=rates, balance=balance,
                         end=end)
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        lst, n = chunks[0]["list"], len(data["dates"])
        steps = [["wait"]]
        for si, (category, _) in enumerate(_MIX_SUBCATS):
            assert "m" in lst["views"][f"s{si}"]
            steps += _mix_tick_steps([_MIX_CATEGORIES.index(category)], [si], f"tag{si}")
        snaps = _run_script(tmp_path, page, steps, strict=True)
        exact = {"abs": 1e-9}
        several = 0
        for si, (category, _) in enumerate(_MIX_SUBCATS):
            snap = snaps[f"tag{si}"]
            ref = _mix_reference(data, trades, [_MIX_CATEGORIES.index(category)], [si],
                                 risk_free=rates, balance=balance)
            view = ref["view"]
            assert view["n"] == lst["views"][f"s{si}"]["n"] > 0
            # Python's own view of the tag, not a mix: its cards as shipped
            assert data["text"]["mix_note"] not in snap["text"]["flt-summary"]
            assert {k: snap["text"][f"kpi-{k}"] for k in view["kpi"]} == view["kpi"]
            perf = _last_react(snap, "perf-cum")["data"]
            assert perf[0]["y"] == pytest.approx(_expand(view["total"], n), **exact)
            assert [trace["name"] for trace in perf[1:]] == [label for label, _ in view["types"]]
            for trace, (_, series) in zip(perf[1:], view["types"], strict=True):
                assert trace["y"] == pytest.approx(_expand(series, n), **exact)
            assert _last_react(snap, "perf-dd")["data"][0]["y"] == pytest.approx(
                _expand(view["dd"], n), **exact)
            assert _last_react(snap, "bench-fig")["data"][0]["y"] == pytest.approx(
                _expand(view["eq"], n), **exact)
            several += len(view["types"]) > 1
        assert several
        return {si: lst["views"][f"s{si}"] for si in range(len(_MIX_SUBCATS))}

    @pytest.mark.parametrize("fixture", sorted(FIXTURES))
    def test_one_tag_alone_draws_the_series_python_computes_for_it(
            self, monkeypatch, tmp_path, fixture):
        # A tag's view ships its curve once, as dollars per trade type ("m"),
        # and the script works the four series it draws out of that. For
        # every tag of every parity fixture they are the very series
        # _view_payload computes for that tag
        seed, quoted, kind, balance, spread, end = self.FIXTURES[fixture]
        views = self._each_tag_alone_is_pythons(
            monkeypatch, tmp_path, _mix_trades(seed, n=60, quoted=quoted, spread=spread),
            rates=self._rates(kind), balance=balance, end=end)
        # None of the four is shipped beside "m" on these pages
        assert not any(name in view for view in views.values()
                       for name in dashboard._MIX_CURVE_SERIES)

    @pytest.mark.parametrize("quoted", [False, True], ids=["at-cost", "at-market"])
    def test_a_series_that_would_round_the_other_way_is_shipped_as_it_was(
            self, monkeypatch, tmp_path, quoted):
        # Profits on a half-cent grid from $10,000: many days' values sit
        # exactly on a rounding tie (x.xx5 dollars; x.xxxx5 percent), where
        # float noise decides which way Python rounded. A series the script
        # would not rebuild to exactly Python's values stays in the view,
        # so one tag alone is still drawn as Python computed it
        rng = random.Random(81)
        trades = []
        for t in _mix_trades(81, n=60):
            profit = rng.randint(-4000, 4000) * 5 / 1000
            trades.append(dataclasses.replace(
                t, profit=profit, actual_payoff=profit + t.total_cost + t.fees))
        if quoted:
            mark_rng = random.Random(82)
            trades = [_random_marks(t, mark_rng) if i % 3 else t for i, t in enumerate(trades)]
        views = self._each_tag_alone_is_pythons(monkeypatch, tmp_path, trades)
        kept = [name for view in views.values() for name in dashboard._MIX_CURVE_SERIES
                if name in view]
        # Both happen on this page: series kept for a tie, and series left out
        assert kept and len(kept) < len(views) * len(dashboard._MIX_CURVE_SERIES)

    def test_a_tag_whose_list_cannot_be_mixed_is_drawn_from_its_shipped_series(
            self, monkeypatch, tmp_path):
        # No tag of this list carries "m" (the slices' curves open two days
        # after the page's dates), so each keeps the four series Python
        # shipped, and one tag alone is drawn from them as they stand
        trades = _mix_trades(47, n=44)
        page = _mix_page(monkeypatch, tmp_path, trades, start=_MIX_START + timedelta(days=2))
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        n, si = len(data["dates"]), 3
        view = chunks[0]["list"]["views"][f"s{si}"]
        assert "m" not in view and all(k in view for k in dashboard._MIX_CURVE_SERIES)
        snap = _run_script(tmp_path, page, [
            ["wait"], _tick_tag(si), ["settle"], ["snap", "tag"]], strict=True)["tag"]
        perf = _last_react(snap, "perf-cum")["data"]
        assert perf[0]["y"] == _expand(view["total"], n)
        assert [(trace["name"], trace["y"]) for trace in perf[1:]] == [
            (label, _expand(series, n)) for label, series in view["types"]]
        assert _last_react(snap, "perf-dd")["data"][0]["y"] == _expand(view["dd"], n)
        assert _last_react(snap, "bench-fig")["data"][0]["y"] == _expand(view["eq"], n)

    @pytest.mark.parametrize("quoted", [False, True], ids=["at-cost", "at-market"])
    def test_a_selection_python_has_a_view_of_is_drawn_alike_by_the_mix(
            self, monkeypatch, tmp_path, quoted):
        # One category, one tag, every tag of a category and every category:
        # each is one of Python's own views, which the page draws as it
        # stands. Sent through the mix instead, the drawing is the same.
        trades = _mix_trades(31, n=44, quoted=quoted)
        page = _mix_page(monkeypatch, tmp_path, trades, risk_free=_rates())
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        views = chunks[0]["list"]["views"]
        own = [i for i, (name, _) in enumerate(_MIX_SUBCATS) if name == _MIX_CATEGORIES[0]]
        every = list(range(len(_MIX_CATEGORIES)))
        cases = {"category": ([1], []), "tag": ([0], [own[1]]), "every_tag": ([0], own),
                 "every_category": (every, [])}
        steps = [["wait"]]
        for name, (cats, tags) in cases.items():
            steps += _mix_tick_steps(cats, tags, name)
        plain = _run_script(tmp_path, page, steps, strict=True)
        mixed = _run_script(tmp_path, page, steps, strict=True, setup_js=_FORCE_MIX)
        note = data["text"]["mix_note"]
        expected = {"category": views["c1"], "tag": views[f"s{own[1]}"],
                    "every_tag": views["c0"], "every_category": views["all"]}
        for name in cases:
            _assert_same_drawing(plain[name], mixed[name])
            # The page's own drawing is Python's view; only the mix says it combined
            assert plain[name]["text"]["hdr-trades"] == str(expected[name]["n"])
            for key, value in expected[name]["kpi"].items():
                assert plain[name]["text"][f"kpi-{key}"] == value, (name, key)
            assert note not in plain[name]["text"]["flt-summary"]
            assert note in mixed[name]["text"]["flt-summary"]

    def test_ticks_follow_the_rule_and_the_menus_say_what_is_ticked(
            self, monkeypatch, tmp_path):
        trades = _mix_trades(41, n=44)
        page = _mix_page(monkeypatch, tmp_path, trades)
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        views = chunks[0]["list"]["views"]
        cat0 = [i for i, (name, _) in enumerate(_MIX_SUBCATS) if name == "Cat0"]
        cat1 = [i for i, (name, _) in enumerate(_MIX_SUBCATS) if name == "Cat1"]
        snaps = _run_script(tmp_path, page, [
            ["snap", "loaded"], ["wait"], ["snap", "ready"],
            # A tag ticked under "All categories" ticks its category
            _tick_tag(cat0[1]), ["settle"], ["snap", "tag"],
            # A second category, whole
            _tick_category(1), ["settle"], ["snap", "two"],
            # ... narrowed to one of its tags, then a second tag of the first
            _tick_tag(cat1[0]), _tick_tag(cat0[2]), ["settle"], ["snap", "narrowed"],
            # Unticking a category unticks its tags
            _tick_category(0), ["settle"], ["snap", "dropped"],
            # The Tag menu's clearing box clears the tags alone
            _tick_tag(""), ["settle"], ["snap", "tags_cleared"],
            # Clicked again with nothing to clear, it changes and redraws nothing
            _tick_tag(""), ["settle"], ["snap", "again"],
            # The Category menu's clearing box clears both menus
            _tick_tag(cat1[1]), _tick_category(""), ["settle"], ["snap", "cleared"]],
            strict=True, pre=(("flt-cat", "0,2"), ("flt-tag", "1")))
        loaded, ready = snaps["loaded"], snaps["ready"]
        # A browser's restored ticks are cleared before anything is enabled
        for menu, words in (("flt-cat", "All categories"), ("flt-tag", "All tags")):
            assert loaded["menus"][menu]["ticked"] == [] and loaded["menus"][menu]["all"]
            assert loaded["menus"][menu]["disabled"] and all(loaded["menus"][menu]["boxesDisabled"])
            assert loaded["menus"][menu]["className"] == "flt-multi flt-off"
            assert not ready["menus"][menu]["disabled"]
            assert not any(ready["menus"][menu]["boxesDisabled"])
            assert ready["menus"][menu]["className"] == "flt-multi"
            assert ready["menus"][menu]["label"] == words
        assert loaded["reacts"] == ready["reacts"] == []

        def state(snap) -> tuple:
            return (snap["menus"]["flt-cat"]["ticked"], snap["menus"]["flt-tag"]["ticked"],
                    snap["menus"]["flt-cat"]["label"], snap["menus"]["flt-tag"]["label"])

        def shown_tags(snap) -> list[int]:
            return [i for i, (_, _, hidden) in enumerate(snap["menus"]["flt-tag"]["rows"])
                    if not hidden]

        n = {key: view["n"] for key, view in views.items()}
        tag = snaps["tag"]
        assert state(tag) == ([0], [cat0[1]], "Cat0", _MIX_SUBCATS[cat0[1]][1])
        # One category ticked: its tags alone are listed, by their own names
        assert shown_tags(tag) == cat0
        assert [tag["menus"]["flt-tag"]["rows"][i][0] for i in cat0] == [
            f"{_MIX_SUBCATS[i][1]} ({n[f's{i}']})" for i in cat0]
        assert not tag["menus"]["flt-cat"]["all"] and not tag["menus"]["flt-tag"]["all"]
        assert tag["text"]["hdr-trades"] == str(n[f"s{cat0[1]}"])
        two = snaps["two"]
        assert state(two) == ([0, 1], [cat0[1]], "2 categories",
                              " · ".join(_MIX_SUBCATS[cat0[1]]))
        # Two categories ticked: both categories' tags, named with their category
        assert shown_tags(two) == cat0 + cat1
        assert [two["menus"]["flt-tag"]["rows"][i][0] for i in cat0 + cat1] == [
            f"{' · '.join(_MIX_SUBCATS[i])} ({n[f's{i}']})" for i in cat0 + cat1]
        # One tag of Cat0 and all of Cat1: a mix, in the page's words
        covered = n[f"s{cat0[1]}"] + n["c1"]
        assert two["text"]["hdr-trades"] == str(covered)
        assert two["text"]["flt-summary"] == dashboard._filter_summary_text(
            data["text"], _phrase(data, 0, 0, 0), True,
            f"{' · '.join(_MIX_SUBCATS[cat0[1]])}, Cat1", covered, n["all"], mix=True)
        narrowed = snaps["narrowed"]
        assert state(narrowed) == ([0, 1], sorted([cat0[1], cat0[2], cat1[0]]),
                                   "2 categories", "3 tags")
        assert narrowed["text"]["hdr-trades"] == str(
            n[f"s{cat0[1]}"] + n[f"s{cat0[2]}"] + n[f"s{cat1[0]}"])
        dropped = snaps["dropped"]
        assert state(dropped) == ([1], [cat1[0]], "Cat1", _MIX_SUBCATS[cat1[0]][1])
        assert shown_tags(dropped) == cat1
        # One tag alone is Python's own view of it: no word of a mix
        assert dropped["text"]["flt-summary"] == dashboard._filter_summary_text(
            data["text"], _phrase(data, 0, 0, 0), True, " · ".join(_MIX_SUBCATS[cat1[0]]),
            n[f"s{cat1[0]}"], n["all"])
        cleared = snaps["tags_cleared"]
        assert state(cleared) == ([1], [], "Cat1", "All tags")
        assert cleared["menus"]["flt-tag"]["all"] and cleared["reacts"]
        assert cleared["text"]["hdr-trades"] == str(n["c1"])
        again = snaps["again"]
        assert state(again) == state(cleared) and again["menus"]["flt-tag"]["all"]
        assert again["reacts"] == []
        done = snaps["cleared"]
        assert state(done) == ([], [], "All categories", "All tags")
        assert done["menus"]["flt-cat"]["all"] and done["menus"]["flt-tag"]["all"]
        assert shown_tags(done) == list(range(len(_MIX_SUBCATS)))
        assert done["text"]["hdr-trades"] == str(n["all"])
        assert html.escape(done["text"]["flt-summary"]) in page

    def test_more_than_three_names_are_counted_not_listed(self, monkeypatch, tmp_path):
        page = _mix_page(monkeypatch, tmp_path, _mix_trades(42, n=44))
        data = TestFilterPage._data(page)
        cat0 = [i for i, (name, _) in enumerate(_MIX_SUBCATS) if name == "Cat0"]
        snap = _run_script(tmp_path, page, [
            ["wait"], _tick_category(1), _tick_category(2), *(_tick_tag(t) for t in cat0),
            ["settle"], ["snap", "s"]], strict=True)["s"]
        names = [" · ".join(_MIX_SUBCATS[t]) for t in cat0] + ["Cat1", "Cat2"]
        assert dashboard._selection_name(names) == (
            "Cat0 · Tag0, Cat0 · Tag3, Cat0 · Tag6, +2 more")
        assert dashboard._selection_name(names[:3]) == "Cat0 · Tag0, Cat0 · Tag3, Cat0 · Tag6"
        assert dashboard._selection_name(["Cat1"]) == "Cat1"
        # Every category's every trade, however the ticks got there: the
        # whole list, named by its ticks
        n = TestFilterPage._chunks(page)[0]["list"]["views"]["all"]["n"]
        assert snap["text"]["flt-summary"] == dashboard._filter_summary_text(
            data["text"], _phrase(data, 0, 0, 0), True,
            dashboard._selection_name(names, data["text"]), n, n)
        assert snap["menus"]["flt-cat"]["label"] == "3 categories"
        assert snap["menus"]["flt-tag"]["label"] == "3 tags"

    def test_a_menu_stays_open_while_ticking_and_closes_outside_or_on_escape(
            self, monkeypatch, tmp_path):
        page = _mix_page(monkeypatch, tmp_path, _mix_trades(43, n=20))
        snaps = _run_script(tmp_path, page, [
            ["wait"], ["open", "flt-cat"], _tick_category(0), _tick_category(1), ["settle"],
            ["snap", "ticking"],
            # A click on one of the menu's own boxes is inside it
            ["outside", "flt-cat-1"], ["snap", "inside"],
            ["outside", "flt-summary"], ["snap", "outside"],
            ["open", "flt-tag"], ["key", "a"], ["snap", "other_key"],
            ["key", "Escape"], ["snap", "escape"]], strict=True)
        assert snaps["ticking"]["menus"]["flt-cat"]["open"]
        assert snaps["ticking"]["menus"]["flt-cat"]["ticked"] == [0, 1]
        assert snaps["inside"]["menus"]["flt-cat"]["open"]
        assert not snaps["outside"]["menus"]["flt-cat"]["open"]
        assert snaps["other_key"]["menus"]["flt-tag"]["open"]
        assert not snaps["escape"]["menus"]["flt-tag"]["open"]
        # Closing a menu changes no tick
        assert snaps["escape"]["menus"]["flt-cat"]["ticked"] == [0, 1]

    def test_a_chunk_that_fails_to_load_puts_the_ticks_back(self, monkeypatch, tmp_path):
        # Two categories are on screen at the primary scenario; a third is
        # ticked and another band chosen, whose chunk cannot be read: the
        # band and the ticks all go back to what the sections still show
        page = TestFilterPage()._page(monkeypatch, tmp_path)
        data = TestFilterPage._data(page)
        other = data["grid"][1][1][0]
        sports = data["categories"].index("Sports")
        commodities = data["categories"].index("Commodities")
        science = data["categories"].index("Science")
        snaps = _run_script(tmp_path, page, [
            ["wait"], _tick_category(sports), _tick_category(commodities), ["settle"],
            ["snap", "shown"],
            ["set", "flt-band", "1"], ["fire", "flt-band"], _tick_category(science),
            ["snap", "pending"], ["reject", f"dash-chunk-{other}"], ["snap", "failed"]],
            strict=True, deferred=(f"dash-chunk-{other}",))
        shown, pending, failed = snaps["shown"], snaps["pending"], snaps["failed"]
        ticks = sorted([sports, commodities])
        assert shown["menus"]["flt-cat"]["ticked"] == ticks
        assert pending["menus"]["flt-cat"]["ticked"] == sorted([*ticks, science])
        assert pending["reacts"] == []
        assert failed["menus"]["flt-cat"]["ticked"] == ticks
        assert failed["menus"]["flt-cat"]["label"] == "2 categories"
        assert failed["selects"]["flt-band"]["value"] == "0"
        assert failed["reacts"] == []
        assert failed["text"]["hdr-trades"] == shown["text"]["hdr-trades"]
        assert failed["text"]["flt-summary"].startswith("The filter could not load the data")

    def test_a_list_that_cannot_be_mixed_says_so_and_puts_the_ticks_back(
            self, monkeypatch, tmp_path, caplog):
        # The slices' curves open two days after the page's own dates, so no
        # tag of the list carries the curve a mix adds up
        trades = _mix_trades(44, n=44)
        with caplog.at_level(logging.WARNING):
            page = _mix_page(monkeypatch, tmp_path, trades,
                             start=_MIX_START + timedelta(days=2))
        assert [r for r in caplog.records
                if "cannot combine categories and tags" in r.getMessage()]
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        views = chunks[0]["list"]["views"]
        assert not any("m" in view for view in views.values())
        cat1 = [i for i, (name, _) in enumerate(_MIX_SUBCATS) if name == "Cat1"]
        snaps = _run_script(tmp_path, page, [
            ["wait"], _tick_category(1), ["settle"], ["snap", "one"],
            _tick_category(2), ["settle"], ["snap", "refused"],
            # One tag is still Python's own view
            _tick_tag(cat1[0]), ["settle"], ["snap", "tag"]],
            strict=True)
        one, refused = snaps["one"], snaps["refused"]
        assert one["text"]["hdr-trades"] == str(views["c1"]["n"]) and one["reacts"]
        # Refused in Python's words: nothing drawn, the second tick taken back
        phrase = _phrase(data, 0, 0, 0)
        assert refused["text"]["flt-summary"] == (
            f"Not available: Cat1, Cat2 cannot be shown together for {phrase} (the page's "
            "build log says why). The page still shows Cat1.")
        assert refused["text"]["flt-summary"] == data["text"]["mix_unavailable"].format(
            selection="Cat1, Cat2", failed=phrase, shown="Cat1", at="")
        # The scenario is named once: the ticks were refused where they stood
        assert refused["text"]["flt-summary"].count(phrase) == 1
        assert refused["reacts"] == []
        assert refused["menus"]["flt-cat"]["ticked"] == [1]
        assert refused["menus"]["flt-cat"]["label"] == "Cat1"
        assert refused["text"]["hdr-trades"] == str(views["c1"]["n"])
        assert not refused["buttons"]["flt-save"]
        assert snaps["tag"]["text"]["hdr-trades"] == str(views[f"s{cat1[0]}"]["n"])
        assert snaps["tag"]["menus"]["flt-tag"]["ticked"] == [cat1[0]]

    def test_the_saved_filter_keeps_exactly_what_the_page_shows(self, monkeypatch, tmp_path):
        # The clicked address, read by the defaults server's own parser, gives
        # live settings whose filter (config.trade_filter) keeps exactly the
        # category · tag pairs the ticks cover on the page — over every pair
        # the page lists — for seeded sets of ticks
        save_config_live_defaults()
        current = config.read_saved_live_defaults()
        trades = _mix_trades(51, n=44)
        page = _mix_page(monkeypatch, tmp_path, trades)
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        views = chunks[0]["list"]["views"]
        pairs = [(data["categories"][c], tag) for c, tag in data["subcats"]]
        assert pairs == _MIX_SUBCATS
        sets = _mix_tick_sets(51, 12)
        steps = [["wait"]]
        for i, (cats, tags) in enumerate(sets):
            steps += _mix_tick_steps(cats, tags, f"drawn{i}")[:-1]
            steps += [["click", "flt-save"], ["snap", f"set{i}"]]
        snaps = _run_script(tmp_path, page, steps, strict=True)
        narrowed = 0
        for i, (cats, tags) in enumerate(sets):
            snap = snaps[f"set{i}"]
            [opened] = snap["opened"]
            query = opened["url"].partition("?")[2]
            sent = parse_qs(query, strict_parsing=True)
            categories, tied, _ = _mix_names(cats, tags)
            # One category per ticked category, one tied tag per ticked tag
            assert sent["category"] == categories
            assert sent.get("tag", []) == tied
            settings, _ = defaults_server._proposal(defaults_server._params(query), current)
            assert settings.categories == tuple(categories)
            assert settings.tags == (tuple(tied) or None)
            keeps = config.trade_filter(settings)
            kept = [si for si, pair in enumerate(pairs) if keeps(*pair)]
            # The page's own resolution of the ticks: per ticked category its
            # ticked tags, or every one of its tags
            covered = [si for si, (name, _) in enumerate(pairs)
                       if _MIX_CATEGORIES.index(name) in cats
                       and (si in tags or not any(pairs[t][0] == name for t in tags))]
            assert kept == covered
            # ... which is what the page drew
            assert snap["text"]["hdr-trades"] == str(
                sum(views[f"s{si}"]["n"] for si in covered if f"s{si}" in views))
            narrowed += bool(tags) and len(covered) < len(pairs)
        assert narrowed

    def test_every_ticked_tag_is_sent_even_when_they_are_all_of_a_categorys(
            self, monkeypatch, tmp_path):
        # The page lists only the tags this backtest saw, so a category whose
        # listed tags are all ticked is still saved tag by tag: naming it
        # alone would let a live run trade tags nobody ticked. Commodities
        # has one listed tag, Sports two.
        save_config_live_defaults()
        current = config.read_saved_live_defaults()
        page = TestSaveLiveDefaultsButton._page(monkeypatch, tmp_path)
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        views = chunks[data["grid"][data["primary"][0]][data["primary"][1]][
            data["primary"][2]]]["list"]["views"]
        commodities = data["categories"].index("Commodities")
        sports = data["categories"].index("Sports")
        oil = data["subcats"].index([commodities, "Oil & Gas"])
        sports_tags = [i for i, (c, _) in enumerate(data["subcats"]) if c == sports]
        assert [c for c, _ in data["subcats"]].count(commodities) == 1
        assert len(sports_tags) == 2
        snaps = _run_script(tmp_path, page, [
            ["wait"], _tick_tag(oil), ["settle"], ["click", "flt-save"], ["snap", "one_tag"],
            _tick_category(""), *(_tick_tag(t) for t in sports_tags), ["settle"],
            ["click", "flt-save"], ["snap", "both_tags"],
            _tick_tag(""), ["settle"], ["click", "flt-save"], ["snap", "whole"]], strict=True)
        expected = {
            "one_tag": (["Commodities"], ["Commodities · Oil & Gas"], f"c{commodities}"),
            "both_tags": (["Sports"], ["Sports · Basketball", "Sports · Hockey"],
                          f"c{sports}"),
            "whole": (["Sports"], [], f"c{sports}"),
        }
        for name, (categories, tied, view) in expected.items():
            [opened] = snaps[name]["opened"]
            query = opened["url"].partition("?")[2]
            sent = parse_qs(query, strict_parsing=True)
            assert sent["category"] == categories and sent.get("tag", []) == tied, name
            settings, _ = defaults_server._proposal(defaults_server._params(query), current)
            assert settings.categories == tuple(categories)
            assert settings.tags == (tuple(tied) or None)
            # The figures are the category's own view either way: the same trades
            assert snaps[name]["text"]["hdr-trades"] == str(views[view]["n"]), name
        # A tag the page never listed is kept only by the whole category
        tagged = config.trade_filter(defaults_server._proposal(defaults_server._params(
            snaps["both_tags"]["opened"][0]["url"].partition("?")[2]), current)[0])
        whole = config.trade_filter(defaults_server._proposal(defaults_server._params(
            snaps["whole"]["opened"][0]["url"].partition("?")[2]), current)[0])
        assert tagged("Sports", "Hockey") and not tagged("Sports", "Tennis")
        assert whole("Sports", "Tennis")

    def test_more_names_than_the_server_takes_cannot_be_saved(self, monkeypatch, tmp_path):
        # The page carries the server's own limit; past it the button is shut
        monkeypatch.setattr(dashboard, "DEFAULTS_SERVER_MAX_FILTER_NAMES", 2)
        page = _mix_page(monkeypatch, tmp_path, _mix_trades(52, n=44))
        data = TestFilterPage._data(page)
        assert data["save"]["max_names"] == 2
        cat0 = [i for i, (name, _) in enumerate(_MIX_SUBCATS) if name == "Cat0"]
        snaps = _run_script(tmp_path, page, [
            ["wait"], _tick_category(0), _tick_category(1), ["settle"], ["snap", "two"],
            _tick_category(2), ["settle"], ["click", "flt-save"], ["snap", "three"],
            _tick_category(2), _tick_category(1), *(_tick_tag(t) for t in cat0), ["settle"],
            ["click", "flt-save"], ["snap", "three_tags"],
            _tick_tag(cat0[0]), ["settle"], ["snap", "two_tags"]], strict=True)
        assert not snaps["two"]["buttons"]["flt-save"]
        assert snaps["three"]["buttons"]["flt-save"] and snaps["three"]["opened"] == []
        assert snaps["three_tags"]["buttons"]["flt-save"]
        assert snaps["three_tags"]["opened"] == []
        assert not snaps["two_tags"]["buttons"]["flt-save"]
        # While the ticks are too many the button's hover text says so, in
        # Python's words; otherwise it reads as Python rendered it
        too_many = "Cannot save: more than 2 categories, or more than 2 tags, are ticked. " \
                   "Untick some."
        assert data["text"]["save_too_many"].format(n=2) == too_many
        assert snaps["three"]["titles"]["flt-save"] == too_many
        assert snaps["three_tags"]["titles"]["flt-save"] == too_many
        assert snaps["two"]["titles"]["flt-save"] == dashboard._SAVE_TITLE
        assert snaps["two_tags"]["titles"]["flt-save"] == dashboard._SAVE_TITLE

    def test_the_khat_figures_of_a_mix_are_its_groups_pooled(self, monkeypatch, tmp_path):
        observations = _mix_observations(61)
        trades = _mix_trades(61, n=44)
        page = _mix_page(monkeypatch, tmp_path, trades, observations=observations)
        data = TestFilterPage._data(page)
        cat0 = [i for i, (name, _) in enumerate(_MIX_SUBCATS) if name == "Cat0"]
        cases = {"mix": ([0, 1], [cat0[0], cat0[2]]), "two_whole": ([1, 2], []),
                 "category": ([2], []), "tag": ([0], [cat0[1]])}
        steps = [["wait"], ["set", "khat-group", "band"], ["fire", "khat-group"]]
        for name, (cats, tags) in cases.items():
            steps += _mix_tick_steps(cats, tags, name)
        steps += [["set", "khat-group", "tag"], ["fire", "khat-group"], ["snap", "by_tag"],
                  ["set", "khat-group", "category"], ["fire", "khat-group"],
                  ["snap", "by_category"]]
        snaps = _run_script(tmp_path, page, steps, strict=True)
        k_shown = data["ks"][data["primary"][1]]["value"]
        for name, (cats, tags) in cases.items():
            keeps = _mix_kept(cats, tags)
            joined = [o for o in observations if keeps(*dashboard._series_labels(
                o.event_ticker, o.category, _MIX_SERIES))]
            assert joined, name
            # The one definition of k-hat, over the joined groups' observations
            bucket = backtester._calibration_bucket("", 0.0, joined)
            snap = snaps[name]
            assert snap["text"]["kpi-khat"] == f"{bucket.empirical_k:.3f}", name
            text, color = dashboard._khat_delta(bucket.empirical_k, k_shown)
            assert snap["text"]["kpi-khat_delta"] == text, name
            assert snap["colors"]["kpi-khat_delta"] == color, name
            # Grouped by spread band: the band's row is the same pooled figures
            stat = dashboard._khat_stat(joined)
            assert snap["rows"]["khat-rows"] == [[data["bands"][0]["label"], *stat["cells"]]]
            react = _last_react(snap, "khat-fig")
            assert react["data"][0]["text"] == [stat["text"]]
            assert react["data"][0]["x"] == pytest.approx([bucket.empirical_k], abs=1e-12)
            _, _, spelled = _mix_names(cats, tags)
            assert react["layout"]["title"]["text"] == (
                f"Empirical k̂ by spread band — {dashboard._selection_name(spelled)}")
        # Grouped by tag or category, every ticked row is marked (the last
        # ticks: one tag of Cat0, which lists that category's tags alone)
        colors = data["styles"]["khat"]
        by_tag = _last_react(snaps["by_tag"], "khat-fig")["data"][0]
        assert by_tag["y"][0] == "All Cat0"
        marked = [label for label, color in zip(by_tag["y"], by_tag["marker"]["color"],
                                                strict=True) if color == colors["selected"]]
        assert marked == [_MIX_SUBCATS[cat0[1]][1]]
        by_category = _last_react(snaps["by_category"], "khat-fig")["data"][0]
        assert [label for label, color in zip(by_category["y"], by_category["marker"]["color"],
                                              strict=True)
                if color == colors["selected"]] == ["Cat0"]

    def test_several_ticked_categories_are_each_marked_in_the_khat_chart(
            self, monkeypatch, tmp_path):
        observations = _mix_observations(62)
        page = _mix_page(monkeypatch, tmp_path, _mix_trades(62, n=44),
                         observations=observations)
        data = TestFilterPage._data(page)
        cat0 = [i for i, (name, _) in enumerate(_MIX_SUBCATS) if name == "Cat0"]
        cat2 = [i for i, (name, _) in enumerate(_MIX_SUBCATS) if name == "Cat2"]
        snaps = _run_script(tmp_path, page, [
            ["wait"], _tick_category(0), _tick_category(2), _tick_tag(cat0[0]), ["settle"],
            ["snap", "by_category"],
            ["set", "khat-group", "tag"], ["fire", "khat-group"], ["snap", "by_tag"]],
            strict=True)
        colors = data["styles"]["khat"]

        def marked(snap) -> list[str]:
            trace = _last_react(snap, "khat-fig")["data"][0]
            return [label for label, color in zip(trace["y"], trace["marker"]["color"],
                                                  strict=True) if color == colors["selected"]]

        assert marked(snaps["by_category"]) == ["Cat0", "Cat2"]
        # By tag, with more than one category ticked: every tag the ticks
        # cover — the one ticked tag of Cat0 and all of Cat2 — among all tags
        by_tag = _last_react(snaps["by_tag"], "khat-fig")["data"][0]
        assert by_tag["y"][0] == "All tags"
        assert marked(snaps["by_tag"]) == [
            " · ".join(_MIX_SUBCATS[i]) for i in [cat0[0], *cat2]
            if f"s{i}" in data["khat"][0]["groups"]]

    def test_tags_that_differ_only_in_letter_case_are_ticked_and_saved_as_one(
            self, monkeypatch, tmp_path):
        # The live filter reads "Health · COVID" and "Health · Covid" as one
        # tag, so the page ticks and unticks them together: whatever a reader
        # clicks, the tags the page shows are exactly the ones the saved
        # filter keeps. Tag i holds 2**i trades, so the trade count on the
        # page names the exact set of tags shown.
        save_config_live_defaults()
        current = config.read_saved_live_defaults()
        categories, pairs = _mix_axes(_TWIN_SERIES)
        page = _mix_page(monkeypatch, tmp_path, _counted_trades(_TWIN_SERIES),
                         series=_TWIN_SERIES)
        data = TestFilterPage._data(page)
        upper, lower = pairs.index(("Health", "COVID")), pairs.index(("Health", "Covid"))
        flu, other = pairs.index(("Health", "Flu")), pairs.index(("Sports", "Covid"))
        health = categories.index("Health")
        twins = {upper: {upper, lower}, lower: {upper, lower}}
        rng = random.Random(71)

        def clicked(boxes: list[tuple[str, int]]) -> tuple[list, set, set]:
            """A reader's clicks on category ("c") and tag ("t") boxes from
            cleared menus: the steps, and the ticks the menus' rule leaves."""
            cats: set = set()
            tags: set = set()
            steps = [_tick_category("")]
            for kind, i in boxes:
                if kind == "c":
                    steps.append(_tick_category(i))
                    if i in cats:
                        cats.discard(i)
                        tags = {t for t in tags if pairs[t][0] != categories[i]}
                    else:
                        cats.add(i)
                else:
                    steps.append(_tick_tag(i))
                    if i in tags:
                        tags -= twins.get(i, {i})
                    else:
                        tags |= twins.get(i, {i})
                        cats.add(categories.index(pairs[i][0]))
            return steps, cats, tags

        def seeded() -> list[tuple[str, int]]:
            """One to six clicks, each on a box the menus show at that point."""
            boxes: list[tuple[str, int]] = []
            for _ in range(rng.randint(1, 6)):
                _, cats, _ = clicked(boxes)
                if rng.random() < 0.4:
                    boxes.append(("c", rng.randrange(len(categories))))
                else:
                    boxes.append(("t", rng.choice(
                        [t for t, (name, _) in enumerate(pairs)
                         if not cats or categories.index(name) in cats])))
            return boxes

        cases = [[("t", upper)], [("t", lower)], [("t", upper), ("t", lower)],
                 [("c", health), ("t", lower), ("t", flu)], [("t", other)],
                 [("t", lower), ("t", other), ("t", upper)]]
        cases += [seeded() for _ in range(18)]
        steps, expected = [["wait"]], []
        for i, boxes in enumerate(cases):
            ticks, cats, tags = clicked(boxes)
            steps += [*ticks, ["settle"], ["click", "flt-save"], ["snap", f"case{i}"]]
            expected.append((sorted(cats), sorted(tags)))
        snaps = _run_script(tmp_path, page, steps, strict=True)
        seen = set()
        for i, (cats, tags) in enumerate(expected):
            snap = snaps[f"case{i}"]
            assert snap["menus"]["flt-cat"]["ticked"] == cats, cases[i]
            assert snap["menus"]["flt-tag"]["ticked"] == tags, cases[i]
            [opened] = snap["opened"]
            query = opened["url"].partition("?")[2]
            sent = parse_qs(query, strict_parsing=True)
            # Every ticked name is sent, both spellings of a twin included
            assert sent.get("category", []) == [categories[c] for c in cats]
            assert sent.get("tag", []) == [config.TAG_SCOPE_SEPARATOR.join(pairs[t])
                                           for t in tags]
            settings, _ = defaults_server._proposal(defaults_server._params(query), current)
            keeps = config.trade_filter(settings)
            kept = {si for si, pair in enumerate(pairs) if keeps(*pair)}
            # The tags on the page, read off its trade count (2**i per tag)
            count = int(snap["text"]["hdr-trades"])
            shown = {si for si in range(len(pairs)) if count >> si & 1}
            assert shown == kept, cases[i]
            seen.add(frozenset(shown))
        # One click on either spelling ticks both; the label counts both; a
        # click on either unticks both and leaves the category ticked
        for i in (0, 1):
            assert expected[i] == ([health], [upper, lower])
            assert snaps[f"case{i}"]["menus"]["flt-tag"]["label"] == "2 tags"
            assert snaps[f"case{i}"]["text"]["hdr-trades"] == str(2 ** upper + 2 ** lower)
        assert expected[2] == ([health], [])
        # The server keeps the first spelling and drops the case-only repeat
        saved, _ = defaults_server._proposal(defaults_server._params(
            snaps["case0"]["opened"][0]["url"].partition("?")[2]), current)
        assert saved.tags == ("Health · COVID",)
        # Sports' own Covid is another category's tag: no twin of Health's
        assert expected[4] == ([categories.index("Sports")], [other])
        assert len(seen) >= 8
        # Python names the twins for the script (by str.casefold): one group,
        # within Health
        assert data["tag_twins"] == [[upper, lower]] and data["category_twins"] == []

    def test_equal_sums_are_ordered_as_python_orders_them(self, monkeypatch, tmp_path):
        # Most of 24 tags end on the very same P&L (+1.00, 0.00 or -1.00), so
        # the bars and the table rows are mostly ties: both sides keep ties
        # in name order, whichever way the chart or the table sorts
        trades: list[BacktestTrade] = []
        for i, ticker in enumerate(sorted(_TIE_SERIES)):
            parts = [2.5, -2.5] if i in (3, 8) else [-1.0] if i in (4, 9, 14) else [1.0]
            for profit in parts:
                n = len(trades)
                entry = _MIX_START + timedelta(days=3 + (n * 7) % 120)
                trades.append(dataclasses.replace(
                    _typed_trade("time_series" if n % 2 == 0 else "same_title", n % 4 == 0,
                                 entry, entry + timedelta(days=[0, 1, 5][n % 3]), profit),
                    event_ticker=f"{ticker}-{n}", ticker_a=f"{ticker}-{n}A",
                    ticker_b=f"{ticker}-{n}B", title_a=f"Question {n}?", category="Other"))
        trades.sort(key=lambda t: t.entry_date)
        page = _mix_page(monkeypatch, tmp_path, trades, series=_TIE_SERIES, balance=1000.0)
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        lst = chunks[0]["list"]
        categories, pairs = _mix_axes(_TIE_SERIES)
        cat0 = [i for i, (name, _) in enumerate(pairs) if name == "Cat0"]
        every = list(range(len(categories)))
        # Every category with Cat0 narrowed (23 tags), four categories, three
        sets = [(every, cat0[:-1]), ([0, 1, 2, 3], []), ([2, 3, 4], [])]
        steps = [["wait"]]
        for i, (cats, tags) in enumerate(sets):
            steps += _mix_tick_steps(cats, tags, f"mix{i}")
        snaps = _run_script(tmp_path, page, steps, strict=True)
        # ... and the whole list, sent through the mix
        forced = _run_script(tmp_path, page, [["wait"], *_mix_tick_steps(every, [], "all")],
                             strict=True, setup_js=_FORCE_MIX)["all"]
        tied = 0
        for snap, (cats, tags) in [*((snaps[f"mix{i}"], ticks) for i, ticks in enumerate(sets)),
                                   (forced, (every, []))]:
            ref = _mix_reference(data, trades, cats, tags, balance=1000.0, series=_TIE_SERIES)
            sums = ref["view"]["sub"]["x"]
            assert len(sums) > 12 and len(set(sums)) <= 4
            tied += len(sums) > 16
            _assert_mix_is_pythons(snap, ref, data, lst)
        # More tags than a small sort keeps in order by accident
        assert tied >= 3

    def test_a_mix_refused_at_another_scenario_says_which_one_is_still_shown(
            self, monkeypatch, tmp_path):
        # The list at k = 0.60 holds a figure that is not a number, so its
        # tags cannot be mixed. Two categories are on screen at k = 0.75 when
        # that k is chosen: the k goes back, and the line names both scenarios
        trades = _mix_trades(45, n=44)
        base = _mix_sweep(trades)
        bad = [dataclasses.replace(trades[0], profit_ratio=float("inf")), *trades[1:]]
        other = SweepPoint(k=0.6, trades=bad, equity_df=base.primary.equity_df,
                           spread_band=(0.0, 1.0), size_cap=0.2)
        sweep = dataclasses.replace(base, points=[other, base.primary],
                                    scenarios=[other, base.primary])
        monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(dashboard.yf, "download",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
        page = dashboard.generate_dashboard(
            trades, base.primary.equity_df, _MIX_START, _MIX_BALANCE, sweep=sweep,
            interval_discount=0.75, series_categories=_MIX_SERIES).read_text(encoding="utf-8")
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        assert [k["value"] for k in data["ks"]] == [0.6, 0.75] and data["primary"][1] == 1
        mixable = {ki: all("m" in view for key, view in chunks[data["grid"][0][ki][0]][
            "list"]["views"].items() if key.startswith("s")) for ki in (0, 1)}
        assert mixable == {0: False, 1: True}
        snaps = _run_script(tmp_path, page, [
            ["wait"], _tick_category(1), _tick_category(2), ["settle"], ["snap", "mixed"],
            ["set", "flt-k", "0"], ["fire", "flt-k"], ["settle"], ["snap", "refused"]],
            strict=True)
        mixed, refused = snaps["mixed"], snaps["refused"]
        assert mixed["reacts"] and refused["reacts"] == []
        assert refused["selects"]["flt-k"]["value"] == "1"
        assert refused["menus"]["flt-cat"]["ticked"] == [1, 2]
        assert refused["text"]["hdr-trades"] == mixed["text"]["hdr-trades"]
        tried, still = _phrase(data, 0, 0, 0), _phrase(data, 0, 1, 0)
        assert tried != still
        assert refused["text"]["flt-summary"] == (
            f"Not available: Cat1, Cat2 cannot be shown together for {tried} (the page's "
            f"build log says why). The page still shows Cat1, Cat2 at {still}.")
        assert not refused["buttons"]["flt-save"]

    def test_the_save_button_says_why_the_ticks_cannot_be_saved(self, monkeypatch, tmp_path):
        # "Health" and "health" are one category to the live filter, which
        # would keep both where the page shows one; a category whose name
        # holds the mark that ties a tag to its category cannot be told from
        # a tied tag. A tick on either cannot be saved, and the button's
        # hover text says why, in Python's words
        series = {"KXD0": ("Health", ("Flu",)), "KXD1": ("health", ("Flu",)),
                  "KXD2": ("Arts · Crafts", ("Glue",)), "KXD3": ("Sports", ("Golf",))}
        categories, _ = _mix_axes(series)
        assert categories == ["Arts · Crafts", "Health", "Sports", "health"]
        page = _mix_page(monkeypatch, tmp_path, _counted_trades(series), series=series)
        data = TestFilterPage._data(page)
        assert data["category_twins"] == [[1, 3]] and data["tag_twins"] == []
        snaps = _run_script(tmp_path, page, [
            ["snap", "loaded"], ["wait"], ["snap", "ready"],
            _tick_category(2), ["settle"], ["snap", "sports"],
            _tick_category(1), ["settle"], ["click", "flt-save"], ["snap", "twin"],
            _tick_category(1), ["settle"], ["snap", "sports_again"],
            _tick_category(0), ["settle"], ["click", "flt-save"], ["snap", "separator"],
            _tick_category(""), _tick_category(3), ["settle"], ["click", "flt-save"],
            ["snap", "other_twin"]], strict=True)
        usual = dashboard._SAVE_TITLE
        for name in ("loaded", "ready", "sports", "sports_again"):
            assert snaps[name]["titles"]["flt-save"] == usual, name
        for name in ("ready", "sports", "sports_again"):
            assert not snaps[name]["buttons"]["flt-save"], name
        for name, why in (("twin", "save_case_twin"), ("separator", "save_separator"),
                          ("other_twin", "save_case_twin")):
            assert snaps[name]["buttons"]["flt-save"], name
            assert snaps[name]["opened"] == [], name
            assert snaps[name]["titles"]["flt-save"] == data["text"][why], name
        assert data["text"]["save_case_twin"].startswith("Cannot save: a ticked category has")
        assert data["text"]["save_separator"].startswith("Cannot save: a ticked category's")

    def test_only_the_last_few_mixes_of_a_list_are_kept(self, monkeypatch, tmp_path):
        # Twenty different mixes of one list (and the mixes each passes
        # through on its way, tick by tick), then the first again: the list
        # remembers at most the script's limit of them, its strings do not
        # grow, and a mix it forgot is worked out again, the same
        limit = 16
        assert dashboard._FILTER_JS.count(f"var MIXES = {limit};") == 1
        trades = _mix_trades(46, n=44)
        page = _mix_page(monkeypatch, tmp_path, trades)
        data, chunks = TestFilterPage._data(page), TestFilterPage._chunks(page)
        lst, strings = chunks[0]["list"], len(chunks[0]["strings"])
        whole = [[si for si, (name, _) in enumerate(_MIX_SUBCATS) if name == category]
                 for category in _MIX_CATEGORIES]
        sets, seen = [], set()
        for cats, tags in _mix_tick_sets(46, 200):
            covered = _mix_covered(cats, tags)
            plain = (len(covered) == 1 or covered in whole
                     or len(covered) == len(_MIX_SUBCATS))
            if not plain and tuple(covered) not in seen:
                seen.add(tuple(covered))
                sets.append((cats, tags))
        sets = sets[:limit + 4]
        assert len(sets) == limit + 4
        steps = [["wait"]]
        for i, (cats, tags) in enumerate(sets):
            steps += _mix_tick_steps(cats, tags, f"mix{i}")
            if i == 0:
                steps += [["call", "__mixProbe", []], ["snap", "early"]]
        steps += [["call", "__mixProbe", []], ["snap", "full"]]
        steps += _mix_tick_steps(*sets[0], "again")
        steps += [["call", "__mixProbe", []], ["snap", "after"]]
        snaps = _run_script(tmp_path, page, steps, strict=True, setup_js=_MIX_PROBE)

        def probed(name: str) -> tuple[list[str], int, int]:
            """The page's one chunk at a snapshot: (the keys of the mixes its
            list remembers, how many mixes it holds, how many strings)."""
            [[keys, held, count]] = json.loads(snaps[name]["text"]["mix-probe"])
            return keys, held, count

        first = ",".join(f"s{si}" for si in _mix_covered(*sets[0]))
        keys, held, count = probed("early")
        assert first in keys and held == len(keys) <= limit and count == strings
        # After every mix: the limit, the first long forgotten, no new string
        keys, held, count = probed("full")
        assert len(keys) == held == limit and first not in keys and count == strings
        keys, held, count = probed("after")
        assert len(keys) == held == limit and first in keys and count == strings
        # Chosen again, the forgotten mix is the same drawing, and still what
        # Python computes for those trades
        _assert_same_drawing(snaps["mix0"], snaps["again"])
        assert snaps["again"]["text"]["flt-summary"] == snaps["mix0"]["text"]["flt-summary"]
        for name, (cats, tags) in (("again", sets[0]), (f"mix{limit + 3}", sets[-1])):
            _assert_mix_is_pythons(snaps[name], _mix_reference(data, trades, cats, tags),
                                   data, lst)

    def test_numbers_are_written_as_python_writes_them(self, tmp_path):
        rng = random.Random(7)
        values = [0.0, -0.0, 0.5, 1.5, 2.5, 0.125, 0.375, 0.625, 12.5, 62.5, 6.25, -0.125,
                  0.045, 1e-9, -1e-9, 0.995, 0.9995, 99.995, 999.5, 1234567.875, 0.05, 0.15,
                  0.25, 0.35, 0.0625, 0.1225, -0.0004, 0.00049999, 1 / 3, 2 / 3, 1e15 + 0.5]
        values += [k / 16 for k in range(-40, 41)] + [k / 1000 for k in range(-25, 26)]
        values += [rng.uniform(-5, 5) for _ in range(300)]
        values += [round(rng.uniform(-100, 100), digits) for digits in (1, 2, 3, 4)
                   for _ in range(60)]
        cases = [(x, digits) for x in values for digits in (0, 1, 2, 3, 4)]
        got = _run_js(tmp_path, ("fixed", "pct", "signed"),
                      f"{json.dumps(cases)}.map(function(c) {{ return [fixed(c[0], c[1]), "
                      "pct(c[0], c[1]), signed(pct(c[0], c[1]))]; })")
        ties = 0
        for (x, digits), drawn in zip(cases, got, strict=True):
            assert drawn == [format(x, f".{digits}f"), format(x, f".{digits}%"),
                             format(x, f"+.{digits}%")], (x, digits)
            # An exact half, which Python rounds to even and toFixed alone
            # rounds up: the grid holds many
            ties += abs(Decimal(x)).scaleb(digits) % 1 == Decimal("0.5")
        assert ties >= 100

    def test_sums_medians_and_rounding_are_numpys_and_pandas(self, tmp_path):
        rng = random.Random(8)
        lists = [[rng.gauss(0, 1e3) * 10 ** rng.randint(-6, 6) for _ in range(n)]
                 for n in (0, 1, 2, 7, 8, 9, 15, 16, 17, 127, 128, 129, 130, 255, 300, 1000)]
        got = _run_js(tmp_path, ("ascending", "pairSum", "npSum", "groupSum", "median", "rint",
                                 "roundTo"),
                      f"{json.dumps(lists)}.map(function(a) {{ return [npSum(a), groupSum(a), "
                      "a.length ? median(a) : null]; })")
        for values, (total, grouped, middle) in zip(lists, got, strict=True):
            # numpy's own pairwise sum, bit for bit: a mean here is Python's mean
            assert total == float(np.sum(np.array(values, dtype=float))), len(values)
            if values:
                # pandas' sum within a group, bit for bit: the decomposition's bars
                frame = pd.DataFrame({"g": ["a"] * len(values), "v": values})
                assert grouped == float(frame.groupby("g")["v"].sum().iloc[0]), len(values)
                assert middle == float(np.median(values)), len(values)
        points = [0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 0.125, 0.135, 2.675, 1e-7, -1e-7,
                  *(rng.uniform(-50, 50) for _ in range(200)),
                  *(k / 8 for k in range(-20, 21))]
        rounded = _run_js(tmp_path, ("rint", "roundTo"),
                          f"{json.dumps(points)}.map(function(x) {{ return [roundTo(x, 0), "
                          "roundTo(x, 2), roundTo(x, 4)]; })")
        for x, drawn in zip(points, rounded, strict=True):
            assert drawn == [float(np.round(x, digits)) for digits in (0, 2, 4)], x
