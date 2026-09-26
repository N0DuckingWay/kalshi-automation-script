"""Tests for dashboard.py — HTML escaping of Kalshi-controlled titles (BS-20),
the _max_drawdown empty/all-NaN guard (BS-30), the _sharpe/_sortino
annualization base and its per-row use in the benchmark table (DR-56), and the
interval-discount (k) section with its native Plotly k selector.

generate_dashboard() pulls in yfinance (network) and Plotly's full HTML
serialization; the escaping and drawdown fixes are exercised directly against
the cheapest reliable seam instead — _section_diagnostics() (the HTML table
render site, _trow) and _section_risk() (the Plotly hover-text render site) —
so these tests stay fully offline. Both sites are Kalshi-controlled
(BacktestTrade.title_a is a market question straight from the API). The
interval-discount section is tested the same way: _section_interval_discount()
is driven from a hand-built BacktestSweep, never through generate_dashboard().

The scenario explorer (PB5) is pinned by VALUE on a non-square band x k grid
whose primary sits off index 0 on both axes, and the seven sections that
predate it are pinned against digests captured on main by
tests/dashboard_golden.py (TestGoldenSections). Tests that do render a whole
page stub dashboard.yf.download and redirect dashboard.PROJECT_ROOT.
"""
import base64
import dataclasses
import gzip
import html
import json
import logging
import math
import re
import shutil
import subprocess
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from kalshi_betting import backtester, config, dashboard
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
    TIME_SERIES_INTERVAL_PROB_DISCOUNT,
    fee_leg_exact,
    fee_per_pair_approx,
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

from . import dashboard_golden

_XSS_TITLE = "<script>alert(1)</script>Will BTC exceed $80k by December 2026 or later?"


def make_trade(title_a: str = "Will BTC exceed $80k?", profit: float | None = None,
               deadline_gap_days: int | None = None) -> BacktestTrade:
    """Factory for a coherent time-series BacktestTrade covering every field the
    dashboard section builders under test read.

    YES on the earlier contract at 0.30 and NO on the later at 0.40 (later YES
    ask 0.60, earlier NO ask 0.70 — reporting only), n=5, settled in the
    "event by A" win cell (A=YES, hence B=YES). _section_risk calls the
    six-argument _kelly_fraction on these entry prices live, threading the run's
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
    prices time-series pairs through config.time_series_profit_prob."""

    def test_time_series_flow_through_fixture(self):
        # YES 0.30 + NO 0.40, later YES ask 0.60: p = 0.775, f* ≈ 0.1620.
        # b's denominator carries the fee — the dollars at risk include it,
        # because a losing pair loses cost + fees (DR-62).
        pA, nA, pB, nB = 0.30, 0.70, 0.60, 0.40
        fee = fee_per_pair_approx(pA, nB)
        net_spread = (1.0 - pA - nB) - fee
        b = net_spread / (pA + nB + fee)
        p = time_series_profit_prob(pA, pB)
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
                            k=TIME_SERIES_INTERVAL_PROB_DISCOUNT)
        )

    def test_explicit_k_of_one_clamps_to_zero(self):
        # k = 1 takes the market at face value: p = 1 - (pB - pA) = 0.70, f* < 0
        assert _kelly_fraction(self._PA, self._NA, self._PB, self._NB,
                               "time_series", k=1.0) == 0.0

    def test_explicit_k_wins_over_a_monkeypatched_constant(self, monkeypatch):
        # The constant is set to the never-trade value; the explicit k must
        # still produce the configured-k fixture's ~0.1620 fraction.
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
    """The interval-discount section renders the calibration table and a native
    Plotly updatemenus dropdown with one button per swept k."""

    def test_none_returns_the_placeholder(self):
        out = _section_interval_discount(None)
        assert "Interval Discount (k) Calibration" in out
        assert "No interval-discount sweep for this run." in out
        # The placeholder must not carry a chart or a selector
        assert "updatemenus" not in out

    def test_empty_points_returns_the_placeholder(self):
        point = _sweep_points([0.75])[0]
        sweep = BacktestSweep(primary=point, points=[], calibration=None)
        assert "No interval-discount sweep for this run." in _section_interval_discount(sweep)

    def test_dropdown_has_one_button_per_k_and_defaults_to_primary(self):
        points = _sweep_points([0.60, 0.75, 0.90])
        sweep = BacktestSweep(primary=points[1], points=points,
                              calibration=_calibration())
        out = _section_interval_discount(sweep)

        # A native Plotly dropdown, not hand-rolled JS
        assert "updatemenus" in out
        assert re.findall(r'"label":\s*"(k = [0-9.]+)"', out) == [
            "k = 0.60", "k = 0.75", "k = 0.90",
        ]
        # Trace visibility serializes as scalars (the buttons' own args use
        # arrays), so these three flags are the traces: only the primary shows.
        assert re.findall(r'"visible":\s*(true|false)\b', out) == ["false", "true", "false"]
        # ...and the dropdown itself opens on the primary's entry
        assert re.findall(r'"active":\s*(\d+)', out) == ["1"]

    def test_primary_matched_by_k_when_points_holds_an_equal_copy(self):
        # BacktestSweep normally shares the object, but a copy must still be
        # located (by k) rather than silently defaulting to the first trace.
        points = _sweep_points([0.60, 0.75, 0.90])
        copy = SweepPoint(k=points[2].k, trades=list(points[2].trades),
                          equity_df=points[2].equity_df.copy())
        sweep = BacktestSweep(primary=copy, points=points, calibration=None)
        out = _section_interval_discount(sweep)
        assert re.findall(r'"visible":\s*(true|false)\b', out) == ["false", "false", "true"]

    def test_kpis_and_calibration_table_render(self):
        points = _sweep_points([0.75])
        sweep = BacktestSweep(primary=points[0], points=points,
                              calibration=_calibration())
        out = _section_interval_discount(sweep)

        # Labelled for the run, not for config.py — on an --interval-discount
        # run this number is the override, which config.py never receives.
        assert "k used (this run)" in out and "0.750" in out
        assert "Configured k" not in out
        assert "Pooled empirical k̂" in out and "0.600" in out
        # Delta = 0.600 - 0.750
        assert "-0.150" in out
        # Bucket row: label, tier, n, realised, implied; empirical_k of None
        # renders as "-", and the pooled tier of 0.0 does too.
        assert "0-7d" in out and "0.15" in out and "0.1800" in out
        assert "POOLED" in out
        assert "Excluded 3 premise violation(s)" in out
        assert "never writes config.py" in out

    def test_sweep_table_reports_per_k_metrics(self):
        points = _sweep_points([0.60, 0.75])
        sweep = BacktestSweep(primary=points[1], points=points, calibration=None)
        out = _section_interval_discount(sweep)

        # One row per point, primary flagged; trade counts come off the points
        assert "k = 0.60" in out and "k = 0.75 (primary)" in out
        # Point 2's curve is 1000 -> 1020 -> 1010: +1.0% total, -1.0% drawdown
        assert "+1.0%" in out
        assert "-1.0%" in out
        # No calibration to show, but the sweep table and selector still render
        assert "No time-series candidate was measurable" in out
        assert "updatemenus" in out

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
        # The banner is additive: the table, the selector and the sweep table
        # must all survive it.
        out = _section_interval_discount(self._sweep(_coverage(2)))
        assert "updatemenus" in out
        assert "POOLED" in out
        assert "k used (this run)" in out

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
        assert "updatemenus" in out and "POOLED" in out
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
        assert "updatemenus" not in out


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
    CALENDAR base (see dashboard._sharpe's own "four of the five calls"
    paragraph); the fourth of them — the Benchmark table's STRATEGY row, which
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
        assert "updatemenus" in out
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
        point = SweepPoint(k=TIME_SERIES_INTERVAL_PROB_DISCOUNT,
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
               halves=None, ex_top=None) -> SweepPoint:
    """One scenario SweepPoint."""
    return SweepPoint(
        k=k, trades=trades if trades is not None else [_scn_trade()],
        equity_df=make_equity(values if values is not None else [1000.0, 1010.0]),
        spread_band=band, population=population, halves=halves, ex_top_event=ex_top,
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


def _scn_data(section_html: str) -> dict:
    """The scn-data payload, cut at the FIRST "</script>" after the block's
    opening tag — exactly where a browser's HTML parser ends the block — and
    parsed strictly (a NaN/Infinity token fails the test)."""
    start = section_html.index('id="scn-data">') + len('id="scn-data">')
    end = section_html.index("</script>", start)
    return json.loads(section_html[start:end], parse_constant=_fail_on_constant)


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
    """(value, selected, text) of every <option> of one <select>."""
    body = re.search(rf"<select id='{select_id}'>(.*?)</select>", section_html).group(1)
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


class TestScenarioExplorerPayload:
    """The data block the inline script reads: its indexing, its per-population
    rows and its values, all pinned by value on a non-square grid."""

    def test_the_grid_is_band_major_and_not_transposed(self):
        data = _scn_data(_Grid.section())
        assert data["bands"] == [[0.0, 1.0], [0.3, 0.6], [0.35, 0.8]]
        assert data["ks"] == [0.65, 0.75]
        assert data["populations"] == ["all", "ladder", "cross", "time_series"]
        pop = {name: i for i, name in enumerate(data["populations"])}
        assert len(data["cells"]) == 3
        assert all(len(row) == 2 for row in data["cells"])
        for bi in range(3):
            for ki in range(2):
                c = bi * 2 + ki
                cell = data["cells"][bi][ki]
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
        data = _scn_data(_section_scenario_explorer(sweep))
        all_row = data["cells"][1][1][0]
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
        data = _scn_data(_Grid.section())
        ladder, cross = data["cells"][1][1][1], data["cells"][1][1][2]
        assert ladder["total_return"] == pytest.approx(0.006)   # 1000 -> 1006
        assert ladder["final_balance"] == pytest.approx(1006.0)
        assert cross["total_return"] == pytest.approx(-0.004)   # 1000 -> 996
        assert "h1_return" not in ladder and "equity" not in ladder
        same_title = data["same_title"]
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
        data = _scn_data(section)
        assert data["cells"][0][0][0]["total_return"] is None
        # PB7: the per-cell curve is the headline (time-series) population's
        assert data["cells"][0][0][3]["equity"][2] is None
        assert "NaN" not in section and "Infinity" not in section

    def test_a_script_closing_ticker_cannot_end_the_data_block(self):
        hostile = "EVT-</script><b>x"
        section = _Grid.section(top_event=hostile)
        assert hostile not in section
        assert "EVT-<\\/script>" in section
        assert _scn_data(section)["cells"][1][1][0]["top_event"] == hostile

    def test_every_curve_is_on_the_shared_axis(self):
        sweep = _Grid.sweep()
        data = _scn_data(_section_scenario_explorer(sweep))
        assert data["dates"] == [d.isoformat() for d in sweep.primary.equity_df["date"]]
        for row in data["cells"]:
            for cell in row:
                # PB7: the per-cell curve is the headline (time-series)
                # population's; the "all" entry no longer ships one
                assert len(cell[3]["equity"]) == len(data["dates"])
                assert "equity" not in cell[0]


class TestScenarioExplorerControls:
    """The selects, the heatmap's metric toggle and the fragility banner."""

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
        data = _scn_data(section)
        assert (data["primary_band_idx"], data["primary_k_idx"]) == (1, 1)

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

    def test_heatmap_buttons_update_z_and_title_together(self):
        heat, layout = _first_figure(_Grid.section())
        # PB7: the title names the population the cells are
        pop = "Time-series (ladders + cross-event; same-title excluded)"
        assert layout["title"]["text"] == (
            f"Mean per trade (equal stake) by spread band x k — {pop}")
        buttons = layout["updatemenus"][0]["buttons"]
        assert [b["label"] for b in buttons] == [
            "Mean per trade (equal stake)", "Median per trade (equal stake)",
            "Total return", "H1 return", "H2 return", "Trade count",
            "Empirical k̂ (pooled per band)"]
        for b in buttons:
            # "restyle" would read the second argument as trace indices and
            # drop the title silently; "update" relayouts it.
            assert b["method"] == "update"
            assert b["args"][1] == {
                "title.text": f"{b['label']} by spread band x k — {pop}"}
        z = {b["label"]: b["args"][0]["z"][0] for b in buttons}
        assert z["Trade count"] == [[1, 2], [3, 4], [5, 6]]
        assert z["Total return"][0][0] is None
        assert z["Total return"][1] == pytest.approx([0.01, 0.02])
        assert z["H1 return"] == [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]]
        assert z["H2 return"] == [[0.03, -0.01], [0.05, 0.0], [0.02, 0.02]]
        assert heat[0]["z"] == z["Mean per trade (equal stake)"]
        # A count is never negative: its own colour scale, auto-ranged.
        count = buttons[5]["args"][0]
        assert count["zmid"] == [None]
        assert count["colorscale"] != buttons[0]["args"][0]["colorscale"]
        assert buttons[0]["args"][0]["zmid"] == [0]
        assert heat[0]["colorscale"] == buttons[0]["args"][0]["colorscale"][0]

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
        heat, layout = _first_figure(_section_scenario_explorer(self.sweep()))
        buttons = {b["label"]: b["args"][0]["z"][0]
                   for b in layout["updatemenus"][0]["buttons"]}
        expected = [(f - 1000.0) / 1000.0 for f in self.TS_FINALS]
        assert buttons["Total return"][0] == pytest.approx(expected[:2])
        assert buttons["Total return"][1][0] == pytest.approx(expected[2])
        # Cell 3 has no time-series point: null, never the "all" cell's +50%
        assert buttons["Total return"][1][1] is None
        assert buttons["Trade count"] == [[1, 1], [1, None]]
        assert buttons["H1 return"] == [[-0.10, -0.30], [-0.20, None]]
        # The title names the population on every metric
        assert layout["title"]["text"].endswith(f"— {_TS_LABEL}")
        assert all(_TS_LABEL in b["args"][1]["title.text"]
                   for b in layout["updatemenus"][0]["buttons"])

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
        data = _scn_data(section)
        pop = {name: i for i, name in enumerate(data["populations"])}
        cell = data["cells"][0][0]
        assert cell[pop["time_series"]]["equity"] == [1000.0, 950.0, 900.0]
        assert "equity" not in cell[pop["all"]]
        # ... and both checked populations carry their own extras
        assert cell[pop["time_series"]]["top_event"] == "EVT-TS"
        assert cell[pop["all"]]["top_event"] == "EVT-ST"
        assert data["cells"][1][1][pop["time_series"]] is None

    def test_every_population_is_labelled(self):
        data = _scn_data(_section_scenario_explorer(self.sweep()))
        assert data["labels"] == {
            "time_series": _TS_LABEL,
            "all": "All (time-series + same-title)",
            "ladder": "Ladders (same-event)",
            "cross": "Cross-event",
            "same_title": "Same-title (independent of band and k)",
        }
        # The script builds every KPI row from those labels, the headline
        # population first, and no longer spells a bare 'All'
        js = dashboard._SCENARIO_EXPLORER_JS
        assert js.index("L.time_series") < js.index("L.ladder") < js.index("L.all")
        assert "kpiRow('All'" not in js

    def test_the_script_reads_the_headline_population(self):
        # The KPI table's headline row, its sub-row and the restyled curve
        # exist only in the inline script, which no test executes; on a
        # corpus with no same-title pair "all" equals "time_series" cell for
        # cell, so a script reading "all" would look right there. Pin the
        # reads themselves.
        js = dashboard._SCENARIO_EXPLORER_JS
        assert "var ts = cell[P.time_series];" in js
        assert "kpiRow(esc(L.time_series), ts) + extrasRow(ts)" in js
        assert "var values = (ts && ts.equity) ? ts.equity : [];" in js
        assert "kpiRow(esc(L.all), cell[P.all]) + extrasRow(cell[P.all])" in js
        # ... and the population is resolved through the payload's names, so
        # the index the script reads is the time-series point's
        data = _scn_data(_section_scenario_explorer(self.sweep()))
        ts_idx = data["populations"].index("time_series")
        cell = data["cells"][0][0]
        assert cell[ts_idx]["equity"] == [1000.0, 950.0, 900.0]
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
        _, layout = _first_figure(section)
        z = {b["label"]: b["args"][0]["z"][0] for b in layout["updatemenus"][0]["buttons"]}
        assert [row[0] for row in z["H1 return"]] == [*h1, None]
        # ... while H2 of that cell, which had entries, is still a measurement
        assert [row[0] for row in z["H2 return"]] == [*h2, 0.10]
        data = _scn_data(section)
        pop = {name: i for i, name in enumerate(data["populations"])}
        assert data["cells"][4][0][pop["time_series"]]["h1_return"] is None
        assert data["cells"][4][0][pop["all"]]["h1_return"] is None

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
        data = _scn_data(section)
        pop = {name: i for i, name in enumerate(data["populations"])}
        for row in data["cells"]:
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
        assert values == [round(v, 2) for v in later["portfolio_value"].iloc[:400]]

    def test_values_are_cents_and_non_finite_is_none(self):
        eq = make_equity([1000.123456, float("nan"), 1001.5])
        values = dashboard._curve_on_axis(eq, dashboard._equity_axis(eq))
        assert values == [1000.12, None, 1001.5]

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
    """The page header names the primary spread band and the ladder setting,
    under the Period line and above every section — both decide which pairs
    exist, so they qualify the whole page."""

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
                in page)
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
        line = f"Primary spread band: max(tier,0.3)-0.6 | same-event ladders: {word}</p>"
        assert line in page
        assert page.index("Period:") < page.index(line) < page.index("Portfolio Performance")

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
        assert f"| same-event ladders: {tail}</p>" in page

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
        assert "| same-event ladders: not recorded</p>" in page


class TestCorpusProvenanceHeader:
    """DR-13 / M2 (P2): directly under the Period line the header says what
    settled-market corpus the run read — its assembly time (the Period runs to
    today, the corpus only to that moment), whether it came from an earlier
    run's cache, and the archive cutoff as of assembly — on EVERY run, healthy
    or not (DR-66). A window at or after that cutoff gets a red banner that
    states a bound (no trade could be entered whatever pairs formed), never a
    cause — unless some simulated point traded, which proves the verdict stale
    and gets an amber stale-verdict line instead. A legacy .json hit shows its
    file time. Rendered through generate_dashboard(), because a unit test that
    does not prove the string reaches the page is the gap DR-66b was about."""

    ASSEMBLED = datetime(2026, 9, 24, 12, 37, 49, tzinfo=UTC)
    CUTOFF = datetime(2026, 7, 25, tzinfo=UTC)

    def _page(self, monkeypatch, tmp_path, prov=None, *, sweep=True,
              n_trades=0, other_point_trades=0) -> str:
        # The page's trades and the sweep's primary point agree, as they do in
        # production (backtest.py passes result.primary.trades); a second k
        # point can carry trades of its own, as the k dropdown can show.
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
                  "archive_cutoff": self.CUTOFF, "post_cutoff": False}
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
        line = self._corpus_line(self._page(monkeypatch, tmp_path,
                                            self._prov(from_cache=True)))
        assert "served from an earlier run&#x27;s cache; --no-cache extends it" in line

    @pytest.mark.parametrize("from_cache", [False, True])
    def test_a_post_cutoff_window_gets_the_banner(self, monkeypatch, tmp_path, from_cache):
        page = self._page(monkeypatch, tmp_path,
                          self._prov(post_cutoff=True, from_cache=from_cache))
        assert ("This window starts at or after the archive cutoff (2026-07-25, as "
                "of the corpus's assembly).") in page
        assert "no trade could be entered in this window whatever pairs formed" in page
        assert "says nothing about the strategy" in page
        # Only a cached verdict can have gone stale — only it says so.
        assert ("a cached run does not re-read it; --no-cache re-checks" in page) \
            is from_cache
        assert page.index("Settled-market corpus:") < page.index(
            "This window starts at or after") < page.index("Portfolio Performance")
        assert "that verdict is stale" not in page

    @pytest.mark.parametrize("from_cache", [False, True])
    @pytest.mark.parametrize("n_trades, other_point_trades", [(2, 0), (0, 3)])
    def test_a_post_cutoff_verdict_contradicted_by_trades_is_reported_as_stale(
            self, monkeypatch, tmp_path, from_cache, n_trades, other_point_trades):
        # P2 review (R3/C3/ADV-3): a stamped verdict goes stale once the cutoff
        # moves past start_date, and the run can then trade — a red "no trade
        # could be entered" beside "Trades found: N" would be false on its
        # face. A trade at ANY simulated point (the primary, or another k the
        # dropdown shows) turns it into a stale-verdict line instead.
        page = self._page(monkeypatch, tmp_path,
                          self._prov(post_cutoff=True, from_cache=from_cache),
                          n_trades=n_trades, other_point_trades=other_point_trades)
        assert f'Trades found: <span id="hdr-trades">{n_trades}</span>' in page
        assert "This window starts at or after" not in page
        assert "no trade could be entered in this window" not in page
        recorded = ("at this corpus&#x27;s assembly" if from_cache else "by this run")
        assert f"The archive cutoff recorded {recorded} (2026-07-25)" in page
        assert (f"this run entered trades (up to {max(n_trades, other_point_trades)} "
                "in one simulated scenario), so that verdict is stale") in page
        assert "--no-cache re-reads the cutoff and re-stamps the cache." in page
        assert page.index("Settled-market corpus:") < page.index(
            "The archive cutoff recorded") < page.index("Portfolio Performance")

    def test_a_legacy_cache_shows_its_file_time(self, monkeypatch, tmp_path):
        # P2 review (C1/ADV-1): a legacy settled_markets_*.json hit carries its
        # file time to the page, named as such, and claims no cutoff.
        line = self._corpus_line(self._page(monkeypatch, tmp_path, self._prov(
            from_cache=True, archive_cutoff=None, post_cutoff=None, legacy=True)))
        assert ("last written 2026-09-24 12:37 UTC (the file time of a legacy "
                "settled_markets_*.json, which records no assembly stamp) — it "
                "holds no market settled after that") in line
        assert ("served from an earlier run&#x27;s cache; --no-cache extends it "
                "and rebuilds it in the streamed format") in line
        assert ("archive cutoff at assembly: not recorded (the legacy format "
                "records none; --no-cache re-checks it)") in line

    def test_an_unrecorded_cutoff_says_so_and_claims_no_verdict(self, monkeypatch, tmp_path):
        page = self._page(monkeypatch, tmp_path, self._prov(
            from_cache=True, archive_cutoff=None, post_cutoff=None))
        assert ("archive cutoff at assembly: not recorded (--no-cache re-checks it)"
                in self._corpus_line(page))
        assert "This window starts at or after" not in page

    def test_an_unrecorded_assembly_time_says_so(self, monkeypatch, tmp_path):
        line = self._corpus_line(self._page(monkeypatch, tmp_path,
                                            self._prov(assembled_at=None)))
        assert "assembly time not recorded" in line

    @pytest.mark.parametrize("sweep", [True, False])
    def test_no_provenance_reads_not_recorded(self, monkeypatch, tmp_path, sweep):
        # An infeasible window, a stubbed corpus, or no sweep at all: said,
        # never silently absent (a legacy .json hit now carries its file time).
        page = self._page(monkeypatch, tmp_path, None, sweep=sweep)
        assert ("Settled-market corpus: assembly time and archive cutoff not "
                "recorded for this run (no sweep was passed to the report") in page
        assert "This window starts at or after" not in page


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
    calibration per band and a top event per cell, stays within 5 MB."""

    def test_page_size_under_5mb_for_a_468_cell_sweep(self, monkeypatch, tmp_path):
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

        scenarios = []
        for band in bands:
            for k in ks:
                # PB7: the "all" AND the "time_series" point each carry both
                # checks, and the time-series one carries the per-cell curve
                for population in ("all", "time_series"):
                    scenarios.append(SweepPoint(
                        k=k, trades=trades, equity_df=shared_equity, spread_band=band,
                        population=population,
                        halves=HalfSplit(float(rng.normal()), float(rng.normal()), 40, 48,
                                         120, 130),
                        ex_top_event=("KXEVENT-26SEP03-ABCDEF", float(rng.normal()))))
                for population in ("ladder", "cross"):
                    scenarios.append(SweepPoint(
                        k=k, trades=trades[:40], equity_df=shared_equity,
                        spread_band=band, population=population))
        sweep = BacktestSweep(
            primary=scenarios[0], points=[scenarios[0]], calibration=None,
            label_coverage=_scn_coverage(), scenarios=scenarios,
            same_title_point=SweepPoint(k=ks[0], trades=trades[:10],
                                        equity_df=shared_equity, population="same_title"),
            calibrations_by_band={b: _scn_calibration() for b in bands},
            same_event_ladders=True, split_date=date(2023, 5, 1),
        )

        out_path = dashboard.generate_dashboard(
            trades, shared_equity, date(2020, 1, 1), 10_000.0, sweep=sweep)
        size = out_path.stat().st_size
        assert size <= 5_000_000, f"page was {size} bytes"


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
        data = _scn_data(section)
        ts = data["cells"][1][1][data["populations"].index("time_series")]
        ratios = sorted(t.profit_ratio for t in _Grid.all_trades(3))
        assert ts["median_per_trade"] == pytest.approx((ratios[1] + ratios[2]) / 2)
        _, layout = _first_figure(section)
        buttons = {b["label"]: b["args"][0] for b in layout["updatemenus"][0]["buttons"]}
        median = buttons["Median per trade (equal stake)"]
        assert median["z"][0][1][1] == pytest.approx(ts["median_per_trade"])
        assert median["zmid"] == [0]


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
        assert dashboard._series_ticker("KXMVECROSSCATEGORY0-S1") == "KXMVECROSSCATEGORY0"
        assert dashboard._series_ticker("") == ""

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

    def test_each_band_row_repeats_its_pooled_khat(self):
        _, layout = _first_figure(_section_scenario_explorer(self._sweep()))
        buttons = {b["label"]: b["args"][0] for b in layout["updatemenus"][0]["buttons"]}
        khat = buttons[self.KHAT]
        assert khat["z"][0] == [[0.60, 0.60], [0.90, 0.90], [None, None]]
        # Its hover reads the entries pooled, not the trade count
        assert khat["customdata"][0] == [[10, 10], [4, 4], [0, 0]]
        # Centred on the run's primary k (the primary cell sits at k = 0.75)
        assert khat["zmid"] == [0.75]
        assert "entries pooled" in khat["hovertemplate"][0]

    def test_every_other_metric_restores_the_trade_counts(self):
        # customdata travels with every button, so switching back from k-hat
        # must put the trade counts back under the other metrics' hovers
        _, layout = _first_figure(_section_scenario_explorer(self._sweep()))
        buttons = {b["label"]: b["args"][0] for b in layout["updatemenus"][0]["buttons"]}
        trades = buttons["Trade count"]["z"][0]
        for label, args in buttons.items():
            if label != self.KHAT:
                assert args["customdata"][0] == trades, label

    def test_a_band_without_a_calibration_is_blank(self):
        sweep = dataclasses.replace(self._sweep(), calibrations_by_band={
            _Grid.BANDS[0]: _scn_calibration()})
        _, layout = _first_figure(_section_scenario_explorer(sweep))
        buttons = {b["label"]: b["args"][0] for b in layout["updatemenus"][0]["buttons"]}
        assert buttons[self.KHAT]["z"][0] == [[0.60, 0.60], [None, None], [None, None]]

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


# ═══ The page-wide filter: spread band x Kalshi category x tag ══════════════

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


def _flt_sweep():
    """Three bands at k 0.75 (the primary, 0-1, holding every trade; 0.3-0.6
    holding only the same-title ones; 0.3-1 an EQUAL copy of 0.3-0.6's list),
    plus decoy points the filter must never read: another k at the primary
    band AND at 0.3-0.6 (listed after that band's own point, so a filter that
    ignored k would keep it), another population, and a point with no band.
    Only the primary band carries a k-hat population; KXSPACEX has no trade
    and no map entry, so its observation files under its fallback category,
    "Science", which no trade carries."""
    trades = _flt_trades()
    st = [t for t in trades if t.pair_type == "same_title"]
    curve = backtester._build_equity_curve
    primary = SweepPoint(k=0.75, trades=trades, equity_df=curve(trades, _FLT_START, 1000.0),
                         spread_band=(0.0, 1.0), population="all")
    narrow = SweepPoint(k=0.75, trades=st, equity_df=curve(st, _FLT_START, 1000.0),
                        spread_band=(0.3, 0.6), population="all")
    wide = SweepPoint(k=0.75, trades=list(st), equity_df=curve(st, _FLT_START, 1000.0),
                      spread_band=(0.3, 1.0), population="all")
    one = curve(st[:1], _FLT_START, 1000.0)
    decoys = [
        SweepPoint(k=0.60, trades=st[:1], equity_df=one, spread_band=(0.0, 1.0),
                   population="all"),
        SweepPoint(k=1.00, trades=st[:1], equity_df=one, spread_band=(0.3, 0.6),
                   population="all"),
        SweepPoint(k=0.75, trades=st[:1], equity_df=one, spread_band=(0.3, 0.6),
                   population="time_series"),
        SweepPoint(k=0.75, trades=st[:1], equity_df=one, spread_band=None, population="all"),
    ]
    cals = {(0.0, 1.0): _flt_calibration([("KXNHLHART-27", "Other"),
                                          ("KXSPACEX-14", "Science"),
                                          ("KXOTHER-1", "Other")]),
            (0.3, 0.6): None, (0.3, 1.0): None}
    return BacktestSweep(primary=primary, points=[primary], calibration=cals[(0.0, 1.0)],
                         label_coverage=_scn_coverage(),
                         scenarios=[primary, narrow, wide, *decoys],
                         calibrations_by_band=cals, same_event_ladders=True)


def _flt_payload(sweep=None):
    sweep = sweep or _flt_sweep()
    runs, primary_idx = dashboard._band_runs(sweep, sweep.primary.trades,
                                             sweep.primary.equity_df)
    payload = dashboard._filter_payload(runs, primary_idx, _FLT_START, 1000.0,
                                        _FLT_SERIES, 0.75, "k = 0.75")
    return sweep, runs, primary_idx, payload


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


def _view(payload: dict, band: int, key: str) -> dict:
    return payload["lists"][payload["bands"][band]["list"]]["views"][key]


def _key(payload: dict, category: str, tag: str | None = None) -> str:
    ci = payload["categories"].index(category)
    if tag is None:
        return f"c{ci}"
    return f"s{payload['subcats'].index([ci, tag])}"


class TestFilterBandRuns:
    """Which run each spread band the bar offers shows."""

    def test_each_band_is_its_all_point_at_the_primary_k(self):
        sweep = _flt_sweep()
        runs, primary_idx = dashboard._band_runs(sweep, sweep.primary.trades,
                                                 sweep.primary.equity_df)
        assert [r.band for r in runs] == [(0.0, 1.0), (0.3, 0.6), (0.3, 1.0)]
        assert primary_idx == 0
        assert [r.label for r in runs] == [dashboard._row_label(r.band) for r in runs]
        # Never another k's point (0.3-0.6 has one listed after its own),
        # another population's, or a point with no band
        assert [len(r.trades) for r in runs] == [5, 3, 3]
        assert runs[1].trades is sweep.scenarios[1].trades
        assert runs[0].calibration is sweep.calibration
        assert runs[1].calibration is runs[2].calibration is None

    def test_the_primary_band_is_what_the_page_renders(self):
        sweep = _flt_sweep()
        page_trades, page_curve = list(sweep.primary.trades), sweep.primary.equity_df.copy()
        runs, i = dashboard._band_runs(sweep, page_trades, page_curve)
        assert runs[i].trades is page_trades and runs[i].equity_df is page_curve

    def test_no_sweep_or_no_recorded_band_is_one_unlabelled_run(self):
        trades = _flt_trades()
        curve = backtester._build_equity_curve(trades, _FLT_START, 1000.0)
        runs, i = dashboard._band_runs(None, trades, curve)
        assert (i, [(r.band, r.label) for r in runs]) == (0, [(None, "not recorded")])
        hand_built = BacktestSweep(primary=SweepPoint(k=0.75, trades=trades, equity_df=curve),
                                   points=[], calibration=None)
        runs, i = dashboard._band_runs(hand_built, trades, curve)
        assert [(r.band, r.label) for r in runs] == [(None, "not recorded")]

    def test_bands_with_equal_trades_share_one_list(self):
        _, _, _, payload = _flt_payload()
        assert [b["list"] for b in payload["bands"]] == [0, 1, 1]
        assert len(payload["lists"]) == 2

    def test_the_primary_list_is_built_first(self):
        # The primary band sorts SECOND here, and a band before it trades the
        # same list: the shared list must keep the primary's own curve (the
        # page's), not the other band's (a curve built later can run a day on)
        st = [t for t in _flt_trades() if t.pair_type == "same_title"]
        page_curve = backtester._build_equity_curve(st, _FLT_START, 1000.0)
        other_curve = page_curve.assign(portfolio_value=page_curve["portfolio_value"] + 1.0)
        primary = SweepPoint(k=0.75, trades=st, equity_df=page_curve,
                             spread_band=(0.3, 0.6), population="all")
        other = SweepPoint(k=0.75, trades=list(st), equity_df=other_curve,
                           spread_band=(0.0, 1.0), population="all")
        sweep = BacktestSweep(primary=primary, points=[primary], calibration=None,
                              scenarios=[primary, other])
        _, runs, primary_idx, payload = _flt_payload(sweep)
        assert primary_idx == 1 and [b["list"] for b in payload["bands"]] == [0, 0]
        n = len(payload["dates"])
        assert _expand(_view(payload, 0, "all")["eq"], n) == pytest.approx(
            list(page_curve["portfolio_value"]))


class TestFilterViews:
    """Every view's figures, computed by the helpers the sections render with."""

    def test_the_default_view_is_exactly_the_rendered_page(self):
        sweep, _, _, payload = _flt_payload()
        trades, curve = sweep.primary.trades, sweep.primary.equity_df
        view = _view(payload, payload["primary"], "all")
        kpis = dashboard._performance_kpis(curve, trades, 1000.0)
        assert view["kpi"] == {key: value for key, _, value, _ in kpis}
        section = dashboard._section_performance(curve, trades, _FLT_START, 1000.0)
        for key, value in view["kpi"].items():
            assert f'id="kpi-{key}" ' in section and f">{value}</div>" in section
        assert view["bench"] == dashboard._strategy_row(curve, 1000.0)
        rel = dashboard._reliability(trades)
        assert view["cal"]["brier"] == f"{rel['brier']:.4f}"
        # The sparse series expand back to the curve the section draws
        n = len(payload["dates"])
        assert _expand(view["eq"], n) == pytest.approx(list(curve["portfolio_value"]))
        total, _, drawdown = dashboard._performance_series(curve, trades, 1000.0)
        assert _expand(view["total"], n) == pytest.approx(list(total), abs=1e-4)
        assert _expand(view["dd"], n) == pytest.approx(list(drawdown), abs=1e-4)
        deployed = dashboard._capital_deployed(trades, curve)
        assert _expand(view["dep"], n) == pytest.approx(deployed, abs=0.01)

    def test_categories_and_tags_partition_the_band(self):
        _, _, _, payload = _flt_payload()
        lst = payload["lists"][0]
        assert payload["categories"] == ["Commodities", "Other", "Science", "Sports"]
        assert payload["subcats"] == [[0, "Oil & Gas"], [1, "General"], [2, "General"],
                                      [3, "Basketball"], [3, "Hockey"]]
        n_all = lst["views"]["all"]["n"]
        cats = {ci: lst["views"][f"c{ci}"]["n"] for ci in range(4) if f"c{ci}" in lst["views"]}
        assert sum(cats.values()) == n_all == 5
        for ci, n_cat in cats.items():
            subs = [lst["views"][f"s{si}"]["n"] for si, (c, _) in enumerate(payload["subcats"])
                    if c == ci and f"s{si}" in lst["views"]]
            assert sum(subs) == n_cat
        # FIRST tag only: KXBRENTW's second tag ("Energy") files nothing
        assert [0, "Energy"] not in payload["subcats"]

    def test_a_slice_shows_its_trades_contribution(self):
        _, _, _, payload = _flt_payload()
        view = _view(payload, 0, _key(payload, "Sports"))
        sports = [t for t in _flt_trades() if t.event_ticker.startswith(("KXNCAA", "KXNHL"))]
        assert view["n"] == len(sports) == 3
        n = len(payload["dates"])
        pnl = sum(t.profit for t in sports)
        assert _expand(view["eq"], n)[-1] == pytest.approx(1000.0 + pnl, abs=0.01)
        assert _expand(view["total"], n)[-1] == pytest.approx(pnl / 1000.0 * 100, abs=1e-4)
        assert view["kpi"]["trades"] == "3"
        assert view["kpi"]["total_return"] == f"{pnl / 1000.0:+.1%}"
        # Its type lines are only the types it holds
        assert [label for label, _ in view["types"]] == ["Same-title", "Time-series: ladder"]

    def test_best_and_worst_rows_are_the_slices_own(self):
        _, _, _, payload = _flt_payload()
        key = _key(payload, "Sports", "Basketball")
        view = _view(payload, 0, key)
        basketball = [t for t in _flt_trades() if t.event_ticker.startswith("KXNCAA")]
        best, worst = dashboard._best_and_worst(basketball)
        assert [payload["strings"][i] for i in view["best"]] == [
            dashboard._trade_row(t, dashboard._BEST_ROW_COLOR) for t in best]
        assert [payload["strings"][i] for i in view["worst"]] == [
            dashboard._trade_row(t, dashboard._WORST_ROW_COLOR) for t in worst]
        lst = payload["lists"][0]
        assert [lst["ret"][i] for i in view["idx"]] == pytest.approx(
            [t.profit_ratio * 100 for t in basketball])

    def test_a_band_without_a_category_has_no_view_for_it(self):
        _, _, _, payload = _flt_payload()
        narrow = payload["lists"][payload["bands"][1]["list"]]["views"]
        assert _key(payload, "Sports", "Hockey") not in narrow
        assert payload["empty"]["n"] == 0
        n = len(payload["dates"])
        assert set(_expand(payload["empty"]["eq"], n)) == {1000.0}

    def test_a_category_seen_only_in_khat_observations_is_offered(self):
        _, _, _, payload = _flt_payload()
        # KXSPACEX has no trade and no map entry: its observation still files
        # it under its fallback category, which the bar must offer for the
        # k-hat breakdown (with no trade view behind it — the trade sections
        # show the empty view)
        assert "Science" in payload["categories"]
        assert _key(payload, "Science") not in payload["lists"][0]["views"]

    def test_an_empty_view_reports_no_drawdown_date(self):
        # A flat curve never falls, so there is no trough to date
        _, _, _, payload = _flt_payload()
        assert payload["empty"]["kpi"]["max_drawdown"] == "0.0%"

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
        _, _, _, payload = _flt_payload(sweep)
        sports = [t for t in trades if t.event_ticker.startswith(("KXNCAA", "KXNHL"))]
        slice_curve = backtester._build_equity_curve(sports, _FLT_START, 1000.0)
        on_axis = slice_curve[pd.to_datetime(slice_curve["date"])
                              <= pd.Timestamp(payload["dates"][-1])]
        assert len(on_axis) == len(cut) < len(slice_curve)
        kpis = {k: v for k, _, v, _ in dashboard._performance_kpis(on_axis, sports, 1000.0)}
        uncut = {k: v for k, _, v, _ in dashboard._performance_kpis(slice_curve, sports, 1000.0)}
        view = _view(payload, 0, _key(payload, "Sports"))
        assert view["kpi"] == kpis != uncut

    def test_the_sparse_encoding_round_trips_and_gaps_are_none(self):
        axis = pd.DatetimeIndex(pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03",
                                                "2026-01-04", "2026-01-05"]))
        dates = [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 4), date(2026, 1, 5)]
        sparse = dashboard._sparse_on_axis(dates, [1.0, 1.0, 2.5, float("nan")], axis, 2)
        assert sparse == [[0, 1.0], [2, None], [3, 2.5], [4, None]]
        assert _expand(sparse, 5) == [1.0, 1.0, None, 2.5, None]
        assert dashboard._sparse_on_axis(dates, [1.0] * 4, pd.DatetimeIndex([]), 2) == []


class TestFilterPage:
    """The bar, the data block and the script, on a whole rendered page."""

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
        """The filter block, inflated as the script inflates it (base64, then
        gzip) and parsed strictly (a NaN/Infinity token fails the test)."""
        opening = '<script type="text/plain" id="dash-data" data-encoding="gzip+base64">'
        start = page.index(opening) + len(opening)
        end = page.index("</script>", start)
        raw = gzip.decompress(base64.b64decode(page[start:end], validate=True))
        return json.loads(raw.decode("utf-8"), parse_constant=_fail_on_constant)

    def test_the_bar_preselects_the_primary_and_counts_its_trades(self, monkeypatch, tmp_path):
        page = self._page(monkeypatch, tmp_path)
        band = re.search(r'<select id="flt-band"[^>]*>(.*?)</select>', page).group(1)
        assert re.findall(r'<option value="(\d+)"( selected)?>(.*?)</option>', band) == [
            ("0", " selected", "max(tier,0)-1 (primary)"),
            ("1", "", "max(tier,0.3)-0.6"), ("2", "", "max(tier,0.3)-1")]
        cats = re.search(r'<select id="flt-cat"[^>]*>(.*?)</select>', page).group(1)
        assert re.findall(r"<option value=\"\d*\">(.*?)</option>", cats) == [
            "All categories", "Commodities (1)", "Other (1)", "Science (0)", "Sports (3)"]
        assert ("Showing every trade of the run at the primary spread band max(tier,0)-1, "
                "k = 0.75: 5 trades. Not filtered by this bar: "
                f"{dashboard._UNFILTERED_SECTIONS}.") in page
        assert re.search(r'Trades found: <span id="hdr-trades">5</span>', page)

    def test_the_selects_wait_for_the_script_and_are_never_restored(
            self, monkeypatch, tmp_path):
        # Disabled until the script has inflated its data; autocomplete off,
        # so a reload cannot restore a choice the rendered page is not showing
        page = self._page(monkeypatch, tmp_path)
        for sel in ("flt-band", "flt-cat", "flt-tag"):
            assert f'<select id="{sel}" disabled autocomplete="off">' in page

    def test_the_block_is_strict_json_and_escapes_kalshi_text(self, monkeypatch, tmp_path):
        trades = [dataclasses.replace(t, title_a="</script><b>x</b>") for t in _flt_trades()]
        curve = backtester._build_equity_curve(trades, _FLT_START, 1000.0)
        sweep = BacktestSweep(
            primary=SweepPoint(k=0.75, trades=trades, equity_df=curve, spread_band=(0.0, 1.0)),
            points=[], calibration=None)
        series = {"KXNCAAMBGAME": ("<i>Sports</i>", ("B</script>",))}
        page = self._page(monkeypatch, tmp_path, sweep, series)
        data = self._data(page)            # strict: a NaN/Infinity token fails
        assert "<i>Sports</i>" in data["categories"]
        assert all("<b>" not in text for text in data["lists"][0]["kt"])
        bar = page[page.index('<div id="flt-bar"'):page.index("</select>", page.index('id="flt-tag"'))]
        assert "<i>" not in bar and "&lt;i&gt;Sports&lt;/i&gt;" in bar
        # Base64 has no "<", so nothing inside the block can close it early:
        # the block ends exactly where the encoded bytes end
        opening = 'data-encoding="gzip+base64">'
        block = page[page.index(opening) + len(opening):]
        assert re.fullmatch(r"[A-Za-z0-9+/=]+", block[:block.index("</script>")])

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
        missing = sorted(i for i in literal | dynamic if f'id="{i}"' not in page)
        assert literal and not missing

    def test_the_script_comes_after_every_section(self, monkeypatch, tmp_path):
        page = self._page(monkeypatch, tmp_path)
        assert page.index('id="flt-bar"') < page.index("Portfolio Performance")
        assert page.index("Benchmark Comparison") < page.index('id="dash-data"')
        assert page.index('id="dash-data"') < page.index("function inflate()")

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
        assert 'id="flt-unavailable"' in page and "Portfolio Performance" in page
        assert "function inflate()" not in page
        assert any("could not be built" in r.getMessage() for r in caplog.records)

    def test_one_k_prices_the_whole_page(self, monkeypatch, tmp_path):
        # No override: the sweep's primary k (0.60 here) prices the Risk
        # section's Kelly scatter AND the filter's views of it, and is the k
        # the summary names — never the config constant beside it
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
        assert self._data(page)["lists"][0]["kx"] == pytest.approx(kx)
        assert kx != pytest.approx(dashboard._kelly_points(sweep.primary.trades, None)[0])
        assert "max(tier,0)-1, k = 0.60: 5 trades." in page


class TestFilterSummary:
    """The line under the bar says what the page shows, from templates the
    script fills too (D.text) — so the script holds no sentence of its own."""

    T = dashboard._SUMMARY_TEMPLATES

    def test_the_whole_band_and_a_slice(self):
        where = "the primary spread band max(tier,0)-1, k = 0.75"
        assert dashboard._filter_summary_text(self.T, where, True, None, 5, 5) == (
            "Showing every trade of the run at the primary spread band max(tier,0)-1, "
            f"k = 0.75: 5 trades. Not filtered by this bar: {dashboard._UNFILTERED_SECTIONS}.")
        other = dashboard._filter_summary_text(self.T, "spread band x, k = 0.75", False,
                                               None, 3, 3)
        assert "3 trades. This band is its own simulation" in other
        sliced = dashboard._filter_summary_text(self.T, where, True, "Sports · Hockey", 1, 5)
        assert sliced.startswith("Showing Sports · Hockey within the run at the primary "
                                 "spread band max(tier,0)-1, k = 0.75: 1 of its 5 trades. ")
        # Every figure a slice draws from an equity curve is its contribution
        for figure in ("return", "drawdown", "Sharpe", "Sortino", "median monthly return",
                       "benchmark's strategy row"):
            assert figure in sliced

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
                "k not recorded: 5 trades.") in page

    def test_the_script_fills_the_templates_and_writes_none_of_its_own(self):
        js = dashboard._FILTER_JS
        assert "T.all" in js and "T.slice" in js and "T.unfiltered" in js
        for phrase in ("Showing", "contribution", "Not filtered", "its own simulation"):
            assert phrase not in js


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
    """The selects (options, "selected", "disabled") and every chart
    (data and layout, typed arrays decoded) of a rendered page."""
    selects = {}
    for m in re.finditer(r'<select id="([^"]+)"([^>]*)>(.*?)</select>', page, re.S):
        options = [{"value": value, "text": html.unescape(text), "selected": bool(chosen)}
                   for value, chosen, text in re.findall(
                       r'<option value="([^"]*)"( selected)?>(.*?)</option>', m.group(3))]
        selects[m.group(1)] = {"options": options, "disabled": " disabled" in m.group(2)}
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
    return {"selects": selects, "charts": charts}


def _script_body() -> str:
    """dashboard._FILTER_JS without its <script> tags, its inflate() handing
    back the already-inflated data (the block's encoding is pinned by
    TestFilterPage's strict decode)."""
    body = dashboard._FILTER_JS.strip()
    body = body[len("<script>"):-len("</script>")]
    start = body.index("  function inflate() {")
    end = body.index("\n  }\n", start) + len("\n  }\n")
    return body[:start] + "  function inflate() { return Promise.resolve(__DATA); }\n" + body[end:]


def _run_script(tmp_path: Path, page: str, steps: list, pre: tuple = (),
                no_decompression: bool = False) -> dict:
    """
    Run the filter script over a rendered page under tests/js/filter_harness.js.

    Args:
        tmp_path (Path): Where the assembled program is written.
        page (str): The rendered page.
        steps (list): The harness's steps (["wait"], ["set", id, value],
            ["fire", id], ["zoom", id], ["hide", id], ["snap", name]).
        pre (tuple): (select id, value) pairs set BEFORE the script runs — a
            browser restoring a reader's last choice.
        no_decompression (bool): Run as a browser without DecompressionStream.

    Returns:
        dict: The snapshots the steps took, by name.
    """
    runtime = _js_runtime()
    if runtime is None:
        pytest.skip("no JavaScript runtime (node or jsc) to run the filter script with")
    program = "\n".join([
        _JS_HARNESS.read_text(encoding="utf-8"),
        f"var __PAGE = {json.dumps(_page_elements(page))};",
        f"var __DATA = {json.dumps(TestFilterPage._data(page))};",
        "__setup(__PAGE);",
        *(f"document.getElementById({json.dumps(i)}).value = {json.dumps(v)};" for i, v in pre),
        "window.DecompressionStream = undefined;" if no_decompression else "",
        _script_body(),
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


_FLT_SELECTS = ("flt-band", "flt-cat", "flt-tag")


class TestFilterScript:
    """The page-wide filter script, run over the page Python rendered: what it
    draws, writes and enables for each choice. Skipped without a runtime."""

    def _page(self, monkeypatch, tmp_path, sweep=None) -> str:
        return TestFilterPage()._page(monkeypatch, tmp_path, sweep)

    def test_on_load_the_bar_is_reset_disabled_and_nothing_is_drawn(
            self, monkeypatch, tmp_path):
        # A browser restored a reader's last choice (band 1, category 1)
        page = self._page(monkeypatch, tmp_path)
        snaps = _run_script(tmp_path, page, [["snap", "loaded"], ["wait"], ["snap", "ready"]],
                            pre=(("flt-band", "1"), ("flt-cat", "1")))
        loaded, ready = snaps["loaded"], snaps["ready"]
        assert {i: loaded["selects"][i]["value"] for i in _FLT_SELECTS} == {
            "flt-band": "0", "flt-cat": "", "flt-tag": ""}
        assert all(loaded["selects"][i]["disabled"] for i in _FLT_SELECTS)
        assert not any(ready["selects"][i]["disabled"] for i in _FLT_SELECTS)
        assert loaded["reacts"] == ready["reacts"] == []

    def test_every_redraw_is_the_chart_python_drew_for_that_view(
            self, monkeypatch, tmp_path):
        # Away to another band and back: the primary's unfiltered view, drawn
        # by the script, must be the page Python rendered
        page = self._page(monkeypatch, tmp_path)
        snap = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-band", "1"], ["fire", "flt-band"],
            ["set", "flt-band", "0"], ["fire", "flt-band"], ["snap", "back"]])["back"]
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
            ["set", "flt-band", "1"], ["fire", "flt-band"], ["snap", "after"]])["after"]
        react = _last_react(snap, "perf-cum")
        python = _page_elements(page)["charts"]["perf-cum"]["layout"]
        assert react["layout"].get("xaxis") == python.get("xaxis")
        assert (react["layout"].get("xaxis") or {}).get("autorange") is not False
        assert "visible" not in react["data"][0]
        assert "selectedpoints" not in _last_react(snap, "risk-kelly")["data"][0]
        # Another band's whole run, in the same words Python would use
        data = TestFilterPage._data(page)
        n = data["lists"][data["bands"][1]["list"]]["views"]["all"]["n"]
        assert snap["text"]["flt-summary"] == dashboard._filter_summary_text(
            data["text"], data["bands"][1]["scenario"], False, None, n, n)

    def test_a_tag_picked_under_all_categories_selects_its_category(
            self, monkeypatch, tmp_path):
        page = self._page(monkeypatch, tmp_path)
        data = TestFilterPage._data(page)
        sports = data["categories"].index("Sports")
        hockey = data["subcats"].index([sports, "Hockey"])
        snap = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-tag", str(hockey)], ["fire", "flt-tag"], ["snap", "s"]])["s"]
        assert snap["selects"]["flt-cat"]["value"] == str(sports)
        assert snap["selects"]["flt-tag"]["value"] == str(hockey)
        assert [text for _, text in snap["selects"]["flt-tag"]["options"]] == [
            "All tags", "Basketball (2)", "Hockey (1)"]
        assert snap["text"]["flt-summary"] == dashboard._filter_summary_text(
            data["text"], data["bands"][0]["scenario"], True, "Sports · Hockey", 1, 5)
        assert snap["text"]["hdr-trades"] == snap["text"]["kpi-trades"] == "1"
        # The row-dependent chart's height lands on both boxes, whichever one
        # the page's plotly.py wrote it on
        h = data["lists"][0]["views"][f"s{hockey}"]["sub"]["h"]
        assert snap["heights"]["dec-sub"] == snap["ownHeights"]["dec-sub"] == f"{h}px"

    def test_a_band_without_the_selection_shows_no_trades(self, monkeypatch, tmp_path):
        # Band 1 holds only same-title trades: none of them is in "Other"
        page = self._page(monkeypatch, tmp_path)
        data = TestFilterPage._data(page)
        other = str(data["categories"].index("Other"))
        snap = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-band", "1"], ["fire", "flt-band"],
            ["set", "flt-cat", other], ["fire", "flt-cat"], ["snap", "s"]])["s"]
        assert [text for _, text in snap["selects"]["flt-cat"]["options"]] == [
            "All categories", "Commodities (1)", "Other (0)", "Science (0)", "Sports (2)"]
        for prefix in ("dec", "cal", "diag", "risk"):
            assert (snap["display"][f"{prefix}-empty"], snap["display"][f"{prefix}-body"]) \
                == ("", "none")
        assert snap["text"]["hdr-trades"] == "0"
        assert snap["text"]["kpi-max_drawdown"] == "0.0%"
        assert "This band is its own simulation" not in snap["text"]["flt-summary"]

    def test_the_khat_table_redraws_as_python_rendered_it(self, monkeypatch, tmp_path):
        page = self._page(monkeypatch, tmp_path)
        snap = _run_script(tmp_path, page, [
            ["wait"], ["set", "flt-band", "1"], ["fire", "flt-band"],
            ["set", "flt-band", "0"], ["fire", "flt-band"], ["snap", "back"]])["back"]
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
            ["set", "flt-cat", commodities], ["fire", "flt-cat"], ["snap", "empty"],
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
            ["wait"], ["set", "flt-cat", sports], ["fire", "flt-cat"],
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
            ["wait"], ["set", "flt-tag", hockey], ["fire", "flt-tag"],
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
            ["wait"], ["set", "flt-band", "1"], ["fire", "flt-band"], ["snap", "s"]])["s"]
        assert (snap["display"]["khat-body"], snap["display"]["khat-empty"]) == ("none", "")
        assert snap["text"]["khat-empty"] == data["text"]["khat_not_recorded"]

    def test_a_browser_that_cannot_inflate_keeps_the_bar_disabled(
            self, monkeypatch, tmp_path):
        page = self._page(monkeypatch, tmp_path)
        snap = _run_script(tmp_path, page, [["wait"], ["snap", "s"]],
                           no_decompression=True)["s"]
        assert all(snap["selects"][i]["disabled"] for i in (*_FLT_SELECTS, "khat-group"))
        assert snap["text"]["flt-summary"].startswith(
            "The filter could not load its data (this browser cannot decompress it)")
        assert snap["reacts"] == []


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
        _, runs, _, payload = _flt_payload()
        assert len(payload["khat"]) == len(runs) == 3
        assert payload["khat"][1] is None and payload["khat"][2] is None
        groups = payload["khat"][0]["groups"]
        # Science (seen only in an observation) is a group; Commodities (seen
        # only in trades) is not
        assert _key(payload, "Science") in groups
        assert _key(payload, "Commodities") not in groups
        assert payload["khat_blank"] == ["—"] * 5
        assert payload["styles"]["khat_height"] == list(dashboard._KHAT_HEIGHT)
        least, per_row, axes = dashboard._KHAT_HEIGHT
        assert dashboard._khat_chart_height(9) == max(least, per_row * 9 + axes)

    def test_the_default_render_is_by_category_at_the_primary_band(self):
        _, _, _, payload = _flt_payload()
        section = dashboard._section_khat(payload, 0.625)
        data, layout = _nth_figure(section, 0)
        groups = payload["khat"][0]["groups"]
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
        # The line marks the k the run was sized at, printed exactly
        assert layout["shapes"][0]["x0"] == 0.625
        assert layout["annotations"][0]["text"] == "sized at k = 0.625"
        # The table repeats the bars, one row each, with Python's cells
        assert section.count("<tr style='border-bottom:1px solid #E0E0E0'>") == 4
        assert "".join(f"<td style='padding:4px 12px;'>{c}</td>"
                       for c in groups["all"]["cells"]) in section

    def test_group_by_stays_outside_the_body_it_hides(self):
        # A selection with nothing to draw hides the body; the grouping that
        # could draw something must stay reachable, so the select sits
        # before the notice and outside the body — disabled until the
        # script has its data, and never restored by a reload
        _, _, _, payload = _flt_payload()
        section = dashboard._section_khat(payload, 0.75)
        select = section.index('<select id="khat-group" disabled autocomplete="off">')
        assert select < section.index('<p id="khat-empty"') < section.index('<div id="khat-body"')
        assert '<option value="band">' in section

    def test_a_band_without_a_calibration_says_it_was_not_recorded(self):
        trades = _flt_trades()
        curve = backtester._build_equity_curve(trades, _FLT_START, 1000.0)
        runs = [dashboard._BandRun((0.0, 1.0), "max(tier,0)-1", trades, curve, None)]
        payload = dashboard._filter_payload(runs, 0, _FLT_START, 1000.0, _FLT_SERIES, 0.75,
                                            "k = 0.75")
        section = dashboard._section_khat(payload, None)
        notice = html.escape(dashboard._KHAT_TEXT["khat_not_recorded"])
        assert f'color:#616161;">{notice}</p>' in section
        assert '<div id="khat-body" style="display:none">' in section
        assert "shapes" not in json.dumps(_nth_figure(section, 0)[1])   # no k to mark

    def test_an_empty_population_is_not_called_unrecorded(self):
        # Every candidate a premise violation: measured, and nothing counted
        empty = IntervalCalibration(
            pooled=backtester._calibration_bucket("POOLED", 0.0, ()), buckets=[],
            excluded_premise_violations=3, observations=())
        trades = _flt_trades()
        curve = backtester._build_equity_curve(trades, _FLT_START, 1000.0)
        runs = [dashboard._BandRun((0.0, 1.0), "max(tier,0)-1", trades, curve, empty)]
        payload = dashboard._filter_payload(runs, 0, _FLT_START, 1000.0, _FLT_SERIES, 0.75,
                                            "k = 0.75")
        section = dashboard._section_khat(payload, 0.75)
        assert html.escape(dashboard._KHAT_TEXT["khat_none"]) in section
        assert '<div id="khat-body" style="display:none">' in section

    def test_without_the_filter_data_the_section_says_so(self):
        section = dashboard._section_khat(None, 0.75)
        assert "could not be built" in section and "khat-group" not in section

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


class TestFilterPageSize:
    """A band sweep whose 36 bands each traded a DIFFERENT list — the
    worst case for the filter, which ships a view per band list x category
    x tag — stays well inside the page budget the scenario explorer set
    (5 MB), because the payload is gzip-packed (_packed_json_script). Each
    band also carries a 300-entry k-hat population, which the k-hat
    breakdown ships per band x category x tag."""

    def test_36_distinct_band_lists_stay_under_budget(self, monkeypatch, tmp_path):
        import random
        monkeypatch.setattr(dashboard, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(dashboard.yf, "download",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
        rng = random.Random(3)
        # 4 categories x 2 tags: 13 views per list, 468 in all
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

        bands = [(lo, hi) for lo in config.SPREAD_BAND_SWEEP_FLOORS
                 for hi in config.SPREAD_BAND_SWEEP_CEILINGS]
        scenarios, cals = [], {}
        for band in bands:
            trades = sorted((trade(i) for i in range(40)), key=lambda t: t.entry_date)
            scenarios.append(SweepPoint(
                k=0.75, trades=trades, spread_band=band,
                equity_df=backtester._build_equity_curve(trades, start, 10_000.0)))
            obs = tuple(backtester.CalibrationObservation(
                rng.randint(1, 30), rng.uniform(0.15, 0.8), rng.random() < 0.4,
                f"{rng.choice(series)}-{j}", "Other") for j in range(300))
            cals[band] = IntervalCalibration(
                pooled=backtester._calibration_bucket("POOLED", 0.0, obs), buckets=[],
                excluded_premise_violations=0, observations=obs)
        sweep = BacktestSweep(primary=scenarios[0], points=[scenarios[0]],
                              calibration=cals[bands[0]], label_coverage=_scn_coverage(),
                              scenarios=scenarios, calibrations_by_band=cals)
        out = dashboard.generate_dashboard(
            scenarios[0].trades, scenarios[0].equity_df, start, 10_000.0, sweep=sweep,
            interval_discount=0.75, series_categories=categories)
        page = out.read_text(encoding="utf-8")
        data = TestFilterPage._data(page)
        assert len(data["lists"]) == 36          # nothing collapsed: every list is its own
        assert all(band is not None for band in data["khat"])
        assert out.stat().st_size <= 3_000_000, f"page was {out.stat().st_size} bytes"


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
        # The risk section's charts carry fixed ids now (the page-wide filter
        # redraws them by id); the k selector's chart still gets Plotly's
        # random UUID, which the normaliser must still replace.
        assert "UUID" in dashboard_golden.normalize(sections["interval_discount_healthy"])
