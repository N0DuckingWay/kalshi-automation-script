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
import dataclasses
import json
import math
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from kalshi_betting import backtester, config, dashboard
from kalshi_betting.backtester import (
    BacktestSweep,
    BacktestTrade,
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
            "Mean per trade (equal stake)", "Total return", "H1 return",
            "H2 return", "Trade count"]
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
        count = buttons[4]["args"][0]
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
        risk = dashboard_golden.render_sections()["risk"]
        once = dashboard_golden.normalize(risk)
        assert once == dashboard_golden.normalize(risk)
        assert "UUID" in once and '"template":' not in once
        assert dashboard_golden.normalize(risk.replace("Kelly", "Kellx")) != once
