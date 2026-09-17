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
"""
import dataclasses
import math
import re
from datetime import date, timedelta

import pandas as pd
import pytest

from kalshi_betting import backtester, config, dashboard
from kalshi_betting.backtester import (
    BacktestSweep,
    BacktestTrade,
    IntervalCalibration,
    IntervalCalibrationBucket,
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
)

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
        # YES 0.30 + NO 0.40, later YES ask 0.60: p = 0.775, f* ≈ 0.1884
        pA, nA, pB, nB = 0.30, 0.70, 0.60, 0.40
        net_spread = (1.0 - pA - nB) - fee_per_pair_approx(pA, nB)
        b = net_spread / (pA + nB)
        p = time_series_profit_prob(pA, pB)
        assert _kelly_fraction(pA, nA, pB, nB, "time_series") == pytest.approx(p - (1 - p) / b)
        assert _kelly_fraction(pA, nA, pB, nB, "time_series") == pytest.approx(0.1884, abs=1e-4)

    def test_time_series_wide_book_clamps_to_zero(self):
        assert _kelly_fraction(0.30, 0.70, 0.60, 0.50, "time_series") == 0.0

    def test_same_title_prices_nA_pB_on_the_prior(self):
        nA, pB = 0.20, 0.30
        net_spread = (1.0 - nA - pB) - fee_per_pair_approx(nA, pB)
        b = net_spread / (nA + pB)
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
        # still produce the configured-k fixture's ~0.1884 fraction.
        monkeypatch.setattr(config, "TIME_SERIES_INTERVAL_PROB_DISCOUNT", 1.0)
        assert _kelly_fraction(self._PA, self._NA, self._PB, self._NB,
                               "time_series", k=0.75) == pytest.approx(0.1884, abs=1e-4)
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
