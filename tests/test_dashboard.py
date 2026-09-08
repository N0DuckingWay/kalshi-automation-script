"""Tests for dashboard.py — HTML escaping of Kalshi-controlled titles (BS-20)
and the _max_drawdown empty/all-NaN guard (BS-30).

generate_dashboard() pulls in yfinance (network) and Plotly's full HTML
serialization; the escaping and drawdown fixes are exercised directly against
the cheapest reliable seam instead — _section_diagnostics() (the HTML table
render site, _trow) and _section_risk() (the Plotly hover-text render site) —
so these tests stay fully offline. Both sites are Kalshi-controlled
(BacktestTrade.title_a is a market question straight from the API).
"""
from datetime import date

import pandas as pd
import pytest

from kalshi_betting.backtester import BacktestTrade
from kalshi_betting.config import (
    SAME_TITLE_CO_RESOLVE_PROB,
    fee_leg_exact,
    fee_per_pair_approx,
    time_series_profit_prob,
)
from kalshi_betting.dashboard import (
    _kelly_fraction,
    _max_drawdown,
    _section_diagnostics,
    _section_risk,
)

_XSS_TITLE = "<script>alert(1)</script>Will BTC exceed $80k by December 2026 or later?"


def make_trade(title_a: str = "Will BTC exceed $80k?", profit: float | None = None) -> BacktestTrade:
    """Factory for a coherent time-series BacktestTrade covering every field the
    dashboard section builders under test read.

    YES on the earlier contract at 0.30 and NO on the later at 0.40 (later YES
    ask 0.60, earlier NO ask 0.70 — reporting only), n=5, settled in the
    "event by A" win cell (A=YES, hence B=YES). _section_risk calls the
    five-argument _kelly_fraction on these entry prices live.
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
    )


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
