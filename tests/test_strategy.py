"""Tests for strategy.py Kelly sizing and portfolio selection.

Time-series pairs buy YES on the earlier contract (market_a, at pA) and NO on
the later one (market_b, at nB); same-title pairs buy NO on market_a (nA) and
YES on market_b (pB). Every fixture therefore carries a REAL float nB — a
MagicMock auto-attribute would TypeError inside compute_trade's arithmetic.
"""
import ast
import inspect
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from kalshi_betting import backtester, config, dashboard, strategy
from kalshi_betting.config import (
    BUDGET_FRACTION,
    SAME_TITLE_CO_RESOLVE_PROB,
    fee_leg_exact,
    fee_per_pair_approx,
    time_series_profit_prob,
)
from kalshi_betting.strategy import TradeSpec, _kelly_p, compute_trade, select_portfolio


def make_pair(
    pA: float = 0.70,
    pB: float = 0.30,
    nA: float = 0.20,
    nB: float = 0.65,
    tradeable: bool = True,
    pair_type: str = "same_title",
    max_contracts: int = 0,
) -> MagicMock:
    """Factory for CandidatePair-like mocks; avoids importing the real dataclass.

    nB is set as a real float (never left to MagicMock auto-vivification):
    scanner.leg_prices reads it directly for time-series pairs.
    """
    pair = MagicMock()
    pair.pA = pA
    pair.pB = pB
    pair.nA = nA
    pair.nB = nB
    pair.tradeable = tradeable
    pair.pair_type = pair_type
    pair.max_contracts = max_contracts
    pair.canonical_title = "test pair"
    now = datetime.now(UTC)
    pair.market_a.close_time = now + timedelta(days=15)
    pair.market_b.close_time = now + timedelta(days=30)
    return pair


def make_spec(
    monthly_profit_ratio: float = 0.10,
    pair_type: str = "time_series",
    total_cost: float = 100.0,
    total_cost_with_fees: float | None = None,
) -> TradeSpec:
    """Factory for TradeSpec with the minimum fields needed by select_portfolio()."""
    pair = make_pair(pair_type=pair_type)
    return TradeSpec(
        pair=pair,
        x=1,
        y=1,
        total_cost=total_cost,
        # select_portfolio budgets against the fee-inclusive cost; default to
        # total_cost so affordability-by-cost tests keep their semantics
        total_cost_with_fees=total_cost if total_cost_with_fees is None else total_cost_with_fees,
        min_payoff=0.10,
        profit_ratio=0.05,
        days_to_close=30,
        monthly_profit_ratio=monthly_profit_ratio,
        kelly_p=0.90,
        kelly_fraction=0.10,
    )


# The plan's flow-through fixture: earlier YES ask 0.30, later YES ask 0.60 /
# NO ask 0.40 (13-day gap in the backtester; the gap only picks the tier here).
_TS_PA, _TS_PB, _TS_NA, _TS_NB = 0.30, 0.60, 0.70, 0.40


def _ts_kelly_fraction(pA: float, pB: float, nB: float) -> float:
    """Uncapped Kelly f* = p - (1-p)/b for a time-series pair, from the config
    helpers alone — the oracle every sizer (strategy, dashboard, backtester)
    must agree with."""
    net_spread = (1.0 - pA - nB) - fee_per_pair_approx(pA, nB)
    b = net_spread / (pA + nB)
    p = time_series_profit_prob(pA, pB)
    return p - (1.0 - p) / b


class TestKellyP:
    """Probability-of-profit models: the discounted market gap for time-series
    (config.time_series_profit_prob) and the fixed co-resolution prior for
    same-title."""

    def test_time_series_discounted_gap_model(self):
        pair = make_pair(pA=0.30, pB=0.60, nB=0.40, pair_type="time_series")
        # p = 1 - k * (pB - pA) = 1 - 0.75 * 0.30 = 0.775
        assert _kelly_p(pair) == pytest.approx(0.775)
        assert _kelly_p(pair) == pytest.approx(time_series_profit_prob(0.30, 0.60))

    def test_time_series_model_is_not_the_old_expression(self):
        # The pre-2026-09 formula 1 - pA*(1-pB) modelled the impossible
        # A=YES/B=NO cell; on this fixture it would read 0.88, not 0.775
        pair = make_pair(pA=0.30, pB=0.60, nB=0.40, pair_type="time_series")
        assert _kelly_p(pair) != pytest.approx(1.0 - 0.30 * (1.0 - 0.60))

    def test_same_title_fixed_prior(self):
        pair = make_pair(pA=0.80, pB=0.40, pair_type="same_title")
        assert _kelly_p(pair) == SAME_TITLE_CO_RESOLVE_PROB

    def test_same_title_ignores_prices(self):
        pair_high = make_pair(pA=0.90, pB=0.80, pair_type="same_title")
        pair_low = make_pair(pA=0.10, pB=0.05, pair_type="same_title")
        assert _kelly_p(pair_high) == _kelly_p(pair_low)


class TestComputeTrade:
    def test_returns_none_when_not_tradeable(self):
        pair = make_pair(tradeable=False)
        assert compute_trade(pair, 100_000) is None

    def test_returns_none_when_no_spread(self):
        # nA + pB = 1.0 means zero gross spread
        pair = make_pair(nA=0.50, pB=0.50)
        assert compute_trade(pair, 100_000) is None

    def test_returns_none_when_spread_negative(self):
        # nA + pB > 1 means the bot pays more than it can earn
        pair = make_pair(nA=0.60, pB=0.50)
        assert compute_trade(pair, 100_000) is None

    def test_returns_none_when_price_at_boundary(self):
        assert compute_trade(make_pair(pB=0.0), 100_000) is None
        assert compute_trade(make_pair(pB=1.0), 100_000) is None
        assert compute_trade(make_pair(nA=0.0), 100_000) is None
        assert compute_trade(make_pair(nA=1.0), 100_000) is None

    def test_time_series_boundary_checks_use_leg_prices(self):
        # For time_series the leg prices are pA/nB — a boundary nA or pB (not
        # leg prices) must NOT reject, while a boundary pA or nB must
        ok = make_pair(pA=0.30, pB=0.60, nA=0.0, nB=0.40, pair_type="time_series")
        assert compute_trade(ok, 1_000_000) is not None
        assert compute_trade(make_pair(pA=0.0, pB=0.60, nB=0.40, pair_type="time_series"), 1_000_000) is None
        assert compute_trade(make_pair(pA=0.30, pB=0.60, nB=1.0, pair_type="time_series"), 1_000_000) is None

    def test_returns_trade_spec_for_valid_same_title_pair(self):
        # same_title pair: p=0.95 fixed prior; nA=0.20+pB=0.30=0.50 < 1, wide spread gives positive Kelly
        pair = make_pair(nA=0.20, pB=0.30, pair_type="same_title")
        result = compute_trade(pair, 100_000)
        assert result is not None
        assert result.x >= 1
        assert result.y == result.x
        assert result.min_payoff > 0

    def test_kelly_fraction_capped_at_budget_fraction(self):
        # Even with a huge edge, the Kelly fraction must not exceed BUDGET_FRACTION
        pair = make_pair(nA=0.01, pB=0.01)  # very cheap pair, massive edge
        result = compute_trade(pair, 1_000_000)
        assert result is not None
        assert result.kelly_fraction <= BUDGET_FRACTION + 1e-9

    def test_respects_max_contracts_limit(self):
        pair = make_pair(nA=0.20, pB=0.30, pair_type="same_title", max_contracts=2)
        result = compute_trade(pair, 1_000_000)
        assert result is not None
        assert result.x <= 2

    def test_returns_none_when_fees_eat_spread_at_small_n(self):
        # At very tight spreads with n=1, ceiling-rounded fees can consume the profit.
        # nA=0.49, pB=0.48 → spread ≈ 0.03, fees ≈ 0.02 per leg → marginal at n=1.
        pair = make_pair(nA=0.49, pB=0.48)
        # This may return None or a spec depending on exact fee math — just ensure no crash.
        result = compute_trade(pair, 100)  # tiny balance forces n=1
        if result is not None:
            assert result.min_payoff > 0

    def test_days_to_close_at_least_one(self):
        pair = make_pair(nA=0.20, pB=0.30, pair_type="same_title")
        result = compute_trade(pair, 100_000)
        assert result is not None
        assert result.days_to_close >= 1

    def test_returns_none_when_budget_cannot_afford_one_contract(self):
        # Balance $1, Kelly cap 20% → budget $0.20 < one pair at $0.50.
        # Forcing n=1 would silently exceed the Kelly fraction, so: None.
        pair = make_pair(nA=0.20, pB=0.30, pair_type="same_title")
        assert compute_trade(pair, 100) is None

    def test_total_cost_with_fees_exceeds_total_cost(self):
        # The fee-inclusive cost must include both legs' exact taker fees
        pair = make_pair(nA=0.20, pB=0.30, pair_type="same_title")
        result = compute_trade(pair, 100_000)
        assert result is not None
        assert result.total_cost_with_fees > result.total_cost

    def test_per_leg_costs_sum_to_total(self):
        # cost_with_fees_a + cost_with_fees_b must equal total_cost_with_fees —
        # same terms, same fee calls, just not summed together. This is the
        # invariant the collateral transfer planner relies on.
        pair = make_pair(nA=0.20, pB=0.30, pair_type="same_title")
        result = compute_trade(pair, 100_000)
        assert result is not None
        assert result.cost_with_fees_a + result.cost_with_fees_b == pytest.approx(
            result.total_cost_with_fees
        )

    def test_per_leg_costs_match_component_construction(self):
        # Each leg's cost is that leg's own contracts-times-price plus its own
        # exact ceiling-rounded fee — verify against a direct re-derivation
        # rather than trusting compute_trade's internal arithmetic.
        pair = make_pair(nA=0.20, pB=0.30, pair_type="same_title")
        result = compute_trade(pair, 100_000)
        assert result is not None
        assert result.cost_with_fees_a == pytest.approx(
            result.x * pair.nA + fee_leg_exact(result.x, pair.nA)
        )
        assert result.cost_with_fees_b == pytest.approx(
            result.y * pair.pB + fee_leg_exact(result.y, pair.pB)
        )

    def test_fee_inclusive_cost_never_exceeds_kelly_budget(self):
        # Regression: n was originally derived from budget_dollars / (nA + pB),
        # which excludes fees entirely — fees were added on top afterward, so
        # total_cost_with_fees could exceed the capped Kelly budget the fraction
        # was supposed to bound. balance_cents=5000 with this pair reproduces
        # the overshoot under the old (pre-fix) sizing: naive n=16 costs $10.08
        # against a $10.00 budget.
        pair = make_pair(nA=0.30, pB=0.30, pair_type="same_title")
        result = compute_trade(pair, balance_cents=5000)
        assert result is not None
        budget_dollars = (5000 / 100.0) * result.kelly_fraction
        assert result.total_cost_with_fees <= budget_dollars + 1e-9


class TestComputeTradeTimeSeries:
    """The plan's flow-through fixture through compute_trade: YES on the
    earlier contract at 0.30 and NO on the later at 0.40 (later YES ask 0.60),
    $10,000 balance. Every expectation is derived from the config helpers in
    the test, not hardcoded, except the contract count and the dollar figures
    the plan pins."""

    @staticmethod
    def _pair(**overrides) -> MagicMock:
        kwargs = {"pA": _TS_PA, "pB": _TS_PB, "nA": _TS_NA, "nB": _TS_NB, "pair_type": "time_series"}
        kwargs.update(overrides)
        return make_pair(**kwargs)

    def test_costs_are_on_the_leg_prices(self):
        result = compute_trade(self._pair(), 1_000_000)
        assert result is not None
        x = result.x
        assert result.y == x
        # total_cost = x * (pA + nB), NOT x * (nA + pB) = x * 1.30
        assert result.total_cost == pytest.approx(x * (_TS_PA + _TS_NB))
        assert result.cost_with_fees_a == pytest.approx(x * _TS_PA + fee_leg_exact(x, _TS_PA))
        assert result.cost_with_fees_b == pytest.approx(x * _TS_NB + fee_leg_exact(x, _TS_NB))
        assert result.cost_with_fees_a + result.cost_with_fees_b == pytest.approx(
            result.total_cost_with_fees
        )

    def test_kelly_p_and_fraction_from_config_helpers(self):
        result = compute_trade(self._pair(), 1_000_000)
        assert result is not None
        assert result.kelly_p == pytest.approx(time_series_profit_prob(_TS_PA, _TS_PB))
        assert result.kelly_p == pytest.approx(0.775)
        expected_f = _ts_kelly_fraction(_TS_PA, _TS_PB, _TS_NB)
        # ~0.1884 — below the 20% cap, so Kelly (not the cap) sizes this pair
        assert expected_f == pytest.approx(0.1884, abs=1e-4)
        assert expected_f < BUDGET_FRACTION
        assert result.kelly_fraction == pytest.approx(expected_f)

    def test_flow_through_dollar_figures(self):
        # Kelly budget 1884.08 → raw n 2691, shrunk by the fee loop to 2575:
        # cost 1802.50, exact fees 37.86 + 43.26 = 81.12, cash out 1883.62,
        # win-scenario profit 2575 * 0.30 - 81.12 = 691.38
        result = compute_trade(self._pair(), 1_000_000)
        assert result is not None
        assert result.x == 2575
        assert result.total_cost == pytest.approx(1802.50)
        assert result.total_cost_with_fees == pytest.approx(1883.62)
        assert result.min_payoff == pytest.approx(691.38)
        assert result.total_cost_with_fees <= 10_000.0 * result.kelly_fraction + 1e-9

    def test_min_payoff_is_the_win_scenario_profit(self):
        # min_payoff = n * (1 - pA - nB) - exact fees; the in-between cell
        # loses total_cost_with_fees in full (there is no floor for time_series)
        result = compute_trade(self._pair(), 1_000_000)
        assert result is not None
        n = result.x
        assert result.min_payoff == pytest.approx(
            n * (1.0 - _TS_PA - _TS_NB) - fee_leg_exact(n, _TS_PA) - fee_leg_exact(n, _TS_NB)
        )

    def test_wide_later_book_returns_none(self):
        # Same YES-ask gap, later NO ask 0.50 instead of 0.40: pA + nB = 0.80
        # drives f* negative — Kelly says the market's in-between mass is not
        # overstated enough to pay for the wider book
        assert _ts_kelly_fraction(_TS_PA, _TS_PB, 0.50) < 0
        assert compute_trade(self._pair(nB=0.50), 1_000_000) is None

    def test_wide_gap_is_capped_at_budget_fraction(self):
        # 0.30 → 0.70 with NO ask 0.30: f* ≈ 0.214 → capped to BUDGET_FRACTION
        assert _ts_kelly_fraction(0.30, 0.70, 0.30) > BUDGET_FRACTION
        result = compute_trade(self._pair(pB=0.70, nB=0.30), 1_000_000)
        assert result is not None
        assert result.kelly_fraction == pytest.approx(BUDGET_FRACTION)

    def test_respects_depth_cap(self):
        result = compute_trade(self._pair(max_contracts=100), 1_000_000)
        assert result is not None
        assert result.x == 100
        assert result.total_cost == pytest.approx(70.0)

    def test_nA_and_pB_are_not_priced(self):
        # Changing the reporting-only nA leaves every dollar figure untouched;
        # pB only moves the probability (and hence the fraction / n)
        base = compute_trade(self._pair(), 1_000_000)
        other_nA = compute_trade(self._pair(nA=0.99), 1_000_000)
        assert base is not None and other_nA is not None
        assert (base.x, base.total_cost, base.min_payoff) == (
            other_nA.x, other_nA.total_cost, other_nA.min_payoff
        )


def _function_calls(module, func_name: str, callee: str) -> bool:
    """True when the named function in `module` contains a call to `callee`
    (as a bare name or an attribute), found by AST walk of the module source."""
    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    fn = sub.func
                    name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
                    if name == callee:
                        return True
            return False
    raise AssertionError(f"{module.__name__}.{func_name} not found")


class TestTimeSeriesKellyParity:
    """The three sizers — strategy._kelly_p / compute_trade, dashboard._kelly_fraction
    and the backtester (run_backtest -> _simulate_at_discount) — must all price
    the time-series probability through config.time_series_profit_prob, so the
    model cannot drift between live sizing, the backtest and the dashboard."""

    def test_kelly_p_equals_config_helper(self):
        pair = make_pair(pA=_TS_PA, pB=_TS_PB, nB=_TS_NB, pair_type="time_series")
        assert _kelly_p(pair) == time_series_profit_prob(_TS_PA, _TS_PB)

    def test_dashboard_fraction_equals_compute_trade_fraction(self):
        dash = dashboard._kelly_fraction(_TS_PA, _TS_NA, _TS_PB, _TS_NB, "time_series")
        assert dash == pytest.approx(_ts_kelly_fraction(_TS_PA, _TS_PB, _TS_NB))
        live = compute_trade(
            make_pair(pA=_TS_PA, pB=_TS_PB, nA=_TS_NA, nB=_TS_NB, pair_type="time_series"),
            1_000_000,
        )
        assert live is not None
        assert live.kelly_fraction == pytest.approx(dash)

    def test_dashboard_fraction_uses_leg_prices_not_nA_pB(self):
        # On this fixture nA + pB = 1.30 — the old leg mapping would return 0.0
        # (no spread), not the ~0.1884 the live sizer computes
        assert dashboard._kelly_fraction(_TS_PA, _TS_NA, _TS_PB, _TS_NB, "time_series") > 0.18

    def test_dashboard_same_title_unchanged(self):
        # Same-title still prices nA + pB on the fixed prior; nB is ignored
        expected_b = ((1.0 - 0.20 - 0.30) - fee_per_pair_approx(0.20, 0.30)) / 0.50
        expected = SAME_TITLE_CO_RESOLVE_PROB - (1 - SAME_TITLE_CO_RESOLVE_PROB) / expected_b
        assert dashboard._kelly_fraction(0.70, 0.20, 0.30, 0.65, "same_title") == pytest.approx(expected)
        assert dashboard._kelly_fraction(0.70, 0.20, 0.30, 0.99, "same_title") == pytest.approx(expected)

    def test_discount_of_one_never_trades(self, monkeypatch):
        # k = 1 (market-implied): p = 1 - (pB - pA) → f* < 0 for every pair;
        # compute_trade returns None and the dashboard clamps to 0.0
        monkeypatch.setattr(config, "TIME_SERIES_INTERVAL_PROB_DISCOUNT", 1.0)
        assert _ts_kelly_fraction(_TS_PA, _TS_PB, _TS_NB) < 0
        pair = make_pair(pA=_TS_PA, pB=_TS_PB, nA=_TS_NA, nB=_TS_NB, pair_type="time_series")
        assert compute_trade(pair, 1_000_000) is None
        assert dashboard._kelly_fraction(_TS_PA, _TS_NA, _TS_PB, _TS_NB, "time_series") == 0.0
        # Also the wide-gap fixture that is capped under k = 0.75
        assert compute_trade(make_pair(pA=0.30, pB=0.70, nA=0.70, nB=0.30, pair_type="time_series"), 1_000_000) is None

    def test_ast_strategy_kelly_p_calls_helper(self):
        assert _function_calls(strategy, "_kelly_p", "time_series_profit_prob")

    def test_ast_dashboard_kelly_fraction_calls_helper(self):
        assert _function_calls(dashboard, "_kelly_fraction", "time_series_profit_prob")

    def test_ast_backtester_prices_through_helper(self):
        # Pass 1's k-dependent scoring moved into _simulate_at_discount when the
        # calibration sweep landed. The invariant is unchanged — the backtester
        # must price time-series through the shared config helper and never
        # reimplement the formula — so it is pinned as a two-link chain.
        assert _function_calls(backtester, "_simulate_at_discount", "time_series_profit_prob")
        assert _function_calls(backtester, "run_backtest", "_simulate_at_discount")


class TestSelectPortfolio:
    def test_empty_input(self):
        assert select_portfolio([], 100_000) == []

    def test_returns_best_monthly_return_first(self):
        high = make_spec(monthly_profit_ratio=0.20, total_cost=50.0)
        low = make_spec(monthly_profit_ratio=0.05, total_cost=50.0)
        result = select_portfolio([low, high], 100_000)
        assert result[0] is high

    def test_greedy_stops_when_balance_insufficient(self):
        spec = make_spec(total_cost=600.0)
        # Balance of $500 cannot afford a $600 trade
        result = select_portfolio([spec], 50_000)  # $500
        assert result == []

    def test_selects_all_that_fit(self):
        a = make_spec(monthly_profit_ratio=0.20, total_cost=200.0)
        b = make_spec(monthly_profit_ratio=0.10, total_cost=200.0)
        # $500 balance fits both ($400 total cost)
        result = select_portfolio([a, b], 50_000)
        assert len(result) == 2

    def test_prefers_same_title_over_time_series_at_equal_return(self):
        # Near-arbitrage (same-title) outranks the directional bet (time-series)
        ts = make_spec(monthly_profit_ratio=0.10, pair_type="time_series", total_cost=100.0)
        st = make_spec(monthly_profit_ratio=0.10, pair_type="same_title", total_cost=100.0)
        result = select_portfolio([ts, st], 100_000)
        assert result[0] is st

    def test_partial_selection_when_second_trade_doesnt_fit(self):
        cheap = make_spec(monthly_profit_ratio=0.10, total_cost=100.0)
        expensive = make_spec(monthly_profit_ratio=0.05, total_cost=500.0)
        # Balance = $200: cheap fits, expensive doesn't
        result = select_portfolio([cheap, expensive], 20_000)
        assert cheap in result
        assert expensive not in result

    def test_budget_accounts_for_taker_fees(self):
        # Contract cost alone fits the $500 balance, but the real cash need
        # (cost + both legs' fees) does not — the trade must be skipped, or
        # order submission would be rejected for insufficient funds.
        spec = make_spec(total_cost=498.0, total_cost_with_fees=503.0)
        assert select_portfolio([spec], 50_000) == []
        # Sanity: with fees still inside the balance, it is selected
        spec_ok = make_spec(total_cost=490.0, total_cost_with_fees=499.0)
        assert select_portfolio([spec_ok], 50_000) == [spec_ok]
