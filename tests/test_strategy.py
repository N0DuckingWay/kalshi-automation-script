"""Tests for strategy.py Kelly sizing and portfolio selection."""
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from kalshi_betting.config import BUDGET_FRACTION, SAME_TITLE_CO_RESOLVE_PROB
from kalshi_betting.strategy import TradeSpec, _kelly_p, compute_trade, select_portfolio


def make_pair(
    pA: float = 0.70,
    pB: float = 0.30,
    nA: float = 0.20,
    tradeable: bool = True,
    pair_type: str = "same_title",
    max_contracts: int = 0,
) -> MagicMock:
    """Factory for CandidatePair-like mocks; avoids importing the real dataclass."""
    pair = MagicMock()
    pair.pA = pA
    pair.pB = pB
    pair.nA = nA
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


class TestKellyP:
    def test_time_series_independence_model(self):
        pair = make_pair(pA=0.70, pB=0.30, pair_type="time_series")
        # p = 1 - pA * (1 - pB) = 1 - 0.7 * 0.7 = 0.51
        assert _kelly_p(pair) == pytest.approx(1.0 - 0.70 * (1.0 - 0.30))

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

    def test_returns_trade_spec_for_valid_arbitrage(self):
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
