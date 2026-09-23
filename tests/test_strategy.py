"""Tests for strategy.py Kelly sizing and portfolio selection.

Time-series pairs buy YES on the earlier contract (market_a, at pA) and NO on
the later one (market_b, at nB); same-title pairs buy NO on market_a (nA) and
YES on market_b (pB). Every fixture therefore carries a REAL float nB — a
MagicMock auto-attribute would TypeError inside compute_trade's arithmetic.
"""
import ast
import inspect
import json
import logging
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kalshi_betting import backtester, config, dashboard, scanner, strategy
from kalshi_betting.config import (
    BUDGET_FRACTION,
    SAME_TITLE_CO_RESOLVE_PROB,
    fee_leg_exact,
    fee_per_pair_approx,
    max_affordable_pairs,
    time_series_profit_prob,
)
from kalshi_betting.scanner import (
    CandidatePair,
    enrich_with_orderbook_prices,
    leg_prices,
    leg_sides,
    prefix_fill_prices,
)
from kalshi_betting.strategy import TradeSpec, _kelly_p, compute_trade, select_portfolio

# Balance handed to enrich_with_orderbook_prices. Deliberately far larger than
# any fixture book: the affordability cap is min(book depth, what the budget
# buys), so an ample balance makes it never bind and these tests keep exercising
# the depth path alone. Tests that mean to exercise the cap set their own.
_AMPLE_BALANCE_CENTS = 100_000_000


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


def make_booked_pair(
    levels: list[tuple[float, float, float]],
    *,
    pair_type: str = "time_series",
    pB: float = 0.62,
    pA: float | None = None,
    nB: float | None = None,
    nA: float = 0.70,
    max_contracts: int | None = None,
) -> CandidatePair:
    """Build a REAL CandidatePair carrying order-book depth.

    A real dataclass rather than make_pair's MagicMock: compute_trade's marginal
    pricing reads depth_levels by type (strategy._depth_levels), so a mock's
    truthy auto-attribute is deliberately treated as "no book" and would exercise
    the wrong path entirely.

    levels are (price_a, price_b, qty) in MARKET order, ascending by combined
    price — the shape enrichment writes. The scalar leg prices default to the
    BEST level, and max_contracts to the full depth, which is what enrichment
    would have written for a balance large enough not to bind.
    """
    depth = tuple(levels)
    best_a, best_b, _ = depth[0]
    total = int(sum(q for _, _, q in depth))
    now = datetime.now(UTC)
    mA = SimpleNamespace(ticker="A1", title="A", close_time=now + timedelta(days=15),
                         exchange_index=0)
    mB = SimpleNamespace(ticker="B1", title="B", close_time=now + timedelta(days=30),
                         exchange_index=0)
    if pair_type == "time_series":
        pA_v, nB_v = (best_a if pA is None else pA), (best_b if nB is None else nB)
        nA_v, pB_v = nA, pB
    else:
        nA_v, pB_v = best_a, best_b
        pA_v, nB_v = (0.70 if pA is None else pA), (0.70 if nB is None else nB)
    return CandidatePair(
        market_a=mA, market_b=mB,
        pA=pA_v, pB=pB_v, nA=nA_v,
        tradeable=True,
        canonical_title="booked pair",
        pair_type=pair_type,
        nB=nB_v,
        max_contracts=total if max_contracts is None else max_contracts,
        depth_levels=depth,
    )


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
    must agree with.

    b divides by the dollars AT RISK, which include the fee: a losing pair loses
    total_cost_with_fees, not just the contracts' cost (DR-62). This is NOT the
    reported TradeSpec.profit_ratio, whose denominator is fee-less.
    """
    fee = fee_per_pair_approx(pA, nB)
    net_spread = (1.0 - pA - nB) - fee
    b = net_spread / (pA + nB + fee)
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
        # ~0.1620 — below the 20% cap, so Kelly (not the cap) sizes this pair.
        # (0.1884 under the pre-DR-62 fee-less Kelly denominator; the gate is
        # strictly tighter now, so every size is the same or smaller.)
        assert expected_f == pytest.approx(0.1620, abs=1e-4)
        assert expected_f < BUDGET_FRACTION
        assert result.kelly_fraction == pytest.approx(expected_f)

    def test_flow_through_dollar_figures(self):
        # Kelly budget 1620.11 → raw n 2314, shrunk by the fee loop to 2214:
        # cost 1549.80, exact fees 32.55 + 37.20 = 69.75, cash out 1619.55,
        # win-scenario profit 2214 * 0.30 - 69.75 = 594.45
        result = compute_trade(self._pair(), 1_000_000)
        assert result is not None
        assert result.x == 2214
        assert result.total_cost == pytest.approx(1549.80)
        assert result.total_cost_with_fees == pytest.approx(1619.55)
        assert result.min_payoff == pytest.approx(594.45)
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
        # 0.30 → 0.85 with NO ask 0.15: f* ≈ 0.216 → capped to BUDGET_FRACTION.
        # The gap had to widen from 0.40 to 0.55 with the fee-inclusive Kelly
        # denominator (DR-62): at the old 0.30 → 0.70 / 0.30 fixture f* is now
        # 0.1905, just under the cap, so the cap no longer binds there.
        assert _ts_kelly_fraction(0.30, 0.85, 0.15) > BUDGET_FRACTION
        result = compute_trade(self._pair(pB=0.85, nB=0.15), 1_000_000)
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


def _kelly_fraction_at(pair, price_a: float, price_b: float) -> float:
    """Uncapped Kelly fraction for a pair priced at (price_a, price_b).

    Mirrors compute_trade's formula so a test can state, in its own terms, what
    the old whole-book average would have concluded about a book. b carries the
    fee in its denominator, as compute_trade's kelly_b does (DR-62).
    """
    fee = fee_per_pair_approx(price_a, price_b)
    net_spread = (1.0 - price_a - price_b) - fee
    if net_spread <= 0:
        return -1.0
    b = net_spread / (price_a + price_b + fee)
    p = strategy._kelly_p_at(pair, price_a)
    return p - (1.0 - p) / b


class TestMarginalFillPricing:
    """compute_trade must price the contracts it is ACTUALLY buying.

    Enrichment bounds its average at the most the budget could ever buy; this is
    the second half — the size and the price are solved together, so the price
    the spec carries (and therefore the FoK limit the trader builds from it) is
    the average over exactly the contracts that will be submitted.
    """

    # Ascending by combined price: 0.70, 0.75, 0.82.
    LEVELS = [(0.30, 0.40, 20.0), (0.33, 0.42, 80.0), (0.37, 0.45, 400.0)]

    def test_spec_price_is_the_average_over_exactly_its_own_size(self):
        # The core invariant. Whatever n the descent lands on, leg_prices of the
        # returned pair is prefix_fill_prices at that same n — never at some
        # other quantity's average.
        for balance in (5_000, 25_000, 100_000, 1_000_000):
            pair = make_booked_pair(self.LEVELS)
            spec = compute_trade(pair, balance)
            if spec is None:
                continue
            expected = prefix_fill_prices(pair.depth_levels, spec.x)
            assert leg_prices(spec.pair) == pytest.approx(expected), balance

    def test_kelly_supports_the_size_it_settled_on(self):
        # The descent's stopping condition, checked from the outside: at the
        # price of n contracts, the capped Kelly budget still affords n.
        spec = compute_trade(make_booked_pair(self.LEVELS), 200_000)
        assert spec is not None
        price_sum = sum(leg_prices(spec.pair))
        assert max_affordable_pairs(200_000, price_sum, spec.kelly_fraction) >= spec.x

    def test_fee_inclusive_cost_still_fits_the_kelly_budget(self):
        # Re-pricing after the fee shrink must not push the trade back over
        # budget — a smaller n can only reach cheaper levels.
        spec = compute_trade(make_booked_pair(self.LEVELS), 200_000)
        assert spec is not None
        assert spec.total_cost_with_fees <= (200_000 / 100.0) * spec.kelly_fraction

    def test_never_sizes_past_the_depth_it_priced(self):
        for balance in (5_000, 50_000, 5_000_000):
            pair = make_booked_pair(self.LEVELS)
            spec = compute_trade(pair, balance)
            if spec is not None:
                assert spec.x <= pair.max_contracts

    def test_small_balance_pays_the_best_level(self):
        # $30 * 20% = $6.00 at 0.70 a pair -> 8 pairs, well inside the 20 resting
        # at the top rung, so the fill is the best level outright.
        spec = compute_trade(make_booked_pair(self.LEVELS), 3_000)
        assert spec is not None
        assert spec.x <= 20
        assert leg_prices(spec.pair) == pytest.approx((0.30, 0.40))

    def test_price_beats_the_whole_book_average(self):
        # The regression this change exists for: the old sizer priced every pair
        # at the average of the ENTIRE qualifying book, including depth no single
        # trade could reach. The solved price must be strictly better than that.
        pair = make_booked_pair(self.LEVELS)
        total = sum(q for _, _, q in pair.depth_levels)
        whole_book = (
            sum(a * q for a, _, q in pair.depth_levels) / total
            + sum(b * q for _, b, q in pair.depth_levels) / total
        )
        spec = compute_trade(pair, 100_000)
        assert spec is not None
        assert sum(leg_prices(spec.pair)) < whole_book

    def test_deep_book_no_longer_kills_a_pair_with_a_real_edge(self):
        # A thin band of genuine edge on top of a wall of near-worthless depth.
        # Averaged whole, the pair's Kelly fraction goes NEGATIVE and the old
        # sizer returned None — "a wide book drives Kelly negative and the pair
        # is skipped", on depth no trade could reach. Priced at what a real
        # trade would consume, the same pair sizes.
        levels = [(0.30, 0.40, 25.0), (0.42, 0.43, 100_000.0)]
        pair = make_booked_pair(levels)
        total = sum(q for _, _, q in levels)
        avg_a = sum(a * q for a, _, q in levels) / total
        avg_b = sum(b * q for _, b, q in levels) / total
        # Confirm the premise: at the whole-book average Kelly says don't bet
        whole_book_kelly = _kelly_fraction_at(pair, avg_a, avg_b)
        assert whole_book_kelly <= 0
        # ...while at the top of the book it is comfortably positive
        assert _kelly_fraction_at(pair, 0.30, 0.40) > 0
        spec = compute_trade(pair, 50_000)
        assert spec is not None
        assert spec.min_payoff > 0
        assert spec.kelly_fraction > 0
        # It may reach a little past the cheap band — what matters is that the
        # size it settled on is justified at its OWN price, not at the average
        # of a book it would never sweep.
        assert _kelly_fraction_at(pair, *leg_prices(spec.pair)) > 0
        assert sum(leg_prices(spec.pair)) < avg_a + avg_b

    def test_bookless_pair_sizes_exactly_as_before(self):
        # depth_levels=() is the no-book path, byte-identical to the pre-change
        # single-shot sizing. This is what keeps the backtester's Kelly-parity
        # test (which builds a bare pair on purpose) meaningful.
        booked = make_booked_pair([(0.30, 0.40, 1_000_000.0)])
        bare = make_pair(pA=0.30, nB=0.40, pB=0.62, nA=0.70, pair_type="time_series")
        bare_spec = compute_trade(bare, 100_000)
        booked_spec = compute_trade(booked, 100_000)
        assert bare_spec is not None and booked_spec is not None
        # One flat, effectively bottomless level prices identically at any size
        assert booked_spec.x == bare_spec.x
        assert booked_spec.total_cost == pytest.approx(bare_spec.total_cost)

    def test_bookless_pair_leaves_its_pair_object_untouched(self):
        # No book, nothing solved, so the spec carries the very same object —
        # the property main's display_specs mapping used to rely on everywhere.
        bare = make_pair(pair_type="same_title")
        spec = compute_trade(bare, 100_000)
        assert spec is not None
        assert spec.pair is bare

    def test_same_title_solves_on_its_own_leg_prices(self):
        # Same-title legs are (nA, pB), so the solved prices must land there —
        # writeback goes through leg_sides, never a hardcoded field name.
        levels = [(0.44, 0.31, 30.0), (0.47, 0.34, 500.0)]
        pair = make_booked_pair(levels, pair_type="same_title")
        spec = compute_trade(pair, 100_000)
        assert spec is not None
        assert leg_prices(spec.pair) == pytest.approx(
            prefix_fill_prices(pair.depth_levels, spec.x)
        )
        assert (spec.pair.nA, spec.pair.pB) == pytest.approx(leg_prices(spec.pair))
        # pA/nB are reporting-only for this pair type and must be left alone
        assert spec.pair.pA == pair.pA
        assert spec.pair.nB == pair.nB

    def test_descent_terminates_on_a_steeply_worsening_book(self):
        # Every rung materially worse than the last, so the descent has to walk
        # rather than settle on its first guess. It must still return.
        levels = [(0.30 + i * 0.002, 0.40 + i * 0.002, 5.0) for i in range(40)]
        spec = compute_trade(make_booked_pair(levels), 500_000)
        if spec is not None:
            assert leg_prices(spec.pair) == pytest.approx(
                prefix_fill_prices(tuple(levels), spec.x)
            )


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
        # (no spread), not the ~0.1620 the live sizer computes
        assert dashboard._kelly_fraction(_TS_PA, _TS_NA, _TS_PB, _TS_NB, "time_series") > 0.16

    def test_dashboard_same_title_unchanged(self):
        # Same-title still prices nA + pB on the fixed prior; nB is ignored
        fee = fee_per_pair_approx(0.20, 0.30)
        expected_b = ((1.0 - 0.20 - 0.30) - fee) / (0.50 + fee)
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
        # A two-link chain, same shape (and same intent) as the backtester pin
        # below: the priced call moved into _kelly_p_at when compute_trade
        # started re-deriving p at each candidate size, since pA is a LEG price
        # and therefore moves with the quantity being bought.
        assert _function_calls(strategy, "_kelly_p_at", "time_series_profit_prob")
        assert _function_calls(strategy, "_kelly_p", "_kelly_p_at")

    def test_ast_compute_trade_prices_through_kelly_helper(self):
        # compute_trade must reach the model through the same helper, never
        # reimplement 1 - k * (pB - pA) against its own per-size prices. The
        # chain runs through _evaluate_size, which is where every gate now
        # lives, and _solve_marginal_size, which searches over it.
        assert _function_calls(strategy, "_evaluate_size", "_kelly_p_at")
        assert _function_calls(strategy, "_solve_marginal_size", "_evaluate_size")
        assert _function_calls(strategy, "compute_trade", "_evaluate_size")
        assert _function_calls(strategy, "compute_trade", "_solve_marginal_size")

    def test_ast_dashboard_kelly_fraction_calls_helper(self):
        assert _function_calls(dashboard, "_kelly_fraction", "time_series_profit_prob")

    def test_ast_backtester_prices_through_helper(self):
        # Pass 1's k-dependent scoring moved into _simulate_at_discount when the
        # calibration sweep landed. The invariant is unchanged — the backtester
        # must price time-series through the shared config helper and never
        # reimplement the formula — so it is pinned as a two-link chain.
        assert _function_calls(backtester, "_simulate_at_discount", "time_series_profit_prob")
        assert _function_calls(backtester, "run_backtest", "_simulate_at_discount")

    def test_ast_both_finders_group_through_the_shared_key_helper(self):
        # DR-01: the live scanner and the backtester must derive the
        # time-series group key from ONE definition. They previously each
        # called normalize_title on their own combined title, so the backtester
        # reproduced the strike-blind key exactly and could never have
        # surfaced the defect on settled history.
        assert _function_calls(scanner, "find_time_series_pairs", "time_series_group_key")
        assert _function_calls(backtester, "_group_by_normalized_title", "time_series_group_key")

    def test_ast_both_finders_apply_the_one_series_rule(self):
        # DR-02/DR-54: identical wording across two events of ONE series is two
        # instances of a recurring fixture. Gating only the same-title finder
        # would merely RELABEL such a pair as a time-series bet — the same two
        # tickers also qualify there, and main._dedup_pairs only ever dropped
        # the time-series copy because a same-title copy existed. Both live
        # finders must therefore reach scanner._same_series, and the
        # backtester's _extract_pairs must reach its dict-world mirror on both
        # of its branches (one function, so one pin covers both).
        assert _function_calls(scanner, "find_same_title_pairs", "_same_series")
        assert _function_calls(scanner, "find_time_series_pairs", "_same_series")
        assert _function_calls(scanner, "find_time_series_pairs", "_identical_wording")
        assert _function_calls(backtester, "_extract_pairs", "_same_series_dicts")
        assert _function_calls(backtester, "_extract_pairs", "_identical_wording_dicts")

    def test_ast_both_finders_apply_the_cumulative_deadline_rule(self):
        # The time-series bet is only coherent between two CUMULATIVE-deadline
        # markets ("by <date>"), whose probabilities nest. Kalshi also lists
        # SNAPSHOT markets ("price ON <date>"), which do not nest at all, and
        # normalize_title strips a dated snapshot title just as readily as a
        # dated deadline one — so such a family lands in ONE group on BOTH
        # paths. Gating only the live finder would leave the backtester
        # replaying the defect instead of detecting it, exactly as it
        # reproduced DR-01 before the shared group key landed.
        assert _function_calls(scanner, "find_time_series_pairs", "cumulative_deadline_pair")
        assert _function_calls(backtester, "_extract_pairs", "cumulative_deadline_pair")

    def test_ast_the_deadline_classifier_has_one_definition(self):
        # Only the FIELD EXTRACTION differs between the two paths (attributes
        # live, dict keys in the backtest); the classification itself is one
        # function, so the two can never disagree about which pairs are
        # eligible. Same shape as the series-prefix pin below.
        assert _function_calls(scanner, "_market_deadline_profile", "deadline_profile")
        assert _function_calls(backtester, "_deadline_profile_dict", "deadline_profile")
        # DR-69: the field walk is _deciding_field, and BOTH the verdict
        # (deadline_phrasing) and the verdict-plus-spans (deadline_profile)
        # read it, so the spans can never come from a different field than
        # the verdict. A second copy of the walk in either would re-open
        # exactly that divergence.
        assert _function_calls(scanner, "deadline_profile", "_deciding_field")
        assert _function_calls(scanner, "deadline_phrasing", "_deciding_field")

    def test_ast_cumulative_deadline_pair_is_defined_through_refusal(self):
        # DR-72: cumulative_deadline_pair() is `deadline_pair_refusal(...) is
        # None`, so the boolean verdict and the refusal REASON both finders
        # log can never disagree — there is exactly one place the rule lives.
        assert _function_calls(scanner, "cumulative_deadline_pair", "deadline_pair_refusal")

    def test_ast_both_finders_name_the_refusal_reason(self):
        # Both finders keep deciding through cumulative_deadline_pair (the
        # pin above is unchanged); on refusal they ALSO call
        # deadline_pair_refusal directly to choose which of the three DR-72
        # skip counters to increment, rather than re-deriving the reason from
        # the profiles themselves or comparing string literals.
        assert _function_calls(scanner, "find_time_series_pairs", "deadline_pair_refusal")
        assert _function_calls(backtester, "_extract_pairs", "deadline_pair_refusal")

    def test_ast_the_live_finder_applies_the_same_event_ladder_rule(self):
        # DR-73: a same-event deadline ladder is ordered and tiered on its two
        # STATED deadlines, so the live finder must reach BOTH halves of that
        # arithmetic — stated_deadline (wording -> calendar day) and
        # same_event_ladder (two days -> leg order and gap). Re-deriving
        # either inside the finder is how the two paths drifted apart before
        # (DR-01), and close_time cannot answer either question for a ladder
        # whose rungs settle at one instant.
        #
        # ONE finder, unlike its both-paths siblings above, and that is the
        # staging rather than the rule: backtester._extract_pairs still
        # refuses every same-event candidate, so DR-73c renames this back to
        # test_ast_both_finders_apply_the_same_event_ladder_rule and adds
        #     assert _function_calls(backtester, "_extract_pairs", "stated_deadline")
        #     assert _function_calls(backtester, "_extract_pairs", "same_event_ladder")
        # Until it lands this pin covers the LIVE finder only, and
        # config.TIME_SERIES_SAME_EVENT_LADDERS must stay off (the two paths
        # would otherwise measure different strategies — CLAUDE.md's DR-73
        # gotcha records it).
        assert _function_calls(scanner, "find_time_series_pairs", "stated_deadline")
        assert _function_calls(scanner, "find_time_series_pairs", "same_event_ladder")

    def test_ast_pair_ceiling_reads_the_pair_gap(self):
        # DR-73: everything downstream of pair formation reads the gap through
        # pair_gap_days, which returns the STATED gap for a ladder and the
        # close_time gap for everything else. _pair_max_sum calling
        # deadline_gap_days itself would silently re-tier a ladder whose rungs
        # share a close_time — from the 30% tier to the 15% one.
        assert _function_calls(scanner, "_pair_max_sum", "pair_gap_days")
        assert not _function_calls(scanner, "_pair_max_sum", "deadline_gap_days")
        assert _function_calls(scanner, "enrich_with_orderbook_prices", "pair_gap_days")
        assert not _function_calls(
            scanner, "enrich_with_orderbook_prices", "deadline_gap_days")

    def test_ast_the_series_prefix_has_one_definition(self):
        # The backtester must not re-split the event ticker itself: the mirror
        # helpers call scanner.event_series, so live and backtest can never
        # disagree about what a series is.
        assert _function_calls(backtester, "_same_series_dicts", "event_series")
        assert _function_calls(scanner, "_same_series", "event_series")


# ── DR-62: Kelly's denominator is the dollars AT RISK, fee included ───────────

# The reproduction fixture: a time-series pair every scanner gate admits —
# pB - pA = 0.82 clears even the 30% long-gap tier and the leg sum 0.37 is far
# under the 0.85 price ceiling — whose TRUE expected value is negative once the
# losing cell's fee is counted. Under the pre-DR-62 fee-less Kelly denominator
# f* = +0.0113 and this pair was sized and submitted with real money.
_DR62_PA, _DR62_PB, _DR62_NB = 0.16, 0.98, 0.21


def _fee_less_kelly_fraction(pA: float, pB: float, nB: float) -> float:
    """The PRE-DR-62 time-series Kelly fraction: net_spread over the fee-LESS
    cost (price_a + price_b). Kept only so the tests below can state, in their
    own terms, what the old gate concluded — never what the code now does."""
    net_spread = (1.0 - pA - nB) - fee_per_pair_approx(pA, nB)
    p = time_series_profit_prob(pA, pB)
    return p - (1.0 - p) / (net_spread / (pA + nB))


def _fee_shrunk_n(kelly_f: float, price_a: float, price_b: float, balance: float) -> int:
    """compute_trade's budget-to-contracts step (capped Kelly budget, then the
    exact-fee shrink loop), so a test can size the SAME pair under a different
    Kelly fraction and compare the two counts."""
    budget = balance * min(BUDGET_FRACTION, kelly_f)
    n = int(budget / (price_a + price_b))
    fee_a, fee_b = fee_leg_exact(n, price_a), fee_leg_exact(n, price_b)
    while n > 0 and n * (price_a + price_b) + fee_a + fee_b > budget:
        n -= 1
        fee_a, fee_b = fee_leg_exact(n, price_a), fee_leg_exact(n, price_b)
    return n


def _backtester_trades(pA: float, pB: float, nA: float, nB: float) -> list:
    """Replay ONE time-series pair through backtester._simulate_at_discount and
    return the trades it entered (empty when its Kelly gate rejected the pair).

    Goes through the real Pass 1b/Pass 2 code rather than re-deriving the
    formula in the test, which is the whole point: the backtester replays the
    live admission rule, so a divergence here is a divergence nothing else in
    the suite would catch.
    """
    entry_date = date(2026, 1, 5)
    mA = {"ticker": "EA", "title": "A", "result": "yes",
          "close_time": "2026-02-01T00:00:00+00:00",
          "settlement_ts": "2026-02-14T00:00:00+00:00"}
    mB = {"ticker": "EB", "title": "B", "result": "yes",
          "close_time": "2026-02-14T00:00:00+00:00",
          "settlement_ts": "2026-02-14T00:00:00+00:00"}
    rec = {
        "pair_type": "time_series",
        "canon": "dr62 pair",
        "group_key": "dr62",
        "entry": {"mA": mA, "mB": mB, "pA": pA, "pB": pB, "nA": nA, "nB": nB,
                  "entry_date": entry_date, "gap_days": 13},
    }
    return backtester._simulate_at_discount([rec], entry_date, 10_000.0).trades


def _backtester_kelly_fraction(pA: float, pB: float, nA: float, nB: float) -> float:
    """The capped Kelly fraction the backtester puts on that one trade."""
    trades = _backtester_trades(pA, pB, nA, nB)
    assert len(trades) == 1
    return trades[0].kelly_fraction


class TestKellyRiskIncludesFees:
    """Kelly's "b" divides by the dollars actually AT RISK, and the fee is one
    of them: a losing pair loses total_cost_with_fees in full, so the fee-less
    denominator made f* > 0 whenever p*net_spread > q*(price_a + price_b) while
    true positive EV needs p*net_spread > q*(price_a + price_b + fee). The gate
    overstated EV by exactly q*fee on every pair (DR-62).

    What the fixed gate guarantees is positive EV under the CONTINUOUS fee
    approximation — nothing about the ceiling-rounded exact fee the trade is
    charged. fee_per_pair_approx sits below fee_leg_exact, so a spec on the
    boundary can still be EV-negative on its own fields at single-digit n; that
    residual is pinned by test_small_n_can_still_be_ev_negative_on_exact_fees
    rather than papered over."""

    def test_the_headline_fixture_is_rejected(self):
        # THE pin. Accepted before DR-62, rejected now.
        assert _fee_less_kelly_fraction(_DR62_PA, _DR62_PB, _DR62_NB) > 0
        assert _ts_kelly_fraction(_DR62_PA, _DR62_PB, _DR62_NB) < 0
        pair = make_pair(pA=_DR62_PA, pB=_DR62_PB, nA=0.85, nB=_DR62_NB,
                         pair_type="time_series")
        assert compute_trade(pair, 1_000_000) is None

    def test_the_headline_fixtures_true_ev_is_negative(self):
        # Stated in EV terms rather than in Kelly terms, so the two readings of
        # this pair are side by side: what the old gate implicitly modelled,
        # and what the module's own settlement model says.
        fee = fee_per_pair_approx(_DR62_PA, _DR62_NB)
        net_spread = (1.0 - _DR62_PA - _DR62_NB) - fee
        p = time_series_profit_prob(_DR62_PA, _DR62_PB)
        q = 1.0 - p
        ev_fee_less = p * net_spread - q * (_DR62_PA + _DR62_NB)
        ev_true = p * net_spread - q * (_DR62_PA + _DR62_NB + fee)
        assert ev_fee_less > 0
        assert ev_true < 0
        # The gap between the two readings is exactly q * fee, always
        assert ev_fee_less - ev_true == pytest.approx(q * fee)

    def test_the_overstatement_is_q_times_the_fee_for_any_pair(self):
        # Not a property of the fixture: the two EV expressions differ by q*fee
        # for every price pair, which is why same-title (q = 0.05) is immune
        # and the time-series bet (q = k*(pB - pA)) is not.
        for price_a, price_b, p in ((0.30, 0.40, 0.775), (0.20, 0.30, 0.95),
                                    (0.16, 0.21, 0.385)):
            fee = fee_per_pair_approx(price_a, price_b)
            net_spread = (1.0 - price_a - price_b) - fee
            q = 1.0 - p
            ev_fee_less = p * net_spread - q * (price_a + price_b)
            ev_true = p * net_spread - q * (price_a + price_b + fee)
            assert ev_fee_less - ev_true == pytest.approx(q * fee)

    def test_a_profitable_pair_is_still_accepted_and_sizes_no_larger(self):
        # The gate is strictly TIGHTER, never looser: the same pair still
        # trades, at the same count or a smaller one.
        pair = make_pair(pA=_TS_PA, pB=_TS_PB, nA=_TS_NA, nB=_TS_NB,
                         pair_type="time_series")
        spec = compute_trade(pair, 1_000_000)
        assert spec is not None
        fee_less_f = _fee_less_kelly_fraction(_TS_PA, _TS_PB, _TS_NB)
        assert spec.kelly_fraction < fee_less_f
        old_n = _fee_shrunk_n(fee_less_f, _TS_PA, _TS_NB, 10_000.0)
        assert 0 < spec.x <= old_n

    def test_same_title_verdict_is_unchanged_on_the_reference_pair(self):
        # q is the fixed 1 - SAME_TITLE_CO_RESOLVE_PROB = 0.05, so q*fee is
        # small: the fraction barely moves and the accept/reject verdict is
        # identical on both a profitable and an unprofitable same-title pair.
        # NOT a universal — the name scopes it to this fixture on purpose.
        # Measured over the same-title admissible region on a whole-cent grid,
        # 12 of 4,465 price points DO flip (0.27%), always accept -> reject
        # (e.g. nA 0.28 / pB 0.64: +0.0256 fee-less, -0.0048 fee-inclusive);
        # test_a_flipping_same_title_pair_is_now_rejected pins one of them.
        nA, pB = 0.20, 0.30
        fee = fee_per_pair_approx(nA, pB)
        net_spread = (1.0 - nA - pB) - fee
        q = 1.0 - SAME_TITLE_CO_RESOLVE_PROB
        fee_less = SAME_TITLE_CO_RESOLVE_PROB - q / (net_spread / (nA + pB))
        fee_incl = SAME_TITLE_CO_RESOLVE_PROB - q / (net_spread / (nA + pB + fee))
        assert (fee_less > 0) == (fee_incl > 0)
        assert fee_incl == pytest.approx(fee_less, abs=0.01)
        spec = compute_trade(make_pair(nA=nA, pB=pB, pair_type="same_title"), 1_000_000)
        assert spec is not None
        # Both readings are far above the cap, so the sizing is byte-identical
        assert spec.kelly_fraction == pytest.approx(BUDGET_FRACTION)
        # ...and a same-title pair with no spread is still rejected either way
        assert compute_trade(make_pair(nA=0.60, pB=0.50, pair_type="same_title"),
                             1_000_000) is None

    def test_a_flipping_same_title_pair_is_now_rejected(self):
        # "Effectively immune" is not "immune": a thin minority of same-title
        # price points inside the admissible region (pA - pB >= 0.05, legs
        # <= 0.95) do change verdict, and the flip is always accept -> reject.
        # Pinned so the immunity claim is never read as a universal.
        nA, pB = 0.28, 0.64
        fee = fee_per_pair_approx(nA, pB)
        net_spread = (1.0 - nA - pB) - fee
        q = 1.0 - SAME_TITLE_CO_RESOLVE_PROB
        fee_less = SAME_TITLE_CO_RESOLVE_PROB - q / (net_spread / (nA + pB))
        fee_incl = SAME_TITLE_CO_RESOLVE_PROB - q / (net_spread / (nA + pB + fee))
        assert fee_less > 0 > fee_incl
        assert compute_trade(make_pair(nA=nA, pB=pB, pair_type="same_title"),
                             1_000_000) is None

    @pytest.mark.parametrize("kwargs", [
        {"pA": _TS_PA, "pB": _TS_PB, "nA": _TS_NA, "nB": _TS_NB,
         "pair_type": "time_series"},
        {"nA": 0.20, "pB": 0.30, "pair_type": "same_title"},
    ])
    def test_accepted_specs_have_positive_ev_on_their_own_fields(self, kwargs):
        # A spot-check on THESE fixtures, not a property of the gate. The gate
        # prices with fee_per_pair_approx, which UNDERESTIMATES the
        # ceiling-rounded fee_leg_exact that min_payoff and total_cost_with_fees
        # are built from — so the identity only holds at sizes where the two
        # converge. Both fixtures size in the thousands, where the rounding is
        # negligible; test_small_n_can_still_be_ev_negative_on_exact_fees below
        # pins the residual at the other end so it is documented rather than
        # rediscovered.
        spec = compute_trade(make_pair(**kwargs), 1_000_000)
        assert spec is not None
        assert spec.x > 100  # the regime this identity is asserted for
        q = 1.0 - spec.kelly_p
        assert spec.kelly_p * spec.min_payoff - q * spec.total_cost_with_fees > 0

    def test_small_n_can_still_be_ev_negative_on_exact_fees(self):
        # The residual the gate does NOT close, pinned so nobody restates the
        # guarantee as "positive EV on the spec's own fields". Accepted at
        # n = 30, yet p*min_payoff - q*total_cost_with_fees is a half-cent
        # NEGATIVE, entirely because the exact per-leg fee is ceiling-rounded
        # above fee_per_pair_approx. compute_trade's min_payoff > 0 check bounds
        # this regime; it does not eliminate it.
        pair = make_pair(pA=0.12, pB=0.33, nA=0.88, nB=0.70,
                         pair_type="time_series")
        spec = compute_trade(pair, 1_000_000)
        assert spec is not None
        assert spec.x == 30
        assert spec.min_payoff > 0  # the backstop that bounds the shortfall
        q = 1.0 - spec.kelly_p
        ev = spec.kelly_p * spec.min_payoff - q * spec.total_cost_with_fees
        assert ev == pytest.approx(-0.005, abs=1e-3)
        # ...and the approximation the gate priced with says the opposite
        price_a, price_b = leg_prices(spec.pair)
        approx_fee = fee_per_pair_approx(price_a, price_b) * spec.x
        assert approx_fee < spec.total_cost_with_fees - spec.total_cost

    def test_all_three_sizers_agree_by_value(self):
        # CLAUDE.md's AST pins cover WHICH helper each sizer calls, never the
        # arithmetic around it — so the fee-inclusive denominator is pinned here
        # by value, across strategy, dashboard and the backtester.
        expected = _ts_kelly_fraction(_TS_PA, _TS_PB, _TS_NB)
        live = compute_trade(
            make_pair(pA=_TS_PA, pB=_TS_PB, nA=_TS_NA, nB=_TS_NB,
                      pair_type="time_series"),
            1_000_000,
        )
        assert live is not None
        assert live.kelly_fraction == pytest.approx(expected)
        assert dashboard._kelly_fraction(
            _TS_PA, _TS_NA, _TS_PB, _TS_NB, "time_series") == pytest.approx(expected)
        assert _backtester_kelly_fraction(
            _TS_PA, _TS_PB, _TS_NA, _TS_NB) == pytest.approx(expected)
        # None of the three is the pre-DR-62 value
        assert expected < _fee_less_kelly_fraction(_TS_PA, _TS_PB, _TS_NB)

    def test_the_backtester_also_rejects_the_headline_fixture(self):
        # The backtest replays the live admission rule, which is why no
        # backtest could ever have surfaced this — so the mirror is pinned on
        # the rejection too, not only on the value.
        assert _backtester_trades(_DR62_PA, _DR62_PB, 0.85, _DR62_NB) == []
        # Sanity: the same harness DOES enter the profitable fixture, so the
        # emptiness above is the Kelly gate and not a broken fixture.
        assert len(_backtester_trades(_TS_PA, _TS_PB, _TS_NA, _TS_NB)) == 1

    def test_reported_profit_ratio_keeps_the_fee_less_denominator(self):
        # The split is deliberate: Kelly's b is the fee-INCLUSIVE risk, while
        # TradeSpec.profit_ratio stays return-on-contract-cost so
        # monthly_profit_ratio (select_portfolio's ranking key) and the prod
        # log's Profit Ratio column keep their published meaning.
        spec = compute_trade(
            make_pair(pA=_TS_PA, pB=_TS_PB, nA=_TS_NA, nB=_TS_NB,
                      pair_type="time_series"),
            1_000_000,
        )
        assert spec is not None
        price_a, price_b = leg_prices(spec.pair)
        fee = fee_per_pair_approx(price_a, price_b)
        net_spread = (1.0 - price_a - price_b) - fee
        assert spec.profit_ratio == pytest.approx(net_spread / (price_a + price_b))
        assert spec.profit_ratio != pytest.approx(net_spread / (price_a + price_b + fee))


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


class TestKellyOperandsShareOneSnapshot:
    """_kelly_p's two time-series operands (pair.pA and pair.pB) must both come
    from the enrichment snapshot.

    config.time_series_profit_prob clamps the gap at zero, so a pB left at its
    scan-time value while pA is refreshed to a depth-weighted fill can return
    p = 1.0 — a riskless model on what is a directional bet, which Kelly then
    sizes at the BUDGET_FRACTION cap. scanner.enrich_with_orderbook_prices
    refreshes pB from the same books and drops any pair whose reference is not
    above the YES fill, so no pair it marks tradeable can reach the clamp
    (TS-34)."""

    @staticmethod
    def _books(*, pA_fill: float, nB_fill: float, pB_ref: float, qty: int = 100):
        """Mock KalshiClient serving EARLY/LATE time-series books, one level each.

        EARLY rests a NO bid of (1 - pA_fill) so its YES ask — the YES leg's
        fill — is pA_fill. LATE rests a YES bid of (1 - nB_fill) so its NO ask
        — the NO leg's fill — is nB_fill, and a NO bid of (1 - pB_ref) so its
        YES ask (the reference quote) is pB_ref. Books arrive in the raw
        orderbook_fp wire format, which is what _fetch_orderbook parses.
        """
        def fake_orderbook(ticker):
            if ticker == "EARLY":
                ob = {"yes_dollars": [],
                      "no_dollars": [[str(round(1.0 - pA_fill, 4)), str(qty)]]}
            else:
                ob = {"yes_dollars": [[str(round(1.0 - nB_fill, 4)), str(qty)]],
                      "no_dollars": [[str(round(1.0 - pB_ref, 4)), str(qty)]]}
            payload = json.dumps({"orderbook_fp": ob}).encode("utf-8")
            return SimpleNamespace(status=200, data=payload)

        client = MagicMock()
        client.get_market_orderbook_without_preload_content = MagicMock(
            side_effect=fake_orderbook
        )
        return client

    @staticmethod
    def _pair(*, pA: float, pB: float, nB: float, gap_days: int = 10):
        """A real CandidatePair (a dataclass, as dc_replace in enrichment needs)."""
        early_close = datetime(2026, 3, 1, tzinfo=UTC)
        mA = SimpleNamespace(ticker="EARLY", close_time=early_close)
        mB = SimpleNamespace(
            ticker="LATE", close_time=early_close + timedelta(days=gap_days),
        )
        return CandidatePair(
            market_a=mA, market_b=mB,
            pA=pA, pB=pB, nA=round(1.0 - pA, 4),
            tradeable=True,
            canonical_title="will btc exceed $80k",
            pair_type="time_series",
            nB=nB,
        )

    def test_kelly_p_operands_come_from_one_snapshot(self):
        # LATE's book is CROSSED (YES bid 0.70 against a NO bid of 0.55), the
        # only shape that can invert a pair once the reference is refreshed:
        # the fills are pA 0.54 + nB 0.30 = 0.84, inside the 10-day ceiling of
        # 0.85 and profitable after fees, while LATE's fresh YES ask is 0.45.
        pair = self._pair(pA=0.30, pB=0.50, nB=0.30)
        client = self._books(pA_fill=0.54, nB_fill=0.30, pB_ref=0.45)

        # Enrichment is the producer of every pair _kelly_p ever prices
        [enriched] = enrich_with_orderbook_prices(client, [pair], _AMPLE_BALANCE_CENTS)

        # The invariant: a tradeable time-series pair still runs in the
        # direction it qualified in, so the max(0, pB - pA) clamp is
        # unreachable; an inverted one is dropped before compute_trade sizes it
        if enriched.tradeable:
            assert enriched.pB > enriched.pA
            assert _kelly_p(enriched) < 1.0
        else:
            assert compute_trade(enriched, 100_000) is None

        # This fixture is the inverted one, so it must take the dropped branch
        assert enriched.tradeable is False

    def test_uninverted_pair_keeps_a_real_loss_probability(self):
        # The control: an UNCROSSED LATE book (YES bid 0.50, NO bid 0.35)
        # leaves the refreshed reference 0.65 above the 0.30 fill, so the pair
        # survives and is priced on a genuine, non-clamped gap
        pair = self._pair(pA=0.30, pB=0.60, nB=0.50)
        client = self._books(pA_fill=0.30, nB_fill=0.50, pB_ref=0.65)
        [enriched] = enrich_with_orderbook_prices(client, [pair], _AMPLE_BALANCE_CENTS)
        assert enriched.tradeable is True
        assert enriched.pB > enriched.pA
        assert _kelly_p(enriched) == pytest.approx(
            time_series_profit_prob(enriched.pA, enriched.pB)
        )
        assert _kelly_p(enriched) < 1.0


# ---------------------------------------------------------------------------
# TS-08: the sized count must be depth the V2 FoK limit can actually reach.
# ---------------------------------------------------------------------------
class TestReachableDepthSizing:
    """
    A V2 taker order is ONE fill-or-kill limit per leg, so it buys only the
    depth resting at or below that limit — and the limit is a PER-CONTRACT cap
    derived from the leg price, which is the average of a MULTI-LEVEL prefix.
    An average over a ladder sits below the ladder's top level, so the top of
    the very prefix being priced could rest above its own cap and the order was
    killed, reported as "NO leg FoK not filled" and indistinguishable from a
    genuine price move (TS-08).

    Fixtures are same_title, so market_a buys NO and market_b buys YES, and
    depth_levels are (market_a price, market_b price, qty) in MARKET order.
    """

    @staticmethod
    def _reach(spec, levels):
        """
        Contracts the spec's own submitted order actually reaches.

        Derived from trader._v2_limit_price — the function that builds the real
        order body — NOT from strategy's own helper. The whole finding is that
        the sizer and the order disagreed, so the oracle here has to be the
        order side, independently of whatever the sizer believes.
        """
        from kalshi_betting import trader
        side_a, side_b = leg_sides(spec.pair.pair_type)
        price_a, price_b = leg_prices(spec.pair)

        def cap(kind, price, market):
            wire = trader._v2_limit_price(f"buy_{kind}", price, market)
            return float(1 - wire) if kind == "no" else float(wire)

        cap_a = cap(side_a, price_a, spec.pair.market_a)
        cap_b = cap(side_b, price_b, spec.pair.market_b)
        return sum(q for pa, pb, q in levels if pa <= cap_a + 1e-9 and pb <= cap_b + 1e-9)

    def test_no_leg_ladder_is_sized_to_reachable_depth(self):
        # Priced over all 600 the NO leg averages 0.345, whose cap is 0.36 —
        # which cannot reach the 0.37 level. Pre-fix this sized 600 and the
        # fill-or-kill died with nothing to unwind (the NO leg goes first).
        levels = [(0.32, 0.30, 300.0), (0.37, 0.30, 300.0)]
        pair = make_booked_pair(levels, pair_type="same_title")
        spec = compute_trade(pair, _AMPLE_BALANCE_CENTS)
        assert spec.x == 300
        assert leg_prices(spec.pair)[0] == pytest.approx(0.32)
        assert self._reach(spec, levels) >= spec.x

    def test_yes_leg_ladder_is_sized_to_reachable_depth(self):
        # The SAME defect with the ladder on the YES leg, which is the
        # expensive half: the NO leg is submitted first and FILLS, then the YES
        # leg is killed, so the pair unwinds. A rollback, not a skipped trade.
        levels = [(0.32, 0.30, 300.0), (0.32, 0.35, 300.0)]
        pair = make_booked_pair(levels, pair_type="same_title")
        spec = compute_trade(pair, _AMPLE_BALANCE_CENTS)
        assert spec.x == 300
        assert leg_prices(spec.pair)[1] == pytest.approx(0.30)
        assert self._reach(spec, levels) >= spec.x

    def test_flat_book_is_unchanged(self):
        # GUARD: one level means the prefix average IS that level's price, so
        # the cap always covers it. The gate must be a no-op here.
        levels = [(0.32, 0.30, 600.0)]
        pair = make_booked_pair(levels, pair_type="same_title")
        spec = compute_trade(pair, _AMPLE_BALANCE_CENTS)
        assert spec.x == 600

    def test_small_balance_path_is_unchanged(self):
        # GUARD: when the budget cannot size past the cheapest level the
        # prefix price and the cap already agree, so #51's affordability bound
        # masks TS-08 entirely. That is why a thin-balance dry run shows no
        # delta — the fix must not move this case either.
        levels = [(0.32, 0.30, 300.0), (0.37, 0.30, 300.0)]
        pair = make_booked_pair(levels, pair_type="same_title", max_contracts=161)
        spec = compute_trade(pair, 500_00)
        assert spec.x == 153
        assert self._reach(spec, levels) >= spec.x

    def test_the_supported_set_has_a_hole_and_the_search_lands_below_it(self):
        # Reachability is NOT downward-closed, which is why the post-shrink
        # backstop exists. On this book n<=1000 is supported, 1001..1500 is
        # not (the prefix average ceils to 0.31, capping at 0.32, which cannot
        # reach the 0.33 level), and 1501..1600 is supported again (the average
        # passes 0.31, capping at 0.33). _solve_marginal_size bisects, so it
        # converges to the top of the LOWER island and never lands above the
        # hole — which is precisely why the backstop has never been observed
        # to fire on a real book.
        levels = [(0.30, 0.30, 1000.0), (0.33, 0.30, 600.0)]
        pair = make_booked_pair(levels, pair_type="same_title")
        supported = {
            n: strategy._reachable_contracts(
                pair, tuple(levels), *prefix_fill_prices(tuple(levels), n)
            ) >= n
            for n in (1000, 1001, 1500, 1501, 1600)
        }
        assert supported == {1000: True, 1001: False, 1500: False,
                             1501: True, 1600: True}
        spec = compute_trade(pair, 5_000_000)
        assert spec.x == 1000
        assert self._reach(spec, levels) >= spec.x

    def test_post_shrink_backstop_pulls_a_stranded_count_back(self, monkeypatch):
        # Drive the backstop directly. No real book reaches it (see the test
        # above), so the only honest way to exercise it is to hand compute_trade
        # a solved size inside the hole's upper island together with a budget
        # the fees overrun — exactly the shape the shrink loop would produce if
        # the search ever landed there.
        levels = [(0.30, 0.30, 1000.0), (0.33, 0.30, 600.0)]
        pair = make_booked_pair(levels, pair_type="same_title")
        price_a, price_b = prefix_fill_prices(tuple(levels), 1600)
        forced = strategy._Sizing(
            n=1600, target=1600, price_a=price_a, price_b=price_b,
            p=SAME_TITLE_CO_RESOLVE_PROB, profit_ratio=0.05,
            kelly_fraction=BUDGET_FRACTION, budget_dollars=940.0,
        )
        monkeypatch.setattr(strategy, "_solve_marginal_size",
                            lambda *a, **k: forced)
        spec = compute_trade(pair, _AMPLE_BALANCE_CENTS)
        # The fee shrink alone would land in [1001, 1500] — unreachable. The
        # backstop snaps to the reachable prefix instead.
        assert spec.x == 1000
        assert self._reach(spec, levels) >= spec.x

    def test_reachability_holds_for_every_ladder_and_balance(self):
        # The invariant, swept: whatever compute_trade returns on a booked
        # pair, the order it implies must be able to buy that many contracts.
        ladders = [
            [(0.30, 0.30, 100.0), (0.33, 0.30, 100.0)],
            [(0.30, 0.30, 1000.0), (0.33, 0.30, 600.0)],
            [(0.32, 0.30, 300.0), (0.37, 0.30, 300.0)],
            [(0.30, 0.30, 50.0), (0.31, 0.30, 50.0), (0.36, 0.30, 400.0)],
            [(0.25, 0.30, 10.0), (0.26, 0.31, 20.0), (0.27, 0.34, 40.0)],
        ]
        for levels in ladders:
            for balance in (20_000, 500_00, 100_000_00, 1_000_000_000):
                pair = make_booked_pair(levels, pair_type="same_title")
                spec = compute_trade(pair, balance)
                if spec is None:
                    continue
                assert self._reach(spec, levels) >= spec.x, (levels, balance, spec.x)

    def test_legacy_path_keeps_the_whole_ladder(self, monkeypatch):
        # The legacy cap is buy_max_cost, a TOTAL-cost cap that CAN sweep a
        # ladder, so narrowing to reachable depth there would shrink sizes for
        # no reason. Gated, not unconditional.
        monkeypatch.setattr(strategy, "ORDER_API_VERSION", "legacy")
        levels = [(0.32, 0.30, 300.0), (0.37, 0.30, 300.0)]
        pair = make_booked_pair(levels, pair_type="same_title")
        spec = compute_trade(pair, _AMPLE_BALANCE_CENTS)
        assert spec.x == 600


class TestPortfolioSummaryIsFeeInclusive:
    """
    TS-12, found by a live prod dry run rather than by the static enumeration:
    select_portfolio BUDGETS against total_cost_with_fees but its summary line
    summed total_cost, so the headline portfolio figure was the one cost on the
    page that was not the cash being committed. Measured live: $60.47 reported
    against $64.39 of per-trade costs and a $52.08 collateral transfer.
    """

    def test_summary_sums_the_figure_the_loop_budgets_against(self, caplog):
        specs = [
            make_spec(pair_type="same_title", total_cost=10.0,
                      total_cost_with_fees=10.70),
            make_spec(pair_type="same_title", total_cost=20.0,
                      total_cost_with_fees=21.40),
        ]
        with caplog.at_level(logging.INFO):
            select_portfolio(specs, 100_000)
        line = next(r.getMessage() for r in caplog.records
                    if "Portfolio:" in r.getMessage())
        assert "$32.10" in line
        assert "$30.00" not in line
        assert "incl. fees" in line
