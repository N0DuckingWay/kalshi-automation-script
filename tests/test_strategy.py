"""Tests for strategy.py Kelly sizing and portfolio selection.

Time-series pairs buy YES on the earlier contract (market_a, at pA) and NO on
the later one (market_b, at nB); same-title pairs buy NO on market_a (nA) and
YES on market_b (pB). Every fixture therefore carries a REAL float nB — a
MagicMock auto-attribute would TypeError inside compute_trade's arithmetic.
"""
import ast
import dataclasses
import inspect
import json
import logging
import random
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kalshi_betting import backtester, config, dashboard, main, scanner, strategy
from kalshi_betting.config import (
    SAME_TITLE_CO_RESOLVE_PROB,
    LiveSettings,
    fee_leg_exact,
    fee_per_pair_approx,
    live_settings,
    max_affordable_pairs,
    time_series_profit_prob,
)
from kalshi_betting.scanner import (
    CandidatePair,
    HeldPair,
    enrich_with_orderbook_prices,
    leg_prices,
    leg_sides,
    pair_held,
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


@pytest.mark.usefixtures("pre_toggle_defaults")
class TestKellyP:
    """Probability-of-profit models: the discounted market gap for time-series
    (config.time_series_profit_prob) and the fixed co-resolution prior for
    same-title, at k 0.75 (pre_toggle_defaults)."""

    def test_time_series_discounted_gap_model(self):
        pair = make_pair(pA=0.30, pB=0.60, nB=0.40, pair_type="time_series")
        # p = 1 - k * (pB - pA) = 1 - 0.75 * 0.30 = 0.775
        assert _kelly_p(pair, live_settings()) == pytest.approx(0.775)
        assert _kelly_p(pair, live_settings()) == pytest.approx(
            time_series_profit_prob(0.30, 0.60))

    def test_time_series_model_is_not_the_old_expression(self):
        # The pre-2026-09 formula 1 - pA*(1-pB) modelled the impossible
        # A=YES/B=NO cell; on this fixture it would read 0.88, not 0.775
        pair = make_pair(pA=0.30, pB=0.60, nB=0.40, pair_type="time_series")
        assert _kelly_p(pair, live_settings()) != pytest.approx(1.0 - 0.30 * (1.0 - 0.60))

    def test_same_title_fixed_prior(self):
        pair = make_pair(pA=0.80, pB=0.40, pair_type="same_title")
        assert _kelly_p(pair, live_settings()) == SAME_TITLE_CO_RESOLVE_PROB

    def test_same_title_ignores_prices(self):
        pair_high = make_pair(pA=0.90, pB=0.80, pair_type="same_title")
        pair_low = make_pair(pA=0.10, pB=0.05, pair_type="same_title")
        assert _kelly_p(pair_high, live_settings()) == _kelly_p(pair_low, live_settings())


@pytest.mark.usefixtures("pre_toggle_defaults")
class TestComputeTrade:
    """compute_trade's gates and caps under pre_toggle_defaults (k 0.75, 20% cap)."""

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
        assert result.kelly_fraction <= config.BUDGET_FRACTION + 1e-9

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


@pytest.mark.usefixtures("pre_toggle_defaults")
class TestComputeTradeTimeSeries:
    """The plan's flow-through fixture through compute_trade: YES on the
    earlier contract at 0.30 and NO on the later at 0.40 (later YES ask 0.60),
    $10,000 balance. Every expectation is derived from the config helpers in
    the test, not hardcoded, except the contract count and the dollar figures
    the plan pins, all worked under pre_toggle_defaults (k 0.75, a 20% cap)."""

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
        assert expected_f < config.BUDGET_FRACTION
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
        assert _ts_kelly_fraction(0.30, 0.85, 0.15) > config.BUDGET_FRACTION
        result = compute_trade(self._pair(pB=0.85, nB=0.15), 1_000_000)
        assert result is not None
        assert result.kelly_fraction == pytest.approx(config.BUDGET_FRACTION)

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


def _live(k: float = 0.75, cap: float = 0.20, st_cap: float = 1.0) -> LiveSettings:
    """A LiveSettings at the given k and caps, tier floors on and no band."""
    return LiveSettings(tier_floors=True, spread_band=(0.0, 1.0), interval_discount=k,
                        size_cap=cap, same_title_size_cap=st_cap)


@pytest.mark.usefixtures("pre_toggle_defaults")
class TestComputeTradeSettings:
    """compute_trade reads k and both caps from ONE LiveSettings, hands it to
    every internal helper, and resolves config.py's only when handed none.
    "The default" is pre_toggle_defaults' (k 0.75, a 20% cap)."""

    _ST = {"nA": 0.20, "pB": 0.30, "pair_type": "same_title"}
    _TS_WIDE = {"pA": 0.30, "pB": 0.85, "nA": 0.70, "nB": 0.15, "pair_type": "time_series"}
    _TS = {"pA": _TS_PA, "pB": _TS_PB, "nA": _TS_NA, "nB": _TS_NB, "pair_type": "time_series"}

    @staticmethod
    def _st_f() -> float:
        fee = fee_per_pair_approx(0.20, 0.30)
        net = 0.50 - fee
        return SAME_TITLE_CO_RESOLVE_PROB - (1 - SAME_TITLE_CO_RESOLVE_PROB) * (0.50 + fee) / net

    def test_the_reference_pairs(self):
        assert self._st_f() == pytest.approx(0.8945, abs=1e-4)
        assert _ts_kelly_fraction(0.30, 0.85, 0.15) == pytest.approx(0.2163, abs=1e-4)

    def test_an_explicit_cap_binds_where_the_default_would(self):
        st = make_pair(**self._ST)
        default = compute_trade(st, 1_000_000)
        assert default.kelly_fraction == pytest.approx(config.BUDGET_FRACTION)
        wider = compute_trade(st, 1_000_000, settings=_live(cap=0.35))
        assert wider.kelly_fraction == pytest.approx(0.35)
        assert wider.x > default.x
        # The time-series f* (~0.216) sits between the caps: 0.20 binds, 0.35 does not
        ts = make_pair(**self._TS_WIDE)
        assert compute_trade(ts, 1_000_000, settings=_live(cap=0.20)).kelly_fraction == \
            pytest.approx(0.20)
        assert compute_trade(ts, 1_000_000, settings=_live(cap=0.35)).kelly_fraction == \
            pytest.approx(_ts_kelly_fraction(0.30, 0.85, 0.15))
        # ... on the booked path too (_solve_marginal_size -> _evaluate_size)
        booked = make_booked_pair([(0.20, 0.30, 1_000_000.0)], pair_type="same_title")
        assert compute_trade(booked, 1_000_000, settings=_live(cap=0.35)).kelly_fraction == \
            pytest.approx(0.35)

    def test_k_changes_p(self):
        pair = make_pair(**self._TS)
        base = compute_trade(pair, 1_000_000, settings=_live())
        at_06 = compute_trade(pair, 1_000_000, settings=_live(k=0.6))
        assert base.kelly_p == pytest.approx(0.775)
        assert at_06.kelly_p == pytest.approx(1.0 - 0.6 * (_TS_PB - _TS_PA))
        assert at_06.kelly_p == pytest.approx(time_series_profit_prob(_TS_PA, _TS_PB, k=0.6))
        assert at_06.kelly_fraction > base.kelly_fraction
        # The booked path prices at the run's k too (one flat level: one price at every n)
        booked = make_booked_pair([(_TS_PA, _TS_NB, 1_000_000.0)], pB=_TS_PB)
        spec = compute_trade(booked, 1_000_000, settings=_live(k=0.6))
        assert spec is not None
        assert spec.kelly_p == pytest.approx(at_06.kelly_p)
        # _kelly_p reads the same k
        assert strategy._kelly_p(pair, _live(k=0.6)) == pytest.approx(at_06.kelly_p)

    def test_the_same_title_cap_binds_same_title_only(self):
        st, ts = make_pair(**self._ST), make_pair(**self._TS_WIDE)
        s = _live(cap=1.0, st_cap=0.20)
        assert compute_trade(st, 1_000_000, settings=s).kelly_fraction == pytest.approx(0.20)
        # With no same-title cap the pair sizes at its full f*
        uncapped = compute_trade(st, 1_000_000, settings=_live(cap=1.0))
        assert uncapped.kelly_fraction == pytest.approx(self._st_f())
        # The same-title cap never reaches a time-series pair
        assert compute_trade(ts, 1_000_000, settings=s).kelly_fraction == pytest.approx(
            _ts_kelly_fraction(0.30, 0.85, 0.15))
        # The tighter of the two caps binds a same-title pair
        tight = compute_trade(st, 1_000_000, settings=_live(cap=0.10, st_cap=0.20))
        assert tight.kelly_fraction == pytest.approx(0.10)
        # ... on the booked path too
        booked = make_booked_pair([(0.20, 0.30, 1_000_000.0)], pair_type="same_title")
        assert compute_trade(booked, 1_000_000, settings=s).kelly_fraction == \
            pytest.approx(0.20)

    def test_none_is_config_pys_settings_read_at_call_time(self, monkeypatch):
        for kwargs in (self._ST, self._TS_WIDE, self._TS):
            pair = make_pair(**kwargs)
            implicit = compute_trade(pair, 1_000_000)
            explicit = compute_trade(pair, 1_000_000, settings=live_settings())
            assert (implicit.x, implicit.kelly_fraction, implicit.kelly_p,
                    implicit.total_cost_with_fees) == (
                explicit.x, explicit.kelly_fraction, explicit.kelly_p,
                explicit.total_cost_with_fees)
        # Resolved at CALL time, from config.py's constants
        monkeypatch.setattr(config, "BUDGET_FRACTION", 0.35)
        assert compute_trade(make_pair(**self._ST), 1_000_000).kelly_fraction == \
            pytest.approx(0.35)
        monkeypatch.setattr(config, "SAME_TITLE_SIZE_CAP", 0.10)
        assert compute_trade(make_pair(**self._ST), 1_000_000).kelly_fraction == \
            pytest.approx(0.10)
        monkeypatch.setattr(config, "TIME_SERIES_INTERVAL_PROB_DISCOUNT", 0.6)
        assert compute_trade(make_pair(**self._TS), 1_000_000).kelly_p == pytest.approx(0.82)

    def test_time_series_fraction_stays_under_one_minus_k(self):
        # With no per-trade cap, 1 - k bounds a time-series f* whenever the
        # reference YES ask is at or above the later book's YES bid (see
        # config.max_kelly_fraction, which relies on it)
        rng = random.Random(20260927)
        balance = 1_000_000
        for k in (0.4, 0.6, 0.75, 0.8, 0.9):
            s = _live(k=k, cap=1.0)
            bound = config.max_kelly_fraction("time_series", s)
            sized = 0
            books = [[(0.02, 0.02, 500.0)], [(0.05, 0.05, 50.0), (0.06, 0.07, 5000.0)]]
            for _ in range(300):
                pa, nb = rng.randint(1, 49) / 100, rng.randint(1, 49) / 100
                levels = []
                for _level in range(rng.randint(1, 3)):
                    levels.append((pa, nb, float(rng.randint(10, 2000))))
                    pa = round(pa + rng.randint(0, 3) / 100, 2)
                    nb = round(nb + rng.randint(0, 3) / 100, 2)
                books.append(levels)
            for levels in books:
                best_nb = levels[0][1]
                # The reference YES ask, at or above the later book's YES bid
                pB = rng.randint(round((1.0 - best_nb) * 100), 99) / 100
                for pair in (make_booked_pair(levels, pB=pB),
                             make_pair(pA=levels[0][0], pB=pB, nA=1 - levels[0][0],
                                       nB=best_nb, pair_type="time_series")):
                    spec = compute_trade(pair, balance, settings=s)
                    if spec is None:
                        continue
                    sized += 1
                    assert spec.kelly_fraction < 1.0 - k, (k, levels, pB, spec.kelly_fraction)
                    assert spec.kelly_fraction <= bound, (k, levels, pB)
            # Non-vacuous at every k, the 0.9 extreme included
            assert sized > 0, k


def _held_pair(cost: float, account_value: float, count: float = 30.0) -> HeldPair:
    """A held pair the account holds on A1 (YES) / B1 (NO), for the sizer."""
    return HeldPair(sides=(("A1", "yes"), ("B1", "no")), count=count,
                    cost_dollars=cost, account_value_dollars=account_value)


class TestComputeTradeAddsToHeldPair:
    """An add-on to a held pair (CandidatePair.held) is sized on its whole
    position: the held stake plus the new one stays within min(f*, cap) of
    the account value, and the new stake alone within what a new pair would
    stake of the cash. Both sizing paths, the bookless one and the book
    search, read it through config.held_pair_fraction. (A book deep enough
    that the fee shrink re-prices at a smaller count can round a leg's fee up
    a cent, so there the bounds hold to within a cent or two, as for every
    trade; these fixtures' one-level book never re-prices.)"""

    # k 0.75, a 10% cap: the time-series fixture's f* (about 0.16) is capped
    _SETTINGS = LiveSettings(tier_floors=True, spread_band=(0.0, 1.0),
                             interval_discount=0.75, size_cap=0.10)
    _BALANCE_CENTS = 15_000

    def _bookless(self, held):
        """The flow-through time-series fixture, bookless, adding to `held`."""
        pair = make_pair(pA=_TS_PA, pB=_TS_PB, nA=_TS_NA, nB=_TS_NB, pair_type="time_series")
        pair.held = held
        return pair

    @staticmethod
    def _booked(held, qty: float = 100):
        """The same pair with one level of book, adding to `held`."""
        pair = make_booked_pair([(_TS_PA, _TS_NB, qty)], pair_type="time_series", pB=_TS_PB)
        return dataclasses.replace(pair, held=held)

    @pytest.mark.parametrize("path", ["bookless", "booked"])
    def test_the_whole_position_stays_within_its_kelly_share(self, path):
        held = _held_pair(cost=8.0, account_value=211.0)
        build = self._bookless if path == "bookless" else self._booked
        spec = compute_trade(build(held), self._BALANCE_CENTS, settings=self._SETTINGS)
        plain = compute_trade(build(None), self._BALANCE_CENTS, settings=self._SETTINGS)
        assert spec is not None and plain is not None
        fraction = config.held_pair_fraction(0.10, 8.0, 211.0, 150.0)
        assert spec.kelly_fraction == pytest.approx(fraction)
        # Old and new together within 10% of the account value ...
        assert 8.0 + spec.total_cost_with_fees <= 0.10 * 211.0 + 1e-9
        # ... the new stake within what the add-on may stake ...
        assert spec.total_cost_with_fees <= fraction * 150.0 + 1e-9
        # ... and never more than a new pair would stake of the cash
        assert spec.total_cost_with_fees <= 0.10 * 150.0 + 1e-9
        # Non-vacuous: the same pair without the held stake buys more
        assert plain.x > spec.x
        # The held pair reaches the spec the trader and the reports read
        assert pair_held(spec.pair) is held

    @pytest.mark.parametrize("path", ["bookless", "booked"])
    def test_a_pair_holding_little_sizes_as_a_new_pair(self, path):
        settings = LiveSettings(tier_floors=True, spread_band=(0.0, 1.0),
                                interval_discount=0.75, size_cap=1.0)
        build = self._bookless if path == "bookless" else self._booked
        spec = compute_trade(build(_held_pair(cost=1.0, account_value=10_000.0)),
                             self._BALANCE_CENTS, settings=settings)
        plain = compute_trade(build(None), self._BALANCE_CENTS, settings=settings)
        assert spec is not None and plain is not None
        assert (spec.x, spec.total_cost_with_fees, spec.kelly_fraction) == (
            plain.x, plain.total_cost_with_fees, plain.kelly_fraction)

    @pytest.mark.parametrize("path", ["bookless", "booked"])
    def test_a_pair_at_its_kelly_share_adds_nothing_and_says_so_once(self, path, caplog):
        # $30 held against 10% of a $211 account: nothing is missing
        held = _held_pair(cost=30.0, account_value=211.0)
        pair = self._bookless(held) if path == "bookless" else self._booked(held, qty=1000)
        with caplog.at_level(logging.INFO):
            assert compute_trade(pair, self._BALANCE_CENTS, settings=self._SETTINGS) is None
        lines = [r.getMessage() for r in caplog.records
                 if r.getMessage().startswith("Not adding to held pair")]
        # One line, however many sizes the book search tried
        assert lines == [
            f"Not adding to held pair '{pair.canonical_title}': it already holds its "
            "Kelly share (30 contracts each, $30.00 staked, account value $211.00)"]

    @pytest.mark.parametrize("case", ["untradeable", "unreachable-book"])
    def test_any_other_refusal_says_no_size_fits(self, case, caplog):
        # The held pair holds $1 of a 10% share of $10,000, so Kelly would add;
        # what refuses it is something else, and the line must not blame Kelly
        held = _held_pair(cost=1.0, account_value=10_000.0)
        if case == "untradeable":
            pair = dataclasses.replace(self._booked(held), tradeable=False)
        else:
            # No fill-or-kill order can buy one contract pair from half a contract
            pair = dataclasses.replace(
                make_booked_pair([(_TS_PA, _TS_NB, 0.5)], pair_type="time_series",
                                 pB=_TS_PB, max_contracts=1), held=held)
        with caplog.at_level(logging.INFO):
            assert compute_trade(pair, 1_000_000, settings=self._SETTINGS) is None
        lines = [r.getMessage() for r in caplog.records
                 if r.getMessage().startswith("Not adding to held pair")]
        assert lines == [f"Not adding to held pair '{pair.canonical_title}': "
                         "no size fits this run"]

    def test_an_ordinary_pair_it_refuses_logs_nothing_new(self, caplog):
        with caplog.at_level(logging.INFO):
            assert compute_trade(make_pair(tradeable=False), self._BALANCE_CENTS,
                                 settings=self._SETTINGS) is None
        assert "Not adding to held pair" not in caplog.text

    def test_the_trade_line_names_the_held_count(self, caplog):
        with caplog.at_level(logging.INFO):
            compute_trade(self._bookless(_held_pair(cost=8.0, account_value=211.0)),
                          self._BALANCE_CENTS, settings=self._SETTINGS)
            [added] = [r.getMessage() for r in caplog.records
                       if r.getMessage().startswith("Trade computed")]
            caplog.clear()
            compute_trade(self._bookless(None), self._BALANCE_CENTS, settings=self._SETTINGS)
            [plain] = [r.getMessage() for r in caplog.records
                       if r.getMessage().startswith("Trade computed")]
        assert added.endswith(" | adds to 30 held")
        # An ordinary line ends where it always did
        assert plain.endswith("%") and "adds to" not in plain

    def test_a_mock_pairs_truthy_held_attribute_is_not_an_add_on(self, caplog):
        mock_pair = make_pair(pA=_TS_PA, pB=_TS_PB, nA=_TS_NA, nB=_TS_NB,
                              pair_type="time_series")
        # MagicMock answers pair.held with a truthy auto-attribute
        assert mock_pair.held
        with caplog.at_level(logging.INFO):
            spec = compute_trade(mock_pair, self._BALANCE_CENTS, settings=self._SETTINGS)
        plain = compute_trade(self._bookless(None), self._BALANCE_CENTS,
                              settings=self._SETTINGS)
        assert spec is not None and spec.x == plain.x
        assert "adds to" not in caplog.text


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


def _keyword_values(module, func_name: str, callee: str, keyword: str, *,
                    source: str | None = None) -> list:
    """
    The value passed as `keyword` in every call to `callee` inside a function.

    Args:
        module: The module to read.
        func_name (str): The function whose calls are read.
        callee (str): The called name, bare or as an attribute.
        keyword (str): The keyword argument to read.
        source (str | None): Source to read instead of the module's own.

    Returns:
        list: One AST node per call, in the order ast.walk visits them; None
            for a call without the keyword.
    """
    tree = ast.parse(source if source is not None else inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            values = []
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    fn = sub.func
                    name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
                    if name == callee:
                        values.append(next((k.value for k in sub.keywords if k.arg == keyword),
                                           None))
            return values
    raise AssertionError(f"{module.__name__}.{func_name} not found")


def _key_homes(tree: ast.AST, key: str) -> list[tuple[str | None, int]]:
    """Every place `key` is written in `tree`, as a string or as a keyword
    argument: each string constant EQUAL to it (a dict key, a subscript, a
    .get) and each keyword argument NAMED it (dict(entry, later=...)). Each
    place is returned as (owner, line): owner is the outermost function
    around it ("Class.method" for a method), or None at module or class
    level. Equality, not substring, so a docstring that merely mentions the
    word never matches. A key built at run time ("lat" + "er", an f-string)
    is not found."""
    homes: list[tuple[str | None, int]] = []

    def visit(node: ast.AST, owner: str | None, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            if owner is None and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, scope + child.name, scope)
            elif owner is None and isinstance(child, ast.ClassDef):
                visit(child, None, f"{scope}{child.name}.")
            elif isinstance(child, ast.Constant):
                if type(child.value) is str and child.value == key:
                    homes.append((owner, child.lineno))
            else:
                if isinstance(child, ast.keyword) and child.arg == key:
                    homes.append((owner, child.lineno))
                visit(child, owner, scope)

    visit(tree, None, "")
    return homes


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
    # config.py's k, as compute_trade(settings=None) resolves it
    p = strategy._kelly_p_at(pair, price_a, k=config.TIME_SERIES_INTERVAL_PROB_DISCOUNT)
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
        # f* < 1 - k = 0.20 at config.py's k of 0.80 (no per-trade cap, pB 0.62 uncrossed):
        # at most $6.00 at 0.70 a pair -> 8 pairs, well inside the 20 resting at the top
        # rung, so the fill is the best level outright.
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
        assert _kelly_p(pair, live_settings()) == time_series_profit_prob(_TS_PA, _TS_PB)

    def test_dashboard_fraction_equals_compute_trade_fraction(self):
        dash = dashboard._kelly_fraction(_TS_PA, _TS_NA, _TS_PB, _TS_NB, "time_series")
        assert dash == pytest.approx(_ts_kelly_fraction(_TS_PA, _TS_PB, _TS_NB))
        live = compute_trade(
            make_pair(pA=_TS_PA, pB=_TS_PB, nA=_TS_NA, nB=_TS_NB, pair_type="time_series"),
            1_000_000,
        )
        assert live is not None
        assert live.kelly_fraction == pytest.approx(dash)

    @pytest.mark.usefixtures("pre_toggle_defaults")
    def test_dashboard_fraction_uses_leg_prices_not_nA_pB(self):
        # On this fixture nA + pB = 1.30 — the old leg mapping would return 0.0
        # (no spread), not the ~0.1620 the live sizer computes at k 0.75 (pre_toggle_defaults)
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
        # SS-1 moved the backtester's call into _ts_group_key, its one
        # definition of the key, so the pin is a two-link chain — and it now
        # also covers _index_eligible_keys, which decides from the SAME two
        # key helpers which records are materialized for grouping at all. A
        # key computed differently there would silently drop records the
        # grouping needs, so both pass-1 keys are pinned beside both groupings.
        assert _function_calls(backtester, "_ts_group_key", "time_series_group_key")
        assert _function_calls(backtester, "_group_by_normalized_title", "_ts_group_key")
        assert _function_calls(backtester, "_group_by_exact_title", "_st_group_key")
        assert _function_calls(backtester, "_index_eligible_keys", "_ts_group_key")
        assert _function_calls(backtester, "_index_eligible_keys", "_st_group_key")

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

    def test_ast_both_finders_apply_the_same_event_ladder_rule(self):
        # DR-73: a same-event deadline ladder is ordered and tiered on its two
        # STATED deadlines, so BOTH paths must reach BOTH halves of that
        # arithmetic — stated_deadline (wording -> calendar day) and
        # same_event_ladder (two days -> leg order and gap). Re-deriving
        # either is how the two paths drifted apart before (DR-01), and
        # close_time cannot answer either question for a ladder whose rungs
        # settle at one instant.
        #
        # Both-paths for the same reason the wording and one-series rules are
        # (see the two pins above): a ladder-enabled LIVE run measured by a
        # backtest that still refused every same-event pair would be measuring
        # a different strategy. _extract_pairs reaches stated_deadline through
        # _stated_deadline_dict, its dict-world field extraction, exactly as
        # it reaches deadline_profile through _deadline_profile_dict.
        assert _function_calls(scanner, "find_time_series_pairs", "stated_deadline")
        assert _function_calls(scanner, "find_time_series_pairs", "same_event_ladder")
        # BOTH of _extract_pairs' reads are pinned, and the second is the
        # load-bearing one: the bare stated_deadline call in that function is
        # the CROSS-CHECK-DISARMED reading (three empty field arguments) that
        # only splits the undated/conflict counters, so asserting it alone
        # leaves green a tree whose ladder rule sources its actual date from
        # somewhere other than the dict-world, cross-checking extractor —
        # exactly the live/backtest drift this pin is named for. The
        # _find_entry half below already asserts _stated_deadline_dict by name.
        assert _function_calls(backtester, "_extract_pairs", "stated_deadline")
        assert _function_calls(backtester, "_extract_pairs", "_stated_deadline_dict")
        assert _function_calls(backtester, "_extract_pairs", "same_event_ladder")
        assert _function_calls(backtester, "_stated_deadline_dict", "stated_deadline")
        # _find_entry orders and gaps the pair it replays on the same helper,
        # so the entry the backtest books cannot disagree with the pair
        # _extract_pairs formed.
        assert _function_calls(backtester, "_find_entry", "same_event_ladder")
        assert _function_calls(backtester, "_find_entry", "_stated_deadline_dict")

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
        # validate_pair_price's ceiling test too: a wider close_time gap could raise
        # the floor, and its BELOW_FLOOR answer would let an over-ceiling spread pass
        assert _function_calls(scanner, "validate_pair_price", "pair_gap_days")
        assert not _function_calls(scanner, "validate_pair_price", "deadline_gap_days")

    def test_ast_the_series_prefix_has_one_definition(self):
        # The backtester must not re-split the event ticker itself: the mirror
        # helpers call scanner.event_series, so live and backtest can never
        # disagree about what a series is.
        assert _function_calls(backtester, "_same_series_dicts", "event_series")
        assert _function_calls(scanner, "_same_series", "event_series")

    def test_ast_both_paths_apply_the_same_title_close_gap(self):
        # DR-74: identical wording on two DIFFERENT series is one question only
        # when both markets close at the same moment (a men's and a women's
        # game between the same schools close hours apart). Gating only the
        # live finder would leave the backtester replaying the M/W trades
        # instead of detecting them — 17 of its 21 trades on the 365-day run.
        # The live finder reaches the gate through its attribute reader; the
        # backtester calls the one definition directly on parsed closes.
        assert _function_calls(scanner, "find_same_title_pairs", "_closes_apart")
        assert _function_calls(backtester, "_extract_pairs", "closes_apart")
        assert _function_calls(backtester, "_extract_pairs", "_comparable_closes_dicts")
        # And deliberately NOT on the time-series finder: identical wording can
        # never form a time-series pair (DR-67), so a same-title-only gate
        # cannot relabel the trade, and a close gate there would refuse every
        # genuine cross-event deadline pair, whose closes differ by design.
        assert not _function_calls(scanner, "find_time_series_pairs", "_closes_apart")
        assert not _function_calls(scanner, "find_time_series_pairs", "closes_apart")

    def test_ast_the_close_gap_has_one_definition(self):
        # Only the close-time READ differs between the paths (an attribute
        # live, a parsed cache string in the backtest); the verdict is one
        # function, so the two can never disagree about which pair closes at
        # one moment. Same shape as the series-prefix pin above.
        assert _function_calls(scanner, "_closes_apart", "closes_apart")
        # And the bound each path's refusal line PRINTS comes from the one
        # renderer, which reads the binding closes_apart reads — so neither
        # line can state a bound its gate did not apply, and the two stay
        # verbatim twins under a patched bound.
        assert _function_calls(scanner, "find_same_title_pairs", "close_gap_bound_text")
        assert _function_calls(backtester, "_extract_pairs", "close_gap_bound_text")

    def test_ast_backtest_band_reaches_find_entry_through_config(self):
        # The backtest band's halves each have ONE home in config: the band is
        # resolved and validated by time_series_spread_band, its floor is
        # layered on the tier inside min_price_diff_for_gap (so the
        # leg-price-sum ceiling moves with it), and its ceiling's
        # PRICE_EPSILON lives only in time_series_spread_too_wide (TS-09). An
        # inline `gap > band_hi + PRICE_EPSILON` in _find_entry behaves
        # identically — every behaviour test stays green — while opening a
        # second copy of the tolerance to drift, so the calls are pinned here.
        assert _function_calls(backtester, "_find_entry", "time_series_spread_band")
        assert _function_calls(backtester, "_find_entry", "time_series_spread_too_wide")
        tree = ast.parse(inspect.getsource(backtester))
        find_entry = next(n for n in ast.walk(tree)
                          if isinstance(n, ast.FunctionDef) and n.name == "_find_entry")
        tier_calls = [
            node for node in ast.walk(find_entry)
            if isinstance(node, ast.Call)
            and (node.func.id if isinstance(node.func, ast.Name)
                 else getattr(node.func, "attr", None)) == "min_price_diff_for_gap"
        ]
        # Non-vacuous, and every tier call carries the band's floor AND hands
        # on the tier-floors switch by name: a call that dropped it would
        # silently re-apply the tiers inside the tier-floors-off family, and
        # every behaviour test of a tier-on band would stay green
        assert tier_calls
        for node in tier_calls:
            keywords = {k.arg for k in node.keywords}
            assert "spread_min" in keywords, node.lineno
            assert "tier_floors" in keywords, node.lineno
        # Which bands the tiers bind at is decided THROUGH the helper, asked
        # both ways, never from a copy of the tier constants
        assert _function_calls(backtester, "_tier_floors_bind", "min_price_diff_for_gap")

    def test_ast_later_mondays_have_one_reader(self):
        # backtester._find_entry stores every qualifying Monday after the
        # first under "later", and _entry_mondays is the one function the
        # rest of the code reads them through (DR-75; its .get default
        # matters, since hand-built entries have no "later"). This checks that
        # the key is written — as a string or as a keyword argument,
        # dict(e, later=...) — only in those two and in _split_halves, which
        # trims the list, anywhere in the package; and, with the last two
        # asserts, that the calibration and the split date, which must use
        # only the first Monday, never read the rest. An unrelated "later"
        # anywhere in the package fails this too: add it to the allowed places
        # on purpose if one is ever needed.
        import importlib
        import pkgutil

        import kalshi_betting

        # The walk finds a keyword spelling, not only a string one
        probe = ast.parse("def f(e):\n    return dict(e, later=())\n")
        assert _key_homes(probe, "later") == [("f", 2)]
        names = sorted(m.name for m in pkgutil.iter_modules(kalshi_betting.__path__))
        assert "backtester" in names
        modules = [kalshi_betting] + [importlib.import_module(f"kalshi_betting.{n}")
                                      for n in names]
        homes = [(module.__name__, owner, line)
                 for module in modules
                 for owner, line in _key_homes(ast.parse(inspect.getsource(module)), "later")]
        found = {(mod, owner) for mod, owner, _line in homes}
        writer = ("kalshi_betting.backtester", "_find_entry")
        reader = ("kalshi_betting.backtester", "_entry_mondays")
        truncator = ("kalshi_betting.backtester", "_split_halves")
        assert found <= {writer, reader, truncator}, homes
        # Not vacuous: the writer spells it (a renamed key would otherwise
        # leave nothing to check), and so do the one reader and the truncator
        assert writer in found and reader in found and truncator in found
        # ... the count line, the Kelly gate, the excluding-top-event check and
        # the cap sweep's event census read the Mondays through that reader ...
        for function in ("_log_qualifying_mondays", "_simulate_at_discount",
                         "_ex_top_event", "entry_events"):
            assert _function_calls(backtester, function, "_entry_mondays"), function
        # ... and the two first-Monday readers never call it
        assert not _function_calls(backtester, "_interval_calibration", "_entry_mondays")
        assert not _function_calls(backtester, "_split_date", "_entry_mondays")

    # The live entry points that may resolve config.py's toggles when handed
    # none, in one statement of one form; every other def taking `settings` requires it.
    _LIVE_SETTINGS_RESOLVERS = frozenset({
        ("scanner", "find_time_series_pairs"),
        ("scanner", "enrich_with_orderbook_prices"),
        ("scanner", "validate_pair_price"),
        ("strategy", "compute_trade"),
        ("trader", "pre_execution_check"),
        ("main", "_run_dev"),
        ("main", "_run_prod"),
        ("main", "_no_pairs_msg"),
    })

    # config.py's constants as LiveSettings are read on the live path only by a
    # whitelisted entry point's None fallback (above); no function calls
    # live_settings() directly
    _DIRECT_RESOLVERS = frozenset()

    # The readers of the SAVED live defaults, each called directly once, and
    # only in these functions: main._resolve_live_settings lays the flags over
    # them for a run; scheduler._check_live_defaults warns at daemon start;
    # defaults_server._current_defaults is the confirmation page's one read
    _DEFAULTS_READERS = {
        "live_defaults": frozenset({("main", "_resolve_live_settings")}),
        "read_saved_live_defaults": frozenset({("scheduler", "_check_live_defaults"),
                                               ("defaults_server", "_current_defaults")}),
    }

    # The writer of the saved live defaults and the functions that may call it,
    # each exactly once: the confirmation page's Confirm, and nothing else. In
    # both tables a method is named with its class
    _DEFAULTS_WRITERS = {
        "save_live_defaults": frozenset({("defaults_server", "_App._post_confirm")}),
    }

    # The saved file's path, config's private parse of it and the seed values: a
    # walked module reaches the saved live defaults only through a
    # _DEFAULTS_READERS reader, so it names none of these (a second read of the
    # file through them would slip past that table)
    _SAVED_FILE_INTERNALS = frozenset({
        "LIVE_DEFAULTS_FILE",
        "LIVE_DEFAULTS_SEED",
        "_saved_settings",
        "_settings_from_bytes",
    })

    # The one module that may name some of those internals, and which: the
    # defaults server proposes the seed values, and prints the saved file's
    # path on its pages and in its log. Each exempted name is importable or
    # print-only (below); either way it may only be read, never assigned,
    # deleted or taken as a parameter
    _SAVED_FILE_EXEMPT = {
        "defaults_server": frozenset({"LIVE_DEFAULTS_FILE", "LIVE_DEFAULTS_SEED"}),
    }

    # The exempted names a module may import by value (from .config import X)
    # and then read freely: the seed is a value, not a way to the file
    _SAVED_FILE_IMPORTABLE = frozenset({"LIVE_DEFAULTS_SEED"})

    # The exempted names a module may only print, always as config.X (never
    # imported by value): config.X as an argument of a logging message call
    # (logging.info and its siblings), or its .name, or its .absolute(), as
    # the one argument of str() or html.escape() or as an f-string field. And
    # every call around the reference in its own statement must print or build
    # text: str(), html.escape(), a logging message call, or one of the
    # module's own functions or methods. So config.X.read_bytes(),
    # Path(str(config.X.absolute())).write_text(...) and
    # logging.FileHandler(config.X) all fail. The check stops at the
    # statement: text of the path bound to a name, or handed to one of the
    # module's own functions, is not followed, so it cannot prove the module
    # never rebuilds the path from that text
    _SAVED_FILE_PRINT_ONLY = frozenset({"LIVE_DEFAULTS_FILE"})

    # The logging calls that write a message. A handler (logging.FileHandler)
    # opens the file it is given, so it is not one of them
    _LOG_MESSAGE_CALLS = frozenset({"debug", "info", "warning", "error", "critical",
                                    "exception", "log"})

    # Code appended to defaults_server's source for the print-only rule's
    # mutant checks, by id: each reaches the saved file through its path
    _PRINT_ONLY_MUTANTS = {
        # text of the path turned back into a path, and written
        "path-from-text": ("def _mutant():\n"
                           "    Path(str(config.LIVE_DEFAULTS_FILE.absolute())).write_text('{}')\n"),
        # the same through an f-string field
        "open-f-string": ("def _mutant():\n"
                          "    open(f'{config.LIVE_DEFAULTS_FILE.absolute()}', 'w')\n"),
        # a logging call that is not a message: the handler opens the file
        "log-file-handler": ("def _mutant():\n"
                             "    logging.FileHandler(config.LIVE_DEFAULTS_FILE)\n"),
        # the path used directly
        "read-bytes": ("def _mutant():\n"
                       "    config.LIVE_DEFAULTS_FILE.read_bytes()\n"),
    }

    def test_ast_live_path_reads_toggles_only_through_live_settings(self):
        # ONE frozen config.LiveSettings per run carries every live toggle, so
        # no site applies a setting while another reads config.py. In every
        # package module but config and the backtest-side band readers:
        #   - no band/tier-floor helper or constant (min_price_diff_for_gap
        #     included) or toggle constant is named, spelled, shadowed or
        #     imported, and no band reader is imported;
        #   - live_settings is read only by each whitelisted entry point's one
        #     resolving statement, and never called directly;
        #   - the saved live defaults are read only where _DEFAULTS_READERS
        #     allows, each allowed function calling its reader exactly once,
        #     and written only where _DEFAULTS_WRITERS allows (the defaults
        #     server's Confirm alone); the saved file's path, config's private
        #     parse of it and the seed (_SAVED_FILE_INTERNALS) are never named,
        #     but that the defaults server may read the seed and print the
        #     path (_SAVED_FILE_EXEMPT, in the shapes _SAVED_FILE_IMPORTABLE
        #     and _SAVED_FILE_PRINT_ONLY allow);
        #   - a def with a `settings` parameter is only ever called (pool.submit
        #     too), with the bare name `settings`, never None or the `reference`,
        #     and every whitelisted def, and only those, defaults it to None;
        #   - time_series_profit_prob, _kelly_p_at and max_affordable_pairs always
        #     get k / the fraction, never None, and are never passed around uncalled.
        # A toggle value hardcoded inline as a literal is outside this pin's
        # reach; the constants-live-in-config rule covers that.
        import importlib
        import pkgutil

        import kalshi_betting

        band_readers = {"config", "backtester", "backtest", "dashboard"}
        names = {m.name for m in pkgutil.iter_modules(kalshi_betting.__path__)}
        # A rename must fail here, not silently shrink the allowlist
        assert band_readers <= names, band_readers - names
        walked = sorted(names - band_readers)
        # The live pipeline and the defaults server are in scope (the walk
        # cannot pass by finding nothing to walk)
        assert {"scanner", "strategy", "trader", "main", "defaults_server"} <= set(walked), walked
        # An exemption names a walked module and only saved-file internals,
        # each with a stated shape (importable or print-only, not both)
        assert set(self._SAVED_FILE_EXEMPT) <= set(walked), self._SAVED_FILE_EXEMPT
        assert not self._SAVED_FILE_IMPORTABLE & self._SAVED_FILE_PRINT_ONLY
        for names_exempt in self._SAVED_FILE_EXEMPT.values():
            assert names_exempt <= self._SAVED_FILE_INTERNALS, names_exempt
            assert names_exempt <= self._SAVED_FILE_IMPORTABLE | self._SAVED_FILE_PRINT_ONLY, (
                names_exempt)
        modules = [kalshi_betting] + [importlib.import_module(f"kalshi_betting.{n}") for n in walked]
        no_import = band_readers - {"config"}

        forbidden = {
            "min_price_diff_for_gap",
            "time_series_spread_band",
            "time_series_spread_too_wide",
            "BACKTEST_DEFAULT_SPREAD_BAND",
            "SPREAD_BAND_SWEEP_FLOORS",
            "SPREAD_BAND_SWEEP_CEILINGS",
            "TIME_SERIES_TIER_FLOORS",
            "TIME_SERIES_SPREAD_BAND",
            "TIME_SERIES_INTERVAL_PROB_DISCOUNT",
            "BUDGET_FRACTION",
            "SAME_TITLE_SIZE_CAP",
            "TRADE_CATEGORIES",
            "TRADE_TAGS",
        } | self._SAVED_FILE_INTERNALS
        resolver = "live_settings"
        # (module, exempted name) -> references found, so no exemption outlives its use
        exempt_uses = {(mod, name): 0 for mod, names in self._SAVED_FILE_EXEMPT.items()
                       for name in names}
        # The saved live defaults' readers and writer: name -> the (module,
        # function) pairs that may call it, each exactly once
        tracked = {**self._DEFAULTS_READERS, **self._DEFAULTS_WRITERS}
        # A rename in config must fail here, not leave the tables naming nothing
        for name in (*tracked, *self._SAVED_FILE_INTERNALS):
            assert hasattr(config, name), name
        # Helpers that read config.py's k / fraction when handed None (or, where
        # it is defaulted, nothing): name -> (positional index, keyword name)
        explicit_args = {
            "time_series_profit_prob": (2, "k"),
            "_kelly_p_at": (2, "k"),
            "max_affordable_pairs": (2, "fraction"),
        }

        def short(module):
            return module.__name__.rsplit(".", 1)[-1]

        trees = {short(m): ast.parse(inspect.getsource(m)) for m in modules}

        def param_default(node, name):
            """(declared, default node or None) for the parameter `name`."""
            positional = node.args.posonlyargs + node.args.args
            defaults = [None] * (len(positional) - len(node.args.defaults)) + list(
                node.args.defaults)
            for arg, default in zip(positional, defaults, strict=True):
                if arg.arg == name:
                    return True, default
            for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True):
                if arg.arg == name:
                    return True, default
            return False, None

        # Every def with a `settings` parameter, walked modules and config:
        # name -> positional index, or None when it is keyword-only
        takes_settings: dict = {}
        for mod, tree in [*trees.items(), ("config", ast.parse(inspect.getsource(config)))]:
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                positional = [a.arg for a in node.args.posonlyargs + node.args.args]
                kwonly = [a.arg for a in node.args.kwonlyargs]
                if "settings" in positional:
                    where = positional.index("settings")
                elif "settings" in kwonly:
                    where = None
                else:
                    continue
                assert takes_settings.get(node.name, where) == where, node.name
                takes_settings[node.name] = where
                # A whitelisted entry point defaults settings to None; every other def requires it
                _, default = param_default(node, "settings")
                if (mod, node.name) in self._LIVE_SETTINGS_RESOLVERS:
                    assert isinstance(default, ast.Constant) and default.value is None, (
                        f"{mod}.{node.name} must default settings to None")
                else:
                    assert default is None, (
                        f"{mod}.{node.name} gives settings a default; only a whitelisted "
                        "entry point may, and it must resolve it")
        # Non-vacuous: the helpers and entry points this rule exists for
        assert {"find_time_series_pairs", "enrich_with_orderbook_prices",
                "validate_pair_price", "pre_execution_check", "_pair_max_sum",
                "live_time_series_floor", "time_series_spread_refusal",
                "max_kelly_fraction", "_run_dev", "_run_prod", "compute_trade",
                "_evaluate_size", "_solve_marginal_size", "_kelly_p",
                "_compute_trade_specs", "_no_pairs_msg", "_log_live_settings",
                "describe_live_settings", "live_rule_warnings", "_filter_by_category",
                "describe_trade_filter"} <= set(takes_settings)
        # _kelly_p_at's k is required: a default would price at config's k
        kelly_p_at = next(n for n in ast.walk(trees["strategy"])
                          if isinstance(n, ast.FunctionDef) and n.name == "_kelly_p_at")
        declared, default = param_default(kelly_p_at, "k")
        assert declared and default is None, "strategy._kelly_p_at must require k"

        def passes_explicitly(call, fn_name):
            """Whether the call hands fn_name its k/fraction, never a literal None."""
            index, keyword = explicit_args[fn_name]
            for kw in call.keywords:
                if kw.arg == keyword:
                    return not (isinstance(kw.value, ast.Constant) and kw.value.value is None)
            if any(isinstance(a, ast.Starred) for a in call.args) or len(call.args) <= index:
                return False
            value = call.args[index]
            return not (isinstance(value, ast.Constant) and value.value is None)

        def passes_settings(call, fn_name, skip):
            """Whether the call hands `fn_name` the bare name `settings`,
            positionally or by keyword; `skip` args precede the callee's own."""
            def is_the_runs(value):
                return isinstance(value, ast.Name) and value.id == "settings"

            for kw in call.keywords:
                if kw.arg == "settings":
                    return is_the_runs(kw.value)
            index = takes_settings[fn_name]
            args = call.args[skip:]
            if index is None or any(isinstance(a, ast.Starred) for a in args):
                return False
            if len(args) <= index:
                return False
            return is_the_runs(args[index])

        def call_name(func):
            return func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)

        def text_function(func):
            """
            Whether a called expression is str or html.escape.

            Args:
                func (ast.AST): A call's func node.

            Returns:
                bool: True for the name str or the attribute html.escape.
            """
            return ((isinstance(func, ast.Name) and func.id == "str")
                    or (isinstance(func, ast.Attribute) and func.attr == "escape"
                        and isinstance(func.value, ast.Name) and func.value.id == "html"))

        def log_message(func):
            """
            Whether a called expression is a logging call that writes a message.

            Args:
                func (ast.AST): A call's func node.

            Returns:
                bool: True for logging.info and its _LOG_MESSAGE_CALLS siblings.
            """
            return (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                    and func.value.id == "logging" and func.attr in self._LOG_MESSAGE_CALLS)

        def as_text(holder, value):
            """
            Whether a node turns an expression straight into text.

            Args:
                holder (ast.AST | None): The node around the expression.
                value (ast.AST): The expression.

            Returns:
                bool: True when holder is an f-string field holding value, or a
                    call of str() or html.escape() with value its one argument.
            """
            if isinstance(holder, ast.FormattedValue):
                return holder.value is value
            return (isinstance(holder, ast.Call) and len(holder.args) == 1
                    and holder.args[0] is value and not holder.keywords
                    and text_function(holder.func))

        def prints_the_path(node, parents, own_functions, own_methods):
            """
            Whether a reference to a _SAVED_FILE_PRINT_ONLY name only prints the path.

            The allowed shapes: config.X as a positional argument of a logging
            message call (log_message); config.X.name, or config.X.absolute(),
            turned straight into text (as_text). A read or write of the file,
            or config.X under another name, is none of them. Then every call
            around the reference, up to its statement, must print or build
            text: str(), html.escape(), a logging message call, or one of the
            module's own functions or methods, called by name or on self. So
            text of the path handed to Path(), open() or anything else fails.
            Text bound to a name, or handed to the module's own functions, is
            not followed further.

            Args:
                node (ast.AST): The reference (a Name or an Attribute).
                parents (dict): Each node of its module's tree's parent, by id.
                own_functions (set[str]): The module's top-level function names.
                own_methods (set[str]): The names of the methods its top-level
                    classes define.

            Returns:
                bool: True when the reference has one of the allowed shapes and
                    every call around it prints or builds text.
            """
            if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id == "config"):
                return False
            parent = parents.get(id(node))
            above = parents.get(id(parent))
            if isinstance(parent, ast.Call) and any(a is node for a in parent.args):
                # config.X handed to a logging message call, which prints it
                shaped = log_message(parent.func)
            elif not (isinstance(parent, ast.Attribute) and parent.value is node):
                shaped = False
            elif parent.attr == "name":
                # config.X.name, as text
                shaped = as_text(above, parent)
            elif (parent.attr == "absolute" and isinstance(above, ast.Call)
                    and above.func is parent and not above.args and not above.keywords):
                # config.X.absolute(), as text
                shaped = as_text(parents.get(id(above)), above)
            else:
                shaped = False
            if not shaped:
                return False
            cur = parent
            while cur is not None and not isinstance(cur, ast.stmt):
                if isinstance(cur, ast.Call):
                    func = cur.func
                    allowed_call = (
                        text_function(func) or log_message(func)
                        # the shape's own .absolute()
                        or (isinstance(func, ast.Attribute) and func.value is node
                            and func.attr == "absolute")
                        or (isinstance(func, ast.Name) and func.id in own_functions)
                        or (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                            and func.value.id == "self" and func.attr in own_methods))
                    if not allowed_call:
                        return False
                cur = parents.get(id(cur))
            return True

        resolutions: dict = {}
        direct_resolutions: dict = {}
        explicit_calls: dict = {}
        # (tracked name, module, function) -> direct calls found there
        defaults_calls: dict = {}
        for mod, tree in trees.items():
            parents = {}
            for node in ast.walk(tree):
                for child in ast.iter_child_nodes(node):
                    parents[id(child)] = node

            def enclosing(node, parents=parents):
                cur = parents.get(id(node))
                while cur is not None and not isinstance(
                        cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    cur = parents.get(id(cur))
                return cur.name if cur is not None else None

            def qualified(node, parents=parents):
                """
                Name the function a node sits in, with its class when it is a method.

                Args:
                    node (ast.AST): The node.
                    parents (dict): Each node's parent, by id.

                Returns:
                    str | None: "Class.method" for a method, the function's own
                        name otherwise (a nested function's is its own), or
                        None at module level.
                """
                cur = parents.get(id(node))
                while cur is not None and not isinstance(
                        cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    cur = parents.get(id(cur))
                if cur is None:
                    return None
                owner = parents.get(id(cur))
                return f"{owner.name}.{cur.name}" if isinstance(owner, ast.ClassDef) else cur.name

            # The one allowed form, recorded by the id of its live_settings Name
            allowed = {}
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Name)
                        and node.targets[0].id == "settings"
                        and isinstance(node.value, ast.IfExp)):
                    continue
                ifexp = node.value
                test, body, orelse = ifexp.test, ifexp.body, ifexp.orelse
                if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
                        and test.left.id == "settings" and len(test.ops) == 1
                        and isinstance(test.ops[0], ast.Is)
                        and isinstance(test.comparators[0], ast.Constant)
                        and test.comparators[0].value is None
                        and isinstance(body, ast.Call) and isinstance(body.func, ast.Name)
                        and body.func.id == resolver and not body.args and not body.keywords
                        and isinstance(orelse, ast.Name) and orelse.id == "settings"):
                    allowed[id(body.func)] = enclosing(node)
            # The bare live_settings() call, checked against _DIRECT_RESOLVERS
            direct = {}
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == resolver and not node.args
                        and not node.keywords and id(node.func) not in allowed):
                    direct[id(node.func)] = enclosing(node)

            # A direct call of a saved-defaults reader or writer, recorded by the
            # id of its called name with the function it sits in (a method
            # with its class), checked against its table below
            tracked_calls = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and call_name(node.func) in tracked:
                    tracked_calls[id(node.func)] = qualified(node)

            # Callees handed as pool.submit's first argument, checked below
            submitted = set()
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = call_name(node.func)
                where = f"{mod}:{node.lineno}"
                if name in explicit_args:
                    assert passes_explicitly(node, name), (
                        f"{where} calls {name} without its {explicit_args[name][1]}, or "
                        "with None: it would read config.py rather than the run's settings")
                    explicit_calls[name] = explicit_calls.get(name, 0) + 1
                if (name == "submit" and node.args
                        and call_name(node.args[0]) in takes_settings):
                    target = call_name(node.args[0])
                    submitted.add(id(node.args[0]))
                    assert passes_settings(node, target, 1), (
                        f"{where} submits {target} without settings=settings")
                    continue
                if name in takes_settings:
                    assert passes_settings(node, name, 0), (
                        f"{where} calls {name} without the run's `settings` (the bare "
                        "name, never config.py's reference or another expression)")

            called = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
            # An ImportFrom's alias nodes are checked with it, so skipped below
            from_aliases = {id(a) for n in ast.walk(tree)
                            if isinstance(n, ast.ImportFrom) for a in n.names}
            # This module may refer to its own exempted internals, in their
            # allowed shapes, import only the importable ones, and name nothing
            # else forbidden; it may still spell or shadow none of them
            exempt = self._SAVED_FILE_EXEMPT.get(mod, frozenset())
            mod_forbidden = forbidden - exempt
            import_forbidden = forbidden - (exempt & self._SAVED_FILE_IMPORTABLE)
            # The module's own functions and methods: a call of one may carry
            # text of a print-only path (a page or message builder)
            own_functions = {n.name for n in tree.body
                             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            own_methods = {m.name for c in tree.body if isinstance(c, ast.ClassDef)
                           for m in c.body
                           if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        assert a.name.split(".")[-1] not in no_import, f"{mod} imports {a.name}"
                    continue
                if isinstance(node, ast.ImportFrom):
                    assert (node.module or "").split(".")[-1] not in no_import, (
                        f"{mod} imports {node.module}")
                    for a in node.names:
                        assert a.name not in no_import, f"{mod} imports {a.name}"
                        assert a.name not in import_forbidden, f"{mod} imports {a.name}"
                        # An exempted internal only under its own name
                        assert not (a.name in exempt and a.asname), (
                            f"{mod} imports {a.name} under another name")
                        # Only the plain import is exempt; an alias could call it
                        assert not (a.name == resolver and a.asname), mod
                        assert not (a.name in tracked and a.asname), (
                            f"{mod} imports {a.name} under another name")
                        # ... and only into a module the tables allow to call it
                        assert a.name not in tracked or any(
                            m == mod for m, _ in tracked[a.name]), (
                            f"{mod} imports {a.name}, which _DEFAULTS_READERS / "
                            "_DEFAULTS_WRITERS allow no function of it to call")
                    continue
                if isinstance(node, ast.Constant):
                    if isinstance(node.value, str):
                        assert node.value not in forbidden | {resolver} | set(tracked), (
                            f"{mod}:{node.lineno} spells {node.value!r}")
                    continue
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    assert node.name not in forbidden | {resolver} | set(tracked), (
                        f"{mod}:{node.lineno} shadows {node.name}")
                    continue
                if isinstance(node, ast.Name):
                    name = node.id
                elif isinstance(node, ast.Attribute):
                    name = node.attr
                elif isinstance(node, ast.arg):
                    name = node.arg
                elif isinstance(node, ast.alias):
                    if id(node) in from_aliases:
                        continue
                    # `import x as y` alias nodes
                    name = node.name
                    assert name not in no_import, f"{mod} imports {name}"
                else:
                    continue
                where = f"{mod}:{node.lineno}"
                assert name not in mod_forbidden, f"{where} references {name}"
                if name in exempt:
                    # Read only: never assigned, deleted or taken as a parameter
                    assert isinstance(node, (ast.Name, ast.Attribute)) and isinstance(
                        node.ctx, ast.Load), f"{where} binds, deletes or takes {name}"
                    # A path it may only print (_SAVED_FILE_PRINT_ONLY's shapes)
                    assert (name not in self._SAVED_FILE_PRINT_ONLY
                            or prints_the_path(node, parents, own_functions, own_methods)), (
                        f"{where} uses {name} other than to print it: only config.{name} "
                        "in a logging message call, or its .name or .absolute() as text, "
                        "with every call around it printing or building text")
                    exempt_uses[(mod, name)] += 1
                if name in tracked:
                    table = ("_DEFAULTS_READERS" if name in self._DEFAULTS_READERS
                             else "_DEFAULTS_WRITERS")
                    assert id(node) in tracked_calls, (
                        f"{where} references {name} other than by calling it; "
                        f"only the functions {table} names may, by calling it")
                    func = tracked_calls[id(node)]
                    assert (mod, func) in tracked[name], (
                        f"{where}: {mod}.{func} calls {name}(); only the functions "
                        f"{table} names for it may")
                    key = (name, mod, func)
                    defaults_calls[key] = defaults_calls.get(key, 0) + 1
                    continue
                if name == resolver and id(node) in direct:
                    func = direct[id(node)]
                    assert (mod, func) in self._DIRECT_RESOLVERS, (
                        f"{where}: {mod}.{func} calls live_settings() directly; no live "
                        "function may (_DIRECT_RESOLVERS): a run starts from the saved "
                        "live defaults, and every live function is handed the run's settings")
                    direct_resolutions[(mod, func)] = (
                        direct_resolutions.get((mod, func), 0) + 1)
                elif name == resolver:
                    assert id(node) in allowed, (
                        f"{where} reads live_settings outside the one allowed "
                        "`settings = live_settings() if settings is None else settings`")
                    func = allowed[id(node)]
                    assert (mod, func) in self._LIVE_SETTINGS_RESOLVERS, (
                        f"{where}: {mod}.{func} may not resolve config.py's settings")
                    resolutions[(mod, func)] = resolutions.get((mod, func), 0) + 1
                elif (name in takes_settings and isinstance(node, (ast.Name, ast.Attribute))
                        and id(node) not in called and id(node) not in submitted):
                    raise AssertionError(
                        f"{where}: {name} takes settings and is passed around uncalled")
                elif (name in explicit_args and isinstance(node, (ast.Name, ast.Attribute))
                        and id(node) not in called):
                    raise AssertionError(
                        f"{where}: {name} is passed around uncalled, so its "
                        f"{explicit_args[name][1]} cannot be checked")

        # Each whitelisted entry point and the direct resolver resolves exactly
        # once: two resolutions could straddle a monkeypatch
        assert resolutions == dict.fromkeys(self._LIVE_SETTINGS_RESOLVERS, 1), resolutions
        assert direct_resolutions == dict.fromkeys(self._DIRECT_RESOLVERS, 1), (
            direct_resolutions)
        # Every function the saved-defaults tables allow is found and calls its
        # reader or writer exactly once (so the tables are never vacuous), and
        # no other call exists
        expected = {(name, mod, func): 1
                    for name, allowed in tracked.items() for mod, func in allowed}
        assert defaults_calls == expected, (
            f"_DEFAULTS_READERS / _DEFAULTS_WRITERS: expected {expected}, found "
            f"{defaults_calls}")
        # Both readers and the writer are in use (a table emptied by an edit fails here)
        assert {name for name, _, _ in defaults_calls} == set(tracked)
        # Every exemption is used, so an exemption the module no longer needs fails here
        assert all(exempt_uses.values()), exempt_uses
        # main() hands both run modes the settings it resolved, positionally
        main_tree = trees["main"]
        main_fn = next(n for n in ast.walk(main_tree)
                       if isinstance(n, ast.FunctionDef) and n.name == "main")
        main_calls = {call_name(n.func) for n in ast.walk(main_fn) if isinstance(n, ast.Call)}
        # (so the walk's check cannot pass on a main() missing a dispatch or the resolution)
        assert {"_run_dev", "_run_prod", "_resolve_live_settings"} <= main_calls, main_calls
        # Non-vacuous: _kelly_p and _evaluate_size price at the run's k, and the
        # sizer and enrichment each turn a fraction into a count
        assert explicit_calls.get("time_series_profit_prob", 0) >= 1, explicit_calls
        assert explicit_calls.get("_kelly_p_at", 0) >= 2, explicit_calls
        assert explicit_calls.get("max_affordable_pairs", 0) >= 2, explicit_calls

        # Every live site of the spread rule (and enrichment's bound) goes through
        # its one definition
        assert _function_calls(scanner, "find_time_series_pairs", "time_series_spread_refusal")
        assert _function_calls(scanner, "enrich_with_orderbook_prices",
                               "time_series_spread_refusal")
        assert _function_calls(scanner, "enrich_with_orderbook_prices", "max_kelly_fraction")
        assert _function_calls(scanner, "validate_pair_price", "time_series_spread_refusal")
        assert _function_calls(scanner, "_pair_max_sum", "live_time_series_floor")
        # ... which reaches the floor through the helper _find_entry uses
        assert _function_calls(config, "live_time_series_floor", "min_price_diff_for_gap")

    def _pin_with_server_code(self, monkeypatch, code):
        """
        Run the live-settings pin with code appended to defaults_server's source.

        Only the pin's reading of that one module's source changes; every
        other module is read as it is.

        Args:
            monkeypatch (pytest.MonkeyPatch): Replaces inspect.getsource for
                the call.
            code (str): Python source appended after the module's own.

        Raises:
            AssertionError: If the pin fails on the module with the code added.
        """
        from kalshi_betting import defaults_server

        real_getsource = inspect.getsource

        def getsource(obj):
            text = real_getsource(obj)
            return f"{text}\n\n{code}" if obj is defaults_server else text

        monkeypatch.setattr(inspect, "getsource", getsource)
        self.test_ast_live_path_reads_toggles_only_through_live_settings()

    @pytest.mark.parametrize("mutant", sorted(_PRINT_ONLY_MUTANTS))
    def test_ast_the_print_only_path_is_never_used_to_reach_the_file(self, monkeypatch,
                                                                       mutant):
        # Each mutant reaches the saved file through its path, directly or by
        # turning printed text of it back into a path, so the pin refuses it
        with pytest.raises(AssertionError,
                           match="uses LIVE_DEFAULTS_FILE other than to print it"):
            self._pin_with_server_code(monkeypatch, self._PRINT_ONLY_MUTANTS[mutant])

    def test_ast_the_print_only_shapes_still_pass_when_appended(self, monkeypatch):
        # control — every allowed shape, appended the same way, still passes,
        # so the mutants above fail on the rule and not on the appending: a
        # logging message, str() and html.escape() of .absolute() and .name,
        # an f-string field, and text handed to the module's own page builder
        code = ("def _allowed_shapes():\n"
                "    logging.info('saved to %s', config.LIVE_DEFAULTS_FILE)\n"
                "    return (html.escape(str(config.LIVE_DEFAULTS_FILE.absolute())),\n"
                "            f'{config.LIVE_DEFAULTS_FILE.absolute()}',\n"
                "            _message_html(500, 't',\n"
                "                          html.escape(config.LIVE_DEFAULTS_FILE.name)))\n")
        self._pin_with_server_code(monkeypatch, code)

    def test_ast_the_reports_never_write_the_saved_live_defaults(self):
        # The backtest, its CLI and the dashboard read the saved live defaults for
        # their reports only: none of them names the writer, the file's path, the
        # seed or config's private parse, so none can change what live runs trade
        from kalshi_betting import backtest

        writes = {"save_live_defaults", "_saved_text", "_sync_directory"}
        names = writes | self._SAVED_FILE_INTERNALS
        # A rename in config must fail here, not leave the check naming nothing
        for name in names:
            assert hasattr(config, name), name
        for module in (backtester, backtest, dashboard):
            mod = module.__name__.rsplit(".", 1)[-1]
            for node in ast.walk(ast.parse(inspect.getsource(module))):
                if isinstance(node, ast.Name):
                    found = node.id
                elif isinstance(node, ast.Attribute):
                    found = node.attr
                elif isinstance(node, ast.alias):
                    found = node.name
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    found = node.name
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    found = node.value
                else:
                    continue
                assert found not in names, f"{mod}:{getattr(node, 'lineno', '?')} names {found}"
        # Non-vacuous: both report reads are there, through the one reader
        assert _function_calls(backtester, "_live_settings_for_report", "live_defaults")
        assert _function_calls(backtest, "main", "live_defaults")

    def test_ast_every_sizer_caps_through_pair_size_cap(self):
        # config.pair_size_cap is the ONE definition of a pair's per-trade cap,
        # so the live sizer, enrichment's bound and the backtester agree
        assert _function_calls(strategy, "_evaluate_size", "pair_size_cap")
        assert _function_calls(config, "max_kelly_fraction", "pair_size_cap")
        assert _function_calls(backtester, "_simulate_at_discount", "pair_size_cap")

    def test_ast_add_ons_size_through_held_pair_fraction(self):
        # config.held_pair_fraction is the ONE definition of an add-on's size,
        # read by the sizer where every other gate is, for the pair
        # scanner.pair_held names (by type, never truthiness)
        assert _function_calls(strategy, "_evaluate_size", "held_pair_fraction")
        assert _function_calls(strategy, "_evaluate_size", "pair_held")
        # The portfolio step tells an add-on by the same reader, and only one
        # whose held pair is the spec's own markets and sides ...
        assert _function_calls(strategy, "select_portfolio", "pair_held")
        assert _function_calls(strategy, "select_portfolio", "matches")
        # ... and both finders check an add-on's sides through HeldPair.matches
        assert _function_calls(scanner, "find_time_series_pairs", "matches")
        assert _function_calls(scanner, "find_same_title_pairs", "matches")

    def test_ast_the_live_run_refuses_pairs_on_held_ladders(self):
        # The production run finds the ladders it holds and hands them to both
        # the finder and the portfolio step. A dropped keyword would silently
        # refuse nothing, since both default to an empty set.
        assert _function_calls(main, "_run_prod", "resolve_held_ladders")
        [finder_value] = _keyword_values(main, "_run_prod", "find_time_series_pairs",
                                         "held_ladders")
        assert isinstance(finder_value, ast.Name) and finder_value.id == "held_ladders"
        # The portfolio step gets the same labels, or an empty set when the
        # lookup failed (the finder is not called then, so no time-series spec exists)
        [portfolio_value] = _keyword_values(main, "_run_prod", "select_portfolio",
                                            "held_ladders")
        assert ast.unparse(portfolio_value) in {"held_ladders", "held_ladders or frozenset()"}
        # Both rules read the ladder labels through the one definition
        assert _function_calls(scanner, "find_time_series_pairs", "ladder_keys")
        assert _function_calls(strategy, "select_portfolio", "pair_ladder_keys")
        assert _function_calls(scanner, "pair_ladder_keys", "market_ladder_keys")
        assert _function_calls(scanner, "market_ladder_keys", "ladder_keys")
        assert _function_calls(scanner, "market_ladder_keys", "time_series_group_key")

    def test_ast_the_backtest_reads_ladders_through_the_one_definition(self):
        # The backtest's one-open-trade-per-ladder rule labels each market
        # through the same scanner.ladder_keys the live rule reads, so the two
        # paths cannot disagree about which markets share a ladder
        assert _function_calls(backtester, "_simulate_at_discount", "_ladder_keys_dict")
        assert _function_calls(backtester, "_ladder_keys_dict", "ladder_keys")
        # ... and a question worked out from the market uses the grouping key
        assert _function_calls(backtester, "_ladder_keys_dict", "_ts_group_key")

    def test_keyword_values_finds_every_call(self):
        # The helper the pin above reads: a call without the keyword is a None,
        # so a second, unwired call cannot hide behind the first one
        source = ("def f():\n"
                  "    g(1, k=a)\n"
                  "    g(2)\n")
        module = SimpleNamespace(__name__="snippet")
        values = _keyword_values(module, "f", "g", "k", source=source)
        assert [ast.unparse(v) if v is not None else None for v in values] == ["a", None]


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
    budget = balance * min(config.BUDGET_FRACTION, kelly_f)
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


@pytest.mark.usefixtures("pre_toggle_defaults")
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
    rather than papered over. Worked under pre_toggle_defaults (k 0.75, a 20% cap)."""

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
        assert spec.kelly_fraction == pytest.approx(config.BUDGET_FRACTION)
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


def _ladder_market(ticker: str, event_ticker: str, title: str) -> SimpleNamespace:
    """A market with just the fields the ladder labels are built from."""
    return SimpleNamespace(ticker=ticker, event_ticker=event_ticker, title=title, subtitle="")


def _ladder_spec(market_a, market_b, *, pair_type: str | None = "time_series",
                 ratio: float = 0.10, cost: float = 10.0) -> TradeSpec:
    """A spec on two given markets, ranked by `ratio`, costing `cost` dollars."""
    pair = SimpleNamespace(pair_type=pair_type, market_a=market_a, market_b=market_b)
    return TradeSpec(pair=pair, x=1, y=1, total_cost=cost, total_cost_with_fees=cost,
                     min_payoff=0.10, profit_ratio=0.05, days_to_close=30,
                     monthly_profit_ratio=ratio, kelly_p=0.90, kelly_fraction=0.10)


# One question asked at four deadlines, all listed in one event
_STAR = "Will SpaceX launch another Starship by %s?"
_R1, _R2, _R3, _R4 = (_ladder_market(f"STAR-{d}", "KXSTAR-14", _STAR % f"March {d}, 2026")
                      for d in (1, 10, 20, 30))
# A question on its own event, sharing no label with the Starship ladder
_OTHER_A = _ladder_market("RAIN-1", "KXRAIN-1", "Will it rain in NYC by March 1, 2026?")
_OTHER_B = _ladder_market("RAIN-2", "KXRAIN-2", "Will it rain in NYC by March 11, 2026?")


class TestSelectPortfolioLadders:
    """select_portfolio picks at most one time-series trade per ladder (one
    question at several deadlines): two markets share a ladder when they share
    an event or ask the same question once the dates are removed. Held
    positions count, and so do specs picked earlier in the run."""

    def test_two_time_series_specs_on_one_ladder_take_only_the_better(self, caplog):
        best = _ladder_spec(_R1, _R3, ratio=0.20)
        second = _ladder_spec(_R2, _R4, ratio=0.10)
        # The two share no ticker, so only the ladder rule can stop the second
        with caplog.at_level(logging.INFO):
            assert select_portfolio([second, best], 100_000) == [best]
        assert ("Time-series trades skipped because the account already holds, or "
                "this run already picked, a trade on the same ladder: 1") in caplog.text

    def test_specs_on_different_ladders_are_both_taken(self, caplog):
        star = _ladder_spec(_R1, _R3, ratio=0.20)
        rain = _ladder_spec(_OTHER_A, _OTHER_B, ratio=0.10)
        with caplog.at_level(logging.INFO):
            assert select_portfolio([rain, star], 100_000) == [star, rain]
        # Silent at zero
        assert "on the same ladder" not in caplog.text

    def test_a_held_event_blocks_a_time_series_spec(self):
        star = _ladder_spec(_R1, _R3, ratio=0.20)
        rain = _ladder_spec(_OTHER_A, _OTHER_B, ratio=0.10)
        held = frozenset({("event", "KXSTAR-14")})
        assert select_portfolio([star, rain], 100_000, held_ladders=held) == [rain]

    def test_a_held_event_of_market_b_alone_blocks_a_time_series_spec(self):
        rain = _ladder_spec(_OTHER_A, _OTHER_B, ratio=0.10)
        # Only the later market is in the held event
        held = frozenset({("event", "KXRAIN-2")})
        assert not held & scanner.market_ladder_keys(_OTHER_A)
        assert select_portfolio([rain], 100_000, held_ladders=held) == []

    def test_a_picked_spec_claims_the_ladders_of_its_market_b_too(self):
        first = _ladder_spec(_OTHER_A, _OTHER_B, ratio=0.20)
        # A different question whose later market shares the rain spec's later event
        snow_a = _ladder_market("SNOW-1", "KXSNOW-1", "Will it snow in NYC by March 1, 2026?")
        snow_b = _ladder_market("SNOW-2", "KXRAIN-2", "Will it snow in NYC by March 11, 2026?")
        second = _ladder_spec(snow_a, snow_b, ratio=0.10)
        shared = scanner.pair_ladder_keys(first.pair) & scanner.pair_ladder_keys(second.pair)
        assert shared == frozenset({("event", "KXRAIN-2")})
        assert select_portfolio([first, second], 100_000) == [first]

    def test_a_spec_of_unknown_type_is_never_blocked(self):
        # Only the exact type "time_series" is refused; anything else is
        # treated as a same-title pair
        ts = _ladder_spec(_R1, _R3, ratio=0.20)
        odd = _ladder_spec(_R2, _R4, pair_type=None, ratio=0.10)
        assert select_portfolio([ts, odd], 100_000) == [ts, odd]

    def test_a_held_question_blocks_a_time_series_spec(self):
        # The same question listed in another event, one we hold a position in
        elsewhere = _ladder_market("STAR-APR", "KXSTAR-15", _STAR % "April 9, 2026")
        held = scanner.market_ladder_keys(elsewhere)
        assert ("event", "KXSTAR-14") not in held
        star = _ladder_spec(_R1, _R3, ratio=0.20)
        # The only label the two share is the question
        assert {kind for kind, _ in held & scanner.pair_ladder_keys(star.pair)} == {"question"}
        assert select_portfolio([star], 100_000, held_ladders=held) == []
        # control: holding a position on an unrelated ladder blocks nothing
        other = scanner.market_ladder_keys(_OTHER_A)
        assert select_portfolio([star], 100_000, held_ladders=other) == [star]

    def test_a_same_title_spec_is_never_blocked(self):
        st = _ladder_spec(_R1, _R3, pair_type="same_title", ratio=0.20)
        held = scanner.pair_ladder_keys(st.pair)
        assert select_portfolio([st], 100_000, held_ladders=held) == [st]

    def test_a_same_title_spec_picked_first_blocks_a_later_time_series_spec(self):
        st = _ladder_spec(_R1, _R3, pair_type="same_title", ratio=0.20)
        ts = _ladder_spec(_R2, _R4, ratio=0.10)
        assert select_portfolio([ts, st], 100_000) == [st]

    def test_a_time_series_spec_picked_first_does_not_block_a_same_title_spec(self):
        ts = _ladder_spec(_R1, _R3, ratio=0.20)
        st = _ladder_spec(_R2, _R4, pair_type="same_title", ratio=0.10)
        assert select_portfolio([st, ts], 100_000) == [ts, st]

    def test_a_spec_skipped_for_cash_leaves_its_ladder_free(self):
        # The better spec does not fit the balance, so it never takes the ladder
        too_dear = _ladder_spec(_R1, _R3, ratio=0.20, cost=600.0)
        fits = _ladder_spec(_R2, _R4, ratio=0.10, cost=100.0)
        assert select_portfolio([too_dear, fits], 50_000) == [fits]

    def test_no_held_ladders_is_the_same_as_an_empty_set(self):
        specs = [_ladder_spec(_R1, _R3, ratio=0.20), _ladder_spec(_R2, _R4, ratio=0.10),
                 _ladder_spec(_OTHER_A, _OTHER_B, ratio=0.05)]
        assert select_portfolio(specs, 100_000) == select_portfolio(
            specs, 100_000, held_ladders=frozenset())


def _add_on_spec(market_a, market_b, *, ratio: float = 0.10) -> TradeSpec:
    """A time-series spec that adds to the pair the account holds on these markets."""
    spec = _ladder_spec(market_a, market_b, ratio=ratio)
    spec.pair.held = HeldPair(sides=tuple(sorted(((market_a.ticker, "yes"),
                                                  (market_b.ticker, "no")))),
                              count=30.0, cost_dollars=18.9, account_value_dollars=168.0)
    return spec


class TestSelectPortfolioAddOns:
    """An add-on to a held pair is exempt from the held ladders (they are its
    own: scanner.held_pairs adds only to a pair no other held market shares a
    ladder with) but not from the picks of this run. Every ordinary
    time-series spec on a held ladder is still refused."""

    _LADDER_LINE = ("Time-series trades skipped because the account already holds, or "
                    "this run already picked, a trade on the same ladder: ")

    def test_an_add_on_on_its_own_held_ladder_is_taken(self, caplog):
        add_on = _add_on_spec(_R1, _R3)
        held = scanner.pair_ladder_keys(add_on.pair)
        with caplog.at_level(logging.INFO):
            assert select_portfolio([add_on], 100_000, held_ladders=held) == [add_on]
        assert "on the same ladder" not in caplog.text

    def test_a_same_title_pick_earlier_in_the_run_blocks_an_add_on(self, caplog):
        st = _ladder_spec(_R2, _R4, pair_type="same_title", ratio=0.20)
        add_on = _add_on_spec(_R1, _R3, ratio=0.10)
        held = scanner.pair_ladder_keys(add_on.pair)
        with caplog.at_level(logging.INFO):
            assert select_portfolio([add_on, st], 100_000, held_ladders=held) == [st]
        assert self._LADDER_LINE + "1" in caplog.text

    def test_an_ordinary_spec_on_a_held_ladder_is_still_refused(self, caplog):
        add_on = _add_on_spec(_R1, _R3, ratio=0.10)
        ordinary = _ladder_spec(_R2, _R4, ratio=0.20)
        held = scanner.pair_ladder_keys(add_on.pair)
        with caplog.at_level(logging.INFO):
            assert select_portfolio([ordinary, add_on], 100_000,
                                    held_ladders=held) == [add_on]
        assert self._LADDER_LINE + "1" in caplog.text

    def test_a_held_pair_naming_other_markets_is_no_exemption(self, caplog):
        # A spec on the held Starship ladder carrying a HeldPair of two other
        # markets is not an add-on to it, so the held ladders still refuse it
        spec = _ladder_spec(_R1, _R3, ratio=0.10)
        spec.pair.held = HeldPair(sides=(("ZZ-1", "yes"), ("ZZ-2", "no")), count=1.0,
                                  cost_dollars=1.0, account_value_dollars=100.0)
        held = scanner.pair_ladder_keys(spec.pair)
        with caplog.at_level(logging.INFO):
            assert select_portfolio([spec], 100_000, held_ladders=held) == []
        assert self._LADDER_LINE + "1" in caplog.text

    def test_a_held_pair_bought_the_other_way_round_is_no_exemption(self):
        # The spec's own two markets, but held NO on market A and YES on B:
        # buying it would close the held pair, not add to it
        spec = _ladder_spec(_R1, _R3, ratio=0.10)
        spec.pair.held = HeldPair(sides=((_R1.ticker, "no"), (_R3.ticker, "yes")),
                                  count=30.0, cost_dollars=18.9, account_value_dollars=168.0)
        held = scanner.pair_ladder_keys(spec.pair)
        assert select_portfolio([spec], 100_000, held_ladders=held) == []

    def test_an_add_on_picked_first_claims_its_ladder(self):
        # Two add-ons on one ladder: held_pairs never builds this, but the
        # second must still wait behind the first's pick
        first = _add_on_spec(_R1, _R3, ratio=0.20)
        second = _add_on_spec(_R2, _R4, ratio=0.10)
        held = scanner.pair_ladder_keys(first.pair) | scanner.pair_ladder_keys(second.pair)
        assert select_portfolio([second, first], 100_000, held_ladders=held) == [first]


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
            assert _kelly_p(enriched, live_settings()) < 1.0
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
        assert _kelly_p(enriched, live_settings()) == pytest.approx(
            time_series_profit_prob(enriched.pA, enriched.pB)
        )
        assert _kelly_p(enriched, live_settings()) < 1.0


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
            kelly_fraction=0.20, budget_dollars=940.0,
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
