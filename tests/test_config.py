"""Tests for config.py fee helpers, the time-series probability model, the
leg-side tuples, the deadline-gap tier (with the backtest's spread band and
tier-floors switch), the live toggles (LiveSettings and its helpers), the
values config.py ships, conftest's apply_pre_toggle_defaults, the V2 order
path's self-trade-prevention value, the order-write pacer's budget, and
PROJECT_ROOT."""
import ast
import dataclasses
import importlib
import logging
import math
import pathlib
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from kalshi_betting import backtester, config, scanner
from kalshi_betting.config import (
    MAX_DEADLINE_GAP_DAYS,
    MIN_PRICE_DIFF_LONG_GAP,
    MIN_PRICE_DIFF_SHORT_GAP,
    PRICE_EPSILON,
    PROJECT_ROOT,
    SAME_TITLE_LEG_SIDES,
    SHORT_DEADLINE_GAP_DAYS,
    SPREAD_ABOVE_CEILING,
    SPREAD_BELOW_FLOOR,
    SPREAD_NOT_POSITIVE,
    TAKER_FEE_RATE,
    TIME_SERIES_LEG_SIDES,
    LiveSettings,
    fee_leg_exact,
    fee_per_pair_approx,
    live_settings,
    max_affordable_pairs,
    max_kelly_fraction,
    min_price_diff_for_gap,
    pair_size_cap,
    time_series_profit_prob,
    time_series_spread_refusal,
)
from kalshi_betting.scanner import leg_prices
from kalshi_betting.strategy import compute_trade

# Modules, never their Test* classes, which pytest would collect twice here
from . import test_backtester as _tb
from . import test_scanner as _ts
from .conftest import apply_pre_toggle_defaults


def _settings(tier_floors=True, spread_band=(0.0, 1.0), interval_discount=0.75, size_cap=0.20,
              same_title_size_cap=1.0, categories=None, tags=None):
    """A LiveSettings with every field named, defaulting to the pre-toggle
    values conftest's apply_pre_toggle_defaults pins."""
    return LiveSettings(tier_floors=tier_floors, spread_band=spread_band,
                        interval_discount=interval_discount, size_cap=size_cap,
                        same_title_size_cap=same_title_size_cap,
                        categories=categories, tags=tags)


class TestProjectRoot:
    def test_resolves_to_repo_root(self):
        assert (PROJECT_ROOT / "kalshi_betting" / "config.py").exists()

    def test_derived_from_file_not_hardcoded(self):
        # PROJECT_ROOT must track config.py's actual location (two levels up:
        # kalshi_betting/config.py -> kalshi_betting/ -> repo root), not a
        # hardcoded absolute path baked in at some point in time.
        assert PROJECT_ROOT == pathlib.Path(config.__file__).resolve().parent.parent


class TestPackagingGuards:
    def test_no_shadow_requirements_file(self):
        # A second dependency list inside the package once pinned the SDK to
        # 3.13.0 — the exact version whose metadata requires Python >= 3.13 and
        # breaks `pip install -e ".[dev]"` on 3.11. pyproject.toml is the single
        # source of truth; this guards against the shadow file reappearing.
        assert not (PROJECT_ROOT / "kalshi_betting" / "requirements.txt").exists()

    def test_sdk_pin_is_3_2_0(self):
        # kalshi-python-sync must stay pinned at 3.2.0: every newer release
        # requires Python >= 3.13, which breaks install (and CI) on 3.11.
        import tomllib

        with open(PROJECT_ROOT / "pyproject.toml", "rb") as fh:
            pyproject = tomllib.load(fh)
        deps = pyproject["project"]["dependencies"]
        sdk_pins = [d for d in deps if d.startswith("kalshi-python-sync")]
        assert sdk_pins == ["kalshi-python-sync==3.2.0"]


class TestFeeLegExact:
    def test_ceiling_rounding(self):
        # ceil(0.07 * 1 * 0.5 * 0.5 * 100) = ceil(1.75) = 2 → 0.02
        assert fee_leg_exact(1, 0.5) == 0.02

    def test_large_n(self):
        # The TRUE fee is exactly 175¢ (0.07 * 100 * 0.5 * 0.5 * 100 = 175).
        # Binary float noise (175.00000000000003) must NOT bump the ceiling to
        # 176¢ — fee_leg_exact rounds before applying the ceiling, matching the
        # fee Kalshi actually charges.
        assert fee_leg_exact(100, 0.5) == 1.75

    def test_minimum_fee_one_cent(self):
        # At extreme prices, fee rounds up to at least 0.01
        assert fee_leg_exact(1, 0.01) == 0.01

    def test_formula_matches_definition(self):
        # Expected mirrors the definition ceil(rate*n*p*(1-p)*100)/100 computed
        # on the true value — round before ceil so float noise on exact-cent
        # amounts (e.g. n=20, p=0.5 → 35¢) doesn't inflate the expectation.
        for n in [1, 5, 20]:
            for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
                expected = math.ceil(round(TAKER_FEE_RATE * n * p * (1 - p) * 100, 6)) / 100
                assert fee_leg_exact(n, p) == pytest.approx(expected)

    def test_symmetric_in_price(self):
        # fee_leg_exact(n, p) == fee_leg_exact(n, 1-p) because p*(1-p) is symmetric
        assert fee_leg_exact(10, 0.3) == fee_leg_exact(10, 0.7)
        assert fee_leg_exact(10, 0.2) == fee_leg_exact(10, 0.8)


class TestFeePairApprox:
    def test_formula_matches_definition(self):
        price_a, price_b = 0.35, 0.45
        expected = TAKER_FEE_RATE * (price_a * (1 - price_a) + price_b * (1 - price_b))
        assert fee_per_pair_approx(price_a, price_b) == pytest.approx(expected)

    def test_is_underestimate_vs_exact(self):
        # The approximation should be <= the sum of two exact leg fees at n=1,
        # because ceiling rounding always rounds up.
        for price_a, price_b in [(0.3, 0.4), (0.2, 0.5), (0.45, 0.35)]:
            approx = fee_per_pair_approx(price_a, price_b)
            exact_sum = fee_leg_exact(1, price_a) + fee_leg_exact(1, price_b)
            assert approx <= exact_sum + 1e-9, (
                f"approx {approx} > exact {exact_sum} for price_a={price_a} price_b={price_b}"
            )

    def test_symmetric(self):
        assert fee_per_pair_approx(0.3, 0.4) == pytest.approx(fee_per_pair_approx(0.4, 0.3))

    def test_side_agnostic_keyword_names(self):
        # The parameters are leg-neutral (price_a/price_b): the same formula
        # prices a same-title pair on (nA, pB) and a time-series pair on (pA, nB)
        assert fee_per_pair_approx(price_a=0.30, price_b=0.40) == pytest.approx(
            fee_per_pair_approx(0.30, 0.40)
        )


class TestTimeSeriesProfitProb:
    """p = 1 - k * max(0, pB - pA): one minus the believed fraction k of the
    market-implied probability that the event first happens between the two
    deadlines (the single loss cell of a YES-on-earlier / NO-on-later pair)."""

    @pytest.mark.usefixtures("pre_toggle_defaults")
    def test_flow_through_fixture(self):
        # pA 0.30, pB 0.60 → p = 1 - 0.75 * 0.30 = 0.775 (k 0.75, pre-toggle)
        assert time_series_profit_prob(0.30, 0.60) == pytest.approx(0.775)

    def test_matches_definition_from_constant(self):
        for pA, pB in [(0.10, 0.25), (0.30, 0.60), (0.40, 0.55), (0.30, 0.70)]:
            expected = 1.0 - config.TIME_SERIES_INTERVAL_PROB_DISCOUNT * (pB - pA)
            assert time_series_profit_prob(pA, pB) == pytest.approx(expected)

    def test_clamps_to_one_when_earlier_is_pricier(self):
        # A pricier earlier contract is never a candidate; reachable only from
        # reporting code, where it must model as riskless, not as p > 1
        assert time_series_profit_prob(0.60, 0.30) == 1.0
        assert time_series_profit_prob(0.50, 0.50) == 1.0

    def test_discount_of_one_is_market_implied(self, monkeypatch):
        # k = 1 takes the market at face value: p = 1 - (pB - pA). Under this
        # model Kelly is <= 0 for every pair (see test_strategy's parity class).
        monkeypatch.setattr(config, "TIME_SERIES_INTERVAL_PROB_DISCOUNT", 1.0)
        assert time_series_profit_prob(0.30, 0.60) == pytest.approx(0.70)

    def test_discount_of_zero_ignores_the_gap(self, monkeypatch):
        monkeypatch.setattr(config, "TIME_SERIES_INTERVAL_PROB_DISCOUNT", 0.0)
        assert time_series_profit_prob(0.30, 0.60) == 1.0

    def test_discount_constant_value_and_range(self):
        # Pinned so a retune is visible in review; within (0, 1], the range
        # LiveSettings accepts
        assert config.TIME_SERIES_INTERVAL_PROB_DISCOUNT == 0.80
        assert 0.0 < config.TIME_SERIES_INTERVAL_PROB_DISCOUNT <= 1.0

    @pytest.mark.usefixtures("pre_toggle_defaults")
    def test_explicit_k_overrides_the_constant(self):
        # The backtester's calibration sweep passes one k per simulation; the
        # override must win over the config constant — 1 - 0.50 * 0.30 = 0.85 —
        # without mutating it, since the live sizer keeps reading it.
        assert time_series_profit_prob(0.30, 0.60, k=0.50) == pytest.approx(0.85)
        assert time_series_profit_prob(0.30, 0.60, k=1.0) == pytest.approx(0.70)
        assert config.TIME_SERIES_INTERVAL_PROB_DISCOUNT == 0.75

    def test_k_none_is_identical_to_omitting_it(self):
        # k=None reads the config constant, so the backtest's default point
        # prices exactly as live sizing does at config.py's k
        for pA, pB in [(0.10, 0.25), (0.30, 0.60), (0.40, 0.55), (0.60, 0.30)]:
            assert time_series_profit_prob(pA, pB, k=None) == time_series_profit_prob(pA, pB)

    def test_constant_read_at_call_time_and_only_when_k_is_omitted(self, monkeypatch):
        # The None sentinel must resolve inside the body rather than binding at
        # def time: a monkeypatched constant still governs an override-free
        # call (1 - 0.50 * 0.30 = 0.85)...
        monkeypatch.setattr(config, "TIME_SERIES_INTERVAL_PROB_DISCOUNT", 0.50)
        assert time_series_profit_prob(0.30, 0.60) == pytest.approx(0.85)
        # ...and is ignored entirely once k is supplied (1 - 0.75 * 0.30)
        assert time_series_profit_prob(0.30, 0.60, k=0.75) == pytest.approx(0.775)


class TestLegSideTuples:
    def test_same_title_buys_no_on_a_yes_on_b(self):
        assert SAME_TITLE_LEG_SIDES == ("no", "yes")

    def test_time_series_buys_yes_on_earlier_no_on_later(self):
        assert TIME_SERIES_LEG_SIDES == ("yes", "no")

    def test_each_pair_type_has_exactly_one_no_leg(self):
        # The trader submits "the NO leg" first and unwinds it — every pair
        # type must have exactly one, and one YES leg to hedge it
        for sides in (SAME_TITLE_LEG_SIDES, TIME_SERIES_LEG_SIDES):
            assert sorted(sides) == ["no", "yes"]


class TestV2SelfTradePrevention:
    def test_is_one_of_the_two_values_the_endpoint_accepts(self):
        # The V2 create-order endpoint requires self_trade_prevention_type and
        # accepts only these two values; any other is rejected with HTTP 400
        assert config.V2_SELF_TRADE_PREVENTION_TYPE in {"taker_at_cross", "maker"}


class TestOrderWriteBudget:
    """trader's write pacer, built from ORDER_WRITES_PER_SECOND and
    ORDER_WRITE_BURST, must stay inside the account's write budget, or the
    exchange answers the excess with HTTP 429 and does not process it."""

    # Kalshi's Basic usage tier: a write bucket of 100 tokens that refills at
    # 100 tokens a second, and an order or transfer POST costs 10 tokens.
    BASIC_TIER_WRITE_BUCKET_CAPACITY = 100
    BASIC_TIER_WRITE_REFILL_PER_SECOND = 100
    TOKENS_PER_WRITE = 10

    def test_the_burst_fits_the_basic_tier_bucket(self):
        assert (config.ORDER_WRITE_BURST * self.TOKENS_PER_WRITE
                <= self.BASIC_TIER_WRITE_BUCKET_CAPACITY)

    def test_the_rate_fits_the_basic_tier_refill(self):
        assert (config.ORDER_WRITES_PER_SECOND * self.TOKENS_PER_WRITE
                <= self.BASIC_TIER_WRITE_REFILL_PER_SECOND)

    def test_the_pacer_admits_writes_at_all(self):
        assert config.ORDER_WRITE_BURST >= 1
        assert config.ORDER_WRITES_PER_SECOND > 0


class TestMinPriceDiffForGap:
    def test_short_tier_from_zero_gap(self):
        # Same-day deadlines are the tightest correlation — short tier applies
        assert min_price_diff_for_gap(0) == MIN_PRICE_DIFF_SHORT_GAP

    def test_short_tier_boundary_inclusive(self):
        # A gap of exactly SHORT_DEADLINE_GAP_DAYS (15) still uses the 15% tier
        assert min_price_diff_for_gap(SHORT_DEADLINE_GAP_DAYS) == MIN_PRICE_DIFF_SHORT_GAP

    def test_long_tier_starts_at_sixteen_days(self):
        assert min_price_diff_for_gap(SHORT_DEADLINE_GAP_DAYS + 1) == MIN_PRICE_DIFF_LONG_GAP

    def test_long_tier_at_max_gap(self):
        # 30 days is still an allowed gap (MAX_DEADLINE_GAP_DAYS) — long tier
        assert min_price_diff_for_gap(MAX_DEADLINE_GAP_DAYS) == MIN_PRICE_DIFF_LONG_GAP

    def test_tier_values(self):
        # The tiers the strategy is specified against: 15% short, 30% long
        assert MIN_PRICE_DIFF_SHORT_GAP == 0.15
        assert MIN_PRICE_DIFF_LONG_GAP == 0.30


def _pre_band_tier(gap_days: int) -> float:
    """min_price_diff_for_gap exactly as it stood before the band keyword
    existed — the oracle the neutrality tests compare against. Written out
    here rather than derived from the helper under test, so a change to the
    helper cannot also change what it is checked against."""
    if gap_days <= SHORT_DEADLINE_GAP_DAYS:
        return MIN_PRICE_DIFF_SHORT_GAP
    return MIN_PRICE_DIFF_LONG_GAP


class TestTimeSeriesSpreadBand:
    """The backtest's time-series spread band (floor, ceiling) on pB - pA.

    The floor rides on min_price_diff_for_gap's backtest-only spread_min
    keyword (max of tier and floor), the ceiling on time_series_spread_too_wide,
    and time_series_spread_band resolves and validates the pair. The live path
    passes none of it — every assertion that the no-band call is unchanged is
    what keeps live trading byte-identical while the explorer is built.
    """

    def test_no_band_is_the_pre_band_tier_in_value_and_type(self):
        # Every gap the live path can hand in (and some it cannot: negatives
        # and far past MAX_DEADLINE_GAP_DAYS), by omission AND by an explicit
        # None — the same value, the same type, the very same constant object.
        for g in range(-5, 400):
            expected = _pre_band_tier(g)
            for got in (min_price_diff_for_gap(g), min_price_diff_for_gap(g, spread_min=None)):
                assert type(got) is type(expected)
                assert got == expected
                assert got is expected

    def test_default_band_is_no_band(self):
        # (0.0, 1.0): a zero floor never beats a tier, and no spread a price in
        # [0, 1] can produce sits above a 1.0 ceiling — so a backtest run on the
        # default band applies exactly the live rule.
        assert config.BACKTEST_DEFAULT_SPREAD_BAND == (0.0, 1.0)
        lo, hi = config.time_series_spread_band()
        assert (lo, hi) == (0.0, 1.0)
        for g in range(-5, 400):
            got = min_price_diff_for_gap(g, spread_min=lo)
            assert type(got) is float
            assert got == _pre_band_tier(g)
        for spread in (0.0, 0.15, 0.60, 0.9999, 1.0):
            assert not config.time_series_spread_too_wide(spread, hi)

    def test_floor_is_the_larger_of_tier_and_band_floor(self):
        # Short tier (0.15): a 0.30 floor raises it, a 0.10 floor is inert, and
        # a floor equal to the tier leaves it where it was
        assert min_price_diff_for_gap(7, spread_min=0.30) == 0.30
        assert min_price_diff_for_gap(7, spread_min=0.10) == MIN_PRICE_DIFF_SHORT_GAP
        assert min_price_diff_for_gap(7, spread_min=MIN_PRICE_DIFF_SHORT_GAP) == MIN_PRICE_DIFF_SHORT_GAP
        # Long tier (0.30): a 0.20 floor is inert, a 0.40 floor raises it
        assert min_price_diff_for_gap(20, spread_min=0.20) == MIN_PRICE_DIFF_LONG_GAP
        assert min_price_diff_for_gap(20, spread_min=0.40) == 0.40
        # Across both tiers' boundaries and the whole sweep's floor grid
        for g in (0, SHORT_DEADLINE_GAP_DAYS, SHORT_DEADLINE_GAP_DAYS + 1, MAX_DEADLINE_GAP_DAYS):
            for floor in config.SPREAD_BAND_SWEEP_FLOORS:
                assert min_price_diff_for_gap(g, spread_min=floor) == max(_pre_band_tier(g), floor)

    def test_ceiling_boundary_absorbs_float_noise_on_the_keep_side(self):
        too_wide = config.time_series_spread_too_wide
        # Exactly on the ceiling, and well inside it: kept
        assert not too_wide(0.60, 0.60)
        assert not too_wide(0.30, 0.60)
        # A spread sitting on the documented bound that float arithmetic nudges
        # one ULP above it (TS-09) — the case is only meaningful if the nudge
        # is real, so assert that first
        spread = 0.90 - 0.30
        assert spread > 0.60
        assert not too_wide(spread, 0.60)
        # Two millionths above: past PRICE_EPSILON (1e-6), a genuine refusal
        assert too_wide(0.600002, 0.60)
        assert too_wide(0.61, 0.60)

    def test_no_ceiling_is_never_too_wide(self):
        for spread in (0.0, 0.60, 1.0, 5.0):
            assert config.time_series_spread_too_wide(spread, None) is False

    def test_default_is_resolved_at_call_time(self, monkeypatch):
        # The default is looked up inside the body, not bound at def time, so
        # a patched constant governs an override-free call...
        monkeypatch.setattr(config, "BACKTEST_DEFAULT_SPREAD_BAND", (0.30, 0.60))
        assert config.time_series_spread_band() == (0.30, 0.60)
        assert config.time_series_spread_band(None) == (0.30, 0.60)
        # ...an explicit band still wins over it...
        assert config.time_series_spread_band((0.20, 0.90)) == (0.20, 0.90)
        # ...and the tier helper never reads it, so the live path never sees
        # the backtest's band
        assert min_price_diff_for_gap(7) is MIN_PRICE_DIFF_SHORT_GAP
        assert min_price_diff_for_gap(20) is MIN_PRICE_DIFF_LONG_GAP
        # A patched default is validated like an override
        monkeypatch.setattr(config, "BACKTEST_DEFAULT_SPREAD_BAND", (0.60, 0.30))
        with pytest.raises(ValueError):
            config.time_series_spread_band()

    def test_band_is_returned_as_a_float_tuple(self):
        # (0, 1), (0.0, 1.0) and (-0.0, 1.0) must label the same scenario —
        # a -0.0 floor passes 0.0 <= -0.0 and compares equal to 0.0, so only
        # its sign and its printed form could tell it apart
        for band in ((0, 1), [0, 1], (0.0, 1.0), (-0.0, 1.0)):
            resolved = config.time_series_spread_band(band)
            assert resolved == (0.0, 1.0)
            assert type(resolved) is tuple
            assert all(type(x) is float for x in resolved)
            assert math.copysign(1.0, resolved[0]) == 1.0
            # the form a log line's %g-%g would print
            assert f"{resolved[0]:g}-{resolved[1]:g}" == "0-1"

    def test_validation_is_tier_agnostic(self):
        # A band whose ceiling sits below a tier validates — the helper checks
        # floor < ceiling only, never the EFFECTIVE band max(tier, floor)..ceiling
        # — so a caller accepting an operator-typed ceiling must warn itself.
        # (0.20, 0.25) empties the long tier; (0.0, 0.10) empties both.
        assert config.time_series_spread_band((0.20, 0.25)) == (0.20, 0.25)
        assert min_price_diff_for_gap(20, spread_min=0.20) > 0.25 + config.PRICE_EPSILON
        assert min_price_diff_for_gap(7, spread_min=0.20) < 0.25
        assert config.time_series_spread_band((0.0, 0.10)) == (0.0, 0.10)
        for g in (0, SHORT_DEADLINE_GAP_DAYS, SHORT_DEADLINE_GAP_DAYS + 1, MAX_DEADLINE_GAP_DAYS):
            assert min_price_diff_for_gap(g, spread_min=0.0) > 0.10 + config.PRICE_EPSILON

    @pytest.mark.parametrize("band", [
        (-0.01, 0.60),        # floor below zero
        (0.30, 1.01),         # ceiling above one
        (0.60, 0.60),         # floor == ceiling: an empty band
        (0.60, 0.30),         # floor above ceiling
        (math.nan, 0.60),     # NaN fails every comparison
        (0.30, math.nan),
        (0.10, 0.20, 0.30),   # not a pair
        (0.30,),
    ])
    def test_invalid_band_raises(self, band):
        with pytest.raises(ValueError):
            config.time_series_spread_band(band)

    def test_sweep_grid_is_36_valid_bands_including_the_default(self):
        floors, ceilings = config.SPREAD_BAND_SWEEP_FLOORS, config.SPREAD_BAND_SWEEP_CEILINGS
        assert len(floors) == 6 and len(ceilings) == 6
        # Every combination validates (no floor reaches the lowest ceiling)...
        bands = {config.time_series_spread_band((f, c)) for f in floors for c in ceilings}
        assert len(bands) == 36
        # ...and the default band is a member, so a default run's sweep adds
        # no 37th band: 36 bands x 13 k = 468 scenarios
        assert config.time_series_spread_band() in bands
        assert len(bands) * len(config.INTERVAL_DISCOUNT_SWEEP) == 468
        # No grid band empties a tier: every ceiling clears both tiers even
        # after the largest floor raises them
        assert min(ceilings) > max(floors)
        assert min(ceilings) > max(MIN_PRICE_DIFF_SHORT_GAP, MIN_PRICE_DIFF_LONG_GAP)


class TestTierFloorsOff:
    """min_price_diff_for_gap's BACKTEST-only tier_floors keyword.

    Only an explicit False drops the deadline-gap tier, leaving the band floor
    alone — the rule the backtest's tier-floors-off family is simulated under.
    Every existing call, the live path's included (which may pass no keyword
    at all), keeps its value, its type and its very object.
    """

    def test_the_default_and_an_explicit_true_are_the_pre_band_tier(self):
        # Every gap the live path can hand in (and some it cannot), by
        # omission AND by an explicit True: the very same constant object
        for g in range(-5, 400):
            expected = _pre_band_tier(g)
            for got in (min_price_diff_for_gap(g),
                        min_price_diff_for_gap(g, tier_floors=True),
                        min_price_diff_for_gap(g, spread_min=None, tier_floors=True)):
                assert got is expected
            # ... and a band floor still layers on the tier exactly as before
            for floor in config.SPREAD_BAND_SWEEP_FLOORS:
                assert min_price_diff_for_gap(g, spread_min=floor, tier_floors=True) == max(
                    expected, floor)

    def test_off_is_the_band_floor_alone(self):
        # Every grid floor at every tier boundary: the floor itself, whether
        # it sits below a tier (0, 0.20, 0.25) or above one — never the tier
        for g in (0, SHORT_DEADLINE_GAP_DAYS, SHORT_DEADLINE_GAP_DAYS + 1, MAX_DEADLINE_GAP_DAYS):
            for floor in config.SPREAD_BAND_SWEEP_FLOORS:
                got = min_price_diff_for_gap(g, spread_min=floor, tier_floors=False)
                assert got == floor and type(got) is float
            # No floor at all is a 0.0 floor, as a float
            got = min_price_diff_for_gap(g, tier_floors=False)
            assert got == 0.0 and type(got) is float

    @pytest.mark.parametrize("not_false", [None, 0, MagicMock()])
    def test_only_false_drops_the_tier(self, not_false):
        # A 20-day gap takes the 0.30 tier; a 0.20 floor sits below it. Any
        # value but False itself falls back to the live rule.
        assert min_price_diff_for_gap(20, spread_min=0.2, tier_floors=not_false) == (
            MIN_PRICE_DIFF_LONG_GAP)
        assert min_price_diff_for_gap(20, spread_min=0.2, tier_floors=False) == 0.2

    def test_tier_floors_is_keyword_only(self):
        # A third positional argument cannot throw the switch by accident
        with pytest.raises(TypeError):
            min_price_diff_for_gap(20, 0.2, False)


class TestMaxAffordablePairs:
    """max_affordable_pairs is the single budget -> contracts definition shared
    by the scanner's depth cap and strategy.compute_trade's sizing. The upper-
    bound property below is what lets enrichment price a pair at a size the
    sizer can never exceed."""

    def test_floors_rather_than_rounds(self):
        # $10.00 at a $0.90 pair sum buys 11.11 pairs -> 11, never 12
        assert max_affordable_pairs(5_000, 0.90, 1.0) == 55
        assert max_affordable_pairs(1_000, 0.90, 1.0) == 11

    def test_defaults_to_budget_fraction(self):
        assert (max_affordable_pairs(100_000, 0.50)
                == max_affordable_pairs(100_000, 0.50, config.BUDGET_FRACTION))

    @pytest.mark.usefixtures("pre_toggle_defaults")
    def test_fraction_default_resolves_at_call_time(self, monkeypatch):
        # Bound as a default argument this would freeze at import time, so a
        # test (or an operator edit) of the constant would silently not apply —
        # the same rule, for the same reason, as time_series_profit_prob's k.
        base = max_affordable_pairs(100_000, 0.50)
        monkeypatch.setattr(config, "BUDGET_FRACTION", 0.40)
        assert max_affordable_pairs(100_000, 0.50) == base * 2

    def test_nonpositive_price_sum_returns_zero_not_zerodivision(self):
        # A nonpositive sum means the book carried no usable level; every
        # caller gets 0 rather than having to guard the division itself.
        assert max_affordable_pairs(100_000, 0.0) == 0
        assert max_affordable_pairs(100_000, -0.5) == 0

    def test_budget_too_small_for_one_pair(self):
        assert max_affordable_pairs(100, 0.90, 0.20) == 0

    @pytest.mark.parametrize("k, cap, st_cap", [
        (0.75, 0.20, 1.0), (0.40, 1.0, 1.0), (0.80, 1.0, 1.0), (0.60, 0.35, 1.0),
        (0.80, 1.0, 0.20), (0.40, 0.35, 0.05),
    ])
    def test_scanner_cap_bounds_the_sizer(self, k, cap, st_cap):
        # Enrichment bounds its average at max_kelly_fraction over the MINIMUM
        # (best-level) price sum, so its count can never be smaller than the
        # sizer's: checked against compute_trade with the SAME settings.
        settings = config.LiveSettings(
            tier_floors=True, spread_band=(0.0, 1.0), interval_discount=k, size_cap=cap,
            same_title_size_cap=st_cap)
        balance = 1_000_000
        now = datetime.now(UTC)

        def pair(pair_type, pA, pB, nA, nB):
            p = MagicMock()
            p.pA, p.pB, p.nA, p.nB = pA, pB, nA, nB
            p.pair_type, p.tradeable, p.max_contracts = pair_type, True, 0
            p.canonical_title = f"{pair_type} {pA}/{pB}"
            p.market_a.close_time = now + timedelta(days=15)
            p.market_b.close_time = now + timedelta(days=30)
            return p

        # Time-series books are UNCROSSED (pB >= 1 - nB), as enrichment
        # enforces on every pair it keeps
        ts = [pair("time_series", pA, pB, 1 - pA, nB)
              for pA, pB, nB in [(0.10, 0.70, 0.30), (0.05, 0.85, 0.15), (0.20, 0.50, 0.50),
                                 (0.30, 0.60, 0.40), (0.12, 0.33, 0.70)]]
        st = [pair("same_title", pA, pB, nA, 0.70)
              for pA, pB, nA in [(0.70, 0.30, 0.20), (0.60, 0.31, 0.44), (0.55, 0.30, 0.45)]]
        sized = {"time_series": 0, "same_title": 0}
        for p in ts + st:
            spec = compute_trade(p, balance, settings=settings)
            if spec is None:
                continue
            sized[p.pair_type] += 1
            bound = config.max_kelly_fraction(p.pair_type, settings)
            assert spec.kelly_fraction <= bound, (p.canonical_title, spec.kelly_fraction, bound)
            # ... and so the scanner's count bounds the sizer's
            assert spec.x <= max_affordable_pairs(balance, sum(leg_prices(p)), bound)
        # Non-vacuous for both types at every setting
        assert sized["time_series"] > 0 and sized["same_title"] > 0, sized


class TestCreateNewOutput:
    """TS-18: exclusive creation is what makes the callers' "never silently
    dropped" promise a guarantee rather than a probability — two writers racing
    for one name cannot both win, however fine the timestamp in it."""

    def test_suffixes_on_collision(self, tmp_path):
        made = []
        for i in range(3):
            path, fh = config.create_new_output(tmp_path / "trade_log_x.xlsx")
            with fh:
                fh.write(f"run{i}".encode())
            made.append(path)

        assert [p.name for p in made] == [
            "trade_log_x.xlsx",
            "trade_log_x-1.xlsx",
            "trade_log_x-2.xlsx",
        ]
        # Each write survives with its own content — nothing was overwritten.
        assert [p.read_bytes() for p in made] == [b"run0", b"run1", b"run2"]

    def test_propagates_non_collision_errors(self, tmp_path, monkeypatch):
        # Only FileExistsError is retried. A permission failure must surface on
        # the first attempt rather than looping OUTPUT_NAME_MAX_ATTEMPTS times
        # against a cause that will not change. Monkeypatch rather than chmod:
        # a directory-mode test silently passes when run as root.
        calls = []

        def boom(self, *args, **kwargs):
            calls.append(self)
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(pathlib.Path, "open", boom)

        with pytest.raises(PermissionError):
            config.create_new_output(tmp_path / "trade_log_x.xlsx")
        assert len(calls) == 1


class TestSameEventLadderSwitch:
    """DR-73: same-event deadline ladders ship ON (operator decision,
    2026-09-26; they shipped OFF with DR-73).

    The switch changes which pairs exist, and turning it on was measured to
    deploy 98% of a $10,000 balance into 6 trades at -31% market-implied EV
    (see the constant's own comment). It must not change by accident in either
    direction: a change to the shipped value must fail a test rather than
    silently change what the next prod run trades.
    """

    def test_same_event_ladders_ship_on(self):
        assert config.TIME_SERIES_SAME_EVENT_LADDERS is True

    def test_every_module_binding_matches_the_config(self):
        # scanner, backtester and backtest each bind the constant BY VALUE at
        # import, and they are what actually run: the live finder, the
        # backtest's pair extraction and entry, and the CLI echo. A stray
        # re-binding in any of them would change what trades or what a
        # backtest measures while the pin above stays green. Value-independent,
        # so a change to the shipped value still fails only that pin.
        from kalshi_betting import backtest, backtester, scanner
        for module in (scanner, backtester, backtest):
            assert module.TIME_SERIES_SAME_EVENT_LADDERS is \
                config.TIME_SERIES_SAME_EVENT_LADDERS, module.__name__


class TestLiveSettings:
    """LiveSettings validates and normalises on construction, is frozen, and
    re-validates under dataclasses.replace."""

    @pytest.mark.parametrize("bad", [1, 0, None, "True", "false", 1.0])
    def test_tier_floors_must_be_exactly_a_bool(self, bad):
        # min_price_diff_for_gap drops the tier only on an explicit False
        with pytest.raises(ValueError, match="tier_floors"):
            _settings(tier_floors=bad)

    @pytest.mark.parametrize("band", [
        (0.5, 0.5), (0.6, 0.5), (-0.1, 0.5), (0.2, 1.1), (float("nan"), 0.5),
        (0.1,), (0.1, 0.2, 0.3), "ab", 0.5, None,
    ])
    def test_band_is_validated_by_the_backtest_band_validator(self, band):
        # None too: the validator would read it as the backtest's default band
        with pytest.raises(ValueError, match="spread"):
            _settings(spread_band=band)

    @pytest.mark.parametrize("k", [0, 0.0, -0.1, 1.01, float("nan"), float("inf"),
                                   True, False, "0.8", None])
    def test_k_outside_zero_one_is_refused(self, k):
        # k = 0 would price every time-series pair as riskless (p = 1)
        with pytest.raises(ValueError, match="interval_discount"):
            _settings(interval_discount=k)

    @pytest.mark.parametrize("k", [1, 1.0, 0.75, 0.8, 1e-9])
    def test_k_in_range_is_accepted_as_a_float(self, k):
        s = _settings(interval_discount=k)
        assert s.interval_discount == k and type(s.interval_discount) is float

    # 1e-7 and 1e-6 round to zero steps (within PRICE_EPSILON of 0)
    @pytest.mark.parametrize("cap", [0, 0.0, 0.37, 1.01, -0.05, float("nan"),
                                     float("inf"), True, False, "0.2", None, 0.051,
                                     1e-7, 1e-6, 0.02])
    def test_cap_off_the_grid_or_out_of_range_is_refused(self, cap):
        with pytest.raises(ValueError, match="size_cap"):
            _settings(size_cap=cap)

    @pytest.mark.parametrize("cap", [0.05, 0.2, 0.35, 0.95, 1.0, 1])
    def test_cap_on_the_grid_is_accepted(self, cap):
        assert _settings(size_cap=cap).size_cap == cap

    def test_the_cap_is_normalised_onto_the_sweep_grid(self):
        # 0.05 * 7 is 0.35000000000000003; a live cap is float-equal to a
        # backtester.SIZE_CAP_SWEEP cell
        from kalshi_betting.backtester import SIZE_CAP_SWEEP
        assert _settings(size_cap=0.35).size_cap == round(0.05 * 7, 2)
        for cap in SIZE_CAP_SWEEP:
            assert _settings(size_cap=cap).size_cap in SIZE_CAP_SWEEP
        assert _settings(size_cap=0.05 * 7).size_cap == 0.35
        assert type(_settings(size_cap=1).size_cap) is float

    def test_the_band_comes_back_as_floats(self):
        s = _settings(spread_band=(0, 1))
        assert s.spread_band == (0.0, 1.0)
        assert all(type(x) is float for x in s.spread_band)
        # -0.0 is normalised to +0.0 by the backtest band's validator
        assert str(_settings(spread_band=(-0.0, 0.5)).spread_band[0]) == "0.0"

    def test_replace_re_validates(self):
        s = _settings()
        with pytest.raises(ValueError):
            dataclasses.replace(s, size_cap=0.37)
        with pytest.raises(ValueError):
            dataclasses.replace(s, interval_discount=0)
        with pytest.raises(ValueError):
            dataclasses.replace(s, spread_band=(0.6, 0.5))
        assert dataclasses.replace(s, size_cap=1.0).size_cap == 1.0

    def test_it_is_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            _settings().size_cap = 1.0

    def test_live_settings_reads_config_at_call_time(self, monkeypatch):
        # Every value differs from the shipped one, so a read of the shipped
        # constants cannot pass
        shipped = live_settings()
        monkeypatch.setattr(config, "TIME_SERIES_TIER_FLOORS", True)
        monkeypatch.setattr(config, "TIME_SERIES_SPREAD_BAND", (0.1, 0.6))
        monkeypatch.setattr(config, "TIME_SERIES_INTERVAL_PROB_DISCOUNT", 0.6)
        monkeypatch.setattr(config, "BUDGET_FRACTION", 0.35)
        monkeypatch.setattr(config, "SAME_TITLE_SIZE_CAP", 0.25)
        monkeypatch.setattr(config, "TRADE_CATEGORIES", ("Economics",))
        monkeypatch.setattr(config, "TRADE_TAGS", ["Oil & Gas"])
        expected = _settings(True, (0.1, 0.6), 0.6, 0.35, 0.25,
                             ("Economics",), ("Oil & Gas",))
        assert live_settings() == expected
        assert all(getattr(expected, f.name) != getattr(shipped, f.name)
                   for f in dataclasses.fields(LiveSettings))

    def test_live_settings_refuses_an_invalid_constant(self, monkeypatch):
        monkeypatch.setattr(config, "BUDGET_FRACTION", 0.37)
        with pytest.raises(ValueError, match="size_cap"):
            live_settings()

    def test_live_settings_refuses_an_invalid_same_title_cap(self, monkeypatch):
        monkeypatch.setattr(config, "SAME_TITLE_SIZE_CAP", 0.37)
        with pytest.raises(ValueError, match="same_title_size_cap"):
            live_settings()

    def test_the_shipped_values_resolve(self):
        s = live_settings()
        assert s == LiveSettings(
            tier_floors=config.TIME_SERIES_TIER_FLOORS,
            spread_band=config.TIME_SERIES_SPREAD_BAND,
            interval_discount=config.TIME_SERIES_INTERVAL_PROB_DISCOUNT,
            size_cap=config.BUDGET_FRACTION,
            same_title_size_cap=config.SAME_TITLE_SIZE_CAP,
            categories=config.TRADE_CATEGORIES,
            tags=config.TRADE_TAGS,
        )
        assert type(s.tier_floors) is bool

    @pytest.mark.parametrize("cap", [0, 0.37, 1.01, float("nan"), True, "0.2", None, 1e-7])
    def test_same_title_cap_off_the_grid_or_out_of_range_is_refused(self, cap):
        # The same grid and validator as size_cap, reported under its own name
        with pytest.raises(ValueError, match="same_title_size_cap"):
            _settings(same_title_size_cap=cap)

    def test_same_title_cap_is_normalised_and_defaults_to_no_cap(self):
        assert _settings(same_title_size_cap=0.05 * 7).same_title_size_cap == 0.35
        assert type(_settings(same_title_size_cap=1).same_title_size_cap) is float
        # Naming only the first four fields adds no same-title cap
        assert LiveSettings(True, (0.0, 1.0), 0.75, 0.2).same_title_size_cap == 1.0
        # ... and replace re-validates it like every other field
        with pytest.raises(ValueError, match="same_title_size_cap"):
            dataclasses.replace(_settings(), same_title_size_cap=0.37)

    @pytest.mark.parametrize("field", ["categories", "tags"])
    @pytest.mark.parametrize("bad", [
        "Sports",            # a bare str would filter on its characters
        (), [],              # matches nothing: None is "any"
        ("",), ("  ",),      # an empty name, before or after stripping
        # "any" would match nothing yet render exactly like None
        ("any",), ("Sports", "ANY"), (" Any ",),
        ("Sports", None), ("Sports", 7), (b"Sports",),
        {"Sports"},          # a set has no order to render or keep
        7, True,
    ])
    def test_a_filter_must_be_none_or_a_non_empty_tuple_of_names(self, field, bad):
        with pytest.raises(ValueError, match=field):
            _settings(**{field: bad})
        # replace re-validates it too
        with pytest.raises(ValueError, match=field):
            dataclasses.replace(_settings(), **{field: bad})

    @pytest.mark.parametrize("field", ["categories", "tags"])
    def test_a_filter_is_normalised_to_a_tuple_of_stripped_names(self, field):
        s = _settings(**{field: [" Sports ", "Oil & Gas"]})
        assert getattr(s, field) == ("Sports", "Oil & Gas")
        assert type(getattr(s, field)) is tuple
        # Order and case are kept (main._filter_by_category ignores case)
        assert getattr(_settings(**{field: ("b", "A")}), field) == ("b", "A")
        assert getattr(_settings(**{field: None}), field) is None

    def test_a_construction_without_the_filters_filters_nothing(self):
        s = LiveSettings(True, (0.0, 1.0), 0.75, 0.2, 1.0)
        assert s.categories is None and s.tags is None
        # Defaulted and last: the first five fields keep their positions
        assert [f.name for f in dataclasses.fields(LiveSettings)][-2:] == ["categories", "tags"]

    @pytest.mark.parametrize("name, value", [("TRADE_CATEGORIES", "Sports"),
                                             ("TRADE_TAGS", ())])
    def test_live_settings_refuses_an_invalid_filter(self, monkeypatch, name, value):
        monkeypatch.setattr(config, name, value)
        with pytest.raises(ValueError, match="categories" if "CATEG" in name else "tags"):
            live_settings()


class TestTimeSeriesSpreadRefusal:
    """The live spread rule: positivity, the entry floor, then the band's
    ceiling, with backtester._find_entry's order and PRICE_EPSILON placement."""

    def test_positivity_epsilon_sits_on_the_reject_side(self):
        s = _settings(tier_floors=False)
        assert time_series_spread_refusal(PRICE_EPSILON, 5, s) == SPREAD_NOT_POSITIVE
        assert time_series_spread_refusal(0.0, 5, s) == SPREAD_NOT_POSITIVE
        assert time_series_spread_refusal(-0.01, 5, s) == SPREAD_NOT_POSITIVE
        assert time_series_spread_refusal(2 * PRICE_EPSILON, 5, s) is None
        assert time_series_spread_refusal(0.0001, 5, s) is None

    def test_floor_epsilon_sits_on_the_keep_side(self):
        s = _settings()  # tiers on: 0.15 up to 15 days
        assert time_series_spread_refusal(0.15 - 2 * PRICE_EPSILON, 5, s) == SPREAD_BELOW_FLOOR
        assert time_series_spread_refusal(0.15 - PRICE_EPSILON / 2, 5, s) is None
        # 0.35 - 0.20 == 0.14999999999999997: kept on the 0.15 floor (TS-09)
        assert 0.35 - 0.20 < 0.15
        assert time_series_spread_refusal(0.35 - 0.20, 5, s) is None

    def test_ceiling_epsilon_sits_on_the_keep_side(self):
        s = _settings(spread_band=(0.0, 0.6))
        assert time_series_spread_refusal(0.6 + PRICE_EPSILON / 2, 5, s) is None
        assert time_series_spread_refusal(0.6 + 2 * PRICE_EPSILON, 5, s) == SPREAD_ABOVE_CEILING
        # 0.90 - 0.30 == 0.6000000000000001: kept at a 0.60 ceiling
        assert 0.90 - 0.30 > 0.6
        assert time_series_spread_refusal(0.90 - 0.30, 5, s) is None

    @pytest.mark.parametrize("spread, gap, verdict", [
        (0.18, 5, SPREAD_BELOW_FLOOR),     # the band's 0.20 floor, above the tier
        (0.20, 5, None),                   # exactly on it (TS-09)
        (0.25, 16, SPREAD_BELOW_FLOOR),    # the 0.30 long tier, above the floor
        (0.60, 5, None),                   # exactly on the ceiling
        (0.61, 5, SPREAD_ABOVE_CEILING),
    ])
    def test_tiers_on_band_020_060(self, spread, gap, verdict):
        s = _settings(spread_band=(0.2, 0.6))
        assert time_series_spread_refusal(spread, gap, s) == verdict

    def test_tiers_off_floor_zero(self):
        s = _settings(tier_floors=False)
        assert time_series_spread_refusal(0.0001, 20, s) is None
        assert time_series_spread_refusal(0.0, 20, s) == SPREAD_NOT_POSITIVE
        assert time_series_spread_refusal(-0.01, 20, s) == SPREAD_NOT_POSITIVE

    def test_the_floor_is_tested_before_the_ceiling(self):
        # 0.12 at 5 days is under the 0.15 tier AND over the 0.10 ceiling:
        # the floor is tested first, as _find_entry does
        s = _settings(spread_band=(0.0, 0.10))
        assert time_series_spread_refusal(0.12, 5, s) == SPREAD_BELOW_FLOOR
        # ... and positivity before both
        assert time_series_spread_refusal(-0.2, 5, s) == SPREAD_NOT_POSITIVE

    def test_the_floor_is_live_time_series_floor(self):
        for tiers in (True, False):
            for band in ((0.0, 1.0), (0.2, 0.6), (0.35, 0.9)):
                s = _settings(tier_floors=tiers, spread_band=band)
                for gap in (0, 15, 16, 30):
                    floor = config.live_time_series_floor(gap, s)
                    assert floor == min_price_diff_for_gap(
                        gap, spread_min=band[0], tier_floors=tiers)
                    if floor > PRICE_EPSILON:
                        assert time_series_spread_refusal(
                            floor - 2 * PRICE_EPSILON, gap, s) == SPREAD_BELOW_FLOOR


class TestMaxKellyFraction:
    """max_kelly_fraction, enrichment's affordability bound. Compared with ==:
    1 - 0.8 is 0.19999999999999996, and the bound must be exactly 0.20."""

    def test_today_both_types_are_the_cap(self):
        s = _settings()
        assert max_kelly_fraction("time_series", s) == 0.20
        assert max_kelly_fraction("same_title", s) == 0.20

    def test_no_cap_time_series_is_one_minus_k_rounded(self):
        s = _settings(interval_discount=0.8, size_cap=1.0)
        assert 1.0 - 0.8 != 0.2
        assert max_kelly_fraction("time_series", s) == 0.2

    def test_no_cap_same_title_is_the_co_resolution_prior(self):
        s = _settings(size_cap=1.0)
        assert max_kelly_fraction("same_title", s) == 0.95
        assert max_kelly_fraction("same_title", s) == config.SAME_TITLE_CO_RESOLVE_PROB

    def test_k_of_one_bounds_time_series_at_zero(self):
        s = _settings(interval_discount=1.0, size_cap=1.0)
        assert max_kelly_fraction("time_series", s) == 0
        # same-title never reads k
        assert max_kelly_fraction("same_title", s) == 0.95

    @pytest.mark.parametrize("pair_type", [None, "bogus", "Time_Series", MagicMock().pair_type])
    def test_anything_but_time_series_reads_as_same_title(self, pair_type):
        s = _settings(interval_discount=0.8, size_cap=1.0)
        assert max_kelly_fraction(pair_type, s) == 0.95

    def test_the_round_is_what_keeps_the_count(self):
        s = _settings(interval_discount=0.8, size_cap=1.0)
        assert max_affordable_pairs(1_000_000, 0.8, max_kelly_fraction("time_series", s)) == 2500
        # the unrounded bound loses a contract on this round-number book
        assert max_affordable_pairs(1_000_000, 0.8, 1.0 - 0.8) == 2499

    def test_the_same_title_cap_bounds_same_title_only(self):
        # No per-trade cap, a 20% same-title cap
        s = _settings(interval_discount=0.8, size_cap=1.0, same_title_size_cap=0.2)
        assert max_kelly_fraction("same_title", s) == 0.2
        assert max_kelly_fraction("time_series", s) == 0.2   # 1 - k, not the cap
        s = _settings(interval_discount=0.6, size_cap=1.0, same_title_size_cap=0.2)
        assert max_kelly_fraction("time_series", s) == 0.4   # never the same-title cap
        # The tighter of the two caps binds a same-title pair
        s = _settings(size_cap=0.1, same_title_size_cap=0.2)
        assert max_kelly_fraction("same_title", s) == 0.1
        assert max_kelly_fraction("time_series", s) == 0.1


class TestPairSizeCap:
    """config.pair_size_cap, the one definition of a pair's per-trade cap."""

    def test_time_series_reads_the_general_cap_alone(self):
        assert pair_size_cap("time_series", 1.0, 0.2) == 1.0
        assert pair_size_cap("time_series", 0.35, 0.05) == 0.35

    def test_same_title_takes_the_tighter_cap(self):
        assert pair_size_cap("same_title", 1.0, 0.2) == 0.2
        assert pair_size_cap("same_title", 0.1, 0.2) == 0.1
        assert pair_size_cap("same_title", 0.2, 1.0) == 0.2

    @pytest.mark.parametrize("pair_type", [None, "bogus", "Time_Series", MagicMock().pair_type])
    def test_anything_but_time_series_reads_as_same_title(self, pair_type):
        # scanner.leg_sides' rule: an unknown type is never the directional bet
        assert pair_size_cap(pair_type, 1.0, 0.2) == 0.2

    def test_the_shipped_caps_are_on_the_grid(self):
        # Both must resolve through LiveSettings, which validates them
        s = live_settings()
        assert s.size_cap == config.BUDGET_FRACTION
        assert s.same_title_size_cap == config.SAME_TITLE_SIZE_CAP


class TestDescribeTimeSeriesRule:
    """describe_time_series_rule says "no spread band" for (0, 1) alone and
    names any band with a floor or a ceiling of its own."""

    def test_only_the_no_band_band_reads_as_no_band(self):
        assert config.describe_time_series_rule(True, (0.0, 1.0)).endswith(", no spread band")
        assert config.describe_time_series_rule(False, (0.0, 1.0)).endswith(", no spread band")

    @pytest.mark.parametrize("band, text", [
        ((0.2, 1.0), "spread band 0.2-1 on pB - pA"),   # a floor alone
        ((0.0, 0.6), "spread band 0-0.6 on pB - pA"),   # a ceiling alone
        ((0.0, 0.5), "spread band 0-0.5 on pB - pA"),
        ((0.25, 0.9), "spread band 0.25-0.9 on pB - pA"),
    ])
    def test_a_band_with_a_floor_or_a_ceiling_is_named(self, band, text):
        for tier_floors in (True, False):
            line = config.describe_time_series_rule(tier_floors, band)
            assert line.endswith(", " + text), line
            assert "no spread band" not in line

    def test_the_tier_clause_follows_the_switch(self):
        on = config.describe_time_series_rule(True, (0.0, 1.0))
        off = config.describe_time_series_rule(False, (0.0, 1.0))
        assert on.startswith("tier floors on (≥15% up to 15 days apart, ≥30% for 16-30)")
        assert off.startswith("tier floors off (pB - pA must still be positive)")

    def test_a_band_bound_renders_as_the_live_settings_line_renders_it(self):
        # %g would print 0.3000001 as 0.3; both lines name the band exactly
        band = (0.0, 0.3000001)
        line = config.describe_time_series_rule(True, band)
        assert line.endswith(", spread band 0-0.3000001 on pB - pA"), line
        assert "spread band 0-0.3000001" in config.describe_live_settings(
            _settings(spread_band=band))


class TestDescribeLiveSettings:
    """describe_live_settings marks each field whose RAW value departs from
    the reference, and never prints two different values alike."""

    def test_config_pys_values_render_with_no_mark(self):
        s = _settings()
        assert config.describe_live_settings(s, s) == (
            "tier floors on | spread band none | k 0.75 | per-trade cap 20% | "
            "same-title cap 100% (no extra cap) | categories any | tags any")
        assert config.describe_live_settings(s) == config.describe_live_settings(s, s)

    def test_the_shipped_values_render_with_no_mark(self):
        s = live_settings()
        assert "(config:" not in config.describe_live_settings(s, s)

    # One departing value per LiveSettings field and the mark it must carry; a
    # new field fails test_every_field_has_a_departure_row until it has a row
    _DEPARTURES = {
        "tier_floors": (False, "tier floors off (config: on)"),
        "spread_band": ((0.1, 1.0), "spread band 0.1-1 (config: none)"),
        "interval_discount": (0.6, "k 0.6 (config: 0.75)"),
        "size_cap": (0.35, "per-trade cap 35% (config: 20%)"),
        "same_title_size_cap": (0.25, "same-title cap 25% (config: 100% (no extra cap))"),
        "categories": (("Economics",), "categories Economics (config: any)"),
        "tags": (("Oil & Gas", "Energy"), "tags Oil & Gas, Energy (config: any)"),
    }

    def test_every_field_has_a_departure_row(self):
        assert set(self._DEPARTURES) == {f.name for f in dataclasses.fields(LiveSettings)}

    def test_every_departing_field_is_marked_and_no_other(self):
        ref = _settings()
        s = _settings(False, (0.0, 0.5), 0.8, 1.0, 0.2)
        assert config.describe_live_settings(s, ref) == (
            "tier floors off (config: on) | spread band 0-0.5 (config: none) | "
            "k 0.8 (config: 0.75) | per-trade cap 100% (no cap) (config: 20%) | "
            "same-title cap 20% (config: 100% (no extra cap)) | categories any | tags any")
        for field in dataclasses.fields(LiveSettings):
            value, mark = self._DEPARTURES[field.name]
            line = config.describe_live_settings(
                dataclasses.replace(ref, **{field.name: value}), ref)
            assert line.count("(config:") == 1, line
            assert mark in line, (field.name, line)

    def test_k_renders_exactly(self):
        # 0.751 and 0.75 must never print alike: a departure would read as none
        ref = _settings()
        line = config.describe_live_settings(_settings(interval_discount=0.751), ref)
        assert "k 0.751 (config: 0.75)" in line

    def test_a_band_bound_renders_exactly_when_the_short_form_would_not(self):
        # %g keeps six significant digits: 0.3 and 0.3000001 print alike there
        ref = _settings(spread_band=(0.3, 0.9))
        s = _settings(spread_band=(0.3000001, 0.9))
        line = config.describe_live_settings(s, ref)
        assert "spread band 0.3000001-0.9 (config: 0.3-0.9)" in line

    def test_no_reference_marks_nothing(self):
        line = config.describe_live_settings(_settings(False, (0.0, 0.5), 0.8, 1.0, 0.2))
        assert "(config:" not in line
        assert line == ("tier floors off | spread band 0-0.5 | k 0.8 | "
                        "per-trade cap 100% (no cap) | same-title cap 20% | "
                        "categories any | tags any")

    def test_a_filter_renders_its_names_and_any_for_none(self):
        # Each field renders the run's value, then config's in the mark; None is "any"
        ref = _settings(categories=("Sports",))
        line = config.describe_live_settings(_settings(tags=("Basketball", "Soccer")), ref)
        assert "categories any (config: Sports)" in line
        assert "tags Basketball, Soccer (config: any)" in line
        # Case is part of the raw value: a case-only change departs and shows
        line = config.describe_live_settings(_settings(categories=("sports",)), ref)
        assert "categories sports (config: Sports)" in line

    def test_a_filter_name_holding_a_separator_renders_exactly(self):
        # ("a, b",) and ("a", "b") must never print alike
        one = config.describe_live_settings(_settings(categories=("a, b",)))
        two = config.describe_live_settings(_settings(categories=("a", "b")))
        assert "categories 'a, b' |" in one
        assert "categories a, b |" in two
        assert one != two
        for sep in (";", "|"):
            line = config.describe_live_settings(_settings(tags=(f"x{sep}y", "z")))
            assert f"tags 'x{sep}y', 'z'" in line

    @pytest.mark.parametrize("field", ["categories", "tags"])
    def test_only_the_name_any_is_refused_never_a_name_containing_it(self, field):
        # Only the name "any" is refused, never one containing it ("Any Awards")
        s = _settings(**{field: ("Companies", "Any Awards")})
        assert getattr(s, field) == ("Companies", "Any Awards")
        assert f"{field} Companies, Any Awards" in config.describe_live_settings(s)
        assert config.describe_live_settings(s) != config.describe_live_settings(_settings())

    def test_describe_trade_filter_names_the_filter_in_the_same_words(self):
        assert config.describe_trade_filter(_settings()) == "categories any; tags any"
        s = _settings(categories=("Economics", "Sports"), tags=("Oil & Gas",))
        assert config.describe_trade_filter(s) == (
            "categories Economics, Sports; tags Oil & Gas")
        line = config.describe_live_settings(s)
        assert "categories Economics, Sports" in line and "tags Oil & Gas" in line


class TestLiveRuleWarnings:
    """live_rule_warnings: settings that empty part of the time-series
    strategy, or let one pair stake more than LIVE_EXPOSURE_WARN_FRACTION."""

    def test_the_shipped_values_warn_nothing(self):
        assert config.live_rule_warnings(live_settings()) == []
        assert config.live_rule_warnings(_settings()) == []

    def test_a_ceiling_on_the_long_tier_keeps_only_that_spread(self):
        (text,) = config.live_rule_warnings(_settings(spread_band=(0.0, 0.30)))
        assert "sits on the 0.3 entry floor" in text and "16-30 days apart" in text

    def test_a_ceiling_below_the_long_tier_empties_it(self):
        (text,) = config.live_rule_warnings(_settings(spread_band=(0.0, 0.29)))
        assert "is below the 0.3 entry floor" in text and "16-30 days apart" in text
        assert "none of them can trade" in text

    def test_a_ceiling_on_the_short_tier(self):
        out = config.live_rule_warnings(_settings(spread_band=(0.0, 0.15)))
        assert len(out) == 2
        assert "sits on the 0.15 entry floor" in out[0] and "0-15 days apart" in out[0]
        assert "is below the 0.3 entry floor" in out[1] and "16-30 days apart" in out[1]

    def test_the_ceiling_is_judged_with_the_live_rules_epsilon(self):
        # Within PRICE_EPSILON under a tier the ceiling still keeps a spread on it
        (text,) = config.live_rule_warnings(
            _settings(spread_band=(0.0, 0.30 - PRICE_EPSILON / 2)))
        assert "sits on" in text

    def test_with_the_tiers_off_the_band_floor_is_the_entry_floor(self):
        # The band floor is the entry floor at every gap, below the ceiling ...
        assert config.live_rule_warnings(_settings(tier_floors=False, spread_band=(0.0, 0.1))) == []
        assert config.live_rule_warnings(
            _settings(tier_floors=False, spread_band=(0.2, 0.200002))) == []
        # ... but within PRICE_EPSILON of it only a spread on the floor trades,
        # judged once for every deadline gap
        (text,) = config.live_rule_warnings(
            _settings(tier_floors=False, spread_band=(0.2, 0.2000001)))
        assert text == ("the spread band's 0.2000001 ceiling sits on the 0.2 entry floor for "
                        "time-series pairs at any deadline gap, so only spreads exactly on "
                        "it can trade")
        # With the tiers on, the same state warns once per tier range
        assert len(config.live_rule_warnings(
            _settings(spread_band=(0.5, 0.5000005)))) == 2
        assert len(config.live_rule_warnings(
            _settings(tier_floors=False, spread_band=(0.5, 0.5000005)))) == 1

    def test_a_ceiling_on_a_floor_of_zero_admits_nothing(self):
        # On a floor of 0 a spread on the floor is not positive: nothing trades
        band = (0.0, PRICE_EPSILON / 2)
        (text,) = config.live_rule_warnings(_settings(tier_floors=False, spread_band=band))
        assert "sits on the 0 entry floor for time-series pairs at any deadline gap" in text
        assert "pB - pA must be positive, so none of them can trade" in text
        assert "only spreads exactly on it" not in text
        # ... as the live rule confirms
        s = _settings(tier_floors=False, spread_band=band)
        assert config.time_series_spread_refusal(0.0, 5, s) == config.SPREAD_NOT_POSITIVE
        assert config.time_series_spread_refusal(band[1], 5, s) == config.SPREAD_NOT_POSITIVE

    def test_a_ceiling_well_above_both_tiers_is_silent(self):
        assert config.live_rule_warnings(_settings(spread_band=(0.2, 0.6))) == []

    def test_a_lifted_cap_at_a_low_k_warns_for_both_types(self):
        # --size-cap 60 --interval-discount 0.4: both types reach the cap
        out = config.live_rule_warnings(_settings(interval_discount=0.4, size_cap=0.6))
        assert out == [
            "one time-series pair may stake up to 60% of the balance, above the 20% "
            "this check accepts",
            "one same-title pair may stake up to 60% of the balance, above the 20% "
            "this check accepts",
        ]

    def test_no_cap_with_a_same_title_cap_of_50_warns_for_both_types(self):
        # --size-cap 100 --same-title-size-cap 50 at k 0.75: 25% and 50%
        out = config.live_rule_warnings(_settings(size_cap=1.0, same_title_size_cap=0.5))
        assert len(out) == 2
        assert "one time-series pair may stake up to 25%" in out[0]
        assert "one same-title pair may stake up to 50%" in out[1]

    def test_a_same_title_cap_above_the_per_trade_cap_warns_nothing(self):
        # --same-title-size-cap 50 alone: the 20% per-trade cap still binds
        assert config.live_rule_warnings(_settings(same_title_size_cap=0.5)) == []

    def test_a_bound_a_hair_over_the_threshold_prints_exactly(self):
        # 1 - 0.7996 = 0.2004: over 20%, and never printed as 20%
        (text,) = config.live_rule_warnings(_settings(interval_discount=0.7996, size_cap=1.0,
                                                     same_title_size_cap=0.2))
        assert "up to 20.04% of the balance" in text

    def test_k_of_one_names_the_dead_time_series_leg(self):
        (text,) = config.live_rule_warnings(_settings(interval_discount=1.0))
        assert text.startswith("k = 1.0: ") and "no time-series trade can size" in text

    def test_the_threshold_is_the_constant(self, monkeypatch):
        monkeypatch.setattr(config, "LIVE_EXPOSURE_WARN_FRACTION", 0.10)
        out = config.live_rule_warnings(_settings())
        assert len(out) == 2 and all("above the 10%" in text for text in out)


class TestShippedLiveToggles:
    """The live toggles config.py ships, read unpatched end to end: any
    positive time-series spread up to 0.5 is admitted, books price up to a
    leg-price sum of 1.0 with no-edge levels cut, and one pair stakes at most
    20% (time-series under 1 - k, same-title at its cap). Changing a toggle
    means re-pinning this class."""

    _SHIPPED = LiveSettings(tier_floors=False, spread_band=(0.0, 0.5), interval_discount=0.8,
                            size_cap=1.0, same_title_size_cap=0.2, categories=None, tags=None)
    # The values conftest's apply_pre_toggle_defaults pins, for the controls
    _BEFORE = LiveSettings(tier_floors=True, spread_band=(0.0, 1.0), interval_discount=0.75,
                           size_cap=0.2, same_title_size_cap=1.0)

    def test_the_shipped_settings(self):
        assert live_settings() == self._SHIPPED
        assert (config.TIME_SERIES_TIER_FLOORS, config.TIME_SERIES_SPREAD_BAND,
                config.TIME_SERIES_INTERVAL_PROB_DISCOUNT, config.BUDGET_FRACTION,
                config.SAME_TITLE_SIZE_CAP, config.TRADE_CATEGORIES, config.TRADE_TAGS) == (
            False, (0.0, 0.5), 0.80, 1.0, 0.20, None, None)

    def test_the_finder_admits_any_positive_spread_up_to_the_ceiling(self, caplog):
        # 0.10 at 10 days and 0.20 at 20 days sit under their tiers and are
        # admitted; 0.55 is over the 0.5 ceiling and refused, and counted
        def scan(gap, pA, pB, settings=None):
            mA, mB = _ts._ts_pair_markets(gap_days=gap, pA=pA, pB=pB)
            return scanner.find_time_series_pairs(MagicMock(), held_tickers=set(),
                                                  markets=[mA, mB], settings=settings)

        for gap, pA, pB in ((10, 0.30, 0.40), (20, 0.30, 0.50)):
            [pair] = scan(gap, pA, pB)
            assert pair.pB - pair.pA == pytest.approx(pB - pA)
            # Control: the pre-toggle rule refuses it at its tier
            assert scan(gap, pA, pB, settings=self._BEFORE) == []
        with caplog.at_level(logging.INFO):
            assert scan(10, 0.30, 0.85) == []
        assert ("Time-series candidates refused above the spread band's 0.5 ceiling "
                "(pB - pA; before the one-best-per-group contest): 1") in caplog.text
        # Control: the pre-toggle rule has no ceiling
        assert len(scan(10, 0.30, 0.85, settings=self._BEFORE)) == 1

    def test_the_price_sum_ceiling_is_one_and_the_fee_cut_is_live(self):
        # TestEnrichmentSpreadRule's fixture: its second level (0.53 + 0.45)
        # sits inside the 1.0 ceiling with an edge under its fee, so it is cut
        pair = _ts._ts_candidate(gap_days=10, pA=0.30, pB=0.56, nB=0.45)
        assert scanner._pair_max_sum(pair, live_settings()) == 1.0
        [enriched] = scanner.enrich_with_orderbook_prices(
            _ts.TestEnrichmentSpreadRule._adversary_client(), [pair], 1_000_000)
        assert enriched.tradeable is True
        assert enriched.depth_levels == (pytest.approx((0.30, 0.45, 100.0)),)
        spec = compute_trade(enriched, 1_000_000)
        assert spec is not None and spec.x == 100
        # With the tiers on, the 0.85 ceiling excludes that level on its own
        assert scanner._pair_max_sum(pair, self._BEFORE) == pytest.approx(0.85)

    def test_the_same_title_reference_pair_is_capped_at_20_percent(self):
        # nA 0.20 + pB 0.30: f* ~0.8945, bound only by SAME_TITLE_SIZE_CAP
        fee = fee_per_pair_approx(0.20, 0.30)
        net = 0.50 - fee
        f_star = config.SAME_TITLE_CO_RESOLVE_PROB - (
            1 - config.SAME_TITLE_CO_RESOLVE_PROB) * (0.50 + fee) / net
        assert f_star == pytest.approx(0.8945, abs=1e-4)
        pair = MagicMock()
        pair.pA, pair.pB, pair.nA, pair.nB = 0.70, 0.30, 0.20, 0.70
        pair.pair_type, pair.tradeable, pair.max_contracts = "same_title", True, 0
        pair.canonical_title = "same-title reference pair"
        pair.market_a.close_time = pair.market_b.close_time = (
            datetime.now(UTC) + timedelta(days=15))
        spec = compute_trade(pair, 1_000_000)
        assert spec is not None
        assert spec.kelly_fraction == pytest.approx(0.20)
        assert spec.total_cost_with_fees <= 10_000.0 * 0.20 + 1e-9
        assert max_kelly_fraction("same_title", live_settings()) == 0.2

    def test_every_time_series_spec_stakes_under_20_percent(self):
        # No per-trade cap: 1 - k = 0.20 bounds it on every uncrossed book
        assert max_kelly_fraction("time_series", live_settings()) == 0.2
        now = datetime.now(UTC)
        sized = 0
        for pA in (0.01, 0.05, 0.10, 0.20, 0.30, 0.40):
            for nB in (0.05, 0.15, 0.30, 0.45):
                for extra in (0.0, 0.02, 0.10):
                    pB = round(1.0 - nB + extra, 2)
                    if pB >= 1.0 or pB - pA > 0.5:
                        continue
                    pair = MagicMock()
                    pair.pA, pair.pB, pair.nA, pair.nB = pA, pB, 1 - pA, nB
                    pair.pair_type, pair.tradeable, pair.max_contracts = "time_series", True, 0
                    pair.canonical_title = f"ts {pA}/{pB}/{nB}"
                    pair.market_a.close_time = now + timedelta(days=5)
                    pair.market_b.close_time = now + timedelta(days=15)
                    spec = compute_trade(pair, 1_000_000)
                    if spec is None:
                        continue
                    sized += 1
                    assert spec.kelly_fraction < 0.2, (pA, pB, nB, spec.kelly_fraction)
        assert sized > 0

    def test_no_live_rule_warning_fires(self):
        assert config.live_rule_warnings(live_settings()) == []

    def test_a_default_backtest_departs_from_the_live_rule(self, monkeypatch, caplog):
        # Its primary follows k and both caps but keeps band (0, 1) with the
        # tiers on, so the live rule is a grid cell, and the last line says so
        golden = _tb.TestPrepareEntriesGolden()
        golden._patch(monkeypatch)
        monkeypatch.setattr(backtester, "SPREAD_BAND_SWEEP_FLOORS", (0.0, 0.35))
        monkeypatch.setattr(backtester, "SPREAD_BAND_SWEEP_CEILINGS", (0.5, 1.0))
        with caplog.at_level(logging.INFO):
            res = backtester.run_backtest_sweep(
                MagicMock(), MagicMock(), golden._START, 10_000.0, sweep=False,
                band_sweep=True, tier_off_sweep=True)
        primary = res.primary
        assert (primary.k, primary.size_cap, primary.spread_band, primary.tier_floors) == (
            0.8, 1.0, (0.0, 1.0), True)
        assert res.same_title_size_cap == 0.2
        assert (res.live_tier_floors, res.live_spread_band) == (False, (0.0, 0.5))
        rule = config.describe_time_series_rule(False, (0.0, 0.5))
        lines = [r.getMessage() for r in caplog.records
                 if r.getMessage().startswith("Live time-series rule (config.py):")]
        assert lines == [
            f"Live time-series rule (config.py): {rule} — this run's primary scenario does "
            "not (tier floors on, band 0-1); its grid simulated the live rule as band 0-0.5 "
            "with the tier floors off, which the dashboard's filter bar shows"]

    def test_validate_re_checks_the_edge_after_the_fee(self, caplog):
        # At a 1.0 ceiling the fee cut is the one pre-submission edge check: a
        # book that moves a tick against this thin-edge spec (FoK caps 0.02 /
        # 0.98, where every cell loses) is dropped
        balance = 1_000_000
        pair = _ts._ts_candidate(gap_days=10, pA=0.01, pB=0.03, nB=0.97)
        sized_on = _ts._ts_orderbook_client(pA_fill=0.01, nB_fill=0.97, qty=5000, pB_ref=0.03)
        [enriched] = scanner.enrich_with_orderbook_prices(sized_on, [pair], balance)
        spec = compute_trade(enriched, balance)
        assert spec is not None and spec.x == 748
        n, fill_a, fill_b = spec.x, 0.02, 0.98
        assert n * (1 - fill_a - fill_b) - fee_leg_exact(n, fill_a) - fee_leg_exact(n, fill_b) < 0
        moved = _ts._ts_orderbook_client(pA_fill=0.02, nB_fill=0.98, qty=5000, pB_ref=0.03)
        with caplog.at_level(logging.INFO):
            assert scanner.validate_pair_price(moved, spec) is False
        assert "keeps an edge after the fee" in caplog.text
        # Control: the book it was sized on still passes
        assert scanner.validate_pair_price(sized_on, spec) is True


class TestPreToggleDefaults:
    """conftest's apply_pre_toggle_defaults moves config's toggle constants AND
    every by-value copy a package module binds, found by walking the imports so
    a new binder is caught."""

    _TOGGLES = frozenset({
        "TIME_SERIES_TIER_FLOORS", "TIME_SERIES_SPREAD_BAND",
        "TIME_SERIES_INTERVAL_PROB_DISCOUNT", "BUDGET_FRACTION", "SAME_TITLE_SIZE_CAP",
        "TRADE_CATEGORIES", "TRADE_TAGS"})

    @classmethod
    def _by_value_binders(cls, directory: pathlib.Path) -> list[tuple[str, str, str]]:
        """(module, bound name, toggle) for every `from X import <toggle>` in
        directory's modules."""
        found = []
        for path in sorted(directory.glob("*.py")):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    found += [(path.stem, alias.asname or alias.name, alias.name)
                              for alias in node.names if alias.name in cls._TOGGLES]
        return found

    def test_every_package_binder_reads_the_pre_flip_value(self, monkeypatch):
        binders = self._by_value_binders(pathlib.Path(config.__file__).parent)
        # Not vacuous: the four known binders are among them
        assert {("backtester", "BUDGET_FRACTION"), ("backtester", "SAME_TITLE_SIZE_CAP"),
                ("backtester", "TIME_SERIES_INTERVAL_PROB_DISCOUNT"),
                ("backtest", "TIME_SERIES_INTERVAL_PROB_DISCOUNT")} <= {
            (module, toggle) for module, _, toggle in binders}
        apply_pre_toggle_defaults(monkeypatch)
        assert live_settings() == LiveSettings(
            tier_floors=True, spread_band=(0.0, 1.0), interval_discount=0.75, size_cap=0.20,
            same_title_size_cap=1.0, categories=None, tags=None)
        for module, bound, toggle in binders:
            assert getattr(importlib.import_module(f"kalshi_betting.{module}"), bound) == (
                getattr(config, toggle)), (module, toggle)

    def test_no_test_module_binds_a_toggle_by_value(self):
        # A by-value import would freeze the shipped value past every patch
        assert self._by_value_binders(pathlib.Path(__file__).parent) == []
