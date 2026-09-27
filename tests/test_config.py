"""Tests for config.py fee helpers, the time-series probability model, the
leg-side tuples, the deadline-gap tier (with the backtest's spread band and
tier-floors switch), the live toggles (LiveSettings, the live spread rule and
the per-pair Kelly bound), and PROJECT_ROOT."""
import dataclasses
import math
import pathlib
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from kalshi_betting import config
from kalshi_betting.config import (
    BUDGET_FRACTION,
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
    TIME_SERIES_INTERVAL_PROB_DISCOUNT,
    TIME_SERIES_LEG_SIDES,
    LiveSettings,
    fee_leg_exact,
    fee_per_pair_approx,
    live_settings,
    max_affordable_pairs,
    max_kelly_fraction,
    min_price_diff_for_gap,
    time_series_profit_prob,
    time_series_spread_refusal,
)
from kalshi_betting.scanner import leg_prices
from kalshi_betting.strategy import compute_trade


def _settings(tier_floors=True, spread_band=(0.0, 1.0), interval_discount=0.75, size_cap=0.20):
    """A LiveSettings with every field named, defaulting to today's values."""
    return LiveSettings(tier_floors=tier_floors, spread_band=spread_band,
                        interval_discount=interval_discount, size_cap=size_cap)


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

    def test_flow_through_fixture(self):
        # pA 0.30, pB 0.60 → gap 0.30 → p = 1 - 0.75 * 0.30 = 0.775
        assert time_series_profit_prob(0.30, 0.60) == pytest.approx(0.775)

    def test_matches_definition_from_constant(self):
        for pA, pB in [(0.10, 0.25), (0.30, 0.60), (0.40, 0.55), (0.30, 0.70)]:
            expected = 1.0 - TIME_SERIES_INTERVAL_PROB_DISCOUNT * (pB - pA)
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
        # The user's conservative choice ("prices converge by 25%") — pinned so
        # a silent retune is visible in review; must stay inside [0, 1]
        assert TIME_SERIES_INTERVAL_PROB_DISCOUNT == 0.75
        assert 0.0 <= TIME_SERIES_INTERVAL_PROB_DISCOUNT <= 1.0

    def test_explicit_k_overrides_the_constant(self):
        # The backtester's calibration sweep passes one k per simulation; the
        # override must win over the config constant — 1 - 0.50 * 0.30 = 0.85 —
        # without mutating it, since the live sizer keeps reading it.
        assert time_series_profit_prob(0.30, 0.60, k=0.50) == pytest.approx(0.85)
        assert time_series_profit_prob(0.30, 0.60, k=1.0) == pytest.approx(0.70)
        assert config.TIME_SERIES_INTERVAL_PROB_DISCOUNT == 0.75

    def test_k_none_is_identical_to_omitting_it(self):
        # None is the sentinel for "read the config constant", so the sweep's
        # default point and the live sizer's override-free call must agree
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
        # ...and the tier helper never reads it: the live path never sees the
        # backtest's band, even when its default is not "no band" (the live
        # band is config.TIME_SERIES_SPREAD_BAND, read through LiveSettings)
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
                == max_affordable_pairs(100_000, 0.50, BUDGET_FRACTION))

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

    @pytest.mark.parametrize("k, cap", [(0.75, 0.20), (0.40, 1.0), (0.80, 1.0), (0.60, 0.35)])
    def test_scanner_cap_bounds_the_sizer(self, monkeypatch, k, cap):
        # The invariant the whole design rests on: enrichment bounds its
        # average at max_kelly_fraction(pair type, settings) over the MINIMUM
        # (best-level) price sum, so its count can never be smaller than the
        # sizer's, whatever Kelly returns. Checked against the capped f* that
        # compute_trade actually returns, for both pair types. compute_trade
        # still reads k from config (TIME_SERIES_INTERVAL_PROB_DISCOUNT, at
        # call time) and its cap from strategy's BUDGET_FRACTION binding, so
        # both are patched to the settings under test.
        from kalshi_betting import strategy

        monkeypatch.setattr(config, "TIME_SERIES_INTERVAL_PROB_DISCOUNT", k)
        monkeypatch.setattr(strategy, "BUDGET_FRACTION", cap)
        settings = config.LiveSettings(
            tier_floors=True, spread_band=(0.0, 1.0), interval_discount=k, size_cap=cap)
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

        # Time-series books are UNCROSSED — the later YES ask pB at or above
        # its own YES bid (1 - nB) — which enrichment's fail-closed fallback
        # and crossed-book guard enforce on every pair it keeps
        ts = [pair("time_series", pA, pB, 1 - pA, nB)
              for pA, pB, nB in [(0.10, 0.70, 0.30), (0.05, 0.85, 0.15), (0.20, 0.50, 0.50),
                                 (0.30, 0.60, 0.40), (0.12, 0.33, 0.70)]]
        st = [pair("same_title", pA, pB, nA, 0.70)
              for pA, pB, nA in [(0.70, 0.30, 0.20), (0.60, 0.31, 0.44), (0.55, 0.30, 0.45)]]
        sized = {"time_series": 0, "same_title": 0}
        for p in ts + st:
            spec = compute_trade(p, balance)
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
    """LiveSettings is one run's live toggles, validated and normalised on
    construction. It is frozen, and dataclasses.replace re-runs __post_init__,
    so an override is validated exactly as config.py's values are."""

    @pytest.mark.parametrize("bad", [1, 0, None, "True", "false", 1.0])
    def test_tier_floors_must_be_exactly_a_bool(self, bad):
        # min_price_diff_for_gap drops the tier only on an explicit False, so
        # a truthy/falsy stand-in would silently read as "on"
        with pytest.raises(ValueError, match="tier_floors"):
            _settings(tier_floors=bad)

    @pytest.mark.parametrize("band", [
        (0.5, 0.5), (0.6, 0.5), (-0.1, 0.5), (0.2, 1.1), (float("nan"), 0.5),
        (0.1,), (0.1, 0.2, 0.3), "ab", 0.5, None,
    ])
    def test_band_is_validated_by_the_backtest_band_validator(self, band):
        # None is refused too, never read as "no band": the backtest band's
        # validator would resolve it to BACKTEST_DEFAULT_SPREAD_BAND, a
        # backtest constant the live path must never read
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

    # 1e-7 and 1e-6 are positive but within PRICE_EPSILON of 0: they round to
    # zero steps, and would otherwise normalise to a cap of 0.0, on no cell
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
        # 0.35 is 0.35000000000000003 as 0.05 * 7; normalised with the very
        # expression backtester.SIZE_CAP_SWEEP builds its grid with, so a live
        # cap is float-equal to one of the grid's cells
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
        monkeypatch.setattr(config, "TIME_SERIES_TIER_FLOORS", False)
        monkeypatch.setattr(config, "TIME_SERIES_SPREAD_BAND", (0.0, 0.5))
        monkeypatch.setattr(config, "TIME_SERIES_INTERVAL_PROB_DISCOUNT", 0.8)
        monkeypatch.setattr(config, "BUDGET_FRACTION", 1.0)
        assert live_settings() == _settings(False, (0.0, 0.5), 0.8, 1.0)

    def test_live_settings_refuses_an_invalid_constant(self, monkeypatch):
        monkeypatch.setattr(config, "BUDGET_FRACTION", 0.37)
        with pytest.raises(ValueError, match="size_cap"):
            live_settings()

    def test_the_shipped_values_resolve(self):
        s = live_settings()
        assert s == LiveSettings(
            tier_floors=config.TIME_SERIES_TIER_FLOORS,
            spread_band=config.TIME_SERIES_SPREAD_BAND,
            interval_discount=config.TIME_SERIES_INTERVAL_PROB_DISCOUNT,
            size_cap=config.BUDGET_FRACTION,
        )
        assert type(s.tier_floors) is bool


class TestTimeSeriesSpreadRefusal:
    """config.time_series_spread_refusal is the one live definition of the
    time-series spread rule: positivity, then the entry floor, then the band's
    ceiling, with backtester._find_entry's PRICE_EPSILON placement."""

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
        # A ceiling under a tier: a 0.12 spread at 5 days is BOTH under the
        # 0.15 tier and over the 0.10 ceiling. _find_entry tests the floor
        # first, and so must the live rule, or its counts would disagree
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
    """max_kelly_fraction is enrichment's affordability bound: the largest
    capped Kelly fraction a sizer pricing with the same k and cap can return
    for a pair type. Compared with
    ==, deliberately: 1 - 0.8 is 0.19999999999999996, and the bound must be
    exactly 0.20 or max_contracts shifts by one on round-number books."""

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


class TestDescribeTimeSeriesRule:
    """describe_time_series_rule words the finder's always-logged
    "Time-series entry rule" line and its floor-refusal lines: it must say
    "no spread band" for (0, 1) alone, and name any band with a floor or a
    ceiling of its own."""

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
