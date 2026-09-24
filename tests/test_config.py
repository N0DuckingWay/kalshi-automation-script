"""Tests for config.py fee helpers, the time-series probability model, the
leg-side tuples, and PROJECT_ROOT."""
import math
import pathlib

import pytest

from kalshi_betting import config
from kalshi_betting.config import (
    BUDGET_FRACTION,
    MAX_DEADLINE_GAP_DAYS,
    MIN_PRICE_DIFF_LONG_GAP,
    MIN_PRICE_DIFF_SHORT_GAP,
    PROJECT_ROOT,
    SAME_TITLE_LEG_SIDES,
    SHORT_DEADLINE_GAP_DAYS,
    TAKER_FEE_RATE,
    TIME_SERIES_INTERVAL_PROB_DISCOUNT,
    TIME_SERIES_LEG_SIDES,
    fee_leg_exact,
    fee_per_pair_approx,
    max_affordable_pairs,
    min_price_diff_for_gap,
    time_series_profit_prob,
)


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
        # ...and the tier helper never reads it: the live path sees no band
        # even when the backtest's default is not "no band"
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

    def test_scanner_cap_bounds_the_sizer(self):
        # The invariant the whole design rests on: the scanner calls with the
        # MAXIMUM fraction and the MINIMUM (best-level) price sum, so its answer
        # can never be smaller than the sizer's, whatever Kelly returns.
        balance, best_sum = 250_000, 0.82
        cap = max_affordable_pairs(balance, best_sum)
        for kelly_f in (0.01, 0.06, 0.13, BUDGET_FRACTION):
            for prefix_sum in (best_sum, 0.85, 0.90, 0.94):
                assert max_affordable_pairs(balance, prefix_sum, kelly_f) <= cap


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
    """DR-73: same-event deadline ladders ship OFF.

    The switch changes which pairs exist, and turning it on was measured to
    deploy 98% of a $10,000 balance into 6 trades at -31% market-implied EV
    (see the constant's own comment). It must not be flipped by accident, and
    a default flip must fail a test rather than reach a Monday prod run.
    """

    def test_same_event_ladders_ship_off(self):
        assert config.TIME_SERIES_SAME_EVENT_LADDERS is False
