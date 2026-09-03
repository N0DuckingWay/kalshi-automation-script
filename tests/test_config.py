"""Tests for config.py fee helpers and PROJECT_ROOT."""
import math
import pathlib

import pytest

from kalshi_betting import config
from kalshi_betting.config import (
    MAX_DEADLINE_GAP_DAYS,
    MIN_PRICE_DIFF_LONG_GAP,
    MIN_PRICE_DIFF_SHORT_GAP,
    PROJECT_ROOT,
    SHORT_DEADLINE_GAP_DAYS,
    TAKER_FEE_RATE,
    fee_leg_exact,
    fee_per_pair_approx,
    min_price_diff_for_gap,
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
        nA, pB = 0.35, 0.45
        expected = TAKER_FEE_RATE * (nA * (1 - nA) + pB * (1 - pB))
        assert fee_per_pair_approx(nA, pB) == pytest.approx(expected)

    def test_is_underestimate_vs_exact(self):
        # The approximation should be <= the sum of two exact leg fees at n=1,
        # because ceiling rounding always rounds up.
        for nA, pB in [(0.3, 0.4), (0.2, 0.5), (0.45, 0.35)]:
            approx = fee_per_pair_approx(nA, pB)
            exact_sum = fee_leg_exact(1, nA) + fee_leg_exact(1, pB)
            assert approx <= exact_sum + 1e-9, f"approx {approx} > exact {exact_sum} for nA={nA} pB={pB}"

    def test_symmetric(self):
        assert fee_per_pair_approx(0.3, 0.4) == pytest.approx(fee_per_pair_approx(0.4, 0.3))


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
