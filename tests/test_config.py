"""Tests for config.py fee helpers and PROJECT_ROOT."""
import math

import pytest

from kalshi_betting.config import (
    PROJECT_ROOT,
    TAKER_FEE_RATE,
    fee_leg_exact,
    fee_per_pair_approx,
)


class TestProjectRoot:
    def test_resolves_to_repo_root(self):
        assert (PROJECT_ROOT / "kalshi_betting" / "config.py").exists()

    def test_not_hardcoded_path(self):
        assert "zdhoffman" not in str(PROJECT_ROOT)
        assert "Users" not in str(PROJECT_ROOT)


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
