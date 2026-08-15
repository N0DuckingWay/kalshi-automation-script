"""Tests for config.py fee helpers, the V2 order-path switch, and PROJECT_ROOT."""
import math
import pathlib
from datetime import date

import pytest

from kalshi_betting import config
from kalshi_betting.config import (
    MAX_DEADLINE_GAP_DAYS,
    MIN_PRICE_DIFF_LONG_GAP,
    MIN_PRICE_DIFF_SHORT_GAP,
    PROJECT_ROOT,
    SHORT_DEADLINE_GAP_DAYS,
    TAKER_FEE_RATE,
    V2_ORDERS_START,
    fee_leg_exact,
    fee_per_pair_approx,
    min_price_diff_for_gap,
    v2_orders_active,
)


class TestProjectRoot:
    def test_resolves_to_repo_root(self):
        assert (PROJECT_ROOT / "kalshi_betting" / "config.py").exists()

    def test_derived_from_file_not_hardcoded(self):
        # PROJECT_ROOT must track config.py's actual location (two levels up:
        # kalshi_betting/config.py -> kalshi_betting/ -> repo root), not a
        # hardcoded absolute path baked in at some point in time.
        assert PROJECT_ROOT == pathlib.Path(config.__file__).resolve().parent.parent


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


class TestV2OrdersActive:
    """The order-path switch: automatic on V2_ORDERS_START (the combo
    migration), with V2_ORDERS_FORCE as a two-way manual override."""

    def test_start_date_is_the_combo_migration(self):
        # Changing this changes when real orders switch wire format.
        assert V2_ORDERS_START == date(2026, 8, 17)

    def test_day_before_start_is_legacy(self):
        assert v2_orders_active(today=date(2026, 8, 16)) is False

    def test_start_date_itself_is_v2(self):
        # Inclusive: the switch happens ON the migration date, not the day after.
        assert v2_orders_active(today=date(2026, 8, 17)) is True

    def test_after_start_is_v2(self):
        assert v2_orders_active(today=date(2026, 8, 18)) is True

    def test_force_true_overrides_an_early_date(self, monkeypatch):
        monkeypatch.setattr(config, "V2_ORDERS_FORCE", True)
        assert v2_orders_active(today=date(2026, 1, 1)) is True

    def test_force_false_overrides_a_late_date(self, monkeypatch):
        # The emergency fallback: legacy stays reachable forever, which is why
        # the legacy order path is kept rather than deleted at the migration.
        monkeypatch.setattr(config, "V2_ORDERS_FORCE", False)
        assert v2_orders_active(today=date(2030, 1, 1)) is False

    def test_force_defaults_to_none(self):
        # None = "the date decides"; a committed bool would mean the repo ships
        # a manual override as its production default.
        assert config.V2_ORDERS_FORCE is None

    def test_omitted_today_uses_the_current_utc_date(self, monkeypatch):
        # Production passes no date. Verified by moving the START date rather
        # than faking the clock: with a start far in the past the live call
        # must be True, and far in the future it must be False.
        monkeypatch.setattr(config, "V2_ORDERS_START", date(2000, 1, 1))
        assert v2_orders_active() is True
        monkeypatch.setattr(config, "V2_ORDERS_START", date(2100, 1, 1))
        assert v2_orders_active() is False
