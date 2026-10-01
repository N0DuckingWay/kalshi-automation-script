"""Tests for config.py fee helpers, the time-series probability model, the
leg-side tuples, the deadline-gap tier (with the backtest's spread band and
tier-floors switch), the live toggles (LiveSettings and its helpers), the saved
live defaults (their file's reader and writer, the seed, and the comparison of
two sets of defaults), the weekly run schedule (ScheduledRun), the values
config.py ships, conftest's apply_pre_toggle_defaults, the V2 order path's
self-trade-prevention value, the startup check that refuses any order path but
"v2" (order_api_version_error), the order-write pacer's budget, the live-run
lock's exit code, file and waits, and PROJECT_ROOT."""
import ast
import dataclasses
import importlib
import importlib.util
import json
import logging
import math
import os
import pathlib
import random
import re
import sys
import threading
from datetime import UTC, date, datetime, timedelta
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pytest

from kalshi_betting import backtester, config, scanner
from kalshi_betting.config import (
    MAX_DEADLINE_GAP_DAYS,
    MIN_PRICE_DIFF_LONG_GAP,
    MIN_PRICE_DIFF_SHORT_GAP,
    PRICE_EPSILON,
    PROJECT_ROOT,
    SAME_TITLE_LEG_SIDES,
    SCHEDULED_RUN,
    SHORT_DEADLINE_GAP_DAYS,
    SPREAD_ABOVE_CEILING,
    SPREAD_BELOW_FLOOR,
    SPREAD_NOT_POSITIVE,
    TAKER_FEE_RATE,
    TIME_SERIES_LEG_SIDES,
    LiveSettings,
    ScheduledRun,
    fee_leg_exact,
    fee_per_pair_approx,
    held_pair_fraction,
    kelly_budget,
    leg_cash_cents,
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
              same_title_size_cap=1.0, categories=None, tags=None, add_to_held_pairs=False):
    """A LiveSettings with every toggle named, defaulting to the pre-toggle
    values conftest's apply_pre_toggle_defaults pins."""
    return LiveSettings(tier_floors=tier_floors, spread_band=spread_band,
                        interval_discount=interval_discount, size_cap=size_cap,
                        same_title_size_cap=same_title_size_cap,
                        categories=categories, tags=tags,
                        add_to_held_pairs=add_to_held_pairs)


# An origin as read_saved_live_defaults writes it: any origin but config.py's
# means the toggles are the saved live defaults
_SAVED_ORIGIN = "live_defaults.json, saved 2026-09-27T21:05:13Z"


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


class TestOrderApiVersionError:
    """config.order_api_version_error, the startup check main.py and
    v2_probe run: exactly the str "v2" passes, and any other value gets a
    message that names it and points only at "v2"."""

    # Values the check must refuse
    _REFUSED = ["legacy", "V2", " v2", "v2 ", "", None, 2, b"v2"]

    def test_the_shipped_value_is_v2_and_passes(self):
        assert config.ORDER_API_VERSION == "v2"
        assert config.order_api_version_error() is None

    @pytest.mark.parametrize("value", _REFUSED, ids=repr)
    def test_any_other_value_is_refused_by_name(self, monkeypatch, value):
        monkeypatch.setattr(config, "ORDER_API_VERSION", value)
        message = config.order_api_version_error()
        assert isinstance(message, str) and message
        assert "\n" not in message
        assert repr(value) in message
        assert '"v2"' in message
        assert config.V2_ORDER_PATH in message

    @pytest.mark.parametrize("value", _REFUSED, ids=repr)
    def test_the_message_never_names_another_value_to_switch_to(self, monkeypatch, value):
        monkeypatch.setattr(config, "ORDER_API_VERSION", value)
        message = config.order_api_version_error()
        # Apart from the refused value, the only quoted value and the only
        # suggested setting in the message are "v2"
        rest = message.replace(repr(value), "", 1)
        assert re.findall(r'"([^"]*)"', rest) == ["v2", "v2"]
        assert re.findall(r"ORDER_API_VERSION\s*=+\s*(\S+)", rest) == ['"v2"']
        # "legacy" appears once, naming the retired endpoint, and the message
        # ends on the fix
        prefix = f"config.ORDER_API_VERSION is {value!r}, "
        assert message.startswith(prefix)
        assert message[len(prefix):].lower().count("legacy") == 1
        assert message.endswith('Set ORDER_API_VERSION = "v2" in config.py.')

    def test_it_reads_the_setting_at_call_time(self, monkeypatch):
        monkeypatch.setattr(config, "ORDER_API_VERSION", "legacy")
        assert "'legacy'" in config.order_api_version_error()
        monkeypatch.setattr(config, "ORDER_API_VERSION", "v2")
        assert config.order_api_version_error() is None

    def test_a_str_subclass_equal_to_v2_is_refused(self, monkeypatch):
        # Only the exact str type passes
        class Named(str):
            pass

        monkeypatch.setattr(config, "ORDER_API_VERSION", Named("v2"))
        assert config.order_api_version_error() is not None

    # A source scan for text that sets ORDER_API_VERSION to "legacy".
    # _JOIN_STRINGS joins string literals split across lines (and escaped
    # quotes are read as plain quotes); _SETS_TO_LEGACY finds the name
    # ORDER_API_VERSION followed within 60 characters by a quoted legacy
    _JOIN_STRINGS = re.compile(r"""(["'])[ \t]*\n[ \t]*[rbfuRBFU]{0,2}\1""")
    _SETS_TO_LEGACY = re.compile(r"""ORDER_API_VERSION\b[\s\S]{0,60}?["']legacy["']""",
                                 re.IGNORECASE)

    # Source-text shapes (comments, docstrings and split string literals)
    # that the scan must catch
    _REMEDY_SHAPES = (
        '    FAILS, set config.ORDER_API_VERSION = "legacy" to hold the bot on the\n',
        '    decision to keep ORDER_API_VERSION = "v2" (or to flip it to "legacy").\n',
        r'            "*** CHECK THE ACCOUNT. *** Set config.ORDER_API_VERSION = \"legacy\" to hold "'
        '\n            "the bot on the legacy order path."\n',
        '            "would add to this exposure. *** FLATTEN THE POSITION MANUALLY. *** "\n'
        """            'Set config.ORDER_API_VERSION = "legacy" to hold the bot on the legacy path.'\n""",
        '            f"{final}; None means the lookup failed). *** CHECK THE ACCOUNT MANUALLY. *** "\n'
        """            'Set config.ORDER_API_VERSION = "legacy" to hold the bot on the legacy path.'\n""",
        '        print(\n'
        """            'Set config.ORDER_API_VERSION = "legacy" to hold the bot on the legacy order '\n"""
        '            "path. Both --step no-mapping and --step unfillable-ask have to PASS before "\n',
        '        fully intact and unmodified so flipping ORDER_API_VERSION back to\n'
        '        "legacy" is an instant, code-free rollback if the V2 mapping misbehaves\n',
        '        automatically on a state we cannot model. Remedy: set\n'
        '        config.ORDER_API_VERSION = "legacy", the instant rollback to the\n',
        '        " must flatten this account position; set config.ORDER_API_VERSION ="\n'
        r'        " \"legacy\" to revert to the proven order path.",'
        '\n',
        '            # path is in use (ORDER_API_VERSION="legacy") a pair spanning\n',
    )

    @classmethod
    def _legacy_settings(cls, text: str) -> list[str]:
        """Every place `text` names ORDER_API_VERSION and then, within 60
        characters, a quoted legacy, after joining split string literals."""
        joined = cls._JOIN_STRINGS.sub("", text).replace('\\"', '"').replace("\\'", "'")
        return [m.group(0) for m in cls._SETS_TO_LEGACY.finditer(joined)]

    def test_the_scan_catches_every_shape_the_package_has_used(self):
        # Each sample is caught; an unquoted mention of the retired endpoint
        # is not
        for sample in self._REMEDY_SHAPES:
            assert self._legacy_settings(sample), sample
        for sample in ('ORDER_API_VERSION = "legacy"', "ORDER_API_VERSION='legacy'",
                       'ORDER_API_VERSION == "legacy"', 'order_api_version = "LEGACY"',
                       'flip ORDER_API_VERSION back to\n    "legacy"'):
            assert self._legacy_settings(sample), sample
        # The retired endpoint may still be named, unquoted, beside the switch
        assert not self._legacy_settings(
            'config.ORDER_API_VERSION is None: Kalshi retired the legacy endpoint'
        )

    def test_no_source_text_sets_the_switch_to_legacy(self):
        # No file in the package, comments and docstrings included, tells
        # anyone to set ORDER_API_VERSION to "legacy": that path does not exist
        package = PROJECT_ROOT / "kalshi_betting"
        hits = []
        for path in sorted(package.rglob("*.py")):
            hits += [f"{path.name}: {found!r}"
                     for found in self._legacy_settings(path.read_text(encoding="utf-8"))]
        assert hits == []

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
        # A pair's NO leg takes two places at once (its own and one held for
        # its YES leg), and trader._WritePacer refuses a smaller burst at import
        assert config.ORDER_WRITE_BURST >= 2
        assert config.ORDER_WRITES_PER_SECOND > 0


class TestLiveRunLockSettings:
    """The exit code a run stopped by the live-run lock returns, and where the
    lock lives. tests/conftest.py points config.LIVE_RUN_LOCK_FILE and the
    waits elsewhere for every test, so the shipped values are read from a
    fresh load of config.py."""

    @staticmethod
    def _config_as_shipped(monkeypatch):
        """
        Load config.py afresh, under a name of its own, with no test's patches.

        Args:
            monkeypatch (pytest.MonkeyPatch): Removes the fresh module from
                sys.modules afterwards.

        Returns:
            module: A second copy of config.py, as it ships.
        """
        spec = importlib.util.spec_from_file_location("_config_as_shipped", config.__file__)
        fresh = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, fresh)
        spec.loader.exec_module(fresh)
        return fresh

    def test_run_in_progress_is_an_exit_code_of_its_own(self):
        codes = {name: getattr(config, name) for name in dir(config)
                 if name.startswith("EXIT_")}
        assert config.EXIT_RUN_IN_PROGRESS == 50
        assert list(codes.values()).count(config.EXIT_RUN_IN_PROGRESS) == 1
        # Not the interpreter's crash code or argparse's usage-error code
        assert config.EXIT_RUN_IN_PROGRESS not in (1, 2)

    def test_the_shipped_lock_is_in_the_home_folder(self, monkeypatch):
        shipped = self._config_as_shipped(monkeypatch)
        # One lock for every checkout and worktree trading the account
        assert shipped.LIVE_RUN_LOCK_FILE == (
            pathlib.Path.home() / ".kalshi_betting" / "live_run.lock")
        assert not shipped.LIVE_RUN_LOCK_FILE.is_relative_to(shipped.PROJECT_ROOT)
        # Not vacuous: this test itself sees conftest's redirect
        assert config.LIVE_RUN_LOCK_FILE != shipped.LIVE_RUN_LOCK_FILE

    def test_the_shipped_wait_rides_out_a_check_and_is_far_shorter_than_a_run(
        self, monkeypatch,
    ):
        shipped = self._config_as_shipped(monkeypatch)
        assert 0 < shipped.LIVE_RUN_LOCK_POLL_SECONDS < shipped.LIVE_RUN_LOCK_WAIT_SECONDS
        # Several tries within the wait, and a small part of the job timeout
        assert shipped.LIVE_RUN_LOCK_WAIT_SECONDS >= 10 * shipped.LIVE_RUN_LOCK_POLL_SECONDS
        assert shipped.LIVE_RUN_LOCK_WAIT_SECONDS * 100 < shipped.SCHEDULER_JOB_TIMEOUT_SECONDS


class TestDefaultsServerCheckoutWait:
    """How long a start that finds the defaults server's port taken waits for the answer."""

    def test_the_wait_outlasts_two_idle_browser_connections(self):
        # The running server answers one connection at a time and holds a
        # connection that sends nothing for DEFAULTS_SERVER_SOCKET_TIMEOUT_SECONDS,
        # so the question must be able to wait behind two such connections
        assert (config.DEFAULTS_SERVER_CHECKOUT_TIMEOUT_SECONDS
                > 2 * config.DEFAULTS_SERVER_SOCKET_TIMEOUT_SECONDS)
        # Short enough that a listener that never answers is refused promptly
        assert config.DEFAULTS_SERVER_CHECKOUT_TIMEOUT_SECONDS <= 30


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

    def test_cash_cents_is_keyword_only(self):
        # A fourth positional argument cannot be read as the cash by accident
        with pytest.raises(TypeError):
            max_affordable_pairs(100_000, 0.50, 0.20, 5_000)

    def test_the_cash_bounds_the_budget(self):
        # $1,000 x 50% = $500 at $0.50 a pair buys 1,000 pairs; $100 of cash
        # buys 200, and cash above the $500 share changes nothing
        assert max_affordable_pairs(100_000, 0.50, 0.50) == 1_000
        assert max_affordable_pairs(100_000, 0.50, 0.50, cash_cents=10_000) == 200
        assert max_affordable_pairs(100_000, 0.50, 0.50, cash_cents=50_000) == 1_000
        assert max_affordable_pairs(100_000, 0.50, 0.50, cash_cents=900_000) == 1_000
        assert max_affordable_pairs(100_000, 0.50, 0.50, cash_cents=0) == 0

    def test_float_identity_with_the_expression_it_replaced(self):
        # With no cash, or cash that does not bind, the count is exactly the one
        # int((bankroll_cents / 100.0) * fraction / price_sum) gave, float for float
        rng = random.Random(20260929)
        grid = [0.05 * i for i in range(1, 21)] + [round(1 - 0.8, 12), 0.19999999999999996]
        for _ in range(20_000):
            bankroll = rng.choice([rng.randrange(0, 10_000), rng.randrange(0, 10**9)])
            price_sum = rng.choice([rng.uniform(0.0001, 1.9999),
                                    round(rng.randrange(1, 200) * 0.01, 2)])
            fraction = rng.choice([rng.random(), rng.choice(grid)])
            old = int((bankroll / 100.0) * fraction / price_sum)
            assert max_affordable_pairs(bankroll, price_sum, fraction) == old
            assert max_affordable_pairs(bankroll, price_sum, fraction,
                                        cash_cents=bankroll) == old

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
            # ... and still does when the cash binds both the same way
            for cash in (500, 5_000, 50_000):
                capped = compute_trade(p, balance, settings=settings, cash_cents=cash)
                if capped is not None:
                    assert capped.x <= max_affordable_pairs(
                        balance, sum(leg_prices(p)), bound, cash_cents=cash)
                    assert capped.total_cost_with_fees <= cash / 100 + 1e-9
        # Non-vacuous for both types at every setting
        assert sized["time_series"] > 0 and sized["same_title"] > 0, sized


class TestKellyBudget:
    """kelly_budget is the one rule for what a trade may spend: a fraction of
    the portfolio value, never more than the cash on hand."""

    def test_a_share_of_the_bankroll(self):
        assert kelly_budget(1_000.0, 0.20) == pytest.approx(200.0)

    def test_the_cash_binds_when_it_is_smaller(self):
        assert kelly_budget(1_000.0, 0.20, 150.0) == pytest.approx(150.0)
        assert kelly_budget(1_000.0, 0.20, 500.0) == pytest.approx(200.0)
        assert kelly_budget(1_000.0, 0.20, 0.0) == 0.0

    def test_no_cash_returns_the_product_exactly(self):
        # The sizer's budget before the cash existed was bankroll * fraction;
        # with no cash the float must be that product, bit for bit
        rng = random.Random(7)
        for _ in range(1_000):
            bankroll, fraction = rng.uniform(0, 1e7), rng.random()
            assert kelly_budget(bankroll, fraction) == bankroll * fraction

    def test_units_in_are_units_out(self):
        assert kelly_budget(100_000, 0.5, 20_000) == 20_000


class TestLegCashCents:
    """leg_cash_cents is the one dollars -> whole cents rounding for what an
    order leg draws, shared by the shard funder and select_portfolio."""

    def test_rounds_up_to_the_cent(self):
        assert leg_cash_cents(1.001) == 101
        assert leg_cash_cents(1.0) == 100
        assert leg_cash_cents(0.0) == 0

    def test_float_noise_does_not_claim_a_cent(self):
        # 0.07 * 100 is 7.000000000000001
        assert leg_cash_cents(0.07) == 7
        assert leg_cash_cents(0.1 + 0.2) == 30


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

    @pytest.mark.parametrize("bad", [1, 0, None, "True", 1.0])
    def test_add_to_held_pairs_must_be_exactly_a_bool(self, bad):
        # A saved file's JSON 1 or "true" must never read as on
        with pytest.raises(ValueError, match="add_to_held_pairs"):
            _settings(add_to_held_pairs=bad)
        with pytest.raises(ValueError, match="add_to_held_pairs"):
            dataclasses.replace(_settings(), add_to_held_pairs=bad)

    def test_add_to_held_pairs_defaults_to_off(self):
        # A construction that does not name it reads it as off (a saved file
        # that leaves it out: TestSavedLiveDefaults)
        assert LiveSettings(True, (0.0, 1.0), 0.75, 0.2).add_to_held_pairs is False
        assert _settings(add_to_held_pairs=True).add_to_held_pairs is True

    def test_every_optional_toggle_is_a_field_with_a_default_at_the_end(self):
        # A toggle a saved file may leave out must have a default to read as,
        # and sit after every required toggle, so a positional construction
        # of the required ones still builds
        names = [f.name for f in dataclasses.fields(LiveSettings) if f.compare]
        assert list(config.LIVE_TOGGLE_FIELDS) == names
        optional = list(config._OPTIONAL_TOGGLES)
        assert optional and names[-len(optional):] == optional
        for f in dataclasses.fields(LiveSettings):
            if f.name in optional:
                assert f.default is not dataclasses.MISSING, f.name
                assert config._TOGGLE_DEFAULTS[f.name] == f.default

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
        monkeypatch.setattr(config, "ADD_TO_HELD_PAIRS", not config.ADD_TO_HELD_PAIRS)
        expected = _settings(True, (0.1, 0.6), 0.6, 0.35, 0.25,
                             ("Economics",), ("Oil & Gas",),
                             add_to_held_pairs=not shipped.add_to_held_pairs)
        assert live_settings() == expected
        assert all(getattr(expected, name) != getattr(shipped, name)
                   for name in config.LIVE_TOGGLE_FIELDS)

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
            add_to_held_pairs=config.ADD_TO_HELD_PAIRS,
        )
        assert type(s.tier_floors) is bool and type(s.add_to_held_pairs) is bool

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
        assert s.add_to_held_pairs is False
        # Defaulted and last of the toggles: the first five fields keep their
        # positions, so a positional construction of up to seven still builds;
        # origin, not a toggle, comes after them
        assert list(config.LIVE_TOGGLE_FIELDS)[-3:] == [
            "categories", "tags", "add_to_held_pairs"]
        assert [f.name for f in dataclasses.fields(LiveSettings)][-1] == "origin"
        assert LiveSettings(True, (0.0, 1.0), 0.75, 0.2, 1.0, ("Sports",),
                            ("Hockey",)).tags == ("Hockey",)

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


class TestHeldPairFraction:
    """config.held_pair_fraction, the one definition of an add-on's size.

    Kelly sizes the whole position: a held pair (its held stake, the worth
    plus the fees paid for it) holds at most `fraction` of the portfolio
    value in all, and an add-on never takes a bigger share than a new pair
    would. Its budget is then kelly_budget(portfolio value, that share,
    cash), so it never spends more than the cash either. A wrong answer here
    stakes real money on a pair the account already holds.
    """

    def test_a_pair_holding_little_sizes_as_a_new_pair(self):
        # Nothing held: exactly the new-pair share
        assert held_pair_fraction(0.10, 0.0, 10_000.0) == 0.10
        # A little held and the cash binding: the add-on's budget is exactly a
        # new pair's, all $150 of the cash (10% of $10,000 less $1 is $999)
        little = held_pair_fraction(0.10, 1.0, 10_000.0)
        assert little == 0.10 - 1.0 / 10_000.0
        assert (kelly_budget(10_000.0, little, 150.0)
                == kelly_budget(10_000.0, 0.10, 150.0) == 150.0)

    def test_the_whole_position_binds_when_the_pair_holds_part_of_its_share(self):
        # 10% of a $211 portfolio is $21.10; the pair stakes $8, so the add-on
        # may take $13.10, and $150 of cash does not bind
        share = held_pair_fraction(0.10, 8.0, 211.0)
        assert share == 0.10 - 8.0 / 211.0
        assert kelly_budget(211.0, share, 150.0) == pytest.approx(13.10, abs=1e-12)
        # With $10 of cash the cash binds instead, as it does for any trade
        assert kelly_budget(211.0, share, 10.0) == 10.0

    def test_a_stake_with_its_fees_leans_safe_at_unchanged_prices(self):
        # A pair bought its whole 10% share of $205.00: $19.60 of contracts
        # and $0.90 of fees. At unchanged prices the portfolio value is
        # $204.10, the fees being spent. Its stake with the fees, $20.50, is
        # above 10% of $204.10 ($20.41): nothing is missing
        assert held_pair_fraction(0.10, 19.60 + 0.90, 204.10) < 0
        # With the fees left out of the stake, $0.81 would read as missing,
        # enough for one more contract pair at these prices every run
        share = held_pair_fraction(0.10, 19.60, 204.10)
        assert share > 0
        assert kelly_budget(204.10, share, 184.50) == pytest.approx(0.81, abs=1e-9)

    @pytest.mark.parametrize("held_stake", [21.1000001, 25.0, 211.0])
    def test_a_pair_above_its_kelly_share_adds_nothing(self, held_stake):
        # Just above 10% of $211 (a stake of exactly $21.10 would rest on
        # 21.1 / 211.0 rounding to exactly 0.1 in floating point), and well above
        assert held_pair_fraction(0.10, held_stake, 211.0) <= 0

    @pytest.mark.parametrize("portfolio_value", [0.0, -5.0, math.nan, math.inf])
    def test_no_portfolio_value_adds_nothing(self, portfolio_value):
        # Not a positive, finite value: at +inf the held pair would read as
        # holding none of its share, and the add-on would take all of it
        assert held_pair_fraction(0.10, 0.0, portfolio_value) == 0.0
        assert held_pair_fraction(0.10, 8.0, portfolio_value) == 0.0

    @pytest.mark.parametrize("fraction, held_stake, portfolio_value", [
        (math.nan, 8.0, 211.0),
        (0.10, math.nan, 211.0),
        (0.10, 8.0, math.nan),
        (0.10, math.inf, 211.0),
        (0.10, 8.0, math.inf),
        (0.10, math.inf, math.inf),
    ])
    def test_a_result_that_is_not_a_number_adds_nothing(self, fraction, held_stake,
                                                         portfolio_value):
        # min(fraction, nan) is fraction, which would read garbage as a full
        # share, so the difference is tested before min
        assert held_pair_fraction(fraction, held_stake, portfolio_value) == 0.0

    @pytest.mark.parametrize("fraction, held_stake", [
        (math.inf, 8.0),
        (0.10, -math.inf),
    ])
    def test_an_infinite_share_adds_nothing(self, fraction, held_stake):
        # Each makes the difference +inf, and min would then return the whole
        # fraction (0.10 for the stake, inf for the share): refused before min
        assert held_pair_fraction(fraction, held_stake, 211.0) == 0.0

    @pytest.mark.parametrize("fraction", [0.05, 0.10, 0.25, 1.0])
    def test_nothing_held_and_no_other_position_is_exactly_a_new_pair(self, fraction):
        assert held_pair_fraction(fraction, 0.0, 200.0) == fraction


class TestContractPayout:
    """CONTRACT_PAYOUT_DOLLARS: what one winning contract pays.

    A contract pays $1 if it wins and nothing if it loses, so an open
    position is worth at most $1 per contract held. A production run refuses
    Kalshi's value of the open positions above that bound; a wrong constant
    would let a value in the wrong units size every trade.
    """

    def test_a_contract_pays_one_dollar(self):
        assert config.CONTRACT_PAYOUT_DOLLARS == 1.0
        assert type(config.CONTRACT_PAYOUT_DOLLARS) is float


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
            "same-title cap 100% (no extra cap) | categories any | tags any | "
            "add to held pairs off")
        assert config.describe_live_settings(s) == config.describe_live_settings(s, s)

    def test_the_shipped_values_render_with_no_mark(self):
        s = live_settings()
        assert "(config:" not in config.describe_live_settings(s, s)

    # One departing value per live toggle (config.LIVE_TOGGLE_FIELDS) and the
    # mark it must carry; a new toggle fails test_every_field_has_a_departure_row
    # until it has a row
    _DEPARTURES = {
        "tier_floors": (False, "tier floors off (config: on)"),
        "spread_band": ((0.1, 1.0), "spread band 0.1-1 (config: none)"),
        "interval_discount": (0.6, "k 0.6 (config: 0.75)"),
        "size_cap": (0.35, "per-trade cap 35% (config: 20%)"),
        "same_title_size_cap": (0.25, "same-title cap 25% (config: 100% (no extra cap))"),
        "categories": (("Economics",), "categories Economics (config: any)"),
        "tags": (("Oil & Gas", "Energy"), "tags Oil & Gas, Energy (config: any)"),
        "add_to_held_pairs": (True, "add to held pairs on (config: off)"),
    }

    def test_every_field_has_a_departure_row(self):
        assert set(self._DEPARTURES) == set(config.LIVE_TOGGLE_FIELDS)

    def test_every_departing_field_is_marked_and_no_other(self):
        ref = _settings()
        s = _settings(False, (0.0, 0.5), 0.8, 1.0, 0.2)
        assert config.describe_live_settings(s, ref) == (
            "tier floors off (config: on) | spread band 0-0.5 (config: none) | "
            "k 0.8 (config: 0.75) | per-trade cap 100% (no cap) (config: 20%) | "
            "same-title cap 20% (config: 100% (no extra cap)) | categories any | tags any | "
            "add to held pairs off")
        for name in config.LIVE_TOGGLE_FIELDS:
            value, mark = self._DEPARTURES[name]
            line = config.describe_live_settings(
                dataclasses.replace(ref, **{name: value}), ref)
            assert line.count("(config:") == 1, line
            assert mark in line, (name, line)

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
                        "categories any | tags any | add to held pairs off")

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

    def test_a_saved_defaults_reference_marks_default(self):
        # A reference read from the saved file (any origin but config.py's)
        ref = dataclasses.replace(_settings(), origin=_SAVED_ORIGIN)
        s = _settings(False, (0.0, 0.5), 0.8, 1.0, 0.2)
        line = config.describe_live_settings(s, ref)
        assert line == (
            "tier floors off (default: on) | spread band 0-0.5 (default: none) | "
            "k 0.8 (default: 0.75) | per-trade cap 100% (no cap) (default: 20%) | "
            "same-title cap 20% (default: 100% (no extra cap)) | categories any | tags any | "
            "add to held pairs off")
        assert "(config:" not in line
        # Every field's mark follows the reference's origin, one mark per departure
        for name in config.LIVE_TOGGLE_FIELDS:
            value, mark = self._DEPARTURES[name]
            line = config.describe_live_settings(
                dataclasses.replace(ref, **{name: value}), ref)
            assert line.count("(default:") == 1 and "(config:" not in line, line
            assert mark.replace("(config:", "(default:") in line, (name, line)
        # Equal toggles mark nothing, whatever either origin says
        assert "(default:" not in config.describe_live_settings(_settings(), ref)

    def test_a_config_reference_still_marks_config(self):
        # The mark reads the REFERENCE's origin, never the settings'
        ref = _settings()
        assert ref.origin == config.LIVE_DEFAULTS_FROM_CONFIG
        s = dataclasses.replace(_settings(interval_discount=0.6), origin=_SAVED_ORIGIN)
        line = config.describe_live_settings(s, ref)
        assert "k 0.6 (config: 0.75)" in line and "(default:" not in line

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
            "one time-series pair may stake up to 60% of the portfolio value, above the "
            "20% this check accepts",
            "one same-title pair may stake up to 60% of the portfolio value, above the "
            "20% this check accepts",
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
        assert "up to 20.04% of the portfolio value" in text

    def test_k_of_one_names_the_dead_time_series_leg(self):
        (text,) = config.live_rule_warnings(_settings(interval_discount=1.0))
        assert text.startswith("k = 1.0: ") and "no time-series trade can size" in text

    @pytest.mark.parametrize("cap", [0.05, 0.10, 0.20, 0.35, 0.60, 1.0])
    @pytest.mark.parametrize("k", [0.4, 0.75, 0.8, 1.0])
    def test_add_on_adds_no_sentence(self, cap, k):
        # Kelly on the whole position keeps a held pair within max_kelly_fraction
        # of the portfolio value, the bound the EXPOSURE sentence already names,
        # so the toggle changes no sentence at any cap or k
        off = _settings(interval_discount=k, size_cap=cap)
        on = dataclasses.replace(off, add_to_held_pairs=True)
        assert config.live_rule_warnings(on) == config.live_rule_warnings(off)

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
                            size_cap=1.0, same_title_size_cap=0.2, categories=None, tags=None,
                            add_to_held_pairs=True)
    # The values conftest's apply_pre_toggle_defaults pins, for the controls
    _BEFORE = LiveSettings(tier_floors=True, spread_band=(0.0, 1.0), interval_discount=0.75,
                           size_cap=0.2, same_title_size_cap=1.0)

    def test_the_shipped_settings(self):
        assert live_settings() == self._SHIPPED
        assert (config.TIME_SERIES_TIER_FLOORS, config.TIME_SERIES_SPREAD_BAND,
                config.TIME_SERIES_INTERVAL_PROB_DISCOUNT, config.BUDGET_FRACTION,
                config.SAME_TITLE_SIZE_CAP, config.TRADE_CATEGORIES, config.TRADE_TAGS,
                config.ADD_TO_HELD_PAIRS) == (
            False, (0.0, 0.5), 0.80, 1.0, 0.20, None, None, True)
        # ... and the seed proposes adding to held pairs too
        assert config.LIVE_DEFAULTS_SEED.add_to_held_pairs is True

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
        # Adding to held pairs on (as shipped) included
        assert live_settings().add_to_held_pairs is True
        assert config.live_rule_warnings(live_settings()) == []

    def test_a_default_backtest_departs_from_the_live_rule(self, monkeypatch, caplog,
                                                            saved_live_defaults):
        # The shipped toggles saved as the live defaults: the primary follows k
        # and both caps but keeps band (0, 1) with the tiers on, so the live
        # rule is a grid cell, and the last line says so (no sizing note: the
        # saved k and caps are the run's own). The shipped defaults add to held
        # pairs, which no point of a backtest run does, so the line ends with
        # that note.
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
        assert res.live_add_to_held_pairs is True
        rule = config.describe_time_series_rule(False, (0.0, 0.5))
        lines = [r.getMessage() for r in caplog.records
                 if r.getMessage().startswith("Live time-series rule")]
        assert lines == [
            f"Live time-series rule (saved live defaults): {rule} — this run's primary "
            "scenario does not (tier floors on, band 0-1); its grid simulated the live rule "
            "as band 0-0.5 with the tier floors off, which the dashboard's filter bar shows; "
            "the live defaults add to held pairs, which this run's primary does not"]

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
        "TRADE_CATEGORIES", "TRADE_TAGS", "ADD_TO_HELD_PAIRS"})

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
            same_title_size_cap=1.0, categories=None, tags=None, add_to_held_pairs=False)
        for module, bound, toggle in binders:
            assert getattr(importlib.import_module(f"kalshi_betting.{module}"), bound) == (
                getattr(config, toggle)), (module, toggle)

    def test_no_test_module_binds_a_toggle_by_value(self):
        # A by-value import would freeze the shipped value past every patch
        assert self._by_value_binders(pathlib.Path(__file__).parent) == []


class TestScheduledRun:
    """ScheduledRun, the weekly live run's schedule. The scheduler fires the
    live run from it and the backtest enters every trade at instant(d), so
    instant() must follow daylight-saving rules and date_problems() must flag
    every run date that is not one UTC moment on that same date."""

    def test_the_shipped_schedule_is_monday_0900_los_angeles(self):
        # A change to the live run's time must fail a test, never pass silently
        assert SCHEDULED_RUN == ScheduledRun(0, 9, 0, "America/Los_Angeles")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"weekday": 7},
            {"weekday": -1},
            {"hour": 24},
            {"minute": 60},
            {"weekday": True},
            {"hour": "9"},
            {"minute": 0.0},
            {"timezone": ""},
            {"timezone": None},
        ],
        ids=lambda kw: "-".join(f"{k}={v!r}" for k, v in kw.items()),
    )
    def test_a_bad_field_is_refused(self, kwargs):
        fields = {"weekday": 0, "hour": 9, "minute": 0, "timezone": "America/Los_Angeles"}
        fields.update(kwargs)
        with pytest.raises(ValueError, match=next(iter(kwargs))):
            ScheduledRun(**fields)

    def test_the_zone_is_resolved_only_on_use(self):
        run = ScheduledRun(0, 9, 0, "No/Such_Zone")  # constructing never looks it up
        with pytest.raises(ZoneInfoNotFoundError):
            run.zone()

    def test_frozen_and_compared_by_value(self):
        run = ScheduledRun(0, 9, 0, "America/Los_Angeles")
        with pytest.raises(dataclasses.FrozenInstanceError):
            run.hour = 10
        assert run == SCHEDULED_RUN
        assert hash(run) == hash(SCHEDULED_RUN)
        assert run != ScheduledRun(0, 9, 0, "UTC")

    def test_accessors(self):
        assert SCHEDULED_RUN.zone() == ZoneInfo("America/Los_Angeles")
        assert SCHEDULED_RUN.weekday_name() == "monday"
        assert SCHEDULED_RUN.at_time() == "09:00"
        assert SCHEDULED_RUN.label() == "Monday 09:00 America/Los_Angeles"
        assert SCHEDULED_RUN.cache_slug() == "mon0900-America-Los_Angeles"
        other = ScheduledRun(6, 7, 5, "Etc/GMT+8")
        assert other.weekday_name() == "sunday"
        assert other.at_time() == "07:05"
        assert other.label() == "Sunday 07:05 Etc/GMT+8"
        assert other.cache_slug() == "sun0705-Etc-GMT+8"

    def test_wall_time_is_the_zones_wall_clock(self):
        wall = SCHEDULED_RUN.wall_time(date(2026, 9, 21))
        assert wall.tzinfo == ZoneInfo("America/Los_Angeles")
        assert (wall.year, wall.month, wall.day, wall.hour, wall.minute) == (2026, 9, 21, 9, 0)

    @pytest.mark.parametrize(
        ("d", "utc_hour"),
        [
            (date(2026, 9, 21), 16),   # daylight time
            (date(2026, 10, 26), 16),  # the last Monday of daylight time
            (date(2026, 11, 2), 17),   # the first Monday of standard time
            (date(2027, 3, 8), 17),    # the last Monday of standard time
            (date(2027, 3, 15), 16),   # the first Monday of daylight time
            (date(2040, 7, 2), 16),    # zoneinfo keeps US daylight time past 2037
        ],
    )
    def test_instant_follows_daylight_saving(self, d, utc_hour):
        assert SCHEDULED_RUN.instant(d) == datetime(d.year, d.month, d.day, utc_hour, 0, tzinfo=UTC)

    def test_the_shipped_schedule_has_no_date_problems(self):
        assert SCHEDULED_RUN.date_problems(date(1990, 1, 1), date(2100, 12, 31)) == []

    def test_a_skipped_wall_time_is_flagged(self):
        # Sunday 02:30 does not exist in Los Angeles on the spring-forward day.
        problems = ScheduledRun(6, 2, 30, "America/Los_Angeles").date_problems(
            date(2026, 1, 1), date(2026, 12, 31))
        assert len(problems) == 1
        assert problems[0].startswith("2026-03-08: ") and "skipped" in problems[0]

    def test_a_repeated_wall_time_is_flagged(self):
        # Sunday 01:30 occurs twice in Los Angeles on the fall-back day.
        problems = ScheduledRun(6, 1, 30, "America/Los_Angeles").date_problems(
            date(2026, 1, 1), date(2026, 12, 31))
        assert len(problems) == 1
        assert problems[0].startswith("2026-11-01: ") and "repeated" in problems[0]

    @pytest.mark.parametrize(
        ("run", "d", "expected"),
        [
            # Spring forward: the clock jumps from 02:00 to 03:00.
            (ScheduledRun(6, 2, 30, "America/Los_Angeles"), date(2026, 3, 8), "skipped"),
            # Fall back: the clock repeats 01:00-02:00.
            (ScheduledRun(6, 1, 30, "America/Los_Angeles"), date(2026, 11, 1), "repeated"),
            # Either side of each change, and a wall time the change never reaches.
            (ScheduledRun(6, 2, 30, "America/Los_Angeles"), date(2026, 3, 1), None),
            (ScheduledRun(6, 1, 30, "America/Los_Angeles"), date(2026, 11, 8), None),
            (ScheduledRun(6, 9, 0, "America/Los_Angeles"), date(2026, 3, 8), None),
            # Southern hemisphere: Sydney springs forward in October, falls back in April.
            (ScheduledRun(6, 2, 30, "Australia/Sydney"), date(2026, 10, 4), "skipped"),
            (ScheduledRun(6, 2, 30, "Australia/Sydney"), date(2026, 4, 5), "repeated"),
        ],
        ids=["la-gap", "la-fold", "la-before-gap", "la-after-fold", "la-0900",
             "sydney-gap", "sydney-fold"],
    )
    def test_clock_change_names_a_skipped_or_repeated_wall_time(self, run, d, expected):
        assert run.clock_change(d) == expected

    @pytest.mark.parametrize(
        "run",
        [
            ScheduledRun(0, 20, 0, "America/Los_Angeles"),  # 03:00Z/04:00Z Tuesday
            ScheduledRun(0, 8, 0, "Asia/Tokyo"),            # 23:00Z Sunday
        ],
        ids=["los-angeles-20h", "tokyo-08h"],
    )
    def test_a_date_shift_is_flagged_on_every_run_date(self, run):
        problems = run.date_problems(date(2026, 9, 1), date(2026, 9, 30))
        assert [p[:10] for p in problems] == ["2026-09-07", "2026-09-14", "2026-09-21", "2026-09-28"]
        assert all("another date in UTC" in p for p in problems)

    def test_midnight_utc_on_the_same_date_passes(self):
        # Tokyo 09:00 is 00:00Z on the same calendar date.
        assert ScheduledRun(0, 9, 0, "Asia/Tokyo").date_problems(
            date(2026, 1, 1), date(2026, 12, 31)) == []

    def test_out_of_range_dates_are_listed_not_raised(self):
        # Tokyo 08:00 on year 1's first Monday is before datetime's range in UTC
        run = ScheduledRun(0, 8, 0, "Asia/Tokyo")
        first = run.date_problems(date(1, 1, 1), date(1, 1, 7))
        assert len(first) == 1 and "outside datetime's range" in first[0]
        last = run.date_problems(date(9999, 12, 20), date.max)
        assert [p[:10] for p in last] == ["9999-12-20", "9999-12-27"]
        assert SCHEDULED_RUN.date_problems(date(9999, 12, 20), date.max) == []
        # No Monday is left before date.max: an empty list, never an overflow
        assert SCHEDULED_RUN.date_problems(date(9999, 12, 28), date.max) == []
        assert SCHEDULED_RUN.date_problems(date.max, date.max) == []


class TestLiveSettingsOrigin:
    """LiveSettings.origin says where the toggles' defaults came from; it is
    one printable line and is never compared."""

    def test_it_defaults_to_config_py(self):
        assert config.LIVE_DEFAULTS_FROM_CONFIG == "config.py"
        assert _settings().origin == config.LIVE_DEFAULTS_FROM_CONFIG
        assert live_settings().origin == config.LIVE_DEFAULTS_FROM_CONFIG

    @pytest.mark.parametrize("bad", [
        None, 7, b"config.py", "", "   ",
        "two\nlines", "a\ttab", "zero\u200bwidth", "\u202edirection",
    ])
    def test_a_blank_multi_line_or_non_printable_origin_is_refused(self, bad):
        with pytest.raises(ValueError, match="origin"):
            LiveSettings(True, (0.0, 1.0), 0.75, 0.2, origin=bad)
        with pytest.raises(ValueError, match="origin"):
            dataclasses.replace(_settings(), origin=bad)

    def test_equal_toggles_are_equal_and_hash_alike_across_origins(self):
        a = _settings()
        b = dataclasses.replace(a, origin=_SAVED_ORIGIN)
        assert a.origin != b.origin
        assert a == b and hash(a) == hash(b) and len({a, b}) == 1
        # ... while any toggle still tells two apart
        assert a != dataclasses.replace(b, size_cap=0.35)

    def test_replace_keeps_it(self):
        s = dataclasses.replace(_settings(), origin=_SAVED_ORIGIN)
        assert dataclasses.replace(s, size_cap=0.35).origin == _SAVED_ORIGIN
        assert dataclasses.replace(s, categories=("Sports",)).origin == _SAVED_ORIGIN

    def test_the_toggle_fields_are_the_eight(self):
        assert config.LIVE_TOGGLE_FIELDS == (
            "tier_floors", "spread_band", "interval_discount", "size_cap",
            "same_title_size_cap", "categories", "tags", "add_to_held_pairs")
        assert [f.name for f in dataclasses.fields(LiveSettings)] == [
            *config.LIVE_TOGGLE_FIELDS, "origin"]


class _FrozenDatetime(datetime):
    """datetime whose now() is fixed at 2026-09-27 21:05:13 UTC; tests put it
    in place of config.datetime, so a save stamps a known time."""

    @classmethod
    def now(cls, tz=None):
        """
        Return the fixed instant, in UTC.

        Args:
            tz: Ignored; the instant is always UTC.

        Returns:
            datetime: 2026-09-27 21:05:13 UTC.
        """
        return datetime(2026, 9, 27, 21, 5, 13, tzinfo=UTC)


def _valid_record(**changes) -> dict:
    """
    Build a saved-defaults record the reader accepts, then apply changes.

    Its toggles are LIVE_DEFAULTS_SEED's, saved at 2026-09-27T21:05:13Z with
    no source note, with add_to_held_pairs left out (the seven-toggle shape),
    which reads as _SEED_ADD_ON_OFF.

    Args:
        **changes: Top-level keys to replace; a value of _DROP removes the key.

    Returns:
        dict: The record, ready for json.dumps.
    """
    record = {
        "format": config.LIVE_DEFAULTS_FORMAT,
        "saved_at": "2026-09-27T21:05:13Z",
        "source": "",
        "settings": {
            "tier_floors": False, "spread_band": [0.0, 0.5], "interval_discount": 0.8,
            "size_cap": 0.1, "same_title_size_cap": 0.2, "categories": None, "tags": None,
        },
    }
    for key, value in changes.items():
        if value is _DROP:
            del record[key]
        else:
            record[key] = value
    return record


def _with_toggles(**changes) -> dict:
    """
    Build _valid_record() with toggles in its "settings" block changed.

    Args:
        **changes: Toggles to replace; a value of _DROP removes the toggle.

    Returns:
        dict: The record, ready for json.dumps.
    """
    record = _valid_record()
    for key, value in changes.items():
        if value is _DROP:
            del record["settings"][key]
        else:
            record["settings"][key] = value
    return record


# Marks a key _valid_record / _with_toggles should remove
_DROP = object()

# LIVE_DEFAULTS_SEED with adding to held pairs off: what _valid_record's
# seven-toggle file reads as, and what a save of it writes as seven toggles
_SEED_ADD_ON_OFF = dataclasses.replace(config.LIVE_DEFAULTS_SEED, add_to_held_pairs=False)

# The exact bytes a save of _SEED_ADD_ON_OFF writes, with the seed's note, at
# _FrozenDatetime's instant: add_to_held_pairs is left out, so code that does
# not know the toggle can read a file saved with it off
_SEVEN_TOGGLE_BYTES = (
    b'{\n'
    b'  "format": "live-defaults-v1",\n'
    b'  "saved_at": "2026-09-27T21:05:13Z",\n'
    b'  "source": "seed values (config.LIVE_DEFAULTS_SEED)",\n'
    b'  "settings": {\n'
    b'    "tier_floors": false,\n'
    b'    "spread_band": [0.0, 0.5],\n'
    b'    "interval_discount": 0.8,\n'
    b'    "size_cap": 0.1,\n'
    b'    "same_title_size_cap": 0.2,\n'
    b'    "categories": null,\n'
    b'    "tags": null\n'
    b'  }\n'
    b'}\n')


def _text(record) -> bytes:
    """
    Render a record as the bytes of a saved-defaults file.

    Args:
        record: Any JSON value; json.dumps writes NaN and Infinity as the bare
            words the reader must refuse.

    Returns:
        bytes: The UTF-8 text.
    """
    return json.dumps(record).encode("utf-8")


def _only(directory: pathlib.Path) -> list[str]:
    """
    List what a save left behind in a directory.

    Args:
        directory (pathlib.Path): The directory to list.

    Returns:
        list[str]: The names of its entries, sorted.
    """
    return sorted(p.name for p in directory.iterdir())


class TestSavedLiveDefaults:
    """read_saved_live_defaults and save_live_defaults: the saved live defaults
    file, written atomically, read strictly, refused whole on any flaw. Each
    test writes to its own path (conftest's _isolate_live_defaults)."""

    _SETTINGS = LiveSettings(
        tier_floors=True, spread_band=(0.1, 0.6), interval_discount=0.6, size_cap=0.35,
        same_title_size_cap=0.5, categories=("Sports",), tags=("Basketball",))

    def test_the_path_is_this_test_s_own(self, tmp_path):
        # conftest's per-test redirect, never the repo's own file
        assert config.LIVE_DEFAULTS_FILE == tmp_path / "live_defaults.json"

    def test_conftest_saves_config_py_s_toggles(self, saved_live_defaults, tmp_path):
        # tests/conftest.py's saved_live_defaults fixture: config.py's toggles,
        # saved into this test's own path, with no source note
        assert config.LIVE_DEFAULTS_FILE == tmp_path / "live_defaults.json"
        saved = config.read_saved_live_defaults()
        assert saved == config.live_settings()
        assert saved.origin.startswith("live_defaults.json, saved ")
        assert " from " not in saved.origin

    def test_a_missing_file_reads_none(self):
        assert not config.LIVE_DEFAULTS_FILE.exists()
        assert config.read_saved_live_defaults() is None

    def test_a_save_round_trips_every_toggle(self):
        # A Category · Tag filter (the dashboard's "Sports · Basketball") and a
        # same-title cap below the per-trade cap
        saved = config.save_live_defaults(self._SETTINGS, source="a note")
        assert saved == self._SETTINGS
        for name in config.LIVE_TOGGLE_FIELDS:
            assert getattr(saved, name) == getattr(self._SETTINGS, name), name
        again = config.read_saved_live_defaults()
        assert again == self._SETTINGS and again.origin == saved.origin
        text = config.LIVE_DEFAULTS_FILE.read_text(encoding="utf-8")
        assert '"categories": ["Sports"],' in text and '"tags": ["Basketball"]\n' in text

    def test_the_written_text_is_pinned(self, monkeypatch):
        # With adding to held pairs off, the key is left out, byte for byte,
        # so code that does not know the toggle can read a file saved with it off
        monkeypatch.setattr(config, "datetime", _FrozenDatetime)
        config.save_live_defaults(_SEED_ADD_ON_OFF, source=config.LIVE_DEFAULTS_SEED_SOURCE)
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == _SEVEN_TOGGLE_BYTES

    def test_an_on_save_writes_the_key_last(self, monkeypatch):
        # The seed (adding to held pairs on) writes the toggle, after the
        # other seven
        monkeypatch.setattr(config, "datetime", _FrozenDatetime)
        config.save_live_defaults(config.LIVE_DEFAULTS_SEED,
                                  source=config.LIVE_DEFAULTS_SEED_SOURCE)
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == _SEVEN_TOGGLE_BYTES.replace(
            b'    "tags": null\n',
            b'    "tags": null,\n    "add_to_held_pairs": true\n')
        assert config.read_saved_live_defaults() == config.LIVE_DEFAULTS_SEED

    def test_a_seven_toggle_file_reads_with_add_on_off(self):
        # A file that leaves add_to_held_pairs out reads as the same
        # settings, adding to held pairs off (nobody confirmed a value for
        # it), whatever config.py or the seed ship
        config.LIVE_DEFAULTS_FILE.write_bytes(_SEVEN_TOGGLE_BYTES)
        saved = config.read_saved_live_defaults()
        assert saved == _SEED_ADD_ON_OFF and saved.add_to_held_pairs is False
        assert config.ADD_TO_HELD_PAIRS is True
        assert config.LIVE_DEFAULTS_SEED.add_to_held_pairs is True
        assert saved.origin == ("live_defaults.json, saved 2026-09-27T21:05:13Z from "
                                "seed values (config.LIVE_DEFAULTS_SEED)")

    @pytest.mark.parametrize("value", [False, True])
    def test_a_file_naming_the_toggle_reads_it(self, value):
        # Written either way by hand, the toggle reads as written
        config.LIVE_DEFAULTS_FILE.write_bytes(_text(_with_toggles(add_to_held_pairs=value)))
        assert config.read_saved_live_defaults() == dataclasses.replace(
            _SEED_ADD_ON_OFF, add_to_held_pairs=value)

    def test_the_origin_names_the_file_the_time_and_the_source(self, monkeypatch):
        monkeypatch.setattr(config, "datetime", _FrozenDatetime)
        saved = config.save_live_defaults(config.LIVE_DEFAULTS_SEED,
                                          source=config.LIVE_DEFAULTS_SEED_SOURCE)
        assert saved.origin == ("live_defaults.json, saved 2026-09-27T21:05:13Z from "
                                "seed values (config.LIVE_DEFAULTS_SEED)")
        # No note, no "from"; surrounding spaces are dropped from a note
        assert config.save_live_defaults(self._SETTINGS, source="").origin == (
            "live_defaults.json, saved 2026-09-27T21:05:13Z")
        assert config.save_live_defaults(self._SETTINGS, source="  a note  ").origin == (
            "live_defaults.json, saved 2026-09-27T21:05:13Z from a note")

    def test_no_staging_file_is_left_after_a_save(self):
        config.save_live_defaults(self._SETTINGS, source="")
        config.save_live_defaults(config.LIVE_DEFAULTS_SEED, source="")
        assert _only(config.LIVE_DEFAULTS_FILE.parent) == ["live_defaults.json"]
        assert config.read_saved_live_defaults() == config.LIVE_DEFAULTS_SEED

    def test_the_staging_name_holds_the_process_id_and_is_gitignored(self, monkeypatch):
        staged = []
        real_replace = os.replace

        def replace_spy(src, dst):
            staged.append(pathlib.Path(src).name)
            real_replace(src, dst)

        monkeypatch.setattr(os, "replace", replace_spy)
        config.save_live_defaults(self._SETTINGS, source="")
        assert staged == [f".live_defaults.json.{os.getpid()}.tmp"]
        ignored = (PROJECT_ROOT / ".gitignore").read_text().splitlines()
        assert "live_defaults.json" in ignored and ".live_defaults.json.*.tmp" in ignored

    def test_a_failed_rename_keeps_the_old_file(self, monkeypatch):
        config.save_live_defaults(config.LIVE_DEFAULTS_SEED, source="")
        before = config.LIVE_DEFAULTS_FILE.read_bytes()

        def refuse(src, dst):
            raise OSError("no rename today")

        monkeypatch.setattr(os, "replace", refuse)
        with pytest.raises(config.LiveDefaultsError, match="could not be written") as err:
            config.save_live_defaults(self._SETTINGS, source="")
        assert str(config.LIVE_DEFAULTS_FILE) in str(err.value)
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == before
        assert _only(config.LIVE_DEFAULTS_FILE.parent) == ["live_defaults.json"]

    def test_a_missing_directory_is_refused(self, tmp_path, monkeypatch):
        path = tmp_path / "missing" / "live_defaults.json"
        monkeypatch.setattr(config, "LIVE_DEFAULTS_FILE", path)
        with pytest.raises(config.LiveDefaultsError, match="could not be written") as err:
            config.save_live_defaults(self._SETTINGS, source="")
        assert str(path) in str(err.value)
        assert not path.parent.exists()

    @pytest.mark.skipif(os.geteuid() == 0, reason="root writes into a read-only directory")
    def test_a_read_only_directory_is_refused(self, tmp_path, monkeypatch):
        directory = tmp_path / "read-only"
        directory.mkdir()
        path = directory / "live_defaults.json"
        monkeypatch.setattr(config, "LIVE_DEFAULTS_FILE", path)
        directory.chmod(0o555)
        try:
            with pytest.raises(config.LiveDefaultsError, match="could not be written"):
                config.save_live_defaults(self._SETTINGS, source="")
            assert _only(directory) == []
        finally:
            directory.chmod(0o755)

    def test_a_failed_directory_flush_says_the_file_was_written(self, monkeypatch):
        def refuse(directory):
            raise OSError("no flush today")

        monkeypatch.setattr(config, "_sync_directory", refuse)
        with pytest.raises(config.LiveDefaultsError, match="written, but its directory"):
            config.save_live_defaults(self._SETTINGS, source="")
        # The rename happened: the new file is in place, and nothing is staged
        assert config.read_saved_live_defaults() == self._SETTINGS
        assert _only(config.LIVE_DEFAULTS_FILE.parent) == ["live_defaults.json"]

    @pytest.mark.parametrize("read_back", [None, _settings()])
    def test_a_read_back_mismatch_raises(self, monkeypatch, read_back):
        monkeypatch.setattr(config, "read_saved_live_defaults", lambda: read_back)
        with pytest.raises(config.LiveDefaultsError, match="read back as") as err:
            config.save_live_defaults(self._SETTINGS, source="")
        assert str(config.LIVE_DEFAULTS_FILE) in str(err.value)

    @pytest.mark.parametrize("source", [7, None, "two\nlines", "zero\u200bwidth",
                                        "x" * (config.LIVE_DEFAULTS_SOURCE_MAX_CHARS + 1)])
    def test_a_bad_source_is_refused_before_anything_is_written(self, monkeypatch, source):
        opened = []
        monkeypatch.setattr(config, "_sync_directory", lambda d: opened.append(d))
        with pytest.raises(config.LiveDefaultsError, match="source") as err:
            config.save_live_defaults(self._SETTINGS, source=source)
        assert str(config.LIVE_DEFAULTS_FILE) in str(err.value)
        assert _only(config.LIVE_DEFAULTS_FILE.parent) == [] and opened == []

    @pytest.mark.parametrize("changes, words", [
        ({"categories": ("Sports\u200b",)}, "printable"),
        ({"tags": ("Basket\u202eball",)}, "printable"),
        ({"categories": tuple(f"Category {i:05d}" for i in range(5000))},
         f"over {config.LIVE_DEFAULTS_MAX_BYTES} bytes"),
    ], ids=["zero-width category", "text-direction tag", "a filter over the size limit"])
    def test_settings_the_reader_would_refuse_are_never_written(self, monkeypatch, changes,
                                                                words):
        # LiveSettings takes these values and the file's rules do not: the good
        # file already in place must survive the attempt untouched
        config.save_live_defaults(config.LIVE_DEFAULTS_SEED, source="")
        before = config.LIVE_DEFAULTS_FILE.read_bytes()
        bad = dataclasses.replace(self._SETTINGS, **changes)
        renamed = []
        monkeypatch.setattr(os, "replace", lambda src, dst: renamed.append(src))
        with pytest.raises(config.LiveDefaultsError, match=re.escape(words)) as err:
            config.save_live_defaults(bad, source="")
        assert str(err.value).startswith(f"{config.LIVE_DEFAULTS_FILE}: not saved: ")
        assert renamed == []
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == before
        assert _only(config.LIVE_DEFAULTS_FILE.parent) == ["live_defaults.json"]

    def test_settings_that_would_read_back_differently_are_never_written(self, monkeypatch):
        config.save_live_defaults(config.LIVE_DEFAULTS_SEED, source="")
        before = config.LIVE_DEFAULTS_FILE.read_bytes()
        real_text = config._saved_text
        # A renderer that writes another valid per-trade cap than the one given
        monkeypatch.setattr(config, "_saved_text", lambda record: real_text(
            {**record, "settings": {**record["settings"], "size_cap": 0.1}}))
        with pytest.raises(config.LiveDefaultsError, match="not saved: it would read back as"):
            config.save_live_defaults(self._SETTINGS, source="")
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == before
        assert _only(config.LIVE_DEFAULTS_FILE.parent) == ["live_defaults.json"]

    def test_the_data_is_flushed_before_the_rename_and_the_directory_after(self, monkeypatch):
        # In order: the staging file's data is fsynced, it is renamed into
        # place, and the directory is flushed with F_FULLFSYNC (a stand-in value
        # here, so the check runs on any platform) and not with plain fsync
        events = []
        real_fsync, real_replace = os.fsync, os.replace

        def fsync_spy(fd):
            """
            Record which file an fsync flushed (by inode), then flush it.

            Args:
                fd (int): The descriptor being flushed.
            """
            events.append(("fsync", os.fstat(fd).st_ino))
            real_fsync(fd)

        def replace_spy(src, dst):
            """
            Record a rename, then make it.

            Args:
                src: The file being renamed.
                dst: The name it takes.
            """
            events.append(("replace",))
            real_replace(src, dst)

        def fcntl_spy(fd, cmd, *args):
            """
            Record which file an fcntl call named (by inode) and its command.

            Args:
                fd (int): The descriptor.
                cmd (int): The fcntl command.
                *args: Ignored.

            Returns:
                int: 0, as a successful fcntl returns.
            """
            events.append(("fcntl", os.fstat(fd).st_ino, cmd))
            return 0

        monkeypatch.setattr(os, "fsync", fsync_spy)
        monkeypatch.setattr(os, "replace", replace_spy)
        monkeypatch.setattr(config.fcntl, "F_FULLFSYNC", 51, raising=False)
        monkeypatch.setattr(config.fcntl, "fcntl", fcntl_spy)
        config.save_live_defaults(self._SETTINGS, source="")
        path = config.LIVE_DEFAULTS_FILE
        # A rename keeps the inode: the file fsynced is the one now in place
        assert events == [("fsync", path.stat().st_ino), ("replace",),
                          ("fcntl", path.parent.stat().st_ino, 51)]

    def test_the_directory_is_flushed_after_the_rename(self, monkeypatch):
        flushed = []

        def spy(directory):
            # Called once the new file is in place under its own name
            flushed.append((directory, config.LIVE_DEFAULTS_FILE.exists()))

        monkeypatch.setattr(config, "_sync_directory", spy)
        config.save_live_defaults(self._SETTINGS, source="")
        assert flushed == [(config.LIVE_DEFAULTS_FILE.parent, True)]

    def test_sync_directory_flushes_a_real_directory(self, tmp_path):
        # F_FULLFSYNC on macOS, fsync elsewhere: either way it returns quietly
        config._sync_directory(tmp_path)

    def test_sync_directory_falls_back_to_fsync(self, tmp_path, monkeypatch):
        synced = []
        real_fsync = os.fsync

        def fsync_spy(fd):
            synced.append(fd)
            real_fsync(fd)

        def refuse(fd, cmd, *args):
            raise OSError("F_FULLFSYNC refused")

        monkeypatch.setattr(os, "fsync", fsync_spy)
        # A filesystem that refuses F_FULLFSYNC ...
        monkeypatch.setattr(config.fcntl, "F_FULLFSYNC", 51, raising=False)
        monkeypatch.setattr(config.fcntl, "fcntl", refuse)
        config._sync_directory(tmp_path)
        assert len(synced) == 1
        # ... and a platform without it
        monkeypatch.delattr(config.fcntl, "F_FULLFSYNC")
        config._sync_directory(tmp_path)
        assert len(synced) == 2

    def test_the_error_is_a_value_error(self):
        assert issubclass(config.LiveDefaultsError, ValueError)
        assert issubclass(config.LiveDefaultsMissing, config.LiveDefaultsError)

    _REFUSED = [
        # The file as a whole
        ("not JSON", b"not json"),
        ("not UTF-8", b'\xff\xfe{"format": 1}'),
        ("over the size limit", _text(_valid_record()) + b" " * config.LIVE_DEFAULTS_MAX_BYTES),
        # Under the size limit, nested past the recursion limit
        ("too deeply nested", b"[" * 60_000),
        ("a JSON array", b"[]"),
        ("a JSON string", b'"live-defaults-v1"'),
        # The top-level keys
        ("a missing key", _text(_valid_record(source=_DROP))),
        ("an extra key", _text(_valid_record(note="hello"))),
        ("a wrong format", _text(_valid_record(format="live-defaults-v2"))),
        ("a saved_at with a space", _text(_valid_record(saved_at="2026-09-27 21:05:13Z"))),
        ("a saved_at without zero padding", _text(_valid_record(saved_at="2026-9-27T21:05:13Z"))),
        ("a saved_at that is a number", _text(_valid_record(saved_at=20260927))),
        ("a source that is a number", _text(_valid_record(source=7))),
        ("a two-line source", _text(_valid_record(source="two\nlines"))),
        ("settings that are a list", _text(_valid_record(settings=[]))),
        # The settings block
        # A toggle that was always there may not be left out; only one added
        # later (add_to_held_pairs) may
        ("a missing toggle", _text(_with_toggles(tags=_DROP))),
        ("a missing toggle beside the later one",
         _text(_with_toggles(tags=_DROP, add_to_held_pairs=False))),
        ("an extra toggle", _text(_with_toggles(origin="config.py"))),
        ("an extra toggle beside the later one",
         _text(_with_toggles(add_to_held_pairs=True, sell_held_pairs=True))),
        # Values
        # LiveSettings alone would read these two as the band (0.0, 1.0): only the
        # reader's own spread_band rule refuses them
        ("a band of booleans", _text(_with_toggles(spread_band=[False, True]))),
        ("a band with a boolean ceiling", _text(_with_toggles(spread_band=[0.0, True]))),
        ("a band with nothing between", _text(_with_toggles(spread_band=[0.5, 0.5]))),
        ("a band of three", _text(_with_toggles(spread_band=[0.0, 0.5, 1.0]))),
        ("a band as a string", _text(_with_toggles(spread_band="0-0.5"))),
        ("a NaN", _text(_with_toggles(interval_discount=float("nan")))),
        ("an Infinity", _text(_with_toggles(size_cap=float("inf")))),
        ("a duplicate key", _text(_valid_record()).replace(
            b'"size_cap": 0.1,', b'"size_cap": 0.1, "size_cap": 1.0,')),
        ("k of 0", _text(_with_toggles(interval_discount=0))),
        ("a cap off the grid", _text(_with_toggles(size_cap=0.33))),
        ("tier floors as 1", _text(_with_toggles(tier_floors=1))),
        ("add to held pairs as 1", _text(_with_toggles(add_to_held_pairs=1))),
        ("add to held pairs as a string", _text(_with_toggles(add_to_held_pairs="true"))),
        ("add to held pairs as null", _text(_with_toggles(add_to_held_pairs=None))),
        # Category and tag names
        ("the name any", _text(_with_toggles(categories=["any"]))),
        ("an empty name", _text(_with_toggles(tags=[""]))),
        ("a zero-width name", _text(_with_toggles(categories=["Sports\u200b"]))),
        ("a bare string filter", _text(_with_toggles(tags="Basketball"))),
    ]

    @pytest.mark.parametrize("label, data", _REFUSED, ids=[label for label, _ in _REFUSED])
    def test_every_refused_file_names_the_file(self, label, data):
        path = config.LIVE_DEFAULTS_FILE
        path.write_bytes(data)
        with pytest.raises(config.LiveDefaultsError) as err:
            config.read_saved_live_defaults()
        assert str(path) in str(err.value), label
        assert not isinstance(err.value, config.LiveDefaultsMissing)

    def test_the_refusals_name_their_cause(self):
        # A few of the rules above, by their words
        path = config.LIVE_DEFAULTS_FILE
        for data, words in [
            (_text(_valid_record()) + b" " * config.LIVE_DEFAULTS_MAX_BYTES, "over 65536 bytes"),
            (b"[" * 60_000, "recursion"),
            (_text(_with_toggles(interval_discount=float("nan"))), "NaN"),
            (_text(_valid_record()).replace(b'"size_cap": 0.1,',
                                            b'"size_cap": 0.1, "size_cap": 1.0,'), "twice"),
            (_text(_with_toggles(categories=["Sports\u200b"])), "printable"),
            (_text(_with_toggles(categories=["any"])), "null in live_defaults.json"),
            (_text(_with_toggles(size_cap=0.33)), "size_cap"),
            (_text(_with_toggles(spread_band=[False, True])),
             '"spread_band" must be [floor, ceiling]'),
            (_text(_with_toggles(spread_band=[0.0, True])),
             '"spread_band" must be [floor, ceiling]'),
            (_text(_with_toggles(add_to_held_pairs=1)), "add_to_held_pairs must be True or False"),
            (_text(_with_toggles(tags=_DROP)),
             "(add_to_held_pairs may be left out, and then reads as off)"),
        ]:
            path.write_bytes(data)
            with pytest.raises(config.LiveDefaultsError, match=re.escape(words)):
                config.read_saved_live_defaults()

    def test_a_directory_at_the_path_is_refused(self):
        path = config.LIVE_DEFAULTS_FILE
        path.mkdir()
        with pytest.raises(config.LiveDefaultsError,
                           match=re.escape("cannot be read (not a regular file)")) as err:
            config.read_saved_live_defaults()
        assert str(path) in str(err.value)

    @pytest.mark.skipif(not hasattr(os, "symlink"), reason="no symbolic links on this platform")
    def test_a_link_to_nothing_is_refused_not_read_as_missing(self):
        # A link whose target is gone is something at the path: it must be
        # refused, never read as "no defaults saved yet"
        path = config.LIVE_DEFAULTS_FILE
        target = path.with_name("moved-away.json")
        os.symlink(target, path)
        words = "cannot be read (a link to a file that does not exist)"
        with pytest.raises(config.LiveDefaultsError, match=re.escape(words)) as err:
            config.read_saved_live_defaults()
        assert str(path) in str(err.value)
        assert not isinstance(err.value, config.LiveDefaultsMissing)
        with pytest.raises(config.LiveDefaultsError, match=re.escape(words)) as err:
            config.live_defaults()
        assert not isinstance(err.value, config.LiveDefaultsMissing)
        # Control: the same link, once its target exists, reads as the saved defaults
        target.write_bytes(_text(_valid_record()))
        assert config.read_saved_live_defaults() == _SEED_ADD_ON_OFF

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no named pipes on this platform")
    def test_a_fifo_at_the_path_is_refused_without_waiting(self):
        # Opening a FIFO for reading normally waits for a writer; the reader
        # must refuse it at once instead of hanging a live run
        path = config.LIVE_DEFAULTS_FILE
        os.mkfifo(path)
        outcome = []

        def read():
            """Read the saved defaults, keeping the refusal's message."""
            try:
                outcome.append(config.read_saved_live_defaults())
            except config.LiveDefaultsError as exc:
                outcome.append(str(exc))

        worker = threading.Thread(target=read, daemon=True)
        worker.start()
        worker.join(10)
        if worker.is_alive():
            # Open the FIFO for writing so the blocked read returns, then fail
            os.close(os.open(path, os.O_WRONLY | os.O_NONBLOCK))
            worker.join(10)
            pytest.fail("reading a FIFO at the path waited for a writer")
        assert outcome == [f"{path}: cannot be read (not a regular file)"]

    @pytest.mark.skipif(os.geteuid() == 0, reason="root reads an unreadable file")
    def test_an_unreadable_file_is_refused(self):
        path = config.LIVE_DEFAULTS_FILE
        path.write_bytes(_text(_valid_record()))
        path.chmod(0o000)
        try:
            with pytest.raises(config.LiveDefaultsError, match="cannot be read"):
                config.read_saved_live_defaults()
        finally:
            path.chmod(0o644)

    def test_a_file_exactly_at_the_size_limit_is_read(self):
        data = _text(_valid_record())
        config.LIVE_DEFAULTS_FILE.write_bytes(
            data + b" " * (config.LIVE_DEFAULTS_MAX_BYTES - len(data)))
        assert config.read_saved_live_defaults() == _SEED_ADD_ON_OFF

    def test_a_hand_written_record_reads_as_its_toggles(self):
        # Whole numbers and names with surrounding spaces are normalised by LiveSettings
        config.LIVE_DEFAULTS_FILE.write_bytes(_text(_with_toggles(
            spread_band=[0, 1], interval_discount=1, size_cap=1,
            categories=[" Sports "], tags=["Basketball"])))
        saved = config.read_saved_live_defaults()
        assert saved == LiveSettings(False, (0.0, 1.0), 1.0, 1.0, 0.2, ("Sports",),
                                     ("Basketball",))
        assert saved.origin == "live_defaults.json, saved 2026-09-27T21:05:13Z"


class TestLiveDefaults:
    """live_defaults: the saved defaults, which must exist; never config.py's."""

    def test_saved_defaults_are_returned(self):
        config.save_live_defaults(config.LIVE_DEFAULTS_SEED, source="")
        got = config.live_defaults()
        assert got == config.LIVE_DEFAULTS_SEED
        assert got.origin.startswith("live_defaults.json, saved ")

    def test_no_file_raises_missing_naming_the_path_and_both_ways_to_save(self):
        with pytest.raises(config.LiveDefaultsMissing) as err:
            config.live_defaults()
        message = str(err.value)
        assert str(config.LIVE_DEFAULTS_FILE) in message
        assert "Save as live defaults…" in message
        assert "python3 -m kalshi_betting.defaults_server --seed" in message
        assert "./start_dashboard.sh --seed" in message
        # It is also a LiveDefaultsError and a ValueError
        assert isinstance(err.value, config.LiveDefaultsError)
        assert isinstance(err.value, ValueError)

    def test_a_refused_file_raises_the_refusal(self):
        config.LIVE_DEFAULTS_FILE.write_bytes(b"not json")
        with pytest.raises(config.LiveDefaultsError) as err:
            config.live_defaults()
        assert not isinstance(err.value, config.LiveDefaultsMissing)

    def test_it_never_reads_config_py(self, monkeypatch):
        # An invalid constant does not reach it, and a valid file is read as saved
        monkeypatch.setattr(config, "BUDGET_FRACTION", 0.37)
        config.LIVE_DEFAULTS_FILE.write_bytes(_text(_valid_record()))
        assert config.live_defaults() == _SEED_ADD_ON_OFF


class TestLiveDefaultsSeed:
    """LIVE_DEFAULTS_SEED, the starting values offered for a first save."""

    def test_the_seed_values(self):
        seed = config.LIVE_DEFAULTS_SEED
        assert seed == LiveSettings(False, (0.0, 0.5), 0.8, 0.10, 0.20,
                                    add_to_held_pairs=True)
        assert seed.categories is None and seed.tags is None
        assert seed.origin == config.LIVE_DEFAULTS_FROM_CONFIG

    def test_the_seed_warns_nothing(self):
        assert config.live_rule_warnings(config.LIVE_DEFAULTS_SEED) == []

    def test_one_pair_stakes_at_most_ten_percent_of_either_type(self):
        assert max_kelly_fraction("time_series", config.LIVE_DEFAULTS_SEED) == 0.10
        assert max_kelly_fraction("same_title", config.LIVE_DEFAULTS_SEED) == 0.10

    def test_the_seed_saves_and_reads_back(self):
        saved = config.save_live_defaults(config.LIVE_DEFAULTS_SEED,
                                          source=config.LIVE_DEFAULTS_SEED_SOURCE)
        assert saved == config.LIVE_DEFAULTS_SEED
        assert config.read_saved_live_defaults() == config.LIVE_DEFAULTS_SEED
        assert saved.origin.endswith(" from " + config.LIVE_DEFAULTS_SEED_SOURCE)


class TestLiveDefaultsSource:
    """live_defaults_source's rules for the saved note, and the two note shapes
    LIVE_DEFAULTS_SOURCE_PATTERN accepts."""

    def test_a_note_is_stripped_and_may_be_empty(self):
        assert config.live_defaults_source("") == ""
        assert config.live_defaults_source("   ") == ""
        assert config.live_defaults_source("  a note ") == "a note"
        at_limit = "x" * config.LIVE_DEFAULTS_SOURCE_MAX_CHARS
        assert config.live_defaults_source(f" {at_limit} ") == at_limit

    @pytest.mark.parametrize("bad", [None, 7, b"note", ["note"], "two\nlines", "a\ttab",
                                     "zero\u200bwidth", "\u202edirection",
                                     "x" * (config.LIVE_DEFAULTS_SOURCE_MAX_CHARS + 1)])
    def test_a_bad_note_is_refused(self, bad):
        with pytest.raises(ValueError, match="source"):
            config.live_defaults_source(bad)

    @pytest.mark.parametrize("note", [
        config.LIVE_DEFAULTS_SEED_SOURCE,
        "backtest dashboard for 2025-09-24 to 2026-09-27",
        "backtest dashboard for 2025-09-24 to 2026-09-27 (same-event ladders off, "
        "config.py on: its pairs are not the live bot's)",
    ])
    def test_the_pattern_accepts_the_two_shapes(self, note):
        assert re.fullmatch(config.LIVE_DEFAULTS_SOURCE_PATTERN, note)
        assert config.live_defaults_source(note) == note

    @pytest.mark.parametrize("note", [
        "",
        "evil words",
        "backtest dashboard for 2025-09-24 to 2026-09-27; and more",
        "backtest dashboard for 2025-09-24",
        "backtest dashboard for 2025-09-24 to 2026-09-27 (same-event ladders maybe, "
        "config.py on: its pairs are not the live bot's)",
        "seed values",
        " " + config.LIVE_DEFAULTS_SEED_SOURCE,
    ])
    def test_the_pattern_refuses_anything_else(self, note):
        assert re.fullmatch(config.LIVE_DEFAULTS_SOURCE_PATTERN, note) is None


class TestLiveSettingsChanges:
    """live_settings_changes: the saved defaults against proposed ones, toggle
    by toggle, in the "Live settings:" line's order and words."""

    def test_labels_and_values_match_describe_live_settings(self):
        current = _settings(categories=("Sports",), tags=("Basketball",))
        proposed = _settings(False, (0.0, 0.5), 0.8, 1.0, 0.2)
        rows = config.live_settings_changes(current, proposed)
        assert [f"{label} {new}" for label, _, new, _ in rows] == \
            config.describe_live_settings(proposed).split(" | ")
        assert [f"{label} {old}" for label, old, _, _ in rows] == \
            config.describe_live_settings(current).split(" | ")
        assert rows == [
            ("tier floors", "on", "off", True),
            ("spread band", "none", "0-0.5", True),
            ("k", "0.75", "0.8", True),
            ("per-trade cap", "20%", "100% (no cap)", True),
            ("same-title cap", "100% (no extra cap)", "20%", True),
            ("categories", "Sports", "any", True),
            ("tags", "Basketball", "any", True),
            ("add to held pairs", "off", "off", False),
        ]

    def test_changed_flags_exactly_the_fields_that_differ(self):
        ref = _settings()
        for name in config.LIVE_TOGGLE_FIELDS:
            value, _ = TestDescribeLiveSettings._DEPARTURES[name]
            rows = config.live_settings_changes(ref, dataclasses.replace(ref, **{name: value}))
            assert [changed for *_, changed in rows] == [
                other == name for other in config.LIVE_TOGGLE_FIELDS], name
        # Equal toggles change nothing, whatever the origins say
        rows = config.live_settings_changes(dataclasses.replace(ref, origin=_SAVED_ORIGIN), ref)
        assert not any(changed for *_, changed in rows)
        # Case is part of the raw value, as on the "Live settings:" line
        rows = config.live_settings_changes(_settings(categories=("Sports",)),
                                            _settings(categories=("sports",)))
        assert rows[5] == ("categories", "Sports", "sports", True)

    def test_no_current_defaults_changes_every_row(self):
        rows = config.live_settings_changes(None, config.LIVE_DEFAULTS_SEED)
        assert len(rows) == len(config.LIVE_TOGGLE_FIELDS) == 8
        assert all(old == "—" and changed for _, old, _, changed in rows)
        assert [new for _, _, new, _ in rows] == [
            "off", "0-0.5", "0.8", "10%", "20%", "any", "any", "on"]

    def test_a_seven_toggle_file_shows_turning_add_on_on_as_a_change(self):
        # A file that leaves the toggle out reads it as off; the seed proposes on
        rows = config.live_settings_changes(_SEED_ADD_ON_OFF, config.LIVE_DEFAULTS_SEED)
        assert [row for row in rows if row[3]] == [("add to held pairs", "off", "on", True)]


class TestCountText:
    """config.count_text writes a contract count or signed position exactly,
    for every add-on marker and every alert that names what the account held
    before a pair: never in %g's six significant digits."""

    @pytest.mark.parametrize("value, text", [
        (30.0, "30"), (-30.0, "-30"), (12.5, "12.5"), (0.01, "0.01"),
        (1234567.0, "1234567"), (-1234567.0, "-1234567"), (-100000.5, "-100000.5"),
        (12345.67, "12345.67"), (1e-6, "0.000001"), (0.0, "0"), (-0.0, "0"),
        (1e-7, "0"), (-1e-7, "0"), (float("nan"), "nan"), (float("inf"), "inf"),
        (float("-inf"), "-inf"),
    ])
    def test_it_writes_the_number(self, value, text):
        """Pins the exact text for whole, fractional, large, tiny and
        non-finite numbers, and that nothing reads "-0"."""
        assert config.count_text(value) == text

    def test_every_two_decimal_count_reads_back_as_itself(self):
        """Pins that every count the exchange can send (a fixed-point string
        with two decimals, as position_fp is) prints as a number that reads
        back as the same float, however large: no count is ever rounded."""
        for cents in (*range(-100_000, 100_001, 7), 123_456_789, -98_765_432_105):
            value = float(f"{cents / 100:.2f}")
            assert float(config.count_text(value)) == value, cents
