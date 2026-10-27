"""Tests for config.py fee helpers, the time-series probability model, the
leg-side tuples, the deadline-gap tier (with the backtest's spread band and
tier-floors switch), the live toggles (LiveSettings and its helpers), the saved
live defaults (their file's reader and writer, the seed, and the comparison of
two sets of defaults), the weekly run schedule (ScheduledRun), the wait
before a Kelly trim (rebalance_rested, rebalance_min_days, trade_day), the values
config.py ships, conftest's apply_pre_toggle_defaults, the V2 order path's
self-trade-prevention value, the startup check that refuses any order path but
"v2" (order_api_version_error), the order-write pacer's budget, the live-run
lock's exit code, file and waits, and PROJECT_ROOT."""
import ast
import dataclasses
import enum
import importlib
import importlib.util
import json
import logging
import math
import numbers
import os
import pathlib
import random
import re
import sys
import threading
from datetime import UTC, date, datetime, timedelta, timezone, tzinfo
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
    time_series_mid_spread,
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
              same_title_size_cap=1.0, exclude_categories=None, exclude_tags=None, rebalance=False,
              min_price_move=None, rebalance_min_days_to_maturity=None, sell_at=None,
              sell_min_days=None):
    """A LiveSettings with every toggle named, defaulting to the pre-toggle
    values conftest's apply_pre_toggle_defaults pins (no rebalancing, no selling)."""
    return LiveSettings(tier_floors=tier_floors, spread_band=spread_band,
                        interval_discount=interval_discount, size_cap=size_cap,
                        same_title_size_cap=same_title_size_cap,
                        exclude_categories=exclude_categories, exclude_tags=exclude_tags,
                        rebalance=rebalance, min_price_move=min_price_move,
                        rebalance_min_days_to_maturity=rebalance_min_days_to_maturity,
                        sell_at=sell_at, sell_min_days=sell_min_days)


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


def _two_midpoints(yes_ask_a, no_ask_a, yes_ask_b, no_ask_b):
    """mid B - mid A, each midpoint taken the plain way: (YES ask + YES bid) / 2,
    the YES bid being 1 - the NO ask."""
    mid_a = (yes_ask_a + (1.0 - no_ask_a)) / 2.0
    mid_b = (yes_ask_b + (1.0 - no_ask_b)) / 2.0
    return mid_b - mid_a


class TestTimeSeriesMidSpread:
    """time_series_mid_spread is the later market's midpoint minus the earlier
    one's, a midpoint being halfway between a market's YES ask and its YES
    bid (1 - its NO ask). It is computed as the ask gap less half the
    difference in the two books' widths (a width being YES ask + NO ask - 1),
    so on books with no width it is the ask gap to the last bit whenever the
    two YES ask + NO ask sums are the same float, and within float noise of
    it otherwise. A missing YES bid (a NO ask of 1.0) puts that market's
    midpoint at half its YES ask."""

    def test_equals_the_ask_gap_on_every_complementary_cent_book(self):
        # Every pair of whole-cent YES asks pA < pB, each market's NO ask its
        # complement: each book's YES ask + NO ask is exactly 1.0 in floating
        # point, so the mid spread is the ask gap to the last bit (the plain
        # two-midpoint form is not, on some)
        checked = plain = 0
        for i in range(1, 100):
            for j in range(i + 1, 100):
                pA, pB = i / 100, j / 100
                nA, nB = (100 - i) / 100, (100 - j) / 100
                assert pA + nA == 1.0 and pB + nB == 1.0, (pA, pB)
                assert time_series_mid_spread(pA, nA, pB, nB) == pB - pA, (pA, pB)
                plain += _two_midpoints(pA, nA, pB, nB) == pB - pA
                checked += 1
        assert checked == 4851
        assert plain < checked

    def test_books_with_no_width_built_as_the_book_reader_builds_them(self):
        # Each ask taken as the book reader takes it, 1 - the other side's
        # bid: a YES bid at i cents and a NO bid at 100 - i cents give a book
        # with no width, but its YES ask + NO ask need not come to exactly
        # 1.0. Where the two books' sums are the same float the mid spread is
        # the ask gap to the last bit; where they are not, it is within about
        # 1e-16 of it
        def quotes(i):
            return 1.0 - float(f"{(100 - i) / 100:.4f}"), 1.0 - float(f"{i / 100:.4f}")

        same_sum = other_sum = 0
        for i in range(1, 100):
            for j in range(i + 1, 100):
                (pA, nA), (pB, nB) = quotes(i), quotes(j)
                spread = time_series_mid_spread(pA, nA, pB, nB)
                if pA + nA == pB + nB:
                    assert spread == pB - pA, (i, j)
                    same_sum += 1
                else:
                    assert abs(spread - (pB - pA)) <= 1.2e-16, (i, j)
                    other_sum += 1
        # Both kinds occur, so both branches above are exercised
        assert same_sum + other_sum == 4851
        assert same_sum and other_sum

    def test_matches_the_two_midpoints_on_random_quotes(self):
        # Any four 4-decimal quotes, crossed books included: the same number
        # as mid B - mid A, to float noise
        rng = random.Random(20261006)
        for _ in range(20_000):
            quotes = [round(rng.uniform(0.0001, 0.9999), 4) for _ in range(4)]
            assert abs(time_series_mid_spread(*quotes) - _two_midpoints(*quotes)) <= 1e-15, quotes

    def test_a_wider_later_book_makes_the_ask_gap_overstate_it(self):
        # A: YES ask 0.30, NO ask 0.72 (YES bid 0.28, midpoint 0.29, width 0.02).
        # B: YES ask 0.62, NO ask 0.42 (YES bid 0.58, midpoint 0.60, width 0.04).
        # The ask gap is 0.32; half the extra 0.02 of B's width comes off it
        assert time_series_mid_spread(0.30, 0.72, 0.62, 0.42) == pytest.approx(0.31)

    def test_no_yes_bid_puts_the_earlier_midpoint_at_half_its_yes_ask(self):
        # A NO ask of 1.0 stands for no YES bid (a bid of 0): mid A = its YES ask / 2
        for pA, pB, nB in ((0.30, 0.60, 0.42), (0.21, 0.55, 0.47), (0.05, 0.31, 0.70)):
            mid_b = (pB + (1.0 - nB)) / 2.0
            assert time_series_mid_spread(pA, 1.0, pB, nB) == pytest.approx(
                mid_b - pA / 2.0, abs=1e-15)


class TestTimeSeriesProfitProb:
    """p = 1 - k * max(0, spread): one minus the believed fraction k of the
    market-implied probability that the event first happens between the two
    deadlines (the single loss cell of a YES-on-earlier / NO-on-later pair),
    read at the midpoints (the mid spread, time_series_mid_spread; DR-78)."""

    @pytest.mark.usefixtures("pre_toggle_defaults")
    def test_flow_through_fixture(self):
        # A mid spread of 0.30 → p = 1 - 0.75 * 0.30 = 0.775 (k 0.75, pre-toggle)
        assert time_series_profit_prob(0.30) == pytest.approx(0.775)

    def test_matches_definition_from_constant(self):
        for spread in (0.15, 0.30, 0.15, 0.40):
            expected = 1.0 - config.TIME_SERIES_INTERVAL_PROB_DISCOUNT * spread
            assert time_series_profit_prob(spread) == pytest.approx(expected)

    def test_it_reads_the_mid_spread_not_the_ask_gap(self):
        # The model takes the spread it is handed: a later book 0.10 wide puts
        # the mid spread 0.05 under the YES-ask gap, and p reads the former
        spread = time_series_mid_spread(0.30, 0.70, 0.60, 0.50)
        assert spread == pytest.approx(0.25)
        assert time_series_profit_prob(spread, k=0.80) == pytest.approx(0.80)
        assert time_series_profit_prob(spread, k=0.80) != pytest.approx(
            1.0 - 0.80 * (0.60 - 0.30))

    def test_clamps_to_one_when_earlier_is_pricier(self):
        # A spread at or below zero (the later midpoint not above the earlier)
        # would model as riskless, never as p > 1: which is why the sizer never
        # asks with one (scanner.pair_mid_spread refuses it)
        assert time_series_profit_prob(-0.30) == 1.0
        assert time_series_profit_prob(0.0) == 1.0

    def test_discount_of_one_is_market_implied(self, monkeypatch):
        # k = 1 takes the market at face value: p = 1 - spread. Under this
        # model Kelly is <= 0 for every pair (see test_strategy's parity class).
        monkeypatch.setattr(config, "TIME_SERIES_INTERVAL_PROB_DISCOUNT", 1.0)
        assert time_series_profit_prob(0.30) == pytest.approx(0.70)

    def test_discount_of_zero_ignores_the_gap(self, monkeypatch):
        monkeypatch.setattr(config, "TIME_SERIES_INTERVAL_PROB_DISCOUNT", 0.0)
        assert time_series_profit_prob(0.30) == 1.0

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
        assert time_series_profit_prob(0.30, k=0.50) == pytest.approx(0.85)
        assert time_series_profit_prob(0.30, k=1.0) == pytest.approx(0.70)
        assert config.TIME_SERIES_INTERVAL_PROB_DISCOUNT == 0.75

    def test_k_none_is_identical_to_omitting_it(self):
        # k=None reads the config constant, so the backtest's default point
        # prices exactly as live sizing does at config.py's k
        for spread in (0.15, 0.30, 0.15, -0.30):
            assert time_series_profit_prob(spread, k=None) == time_series_profit_prob(spread)

    def test_constant_read_at_call_time_and_only_when_k_is_omitted(self, monkeypatch):
        # The None sentinel must resolve inside the body rather than binding at
        # def time: a monkeypatched constant still governs an override-free
        # call (1 - 0.50 * 0.30 = 0.85)...
        monkeypatch.setattr(config, "TIME_SERIES_INTERVAL_PROB_DISCOUNT", 0.50)
        assert time_series_profit_prob(0.30) == pytest.approx(0.85)
        # ...and is ignored entirely once k is supplied (1 - 0.75 * 0.30)
        assert time_series_profit_prob(0.30, k=0.75) == pytest.approx(0.775)


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
            # The forecast's input, as enrichment writes it from the books' tops
            p.mid_spread = (time_series_mid_spread(pA, nA, pB, nB)
                            if pair_type == "time_series" else None)
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


    # 1e-7 and 1e-6 round to zero steps (within PRICE_EPSILON of 0); 0.855 sits
    # between two 1% steps
    @pytest.mark.parametrize("bad", [0, 0.0, -0.05, 1.01, 2, 0.855, 0.001, 1e-7, 1e-6,
                                     float("nan"), float("inf"), True, False, "0.85", [0.85]])
    def test_sell_at_off_the_grid_or_out_of_range_is_refused(self, bad):
        with pytest.raises(ValueError, match="sell_at"):
            _settings(sell_at=bad)
        with pytest.raises(ValueError, match="sell_at"):
            dataclasses.replace(_settings(), sell_at=bad)

    @pytest.mark.parametrize("level", [0.01, 0.29, 0.5, 0.8, 0.85, 0.99, 1.0, 1])
    def test_sell_at_on_the_grid_is_accepted_as_a_float(self, level):
        s = _settings(sell_at=level)
        assert s.sell_at == level and type(s.sell_at) is float

    def test_sell_at_is_normalised_onto_the_grid(self):
        # 0.01 * 35 is not the float 0.35; every level from any source is
        # float-equal to its percent over 100, which is how
        # TAKE_PROFIT_LEVELS spells them
        for percent in range(1, 101):
            assert _settings(sell_at=0.01 * percent).sell_at == percent / 100
            assert _settings(sell_at=percent / 100).sell_at == percent / 100
        for level in config.TAKE_PROFIT_LEVELS:
            assert _settings(sell_at=level).sell_at == level

    def test_sell_at_defaults_to_never_selling(self):
        s = LiveSettings(True, (0.0, 1.0), 0.75, 0.2)
        assert s.sell_at is None and s.sell_min_days is None
        assert _settings().sell_at is None and _settings().sell_min_days is None
        # None is a value, and replace takes it back
        on = _settings(sell_at=0.85, sell_min_days=3)
        assert dataclasses.replace(on, sell_at=None, sell_min_days=None) == _settings()

    @pytest.mark.parametrize("bad", [0, -1, 1.0, 3.0, 2.5, True, False, "3", [3],
                                     float("nan"), float("inf")])
    def test_sell_min_days_must_be_a_whole_number_of_days_one_or_more(self, bad):
        # A bool is an int in Python and a float such as 1.0 an int in JSON's
        # eyes: neither may pass for a day count
        with pytest.raises(ValueError, match="sell_min_days"):
            _settings(sell_at=0.85, sell_min_days=bad)
        with pytest.raises(ValueError, match="sell_min_days"):
            dataclasses.replace(_settings(sell_at=0.85), sell_min_days=bad)

    @pytest.mark.parametrize("days", [1, 2, 7, 14, 21, 365])
    def test_sell_min_days_is_accepted_with_a_level(self, days):
        s = _settings(sell_at=0.85, sell_min_days=days)
        assert s.sell_min_days == days and type(s.sell_min_days) is int

    def test_sell_min_days_without_a_level_is_refused(self):
        with pytest.raises(ValueError, match="sell_min_days needs a sell level"):
            _settings(sell_min_days=3)
        # replace re-validates: taking the level away from a minimum is refused too
        on = _settings(sell_at=0.85, sell_min_days=3)
        with pytest.raises(ValueError, match="sell_min_days needs a sell level"):
            dataclasses.replace(on, sell_at=None)
        # A level without a minimum is fine
        assert dataclasses.replace(on, sell_min_days=None).sell_at == 0.85

    @pytest.mark.parametrize("bad", [1, 0, None, "True", "on", 1.0])
    def test_rebalance_must_be_exactly_a_bool(self, bad):
        # A saved file's JSON 1 or "true" must never read as on
        with pytest.raises(ValueError, match="rebalance must be True or False"):
            _settings(rebalance=bad)
        with pytest.raises(ValueError, match="rebalance must be True or False"):
            dataclasses.replace(_settings(), rebalance=bad)

    def test_rebalancing_defaults_to_off_with_no_minimum(self):
        # A construction that names neither reads rebalancing as off with no
        # minimum (a saved file that leaves them out: TestSavedLiveDefaults)
        s = LiveSettings(True, (0.0, 1.0), 0.75, 0.2)
        assert s.rebalance is False and s.min_price_move is None
        assert _settings().rebalance is False and _settings().min_price_move is None
        on = _settings(rebalance=True)
        assert on.rebalance is True and on.min_price_move is None
        # replace can turn both back off
        both = _settings(rebalance=True, min_price_move=0.10)
        assert dataclasses.replace(both, rebalance=False, min_price_move=None) == _settings()

    def test_the_old_settings_are_no_fields(self):
        # Rebalance replaced the two switches and the trim minimum: none is a field
        assert not set(config._RETIRED_TOGGLES) & {f.name for f in dataclasses.fields(
            LiveSettings)}
        with pytest.raises(TypeError):
            LiveSettings(True, (0.0, 1.0), 0.75, 0.2, add_to_held_pairs=True)

    # 0.07 and 0.125 sit between 5c steps; 1e-7 and 1e-6 round to zero steps;
    # 1.05 is past $1; 10 is ten dollars, not ten cents
    @pytest.mark.parametrize("bad", [
        0, 0.0, -0.05, 0.01, 0.07, 0.125, 0.049, 1.05, 1.01, 2, 10, 100, 1e-7, 1e-6,
        float("nan"), float("inf"), float("-inf"), 1e308, 10 ** 400, True, False,
        "0.10", "10c", [0.10], (0.10,)])
    def test_min_price_move_off_the_5c_grid_or_out_of_range_is_refused(self, bad):
        with pytest.raises(ValueError, match="min_price_move"):
            _settings(rebalance=True, min_price_move=bad)
        with pytest.raises(ValueError, match="min_price_move"):
            dataclasses.replace(_settings(rebalance=True), min_price_move=bad)

    def test_min_price_move_names_its_grid_or_its_type(self):
        # Off the grid names the grid; a non-number (a bool too) asks for dollars
        with pytest.raises(ValueError, match=re.escape(
                "min_price_move must be a multiple of 5c from 5c to $1, got 0.07")):
            _settings(rebalance=True, min_price_move=0.07)
        for bad in (True, "0.10"):
            with pytest.raises(ValueError, match=re.escape(
                    f"min_price_move must be a number of dollars, got {bad!r}")):
                _settings(rebalance=True, min_price_move=bad)

    def test_min_price_move_on_the_grid_is_accepted_as_a_float(self):
        # Every 5c step from 5c to $1, as its cents over 100
        step = config.PRICE_MOVE_STEP_CENTS
        assert step == 5
        for cents in range(step, 101, step):
            s = _settings(rebalance=True, min_price_move=cents / 100)
            assert s.min_price_move == cents / 100 and type(s.min_price_move) is float
        # A whole dollar given as an int reads as the float 1.0
        one = _settings(rebalance=True, min_price_move=1)
        assert one.min_price_move == 1.0 and type(one.min_price_move) is float

    def test_min_price_move_is_normalised_onto_the_grid(self):
        # 0.05 * 3 is 0.15000000000000002; every minimum comes back as its cents
        # over 100, as the dashboard's options (MIN_PRICE_MOVES) spell them
        assert 0.05 * 3 != 0.15
        assert _settings(rebalance=True, min_price_move=0.05 * 3).min_price_move == 0.15
        for steps in range(1, 21):
            assert _settings(rebalance=True,
                             min_price_move=0.05 * steps).min_price_move == steps * 5 / 100
        for move in config.MIN_PRICE_MOVES:
            assert _settings(rebalance=True, min_price_move=move).min_price_move == move
        # Float noise well inside PRICE_EPSILON snaps to the step
        assert _settings(rebalance=True, min_price_move=0.1 + 1e-9).min_price_move == 0.1

    def test_min_price_move_with_rebalancing_off_is_refused(self):
        with pytest.raises(ValueError, match="min_price_move needs rebalance on"):
            _settings(min_price_move=0.10)
        with pytest.raises(ValueError, match="min_price_move needs rebalance on"):
            _settings(rebalance=False, min_price_move=0.10)
        # replace re-validates: turning rebalancing off under a minimum is refused too
        on = _settings(rebalance=True, min_price_move=0.10)
        with pytest.raises(ValueError, match="min_price_move needs rebalance on"):
            dataclasses.replace(on, rebalance=False)
        # Rebalancing with no minimum is fine, and so is turning both off together
        assert dataclasses.replace(on, min_price_move=None).rebalance is True
        assert dataclasses.replace(on, rebalance=False, min_price_move=None) == _settings()

    def test_a_bad_minimum_is_named_before_the_switch_it_needs(self):
        # Off the grid with rebalancing off: the bad value is named first
        with pytest.raises(ValueError, match="multiple of 5c"):
            _settings(min_price_move=0.07)

    def test_rebalance_until_defaults_to_maturity(self):
        # A construction that names no days reads "maturity": always rebalance
        assert LiveSettings(True, (0.0, 1.0), 0.75, 0.2).rebalance_min_days_to_maturity is None
        assert _settings().rebalance_min_days_to_maturity is None
        assert _settings(rebalance=True).rebalance_min_days_to_maturity is None

    def test_rebalance_until_takes_the_dashboards_three_choices(self):
        assert config.REBALANCE_MATURITY_DAYS == (0, 1, 7)
        for days in (1, 7):
            s = _settings(rebalance=True, rebalance_min_days_to_maturity=days)
            assert s.rebalance_min_days_to_maturity == days
            assert type(s.rebalance_min_days_to_maturity) is int

    def test_rebalance_until_zero_is_stored_as_maturity(self):
        # 0 days is "always", the same as None, with rebalancing on or off
        for rebalance in (True, False):
            s = _settings(rebalance=rebalance, rebalance_min_days_to_maturity=0)
            assert s.rebalance_min_days_to_maturity is None
            assert s == _settings(rebalance=rebalance)
        # replace re-validates, so it turns a set value back to "maturity" too
        on = _settings(rebalance=True, rebalance_min_days_to_maturity=7)
        assert dataclasses.replace(on, rebalance_min_days_to_maturity=0) == _settings(
            rebalance=True)

    @pytest.mark.parametrize("bad", [
        2, 3, 6, 8, 14, 30, -1, -7, 100, True, False, 1.0, 7.0, 0.0, 0.5, float("nan"),
        float("inf"), "7", "1", "maturity", [7], (7,), 10 ** 400])
    def test_rebalance_until_other_than_zero_one_or_seven_is_refused(self, bad):
        with pytest.raises(ValueError, match=re.escape(
                f"rebalance_min_days_to_maturity must be 0, 1 or 7, got {bad!r}")):
            _settings(rebalance=True, rebalance_min_days_to_maturity=bad)
        with pytest.raises(ValueError, match="rebalance_min_days_to_maturity"):
            dataclasses.replace(_settings(rebalance=True), rebalance_min_days_to_maturity=bad)

    def test_rebalance_until_refuses_a_numpy_integer(self):
        import numpy as np

        # A numpy whole number is not a plain int, as sell_min_days has it
        for bad in (np.int64(1), np.int64(7), np.int64(0), np.uint8(7), np.bool_(True)):
            with pytest.raises(ValueError, match="rebalance_min_days_to_maturity must be"):
                _settings(rebalance=True, rebalance_min_days_to_maturity=bad)

    def test_rebalance_until_with_rebalancing_off_is_refused(self):
        with pytest.raises(ValueError, match=re.escape(
                "rebalance_min_days_to_maturity needs rebalance on, got "
                "rebalance_min_days_to_maturity 7 with rebalance off")):
            _settings(rebalance_min_days_to_maturity=7)
        with pytest.raises(ValueError, match="rebalance_min_days_to_maturity needs rebalance on"):
            _settings(rebalance=False, rebalance_min_days_to_maturity=1)
        # replace re-validates: turning rebalancing off under days is refused too
        on = _settings(rebalance=True, rebalance_min_days_to_maturity=1)
        with pytest.raises(ValueError, match="rebalance_min_days_to_maturity needs rebalance on"):
            dataclasses.replace(on, rebalance=False)
        # Rebalancing with no days is fine, and so is turning both off together
        assert dataclasses.replace(
            on, rebalance_min_days_to_maturity=None).rebalance is True
        assert dataclasses.replace(
            on, rebalance=False, rebalance_min_days_to_maturity=None) == _settings()

    def test_a_bad_rebalance_until_is_named_before_the_switch_it_needs(self):
        # Not 0, 1 or 7 with rebalancing off: the bad value is named first
        with pytest.raises(ValueError, match="must be 0, 1 or 7"):
            _settings(rebalance_min_days_to_maturity=2)

    @pytest.mark.parametrize("doc, stale", [
        ("README.md", "since live trading has no such setting"),
        ("CLAUDE.md", "there is no live setting to save it as"),
        ("CLAUDE.md", "What is missing: a toggle on `LiveSettings`"),
        ("CLAUDE.md", "missing for the minimum: a `trim_min_rise` setting on `LiveSettings`"),
    ])
    def test_the_docs_do_not_say_live_has_no_trim_setting(self, doc, stale):
        # Rebalancing and its minimum are live settings, so neither document may
        # say live trading lacks a trim setting
        assert {"rebalance", "min_price_move"} <= set(config.LIVE_TOGGLE_FIELDS)
        text = (pathlib.Path(__file__).resolve().parent.parent / doc).read_text(
            encoding="utf-8")
        # Whitespace is collapsed, so a sentence wrapped over lines still matches
        assert stale not in " ".join(text.split())

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
        monkeypatch.setattr(config, "TRADE_EXCLUDE_CATEGORIES", ("Economics",))
        monkeypatch.setattr(config, "TRADE_EXCLUDE_TAGS", ["Oil & Gas"])
        monkeypatch.setattr(config, "REBALANCE", True)
        monkeypatch.setattr(config, "MIN_PRICE_MOVE", 0.10)
        monkeypatch.setattr(config, "REBALANCE_MIN_DAYS_TO_MATURITY", 7)
        monkeypatch.setattr(config, "SELL_AT", 0.85)
        monkeypatch.setattr(config, "SELL_MIN_DAYS", 3)
        expected = _settings(True, (0.1, 0.6), 0.6, 0.35, 0.25,
                             ("Economics",), ("Oil & Gas",),
                             rebalance=True, min_price_move=0.10,
                             rebalance_min_days_to_maturity=7,
                             sell_at=0.85, sell_min_days=3)
        assert live_settings() == expected
        # A minimum needs rebalancing on, so rebalance itself is checked apart
        assert all(getattr(expected, name) != getattr(shipped, name)
                   for name in config.LIVE_TOGGLE_FIELDS if name != "rebalance")
        monkeypatch.setattr(config, "REBALANCE", not shipped.rebalance)
        monkeypatch.setattr(config, "MIN_PRICE_MOVE", None)
        monkeypatch.setattr(config, "REBALANCE_MIN_DAYS_TO_MATURITY", None)
        assert live_settings().rebalance is (not shipped.rebalance)

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
            exclude_categories=config.TRADE_EXCLUDE_CATEGORIES,
            exclude_tags=config.TRADE_EXCLUDE_TAGS,
            rebalance=config.REBALANCE,
            min_price_move=config.MIN_PRICE_MOVE,
            rebalance_min_days_to_maturity=config.REBALANCE_MIN_DAYS_TO_MATURITY,
            sell_at=config.SELL_AT,
            sell_min_days=config.SELL_MIN_DAYS,
        )
        assert type(s.tier_floors) is bool and type(s.rebalance) is bool

    def test_live_settings_refuses_an_invalid_sell_constant(self, monkeypatch):
        monkeypatch.setattr(config, "SELL_AT", 0.855)
        with pytest.raises(ValueError, match="sell_at"):
            live_settings()
        monkeypatch.setattr(config, "SELL_AT", None)
        monkeypatch.setattr(config, "SELL_MIN_DAYS", 3)
        with pytest.raises(ValueError, match="sell_min_days needs a sell level"):
            live_settings()

    def test_live_settings_refuses_an_invalid_rebalance_constant(self, monkeypatch):
        monkeypatch.setattr(config, "REBALANCE", 1)
        with pytest.raises(ValueError, match="rebalance"):
            live_settings()
        # A minimum off the 5c grid, then one with rebalancing off
        monkeypatch.setattr(config, "REBALANCE", True)
        monkeypatch.setattr(config, "MIN_PRICE_MOVE", 0.07)
        with pytest.raises(ValueError, match="min_price_move must be a multiple of 5c"):
            live_settings()
        monkeypatch.setattr(config, "REBALANCE", False)
        monkeypatch.setattr(config, "MIN_PRICE_MOVE", 0.10)
        with pytest.raises(ValueError, match="min_price_move needs rebalance on"):
            live_settings()
        # The days to maturity: not 0, 1 or 7, then one with rebalancing off
        monkeypatch.setattr(config, "MIN_PRICE_MOVE", None)
        monkeypatch.setattr(config, "REBALANCE", True)
        monkeypatch.setattr(config, "REBALANCE_MIN_DAYS_TO_MATURITY", 2)
        with pytest.raises(ValueError, match="rebalance_min_days_to_maturity must be 0, 1 or 7"):
            live_settings()
        monkeypatch.setattr(config, "REBALANCE", False)
        monkeypatch.setattr(config, "REBALANCE_MIN_DAYS_TO_MATURITY", 7)
        with pytest.raises(ValueError, match="rebalance_min_days_to_maturity needs rebalance on"):
            live_settings()

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

    @pytest.mark.parametrize("field", ["exclude_categories", "exclude_tags"])
    @pytest.mark.parametrize("bad", [
        "Sports",            # a bare str would filter on its characters
        (), [],              # matches nothing: None is "any"
        ("",), ("  ",),      # an empty name, before or after stripping
        # "any" and "none" would read exactly like None
        ("any",), ("Sports", "ANY"), (" Any ",), ("none",), ("Sports", "NONE"),
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

    @pytest.mark.parametrize("field", ["exclude_categories", "exclude_tags"])
    def test_a_filter_is_normalised_to_a_tuple_of_stripped_names(self, field):
        s = _settings(**{field: [" Sports ", "Oil & Gas"]})
        assert getattr(s, field) == ("Sports", "Oil & Gas")
        assert type(getattr(s, field)) is tuple
        # Order and case are kept (main._filter_by_category ignores case)
        assert getattr(_settings(**{field: ("b", "A")}), field) == ("b", "A")
        assert getattr(_settings(**{field: None}), field) is None

    def test_a_construction_without_the_filters_filters_nothing(self):
        s = LiveSettings(True, (0.0, 1.0), 0.75, 0.2, 1.0)
        assert s.exclude_categories is None and s.exclude_tags is None
        assert s.rebalance is False and s.min_price_move is None
        assert s.rebalance_min_days_to_maturity is None
        assert s.sell_at is None and s.sell_min_days is None
        # Defaulted and last of the toggles: the first five fields keep their
        # positions, so a positional construction of up to seven still builds;
        # origin, not a toggle, comes after them
        assert list(config.LIVE_TOGGLE_FIELDS)[-7:] == [
            "exclude_categories", "exclude_tags", "rebalance", "min_price_move",
            "rebalance_min_days_to_maturity", "sell_at", "sell_min_days"]
        assert [f.name for f in dataclasses.fields(LiveSettings)][-1] == "origin"
        assert LiveSettings(True, (0.0, 1.0), 0.75, 0.2, 1.0, ("Sports",),
                            ("Hockey",)).exclude_tags == ("Hockey",)

    @pytest.mark.parametrize("name, value", [("TRADE_EXCLUDE_CATEGORIES", "Sports"),
                                             ("TRADE_EXCLUDE_TAGS", ())])
    def test_live_settings_refuses_an_invalid_filter(self, monkeypatch, name, value):
        monkeypatch.setattr(config, name, value)
        with pytest.raises(ValueError, match="exclude_categories" if "CATEG" in name else "exclude_tags"):
            live_settings()


# Pairs as (category, first tag) for TestTradeFilter's truth table
_FILED = (("Economics", "Fed"), ("Economics", "Oil"), ("Sports", "Basketball"),
          ("Sports", "Soccer"), ("Politics", "Fed"), ("Politics", "Basketball"))


class TestFilterNameRefusal:
    """
    config.filter_name_refusal says why one excluded category or tag would be
    refused — by the form check the defaults server reads every link's names
    through (config.check_filter_name, which defaults_server._name calls) or by
    LiveSettings — so the dashboard's Save button can keep such a name out of
    its link. A name it passes is one LiveSettings takes.
    """

    @pytest.mark.parametrize("field, name", [
        ("exclude_categories", "Sports"), ("exclude_categories", "Companies"),
        ("exclude_tags", "Golf"), ("exclude_tags", "Sports · Golf"),
        ("exclude_tags", "·Golf"), ("exclude_tags", "Golf (men's) ~ open!")])
    def test_a_name_it_passes_is_taken(self, field, name):
        assert config.filter_name_refusal(field, name) is None
        config.LiveSettings(tier_floors=True, spread_band=(0.0, 1.0), interval_discount=0.5,
                            size_cap=1.0, **{field: (name,)})

    @pytest.mark.parametrize("field, name, words", [
        ("exclude_categories", "Any", "cannot hold the name"),
        ("exclude_categories", "none", "cannot hold the name"),
        ("exclude_categories", "Arts · Crafts", "is a tag tied to a category"),
        ("exclude_categories", " Sports", "no spaces around it"),
        ("exclude_categories", "Sports\u200b", "one printable name"),
        ("exclude_tags", "Sports · Any", "names no tag"),
        ("exclude_tags", "Sports ·", "must read"),
        ("exclude_tags", "Sports ·  Golf", "two spaces in a row"),
        ("exclude_tags", "NONE", "cannot hold the name")])
    def test_a_name_it_refuses_says_why(self, field, name, words):
        refusal = config.filter_name_refusal(field, name)
        assert refusal is not None and words in refusal, refusal

    def test_the_server_reads_names_through_the_same_check(self):
        from kalshi_betting import defaults_server
        for text in ("Sports ·  Golf", " Golf", "Golf\u200b", "Golf"):
            try:
                expected = config.check_filter_name(text, "exclude_tag")
            except ValueError as exc:
                with pytest.raises(ValueError) as caught:
                    defaults_server._name(text, "exclude_tag")
                assert str(caught.value) == str(exc)
            else:
                assert defaults_server._name(text, "exclude_tag") == expected == text


class TestTradeFilter:
    """config.trade_filter, the one definition of the category/tag filter: a
    pair trades unless its category is excluded or its first tag is excluded,
    where a plain excluded tag applies under every category and a tag tied to
    a category ("Sports · Basketball") under that one alone."""

    @staticmethod
    def _kept(excluded, tags):
        """The pairs of _FILED the filter keeps, in order."""
        keeps = config.trade_filter(_settings(exclude_categories=excluded, exclude_tags=tags))
        return [pair for pair in _FILED if keeps(*pair)]

    def test_the_separator_is_a_spaced_middle_dot(self):
        assert config.TAG_SCOPE_SEPARATOR == " · "

    @pytest.mark.parametrize("name, expected", [
        ("Sports · Basketball", ("Sports", "Basketball")),
        ("Basketball", (None, "Basketball")),
        # Only the spaced middle dot ties a tag
        ("Sports·Basketball", (None, "Sports·Basketball")),
        ("Sports - Basketball", (None, "Sports - Basketball")),
        ("Sports . Basketball", (None, "Sports . Basketball")),
        # Each half is stripped; the split is at the first separator
        ("Sports  ·  Basketball", ("Sports", "Basketball")),
        ("Climate and Weather · Oil & Gas", ("Climate and Weather", "Oil & Gas")),
        ("A · B · C", ("A", "B · C")),
    ])
    def test_split_tag(self, name, expected):
        assert config.split_tag(name) == expected

    @pytest.mark.parametrize("excluded, tags, expected", [
        # No filter keeps everything
        (None, None, list(_FILED)),
        # Excluded categories only: every other category in full
        (("Economics",), None, list(_FILED[2:])),
        (("Economics", "Sports"), None, list(_FILED[4:])),
        # Plain excluded tags only: that tag left out under every category
        (None, ("Basketball",),
         [("Economics", "Fed"), ("Economics", "Oil"), ("Sports", "Soccer"),
          ("Politics", "Fed")]),
        (None, ("Fed", "Soccer"),
         [("Economics", "Oil"), ("Sports", "Basketball"), ("Politics", "Basketball")]),
        # An excluded category and an excluded plain tag both leave pairs out
        (("Politics",), ("Basketball",),
         [("Economics", "Fed"), ("Economics", "Oil"), ("Sports", "Soccer")]),
        (("Sports",), ("Fed",), [("Economics", "Oil"), ("Politics", "Basketball")]),
        # A tied tag leaves out that tag in its own category alone
        (None, ("Sports · Basketball",),
         [("Economics", "Fed"), ("Economics", "Oil"), ("Sports", "Soccer"),
          ("Politics", "Fed"), ("Politics", "Basketball")]),
        (("Economics", "Politics"), ("Sports · Basketball",), [("Sports", "Soccer")]),
        # Every tag of a category left out leaves nothing of it
        (None, ("Sports · Basketball", "Sports · Soccer"),
         [("Economics", "Fed"), ("Economics", "Oil"), ("Politics", "Fed"),
          ("Politics", "Basketball")]),
        # A tied tag never applies under another category
        (("Economics",), ("Sports · Basketball", "Politics · Fed"),
         [("Sports", "Soccer"), ("Politics", "Basketball")]),
        # Plain and tied together: each leaves out its own pairs
        (None, ("Fed", "Sports · Basketball"),
         [("Economics", "Oil"), ("Sports", "Soccer"), ("Politics", "Basketball")]),
        # A tied tag in an excluded category is allowed, and leaves out nothing more
        (("Sports",), ("Sports · Basketball",), [pair for pair in _FILED if pair[0] != "Sports"]),
        # Excluding every category keeps nothing
        (("Economics", "Sports", "Politics"), None, []),
        # An excluded name no pair is filed under excludes nothing
        (("World",), None, list(_FILED)),
        (None, ("Hockey", "World · Fed"), list(_FILED)),
    ])
    def test_the_truth_table(self, excluded, tags, expected):
        assert self._kept(excluded, tags) == expected

    def test_a_category_or_tag_never_seen_trades(self):
        # The point of an exclude list: a category or tag nobody listed still trades
        keeps = config.trade_filter(_settings(exclude_categories=("Politics", "Sports"),
                                              exclude_tags=("Fed", "Economics · Oil")))
        assert keeps("Mentions", "Earnings") and keeps("A brand-new category", "General")
        assert keeps("Economics", "A brand-new tag")
        assert not keeps("Politics", "Fed") and not keeps("sports", "Soccer")
        assert not keeps("World", "fed") and not keeps("ECONOMICS", "oil")

    @pytest.mark.parametrize("excluded, tags", [
        (None, None), (("Economics",), None), (None, ("Basketball",)),
        (("Economics",), ("Sports · Basketball", "Politics · Fed")),
        (("economics", "POLITICS"), ("Fed", "sports · soccer")),
        (("Sports",), ("sports · Soccer", "Basketball")),
    ])
    def test_the_two_field_entry_point_is_the_same_rule(self, excluded, tags):
        # The dashboard holds the recorded fields only; it asks the very rule a run's
        # settings are kept by
        by_settings = config.trade_filter(_settings(exclude_categories=excluded,
                                                    exclude_tags=tags))
        by_fields = config.trade_filter_for(excluded, tags)
        assert all(by_fields(*pair) is by_settings(*pair) for pair in _FILED)
        assert [pair for pair in _FILED if by_fields(*pair)] == self._kept(excluded, tags)

    def test_the_trade_filter_is_built_by_the_two_field_entry_point(self):
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(config.trade_filter))
        called = {n.func.id for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "trade_filter_for" in called

    def test_names_are_compared_without_regard_to_case(self):
        # Both halves of a tied tag, the excluded categories, and the pair's own labels
        assert self._kept(("pOLITICS",), ("SPORTS · basketBALL",)) == [
            ("Economics", "Fed"), ("Economics", "Oil"), ("Sports", "Soccer")]
        keeps = config.trade_filter(
            _settings(exclude_categories=("Politics",), exclude_tags=("Sports · Basketball",)))
        assert not keeps("sports", "BASKETBALL") and not keeps("SPORTS", "basketball")
        assert keeps("ECONOMICS", "anything") and keeps("sports", "SOCCER")
        assert keeps("economics", "basketball")
        assert not keeps("politics", "anything")
        # Casefold, as the filter always compared: "ß" matches "SS"
        assert not config.trade_filter(_settings(exclude_tags=("Fußball",)))("Sports", "FUSSBALL")
        assert not config.trade_filter(
            _settings(exclude_categories=("Fußball",)))("FUSSBALL", "x")

    def test_plain_tags_keep_what_two_exclusions_say(self):
        # With no tied tag the filter is two tests: neither the category nor
        # the tag is excluded, in any case
        rng = random.Random(20261023)
        categories = ("Economics", "Sports", "Politics", "World", "Climate and Weather")
        tags = ("Basketball", "Soccer", "Fed", "General", "Oil & Gas", "Sports", "COVID")

        def recase(name):
            return rng.choice((name, name.upper(), name.lower(), name.swapcase()))

        def draw(pool):
            if rng.random() < 0.25:
                return None
            return tuple(recase(n) for n in rng.sample(pool, rng.randint(1, len(pool))))

        filters = 0
        kept = 0
        for _ in range(2000):
            settings = _settings(exclude_categories=draw(categories), exclude_tags=draw(tags))
            keeps = config.trade_filter(settings)
            out = {c.casefold() for c in settings.exclude_categories or ()}
            out_tags = {t.casefold() for t in settings.exclude_tags or ()}
            for c in categories:
                for t in tags:
                    for category, tag in ((c, t), (recase(c), recase(t))):
                        expected = (category.casefold() not in out
                                    and tag.casefold() not in out_tags)
                        assert keeps(category, tag) is expected, (settings, category, tag)
                        kept += expected
            filters += 1
        # Both verdicts were exercised
        assert filters == 2000 and 0 < kept < 2000 * len(categories) * len(tags) * 2

    def test_tied_tags_keep_what_a_category_by_category_exclusion_says(self):
        # Tied and plain tags, drawn at random, against the rule read per pair:
        # out when its category is excluded, its tag is excluded plainly, or
        # its (category, tag) is excluded tied
        rng = random.Random(20261024)
        categories = ("Economics", "Sports", "Politics", "World")
        tags = ("Basketball", "Soccer", "Fed", "General")
        tied_pool = [f"{c}{config.TAG_SCOPE_SEPARATOR}{t}" for c in categories for t in tags]
        for _ in range(500):
            excluded = (tuple(rng.sample(categories, rng.randint(1, 2)))
                        if rng.random() < 0.5 else None)
            names = rng.sample(tied_pool + list(tags), rng.randint(1, 6))
            names = tuple(rng.choice((n, n.upper(), n.lower())) for n in names)
            keeps = config.trade_filter(_settings(exclude_categories=excluded,
                                                  exclude_tags=names))
            out = {c.casefold() for c in excluded or ()}
            plain = {n.casefold() for n in names if config.split_tag(n)[0] is None}
            tied = {(c.casefold(), t.casefold()) for c, t in map(config.split_tag, names)
                    if c is not None}
            for c in categories:
                for t in tags:
                    expected = not (c.casefold() in out or t.casefold() in plain
                                    or (c.casefold(), t.casefold()) in tied)
                    assert keeps(c, t) is expected, (excluded, names, c, t)

    @pytest.mark.parametrize("excluded", [("Sports",), ("Economics", "sports"), ("SPORTS",)])
    def test_a_tied_tag_in_an_excluded_category_is_allowed(self, excluded):
        # Redundant, never an error: it can leave out nothing the category does not
        s = _settings(exclude_categories=excluded, exclude_tags=("Sports · Basketball",))
        assert s.exclude_tags == ("Sports · Basketball",)
        assert not config.trade_filter(s)("Sports", "Soccer")
        s = _settings(exclude_categories=excluded, exclude_tags=("Fed", "Sports · Basketball"))
        assert s.exclude_tags == ("Fed", "Sports · Basketball")

    @pytest.mark.parametrize("excluded", [None, ("Economics",), ("Economics", "Politics")])
    def test_a_tied_tag_with_its_category_not_excluded_excludes_that_tag(self, excluded):
        s = _settings(exclude_categories=excluded, exclude_tags=("Sports · Basketball",))
        assert s.exclude_tags == ("Sports · Basketball",)
        keeps = config.trade_filter(s)
        assert not keeps("Sports", "Basketball") and keeps("Sports", "Soccer")

    def test_a_tied_tag_is_kept_when_a_field_is_replaced(self):
        s = _settings(exclude_categories=("economics",), exclude_tags=("SPORTS · Basketball",))
        assert s.exclude_tags == ("SPORTS · Basketball",)
        # main.py lays --exclude-category over saved defaults this way: excluding
        # the tied tag's category too is allowed
        for excluded in (("Sports",), ("Economics", "sports"), None):
            assert (dataclasses.replace(s, exclude_categories=excluded).exclude_tags
                    == s.exclude_tags)
        assert dataclasses.replace(s, exclude_tags=None).exclude_tags is None
        # ... and a bad tag is still refused when the field is replaced
        with pytest.raises(ValueError, match="exclude_tags: .* names no tag"):
            dataclasses.replace(s, exclude_tags=("Sports · any",))

    @pytest.mark.parametrize("bad", [
        "Sports · ", " · Basketball", "Sports ·", "· Basketball", " · ", "·",
        "Sports ·  ", "  · Basketball", " ·  · ",
    ])
    def test_a_tied_tag_with_a_blank_half_is_refused(self, bad):
        # Never read as a plain tag, with a category excluded or not
        for excluded in (("Economics",), None):
            with pytest.raises(ValueError, match="exclude_tags: .* must read 'Category · Tag'"):
                _settings(exclude_categories=excluded, exclude_tags=(bad,))
            with pytest.raises(ValueError, match="must read 'Category · Tag'"):
                _settings(exclude_categories=excluded, exclude_tags=("Basketball", bad))

    @pytest.mark.parametrize("bad", ["Sports · any", "Sports · ANY", "Sports ·  Any "])
    def test_a_tied_tag_cannot_name_any(self, bad):
        with pytest.raises(ValueError, match="exclude_tags: .* names no tag; to exclude every "
                                             "tag of 'Sports', exclude the category"):
            _settings(exclude_tags=(bad,))

    def test_a_dot_inside_a_plain_name_is_not_a_tie(self):
        # A tie with a blank half is the dot alone, or a name that starts with
        # the dot and a space or ends with a space and the dot; any other dot
        # is part of a plain tag's name
        s = _settings(exclude_tags=("A·B", "U.S.·ish", "Q·R S"))
        assert s.exclude_tags == ("A·B", "U.S.·ish", "Q·R S")
        assert not config.trade_filter(s)("Sports", "a·b")
        assert config.trade_filter(s)("Sports", "a")

    @pytest.mark.parametrize("plain", ["·Basketball", "Sports·", "·Sports·", "··", "a ·b", "a· b"])
    def test_a_dot_with_no_space_beside_it_at_either_end_is_a_plain_tag(self, plain):
        # The boundary of the blank-half refusal: these have no category
        for excluded in (None, ("Economics",)):
            s = _settings(exclude_categories=excluded, exclude_tags=(plain,))
            assert s.exclude_tags == (plain,)
            assert config.split_tag(plain) == (None, plain)
            keeps = config.trade_filter(s)
            assert not keeps("Sports", plain.upper()) and keeps("Sports", "Basketball")
        # ... while one more space makes each a tie with a blank half
        for bad in ("· Basketball", "Sports ·"):
            with pytest.raises(ValueError, match="must read 'Category · Tag'"):
                _settings(exclude_tags=(bad,))

    @pytest.mark.parametrize("bad", [
        "Sports · Basketball", "Sports · ", " · Basketball", "Sports ·", "· Basketball",
        " · ", "·", "A · B · C", "Sports  ·  Basketball",
    ])
    def test_a_category_written_as_a_tied_tag_is_refused(self, bad):
        # With tags or without: excluding it would read as "Basketball of Sports
        # excluded" while it excluded nothing
        message = (rf"exclude_categories: {re.escape(repr(bad.strip()))} is a tag tied to a "
                   r"category; only a whole category can be excluded")
        for tags in (None, ("Basketball",), ("Politics · Fed",)):
            with pytest.raises(ValueError, match=message):
                _settings(exclude_categories=("Economics", bad), exclude_tags=tags)
            with pytest.raises(ValueError, match=message):
                _settings(exclude_categories=(bad,), exclude_tags=tags)
        # ... and when a field is replaced, as main.py lays --exclude-category over saved
        # defaults
        with pytest.raises(ValueError, match=message):
            dataclasses.replace(_settings(exclude_categories=("Economics",)),
                                exclude_categories=("Economics", bad))

    @pytest.mark.parametrize("name", ["A·B", "·Sports", "Sports·", "U.S. - Politics", "Oil & Gas"])
    def test_a_category_may_hold_a_dot_that_ties_nothing(self, name):
        s = _settings(exclude_categories=(name,))
        assert s.exclude_categories == (name,)
        assert not config.trade_filter(s)(name.upper(), "anything")
        assert config.trade_filter(s)("Sports", "anything")

    def test_a_saved_file_with_a_category_written_as_a_tied_tag_is_refused(self):
        # A file edited by hand, with no tags at all
        record = _with_toggles(exclude_categories=["Economics", "Sports · Basketball"],
                               exclude_tags=None)
        config.LIVE_DEFAULTS_FILE.write_bytes(_text(record))
        for read in (config.read_saved_live_defaults, config.live_defaults):
            with pytest.raises(config.LiveDefaultsError,
                               match="'Sports · Basketball' is a tag tied to a category"):
                read()

    def test_a_save_round_trips_tied_tags(self):
        settings = _settings(exclude_categories=("Economics", "Politics"),
                             exclude_tags=("Sports · Basketball", "Sports · Pro Football"))
        saved = config.save_live_defaults(settings, source="")
        assert saved == settings and saved.exclude_tags == settings.exclude_tags
        again = config.read_saved_live_defaults()
        assert again == settings
        # The file still holds a plain list of names, the tie written as text
        stored = json.loads(config.LIVE_DEFAULTS_FILE.read_text(encoding="utf-8"))["settings"]
        assert stored["exclude_categories"] == ["Economics", "Politics"]
        assert stored["exclude_tags"] == ["Sports · Basketball", "Sports · Pro Football"]
        # The read-back filter is the same test
        keeps = config.trade_filter(again)
        assert [pair for pair in _FILED if keeps(*pair)] == [("Sports", "Soccer")]

    def test_a_saved_file_with_a_tied_tag_in_an_excluded_category_is_read(self):
        # Redundant, so allowed in a file too
        record = _with_toggles(exclude_categories=["Sports"],
                               exclude_tags=["Sports · Basketball"])
        config.LIVE_DEFAULTS_FILE.write_bytes(_text(record))
        saved = config.read_saved_live_defaults()
        assert saved.exclude_categories == ("Sports",)
        assert saved.exclude_tags == ("Sports · Basketball",)
        assert config.live_defaults() == saved

    def test_the_argv_spells_a_tied_tag_as_one_flag(self):
        settings = _settings(exclude_categories=("Economics", "Politics"),
                             exclude_tags=("Sports · Basketball", "Fed"))
        assert config.live_settings_argv(settings)[-4:] == [
            "--exclude-category=Economics", "--exclude-category=Politics",
            "--exclude-tag=Sports · Basketball", "--exclude-tag=Fed"]
        assert config.describe_trade_filter(settings) == (
            "exclude categories Economics, Politics; exclude tags Sports · Basketball, Fed")
        assert config.live_settings_argv(_settings())[-2:] == [
            "--no-exclude-category", "--no-exclude-tag"]
        assert config.describe_trade_filter(_settings()) == (
            "exclude categories none; exclude tags none")


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


class TestTakeProfitHoldDays:
    """TAKE_PROFIT_HOLD_DAYS: how many days in a row the backtest's sell rule
    needs a position at its level before selling it (backtest-only).

    Shipped at 3: the checkpoint and the two days before it. It must be a
    whole number from 1 to 7 (backtester._resolve_hold_days refuses anything
    else), so every daily check falls within the week since the previous
    checkpoint.
    """

    def test_three_days_ship(self):
        assert config.TAKE_PROFIT_HOLD_DAYS == 3
        assert type(config.TAKE_PROFIT_HOLD_DAYS) is int


class TestTakeProfitLevelsAndMinDays:
    """TAKE_PROFIT_LEVELS and TAKE_PROFIT_MIN_DAYS: the backtest dashboard's
    sell levels and its minimum-days options (backtest-only).

    The levels are every 1% from 80% to 100%, each the float nearest its
    percent over 100, so a level prints as its percent (0.81 reads "81") and
    equals the literal a test or a reader would write. The day options are
    1 to 7, 14 and 21, ascending whole numbers, none repeated, which
    backtester.run_backtest_sweep checks before its fetch.
    """

    def test_twenty_one_levels_from_80_to_100_percent_ship(self):
        assert len(config.TAKE_PROFIT_LEVELS) == 21
        assert config.TAKE_PROFIT_LEVELS[0] == 0.80 and config.TAKE_PROFIT_LEVELS[-1] == 1.0
        assert config.TAKE_PROFIT_LEVELS == tuple(sorted(set(config.TAKE_PROFIT_LEVELS)))
        for percent, level in zip(range(80, 101), config.TAKE_PROFIT_LEVELS, strict=True):
            assert level == float(f"0.{percent}" if percent < 100 else "1.0")
            assert type(level) is float

    def test_nine_day_options_ship(self):
        assert config.TAKE_PROFIT_MIN_DAYS == (1, 2, 3, 4, 5, 6, 7, 14, 21)
        assert all(type(days) is int for days in config.TAKE_PROFIT_MIN_DAYS)


class TestCandleNoAskCeiling:
    """CANDLE_NO_ASK_CEILING: the highest NO ask a candle can carry, which is
    no quote. historical clamps to it and refuses it; the backtester's alias
    is the same object."""

    def test_one_value_everywhere(self):
        from kalshi_betting import historical

        assert config.CANDLE_NO_ASK_CEILING == 0.99
        assert historical.CANDLE_NO_ASK_CEILING is config.CANDLE_NO_ASK_CEILING
        assert backtester._CANDLE_NO_ASK_CEILING is config.CANDLE_NO_ASK_CEILING


class TestTakeProfitReached:
    """take_profit_reached: the sell rule's one test at a check — a position
    with a potential profit sells once its realized profit reaches sell_at of
    it, less PRICE_EPSILON of float noise. backtester._sells_at decides with
    it, so it is the backtest's rule exactly."""

    def test_at_and_around_the_level(self):
        # 80% of a $10 potential profit is $8
        assert config.take_profit_reached(0.8, 8.0, 10.0) is True
        assert config.take_profit_reached(0.8, 9.5, 10.0) is True
        # Float noise below the level still reaches it ...
        assert config.take_profit_reached(0.8, 8.0 - PRICE_EPSILON, 10.0) is True
        assert config.take_profit_reached(0.8, 8.0 - PRICE_EPSILON / 2, 10.0) is True
        # ... a real shortfall does not
        assert config.take_profit_reached(0.8, 8.0 - 2 * PRICE_EPSILON, 10.0) is False
        assert config.take_profit_reached(0.8, 7.99, 10.0) is False
        # 0.1 + 0.2 is 0.30000000000000004 and 0.3 is a hair below it: both reach 0.3
        assert config.take_profit_reached(0.3, 0.1 + 0.2, 1.0) is True
        assert config.take_profit_reached(1.0, 0.3, 0.1 + 0.2) is True

    @pytest.mark.parametrize("potential", [0.0, -0.0, -0.01, -5.0])
    def test_no_potential_profit_never_sells(self, potential):
        assert config.take_profit_reached(0.8, 100.0, potential) is False
        assert config.take_profit_reached(0.01, 0.0, potential) is False

    def test_it_is_the_backtests_test(self):
        rng = random.Random(1207)
        for _ in range(2000):
            level = rng.choice(config.TAKE_PROFIT_LEVELS)
            potential = rng.uniform(-1.0, 20.0)
            realized = level * potential + rng.choice([-1, 1]) * rng.choice(
                [0.0, PRICE_EPSILON / 2, PRICE_EPSILON, 2 * PRICE_EPSILON, rng.uniform(0, 1)])
            assert (config.take_profit_reached(level, realized, potential)
                    is backtester._sells_at(level, realized, potential))
            assert config.take_profit_reached(level, realized, potential) is (
                potential > 0 and realized >= level * potential - PRICE_EPSILON)

    def test_a_level_reached_is_reached_at_every_lower_level(self):
        rng = random.Random(1208)
        levels = config.TAKE_PROFIT_LEVELS
        for _ in range(500):
            potential = rng.uniform(0.01, 20.0)
            realized = rng.uniform(0.7, 1.05) * potential
            reached = [config.take_profit_reached(level, realized, potential) for level in levels]
            # True for a prefix of the ascending levels, then False
            assert reached == sorted(reached, reverse=True)


class TestReachedEveryCheck:
    """reached_every_check: take_profit_reached at every daily check."""

    def test_every_check_must_reach_the_level(self):
        assert config.reached_every_check(0.8, [(8.0, 10.0), (9.0, 10.0), (8.5, 10.0)]) is True
        assert config.reached_every_check(0.8, [(8.0, 10.0), (7.0, 10.0), (8.5, 10.0)]) is False
        # One check with no potential profit holds the sale back
        assert config.reached_every_check(0.8, [(8.0, 10.0), (1.0, 0.0)]) is False

    @pytest.mark.parametrize("checks", [[], (), iter([])])
    def test_no_checks_never_sells(self, checks):
        # A position that was never valued has shown nothing: fail closed,
        # where all() of nothing would read as a sale
        assert config.reached_every_check(0.8, checks) is False
        assert config.reached_every_check(0.01, checks) is False

    def test_any_iterable_of_checks(self):
        checks = [(8.0, 10.0), (9.0, 10.0)]
        assert config.reached_every_check(0.8, iter(checks)) is True
        assert config.reached_every_check(0.8, tuple(checks)) is True

    def test_it_is_the_backtests_rule(self):
        rng = random.Random(1209)
        for _ in range(500):
            level = rng.choice(config.TAKE_PROFIT_LEVELS)
            checks = [(rng.uniform(-2.0, 10.0), rng.uniform(-1.0, 10.0))
                      for _ in range(rng.randint(1, 4))]
            assert (config.reached_every_check(level, checks)
                    is backtester._reached_every_day(level, checks))


class TestDaysToMaturity:
    """days_to_maturity: whole days from a date to a position's last close
    date — what the sell rule's minimum of days reads. Unknown when any close
    date is (a position whose maturity cannot be read is never sold under a
    minimum)."""

    def test_the_latest_close_date_decides(self):
        day = date(2026, 1, 12)
        assert config.days_to_maturity([date(2026, 1, 20), date(2026, 2, 2)], day) == 21
        assert config.days_to_maturity([date(2026, 2, 2), date(2026, 1, 20)], day) == 21
        assert config.days_to_maturity([date(2026, 1, 13)], day) == 1

    def test_zero_on_the_closing_date_and_negative_after_it(self):
        day = date(2026, 1, 12)
        assert config.days_to_maturity([date(2026, 1, 12), date(2026, 1, 5)], day) == 0
        assert config.days_to_maturity([date(2026, 1, 10), date(2026, 1, 5)], day) == -2

    @pytest.mark.parametrize("closes", [[], [None], [date(2026, 2, 2), None],
                                        [None, date(2026, 2, 2)]])
    def test_unknown_when_any_date_is_or_there_are_none(self, closes):
        assert config.days_to_maturity(closes, date(2026, 1, 12)) is None

    def test_any_iterable_of_dates(self):
        closes = (d for d in [date(2026, 1, 20), date(2026, 2, 2)])
        assert config.days_to_maturity(closes, date(2026, 1, 12)) == 21


class TestDaysLeftReached:
    """days_left_reached: the "Rebalance until" test. Always with no minimum,
    never with an unknown number of days, otherwise at least the minimum."""

    @pytest.mark.parametrize("left", [None, -2, 0, 1, 6, 7, 8, 400])
    def test_no_minimum_always_allows(self, left):
        assert config.days_left_reached(left, None) is True

    @pytest.mark.parametrize("min_days", [1, 7])
    def test_an_unknown_number_of_days_never_allows(self, min_days):
        assert config.days_left_reached(None, min_days) is False

    @pytest.mark.parametrize("left, at_one, at_seven", [
        (-2, False, False), (0, False, False), (1, True, False), (6, True, False),
        # 7 days left still rebalances at 7; 6 does not
        (7, True, True), (8, True, True)])
    def test_at_least_the_minimum(self, left, at_one, at_seven):
        assert config.days_left_reached(left, 1) is at_one
        assert config.days_left_reached(left, 7) is at_seven

    def test_it_reads_days_to_maturity_on_utc_dates(self):
        # A run at 16:00 UTC and a close at 03:59 UTC the next day: one UTC date
        # apart, so 1 day left, though under 12 hours
        left = config.days_to_maturity([date(2026, 1, 13)], date(2026, 1, 12))
        assert left == 1
        assert config.days_left_reached(left, 1) and not config.days_left_reached(left, 7)

    def test_a_datetime_is_refused(self):
        # Two datetimes differ by whole 24-hour spans: a close at 08:00 three
        # calendar days after a 16:00 check would read 2 days, not 3. A
        # datetime is a date by subclass, so it is refused by name
        close = datetime(2026, 1, 15, 8, 0, tzinfo=UTC)
        now = datetime(2026, 1, 12, 16, 0, tzinfo=UTC)
        assert config.days_to_maturity([close.date()], now.date()) == 3
        for closes, day in (([close], now), ([close], now.date()),
                            ([close.date()], now), ([date(2026, 2, 2), close], now.date()),
                            ([None, close], now.date())):
            with pytest.raises(TypeError, match="calendar dates"):
                config.days_to_maturity(closes, day)


class TestHeldPairWinProb:
    """config.held_pair_win_prob: the chance a held pair pays, at today's quotes.

    It is the chance a new trade of the pair would be sized with, so a held
    pair and a new one are never judged by two models.
    """

    def test_a_same_title_pair_has_the_fixed_chance(self):
        assert config.held_pair_win_prob("same_title", 0.5, 0.5, 0.5, 0.5, 0.8) \
            == config.SAME_TITLE_CO_RESOLVE_PROB
        # Anything but the exact string "time_series" is same-title, whatever the quotes
        assert config.held_pair_win_prob(None, float("nan"), 0, 0, 0, 0.8) \
            == config.SAME_TITLE_CO_RESOLVE_PROB

    def test_a_time_series_pair_reads_the_mid_spread(self):
        quotes = (0.22, 0.80, 0.63, 0.40)
        expected = config.time_series_profit_prob(config.time_series_mid_spread(*quotes), 0.8)
        assert config.held_pair_win_prob("time_series", *quotes, 0.8) == expected
        # mid A = (0.22 + 0.20) / 2 = 0.21, mid B = (0.63 + 0.60) / 2 = 0.615
        assert expected == pytest.approx(1 - 0.8 * (0.615 - 0.21), abs=1e-12)

    @pytest.mark.parametrize("quotes", [(0.30, 0.65, 0.60, 0.42), (0.30, 0.72, 0.60, 0.35)])
    def test_a_crossed_book_on_either_market_has_no_chance_to_read(self, quotes):
        # YES ask + NO ask under 1: the YES ask sits below the market's own YES bid
        assert config.held_pair_win_prob("time_series", *quotes, 0.8) is None

    @pytest.mark.parametrize("quotes", [(0.60, 0.45, 0.30, 0.75), (0.30, 0.72, 0.30, 0.72)])
    def test_a_mid_spread_not_above_zero_has_no_chance_to_read(self, quotes):
        # The model would read it as a sure win; a new trade is refused here too
        assert config.time_series_mid_spread(*quotes) <= 0
        assert config.held_pair_win_prob("time_series", *quotes, 0.8) is None

    @pytest.mark.parametrize("k", [float("nan"), float("inf")])
    def test_a_k_that_is_not_a_number_has_no_chance_to_read(self, k):
        assert config.held_pair_win_prob("time_series", 0.22, 0.80, 0.63, 0.40, k) is None

    def test_a_book_one_float_step_short_of_uncrossed_is_read(self):
        quotes = (0.3, 1.0 - 0.3 - 1e-12, 0.6, 0.4)
        assert config.held_pair_win_prob("time_series", *quotes, 0.8) is not None

    @pytest.mark.parametrize("place", range(4))
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_a_quote_that_is_not_a_number_has_no_chance_to_read(self, place, bad):
        quotes = [0.22, 0.80, 0.63, 0.40]
        quotes[place] = bad
        # Without the check a NaN quote reads as a mid spread of zero: a sure win
        assert config.held_pair_win_prob("time_series", *quotes, 0.8) is None


class TestSettledBids:
    """config.settled_bids: the bids a paid-out market reads as, its settlement price.

    A Kelly trim of a lone leg reads the paid-out earlier market this way, so the
    pair arithmetic (held_pair_win_prob, the sale value) runs unchanged, live and
    in the backtest alike.
    """

    def test_each_result_reads_as_its_settlement_price(self):
        assert config.settled_bids("yes") == (1.0, 0.0)
        assert config.settled_bids("no") == (0.0, 1.0)

    @pytest.mark.parametrize("result", ["void", "", "Yes", None, 1])
    def test_any_other_result_is_refused(self, result):
        with pytest.raises(ValueError, match="'yes' or 'no'"):
            config.settled_bids(result)

    def test_a_lone_later_leg_after_the_earlier_market_paid_nothing(self):
        # YES was held on the earlier market A and NO is still held on the later
        # market B. A settled "no", so its asks read 0 (YES) and 1 (NO): the
        # chance is 1 - k x B's midpoint. B: YES ask 0.14, NO ask 0.88, so its
        # YES bid is 0.12 and its midpoint 0.13; at k 0.7 that is 0.909
        yes_bid_a, no_bid_a = config.settled_bids("no")
        chance = config.held_pair_win_prob("time_series", 1.0 - no_bid_a, 1.0 - yes_bid_a,
                                           0.14, 0.88, 0.7)
        assert chance == pytest.approx(1 - 0.7 * 0.13, abs=1e-12)

    def test_the_later_leg_cannot_win_once_the_earlier_market_paid_out_yes(self):
        # The event happened by A's deadline, so B must settle "yes" too and the
        # NO held on B cannot win: no chance to read, so nothing is trimmed
        yes_bid_a, no_bid_a = config.settled_bids("yes")
        assert config.held_pair_win_prob("time_series", 1.0 - no_bid_a, 1.0 - yes_bid_a,
                                         0.14, 0.88, 0.7) is None


class TestKellyHoldFraction:
    """config.kelly_hold_fraction: the share of the portfolio a held pair may keep.

    Keeping a pair that sells for s is the bet of buying it at s, so the
    share is Kelly's for that bet: (p - s) / (1 - s) at a $1 payout.
    """

    def test_it_is_kellys_share_for_a_bet_bought_at_the_sale_value(self):
        # p - q / b with b = (1 - s) / s: 0.9 - 0.1 * 0.6 / 0.4
        assert config.kelly_hold_fraction(0.9, 0.6) == pytest.approx(0.9 - 0.1 * 0.6 / 0.4)
        assert config.kelly_hold_fraction(0.9, 0.6) == pytest.approx(0.75)

    def test_a_better_sale_price_lets_the_pair_keep_less(self):
        shares = [config.kelly_hold_fraction(0.9, s / 100) for s in range(1, 100)]
        assert shares == sorted(shares, reverse=True)
        assert all(0.0 <= share <= 1.0 for share in shares)

    @pytest.mark.parametrize("sale", [0.9, 0.95, 1.0, 1.5])
    def test_nothing_is_kept_once_the_sale_pays_what_holding_is_expected_to(self, sale):
        assert config.kelly_hold_fraction(0.9, sale) == 0.0

    @pytest.mark.parametrize("win, sale", [(1.5, 1.0), (3.0, 2.0), (1.2, 1.1)])
    def test_a_sale_at_or_above_the_payout_keeps_nothing_whatever_the_chance(self, win, sale):
        # Never a division by zero or a share below 0, even for a chance above 1
        assert config.kelly_hold_fraction(win, sale) == 0.0

    @pytest.mark.parametrize("sale", [0.0, -0.01])
    def test_everything_is_kept_when_the_sale_returns_nothing(self, sale):
        assert config.kelly_hold_fraction(0.9, sale) == 1.0

    @pytest.mark.parametrize("win, sale", [(float("nan"), 0.5), (0.9, float("nan")),
                                           (float("inf"), 0.5), (0.9, float("inf"))])
    def test_everything_is_kept_when_a_number_cannot_be_read(self, win, sale):
        assert config.kelly_hold_fraction(win, sale) == 1.0

    def test_it_sits_above_the_share_a_new_trade_would_buy(self):
        # Selling returns less than buying costs, so the size a run sells down
        # to is above the size it buys up to: it never does both at one price
        p, bid, ask = 0.9, 0.58, 0.62
        buy = (p - ask) / (1 - ask)
        assert config.kelly_hold_fraction(p, bid) > buy


def _flat_sale(price: float, depth: int | None = None):
    """A net_sale that sells any count at one net price, up to `depth` pairs."""
    def net_sale(pairs: int) -> float | None:
        if depth is not None and pairs > depth:
            return None
        return pairs * price
    return net_sale


class TestKellyTrimCount:
    """config.kelly_trim_count: how many contract pairs of a held pair to sell.

    The pair keeps at most min(cap, kelly_hold_fraction) of the portfolio
    value, counted at what selling returns; a worse sale price only ever
    lowers the count.
    """

    def test_a_pair_within_its_share_sells_nothing(self):
        # 100 pairs worth $0.60 each are 6% of $1,000, under the 10% cap
        trim = config.kelly_trim_count(100, 1_000.0, 0.9, 0.10, 0.60, _flat_sale(0.60))
        assert trim.sell == 0

    def test_a_pair_past_the_cap_is_sold_down_to_it(self):
        # 300 pairs at $0.60 are 18% of $1,000; 10% keeps floor(100 / 0.60) = 166
        trim = config.kelly_trim_count(300, 1_000.0, 0.9, 0.10, 0.60, _flat_sale(0.60))
        assert trim.sell == 300 - 166
        assert (300 - trim.sell) * 0.60 <= 0.10 * 1_000.0 < (300 - trim.sell + 1) * 0.60

    def test_kelly_binds_when_it_is_below_the_cap(self):
        # p 0.70, sale value 0.60: Kelly's share is 0.10 / 0.40 = 25%, under a 100% cap
        trim = config.kelly_trim_count(1_000, 1_000.0, 0.70, 1.0, 0.60, _flat_sale(0.60))
        assert trim.sell == 1_000 - math.floor(0.25 * 1_000.0 / 0.60 + 1e-9)

    def test_a_pair_with_no_edge_left_is_sold_whole(self):
        # Selling pays 0.96 a pair against a 0.95 chance of $1
        trim = config.kelly_trim_count(40, 1_000.0, 0.95, 1.0, 0.96, _flat_sale(0.96))
        assert trim == config.KellyTrim(40, 0.0)

    def test_a_worse_real_price_sells_fewer(self):
        # At the best bids (0.60) the cap keeps 166 of 300. The sale really
        # nets 0.50 a pair, where 10% of $1,000 keeps 200
        trim = config.kelly_trim_count(300, 1_000.0, 0.9, 0.10, 0.60, _flat_sale(0.50))
        assert trim.sell == 100

    def test_a_much_worse_real_price_sells_nothing(self):
        # At 0.30 a pair the 300 pairs are 9% of the portfolio: within the cap
        trim = config.kelly_trim_count(300, 1_000.0, 0.9, 0.10, 0.60, _flat_sale(0.30))
        assert trim.sell == 0

    def test_a_steep_book_sells_the_most_that_is_wanted_at_its_own_price(self):
        # 50 pairs net 0.60 each, every one after them 0.20. At the best bids
        # the cap wants 134 sold, but 134 would average 0.349 a pair, where
        # the 300 are within the cap. 79 is the most still wanted at its own
        # price (0.453 a pair keeps 220, so wants 80 sold); 80 is not
        def net_sale(pairs):
            return min(pairs, 50) * 0.60 + max(0, pairs - 50) * 0.20
        trim = config.kelly_trim_count(300, 1_000.0, 0.9, 0.10, 0.60, net_sale)
        assert trim.sell == 79
        for pairs, wanted in ((79, True), (80, False)):
            price = net_sale(pairs) / pairs
            keep = math.floor(0.10 * 1_000.0 / price)
            assert (300 - keep >= pairs) is wanted

    def test_a_small_excess_is_sold_though_a_smaller_sale_would_not_be_wanted(self):
        # Two legs at 0.24 and 0.20, each sale's fee rounded up to the cent
        # per leg. The cap wants 20 sold at the best bids. A sale of a few
        # pairs nets too little per pair to be wanted (the two rounded fees
        # weigh on it), so halving from 20 ends on none; 16 is wanted
        def net_sale(pairs):
            return sum(pairs * bid - config.fee_leg_exact(pairs, bid) for bid in (0.24, 0.20))
        top = 0.44 - config.fee_per_pair_approx(0.24, 0.20)
        count, value, cap = 4396, 18_207.29, 0.10
        trim = config.kelly_trim_count(count, value, 0.9, cap, top, net_sale)

        def wanted_at_its_own_price(pairs):
            price = net_sale(pairs) / pairs
            return count - math.floor(cap * value / price) >= pairs
        assert trim.sell > 0 and wanted_at_its_own_price(trim.sell)
        assert not wanted_at_its_own_price(1)
        # ... and it is the largest such count
        assert not any(wanted_at_its_own_price(pairs) for pairs in range(trim.sell + 1, 41))

    def test_a_better_real_price_never_sells_more_than_the_best_bids_asked_for(self):
        trim = config.kelly_trim_count(300, 1_000.0, 0.9, 0.10, 0.60, _flat_sale(0.70))
        assert trim.sell == 300 - 166

    def test_thin_bids_sell_the_most_they_hold(self):
        asked = []

        def net_sale(pairs):
            asked.append(pairs)
            return None if pairs > 37 else pairs * 0.60
        trim = config.kelly_trim_count(300, 1_000.0, 0.9, 0.10, 0.60, net_sale)
        assert trim.sell == 37
        # Found by halving, then a fixed number of counts above: never one
        # count at a time over the whole range
        assert len(asked) <= 12 + config._TRIM_COUNT_SCAN

    def test_bids_that_hold_nothing_sell_nothing(self):
        trim = config.kelly_trim_count(300, 1_000.0, 0.9, 0.10, 0.60, _flat_sale(0.60, depth=0))
        assert trim.sell == 0

    @pytest.mark.parametrize("cap", [0.0, -0.1, float("nan")])
    def test_a_cap_that_is_not_above_zero_sells_nothing(self, cap):
        assert config.kelly_trim_count(300, 1_000.0, 0.9, cap, 0.60, _flat_sale(0.60)) \
            == config.KellyTrim(0, 0.0)

    def test_a_large_position_keeps_a_whole_count_float_division_left_short(self):
        # The cap is exactly the position's share, but cap x value / price
        # comes back as 6116611.999999998: nothing is over the cap
        price = 0.01449163128142346
        trim = config.kelly_trim_count(6116612, 134602.51498764087, 1.0, 0.6585291946712062,
                                       price, lambda pairs: pairs * price)
        assert trim.sell == 0

    @pytest.mark.parametrize("proceeds", [0.0, -1.0, float("nan"), float("inf")])
    def test_a_sale_that_returns_nothing_sells_nothing(self, proceeds):
        trim = config.kelly_trim_count(300, 1_000.0, 0.9, 0.10, 0.60, lambda pairs: proceeds)
        assert trim.sell == 0

    @pytest.mark.parametrize("count, value, win, top", [
        (0, 1_000.0, 0.9, 0.6), (-5, 1_000.0, 0.9, 0.6), (300, 0.0, 0.9, 0.6),
        (300, float("nan"), 0.9, 0.6), (300, float("inf"), 0.9, 0.6),
        (300, 1_000.0, float("nan"), 0.6), (300, 1_000.0, 0.9, 0.0),
        (300, 1_000.0, 0.9, float("nan")), (300, 1_000.0, 0.9, -0.2)])
    def test_inputs_that_cannot_be_read_sell_nothing(self, count, value, win, top):
        def never(pairs):
            raise AssertionError("no sale should be priced")
        assert config.kelly_trim_count(count, value, win, 0.10, top, never) \
            == config.KellyTrim(0, 0.0)

    def test_a_whole_count_a_float_hair_short_is_kept_whole(self):
        # 0.1 * 3 / 0.1 is 3.0000000000000004 or 2.9999999999999996 by turns
        # in floats; three pairs at $0.10 are exactly 10% of $3
        trim = config.kelly_trim_count(3, 3.0, 0.9, 0.10, 0.1, _flat_sale(0.1))
        assert trim.sell == 0

    def test_the_cap_it_names_is_where_the_answer_stops_depending_on_the_cap(self):
        """At every cap at or above cap_free_from the decision is the uncapped one.

        The backtest's size-cap sweep shares one simulation across those
        caps, so a cap_free_from that is too low would show one cap's trades
        under another's name. Seeded cases over prices, sizes, thin and deep
        bids and a real price better or worse than the best bids.
        """
        rng = random.Random(20261010)
        caps = [round(0.05 * i, 2) for i in range(1, 21)]
        trimmed = bound = 0
        for _ in range(2_000):
            count = rng.randint(1, 3_000)
            value = rng.uniform(50.0, 5_000.0)
            win = rng.uniform(0.3, 1.0)
            top = rng.uniform(0.02, 0.99)
            real = top * rng.uniform(0.4, 1.1)
            depth = rng.choice([None, None, rng.randint(0, 3_000)])
            # A ladder: the deeper the sale, the lower its average price
            slope = rng.choice([0.0, real / 6_000.0])

            def net_sale(pairs, real=real, depth=depth, slope=slope):
                if depth is not None and pairs > depth:
                    return None
                return pairs * (real - slope * pairs)
            free = config.kelly_trim_count(count, value, win, 1.0, top, net_sale)
            trimmed += free.sell > 0
            for cap in caps:
                capped = config.kelly_trim_count(count, value, win, cap, top, net_sale)
                if cap >= capped.cap_free_from:
                    assert capped.sell == free.sell, (count, value, win, top, real, depth, cap)
                else:
                    bound += capped.sell != free.sell
        # The cases cover both sides: pairs that are trimmed, and caps that bind
        assert trimmed > 200 and bound > 200


class TestKellyTrimBids:
    """config.kelly_trim_bids: which bids of a held side a live Kelly trim may
    sell into (those within KELLY_TRIM_MAX_BID_DROP, 20c, of the best bid)."""

    def test_the_reach_is_the_backtests_modeled_ladders(self):
        # Live reaches as deep as the backtest's modeled bid ladder, whose last
        # step is the last of DEPTH_MODEL_DISTANCES
        assert config.KELLY_TRIM_MAX_BID_DROP == config.DEPTH_MODEL_DISTANCES[-1] == 0.20
        assert max(config.DEPTH_MODEL_DISTANCES) == config.DEPTH_MODEL_DISTANCES[-1]

    def test_bids_further_below_the_best_bid_are_left_out(self):
        ladder = [[0.42, 1], [0.22, 2], [0.21, 3]]
        assert config.kelly_trim_bids(ladder, 0.20) == [[0.42, 1], [0.22, 2]]
        # A gap: one contract at the best bid, then a stub far below
        assert config.kelly_trim_bids([[0.42, 1.0], [0.02, 100_000.0]], 0.20) == [[0.42, 1.0]]
        # Nothing below the reach: the whole ladder
        assert config.kelly_trim_bids([[0.52, 1], [0.45, 2], [0.32, 2]], 0.20) == [
            [0.52, 1], [0.45, 2], [0.32, 2]]

    def test_it_hands_back_the_very_levels_in_a_new_list(self):
        ladder = [[0.42, 1], [0.30, 2], [0.10, 3]]
        kept = config.kelly_trim_bids(ladder, 0.20)
        assert kept is not ladder and len(ladder) == 3
        assert kept[0] is ladder[0] and kept[1] is ladder[1]

    def test_a_bid_exactly_the_reach_below_is_kept_on_every_whole_cent(self):
        # Float subtraction lands a hair either side of the cent; a bid exactly
        # 20c under the best is kept either way, one a cent lower is not
        assert 0.55 - 0.20 > 0.35 and 0.42 - 0.20 < 0.22
        for cents in range(22, 100):
            best, at, under = cents / 100, (cents - 20) / 100, (cents - 21) / 100
            assert config.kelly_trim_bids([[best, 1], [at, 2], [under, 3]], 0.20) == [
                [best, 1], [at, 2]], cents

    def test_a_tick_past_the_reach_is_left_out_on_the_finest_grid(self):
        # The float allowance is far under Kalshi's finest step ($0.0001)
        assert config.kelly_trim_bids([[0.42, 1], [0.2199, 2]], 0.20) == [[0.42, 1]]
        assert config.kelly_trim_bids([[0.9999, 1], [0.7999, 2], [0.7998, 3]], 0.20) == [
            [0.9999, 1], [0.7999, 2]]

    def test_no_reach_keeps_only_the_best_bids_own_price(self):
        assert config.kelly_trim_bids([[0.42, 1], [0.42, 4], [0.41, 2]], 0) == [
            [0.42, 1], [0.42, 4]]

    def test_the_best_bid_is_always_kept_and_an_empty_ladder_is_empty(self):
        assert config.kelly_trim_bids([[0.05, 7]], 0.20) == [[0.05, 7]]
        assert config.kelly_trim_bids([], 0.20) == []
        assert config.kelly_trim_bids((), 0.20) == []

    @pytest.mark.parametrize("drop", [None, True, False, "0.2", -0.01, float("nan"),
                                      float("inf")])
    def test_a_bad_reach_is_refused(self, drop):
        with pytest.raises(ValueError, match="KELLY_TRIM_MAX_BID_DROP"):
            config.kelly_trim_bids([[0.42, 1]], drop)
        # ... even when there is nothing to cut
        with pytest.raises(ValueError, match="KELLY_TRIM_MAX_BID_DROP"):
            config.kelly_trim_bids([], drop)


class TestMinPriceMoves:
    """config.MIN_PRICE_MOVES: the minimum price moves, up or down, the
    dashboard's Min. price movement select offers, every 5c from 5c to 95c."""

    def test_every_five_cents_from_5c_to_95c(self):
        rises = config.MIN_PRICE_MOVES
        assert isinstance(rises, tuple) and len(rises) == 19
        assert config.PRICE_MOVE_STEP_CENTS == 5
        # Each is the float nearest its cents over 100, so 15c is exactly 0.15
        assert list(rises) == [cents / 100 for cents in range(5, 100, 5)]
        assert rises[0] == 0.05 and rises[2] == 0.15 and rises[-1] == 0.95

    def test_they_are_distinct_ascending_shares_of_a_dollar(self):
        rises = config.MIN_PRICE_MOVES
        assert list(rises) == sorted(set(rises))
        assert all(0.0 < rise <= 1.0 for rise in rises)


class TestPairPriceMove:
    """config.pair_price_move: how far a held pair's total price has moved
    since the last trade on it, signed (above zero for a rise, below for a
    fall), the one place the sign is decided."""

    def test_a_rise_is_above_zero_and_a_fall_below(self):
        assert config.pair_price_move(0.62, 0.87) == pytest.approx(0.25)
        assert config.pair_price_move(0.87, 0.77) == pytest.approx(-0.10)
        assert config.pair_price_move(0.70, 0.70) == 0.0

    def test_it_is_the_price_now_less_the_price_then(self):
        rng = random.Random(20261013)
        for _ in range(200):
            then, now = rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0)
            assert config.pair_price_move(then, now) == now - then


class TestPriceMoveReached:
    """config.price_move_reached: whether a held pair's price has moved far
    enough, up or down, for a rebalance (an add-on or a Kelly trim). A rise
    and a fall of the same size count alike; a move that is not a number
    holds the rebalance back; a larger minimum never passes what a smaller
    one held back, which is what lets the backtest reuse one run for a
    stricter minimum."""

    def test_no_minimum_passes_anything(self):
        for move in (0.5, 0.0, -0.9, float("nan")):
            assert config.price_move_reached(move, None) is True

    def test_a_move_at_or_above_the_minimum_passes_either_way(self):
        assert config.price_move_reached(0.10, 0.10)
        assert config.price_move_reached(0.33, 0.30)
        assert config.price_move_reached(-0.10, 0.10)
        assert config.price_move_reached(-0.33, 0.30)
        assert not config.price_move_reached(0.08, 0.10)
        assert not config.price_move_reached(-0.08, 0.10)
        assert not config.price_move_reached(0.0, 0.05)

    def test_a_move_exactly_on_the_minimum_passes_through_float_noise(self):
        # 87c less 62c is the 25c minimum, whatever the floats make of it, up or down
        assert config.price_move_reached(config.pair_price_move(0.62, 0.87), 0.25)
        assert config.price_move_reached(config.pair_price_move(0.87, 0.62), 0.25)
        assert config.price_move_reached(config.pair_price_move(0.30, 0.35), 0.05)
        assert config.price_move_reached(config.pair_price_move(0.35, 0.30), 0.05)
        # One real cent short is short, either way
        assert not config.price_move_reached(config.pair_price_move(0.62, 0.86), 0.25)
        assert not config.price_move_reached(config.pair_price_move(0.86, 0.62), 0.25)

    def test_a_fall_of_the_minimum_passes_as_its_rise_does(self):
        for move in (0.01, 0.10, 0.50, 0.95):
            for minimum in config.MIN_PRICE_MOVES:
                assert (config.price_move_reached(-move, minimum)
                        is config.price_move_reached(move, minimum))
        # ... and a large fall passes every minimum on the grid
        assert all(config.price_move_reached(-0.95, m) for m in config.MIN_PRICE_MOVES)

    def test_a_move_or_a_minimum_that_is_not_a_number_holds_the_rebalance_back(self):
        assert not config.price_move_reached(float("nan"), 0.05)
        assert not config.price_move_reached(-float("nan"), 0.05)
        assert not config.price_move_reached(0.50, float("nan"))
        assert not config.price_move_reached(0.50, float("inf"))
        assert not config.price_move_reached(-0.50, float("inf"))

    def test_a_larger_minimum_never_passes_what_a_smaller_one_held_back(self):
        # What lets one run's result stand for a stricter minimum: the test
        # only ever gets harder to pass as the minimum grows, for a fall too
        rng = random.Random(20261013)
        minimums = config.MIN_PRICE_MOVES
        for _ in range(500):
            move = rng.uniform(-1.0, 1.0)
            passed = [config.price_move_reached(move, minimum) for minimum in minimums]
            # Once it fails it fails at every larger minimum
            assert passed == sorted(passed, reverse=True)

    def test_the_rule_reads_only_what_it_is_handed(self):
        # Plain arithmetic on its arguments, so a live seller can call it with
        # its own prices: no name from outside but math, PRICE_EPSILON and the
        # builtin abs
        source = pathlib.Path(config.__file__).read_text()
        tree = ast.parse(source)
        allowed = {"pair_price_move": set(),
                   "price_move_reached": {"math", "PRICE_EPSILON", "abs"}}
        found = 0
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in allowed:
                found += 1
                params = {arg.arg for arg in node.args.args}
                # The body alone: the signature's type names are not reads
                names = {sub.id for statement in node.body for sub in ast.walk(statement)
                         if isinstance(sub, ast.Name)}
                assert names - params <= allowed[node.name], node.name
        assert found == 2


# The message rebalance_min_days refuses a bad number of days with
_BAD_TRIM_DAYS = r"REBALANCE_MIN_DAYS_SINCE_TRADE must be a whole number from 0 to 30, got "


class TestRebalanceRested:
    """config.rebalance_rested and config.rebalance_min_days: the calendar
    days a held pair must go without a trade before a rebalance (a Kelly trim
    or an add-on, live and in the backtest; 7 as shipped).

    The rule compares two plain dates with a checked number of days and reads
    no clock or schedule, so it holds however often runs happen. A missing
    last trade is an error, never "rested".
    """

    def test_seven_days_ship(self):
        assert config.REBALANCE_MIN_DAYS_SINCE_TRADE == 7
        assert type(config.REBALANCE_MIN_DAYS_SINCE_TRADE) is int
        assert config.REBALANCE_MIN_DAYS_SINCE_TRADE_MAX == 30
        assert type(config.REBALANCE_MIN_DAYS_SINCE_TRADE_MAX) is int
        # The shipped wait passes its own check
        assert config.rebalance_min_days(config.REBALANCE_MIN_DAYS_SINCE_TRADE) == 7

    @pytest.mark.parametrize("gap", range(0, 9))
    def test_every_gap_from_0_to_8_days(self, gap):
        # Bought on a Tuesday: not rested the same day or on any of the next
        # six, rested from the next Tuesday on
        last = date(2026, 10, 6)
        assert last.weekday() == 1
        day = last + timedelta(days=gap)
        assert config.rebalance_rested(last, day, 7) is (gap >= 7)

    def test_the_gap_alone_decides_from_any_start_day(self):
        # Every start day of 2027-2028 (a leap day and clock changes among
        # them), gaps up to 32 days, several waits: only the calendar gap decides
        first = date(2027, 1, 1)
        assert date(2028, 2, 29) > first  # 2028 is a leap year
        for offset in range(731):
            last = first + timedelta(days=offset)
            for gap in range(33):
                day = last + timedelta(days=gap)
                for wait in (0, 1, 7, 8, 14, 15, 30):
                    assert config.rebalance_rested(last, day, wait) is (gap >= wait), (
                        last, day, wait)

    @pytest.mark.parametrize("wait", [0, 1, 7, 30])
    @pytest.mark.parametrize("ahead", [1, 2, 7, 8, 30, 365])
    def test_a_last_trade_after_the_day_never_rests(self, ahead, wait):
        # A last trade dated after the day never counts as rested, even with no wait
        day = date(2026, 10, 13)
        assert config.rebalance_rested(day + timedelta(days=ahead), day, wait) is False

    def test_no_wait_rests_from_the_day_of_the_trade(self):
        day = date(2026, 10, 13)
        assert config.rebalance_rested(day, day, 0) is True
        assert config.rebalance_rested(day, day, 1) is False
        assert config.rebalance_rested(day - timedelta(days=1), day, 1) is True

    def test_once_rested_a_pair_stays_rested_and_a_longer_wait_never_rests_sooner(self):
        last = date(2026, 10, 6)
        for wait in range(0, 31):
            rested = [config.rebalance_rested(last, last + timedelta(days=gap), wait)
                      for gap in range(0, 40)]
            # False for the first `wait` days, True from then on
            assert rested == [False] * wait + [True] * (40 - wait)

    def test_the_worked_example(self):
        # Bought on Tuesday the 6th: runs on the 7th through the 12th do not
        # trim it, a run on Tuesday the 13th may
        bought = date(2026, 10, 6)
        for run_day in range(7, 13):
            assert not config.rebalance_rested(bought, date(2026, 10, run_day), 7)
        assert config.rebalance_rested(bought, date(2026, 10, 13), 7)
        # Trimmed on the 13th: a second run that day and runs on the 14th
        # through the 19th leave it alone
        trimmed = date(2026, 10, 13)
        for run_day in range(13, 20):
            assert not config.rebalance_rested(trimmed, date(2026, 10, run_day), 7)
        assert config.rebalance_rested(trimmed, date(2026, 10, 20), 7)
        # An add-on on Friday the 9th instead moves the first day to Friday the 16th
        added = date(2026, 10, 9)
        assert not config.rebalance_rested(added, date(2026, 10, 13), 7)
        assert not config.rebalance_rested(added, date(2026, 10, 15), 7)
        assert config.rebalance_rested(added, date(2026, 10, 16), 7)

    def test_a_missing_last_trade_is_an_error_never_rested(self):
        day = date(2026, 10, 13)
        # Not with the shipped wait, and not with no wait either
        for wait in (7, 0):
            with pytest.raises(TypeError, match="two calendar dates"):
                config.rebalance_rested(None, day, wait)
            with pytest.raises(TypeError, match="two calendar dates"):
                config.rebalance_rested(day, None, wait)

    def test_a_datetime_is_refused(self):
        # Datetimes count 24-hour spans, not calendar days (11pm on the 6th to
        # 9am on the 13th is 6 spans but 7 days), so they are refused
        last = datetime(2026, 10, 6, 23, 0, tzinfo=UTC)
        now = datetime(2026, 10, 13, 9, 0, tzinfo=UTC)
        assert (now - last).days == 6
        assert config.rebalance_rested(last.date(), now.date(), 7) is True
        for last_trade, day in ((last, now), (last, now.date()), (last.date(), now),
                                (last.replace(tzinfo=None), now.date())):
            with pytest.raises(TypeError, match="two calendar dates"):
                config.rebalance_rested(last_trade, day, 7)

    def test_anything_but_a_plain_date_is_refused(self):
        class _Day(date):
            """A date by subclass, as a clock-freezing stand-in might be."""

        good = date(2026, 10, 13)
        for bad in ("2026-10-06", 20261006, 1791244800.0, (2026, 10, 6), _Day(2026, 10, 6),
                    MagicMock(), timedelta(days=7)):
            for last_trade, day in ((bad, good), (good, bad), (bad, bad)):
                with pytest.raises(TypeError, match="two calendar dates"):
                    config.rebalance_rested(last_trade, day, 7)

    @pytest.mark.parametrize("bad", [True, False, 7.0, 7.5, -1, 31, None, "7", float("nan")],
                             ids=repr)
    def test_a_wait_that_is_not_a_checked_whole_number_gets_no_answer(self, bad):
        # Checked on every call: a negative wait would rest a pair traded
        # tomorrow, and True would read as one day
        for last, day in ((date(2026, 10, 6), date(2026, 10, 13)),
                          (date(2026, 10, 13), date(2026, 10, 13)),
                          (date(2026, 10, 14), date(2026, 10, 13))):
            with pytest.raises(ValueError, match=_BAD_TRIM_DAYS):
                config.rebalance_rested(last, day, bad)

    @pytest.mark.parametrize("days", [0, 1, 7, 8, 14, 15, 30])
    def test_min_days_returns_a_whole_number_in_range_as_a_plain_int(self, days):
        checked = config.rebalance_min_days(days)
        assert checked == days and type(checked) is int

    @pytest.mark.parametrize("bad", [True, False, 7.0, 7.5, -1, 31, 1000, None, "7", b"7", [7],
                                     float("nan"), float("inf")], ids=repr)
    def test_min_days_refuses_anything_else(self, bad):
        with pytest.raises(ValueError, match=_BAD_TRIM_DAYS) as refused:
            config.rebalance_min_days(bad)
        # The message shows what was handed in
        assert str(refused.value).endswith(f"got {bad!r}")

    def test_min_days_reads_numpy_numbers_by_kind(self):
        import numpy as np

        # A numpy whole number is a whole number, and comes back a plain int
        for good in (np.int64(7), np.uint8(7), np.int8(7), np.int32(7)):
            checked = config.rebalance_min_days(good)
            assert checked == 7 and type(checked) is int
        for bad in (np.bool_(True), np.float64(7.0), np.int64(31), np.int64(-1)):
            with pytest.raises(ValueError, match=_BAD_TRIM_DAYS):
                config.rebalance_min_days(bad)

    @pytest.mark.parametrize("unit", [None, "ns", "us", "ms", "s", "m", "h", "D", "W", "M", "Y"],
                             ids=str)
    def test_min_days_refuses_a_numpy_duration(self, unit):
        import numpy as np

        # numpy counts a duration (timedelta64) as a whole number, but it is
        # not a plain number of days, so it is refused in any unit
        duration = np.timedelta64(7) if unit is None else np.timedelta64(7, unit)
        assert isinstance(duration, numbers.Integral)
        with pytest.raises(ValueError, match=_BAD_TRIM_DAYS) as refused:
            config.rebalance_min_days(duration)
        assert str(refused.value).endswith(f"got {duration!r}")

    def test_a_wait_written_as_a_month_is_not_read_as_one_day(self):
        import numpy as np

        # A duration gets no answer, so a month is never read as one day
        day = date(2026, 10, 13)
        for wait in (np.timedelta64(1, "M"), np.timedelta64(1, "Y"), np.timedelta64(7, "ns"),
                     np.timedelta64(1), np.timedelta64(1, "D")):
            for last in (day - timedelta(days=1), day - timedelta(days=7), day):
                with pytest.raises(ValueError, match=_BAD_TRIM_DAYS):
                    config.rebalance_rested(last, day, wait)

    def test_min_days_reads_a_whole_number_by_subclass_as_a_plain_int(self):
        class _Days(enum.IntEnum):
            """A whole number by subclass of int."""

            WEEK = 7

        checked = config.rebalance_min_days(_Days.WEEK)
        assert checked == 7 and type(checked) is int

    def test_the_maximum_is_read_when_the_check_runs(self, monkeypatch):
        monkeypatch.setattr(config, "REBALANCE_MIN_DAYS_SINCE_TRADE_MAX", 10)
        assert config.rebalance_min_days(10) == 10
        with pytest.raises(ValueError, match="whole number from 0 to 10, got 11"):
            config.rebalance_min_days(11)

    def test_the_rule_reads_only_what_it_is_handed(self):
        # Each function may read only the names listed: no clock, no weekday,
        # and no schedule but the one handed to trade_day, so the wait holds
        # however often runs happen
        tree = ast.parse(pathlib.Path(config.__file__).read_text())
        allowed = {
            "rebalance_min_days": {"isinstance", "bool", "numbers", "operator", "TypeError",
                                    "ValueError", "REBALANCE_MIN_DAYS_SINCE_TRADE_MAX"},
            "trade_day": {"isinstance", "datetime", "ScheduledRun", "TypeError"},
            "rebalance_rested": {"type", "date", "TypeError", "rebalance_min_days"},
        }
        # Attributes allowed: trade_day reads only its schedule's zone, and
        # rebalance_min_days only operator.index (which refuses a numpy duration)
        attributes = {("trade_day", "run"): {"zone"},
                      ("rebalance_min_days", "operator"): {"index"}}
        found = 0
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in allowed:
                found += 1
                # The body alone: the signature's type names are not reads
                inside = [sub for statement in node.body for sub in ast.walk(statement)]
                names = [sub for sub in inside if isinstance(sub, ast.Name)]
                # Its parameters and the names it assigns are its own
                own = ({arg.arg for arg in node.args.args}
                       | {sub.id for sub in names if isinstance(sub.ctx, ast.Store)})
                assert {sub.id for sub in names} - own <= allowed[node.name], node.name
                for (function, name), expected in attributes.items():
                    if function == node.name:
                        read = {sub.attr for sub in inside if isinstance(sub, ast.Attribute)
                                and isinstance(sub.value, ast.Name) and sub.value.id == name}
                        assert read == expected, (function, name)
        assert found == 3


_PACIFIC = ZoneInfo("America/Los_Angeles")


def _pacific_day_by_hand(moment: datetime) -> date:
    """
    Work out a 2026 moment's Pacific day by hand, as a check on trade_day.

    Daylight time (UTC-7) runs from 10:00 UTC on 2026-03-08 to 09:00 UTC on
    2026-11-01; standard time (UTC-8) holds otherwise.

    Args:
        moment (datetime): A UTC time in 2026.

    Returns:
        date: The day a Pacific wall clock shows at that moment.
    """
    daylight = (datetime(2026, 3, 8, 10, 0, tzinfo=UTC) <= moment
                < datetime(2026, 11, 1, 9, 0, tzinfo=UTC))
    return (moment.replace(tzinfo=None) - timedelta(hours=7 if daylight else 8)).date()


class TestTradeDay:
    """config.trade_day: the calendar day a moment falls on, in the zone of the
    bot's schedule (Pacific as shipped).

    The trim wait counts these days, so a late-evening Pacific trade counts on
    that evening's day, not the next UTC day. A time with no zone is refused.
    """

    def test_1159pm_and_1201am_pacific_are_two_days(self):
        before = datetime(2026, 10, 13, 23, 59, tzinfo=_PACIFIC)
        after = datetime(2026, 10, 14, 0, 1, tzinfo=_PACIFIC)
        assert config.trade_day(before, SCHEDULED_RUN) == date(2026, 10, 13)
        assert config.trade_day(after, SCHEDULED_RUN) == date(2026, 10, 14)
        # The same two moments as Kalshi stamps them, in UTC (one UTC date)
        before_utc, after_utc = before.astimezone(UTC), after.astimezone(UTC)
        assert before_utc.date() == after_utc.date() == date(2026, 10, 14)
        assert config.trade_day(before_utc, SCHEDULED_RUN) == date(2026, 10, 13)
        assert config.trade_day(after_utc, SCHEDULED_RUN) == date(2026, 10, 14)

    def test_459pm_and_501pm_pacific_are_one_day(self):
        # Under daylight time 5pm Pacific is midnight UTC, so the two straddle it
        before = datetime(2026, 10, 13, 16, 59, tzinfo=_PACIFIC)
        after = datetime(2026, 10, 13, 17, 1, tzinfo=_PACIFIC)
        before_utc, after_utc = before.astimezone(UTC), after.astimezone(UTC)
        assert (before_utc.date(), after_utc.date()) == (date(2026, 10, 13), date(2026, 10, 14))
        for moment in (before, after, before_utc, after_utc):
            assert config.trade_day(moment, SCHEDULED_RUN) == date(2026, 10, 13)
        # Under standard time midnight UTC is 4pm Pacific
        winter_before = datetime(2026, 12, 15, 23, 59, tzinfo=UTC)   # 3:59pm Pacific
        winter_after = datetime(2026, 12, 16, 0, 1, tzinfo=UTC)      # 4:01pm Pacific
        assert config.trade_day(winter_before, SCHEDULED_RUN) == date(2026, 12, 15)
        assert config.trade_day(winter_after, SCHEDULED_RUN) == date(2026, 12, 15)

    @pytest.mark.parametrize(
        ("moment", "expected"),
        [
            # Spring, 2026-03-08: Pacific clocks jump from 2:00am to 3:00am.
            # Midnight Pacific is 08:00 UTC before the change and 07:00 UTC after
            (datetime(2026, 3, 8, 7, 59, tzinfo=UTC), date(2026, 3, 7)),    # 11:59pm Saturday
            (datetime(2026, 3, 8, 8, 1, tzinfo=UTC), date(2026, 3, 8)),     # 12:01am Sunday
            (datetime(2026, 3, 8, 9, 59, tzinfo=UTC), date(2026, 3, 8)),    # 1:59am, before the jump
            (datetime(2026, 3, 8, 10, 1, tzinfo=UTC), date(2026, 3, 8)),    # 3:01am, after it
            (datetime(2026, 3, 9, 6, 59, tzinfo=UTC), date(2026, 3, 8)),    # 11:59pm Sunday
            (datetime(2026, 3, 9, 7, 1, tzinfo=UTC), date(2026, 3, 9)),     # 12:01am Monday
            (datetime(2026, 3, 9, 7, 30, tzinfo=UTC), date(2026, 3, 9)),    # 12:30am Monday ...
            (datetime(2026, 3, 8, 7, 30, tzinfo=UTC), date(2026, 3, 7)),    # ... 11:30pm a day before
            # Autumn, 2026-11-01: 2:00am goes back to 1:00am. Midnight Pacific
            # is 07:00 UTC before the change and 08:00 UTC after
            (datetime(2026, 11, 1, 6, 59, tzinfo=UTC), date(2026, 10, 31)),  # 11:59pm Saturday
            (datetime(2026, 11, 1, 7, 1, tzinfo=UTC), date(2026, 11, 1)),    # 12:01am Sunday
            (datetime(2026, 11, 1, 8, 30, tzinfo=UTC), date(2026, 11, 1)),   # 1:30am, the first time
            (datetime(2026, 11, 1, 9, 30, tzinfo=UTC), date(2026, 11, 1)),   # 1:30am, the second
            (datetime(2026, 11, 2, 7, 59, tzinfo=UTC), date(2026, 11, 1)),   # 11:59pm Sunday
            (datetime(2026, 11, 2, 8, 1, tzinfo=UTC), date(2026, 11, 2)),    # 12:01am Monday
            (datetime(2026, 11, 2, 7, 30, tzinfo=UTC), date(2026, 11, 1)),   # 11:30pm Sunday ...
            (datetime(2026, 11, 1, 7, 30, tzinfo=UTC), date(2026, 11, 1)),   # ... 12:30am that day
        ],
        ids=lambda value: value.isoformat() if isinstance(value, datetime) else None,
    )
    def test_across_both_clock_changes(self, moment, expected):
        assert config.trade_day(moment, SCHEDULED_RUN) == expected
        assert _pacific_day_by_hand(moment) == expected

    def test_a_repeated_wall_time_is_one_day_either_time(self):
        # 1:30am happens twice on the autumn clock-change day
        first = datetime(2026, 11, 1, 1, 30, tzinfo=_PACIFIC)
        second = first.replace(fold=1)
        assert first.astimezone(UTC) != second.astimezone(UTC)
        assert config.trade_day(first, SCHEDULED_RUN) == date(2026, 11, 1)
        assert config.trade_day(second, SCHEDULED_RUN) == date(2026, 11, 1)

    def test_every_hour_of_a_year_matches_the_pacific_wall_clock(self):
        # Each hour of 2026 at 29 minutes past, so no moment sits on a change
        moment = datetime(2026, 1, 1, 0, 29, tzinfo=UTC)
        while moment.year == 2026:
            assert config.trade_day(moment, SCHEDULED_RUN) == _pacific_day_by_hand(moment), moment
            moment += timedelta(hours=1)

    def test_the_same_moment_in_any_zone_is_the_same_day(self):
        moment = datetime(2026, 10, 14, 6, 30, tzinfo=UTC)   # 11:30pm Tuesday Pacific
        zones = [ZoneInfo(name) for name in ("UTC", "America/Los_Angeles", "America/New_York",
                                             "Asia/Tokyo", "Australia/Sydney")]
        zones.append(timezone(timedelta(hours=5, minutes=30)))
        for zone in zones:
            assert config.trade_day(moment.astimezone(zone), SCHEDULED_RUN) == date(2026, 10, 13)

    def test_the_day_is_read_on_the_schedules_zone_and_nothing_else_of_it(self):
        moment = datetime(2026, 10, 14, 0, 30, tzinfo=UTC)   # 5:30pm Tuesday Pacific
        assert config.trade_day(moment, SCHEDULED_RUN) == date(2026, 10, 13)
        assert config.trade_day(moment, ScheduledRun(0, 9, 0, "UTC")) == date(2026, 10, 14)
        assert config.trade_day(moment, ScheduledRun(0, 9, 0, "Asia/Tokyo")) == date(2026, 10, 14)
        # The schedule's weekday and time of day play no part
        for weekday, hour, minute in ((0, 9, 0), (3, 23, 59), (6, 0, 0)):
            run = ScheduledRun(weekday, hour, minute, "America/Los_Angeles")
            assert config.trade_day(moment, run) == date(2026, 10, 13)

    def test_it_returns_a_plain_date_even_from_a_datetime_subclass(self):
        class _Moment(datetime):
            """A datetime by subclass, as a clock-freezing stand-in is."""

        day = config.trade_day(_Moment(2026, 10, 14, 0, 30, tzinfo=UTC), SCHEDULED_RUN)
        # Exactly what rebalance_rested accepts
        assert type(day) is date and day == date(2026, 10, 13)
        assert config.rebalance_rested(day, day, 0) is True

    def test_a_time_with_no_zone_is_refused(self):
        class _NoOffset(tzinfo):
            """A zone object that gives no offset, so the time is still zoneless."""

            def utcoffset(self, dt):
                return None

            def dst(self, dt):
                return None

            def tzname(self, dt):
                return "none"

        for bad in (datetime(2026, 10, 13, 9, 0), datetime(2026, 10, 13, 9, 0, tzinfo=_NoOffset()),
                    date(2026, 10, 13), None, "2026-10-13T16:00:00Z", 1791907200, 1791907200.0):
            with pytest.raises(TypeError, match="trade_day reads a time with a zone"):
                config.trade_day(bad, SCHEDULED_RUN)

    def test_an_unknown_zone_is_an_error(self):
        with pytest.raises(ZoneInfoNotFoundError):
            config.trade_day(datetime(2026, 10, 13, 16, 0, tzinfo=UTC),
                             ScheduledRun(0, 9, 0, "No/Such_Zone"))

    def test_only_a_schedule_can_be_the_bots_clock(self):
        class _NoZone:
            """Has a zone() like a schedule's, but it names no zone."""

            def zone(self):
                return None

        moment = datetime(2026, 10, 14, 0, 30, tzinfo=UTC)
        # Only a real ScheduledRun is accepted: a stand-in whose zone() returns
        # None would put the day in this computer's own time zone
        for bad in (None, "America/Los_Angeles", _PACIFIC, UTC, _NoZone(), MagicMock()):
            with pytest.raises(TypeError, match="trade_day reads the zone of a ScheduledRun"):
                config.trade_day(moment, bad)

    def test_evening_trades_on_either_side_of_midnight_pacific(self):
        def rested(fill: datetime, run: datetime) -> bool:
            return config.rebalance_rested(config.trade_day(fill, SCHEDULED_RUN),
                                            config.trade_day(run, SCHEDULED_RUN), 7)

        # Traded at 11:59pm on Tuesday the 6th: a run just after midnight on
        # Tuesday the 13th may trim, a little under 7 x 24 hours later
        late = datetime(2026, 10, 6, 23, 59, tzinfo=_PACIFIC)
        early_run = datetime(2026, 10, 13, 0, 1, tzinfo=_PACIFIC)
        assert early_run - late < timedelta(days=7)
        assert rested(late, early_run) is True
        assert rested(late, datetime(2026, 10, 12, 23, 59, tzinfo=_PACIFIC)) is False
        # Traded two minutes later, at 12:01am on Wednesday the 7th: no run on
        # Tuesday the 13th may, not even its last minute
        after_midnight = datetime(2026, 10, 7, 0, 1, tzinfo=_PACIFIC)
        assert rested(after_midnight, early_run) is False
        assert rested(after_midnight, datetime(2026, 10, 13, 23, 59, tzinfo=_PACIFIC)) is False
        assert rested(after_midnight, datetime(2026, 10, 14, 0, 1, tzinfo=_PACIFIC)) is True

    def test_an_evening_trade_counts_on_its_pacific_day_not_its_utc_day(self):
        # 5:01pm Pacific on Tuesday the 6th is already Wednesday the 7th in UTC
        fill = datetime(2026, 10, 7, 0, 1, tzinfo=UTC)
        run = datetime(2026, 10, 13, 16, 0, tzinfo=UTC)     # 9am Pacific, Tuesday the 13th
        last, today = config.trade_day(fill, SCHEDULED_RUN), config.trade_day(run, SCHEDULED_RUN)
        assert (last, today) == (date(2026, 10, 6), date(2026, 10, 13))
        assert config.rebalance_rested(last, today, 7) is True
        # On a UTC clock the same two moments are six days apart
        utc_clock = ScheduledRun(0, 9, 0, "UTC")
        assert config.rebalance_rested(config.trade_day(fill, utc_clock),
                                        config.trade_day(run, utc_clock), 7) is False

    def test_a_run_at_any_hour_of_days_1_to_8_after_a_purchase(self):
        # Bought at any hour of Tuesday the 6th; runs at any hour of the 7th
        # through the 14th: only the two dates decide
        for bought_hour in (0, 9, 16, 17, 23):
            fill = datetime(2026, 10, 6, bought_hour, 30, tzinfo=_PACIFIC)
            last = config.trade_day(fill, SCHEDULED_RUN)
            for days_after in range(1, 9):
                for run_hour in (0, 9, 16, 17, 23):
                    run = datetime(2026, 10, 6 + days_after, run_hour, 15, tzinfo=_PACIFIC)
                    today = config.trade_day(run, SCHEDULED_RUN)
                    assert config.rebalance_rested(last, today, 7) is (days_after >= 7), (
                        fill, run)

    def test_the_wait_at_any_two_moments_of_a_year(self):
        # Fills stamped in UTC at any second of 2026, runs up to ten days
        # later: rested exactly when the Pacific dates are 7 or more apart
        rng = random.Random(20261013)
        start = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
        rested_count = 0
        for _ in range(4000):
            fill = start + timedelta(seconds=rng.randrange(0, 350 * 86_400))
            run = fill + timedelta(seconds=rng.randrange(0, 10 * 86_400))
            rested = config.rebalance_rested(config.trade_day(fill, SCHEDULED_RUN),
                                              config.trade_day(run, SCHEDULED_RUN), 7)
            by_hand = (_pacific_day_by_hand(run) - _pacific_day_by_hand(fill)).days >= 7
            assert rested is by_hand, (fill, run)
            # In elapsed time, rested needs over 6 days less an hour, and 7 days
            # plus an hour is always enough (a clock change moves an hour)
            if rested:
                assert run - fill > timedelta(days=6) - timedelta(hours=1), (fill, run)
            if run - fill >= timedelta(days=7, hours=1):
                assert rested, (fill, run)
            rested_count += rested
        # Both answers are well covered
        assert 500 < rested_count < 3500


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
            "same-title cap 100% (no extra cap) | exclude categories none | exclude tags none | "
            "rebalance no | min price movement any | rebalance until maturity | "
            "sell at off | min days to maturity any")
        assert config.describe_live_settings(s) == config.describe_live_settings(s, s)

    def test_the_shipped_values_render_with_no_mark(self):
        s = live_settings()
        assert "(config:" not in config.describe_live_settings(s, s)

    # One departing value per live toggle (config.LIVE_TOGGLE_FIELDS) and the
    # mark it must carry; a new toggle fails test_every_field_has_a_departure_row
    # until it has a row. _REFERENCE sells and rebalances, since a minimum of
    # days needs a sell level and a minimum price move needs rebalancing; those
    # two depart by turning off
    _REFERENCE = _settings(sell_at=0.85, rebalance=True)
    _DEPARTURES = {
        "tier_floors": (False, "tier floors off (config: on)"),
        "spread_band": ((0.1, 1.0), "spread band 0.1-1 (config: none)"),
        "interval_discount": (0.6, "k 0.6 (config: 0.75)"),
        "size_cap": (0.35, "per-trade cap 35% (config: 20%)"),
        "same_title_size_cap": (0.25, "same-title cap 25% (config: 100% (no extra cap))"),
        "exclude_categories": (("Economics",), "exclude categories Economics (config: none)"),
        "exclude_tags": (("Oil & Gas", "Energy"), "exclude tags Oil & Gas, Energy (config: none)"),
        "rebalance": (False, "rebalance no (config: yes)"),
        "min_price_move": (0.10, "min price movement 10c (config: any)"),
        "rebalance_min_days_to_maturity": (
            7, "rebalance until 7 days before maturity (config: maturity)"),
        "sell_at": (None, "sell at off (config: 85% of potential profit)"),
        "sell_min_days": (3, "min days to maturity 3 (config: any)"),
    }

    def test_every_field_has_a_departure_row(self):
        assert set(self._DEPARTURES) == set(config.LIVE_TOGGLE_FIELDS)

    def test_every_departing_field_is_marked_and_no_other(self):
        ref = _settings()
        s = _settings(False, (0.0, 0.5), 0.8, 1.0, 0.2)
        assert config.describe_live_settings(s, ref) == (
            "tier floors off (config: on) | spread band 0-0.5 (config: none) | "
            "k 0.8 (config: 0.75) | per-trade cap 100% (no cap) (config: 20%) | "
            "same-title cap 20% (config: 100% (no extra cap)) | exclude categories none | exclude tags none | "
            "rebalance no | min price movement any | rebalance until maturity | "
            "sell at off | min days to maturity any")
        for name in config.LIVE_TOGGLE_FIELDS:
            value, mark = self._DEPARTURES[name]
            line = config.describe_live_settings(
                dataclasses.replace(self._REFERENCE, **{name: value}), self._REFERENCE)
            assert line.count("(config:") == 1, line
            assert mark in line, (name, line)

    def test_rebalancing_renders_yes_or_no_and_the_move_in_whole_cents(self):
        # Each 5c step prints as its exact cents (0.55 * 100 is 55.00000000000001)
        assert [c for c in range(5, 101, 5) if c / 100 * 100 != c] == [55]
        assert "| rebalance no | min price movement any | rebalance until maturity |" in (
            config.describe_live_settings(_settings()))
        assert "| rebalance yes | min price movement any | rebalance until maturity |" in (
            config.describe_live_settings(_settings(rebalance=True)))
        for cents in range(5, 101, 5):
            line = config.describe_live_settings(
                _settings(rebalance=True, min_price_move=cents / 100))
            assert (f"| rebalance yes | min price movement {cents}c | rebalance until "
                    "maturity | sell at") in line, (cents, line)
        # Turning a minimum on over rebalancing with none marks that one field
        ref = _settings(rebalance=True)
        line = config.describe_live_settings(
            _settings(rebalance=True, min_price_move=0.25), ref)
        assert "| rebalance yes | min price movement 25c (config: any) |" in line
        assert line.count("(config:") == 1
        # Turning rebalancing on with a minimum, over rebalancing off, marks both
        line = config.describe_live_settings(
            _settings(rebalance=True, min_price_move=0.10), _settings())
        assert ("| rebalance yes (config: no) | min price movement 10c (config: any) |"
                in line)
        assert line.count("(config:") == 2

    def test_rebalance_until_renders_maturity_or_its_days(self):
        # None is "maturity" (always rebalance); 1 and 7 name the days before it
        for days, text in ((None, "maturity"), (1, "1 day before maturity"),
                           (7, "7 days before maturity")):
            line = config.describe_live_settings(
                _settings(rebalance=True, rebalance_min_days_to_maturity=days))
            assert f"| min price movement any | rebalance until {text} | sell at off |" in line
        # The mark names the reference's value, and only this field departs
        ref = _settings(rebalance=True, rebalance_min_days_to_maturity=7)
        line = config.describe_live_settings(_settings(rebalance=True), ref)
        assert "rebalance until maturity (config: 7 days before maturity)" in line
        assert line.count("(config:") == 1
        line = config.describe_live_settings(
            _settings(rebalance=True, rebalance_min_days_to_maturity=1), ref)
        assert "rebalance until 1 day before maturity (config: 7 days before maturity)" in line
        # A saved-defaults reference marks it "default"
        saved = dataclasses.replace(ref, origin=_SAVED_ORIGIN)
        line = config.describe_live_settings(_settings(rebalance=True), saved)
        assert "rebalance until maturity (default: 7 days before maturity)" in line

    def test_selling_renders_the_exact_whole_percent(self):
        # Off, then each level of the 1% grid as its whole percent: 0.29 * 100
        # is 28.999999999999996, which must never print as anything but 29
        assert "sell at off |" in config.describe_live_settings(_settings())
        for percent in range(1, 101):
            line = config.describe_live_settings(_settings(sell_at=percent / 100))
            assert f"sell at {percent}% of potential profit |" in line, (percent, line)
        line = config.describe_live_settings(_settings(sell_at=0.85, sell_min_days=7))
        assert line.endswith("sell at 85% of potential profit | min days to maturity 7")

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
                        "exclude categories none | exclude tags none | rebalance no | "
                        "min price movement any | rebalance until maturity | "
                        "sell at off | min days to maturity any")

    def test_a_filter_renders_its_names_and_none_for_none(self):
        # Each field renders the run's value, then config's in the mark; None is
        # "none" for the excluded categories and the excluded tags alike
        ref = _settings(exclude_categories=("Sports",))
        line = config.describe_live_settings(_settings(exclude_tags=("Basketball", "Soccer")), ref)
        assert "exclude categories none (config: Sports)" in line
        assert "exclude tags Basketball, Soccer (config: none)" in line
        # Case is part of the raw value: a case-only change departs and shows
        line = config.describe_live_settings(_settings(exclude_categories=("sports",)), ref)
        assert "exclude categories sports (config: Sports)" in line

    def test_a_filter_name_holding_a_separator_renders_exactly(self):
        # ("a, b",) and ("a", "b") must never print alike
        one = config.describe_live_settings(_settings(exclude_categories=("a, b",)))
        two = config.describe_live_settings(_settings(exclude_categories=("a", "b")))
        assert "exclude categories 'a, b' |" in one
        assert "exclude categories a, b |" in two
        assert one != two
        for sep in (";", "|"):
            line = config.describe_live_settings(_settings(exclude_tags=(f"x{sep}y", "z")))
            assert f"exclude tags 'x{sep}y', 'z'" in line

    @pytest.mark.parametrize("field", ["exclude_categories", "exclude_tags"])
    def test_only_the_name_any_is_refused_never_a_name_containing_it(self, field):
        # Only the name "any" is refused, never one containing it ("Any Awards")
        s = _settings(**{field: ("Companies", "Any Awards")})
        assert getattr(s, field) == ("Companies", "Any Awards")
        label = {"exclude_categories": "exclude categories", "exclude_tags": "exclude tags"}[field]
        assert f"{label} Companies, Any Awards" in config.describe_live_settings(s)
        assert config.describe_live_settings(s) != config.describe_live_settings(_settings())

    def test_a_saved_defaults_reference_marks_default(self):
        # A reference read from the saved file (any origin but config.py's)
        ref = dataclasses.replace(_settings(), origin=_SAVED_ORIGIN)
        s = _settings(False, (0.0, 0.5), 0.8, 1.0, 0.2)
        line = config.describe_live_settings(s, ref)
        assert line == (
            "tier floors off (default: on) | spread band 0-0.5 (default: none) | "
            "k 0.8 (default: 0.75) | per-trade cap 100% (no cap) (default: 20%) | "
            "same-title cap 20% (default: 100% (no extra cap)) | exclude categories none | exclude tags none | "
            "rebalance no | min price movement any | rebalance until maturity | "
            "sell at off | min days to maturity any")
        assert "(config:" not in line
        # Every field's mark follows the reference's origin, one mark per departure
        saved = dataclasses.replace(self._REFERENCE, origin=_SAVED_ORIGIN)
        for name in config.LIVE_TOGGLE_FIELDS:
            value, mark = self._DEPARTURES[name]
            line = config.describe_live_settings(
                dataclasses.replace(saved, **{name: value}), saved)
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
        assert config.describe_trade_filter(_settings()) == "exclude categories none; exclude tags none"
        s = _settings(exclude_categories=("Economics", "Sports"), exclude_tags=("Oil & Gas",))
        assert config.describe_trade_filter(s) == (
            "exclude categories Economics, Sports; exclude tags Oil & Gas")
        line = config.describe_live_settings(s)
        assert "exclude categories Economics, Sports" in line and "exclude tags Oil & Gas" in line


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
    def test_rebalancing_adds_no_sentence(self, cap, k):
        # Kelly on the whole position keeps a held pair within max_kelly_fraction
        # of the portfolio value, the bound the EXPOSURE sentence already names,
        # and a trim only sells, so rebalancing changes no sentence at any cap or k
        off = _settings(interval_discount=k, size_cap=cap)
        on = dataclasses.replace(off, rebalance=True)
        assert config.live_rule_warnings(on) == config.live_rule_warnings(off)
        moved = dataclasses.replace(on, min_price_move=0.10)
        assert config.live_rule_warnings(moved) == config.live_rule_warnings(off)

    def test_the_threshold_is_the_constant(self, monkeypatch):
        monkeypatch.setattr(config, "LIVE_EXPOSURE_WARN_FRACTION", 0.10)
        out = config.live_rule_warnings(_settings())
        assert len(out) == 2 and all("above the 10%" in text for text in out)


class TestRebalanceChangeWarnings:
    """rebalance_change_warnings: with Rebalance yes, a lower size cap or a
    higher k can sell more of the held pairs (through a Kelly trim). One
    sentence names every such change, and nothing else is warned about."""

    # A per-trade and a same-title cap of 20%, k 0.8, rebalancing on
    _NOW = LiveSettings(False, (0.0, 0.5), 0.8, 0.2, 0.2, rebalance=True)

    @staticmethod
    def _sentence(named: str, lone: bool = True) -> list:
        """The sentence for these changes; `lone` False when none can reach a lone leg."""
        who = ("each exact held pair, or lone leg whose partner has paid out, worth"
               if lone else "each exact held pair worth")
        return [f"with Rebalance yes, these settings can sell more of the held pairs: "
                f"{named}; {who} more than its new Kelly size is sold down to it before "
                "anything is bought, once the wait since its last trade has passed"]

    def test_each_change_that_sells_is_named(self):
        new = dataclasses.replace
        assert config.rebalance_change_warnings(self._NOW, new(self._NOW, size_cap=0.1)) == (
            self._sentence("the time-series size cap goes from 20% to 10% and the "
                           "same-title size cap goes from 20% to 10%"))
        assert config.rebalance_change_warnings(
            self._NOW, new(self._NOW, same_title_size_cap=0.1)) == self._sentence(
                "the same-title size cap goes from 20% to 10%", lone=False)
        assert config.rebalance_change_warnings(
            self._NOW, new(self._NOW, interval_discount=0.85)) == self._sentence(
                "k goes from 0.8 to 0.85, which lowers a time-series pair's chance of paying")
        # Three at once: a comma, then "and" before the last
        assert config.rebalance_change_warnings(
            self._NOW, new(self._NOW, size_cap=0.1, interval_discount=0.9)) == self._sentence(
                "the time-series size cap goes from 20% to 10%, the same-title size cap goes "
                "from 20% to 10% and k goes from 0.8 to 0.9, which lowers a time-series "
                "pair's chance of paying")
        # No cap at all reads as such
        uncapped = new(self._NOW, size_cap=1.0, same_title_size_cap=1.0)
        assert config.rebalance_change_warnings(uncapped, new(uncapped, size_cap=0.5)) == (
            self._sentence("the time-series size cap goes from 100% (no cap) to 50% and the "
                           "same-title size cap goes from 100% (no cap) to 50%"))

    def test_a_lone_leg_is_named_only_when_a_time_series_setting_can_reach_it(self):
        # A lone leg is the later market of a time-series pair: a same-title cap
        # alone never reaches one, a time-series cap or k does, alone or with a
        # same-title change
        new = dataclasses.replace
        [same_title] = config.rebalance_change_warnings(
            self._NOW, new(self._NOW, same_title_size_cap=0.1))
        assert "lone leg" not in same_title
        # Only the time-series cap falls: the same-title cap was already the lower
        held = new(self._NOW, size_cap=0.3, same_title_size_cap=0.2)
        assert config.rebalance_change_warnings(held, new(held, size_cap=0.25)) == (
            self._sentence("the time-series size cap goes from 30% to 25%"))
        [k_up] = config.rebalance_change_warnings(self._NOW, new(self._NOW, interval_discount=0.9))
        assert "or lone leg whose partner has paid out, worth more than" in k_up
        both = config.rebalance_change_warnings(
            self._NOW, new(self._NOW, same_title_size_cap=0.1, interval_discount=0.9))
        assert "or lone leg whose partner has paid out" in both[0]

    @pytest.mark.parametrize("change", [
        # Rebalancing off in the proposed settings: nothing is trimmed at all
        {"rebalance": False, "size_cap": 0.1},
        # A cap raised, k lowered, or nothing changed: nothing more is sold
        {"size_cap": 0.3}, {"interval_discount": 0.7}, {},
        # Settings that only buy differently
        {"tier_floors": True, "spread_band": (0.0, 1.0)},
        # A minimum price move only holds a trim back
        {"min_price_move": 0.1},
    ])
    def test_nothing_is_said_when_no_more_is_sold(self, change):
        assert config.rebalance_change_warnings(
            self._NOW, dataclasses.replace(self._NOW, **change)) == []

    @pytest.mark.parametrize("before, after", [(7, 1), (7, None), (1, None), (None, 1),
                                               (None, 7), (1, 7)])
    def test_rebalance_until_is_never_named(self, before, after):
        # "Rebalance until" only holds a trim back near maturity and never makes
        # one bigger, so no sentence names it, lowered or raised
        current = dataclasses.replace(self._NOW, rebalance_min_days_to_maturity=before)
        proposed = dataclasses.replace(self._NOW, rebalance_min_days_to_maturity=after)
        assert config.rebalance_change_warnings(current, proposed) == []

    def test_a_same_title_cap_held_below_by_the_per_trade_cap_moves_nothing(self):
        # The same-title cap a pair is trimmed to is the lower of the two
        before = dataclasses.replace(self._NOW, same_title_size_cap=1.0)
        after = dataclasses.replace(before, same_title_size_cap=0.5)
        assert config.pair_size_cap("same_title", 0.2, 1.0) == config.pair_size_cap(
            "same_title", 0.2, 0.5)
        assert config.rebalance_change_warnings(before, after) == []

    def test_turning_rebalancing_on_is_judged_by_the_settings_it_trims_by(self):
        # Saved off, proposed on with a lower cap: the first trims use the new
        # cap, so it is named
        off = dataclasses.replace(self._NOW, rebalance=False)
        assert config.rebalance_change_warnings(off, dataclasses.replace(self._NOW, size_cap=0.1))


class TestShippedLiveToggles:
    """The live toggles config.py ships, read unpatched end to end: any
    positive time-series spread up to 0.5 is admitted, books price up to a
    leg-price sum of 1.0 with no-edge levels cut, and one pair stakes at most
    20% (time-series under 1 - k, same-title at its cap). Changing a toggle
    means re-pinning this class."""

    _SHIPPED = LiveSettings(tier_floors=False, spread_band=(0.0, 0.5), interval_discount=0.8,
                            size_cap=1.0, same_title_size_cap=0.2, exclude_categories=None, exclude_tags=None,
                            rebalance=True)
    # The values conftest's apply_pre_toggle_defaults pins, for the controls
    _BEFORE = LiveSettings(tier_floors=True, spread_band=(0.0, 1.0), interval_discount=0.75,
                           size_cap=0.2, same_title_size_cap=1.0)

    def test_the_shipped_settings(self):
        assert live_settings() == self._SHIPPED
        assert (config.TIME_SERIES_TIER_FLOORS, config.TIME_SERIES_SPREAD_BAND,
                config.TIME_SERIES_INTERVAL_PROB_DISCOUNT, config.BUDGET_FRACTION,
                config.SAME_TITLE_SIZE_CAP, config.TRADE_EXCLUDE_CATEGORIES, config.TRADE_EXCLUDE_TAGS,
                config.REBALANCE, config.MIN_PRICE_MOVE, config.REBALANCE_MIN_DAYS_TO_MATURITY,
                config.SELL_AT, config.SELL_MIN_DAYS) == (
            False, (0.0, 0.5), 0.80, 1.0, 0.20, None, None, True, None, None, None, None)
        # ... and the seed proposes rebalancing too, with no minimum price move and
        # no limit on how close to maturity a position is rebalanced
        assert (config.LIVE_DEFAULTS_SEED.rebalance, config.LIVE_DEFAULTS_SEED.min_price_move,
                config.LIVE_DEFAULTS_SEED.rebalance_min_days_to_maturity) == (True, None, None)

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
                    # The forecast's input, as enrichment writes it from the books' tops
                    pair.mid_spread = time_series_mid_spread(pA, 1 - pA, pB, nB)
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
        # Rebalancing on (as shipped) included
        assert live_settings().rebalance is True
        assert config.live_rule_warnings(live_settings()) == []

    def test_a_default_backtest_departs_from_the_live_rule(self, monkeypatch, caplog,
                                                            saved_live_defaults):
        # The shipped toggles saved as the live defaults: the primary follows k
        # and both caps but keeps band (0, 1) with the tiers on, so the live
        # rule is a grid cell, and the last line says so (no sizing note: the
        # saved k and caps are the run's own). The shipped defaults rebalance,
        # which no point of a backtest run does, so the line ends with that
        # note (backtester._live_rebalance_note, whose words its own tests pin).
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
        assert (res.live_rebalance, res.live_min_price_move) == (True, None)
        note = backtester._live_rebalance_note(res)
        assert note.startswith("; the live defaults rebalance")
        rule = config.describe_time_series_rule(False, (0.0, 0.5))
        lines = [r.getMessage() for r in caplog.records
                 if r.getMessage().startswith("Live time-series rule")]
        assert lines == [
            f"Live time-series rule (saved live defaults): {rule} — this run's primary "
            "scenario does not (tier floors on, band 0-1); its grid simulated the live rule "
            "as band 0-0.5 with the tier floors off, which the dashboard's filter bar shows"
            + note]

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
        "TRADE_EXCLUDE_CATEGORIES", "TRADE_EXCLUDE_TAGS", "REBALANCE", "MIN_PRICE_MOVE",
        "REBALANCE_MIN_DAYS_TO_MATURITY", "SELL_AT", "SELL_MIN_DAYS"})

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
            same_title_size_cap=1.0, exclude_categories=None, exclude_tags=None, rebalance=False,
            min_price_move=None, rebalance_min_days_to_maturity=None, sell_at=None,
            sell_min_days=None)
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
        assert dataclasses.replace(s, exclude_categories=("Sports",)).origin == _SAVED_ORIGIN

    def test_the_toggle_fields_are_the_twelve(self):
        assert config.LIVE_TOGGLE_FIELDS == (
            "tier_floors", "spread_band", "interval_discount", "size_cap",
            "same_title_size_cap", "exclude_categories", "exclude_tags", "rebalance", "min_price_move",
            "rebalance_min_days_to_maturity", "sell_at", "sell_min_days")
        assert len(config.LIVE_TOGGLE_FIELDS) == 12
        # The five a saved file may leave out, in field order, at the end
        assert config._OPTIONAL_TOGGLES == ("rebalance", "min_price_move",
                                            "rebalance_min_days_to_maturity", "sell_at",
                                            "sell_min_days")
        assert config._TOGGLE_DEFAULTS == {
            "rebalance": False, "min_price_move": None,
            "rebalance_min_days_to_maturity": None, "sell_at": None, "sell_min_days": None}
        # The settings an older version saved in their place, refused when read
        assert config._RETIRED_TOGGLES == ("add_to_held_pairs", "trim_to_kelly",
                                           "trim_min_rise", "categories", "tags")
        assert [f.name for f in dataclasses.fields(LiveSettings)] == [
            *config.LIVE_TOGGLE_FIELDS, "origin"]

    def test_every_toggle_count_the_package_states_is_the_number_of_toggles(self):
        # Wherever a module's comments, docstrings or messages count the toggles
        # ("the twelve toggles", "twelve live toggles", "twelve toggle names"),
        # the count is the number of toggle fields or the number a saved file
        # must hold. Comment lines are joined first, so a split count is found
        names = ("one two three four five six seven eight nine ten eleven twelve "
                 "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty").split()
        total = len(config.LIVE_TOGGLE_FIELDS)
        required = total - len(config._OPTIONAL_TOGGLES)
        allowed = {names[total - 1], names[required - 1]}
        # Counts below the required number ("these two toggles") name a few
        # toggles, not all of them, and are left alone
        counted = re.compile(r"\b(" + "|".join(names[required - 1:])
                             + r")[\s-]+(?:live\s+)?toggles?\b", re.IGNORECASE)
        stated: dict[str, set[str]] = {}
        for path in sorted(pathlib.Path(config.__file__).parent.glob("*.py")):
            source = re.sub(r"\n[ \t]*#[ \t]*", " ", path.read_text(encoding="utf-8"))
            for match in counted.finditer(source):
                stated.setdefault(match.group(1).lower(), set()).add(path.name)
        # The scan finds the counts config.py and defaults_server.py state
        assert {"config.py", "defaults_server.py"} <= stated.get(names[total - 1], set())
        assert set(stated) <= allowed, {
            word: sorted(files) for word, files in stated.items() if word not in allowed}


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
    no source note and the optional toggles left out (the seven-toggle
    shape), so it reads as _SEED_REBALANCE_OFF.

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
            "size_cap": 0.1, "same_title_size_cap": 0.2, "exclude_categories": None, "exclude_tags": None,
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

# LIVE_DEFAULTS_SEED with rebalancing off (no selling either): what
# _valid_record's seven-toggle file reads as, and what a save of it writes as
# seven toggles
_SEED_REBALANCE_OFF = dataclasses.replace(config.LIVE_DEFAULTS_SEED, rebalance=False)

# The exact bytes a save of _SEED_REBALANCE_OFF writes, with the seed's note, at
# _FrozenDatetime's instant: the optional toggles are left out while off, so
# code that does not know them can still read the file
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
    b'    "exclude_categories": null,\n'
    b'    "exclude_tags": null\n'
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
        same_title_size_cap=0.5, exclude_categories=("Sports",), exclude_tags=("Basketball",))

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
        assert '"exclude_categories": ["Sports"],' in text and '"exclude_tags": ["Basketball"]\n' in text

    def test_the_written_text_is_pinned(self, monkeypatch):
        # With rebalancing off, the key is left out, byte for byte, so code
        # that does not know the toggle can read a file saved with it off
        monkeypatch.setattr(config, "datetime", _FrozenDatetime)
        config.save_live_defaults(_SEED_REBALANCE_OFF, source=config.LIVE_DEFAULTS_SEED_SOURCE)
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == _SEVEN_TOGGLE_BYTES

    def test_an_on_save_writes_the_key_after_the_filters(self, monkeypatch):
        # The seed (rebalancing on, no minimum) writes the toggle, after the
        # other seven
        monkeypatch.setattr(config, "datetime", _FrozenDatetime)
        config.save_live_defaults(config.LIVE_DEFAULTS_SEED,
                                  source=config.LIVE_DEFAULTS_SEED_SOURCE)
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == _SEVEN_TOGGLE_BYTES.replace(
            b'    "exclude_tags": null\n',
            b'    "exclude_tags": null,\n    "rebalance": true\n')
        assert config.read_saved_live_defaults() == config.LIVE_DEFAULTS_SEED

    def test_selling_off_writes_exactly_the_bytes_written_before_it_existed(self, monkeypatch):
        # sell_at and sell_min_days at their defaults are not in the file at
        # all, whether rebalancing is off or on
        monkeypatch.setattr(config, "datetime", _FrozenDatetime)
        for settings in (_SEED_REBALANCE_OFF, config.LIVE_DEFAULTS_SEED):
            config.save_live_defaults(settings, source=config.LIVE_DEFAULTS_SEED_SOURCE)
            text = config.LIVE_DEFAULTS_FILE.read_text()
            assert "sell_at" not in text and "sell_min_days" not in text
        assert config.LIVE_DEFAULTS_FILE.read_bytes().endswith(
            b'    "exclude_tags": null,\n    "rebalance": true\n  }\n}\n')

    def test_a_selling_save_writes_its_keys_last_in_field_order(self, monkeypatch):
        monkeypatch.setattr(config, "datetime", _FrozenDatetime)
        # A level and a minimum, after rebalancing
        both = dataclasses.replace(config.LIVE_DEFAULTS_SEED, sell_at=0.85, sell_min_days=3)
        config.save_live_defaults(both, source=config.LIVE_DEFAULTS_SEED_SOURCE)
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == _SEVEN_TOGGLE_BYTES.replace(
            b'    "exclude_tags": null\n',
            b'    "exclude_tags": null,\n    "rebalance": true,\n'
            b'    "sell_at": 0.85,\n    "sell_min_days": 3\n')
        assert config.read_saved_live_defaults() == both
        # A level alone leaves the minimum out; rebalancing off leaves its key out too
        level = dataclasses.replace(_SEED_REBALANCE_OFF, sell_at=0.5)
        config.save_live_defaults(level, source=config.LIVE_DEFAULTS_SEED_SOURCE)
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == _SEVEN_TOGGLE_BYTES.replace(
            b'    "exclude_tags": null\n', b'    "exclude_tags": null,\n    "sell_at": 0.5\n')
        assert config.read_saved_live_defaults() == level
        # Every level of the grid round-trips as the same float
        for percent in range(1, 101):
            wanted = dataclasses.replace(_SEED_REBALANCE_OFF, sell_at=percent / 100,
                                         sell_min_days=percent)
            assert config.save_live_defaults(wanted, source="") == wanted
            assert config.read_saved_live_defaults().sell_at == percent / 100

    def test_a_save_round_trips_the_sell_settings(self):
        settings = dataclasses.replace(self._SETTINGS, sell_at=0.9, sell_min_days=14)
        saved = config.save_live_defaults(settings, source="a note")
        assert saved == settings
        assert (saved.sell_at, saved.sell_min_days) == (0.9, 14)
        assert config.read_saved_live_defaults() == settings

    def test_rebalancing_off_writes_exactly_the_bytes_written_before_it_existed(
            self, monkeypatch):
        # Both rebalance settings at their defaults are left out of the file:
        # with selling off too it is byte for byte the seven-toggle file
        monkeypatch.setattr(config, "datetime", _FrozenDatetime)
        assert (_SEED_REBALANCE_OFF.rebalance, _SEED_REBALANCE_OFF.min_price_move) == (
            False, None)
        config.save_live_defaults(_SEED_REBALANCE_OFF, source=config.LIVE_DEFAULTS_SEED_SOURCE)
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == _SEVEN_TOGGLE_BYTES
        for settings in (_SEED_REBALANCE_OFF,
                         dataclasses.replace(_SEED_REBALANCE_OFF, sell_at=0.85,
                                             sell_min_days=3)):
            config.save_live_defaults(settings, source=config.LIVE_DEFAULTS_SEED_SOURCE)
            text = config.LIVE_DEFAULTS_FILE.read_bytes()
            assert b"rebalance" not in text and b"min_price_move" not in text
        assert config.LIVE_DEFAULTS_FILE.read_bytes().endswith(
            b'    "sell_at": 0.85,\n    "sell_min_days": 3\n  }\n}\n')

    def test_a_rebalancing_save_writes_its_keys_in_field_order(self, monkeypatch):
        monkeypatch.setattr(config, "datetime", _FrozenDatetime)
        # With a minimum and both sell keys: rebalance, then its minimum, then
        # the sell level and its minimum of days
        everything = dataclasses.replace(config.LIVE_DEFAULTS_SEED, min_price_move=0.10,
                                         sell_at=0.85, sell_min_days=3)
        config.save_live_defaults(everything, source=config.LIVE_DEFAULTS_SEED_SOURCE)
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == _SEVEN_TOGGLE_BYTES.replace(
            b'    "exclude_tags": null\n',
            b'    "exclude_tags": null,\n    "rebalance": true,\n    "min_price_move": 0.1,\n'
            b'    "sell_at": 0.85,\n    "sell_min_days": 3\n')
        assert config.read_saved_live_defaults() == everything
        # Every 5c step round-trips as the same float
        for cents in range(5, 101, 5):
            wanted = dataclasses.replace(config.LIVE_DEFAULTS_SEED, min_price_move=cents / 100)
            assert config.save_live_defaults(wanted, source="") == wanted
            assert config.read_saved_live_defaults().min_price_move == cents / 100

    def test_maturity_writes_exactly_the_bytes_written_before_it_existed(self, monkeypatch):
        # The days to maturity at "maturity" (None) are left out of the file, so a
        # file saved with it so is byte for byte the file saved without the setting
        monkeypatch.setattr(config, "datetime", _FrozenDatetime)
        for settings in (_SEED_REBALANCE_OFF, config.LIVE_DEFAULTS_SEED):
            assert settings.rebalance_min_days_to_maturity is None
            config.save_live_defaults(settings, source=config.LIVE_DEFAULTS_SEED_SOURCE)
            assert b"rebalance_min_days_to_maturity" not in config.LIVE_DEFAULTS_FILE.read_bytes()
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == _SEVEN_TOGGLE_BYTES.replace(
            b'    "exclude_tags": null\n',
            b'    "exclude_tags": null,\n    "rebalance": true\n')
        # A 0 given in is stored as None, so it is left out too
        zero = dataclasses.replace(config.LIVE_DEFAULTS_SEED, rebalance_min_days_to_maturity=0)
        config.save_live_defaults(zero, source=config.LIVE_DEFAULTS_SEED_SOURCE)
        assert b"rebalance_min_days_to_maturity" not in config.LIVE_DEFAULTS_FILE.read_bytes()

    def test_a_save_writes_the_days_to_maturity_after_the_minimum_move(self, monkeypatch):
        monkeypatch.setattr(config, "datetime", _FrozenDatetime)
        # Rebalancing, its minimum move, its days to maturity, then the sell keys
        everything = dataclasses.replace(
            config.LIVE_DEFAULTS_SEED, min_price_move=0.10, rebalance_min_days_to_maturity=7,
            sell_at=0.85, sell_min_days=3)
        config.save_live_defaults(everything, source=config.LIVE_DEFAULTS_SEED_SOURCE)
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == _SEVEN_TOGGLE_BYTES.replace(
            b'    "exclude_tags": null\n',
            b'    "exclude_tags": null,\n    "rebalance": true,\n    "min_price_move": 0.1,\n'
            b'    "rebalance_min_days_to_maturity": 7,\n'
            b'    "sell_at": 0.85,\n    "sell_min_days": 3\n')
        assert config.read_saved_live_defaults() == everything
        # Alone, it follows rebalance directly
        one = dataclasses.replace(config.LIVE_DEFAULTS_SEED, rebalance_min_days_to_maturity=1)
        config.save_live_defaults(one, source=config.LIVE_DEFAULTS_SEED_SOURCE)
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == _SEVEN_TOGGLE_BYTES.replace(
            b'    "exclude_tags": null\n',
            b'    "exclude_tags": null,\n    "rebalance": true,\n'
            b'    "rebalance_min_days_to_maturity": 1\n')
        assert config.read_saved_live_defaults() == one

    @pytest.mark.parametrize("days", [1, 7])
    def test_a_save_round_trips_the_days_to_maturity(self, days):
        settings = dataclasses.replace(self._SETTINGS, rebalance=True,
                                       rebalance_min_days_to_maturity=days)
        saved = config.save_live_defaults(settings, source="a note")
        assert saved == settings and saved.rebalance_min_days_to_maturity == days
        again = config.read_saved_live_defaults()
        assert again == settings and type(again.rebalance_min_days_to_maturity) is int

    def test_a_save_round_trips_the_rebalance_settings(self):
        settings = dataclasses.replace(self._SETTINGS, rebalance=True, min_price_move=0.25)
        saved = config.save_live_defaults(settings, source="a note")
        assert saved == settings
        assert (saved.rebalance, saved.min_price_move) == (True, 0.25)
        assert config.read_saved_live_defaults() == settings

    @pytest.mark.parametrize("settings", [
        _SEED_REBALANCE_OFF, config.LIVE_DEFAULTS_SEED,
        dataclasses.replace(config.LIVE_DEFAULTS_SEED, min_price_move=0.10, sell_at=0.85,
                            sell_min_days=3),
    ])
    def test_the_writer_never_writes_a_retired_key(self, settings):
        # Only the twelve toggles' names reach the file, so a save always
        # writes a file every live run reads
        config.save_live_defaults(settings, source="")
        record = json.loads(config.LIVE_DEFAULTS_FILE.read_text(encoding="utf-8"))
        assert set(record["settings"]) <= set(config.LIVE_TOGGLE_FIELDS)
        assert not set(record["settings"]) & set(config._RETIRED_TOGGLES)

    def test_a_minimum_with_rebalancing_off_is_never_written(self):
        # LiveSettings and the saved file's parser both refuse a minimum with
        # rebalancing off, and the file on disk is left as it was
        config.save_live_defaults(config.LIVE_DEFAULTS_SEED, source="")
        before = config.LIVE_DEFAULTS_FILE.read_bytes()
        with pytest.raises(ValueError, match="min_price_move needs rebalance on"):
            dataclasses.replace(_SEED_REBALANCE_OFF, min_price_move=0.10)
        with pytest.raises(ValueError, match="min_price_move needs rebalance on"):
            config._settings_from_bytes(_text(_with_toggles(min_price_move=0.10)))
        assert config.LIVE_DEFAULTS_FILE.read_bytes() == before

    def test_a_seven_toggle_file_reads_with_rebalancing_off(self):
        # A file that leaves rebalance out reads as the same settings,
        # rebalancing off with no minimum (nobody confirmed a value for it),
        # whatever config.py or the seed ship
        config.LIVE_DEFAULTS_FILE.write_bytes(_SEVEN_TOGGLE_BYTES)
        saved = config.read_saved_live_defaults()
        assert saved == _SEED_REBALANCE_OFF and saved.rebalance is False
        assert saved.min_price_move is None
        # ... and it reads selling off, the same way
        assert saved.sell_at is None and saved.sell_min_days is None
        assert config.REBALANCE is True
        assert config.LIVE_DEFAULTS_SEED.rebalance is True
        assert saved.origin == ("live_defaults.json, saved 2026-09-27T21:05:13Z from "
                                "seed values (config.LIVE_DEFAULTS_SEED)")

    @pytest.mark.parametrize("value", [False, True])
    def test_a_file_naming_the_toggle_reads_it(self, value):
        # Written either way by hand, the toggle reads as written
        config.LIVE_DEFAULTS_FILE.write_bytes(_text(_with_toggles(rebalance=value)))
        assert config.read_saved_live_defaults() == dataclasses.replace(
            _SEED_REBALANCE_OFF, rebalance=value)

    def test_a_file_that_leaves_selling_out_reads_it_off(self):
        # Neither key is there (a file saved before they existed), so nobody
        # confirmed a value: never selling, whatever config.py ships
        config.LIVE_DEFAULTS_FILE.write_bytes(_text(_with_toggles(rebalance=True)))
        saved = config.read_saved_live_defaults()
        assert saved.sell_at is None and saved.sell_min_days is None
        assert config.SELL_AT is None and config.LIVE_DEFAULTS_SEED.sell_at is None

    def test_a_file_that_leaves_rebalancing_out_reads_it_off(self):
        # Neither key is in the file, so nobody chose to rebalance: it reads
        # as off, whatever config.py ships
        config.LIVE_DEFAULTS_FILE.write_bytes(_text(_with_toggles(sell_at=0.85,
                                                                  sell_min_days=3)))
        saved = config.read_saved_live_defaults()
        assert saved.rebalance is False and saved.min_price_move is None
        assert saved == dataclasses.replace(_SEED_REBALANCE_OFF, sell_at=0.85,
                                            sell_min_days=3)

    @pytest.mark.parametrize("changes, expected", [
        ({"rebalance": False}, {}),
        ({"rebalance": True}, {"rebalance": True}),
        ({"rebalance": True, "min_price_move": 0.1},
         {"rebalance": True, "min_price_move": 0.10}),
        ({"rebalance": True, "min_price_move": 0.25},
         {"rebalance": True, "min_price_move": 0.25}),
        # A whole dollar written as the JSON integer 1
        ({"rebalance": True, "min_price_move": 1},
         {"rebalance": True, "min_price_move": 1.0}),
        # Written as null by hand, the minimum reads as none, on or off
        ({"rebalance": True, "min_price_move": None}, {"rebalance": True}),
        ({"rebalance": False, "min_price_move": None}, {}),
        ({"min_price_move": None}, {}),
    ])
    def test_a_file_naming_the_rebalance_settings_reads_them(self, changes, expected):
        config.LIVE_DEFAULTS_FILE.write_bytes(_text(_with_toggles(**changes)))
        saved = config.read_saved_live_defaults()
        assert saved == dataclasses.replace(_SEED_REBALANCE_OFF, **expected)
        assert type(saved.rebalance) is bool
        assert saved.min_price_move is None or type(saved.min_price_move) is float

    @pytest.mark.parametrize("changes, expected", [
        ({"rebalance": True, "rebalance_min_days_to_maturity": 1},
         {"rebalance": True, "rebalance_min_days_to_maturity": 1}),
        ({"rebalance": True, "rebalance_min_days_to_maturity": 7},
         {"rebalance": True, "rebalance_min_days_to_maturity": 7}),
        ({"rebalance": True, "min_price_move": 0.1, "rebalance_min_days_to_maturity": 7},
         {"rebalance": True, "min_price_move": 0.1, "rebalance_min_days_to_maturity": 7}),
        # Written as 0 or null by hand, it reads "maturity" (none), on or off
        ({"rebalance": True, "rebalance_min_days_to_maturity": 0}, {"rebalance": True}),
        ({"rebalance": True, "rebalance_min_days_to_maturity": None}, {"rebalance": True}),
        ({"rebalance": False, "rebalance_min_days_to_maturity": None}, {}),
        ({"rebalance": False, "rebalance_min_days_to_maturity": 0}, {}),
        ({"rebalance_min_days_to_maturity": None}, {}),
    ])
    def test_a_file_naming_the_days_to_maturity_reads_them(self, changes, expected):
        config.LIVE_DEFAULTS_FILE.write_bytes(_text(_with_toggles(**changes)))
        saved = config.read_saved_live_defaults()
        assert saved == dataclasses.replace(_SEED_REBALANCE_OFF, **expected)
        assert (saved.rebalance_min_days_to_maturity is None
                or type(saved.rebalance_min_days_to_maturity) is int)

    def test_a_file_that_leaves_the_days_out_reads_maturity(self):
        # Nobody chose a limit, so a run rebalances however near maturity a
        # position is, with rebalancing on or off
        for rebalance in (False, True):
            config.LIVE_DEFAULTS_FILE.write_bytes(_text(_with_toggles(rebalance=rebalance)))
            assert config.read_saved_live_defaults().rebalance_min_days_to_maturity is None
        assert config.REBALANCE_MIN_DAYS_TO_MATURITY is None
        assert config.LIVE_DEFAULTS_SEED.rebalance_min_days_to_maturity is None

    @pytest.mark.parametrize("changes, expected", [
        ({"sell_at": 0.85}, {"sell_at": 0.85}),
        ({"sell_at": 0.85, "sell_min_days": 3}, {"sell_at": 0.85, "sell_min_days": 3}),
        ({"sell_at": 1}, {"sell_at": 1.0}),
        # Written as null by hand, a setting reads as off
        ({"sell_at": None, "sell_min_days": None}, {}),
    ])
    def test_a_file_naming_the_sell_settings_reads_them(self, changes, expected):
        config.LIVE_DEFAULTS_FILE.write_bytes(_text(_with_toggles(**changes)))
        assert config.read_saved_live_defaults() == dataclasses.replace(
            _SEED_REBALANCE_OFF, **expected)

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
        ({"exclude_categories": ("Sports\u200b",)}, "printable"),
        ({"exclude_tags": ("Basket\u202eball",)}, "printable"),
        ({"exclude_categories": tuple(f"Category {i:05d}" for i in range(5000))},
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
        # later (rebalance, say) may
        ("a missing toggle", _text(_with_toggles(exclude_tags=_DROP))),
        ("a missing toggle beside the later one",
         _text(_with_toggles(exclude_tags=_DROP, rebalance=False))),
        ("an extra toggle", _text(_with_toggles(origin="config.py"))),
        ("an extra toggle beside the later one",
         _text(_with_toggles(rebalance=True, sell_held_pairs=True))),
        # A setting an older version saved: refused until saved again
        ("an older version's adding switch", _text(_with_toggles(add_to_held_pairs=True))),
        ("an older version's trim settings",
         _text(_with_toggles(trim_to_kelly=True, trim_min_rise=0.1))),
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
        ("rebalance as 1", _text(_with_toggles(rebalance=1))),
        ("rebalance as a string", _text(_with_toggles(rebalance="true"))),
        ("rebalance as null", _text(_with_toggles(rebalance=None))),
        # Selling: a JSON true is a Python bool, and a float is not a day count
        ("a sell level of true", _text(_with_toggles(sell_at=True))),
        ("a sell level of false", _text(_with_toggles(sell_at=False))),
        ("a sell level of zero", _text(_with_toggles(sell_at=0))),
        ("a sell level above 100%", _text(_with_toggles(sell_at=1.01))),
        ("a sell level between two percents", _text(_with_toggles(sell_at=0.855))),
        ("a sell level as a string", _text(_with_toggles(sell_at="0.85"))),
        ("a sell level that is NaN", _text(_with_toggles(sell_at=float("nan")))),
        ("a sell minimum of true", _text(_with_toggles(sell_at=0.85, sell_min_days=True))),
        ("a sell minimum of false", _text(_with_toggles(sell_at=0.85, sell_min_days=False))),
        ("a sell minimum as a float", _text(_with_toggles(sell_at=0.85, sell_min_days=1.0))),
        ("a sell minimum of zero", _text(_with_toggles(sell_at=0.85, sell_min_days=0))),
        ("a sell minimum as a string", _text(_with_toggles(sell_at=0.85, sell_min_days="3"))),
        ("a sell minimum with no level", _text(_with_toggles(sell_min_days=3))),
        ("a sell minimum with a null level",
         _text(_with_toggles(sell_at=None, sell_min_days=3))),
        # The minimum price move is dollars on the 5c grid from 5c to $1, and
        # needs rebalancing on
        ("a minimum move of true", _text(_with_toggles(rebalance=True, min_price_move=True))),
        ("a minimum move of false",
         _text(_with_toggles(rebalance=True, min_price_move=False))),
        ("a minimum move of zero", _text(_with_toggles(rebalance=True, min_price_move=0))),
        ("a minimum move above a dollar",
         _text(_with_toggles(rebalance=True, min_price_move=1.05))),
        ("a minimum move in cents", _text(_with_toggles(rebalance=True, min_price_move=10))),
        ("a minimum move between two steps",
         _text(_with_toggles(rebalance=True, min_price_move=0.07))),
        ("a minimum move as a string",
         _text(_with_toggles(rebalance=True, min_price_move="0.10"))),
        ("a minimum move that is NaN",
         _text(_with_toggles(rebalance=True, min_price_move=float("nan")))),
        ("a minimum move too large to count",
         _text(_with_toggles(rebalance=True)).replace(
             b'"rebalance": true', b'"rebalance": true, "min_price_move": 1' + b"0" * 400)),
        ("a minimum move with rebalancing left out", _text(_with_toggles(min_price_move=0.10))),
        ("a minimum move with rebalancing off",
         _text(_with_toggles(rebalance=False, min_price_move=0.10))),
        # The days to maturity are 0 (read as none), 1 or 7, and need rebalancing on
        ("rebalance days of true",
         _text(_with_toggles(rebalance=True, rebalance_min_days_to_maturity=True))),
        ("rebalance days of false",
         _text(_with_toggles(rebalance=True, rebalance_min_days_to_maturity=False))),
        ("rebalance days of two",
         _text(_with_toggles(rebalance=True, rebalance_min_days_to_maturity=2))),
        ("rebalance days below zero",
         _text(_with_toggles(rebalance=True, rebalance_min_days_to_maturity=-1))),
        ("rebalance days as a float",
         _text(_with_toggles(rebalance=True, rebalance_min_days_to_maturity=1.0))),
        ("rebalance days as a string",
         _text(_with_toggles(rebalance=True, rebalance_min_days_to_maturity="7"))),
        ("rebalance days with rebalancing left out",
         _text(_with_toggles(rebalance_min_days_to_maturity=7))),
        ("rebalance days with rebalancing off",
         _text(_with_toggles(rebalance=False, rebalance_min_days_to_maturity=1))),
        # Category and tag names
        ("the name any", _text(_with_toggles(exclude_categories=["any"]))),
        ("an empty name", _text(_with_toggles(exclude_tags=[""]))),
        ("a zero-width name", _text(_with_toggles(exclude_categories=["Sports\u200b"]))),
        ("a bare string filter", _text(_with_toggles(exclude_tags="Basketball"))),
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
            (_text(_with_toggles(exclude_categories=["Sports\u200b"])), "printable"),
            (_text(_with_toggles(exclude_categories=["any"])), "null in live_defaults.json"),
            (_text(_with_toggles(size_cap=0.33)), "size_cap"),
            (_text(_with_toggles(spread_band=[False, True])),
             '"spread_band" must be [floor, ceiling]'),
            (_text(_with_toggles(spread_band=[0.0, True])),
             '"spread_band" must be [floor, ceiling]'),
            (_text(_with_toggles(rebalance=1)), "rebalance must be True or False"),
            (_text(_with_toggles(exclude_tags=_DROP)),
             "(rebalance, min_price_move, rebalance_min_days_to_maturity, sell_at, "
             "sell_min_days may be left out, and then each reads as off)"),
            (_text(_with_toggles(sell_at=True)), '"sell_at" must be a number or null'),
            (_text(_with_toggles(sell_at=0.85, sell_min_days=False)),
             '"sell_min_days" must be a number or null'),
            (_text(_with_toggles(sell_at=0.85, sell_min_days=1.0)),
             "sell_min_days must be a whole number of days"),
            (_text(_with_toggles(sell_min_days=3)), "sell_min_days needs a sell level"),
            (_text(_with_toggles(sell_at=0.855)), "sell_at must be a multiple of 1%"),
            (_text(_with_toggles(rebalance=True, min_price_move=True)),
             '"min_price_move" must be a number or null'),
            (_text(_with_toggles(rebalance=True, min_price_move=False)),
             '"min_price_move" must be a number or null'),
            (_text(_with_toggles(rebalance=True, min_price_move=0.07)),
             "min_price_move must be a multiple of 5c from 5c to $1"),
            (_text(_with_toggles(rebalance=True, min_price_move="0.10")),
             "min_price_move must be a number of dollars"),
            (_text(_with_toggles(min_price_move=0.10)), "min_price_move needs rebalance on"),
            (_text(_with_toggles(rebalance=False, min_price_move=0.10)),
             "min_price_move needs rebalance on"),
            (_text(_with_toggles(rebalance=True, rebalance_min_days_to_maturity=True)),
             '"rebalance_min_days_to_maturity" must be a number or null'),
            (_text(_with_toggles(rebalance=True, rebalance_min_days_to_maturity=False)),
             '"rebalance_min_days_to_maturity" must be a number or null'),
            (_text(_with_toggles(rebalance=True, rebalance_min_days_to_maturity=2)),
             "rebalance_min_days_to_maturity must be 0, 1 or 7, got 2"),
            (_text(_with_toggles(rebalance=True, rebalance_min_days_to_maturity=1.0)),
             "rebalance_min_days_to_maturity must be 0, 1 or 7, got 1.0"),
            (_text(_with_toggles(rebalance_min_days_to_maturity=7)),
             "rebalance_min_days_to_maturity needs rebalance on"),
            (_text(_with_toggles(add_to_held_pairs=True)),
             "which rebalance and min_price_move replaced"),
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
        assert config.read_saved_live_defaults() == _SEED_REBALANCE_OFF

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
        assert config.read_saved_live_defaults() == _SEED_REBALANCE_OFF

    def test_a_hand_written_record_reads_as_its_toggles(self):
        # Whole numbers and names with surrounding spaces are normalised by LiveSettings
        config.LIVE_DEFAULTS_FILE.write_bytes(_text(_with_toggles(
            spread_band=[0, 1], interval_discount=1, size_cap=1,
            exclude_categories=[" Sports "], exclude_tags=["Basketball"])))
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
        assert config.live_defaults() == _SEED_REBALANCE_OFF


# A file the previous version's writer wrote, in its exact layout: adding to
# held pairs on, the Kelly trim on with a 10c minimum price rise
_OLD_VERSION_BYTES = (
    b'{\n'
    b'  "format": "live-defaults-v1",\n'
    b'  "saved_at": "2026-10-11T18:22:05Z",\n'
    b'  "source": "backtest dashboard for 2025-10-11 to 2026-10-11",\n'
    b'  "settings": {\n'
    b'    "tier_floors": false,\n'
    b'    "spread_band": [0.0, 0.8],\n'
    b'    "interval_discount": 0.7,\n'
    b'    "size_cap": 0.15,\n'
    b'    "same_title_size_cap": 0.2,\n'
    b'    "categories": null,\n'
    b'    "tags": null,\n'
    b'    "add_to_held_pairs": true,\n'
    b'    "trim_to_kelly": true,\n'
    b'    "trim_min_rise": 0.1\n'
    b'  }\n'
    b'}\n')


def _outdated(data: bytes) -> config.LiveDefaultsOutdated:
    """
    Write data as the saved file and return the refusal its read raises.

    Args:
        data (bytes): The file's bytes.

    Returns:
        config.LiveDefaultsOutdated: What read_saved_live_defaults raised.
    """
    config.LIVE_DEFAULTS_FILE.write_bytes(data)
    with pytest.raises(config.LiveDefaultsOutdated) as err:
        config.read_saved_live_defaults()
    return err.value


class TestOutdatedSavedFile:
    """A saved file holding a setting an older version saved (add_to_held_pairs,
    trim_to_kelly, trim_min_rise) is refused as LiveDefaultsOutdated, so no run
    trades it, and carries what it reads as now (rebalance on only when both
    old switches were on, the old minimum then) for the defaults server to
    offer. Each test writes to its own path (conftest's _isolate_live_defaults)."""

    def test_the_old_file_is_refused_and_carries_its_new_form(self):
        # The previous writer's own bytes: both switches on, a 10c minimum
        err = _outdated(_OLD_VERSION_BYTES)
        assert isinstance(err, config.LiveDefaultsError)
        assert not isinstance(err, config.LiveDefaultsMissing)
        # The previous writer's file holds categories and tags too (null: any
        # category, any tag), which exclude_categories and exclude_tags replaced
        assert str(err) == (
            f'{config.LIVE_DEFAULTS_FILE}: "settings" holds add_to_held_pairs, trim_to_kelly '
            "and trim_min_rise, which rebalance and min_price_move replaced, and categories, "
            "which exclude_categories replaced, and tags, which exclude_tags replaced: save "
            "the live defaults again (start ./start_dashboard.sh; its pages offer them as "
            "rebalance yes, min price movement 10c, no category excluded, no tag excluded)")
        assert err.retired == ("add_to_held_pairs", "trim_to_kelly", "trim_min_rise",
                               "categories", "tags")
        # Every other setting as saved; the origin names the file as a read would
        assert err.proposal == LiveSettings(False, (0.0, 0.8), 0.7, 0.15, 0.2,
                                            rebalance=True, min_price_move=0.10)
        assert err.proposal.origin == (
            "live_defaults.json, saved 2026-10-11T18:22:05Z from backtest dashboard for "
            "2025-10-11 to 2026-10-11")

    def test_live_defaults_refuses_it_too(self):
        # The read every live run makes: refused, never the proposal
        config.LIVE_DEFAULTS_FILE.write_bytes(_OLD_VERSION_BYTES)
        with pytest.raises(config.LiveDefaultsOutdated) as err:
            config.live_defaults()
        assert err.value.proposal.rebalance is True

    def test_both_switches_on_with_no_minimum_proposes_no_minimum(self):
        err = _outdated(_text(_with_toggles(add_to_held_pairs=True, trim_to_kelly=True)))
        assert err.proposal == config.LIVE_DEFAULTS_SEED
        assert str(err).endswith("offer them as rebalance yes, min price movement any)")

    @pytest.mark.parametrize("old", [
        {"add_to_held_pairs": True},
        {"add_to_held_pairs": True, "trim_to_kelly": False},
        {"trim_to_kelly": True},
        {"trim_to_kelly": True, "trim_min_rise": 0.1},
        {"add_to_held_pairs": False, "trim_to_kelly": True, "trim_min_rise": 0.25},
        {"add_to_held_pairs": False},
        {"trim_to_kelly": False, "trim_min_rise": None},
    ])
    def test_only_one_old_switch_on_proposes_rebalancing_off(self, old):
        # Rebalance does both together, so one old switch alone is not it, and
        # the old minimum goes with it
        err = _outdated(_text(_with_toggles(**old)))
        assert err.proposal == _SEED_REBALANCE_OFF
        assert err.proposal.min_price_move is None
        assert str(err).endswith("offer them as rebalance no)")

    def test_the_other_settings_are_kept_as_saved(self):
        # Selling and the filter come through unchanged
        err = _outdated(_text(_with_toggles(
            add_to_held_pairs=True, trim_to_kelly=True, trim_min_rise=0.25, sell_at=0.85,
            sell_min_days=3, exclude_categories=["Politics"], exclude_tags=["Sports · Basketball"])))
        assert err.proposal == dataclasses.replace(
            config.LIVE_DEFAULTS_SEED, min_price_move=0.25, sell_at=0.85, sell_min_days=3,
            exclude_categories=("Politics",), exclude_tags=("Sports · Basketball",))

    @pytest.mark.parametrize("old, held", [
        ({"trim_to_kelly": False}, "trim_to_kelly"),
        ({"trim_min_rise": None}, "trim_min_rise"),
        ({"add_to_held_pairs": True, "trim_to_kelly": False},
         "add_to_held_pairs and trim_to_kelly"),
    ])
    def test_the_message_names_only_the_old_settings_the_file_holds(self, old, held):
        err = _outdated(_text(_with_toggles(**old)))
        assert f'"settings" holds {held}, which rebalance and min_price_move replaced' in str(err)

    @pytest.mark.parametrize("label, old, reason", [
        ("a cap off the grid", {"add_to_held_pairs": True, "size_cap": 0.07},
         "size_cap must be a multiple of 5%"),
        ("a missing setting", {"add_to_held_pairs": True, "exclude_tags": _DROP},
         '"settings" must hold exactly'),
        ("an unknown setting", {"add_to_held_pairs": True, "sell_held_pairs": True},
         '"settings" must hold exactly'),
        ("a band of booleans", {"add_to_held_pairs": True, "spread_band": [False, True]},
         '"spread_band" must be [floor, ceiling]'),
        ("an old switch as 1", {"add_to_held_pairs": 1}, '"add_to_held_pairs" must be true'),
        ("an old switch as null", {"trim_to_kelly": None}, '"trim_to_kelly" must be true'),
        ("an old minimum off the grid", {"trim_to_kelly": True, "trim_min_rise": 0.07},
         "trim_min_rise must be a multiple of 5c"),
        ("an old minimum of true", {"trim_to_kelly": True, "trim_min_rise": True},
         '"trim_min_rise" must be a number or null'),
        ("an old minimum with trimming off",
         {"add_to_held_pairs": True, "trim_to_kelly": False, "trim_min_rise": 0.1},
         '"trim_min_rise" needs trim_to_kelly on'),
        ("old and new names together", {"add_to_held_pairs": True, "rebalance": True},
         "it also holds rebalance, min_price_move or rebalance_min_days_to_maturity"),
        ("an old switch beside a minimum move", {"trim_to_kelly": True, "min_price_move": 0.1},
         "it also holds rebalance, min_price_move or rebalance_min_days_to_maturity"),
        ("an old switch beside the days to maturity",
         {"add_to_held_pairs": True, "trim_to_kelly": True,
          "rebalance_min_days_to_maturity": 7},
         "it also holds rebalance, min_price_move or rebalance_min_days_to_maturity"),
    ])
    def test_a_broken_setting_beside_an_old_one_leaves_no_proposal(self, label, old, reason):
        # Still refused as outdated, but with nothing to offer: the message says
        # why and names the other ways to save
        err = _outdated(_text(_with_toggles(**old)))
        assert err.proposal is None, label
        message = str(err)
        assert "and its other settings are refused too (" in message, label
        assert reason in message, (label, message)
        assert "offer them" not in message
        assert message.endswith("save new ones from its pages, from the seed values or the "
                                "backtest dashboard)")

    def test_the_record_rules_come_first(self):
        # A file that breaks a rule outside its settings is an ordinary refusal
        record = json.loads(_OLD_VERSION_BYTES)
        record["format"] = "live-defaults-v2"
        config.LIVE_DEFAULTS_FILE.write_bytes(_text(record))
        with pytest.raises(config.LiveDefaultsError) as err:
            config.read_saved_live_defaults()
        assert not isinstance(err.value, config.LiveDefaultsOutdated)
        assert '"format" must be' in str(err.value)

    def test_one_save_of_the_proposal_replaces_the_file(self):
        # What the defaults server's Confirm and save does: the new file holds
        # no old setting and every run reads it
        proposal = _outdated(_OLD_VERSION_BYTES).proposal
        saved = config.save_live_defaults(proposal, source="")
        assert saved == proposal
        assert config.read_saved_live_defaults() == proposal
        text = config.LIVE_DEFAULTS_FILE.read_text(encoding="utf-8")
        stored = json.loads(text)["settings"]
        assert not any(name in stored for name in config._RETIRED_TOGGLES)
        assert '"rebalance": true,\n    "min_price_move": 0.1\n' in text


class TestCategoriesRetirement:
    """A saved file holding "categories" (a list of categories to trade, which
    exclude_categories replaced) is refused as LiveDefaultsOutdated. A null
    value (any category) reads exactly as no category excluded; a list of
    names has no exact reading (which categories it leaves out depends on
    every category Kalshi lists), so its proposal excludes none too, and
    old_categories names the list, so every page that offers the proposal
    can say it widens live trading."""

    @staticmethod
    def _current_era(**changes) -> bytes:
        """A file the version before this one wrote: categories in place of
        exclude_categories, with rebalance and selling saved."""
        record = _with_toggles(rebalance=True, min_price_move=0.1, sell_at=0.9,
                               sell_min_days=2, **changes)
        settings = record["settings"]
        settings["categories"] = settings.pop("exclude_categories")
        # That version's tags were tags to trade, under the old key
        settings["tags"] = settings.pop("exclude_tags")
        return _text(record)

    def test_a_null_list_proposes_no_category_excluded_and_everything_else(self):
        # That version's file holds tags too (null: any tag), which exclude_tags
        # replaced: both retirements at once
        err = _outdated(self._current_era())
        assert err.retired == ("categories", "tags")
        assert err.proposal == dataclasses.replace(
            config.LIVE_DEFAULTS_SEED, min_price_move=0.1, sell_at=0.9, sell_min_days=2,
            exclude_categories=None)
        assert str(err) == (
            f'{config.LIVE_DEFAULTS_FILE}: "settings" holds categories, which '
            "exclude_categories replaced, and tags, which exclude_tags replaced: save the "
            "live defaults again (start ./start_dashboard.sh; its pages offer them as no "
            "category excluded, no tag excluded)")
        # live_defaults refuses it too: no run trades the proposal
        with pytest.raises(config.LiveDefaultsOutdated):
            config.live_defaults()

    def test_a_null_list_with_the_rebalance_settings_applies_both(self):
        # The Rebalance-era writer's bytes: both retirements at once
        err = _outdated(_OLD_VERSION_BYTES)
        assert err.proposal.exclude_categories is None and err.proposal.rebalance is True
        assert err.proposal.min_price_move == 0.10

    @pytest.mark.parametrize("names", [["Sports"], ["Economics", " Politics "]])
    def test_a_list_of_names_proposes_no_exclusion_and_names_the_old_list(self, names):
        # A list of categories to trade has no exact reading as categories to
        # exclude: the proposal is the file's other settings with nothing
        # excluded, and old_categories says it widens live trading
        err = _outdated(self._current_era(exclude_tags=None).replace(
            b'"categories": null', b'"categories": ' + json.dumps(names).encode()))
        old = tuple(name.strip() for name in names)
        assert err.old_categories == old
        assert err.proposal == dataclasses.replace(
            config.LIVE_DEFAULTS_SEED, min_price_move=0.1, sell_at=0.9, sell_min_days=2,
            exclude_categories=None)
        assert str(err) == (
            f'{config.LIVE_DEFAULTS_FILE}: "settings" holds categories, which '
            "exclude_categories replaced, and tags, which exclude_tags replaced: these "
            f"defaults traded only {', '.join(repr(name) for name in old)}, and this "
            "version saves what NOT to trade instead. Saved again as offered (no category "
            "excluded, no tag excluded), they trade everything: exclude each category not "
            "to trade before saving (start ./start_dashboard.sh, or the backtest "
            "dashboard's \"Save as live defaults…\" button)")

    def test_a_null_list_has_no_old_categories(self):
        assert _outdated(self._current_era()).old_categories is None

    @pytest.mark.parametrize("value", [
        b'"Sports"', b"3", b"[]", b'["Sports", "any"]', b'["Sports", ""]', b"[1]",
        b'["Sports \xc2\xb7 Basketball"]', b"{}",
    ])
    def test_a_malformed_list_has_no_proposal(self, value):
        # What the older version refused has no reading now either
        err = _outdated(self._current_era(exclude_tags=None).replace(
            b'"categories": null', b'"categories": ' + value))
        assert err.proposal is None and err.old_categories is None
        assert "and its other settings are refused too" in str(err)

    def test_a_tied_tag_with_null_categories_has_no_proposal(self):
        # The older version accepted a tied tag only with its category listed,
        # so null categories with one was a file it refused
        err = _outdated(self._current_era(exclude_tags=["Sports · Basketball"]))
        assert err.proposal is None and err.old_categories is None
        assert "is tied to 'Sports', which its categories did not list" in str(err)

    def test_a_tied_tag_needs_its_category_listed_as_the_old_version_did(self):
        listed = self._current_era(exclude_tags=["sports · Basketball"]).replace(
            b'"categories": null', b'"categories": ["Sports"]')
        err = _outdated(listed)
        # Tags to trade too, which exclude_tags replaced: offered as excluding none
        assert err.proposal.exclude_tags is None
        assert err.proposal.exclude_categories is None
        assert err.old_categories == ("Sports",)
        assert err.old_tags == ("sports · Basketball",)
        unlisted = self._current_era(exclude_tags=["Sports · Basketball"]).replace(
            b'"categories": null', b'"categories": ["Economics"]')
        assert _outdated(unlisted).proposal is None

    def test_a_list_of_names_beside_the_rebalance_settings_applies_both(self):
        record = json.loads(_OLD_VERSION_BYTES)
        record["settings"]["categories"] = ["Sports"]
        err = _outdated(_text(record))
        assert err.old_categories == ("Sports",)
        assert err.proposal.rebalance is True and err.proposal.exclude_categories is None
        assert ('"settings" holds add_to_held_pairs, trim_to_kelly and trim_min_rise, '
                "which rebalance and min_price_move replaced, and categories, which "
                "exclude_categories replaced, and tags, which exclude_tags replaced: these "
                "defaults traded only 'Sports'") in str(err)
        assert ("Saved again as offered (rebalance yes, min price movement 10c, no "
                "category excluded, no tag excluded), they trade everything") in str(err)

    def test_both_filters_together_are_refused_with_no_proposal(self):
        record = _with_toggles()
        record["settings"]["categories"] = None
        err = _outdated(_text(record))
        assert err.proposal is None
        assert ("and its other settings are refused too (it also holds exclude_categories, "
                "which no version saved beside categories)") in str(err)

    def test_the_rebalance_conflict_applies_only_to_the_rebalance_settings(self):
        # A file of the version before this one holds rebalance beside categories:
        # not two things at once, since rebalance replaced nothing there
        err = _outdated(self._current_era())
        assert err.proposal is not None and err.proposal.rebalance is True

    def test_one_save_of_the_proposal_replaces_the_file(self):
        proposal = _outdated(self._current_era()).proposal
        assert config.save_live_defaults(proposal, source="") == proposal
        stored = json.loads(config.LIVE_DEFAULTS_FILE.read_text(encoding="utf-8"))["settings"]
        assert "categories" not in stored and stored["exclude_categories"] is None
        assert config.read_saved_live_defaults() == proposal


class TestTagsRetirement:
    """A saved file holding "tags" (a list of tags to trade, which exclude_tags
    replaced) is refused as LiveDefaultsOutdated, as categories was. A null
    value (any tag) reads exactly as no tag excluded; a list of names has no
    exact reading (which tags it leaves out depends on every tag Kalshi
    lists), so its proposal excludes none too, and old_tags names the list,
    so every page that offers the proposal can say it widens live trading.
    Every file the version before this one saved holds "tags", since it was
    required."""

    @staticmethod
    def _current_era(**changes) -> bytes:
        """A file the version before this one wrote: exclude_categories, and
        tags (tags to trade) in place of exclude_tags, with rebalance and
        selling saved."""
        record = _with_toggles(rebalance=True, min_price_move=0.1, sell_at=0.9,
                               sell_min_days=2, **changes)
        settings = record["settings"]
        settings["tags"] = settings.pop("exclude_tags")
        return _text(record)

    @staticmethod
    def _with_tags(value: bytes, **changes) -> bytes:
        """That file with its tags set to value (raw JSON)."""
        return TestTagsRetirement._current_era(**changes).replace(
            b'"tags": null', b'"tags": ' + value)

    def test_a_null_list_proposes_no_tag_excluded_and_everything_else(self):
        err = _outdated(self._current_era(exclude_categories=["Politics"]))
        assert err.retired == ("tags",)
        assert err.proposal == dataclasses.replace(
            config.LIVE_DEFAULTS_SEED, min_price_move=0.1, sell_at=0.9, sell_min_days=2,
            exclude_categories=("Politics",), exclude_tags=None)
        assert err.old_tags is None and err.old_categories is None
        assert str(err) == (
            f'{config.LIVE_DEFAULTS_FILE}: "settings" holds tags, which exclude_tags '
            "replaced: save the live defaults again (start ./start_dashboard.sh; its pages "
            "offer them as no tag excluded)")
        # live_defaults refuses it too: no run trades the proposal
        with pytest.raises(config.LiveDefaultsOutdated):
            config.live_defaults()

    @pytest.mark.parametrize("names", [["Basketball"], ["Sports · Basketball", " Fed "]])
    def test_a_list_of_names_proposes_no_exclusion_and_names_the_old_list(self, names):
        err = _outdated(self._with_tags(json.dumps(names).encode()))
        old = tuple(name.strip() for name in names)
        assert err.old_tags == old and err.old_categories is None
        assert err.proposal == dataclasses.replace(
            config.LIVE_DEFAULTS_SEED, min_price_move=0.1, sell_at=0.9, sell_min_days=2,
            exclude_tags=None)
        assert str(err) == (
            f'{config.LIVE_DEFAULTS_FILE}: "settings" holds tags, which exclude_tags '
            "replaced: these defaults limited trading to the tags "
            f"{', '.join(repr(name) for name in old)} (a 'Category · Tag' limited that "
            "category alone), and this version saves what NOT to "
            "trade instead. Saved again as offered (no tag excluded), they trade "
            "everything: exclude each tag not to trade before saving (start "
            "./start_dashboard.sh, or the backtest dashboard's \"Save as live "
            "defaults…\" button)")

    @pytest.mark.parametrize("value", [
        b'"Basketball"', b"3", b"[]", b'["Basketball", "any"]', b'["Basketball", "NONE"]',
        b'["Basketball", ""]', b"[1]", b"{}", b'["Sports \xc2\xb7 any"]',
        b'["Sports \xc2\xb7 "]', b'["\xc2\xb7 Basketball"]', b'["\xc2\xb7"]',
        b'["Basket\\u202eball"]',
    ])
    def test_a_malformed_list_has_no_proposal(self, value):
        # What the older version refused has no reading now either
        err = _outdated(self._with_tags(value))
        assert err.proposal is None and err.old_tags is None
        assert "and its other settings are refused too" in str(err)

    def test_a_tied_tag_in_an_excluded_category_has_no_proposal(self):
        # The older version refused a tied tag whose category it excluded
        err = _outdated(self._with_tags(b'["sports \xc2\xb7 Basketball"]',
                                        exclude_categories=["Sports"]))
        assert err.proposal is None and err.old_tags is None
        assert "is tied to 'sports', which its exclude_categories excluded" in str(err)
        # ... while one in a category it did not exclude was a file it read
        err = _outdated(self._with_tags(b'["Sports \xc2\xb7 Basketball"]',
                                        exclude_categories=["Politics"]))
        assert err.old_tags == ("Sports · Basketball",)
        assert err.proposal.exclude_categories == ("Politics",)

    def test_an_older_file_with_categories_took_the_name_none(self):
        # The version with categories refused only "any"; its tied-tag rule
        # is categories' own (_old_categories)
        record = json.loads(_OLD_VERSION_BYTES)
        record["settings"]["tags"] = ["None", "Fed"]
        err = _outdated(_text(record))
        assert err.old_tags == ("None", "Fed")
        assert err.proposal.exclude_tags is None

    def test_both_old_lists_are_named(self):
        record = json.loads(_OLD_VERSION_BYTES)
        record["settings"]["categories"] = ["Sports", "Economics"]
        record["settings"]["tags"] = ["Sports · Basketball", "Fed"]
        err = _outdated(_text(record))
        assert err.old_categories == ("Sports", "Economics")
        assert err.old_tags == ("Sports · Basketball", "Fed")
        assert err.proposal.exclude_categories is None and err.proposal.exclude_tags is None
        assert ("these defaults traded only the categories 'Sports', 'Economics' and "
                "limited trading to the tags 'Sports · Basketball', 'Fed' (a 'Category · "
                "Tag' limited that category alone), and this version saves what NOT to "
                "trade instead") in str(err)
        assert "exclude each category and tag not to trade before saving" in str(err)

    def test_both_filters_together_are_refused_with_no_proposal(self):
        record = _with_toggles()
        record["settings"]["tags"] = None
        err = _outdated(_text(record))
        assert err.proposal is None
        assert ("and its other settings are refused too (it also holds exclude_tags, "
                "which no version saved beside tags)") in str(err)

    @pytest.mark.parametrize("null", [True, False])
    def test_categories_beside_exclude_tags_is_refused_with_no_proposal(self, null):
        # No version saved categories (the oldest filter) beside exclude_tags,
        # as none saved tags beside exclude_tags: such a file says two things
        # at once and is offered nothing, whatever its categories says
        record = _with_toggles(exclude_tags=["Sports · Basketball"])
        record["settings"]["categories"] = None if null else ["Sports"]
        del record["settings"]["exclude_categories"]
        err = _outdated(_text(record))
        assert err.retired == ("categories",)
        assert err.proposal is None and err.old_categories is None
        assert ("and its other settings are refused too (it also holds exclude_tags, "
                "which no version saved beside categories)") in str(err)

    def test_every_filter_era_a_version_saved_still_has_its_proposal(self):
        # categories beside tags (the oldest filter): both retirements
        record = json.loads(_OLD_VERSION_BYTES)
        record["settings"]["categories"] = ["Sports"]
        record["settings"]["tags"] = ["Basketball"]
        err = _outdated(_text(record))
        assert err.retired == ("add_to_held_pairs", "trim_to_kelly", "trim_min_rise",
                               "categories", "tags")
        assert err.proposal is not None
        assert err.old_categories == ("Sports",) and err.old_tags == ("Basketball",)
        # exclude_categories beside tags (the version before this one)
        err = _outdated(self._with_tags(b'["Basketball"]', exclude_categories=["Politics"]))
        assert err.retired == ("tags",)
        assert err.proposal.exclude_categories == ("Politics",)
        assert err.old_tags == ("Basketball",)
        # The Rebalance settings are not part of the filter: beside this
        # version's filter keys they still read as saved
        err = _outdated(_text(_with_toggles(add_to_held_pairs=True, trim_to_kelly=True,
                                            exclude_tags=["Sports · Basketball"])))
        assert err.proposal.exclude_tags == ("Sports · Basketball",)

    def test_one_save_of_the_proposal_replaces_the_file(self):
        proposal = _outdated(self._with_tags(b'["Basketball"]')).proposal
        assert config.save_live_defaults(proposal, source="") == proposal
        stored = json.loads(config.LIVE_DEFAULTS_FILE.read_text(encoding="utf-8"))["settings"]
        assert "tags" not in stored and stored["exclude_tags"] is None
        assert config.read_saved_live_defaults() == proposal


class TestLiveDefaultsSeed:
    """LIVE_DEFAULTS_SEED, the starting values offered for a first save."""

    def test_the_seed_values(self):
        seed = config.LIVE_DEFAULTS_SEED
        assert seed == LiveSettings(False, (0.0, 0.5), 0.8, 0.10, 0.20, rebalance=True)
        assert seed.min_price_move is None
        assert seed.exclude_categories is None and seed.exclude_tags is None
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
        current = _settings(exclude_categories=("Sports",), exclude_tags=("Basketball",))
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
            ("exclude categories", "Sports", "none", True),
            ("exclude tags", "Basketball", "none", True),
            ("rebalance", "no", "no", False),
            ("min price movement", "any", "any", False),
            ("rebalance until", "maturity", "maturity", False),
            ("sell at", "off", "off", False),
            ("min days to maturity", "any", "any", False),
        ]

    def test_changed_flags_exactly_the_fields_that_differ(self):
        ref = TestDescribeLiveSettings._REFERENCE
        for name in config.LIVE_TOGGLE_FIELDS:
            value, _ = TestDescribeLiveSettings._DEPARTURES[name]
            rows = config.live_settings_changes(ref, dataclasses.replace(ref, **{name: value}))
            assert [changed for *_, changed in rows] == [
                other == name for other in config.LIVE_TOGGLE_FIELDS], name
        # Equal toggles change nothing, whatever the origins say
        rows = config.live_settings_changes(dataclasses.replace(ref, origin=_SAVED_ORIGIN), ref)
        assert not any(changed for *_, changed in rows)
        # Case is part of the raw value, as on the "Live settings:" line
        rows = config.live_settings_changes(_settings(exclude_categories=("Sports",)),
                                            _settings(exclude_categories=("sports",)))
        assert rows[5] == ("exclude categories", "Sports", "sports", True)

    def test_no_current_defaults_changes_every_row(self):
        rows = config.live_settings_changes(None, config.LIVE_DEFAULTS_SEED)
        assert len(rows) == len(config.LIVE_TOGGLE_FIELDS) == 12
        assert all(old == "—" and changed for _, old, _, changed in rows)
        assert [new for _, _, new, _ in rows] == [
            "off", "0-0.5", "0.8", "10%", "20%", "none", "none", "yes", "any", "maturity",
            "off", "any"]

    def test_turning_selling_on_shows_as_the_two_sell_rows(self):
        on = dataclasses.replace(config.LIVE_DEFAULTS_SEED, sell_at=0.85, sell_min_days=3)
        rows = config.live_settings_changes(config.LIVE_DEFAULTS_SEED, on)
        assert [row for row in rows if row[3]] == [
            ("sell at", "off", "85% of potential profit", True),
            ("min days to maturity", "any", "3", True)]
        # ... and back off again
        rows = config.live_settings_changes(on, config.LIVE_DEFAULTS_SEED)
        assert [row for row in rows if row[3]] == [
            ("sell at", "85% of potential profit", "off", True),
            ("min days to maturity", "3", "any", True)]

    def test_turning_rebalancing_on_shows_as_the_two_rebalance_rows(self):
        on = dataclasses.replace(config.LIVE_DEFAULTS_SEED, min_price_move=0.10)
        rows = config.live_settings_changes(_SEED_REBALANCE_OFF, on)
        assert [row for row in rows if row[3]] == [
            ("rebalance", "no", "yes", True),
            ("min price movement", "any", "10c", True)]
        # ... and back off again
        rows = config.live_settings_changes(on, _SEED_REBALANCE_OFF)
        assert [row for row in rows if row[3]] == [
            ("rebalance", "yes", "no", True),
            ("min price movement", "10c", "any", True)]
        # A minimum over rebalancing with none changes the one row
        rows = config.live_settings_changes(config.LIVE_DEFAULTS_SEED, on)
        assert [row for row in rows if row[3]] == [("min price movement", "any", "10c", True)]

    def test_choosing_a_rebalance_until_shows_as_its_one_row(self):
        seed = config.LIVE_DEFAULTS_SEED
        for days, text in ((1, "1 day before maturity"), (7, "7 days before maturity")):
            on = dataclasses.replace(seed, rebalance_min_days_to_maturity=days)
            rows = config.live_settings_changes(seed, on)
            assert [row for row in rows if row[3]] == [
                ("rebalance until", "maturity", text, True)]
            # ... and back to maturity again
            rows = config.live_settings_changes(on, seed)
            assert [row for row in rows if row[3]] == [
                ("rebalance until", text, "maturity", True)]
        # Turning rebalancing off clears it: both rows change
        on = dataclasses.replace(seed, rebalance_min_days_to_maturity=7)
        rows = config.live_settings_changes(on, _SEED_REBALANCE_OFF)
        assert [row for row in rows if row[3]] == [
            ("rebalance", "yes", "no", True),
            ("rebalance until", "7 days before maturity", "maturity", True)]

    def test_a_seven_toggle_file_shows_turning_rebalancing_on_as_a_change(self):
        # A file that leaves the toggle out reads it as off; the seed proposes on
        rows = config.live_settings_changes(_SEED_REBALANCE_OFF, config.LIVE_DEFAULTS_SEED)
        assert [row for row in rows if row[3]] == [("rebalance", "no", "yes", True)]


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


class TestFeeMultipliers:
    """Each leg's fee is scaled by its own market's series fee multiplier, but
    only for a multiplier config.CONFIRMED_FEE_MULTIPLIERS holds; with the
    multipliers at 1.0 both fee functions are the formula they always were."""

    @staticmethod
    def _old_exact(n, p):
        # The parent commit's fee_leg_exact, verbatim
        return math.ceil(round(config.TAKER_FEE_RATE * n * p * (1.0 - p) * 100, 6)) / 100

    @staticmethod
    def _old_approx(price_a, price_b):
        # The parent commit's fee_per_pair_approx, verbatim
        return config.TAKER_FEE_RATE * (price_a * (1.0 - price_a) + price_b * (1.0 - price_b))

    def test_only_the_full_fee_ships_confirmed(self):
        assert config.CONFIRMED_FEE_MULTIPLIERS == frozenset({1.0})

    def test_bit_identical_at_one_over_a_sweep(self):
        rng = random.Random(406)
        prices = [i / 10_000 for i in range(1, 10_000, 37)] + [rng.random() for _ in range(500)]
        for p in prices:
            for n in (1, 2, 3, 7, 10, 31, 100, 748, 1_000, 25_000):
                assert config.fee_leg_exact(n, p, multiplier=1.0) == self._old_exact(n, p)
                assert config.fee_leg_exact(n, p) == self._old_exact(n, p)
        for _ in range(5_000):
            a, b = rng.random(), rng.random()
            assert config.fee_per_pair_approx(a, b, multiplier_a=1.0, multiplier_b=1.0) \
                == self._old_approx(a, b)

    def test_a_half_multiplier_halves_the_rate(self):
        # 0.07 x 0.5 x 10 x 0.25 = 0.0875, rounded up to the cent
        assert config.fee_leg_exact(10, 0.5, multiplier=0.5) == 0.09
        assert config.fee_leg_exact(10, 0.5) == 0.18
        assert config.fee_per_pair_approx(0.5, 0.5, multiplier_a=0.5, multiplier_b=1.0) \
            == pytest.approx(0.07 * (0.5 * 0.25 + 0.25))
        # Each multiplier goes with its own price, not the other one
        assert config.fee_per_pair_approx(0.1, 0.5, multiplier_a=0.0, multiplier_b=1.0) \
            == pytest.approx(0.07 * 0.25)
        assert config.fee_per_pair_approx(0.1, 0.5, multiplier_a=1.0, multiplier_b=0.0) \
            == pytest.approx(0.07 * 0.09)

    def test_a_zero_multiplier_charges_nothing(self):
        assert config.fee_leg_exact(1_000, 0.37, multiplier=0.0) == 0.0
        assert config.fee_per_pair_approx(0.3, 0.4, multiplier_a=0.0, multiplier_b=0.0) == 0.0

    def test_fee_multiplier_charges_an_unconfirmed_one_in_full(self, monkeypatch):
        for raw in (0.5, 0, 0.0, 2.0, float("nan"), float("inf"), None, "0.5", True, False,
                    MagicMock(), [0.5]):
            assert config.confirmed_fee_multiplier(raw) == 1.0, raw
        assert config.confirmed_fee_multiplier(1) == 1.0
        monkeypatch.setattr(config, "CONFIRMED_FEE_MULTIPLIERS", frozenset({1.0, 0.5, 0.0}))
        assert config.confirmed_fee_multiplier(0.5) == 0.5
        assert config.confirmed_fee_multiplier(0) == 0.0
        assert config.confirmed_fee_multiplier(0.25) == 1.0
        assert config.confirmed_fee_multiplier(False) == 1.0

    def test_the_unconfirmed_count_line(self, monkeypatch):
        assert config.unconfirmed_fee_multipliers_text([1.0, 1.0, None]) is None
        assert config.unconfirmed_fee_multipliers_text(
            [1.0, 0.5, 0.0, 0.0, 0.5, 0.0, "x", True]) == (
            "Fee multipliers listed but not confirmed (charged in full): 5 markets "
            "(multipliers: 0.5 x 2, 0 x 3)")
        # A confirmed multiplier is not counted
        monkeypatch.setattr(config, "CONFIRMED_FEE_MULTIPLIERS", frozenset({1.0, 0.5}))
        assert config.unconfirmed_fee_multipliers_text([0.5, 0.0]) == (
            "Fee multipliers listed but not confirmed (charged in full): 1 markets "
            "(multipliers: 0 x 1)")
