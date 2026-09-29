"""
File: test_main.py
Author: Zachary Hoffman
Last edited by: Zachary Hoffman

Purpose:
    Offline tests for kalshi_betting.main — the live-pipeline orchestrator's
    pure helpers (_truncate, _format_deadline, _dedup_pairs,
    _compute_trade_specs, _no_pairs_msg), its logging setup (_setup_logging,
    including the BS-25 rotating file handler), its process exit-code contract
    (BS-14), and end-to-end "live-shape replay" runs of _run_dev/_run_prod
    against a MagicMock client wired with CURRENT-generation (2026-08+) Kalshi
    payload shapes — dollar-string prices, yes_sub_title, orderbook_fp books,
    shard-aware balance_breakdown, position_fp positions, V2 order responses —
    the closest offline substitute for a live smoke test.

    On the exit-code half: _run_prod / _run_dev return an int outcome code and
    main() propagates it to the OS via sys.exit(), so the scheduler (a separate
    subprocess, see scheduler.run_job) can distinguish a clean run from a
    low-balance skip or a run whose trades need manual review.

    On the live toggles: TestLiveSettingsFlags, TestLogLiveSettings and
    TestLiveSettingsReachEverySite (the runtime half of test_strategy.py's
    live-toggle AST pin) run under pinned_config_toggles; TestCategoryFilter
    covers main._filter_by_category.

Dependencies:
    Imports _run_dev/_run_prod and the pure helpers from kalshi_betting.main,
    plus config constants asserted against and conftest's
    apply_pre_toggle_defaults; for the category/tag filter, historical,
    dashboard and backtester.BacktestTrade. The live-shape replays mock all
    Kalshi API interaction at the HTTP boundary (raw-response mocks and
    rest_client.request); the exit-code tests mock the heavy collaborators
    (auth, scanner, strategy, trader, reporter) at their main-module import
    sites per project policy. Reporter Excel writers are patched out so no
    files are written, and PROJECT_ROOT is redirected to tmp_path wherever
    main() configures logging, so the real kalshi_arb.log / trade_log.xlsx are
    never touched.

Notes:
    The V2 live-execution replays run with dry_run=False and
    config.ORDER_API_VERSION left at its shipped "v2". Order responses are
    generated from each request's own submitted count, so the tests don't
    depend on exact Kelly sizing.

    verify_auth() returns dict[int, int] (exchange_index -> cents), never a
    scalar — every mock of it here must return a dict, and _run_prod sizes on
    sum(...) of it.
"""
import ast
import dataclasses
import inspect
import json
import logging
import logging.handlers
import pathlib
import re
import sys
from datetime import UTC, date, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from kalshi_betting import config, dashboard, historical, main
from kalshi_betting import scanner as scanner_mod
from kalshi_betting import strategy as strategy_mod
from kalshi_betting import trader as trader_mod
from kalshi_betting.backtester import BacktestTrade
from kalshi_betting.config import (
    DEFAULT_EXCHANGE_INDEX,
    EXIT_NO_TRADEABLE_SHARDS,
    EXIT_OK,
    EXIT_SKIPPED_LOW_BALANCE,
    EXIT_TIME_SERIES_SKIPPED,
    EXIT_TRADES_NEED_ATTENTION,
    MIN_BALANCE_CENTS,
    MIN_PRICE_DIFF_LONG_GAP,
    MIN_PRICE_DIFF_SHORT_GAP,
    ORDER_API_VERSION,
    SAME_TITLE_MAX_CLOSE_GAP_SECONDS,
    SAME_TITLE_MIN_PRICE_DIFF,
    TRANSFER_PATH,
    V2_ORDER_PATH,
    LiveSettings,
    describe_live_settings,
    live_settings,
)
from kalshi_betting.historical import infer_category
from kalshi_betting.reporter import TradeResult

from .conftest import apply_pre_toggle_defaults


def make_pair(ticker_a: str, ticker_b: str, pair_type: str = "time_series"):
    """Minimal stand-in for a CandidatePair — only the attributes
    _dedup_pairs actually reads (market_a.ticker, market_b.ticker)."""
    return SimpleNamespace(
        market_a=SimpleNamespace(ticker=ticker_a),
        market_b=SimpleNamespace(ticker=ticker_b),
        pair_type=pair_type,
    )


class TestTruncate:
    def test_truncate_boundary(self):
        # Exactly n chars: no truncation, no ellipsis appended
        text_at_boundary = "x" * 40
        assert main._truncate(text_at_boundary) == text_at_boundary

        # One char over: truncated to n chars plus the ellipsis marker
        text_over_boundary = "x" * 41
        result = main._truncate(text_over_boundary)
        assert result == "x" * 40 + "…"
        assert len(result) == 41


class TestFormatDeadline:
    def test_format_deadline_none(self):
        assert main._format_deadline(None) == "?"


class TestDedupPairs:
    def test_dedup_pairs_prefers_same_title(self):
        # Same ticker pair detected by both scanners — the same-title (primary)
        # entry must win, and the time-series (secondary) duplicate must be dropped.
        same_title = make_pair("TICK-A", "TICK-B", pair_type="same_title")
        time_series_dup = make_pair("TICK-A", "TICK-B", pair_type="time_series")

        result = main._dedup_pairs([same_title], [time_series_dup])

        assert result == [same_title]

    def test_dedup_pairs_preserves_order_and_appends_unique_secondary(self):
        primary_1 = make_pair("A1", "A2", pair_type="same_title")
        primary_2 = make_pair("B1", "B2", pair_type="same_title")
        # Duplicate of primary_1's ticker pair — must be dropped, order-independent of tickers
        secondary_dup = make_pair("A2", "A1", pair_type="time_series")
        # Genuinely unique pair — must be appended after all primary entries
        secondary_unique = make_pair("C1", "C2", pair_type="time_series")

        result = main._dedup_pairs(
            [primary_1, primary_2], [secondary_dup, secondary_unique],
        )

        assert result == [primary_1, primary_2, secondary_unique]


class TestComputeTradeSpecs:
    def test_compute_trade_specs_excludes_none(self, monkeypatch):
        pair_ok = make_pair("A1", "A2")
        pair_none = make_pair("B1", "B2")

        settings = live_settings()
        seen = []

        def fake_compute_trade(pair, balance_cents, *, settings):
            # Only pair_ok produces a spec — pair_none has no edge (returns None)
            seen.append(settings)
            if pair is pair_ok:
                return SimpleNamespace(pair=pair)
            return None

        monkeypatch.setattr(main, "compute_trade", fake_compute_trade)

        specs = main._compute_trade_specs([pair_ok, pair_none], balance_cents=100_000,
                                          settings=settings)

        assert list(specs.keys()) == [id(pair_ok)]
        assert specs[id(pair_ok)].pair is pair_ok
        # Every pair is sized under the ONE settings object the run handed in
        assert len(seen) == 2 and all(s is settings for s in seen)


class TestNoPairsMsg:
    def test_no_pairs_log_lines_format_thresholds_from_config(self):
        # A tier-on rule, the only one whose wording names the tier thresholds
        tiers_on = LiveSettings(True, (0.0, 1.0), 0.75, 0.2)
        prod_msg = main._no_pairs_msg(settings=tiers_on)
        dev_msg = main._no_pairs_msg(sandbox=True, settings=tiers_on)

        for msg in (prod_msg, dev_msg):
            assert f"{MIN_PRICE_DIFF_SHORT_GAP:.0%}" in msg
            assert f"{MIN_PRICE_DIFF_LONG_GAP:.0%}" in msg
            assert f"{SAME_TITLE_MIN_PRICE_DIFF:.0%}" in msg
            # DR-74: the same-title rule also needs two different series whose
            # markets close within SAME_TITLE_MAX_CLOSE_GAP_SECONDS — named in
            # the message, with the minutes derived from the constant.
            assert (
                f"on two different series closing within "
                f"{SAME_TITLE_MAX_CLOSE_GAP_SECONDS // 60} minutes"
            ) in msg

        assert "sandbox" in dev_msg
        assert "sandbox" not in prod_msg

    def test_the_close_gap_clause_follows_the_constant(self, monkeypatch):
        # Derived, not spelled: a different bound changes the message.
        monkeypatch.setattr(main, "SAME_TITLE_MAX_CLOSE_GAP_SECONDS", 15 * 60)
        assert "closing within 15 minutes" in main._no_pairs_msg()
        assert "closing within 60 minutes" not in main._no_pairs_msg()

    def test_the_time_series_clause_is_the_runs_own_rule(self):
        # The rule the run applied, never the tiers alone when it dropped them
        on = main._no_pairs_msg(settings=LiveSettings(True, (0.0, 1.0), 0.75, 0.2))
        off = main._no_pairs_msg(settings=LiveSettings(False, (0.0, 0.5), 0.75, 0.2))
        assert config.describe_time_series_rule(True, (0.0, 1.0)) in on
        assert config.describe_time_series_rule(False, (0.0, 0.5)) in off
        assert "tier floors off" in off and "spread band 0-0.5" in off
        assert f"≥{MIN_PRICE_DIFF_SHORT_GAP:.0%}" not in off
        # The same-title clause is the same under any time-series rule
        assert off.endswith(on[on.index(" — or same-title"):])

    def test_no_settings_reads_config_at_call_time(self, monkeypatch):
        # Both values, so the test cannot pass on whichever one config.py ships
        monkeypatch.setattr(config, "TIME_SERIES_TIER_FLOORS", False)
        assert "tier floors off" in main._no_pairs_msg()
        monkeypatch.setattr(config, "TIME_SERIES_TIER_FLOORS", True)
        assert "tier floors on" in main._no_pairs_msg()

    def test_a_run_that_did_not_search_time_series_says_so(self):
        settings = LiveSettings(True, (0.0, 1.0), 0.75, 0.2)
        searched = main._no_pairs_msg(settings=settings)
        skipped = main._no_pairs_msg(settings=settings, time_series_searched=False)
        assert skipped.startswith(
            "No qualifying pairs found (time-series: not searched this run, because a "
            "held market could not be identified (see the ERROR above)")
        assert config.describe_time_series_rule(True, (0.0, 1.0)) not in skipped
        # The same-title clause does not change
        assert skipped.endswith(searched[searched.index(" — or same-title"):])

    def test_a_category_or_tag_filter_is_named_only_when_set(self):
        # A filter alone can empty the list: named when set, unmentioned when not
        plain = main._no_pairs_msg(settings=LiveSettings(True, (0.0, 1.0), 0.75, 0.2))
        assert "category/tag filter" not in plain
        for categories, tags in ((("Economics",), None), (None, ("Oil & Gas",))):
            settings = LiveSettings(True, (0.0, 1.0), 0.75, 0.2,
                                    categories=categories, tags=tags)
            msg = main._no_pairs_msg(settings=settings)
            assert (f"among pairs filed under the run's category/tag filter "
                    f"({config.describe_trade_filter(settings)})") in msg
            # Everything before the clause is the unfiltered message's
            assert msg.startswith(plain[:-2])


@pytest.fixture
def pinned_config_toggles(monkeypatch):
    """Pin the seven live toggles through conftest's apply_pre_toggle_defaults, the
    one definition of their values, so every "(config: X)" mark reads the same
    whatever config.py ships."""
    apply_pre_toggle_defaults(monkeypatch)


def _seed_series_listing(series: dict, fetched_at: datetime | None = None) -> None:
    """
    Seed historical.load_series_categories' cached /series listing (in tmp_path).

    Args:
        series (dict): series ticker -> [category, [tags...]].
        fetched_at (datetime | None): When it was fetched; None means now (fresh).
    """
    stamp = datetime.now(UTC) if fetched_at is None else fetched_at
    historical._SERIES_CATEGORIES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    historical._SERIES_CATEGORIES_CACHE.write_text(json.dumps(
        {"fetched_at": stamp.isoformat(), "series": series}))


def _main_with(monkeypatch, argv: list, **patches) -> dict:
    """
    Run main.main() with argv up to its dispatch and return what it did.

    _run_dev/_run_prod (unless `patches` supplies them), build_client and
    _setup_logging are recorders. The dict holds "code", "logging_set_up",
    "client_built" and, when a run-mode recorder ran, "settings", "reference", "mode".
    """
    seen: dict = {"logging_set_up": False, "client_built": False}

    def record_run(client, args, settings, reference):
        seen.update(settings=settings, reference=reference, mode=args.mode)
        return EXIT_OK

    def record_client(mode):
        seen["client_built"] = True
        return MagicMock()

    def record_logging(path):
        seen["logging_set_up"] = True

    monkeypatch.setattr(sys, "argv", ["kalshi_betting.main", *argv])
    monkeypatch.setattr(main, "_setup_logging", record_logging)
    monkeypatch.setattr(main, "build_client", record_client)
    monkeypatch.setattr(main, "_run_dev", patches.get("_run_dev", record_run))
    monkeypatch.setattr(main, "_run_prod", patches.get("_run_prod", record_run))
    for name, value in patches.items():
        if name not in ("_run_dev", "_run_prod"):
            monkeypatch.setattr(main, name, value)
    with pytest.raises(SystemExit) as exc_info:
        main.main()
    seen["code"] = exc_info.value.code
    return seen


class TestOrderApiVersionGate:
    """main() exits 2 on any config.ORDER_API_VERSION but "v2", in either
    mode, before the live settings are resolved, logging is configured, a
    client is built or a run mode starts; "v2" reaches the run mode."""

    @pytest.mark.parametrize("mode", ["dev", "prod"])
    @pytest.mark.parametrize("value", ["legacy", "V2", "", None])
    def test_a_non_v2_order_path_exits_2_before_anything_runs(
        self, monkeypatch, capsys, mode, value,
    ):
        monkeypatch.setattr(config, "ORDER_API_VERSION", value)
        seen = _main_with(
            monkeypatch, ["--mode", mode],
            _resolve_live_settings=lambda *a: pytest.fail("resolved settings first"),
        )
        assert seen["code"] == 2
        assert not seen["logging_set_up"] and not seen["client_built"]
        assert "settings" not in seen and "mode" not in seen
        err = capsys.readouterr().err
        assert config.order_api_version_error() in err
        assert repr(value) in err and '"v2"' in err

    def test_the_shipped_value_reaches_the_run_mode(self, monkeypatch):
        assert config.ORDER_API_VERSION == "v2"
        seen = _main_with(monkeypatch, ["--mode", "dev"])
        assert seen["code"] == EXIT_OK and seen["mode"] == "dev"


@pytest.mark.usefixtures("pinned_config_toggles")
class TestLiveSettingsFlags:
    """Each live-toggle flag overrides ONE config.py toggle for one run, over
    config.live_settings() read at call time; a value LiveSettings refuses is a
    usage error (exit 2) before logging is configured (TS-20) or a client built."""

    @pytest.mark.parametrize("mode", ["dev", "prod"])
    @pytest.mark.parametrize("argv, field, value", [
        (["--no-tier-floors"], "tier_floors", False),
        (["--spread-min", "0.1"], "spread_band", (0.1, 1.0)),
        (["--spread-max", "0.5"], "spread_band", (0.0, 0.5)),
        (["--interval-discount", "0.6"], "interval_discount", 0.6),
        (["--size-cap", "35"], "size_cap", 0.35),
        (["--same-title-size-cap", "25"], "same_title_size_cap", 0.25),
        (["--category", "Economics"], "categories", ("Economics",)),
        (["--category", "Economics", "--category", " Sports "], "categories",
         ("Economics", "Sports")),
        (["--tag", "Oil & Gas"], "tags", ("Oil & Gas",)),
    ])
    def test_each_flag_overrides_only_its_own_field(self, monkeypatch, mode, argv, field, value):
        seen = _main_with(monkeypatch, ["--mode", mode, *argv])
        assert seen["code"] == EXIT_OK and seen["mode"] == mode
        settings, reference = seen["settings"], seen["reference"]
        assert reference == live_settings()
        assert getattr(settings, field) == value
        assert getattr(settings, field) != getattr(reference, field)
        for other in dataclasses.fields(LiveSettings):
            if other.name != field:
                assert getattr(settings, other.name) == getattr(reference, other.name), other.name

    def test_no_flag_hands_the_run_config_py_itself(self, monkeypatch):
        # The scheduler's exact argv (tests/test_scheduler.py pins it)
        seen = _main_with(monkeypatch, ["--mode", "prod"])
        assert seen["settings"] == seen["reference"] == live_settings()

    def test_a_flag_equal_to_config_departs_nothing(self, monkeypatch):
        cfg = live_settings()
        seen = _main_with(monkeypatch, [
            "--mode", "prod", "--tier-floors" if cfg.tier_floors else "--no-tier-floors",
            "--interval-discount", repr(cfg.interval_discount),
            "--size-cap", str(round(cfg.size_cap * 100)),
        ])
        assert seen["settings"] == seen["reference"]

    def test_one_band_flag_keeps_config_pys_other_bound(self, monkeypatch):
        # Read at call time from config.py, never bound at import
        monkeypatch.setattr(config, "TIME_SERIES_SPREAD_BAND", (0.1, 0.8))
        assert _main_with(monkeypatch, ["--spread-max", "0.5"])["settings"].spread_band == (0.1, 0.5)
        assert _main_with(monkeypatch, ["--spread-min", "0.2"])["settings"].spread_band == (0.2, 0.8)
        # 0 is a value, not "not given": it lowers config.py's floor to 0
        assert _main_with(monkeypatch, ["--spread-min", "0"])["settings"].spread_band == (0.0, 0.8)
        both = _main_with(monkeypatch, ["--spread-min", "0.05", "--spread-max", "0.9"])
        assert both["settings"].spread_band == (0.05, 0.9)
        assert both["reference"].spread_band == (0.1, 0.8)

    # The unit note on a cap flag's usage error: the flag takes a whole percent
    _STEP = f"{config.SIZE_CAP_STEP * 100:g}"
    _PERCENT = f"a whole percent, a multiple of {_STEP} from {_STEP} to 100"

    @pytest.mark.parametrize("argv, words, percent", [
        (["--size-cap", "37"],
         ["--size-cap", "size_cap", f"multiple of {config.SIZE_CAP_STEP:.0%}"], True),
        (["--size-cap", "0"], ["--size-cap", "size_cap", "(0, 1]"], True),
        (["--size-cap", "150"], ["--size-cap", "size_cap", "(0, 1]"], True),
        (["--same-title-size-cap", "105"], ["--same-title-size-cap", "same_title_size_cap"],
         True),
        # 0 is a value, not "not given": refused, never silently config.py's
        (["--same-title-size-cap", "0"],
         ["--same-title-size-cap", "same_title_size_cap", "(0, 1]"], True),
        (["--interval-discount", "0"], ["--interval-discount", "interval_discount"], False),
        (["--interval-discount", "1.5"], ["--interval-discount", "interval_discount"], False),
        (["--spread-min", "0.6", "--spread-max", "0.5"], ["--spread-min", "--spread-max",
                                                          "spread_band"], False),
        (["--spread-max", "0"], ["--spread-max", "spread_band"], False),
        # An empty name would match nothing: refused, never "any"
        (["--category", ""], ["--category", "categories"], False),
        (["--tag", "Soccer", "--tag", "  "], ["--tag", "tags"], False),
        # "any" is the --any-* switch, not a name: it would print like no filter (config._names)
        (["--category", "any"], ["--category", "categories", "--any-category / --any-tag"],
         False),
        (["--tag", "ANY"], ["--tag", "tags", "--any-category / --any-tag"], False),
        # The error names exactly the flags given, --no-tier-floors and --any-* included
        (["--no-tier-floors", "--spread-max", "0"],
         ["invalid live setting for this run (--tier-floors/--no-tier-floors, "
          "--spread-max):", "spread_band"], False),
        (["--any-category", "--spread-max", "0"],
         ["invalid live setting for this run (--spread-max, --any-category):"], False),
    ])
    def test_an_invalid_flag_is_a_usage_error_before_anything_runs(
        self, monkeypatch, capsys, argv, words, percent,
    ):
        seen = _main_with(monkeypatch, ["--mode", "prod", *argv])
        assert seen["code"] == 2
        # Refused before logging (TS-20), the client and either run mode
        assert not seen["logging_set_up"] and not seen["client_built"]
        assert "settings" not in seen
        err = capsys.readouterr().err
        for word in words:
            assert word in err, (word, err)
        # Only a cap flag's error names the percent unit its value was read in
        if percent:
            assert f"{argv[0]} takes {self._PERCENT}, read as that percent / 100" in err, err
        else:
            assert "whole percent" not in err, err

    @pytest.mark.parametrize("field, constant, flag", [
        ("categories", "TRADE_CATEGORIES", "--any-category"),
        ("tags", "TRADE_TAGS", "--any-tag"),
    ])
    def test_an_any_flag_clears_config_pys_filter_for_one_run(
        self, monkeypatch, field, constant, flag,
    ):
        monkeypatch.setattr(config, constant, ("Sports",))
        seen = _main_with(monkeypatch, ["--mode", "prod", flag])
        assert getattr(seen["reference"], field) == ("Sports",)
        assert getattr(seen["settings"], field) is None
        line = describe_live_settings(seen["settings"], seen["reference"])
        assert f"{field} any (config: Sports)" in line and line.count("(config:") == 1
        # No flag keeps config.py's filter
        seen = _main_with(monkeypatch, ["--mode", "prod"])
        assert getattr(seen["settings"], field) == ("Sports",)

    @pytest.mark.parametrize("argv, words", [
        (["--category", "Sports", "--any-category"], ["--any-category", "--category"]),
        (["--any-tag", "--tag", "Soccer"], ["--tag", "--any-tag"]),
    ])
    def test_a_filter_and_its_any_twin_are_mutually_exclusive(
        self, monkeypatch, capsys, argv, words,
    ):
        seen = _main_with(monkeypatch, ["--mode", "prod", *argv])
        assert seen["code"] == 2
        assert not seen["logging_set_up"] and not seen["client_built"]
        err = capsys.readouterr().err
        assert "not allowed with argument" in err, err
        for word in words:
            assert word in err, (word, err)

    def test_two_cap_flags_share_one_unit_note(self, monkeypatch, capsys):
        seen = _main_with(monkeypatch, ["--mode", "prod", "--size-cap", "37",
                                        "--same-title-size-cap", "25"])
        assert seen["code"] == 2
        err = capsys.readouterr().err
        assert f"--size-cap and --same-title-size-cap take {self._PERCENT}" in err, err

    def test_an_invalid_config_value_is_a_usage_error(self, monkeypatch, capsys):
        monkeypatch.setattr(config, "BUDGET_FRACTION", 0.37)
        seen = _main_with(monkeypatch, ["--mode", "prod"])
        assert seen["code"] == 2
        assert not seen["logging_set_up"] and not seen["client_built"]
        err = capsys.readouterr().err
        assert "config.py's live settings are invalid" in err and "size_cap" in err

    def test_help_names_every_flag_and_the_grid(self, monkeypatch, capsys):
        # argparse %-formats help: a bare "%" would raise here, so "%%" is pinned
        monkeypatch.setattr(sys, "argv", ["kalshi_betting.main", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            main.main()
        assert exc_info.value.code == 0
        # argparse wraps help to the terminal: compare words, not line breaks
        out = " ".join(capsys.readouterr().out.split())
        for flag in ("--tier-floors", "--no-tier-floors", "--spread-min", "--spread-max",
                     "--interval-discount", "--size-cap", "--same-title-size-cap",
                     "--category", "--any-category", "--tag", "--any-tag"):
            assert flag in out, flag
        assert "config.TRADE_CATEGORIES" in out and "config.TRADE_TAGS" in out
        assert out.count(f"in {config.SIZE_CAP_STEP * 100:g}% steps") == 2
        assert "100 = no cap" in out
        assert "100 = no extra cap beyond --size-cap" in out
        assert "live trading toggles" in out

    def test_the_echo_marks_exactly_the_departing_fields(self, monkeypatch):
        seen = _main_with(monkeypatch, ["--interval-discount", "0.751"])
        line = describe_live_settings(seen["settings"], seen["reference"])
        # k renders exactly, so 0.751 never prints as config.py's 0.75
        assert "k 0.751 (config: 0.75)" in line
        assert line.count("(config:") == 1

        seen = _main_with(monkeypatch, [
            "--no-tier-floors", "--spread-max", "0.5", "--size-cap", "35"])
        line = describe_live_settings(seen["settings"], seen["reference"])
        assert "tier floors off (config: on)" in line
        assert "spread band 0-0.5 (config: none)" in line
        assert "per-trade cap 35% (config: 20%)" in line
        assert line.count("(config:") == 3


def _prod_until_the_balance_gate(monkeypatch, argv: list, caplog) -> int:
    """Run the real _run_prod up to the MIN_BALANCE_CENTS gate (it logs its
    settings first) and return the exit code."""
    with caplog.at_level(logging.INFO):
        seen = _main_with(
            monkeypatch, ["--mode", "prod", *argv],
            _run_prod=main._run_prod,
            verify_auth=lambda client: {DEFAULT_EXCHANGE_INDEX: MIN_BALANCE_CENTS - 1},
        )
    return seen["code"]


@pytest.mark.usefixtures("pinned_config_toggles")
class TestLogLiveSettings:
    """Each live run logs its toggles on one INFO line, marking each departure from
    config.py; it WARNS on every live_rule_warnings sentence and, when a prod run
    that submits orders departs, on that."""

    _DEPARTURE = "This PRODUCTION run overrides config.py's live settings"

    def test_a_scheduled_run_logs_config_py_with_no_mark_and_no_warning(
        self, monkeypatch, caplog,
    ):
        code = _prod_until_the_balance_gate(monkeypatch, [], caplog)
        assert code == EXIT_SKIPPED_LOW_BALANCE
        lines = [r.getMessage() for r in caplog.records
                 if r.getMessage().startswith("Live settings:")]
        assert lines == [f"Live settings: {describe_live_settings(live_settings())}"]
        assert "(config:" not in lines[0]
        assert self._DEPARTURE not in caplog.text
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING
                    and r.getMessage().startswith("Live settings:")]

    # One departing flag per LiveSettings field, with the mark its line must carry
    _ONE_FLAG_PER_FIELD = {
        "tier_floors": (["--no-tier-floors"], "tier floors off (config: on)"),
        "spread_band": (["--spread-max", "0.5"], "spread band 0-0.5 (config: none)"),
        "interval_discount": (["--interval-discount", "0.6"], "k 0.6 (config: 0.75)"),
        "size_cap": (["--size-cap", "35"], "per-trade cap 35% (config: 20%)"),
        "same_title_size_cap": (["--same-title-size-cap", "15"],
                                "same-title cap 15% (config: 100% (no extra cap))"),
        "categories": (["--category", "Economics"], "categories Economics (config: any)"),
        "tags": (["--tag", "Oil & Gas"], "tags Oil & Gas (config: any)"),
    }

    def test_every_field_has_a_departing_flag(self):
        # A new LiveSettings field needs a row, so its departure WARNING is tested
        assert set(self._ONE_FLAG_PER_FIELD) == {f.name for f in dataclasses.fields(LiveSettings)}

    @pytest.mark.parametrize("field", sorted(_ONE_FLAG_PER_FIELD))
    def test_a_departing_production_run_warns(self, monkeypatch, caplog, field):
        argv, mark = self._ONE_FLAG_PER_FIELD[field]
        code = _prod_until_the_balance_gate(monkeypatch, argv, caplog)
        assert code == EXIT_SKIPPED_LOW_BALANCE
        (line,) = [r.getMessage() for r in caplog.records
                   if r.getMessage().startswith("Live settings: tier floors")]
        assert mark in line and line.count("(config:") == 1, line
        warnings = [r for r in caplog.records
                    if r.levelno == logging.WARNING and self._DEPARTURE in r.getMessage()]
        assert len(warnings) == 1

    def test_a_departing_dry_run_marks_but_does_not_warn(self, monkeypatch, caplog):
        _prod_until_the_balance_gate(
            monkeypatch, ["--dry-run", "--interval-discount", "0.6"], caplog)
        assert "k 0.6 (config: 0.75)" in caplog.text
        assert self._DEPARTURE not in caplog.text

    def test_a_departing_dev_run_marks_but_does_not_warn(self, caplog):
        settings = dataclasses.replace(live_settings(), interval_discount=0.6)
        with (
            patch("kalshi_betting.main.fetch_shard_statuses", return_value=None),
            patch("kalshi_betting.main.fetch_open_events_with_markets", return_value=[]),
            caplog.at_level(logging.INFO),
        ):
            code = main._run_dev(
                MagicMock(), SimpleNamespace(sandbox_balance=1000.0, max_horizon_days=None),
                settings, live_settings(),
            )
        assert code == EXIT_NO_TRADEABLE_SHARDS
        assert "k 0.6 (config: 0.75)" in caplog.text
        assert self._DEPARTURE not in caplog.text

    def test_every_rule_warning_is_logged(self, monkeypatch, caplog):
        _prod_until_the_balance_gate(
            monkeypatch, ["--dry-run", "--size-cap", "60", "--interval-discount", "0.4"], caplog)
        warned = [r.getMessage() for r in caplog.records
                  if r.levelno == logging.WARNING and r.getMessage().startswith("Live settings:")]
        assert warned == [
            f"Live settings: {text}"
            for text in config.live_rule_warnings(LiveSettings(
                config.TIME_SERIES_TIER_FLOORS, config.TIME_SERIES_SPREAD_BAND, 0.4, 0.6,
                config.SAME_TITLE_SIZE_CAP))
        ]
        assert any("one time-series pair may stake up to 60%" in w for w in warned)

    def test_it_never_resolves_config_py_itself(self, monkeypatch, caplog):
        def tripwire():
            raise AssertionError("_log_live_settings resolved config.py's settings")
        s = LiveSettings(False, (0.0, 0.5), 0.8, 1.0, 0.2)
        r = LiveSettings(True, (0.0, 1.0), 0.75, 0.2, 1.0)
        monkeypatch.setattr(main, "live_settings", tripwire)
        monkeypatch.setattr(config, "live_settings", tripwire)
        with caplog.at_level(logging.INFO):
            main._log_live_settings(s, r, real_money=True)
        assert f"Live settings: {describe_live_settings(s, r)}" in caplog.text
        assert self._DEPARTURE in caplog.text


def _filter_pair(event_ticker: str) -> SimpleNamespace:
    """A pair stand-in carrying market A's event ticker, all the filter reads."""
    return SimpleNamespace(market_a=SimpleNamespace(event_ticker=event_ticker),
                           market_b=SimpleNamespace(event_ticker="IGNORED-B"))


def _filtered(pairs: list, categories=None, tags=None, listing_client=None) -> list:
    """Run main._filter_by_category with only the category/tag fields set."""
    settings = LiveSettings(True, (0.0, 1.0), 0.75, 0.2, categories=categories, tags=tags)
    return main._filter_by_category(pairs, settings, listing_client)


class TestCategoryFilter:
    """main._filter_by_category keeps the pairs filed under the run's categories
    and tags by historical.series_labels (the dashboard's rule): case-insensitive,
    categories AND tags; no filter, no request; no listing, no pair (fail closed)."""

    # Kalshi's /series listing as cached: series ticker -> [category, tags]
    LISTING = {
        "KXNCAAMBGAME": ["Sports", ["Basketball"]],
        "KXUCLGAME": ["Sports", ["Soccer", "Europe"]],
        "KXBRENTW": ["Commodities", ["Oil & Gas", "Energy"]],
        "KXFISAEXTEND": ["Politics", ["Congress"]],
        "KXSCOTUSLAST": ["Politics", []],
        "KXMVECROSSCATEGORY": ["", []],
    }

    def _pairs(self) -> list:
        return [_filter_pair(t) for t in (
            "KXNCAAMBGAME-26JAN13WIUEIU", "KXUCLGAME-26APR14ATMBAR", "KXBRENTW-26SEP19",
            "KXFISAEXTEND-26", "KXSCOTUSLAST-26", "KXMVECROSSCATEGORY-S2026X",
            "KXMVECROSSCATEGORY0-S2026Y", "KXBTCD-26SEP1517", "", None,
        )]

    @staticmethod
    def _tickers(pairs: list) -> list:
        return [p.market_a.event_ticker for p in pairs]

    def test_no_filter_makes_no_request_and_returns_the_pairs_untouched(
        self, monkeypatch, caplog,
    ):
        def no_listing(*args, **kwargs):
            raise AssertionError("the listing was read with no filter set")

        monkeypatch.setattr(main, "load_series_categories", no_listing)
        pairs = self._pairs()
        with caplog.at_level(logging.INFO):
            out = _filtered(pairs, listing_client=MagicMock())
        assert out is pairs
        assert "Category/tag filter" not in caplog.text

    def test_a_category_matches_case_insensitively(self, caplog):
        _seed_series_listing(self.LISTING)
        with caplog.at_level(logging.INFO):
            out = _filtered(self._pairs(), categories=("sPoRtS",))
        assert self._tickers(out) == ["KXNCAAMBGAME-26JAN13WIUEIU", "KXUCLGAME-26APR14ATMBAR"]
        assert "kept 2 of 10 candidate pairs — dropped " in caplog.text
        assert "Politics · General 1" in caplog.text and "Other · General 3" in caplog.text

    def test_the_first_tag_is_the_one_matched(self):
        _seed_series_listing(self.LISTING)
        # Only KXBRENTW's first tag files it, as on the dashboard's Tag select
        assert self._tickers(_filtered(self._pairs(), tags=("oil & gas",))) == [
            "KXBRENTW-26SEP19"]
        assert _filtered(self._pairs(), tags=("Energy",)) == []
        assert self._tickers(_filtered(self._pairs(), tags=("Europe",))) == []

    def test_categories_and_tags_combine_by_and(self):
        _seed_series_listing(self.LISTING)
        assert self._tickers(_filtered(self._pairs(), categories=("Sports",),
                                       tags=("Soccer",))) == ["KXUCLGAME-26APR14ATMBAR"]
        # A tag of another category keeps nothing
        assert _filtered(self._pairs(), categories=("Politics",), tags=("Soccer",)) == []
        # Several names on one axis are an OR within it
        assert self._tickers(_filtered(self._pairs(), categories=("Politics", "Commodities"),
                                       tags=("Congress", "Oil & Gas"))) == [
            "KXBRENTW-26SEP19", "KXFISAEXTEND-26"]

    def test_a_combo_series_is_looked_up_literally(self):
        # The literal series, not scanner.event_series (which folds KXMVE*):
        # KXMVECROSSCATEGORY is listed uncategorised, KXMVECROSSCATEGORY0 not at all
        _seed_series_listing(self.LISTING)
        assert self._tickers(_filtered(self._pairs(), categories=("Uncategorised",))) == [
            "KXMVECROSSCATEGORY-S2026X"]
        assert "KXMVECROSSCATEGORY0-S2026Y" in self._tickers(
            _filtered(self._pairs(), categories=("Other",)))

    def test_an_unlisted_series_falls_back_to_the_ticker_prefix_label(self, caplog):
        _seed_series_listing(self.LISTING)
        assert infer_category("KXBTCD-26SEP1517") == "Crypto"
        with caplog.at_level(logging.WARNING):
            out = _filtered(self._pairs(), categories=("crypto",), tags=("General",))
        assert self._tickers(out) == ["KXBTCD-26SEP1517"]
        # A label only a fallback files under is not a typo
        assert "check the spelling" not in caplog.text
        # A missing or empty event ticker files under infer_category's "Other"
        assert self._tickers(_filtered(self._pairs(), categories=("Other",)))[-2:] == ["", None]

    def test_no_listing_keeps_nothing_and_says_why(self, monkeypatch, caplog):
        # Dev (no client) with no cached copy
        with caplog.at_level(logging.WARNING):
            assert _filtered(self._pairs(), categories=("Sports",)) == []
        assert "no cached copy exists (a dev run reads the cached copy only" in caplog.text
        assert "this run trades none" in caplog.text
        # Prod: the request fails and there is no cached copy
        caplog.clear()

        def offline(client, path, **params):
            raise RuntimeError("offline")

        monkeypatch.setattr(historical, "_historical_get", offline)
        with caplog.at_level(logging.WARNING):
            assert _filtered(self._pairs(), tags=("Soccer",), listing_client=MagicMock()) == []
        assert "Kalshi's /series listing could not be read and no cached copy exists" in (
            caplog.text)

    def test_a_name_no_series_carries_is_warned_as_a_typo(self, caplog):
        _seed_series_listing(self.LISTING)
        with caplog.at_level(logging.WARNING):
            assert _filtered(self._pairs(), categories=("Sprots",)) == []
        assert "Category 'Sprots' names no category in Kalshi's listing of 6 series" in (
            caplog.text)
        caplog.clear()
        # A dashboard Tag option ("category · tag"): the warning spells it as flags
        with caplog.at_level(logging.WARNING):
            assert _filtered(self._pairs(), tags=("Sports · Soccer",)) == []
        assert "Tag 'Sports · Soccer' is no series' first tag" in caplog.text
        assert "is --category 'Sports' --tag 'Soccer'" in caplog.text
        caplog.clear()
        # A real name draws no warning, even when it matches no pair this run
        with caplog.at_level(logging.WARNING):
            _filtered([_filter_pair("KXBRENTW-1")], categories=("Politics",), tags=("congress",))
        assert "check the spelling" not in caplog.text

    def test_a_fallback_tag_is_not_a_typo_when_no_listed_series_is_untagged(self, caplog):
        # "General" also files a series the listing lacks: with no untagged listed
        # series, only this run's filing knows it, and its pair is kept unwarned
        _seed_series_listing({"KXBRENTW": ["Commodities", ["Oil & Gas"]]})
        pairs = [_filter_pair("KXBRENTW-26SEP19"), _filter_pair("KXNOTLISTED-1")]
        with caplog.at_level(logging.WARNING):
            assert self._tickers(_filtered(pairs, tags=("general",))) == ["KXNOTLISTED-1"]
        assert "check the spelling" not in caplog.text

    def test_dev_reads_the_cached_listing_only_however_old(self, monkeypatch, caplog):
        # Recorded as well as raised: load_series_categories turns a raise into a WARNING
        requests = []

        def no_request(client, path, **params):
            requests.append((client, path))
            raise AssertionError("a request was made with no listing client")

        monkeypatch.setattr(historical, "_historical_get", no_request)
        _seed_series_listing(self.LISTING, fetched_at=datetime(2020, 1, 1, tzinfo=UTC))
        with caplog.at_level(logging.WARNING):
            assert self._tickers(_filtered(self._pairs(), categories=("Commodities",))) == [
                "KXBRENTW-26SEP19"]
        assert requests == []
        assert "Series category listing unavailable" not in caplog.text

    def test_prod_refreshes_a_stale_listing_with_its_own_client(self, monkeypatch):
        _seed_series_listing({"KXBRENTW": ["Sports", ["Soccer"]]},
                             fetched_at=datetime(2020, 1, 1, tzinfo=UTC))
        clients = []

        def fresh(client, path, **params):
            clients.append(client)
            return {"series": [{"ticker": "KXBRENTW", "category": "Commodities",
                                "tags": ["Oil & Gas"]}]}

        monkeypatch.setattr(historical, "_historical_get", fresh)
        client = MagicMock()
        out = _filtered(self._pairs(), categories=("Commodities",), listing_client=client)
        assert self._tickers(out) == ["KXBRENTW-26SEP19"]
        assert clients == [client]

    # A copy of test_dashboard.py's TestReturnsByCategory.SERIES: importing a
    # test class would make pytest collect it twice
    DASHBOARD_SERIES = {
        "KXNCAAMBGAME": ("Sports", ("Basketball",)),
        "KXUCLGAME": ("Sports", ("Soccer", "Europe")),
        "KXFISAEXTEND": ("Politics", ("Congress",)),
        "KXSCOTUSLAST": ("Politics", ()),
        "KXMVECROSSCATEGORY": ("", ()),
    }

    @staticmethod
    def _trade(event_ticker: str) -> BacktestTrade:
        """A BacktestTrade carrying market A's event ticker and its infer_category label."""
        return BacktestTrade(
            pair_type="same_title", ticker_a="A", ticker_b="B", title_a="Q", title_b="Q",
            category=infer_category(event_ticker), entry_date=date(2026, 1, 5),
            exit_date=date(2026, 1, 12), entry_pA=0.5, entry_pB=0.4, entry_nA=0.5,
            entry_nB=0.6, n=1, total_cost=0.9, fees=0.02, outcome_a="yes", outcome_b="yes",
            actual_payoff=1.0, profit=0.08, profit_ratio=0.1, monthly_profit_ratio=0.4,
            kelly_fraction=0.1, expected_payoff=0.08, slippage=0.0, holding_days=7,
            balance_at_entry=1000.0, event_ticker=event_ticker,
        )

    def test_it_files_every_pair_as_the_dashboard_files_the_trade(self):
        # For each (category, tag) the dashboard files a trade under, the filter
        # keeps exactly those pairs (this map spells no tag two ways; the next
        # test covers one that does)
        _seed_series_listing({t: [c, list(tags)] for t, (c, tags) in
                              self.DASHBOARD_SERIES.items()})
        events = ["KXNCAAMBGAME-26JAN13WIUEIU", "KXUCLGAME-26APR14ATMBAR", "KXSCOTUSLAST-26",
                  "KXFISAEXTEND-26", "KXMVECROSSCATEGORY-S2026X",
                  "KXMVECROSSCATEGORY0-S2026Y", "KXBTCD-26SEP1517", "KXNOTLISTED-1", ""]
        pairs = [_filter_pair(e) for e in events]
        filed: dict = {}
        for event in events:
            category, subcategory = dashboard._trade_category(
                self._trade(event), self.DASHBOARD_SERIES)
            assert subcategory.startswith(f"{category} · ")
            filed.setdefault((category, subcategory[len(category) + 3:]), []).append(event)
        assert len(filed) >= 6
        for (category, tag), expected in filed.items():
            kept = _filtered(pairs, categories=(category,), tags=(tag,))
            assert self._tickers(kept) == expected, (category, tag)
        # ... and each category alone keeps exactly its trades, whatever the tag
        for category in {c for c, _ in filed}:
            expected = [e for (c, _), es in filed.items() if c == category for e in es]
            kept = _filtered(pairs, categories=(category,))
            assert sorted(self._tickers(kept)) == sorted(expected), category

    def test_a_tag_is_matched_under_every_category_and_in_any_case(self):
        # Dashboard Tag options are category-scoped and case-exact; the filter's are not
        listing = {"KXCLUBWC": ["Sports", ["Soccer"]], "KXHKANE": ["Economics", ["Soccer"]],
                   "KXANIMEB": ["Entertainment", ["Anime Awards"]],
                   "ANIMEB": ["Entertainment", ["Anime awards"]]}
        _seed_series_listing(listing)
        events = ["KXCLUBWC-1", "KXHKANE-1", "KXANIMEB-1", "ANIMEB-1"]
        pairs = [_filter_pair(e) for e in events]
        series = {t: (c, tuple(tags)) for t, (c, tags) in listing.items()}
        # Four Tag options on the page ...
        assert [dashboard._trade_category(self._trade(e), series)[1] for e in events] == [
            "Sports · Soccer", "Economics · Soccer", "Entertainment · Anime Awards",
            "Entertainment · Anime awards"]
        # ... "Sports · Soccer" is --category Sports --tag Soccer; --tag Soccer
        # alone keeps every category's Soccer series
        assert self._tickers(_filtered(pairs, categories=("Sports",), tags=("Soccer",))) == [
            "KXCLUBWC-1"]
        assert self._tickers(_filtered(pairs, tags=("Soccer",))) == ["KXCLUBWC-1", "KXHKANE-1"]
        # ... and either spelling of one tag keeps both options' series
        for spelling in ("Anime Awards", "Anime awards"):
            assert self._tickers(_filtered(pairs, categories=("Entertainment",),
                                           tags=(spelling,))) == ["KXANIMEB-1", "ANIMEB-1"]

    def test_dashboard_and_live_filter_share_one_filing_rule(self):
        assert dashboard._series_labels is historical.series_labels
        assert main.series_labels is historical.series_labels


class TestSetupLogging:
    def test_setup_logging_has_console_and_file_handlers(self, tmp_path):
        root = logging.getLogger()
        saved_handlers = root.handlers[:]
        saved_level = root.level
        root.handlers = []
        try:
            log_path = tmp_path / "x.log"
            main._setup_logging(log_path)

            handler_types = [type(h) for h in root.handlers]
            assert logging.StreamHandler in handler_types
            # RotatingFileHandler, not a plain FileHandler (BS-25) — a
            # scheduler daemon runs this weekly forever, so the log must rotate
            assert logging.handlers.RotatingFileHandler in handler_types
            # Exactly one of each — basicConfig should not have added extras
            assert sum(1 for h in root.handlers if type(h) is logging.StreamHandler) == 1
            assert sum(1 for h in root.handlers if isinstance(h, logging.FileHandler)) == 1

            # delay=True: the file must not be created until a record is emitted
            assert not log_path.exists()
            logging.getLogger("test_setup_logging").info("trigger the delayed file open")
            assert log_path.exists()
        finally:
            for h in root.handlers:
                h.close()
            root.handlers = saved_handlers
            root.level = saved_level


# ═══════════════════════════════════════════════════════════════════════════
# Live-shape replay suite — end-to-end _run_dev / _run_prod runs against a
# MagicMock client wired with current-generation (2026-08+) Kalshi payload
# shapes only: yes_sub_title (no "subtitle" key), *_dollars price strings,
# balance_breakdown shard entries, position_fp counts, and orderbook_fp
# dollar-string book levels. This exercises the real scanner ingest/pairing,
# real strategy sizing, and real trader execution (mocked at the HTTP
# boundary only) — the closest offline substitute for a live smoke test.
#
# Fixed 4-group market set built by _live_shape_client():
#   Recurring Q / Will X happen?  -> SAME-EXP (0.50) / SAME-CHEAP (0.20)
#       the one tradeable same-title pair every test exercises.
#   Tick Event / Tick Test Market -> TICK-A (0.90, sub-cent tick fields) /
#       TICK-B (0.50) — priced to be non-tradeable (nA+pB > 1), so it always
#       appears in scan output but never in a portfolio.
#   Shard Event / Shard Test Market -> SHARD1-A (exchange_index=1) /
#       SHARD1-B (shard 0) — a cross-shard pair that IS ingested (market data
#       is cross-shard) but is priced non-tradeable, so it only ever proves
#       ingest tagging, never execution.
#   Held Event / Held Question -> HELD-A (held in prod) / HELD-B — forms a
#       tradeable pair in dev (no held-ticker filtering) but never in prod.
#
# Every market above shares ONE close time (_CLOSE), so no time-series pair
# with a real deadline gap exists among them. Since DR-67 the time-series
# finder does not even form a copy of these same-title groups: their titles
# ("Will X happen?") state no cumulative "by <date>" deadline, so the wording
# screen refuses them outright — a same-title pair requires identical wording
# and a time-series pair requires two DIFFERENT stated deadlines, so the two
# finders can no longer collide on one ticker pair (see main._dedup_pairs'
# docstring). The candidate set of every replay is the same-title set only.
#
# An OPT-IN fifth group (_live_shape_client(include_time_series=True)) adds
# the 2026-09 time-series fixture — TS-EARLY (closes _CLOSE, YES 0.30) /
# TS-LATE (closes 10 days later, YES 0.60 / NO 0.40) on distinct event
# tickers whose titles differ ONLY by date, so the same-title scanner cannot
# pair them and the time-series finder must. It defaults OFF because the
# ingest census pins below ({0: 7, 1: 1} / {0: 7}) count the fixed set.
#
# The one tradeable pair's cheap leg can be moved onto another shard with
# _live_shape_client(same_cheap_shard=...), which is what the cross-shard
# routing and collateral-transfer replays use.
# ═══════════════════════════════════════════════════════════════════════════

_CLOSE = "2026-12-01T00:00:00Z"
# 10 days after _CLOSE — a short-tier deadline gap (<= SHORT_DEADLINE_GAP_DAYS)
_CLOSE_LATE = "2026-12-11T00:00:00Z"

_TICKER_SAME_EXP = "SAME-EXP"
_TICKER_SAME_CHEAP = "SAME-CHEAP"
_TICKER_TICK_A = "TICK-A"
_TICKER_TICK_B = "TICK-B"
_TICKER_SHARD_A = "SHARD1-A"
_TICKER_SHARD_B = "SHARD1-B"
_TICKER_HELD_A = "HELD-A"
_TICKER_HELD_B = "HELD-B"
_TICKER_TS_EARLY = "TS-EARLY"
_TICKER_TS_LATE = "TS-LATE"
# Titles differ only by their date token, so normalize_title collapses both
# to the same key (a time-series group) while the exact-title same-title
# grouping keeps them apart.
_TITLE_TS_EARLY = "Will Z happen by December 1, 2026?"
_TITLE_TS_LATE = "Will Z happen by December 11, 2026?"

_TICK_PRICE_RANGES = [
    {"start": "0", "end": "0.01", "step": "0.0001"},
    {"start": "0.01", "end": "0.99", "step": "0.001"},
    {"start": "0.99", "end": "1", "step": "0.0001"},
]

# Live-shape balance payload (2026-08-14 shard scoping): shard 0 holds
# $250.00 and shard 1 holds $9,999.00 — sizing sums the BREAKDOWN
# (1024900 cents = $10,249.00), deliberately != the $10,250.00 top-level
# balance_dollars aggregate, so the replay proves the breakdown sum is what
# Kelly sizing sees, not the top-level field.
_LIVE_BALANCE_PAYLOAD = {
    "balance": 114,
    "balance_dollars": "10250.0000",
    "balance_breakdown": [
        {"exchange_index": 0, "balance": "250.0000"},
        {"exchange_index": 1, "balance": "9999.0000"},
    ],
}

# Same shape, but the breakdown SUM is below MIN_BALANCE_CENTS ($50 = 5000
# cents): shard0 $5.00 + shard1 $3.00 = 800 cents. Under multi-shard
# semantics an account with money parked on shard 1 must still be summed in
# — it's the total across shards that must clear the floor, not any single
# shard — so this fixture only aborts because the TOTAL is sub-minimum.
_LOW_BALANCE_PAYLOAD = {
    "balance": 1,
    "balance_dollars": "8.0000",
    "balance_breakdown": [
        {"exchange_index": 0, "balance": "5.0000"},
        {"exchange_index": 1, "balance": "3.0000"},
    ],
}

# Everything on shard 0, nothing on shard 1 — the collateral-transfer replays
# put the cheap leg on shard 1, so market B's (the YES leg's) cash requirement is a pure deficit
# that only an intra-exchange transfer out of shard 0's surplus can cover.
_SHARD1_EMPTY_BALANCE = {
    "balance": 114,
    "balance_dollars": "10000.0000",
    "balance_breakdown": [
        {"exchange_index": 0, "balance": "10000.0000"},
        {"exchange_index": 1, "balance": "0.0000"},
    ],
}

# What the account reads back AFTER a transfer settles — trader's settlement
# poll re-reads the balance through auth.verify_auth, so this is how the
# replay proves the money landed before any order relies on it.
_SHARD1_SETTLED_BALANCE = {
    "balance": 114,
    "balance_dollars": "10000.0000",
    "balance_breakdown": [
        {"exchange_index": 0, "balance": "5000.0000"},
        {"exchange_index": 1, "balance": "5000.0000"},
    ],
}

# Funds parked on an advertised, trading-active shard that produced ZERO
# ingested markets — the one coverage gap check_shard_coverage calls CRITICAL.
_SHARD2_FUNDED_BALANCE = {
    "balance": 114,
    "balance_dollars": "10349.0000",
    "balance_breakdown": [
        {"exchange_index": 0, "balance": "250.0000"},
        {"exchange_index": 1, "balance": "9999.0000"},
        {"exchange_index": 2, "balance": "100.0000"},
    ],
}


def _stub_ingest() -> list:
    """One market-shaped stand-in for "ingest returned something".

    Tests that stub the pair finders out still have to hand _run_dev/_run_prod
    a NON-empty ingest: a run whose ingest produced zero markets scanned
    nothing and is now reported as a blind run (EXIT_NO_TRADEABLE_SHARDS,
    VI-02), and "zero ingested markets but the finders returned pairs" is not
    a shape the live pipeline can ever take. Only .ticker (prod's held-ticker
    filter) and .exchange_index (the coverage census) are read on those
    stubbed paths.
    """
    return [SimpleNamespace(ticker="STUB-INGEST-MKT", exchange_index=DEFAULT_EXCHANGE_INDEX)]


def _status_entry(
    idx: int, *, trading_active: bool = True, transfers_active: bool = True,
    description: str = "Main",
) -> dict:
    """One entry of GET /exchange/status's exchange_index_statuses array."""
    return {
        "exchange_index": idx,
        "trading_active": trading_active,
        "exchange_active": True,
        "intra_exchange_transfers_active": transfers_active,
        "description": description,
    }


def _status_payload(*entries: dict) -> dict:
    """A full GET /exchange/status body wrapping the given shard entries."""
    return {
        "exchange_active": True,
        "trading_active": True,
        "exchange_index_statuses": list(entries),
    }


# Default advertised topology: shards 0 and 1, both trading and both accepting
# intra-exchange transfers — matching the fixed market set, which spans exactly
# those two shards.
_EXCHANGE_STATUS_PAYLOAD = _status_payload(
    _status_entry(0, description="Main"),
    _status_entry(1, description="Combos"),
)


def _raw_json_response(payload: dict, status: int = 200, reason: str = "OK") -> SimpleNamespace:
    """Raw-response stand-in matching every *_without_preload_content call and
    a hand-built rest_client.request call. reason/getheaders are what
    ApiException.from_response reads off a real RESTResponse on the non-2xx
    path (see test_http.py's TestSignedRequestJson._response)."""
    return SimpleNamespace(
        status=status,
        data=json.dumps(payload).encode("utf-8"),
        reason=reason,
        getheaders=lambda: {},
    )


def _mk_market(
    ticker: str,
    event_ticker: str,
    title: str,
    sub: str,
    yes_ask: str,
    no_ask: str,
    *,
    exchange_index: int = 0,
    price_level_structure: str = "",
    price_ranges: list | None = None,
    close_time: str = _CLOSE,
) -> dict:
    """One raw market JSON dict in the current (2026-08+) wire shape: no
    "subtitle" key at all (only yes_sub_title), *_dollars price strings, and
    an explicit exchange_index. `close_time` defaults to the shared _CLOSE so
    the fixed market set never forms a deadline-gapped time-series pair; the
    opt-in time-series group overrides it."""
    d = {
        "ticker": ticker,
        "event_ticker": event_ticker,
        "title": title,
        "yes_sub_title": sub,
        "status": "active",
        "close_time": close_time,
        "yes_ask_dollars": yes_ask,
        "no_ask_dollars": no_ask,
        "yes_bid_dollars": no_ask,
        "price_level_structure": price_level_structure,
        "exchange_index": exchange_index,
    }
    if price_ranges is not None:
        d["price_ranges"] = price_ranges
    return d


def _ev(title: str, market: dict) -> dict:
    return {"title": title, "markets": [market]}


def _build_events(same_cheap_shard: int = 0, include_time_series: bool = False) -> list:
    """The fixed market set. `same_cheap_shard` moves the tradeable pair's
    cheap leg (market B) onto another exchange shard, which is what turns the
    one selectable trade into a cross-shard one. `include_time_series` appends
    the opt-in later-pricier time-series pair (TS-EARLY / TS-LATE, 10-day
    deadline gap) described in the suite header — off by default so the
    ingest census pins on the fixed set stay exact.

    Every same-title pair here deliberately puts its two markets in two
    DIFFERENT event series (EXPEVT-1 / CHEAPEVT-1, TICKAEVT-1 / TICKBEVT-1,
    ...). They all used to share the prefix "EVT", i.e. one series, which
    scanner.find_same_title_pairs now refuses: two events of one series are two
    instances of one recurring fixture, so identical wording across them is two
    questions rather than one question listed twice, and the 95% co-resolution
    prior does not apply (DR-02, DR-54). Distinct series are the shape the
    same-title strategy was built for. The time-series pair keeps ONE series
    (EVT-TS-EARLY / EVT-TS-LATE) on purpose — a real cumulative-deadline family
    is one series, and its two titles differ by the deadline, so the finder's
    identical-wording conjunct never fires on it."""
    events = [
        _ev("Recurring Q", _mk_market(
            _TICKER_SAME_EXP, "EXPEVT-1", "Will X happen?", "Outcome Main",
            "0.50", "0.45", price_level_structure="linear_cent",
        )),
        _ev("Recurring Q", _mk_market(
            _TICKER_SAME_CHEAP, "CHEAPEVT-1", "Will X happen?", "Outcome Main",
            "0.20", "0.75", price_level_structure="linear_cent",
            exchange_index=same_cheap_shard,
        )),
        _ev("Tick Event", _mk_market(
            _TICKER_TICK_A, "TICKAEVT-1", "Tick Test Market", "Outcome",
            "0.90", "0.85", price_level_structure="center_deci_edge_centi_cent",
            price_ranges=_TICK_PRICE_RANGES,
        )),
        _ev("Tick Event", _mk_market(
            _TICKER_TICK_B, "TICKBEVT-1", "Tick Test Market", "Outcome",
            "0.50", "0.45",
        )),
        _ev("Shard Event", _mk_market(
            _TICKER_SHARD_A, "SHARDAEVT-1", "Shard Test Market", "Outcome",
            "0.60", "0.35", exchange_index=1,
        )),
        _ev("Shard Event", _mk_market(
            _TICKER_SHARD_B, "SHARDBEVT-1", "Shard Test Market", "Outcome",
            "0.50", "0.45",
        )),
        _ev("Held Event", _mk_market(
            _TICKER_HELD_A, "HELDAEVT-1", "Held Question", "Outcome",
            "0.50", "0.45",
        )),
        _ev("Held Event", _mk_market(
            _TICKER_HELD_B, "HELDBEVT-1", "Held Question", "Outcome",
            "0.20", "0.75",
        )),
    ]
    if include_time_series:
        # The 2026-09 flow-through fixture: the LATER contract's YES ask (0.60)
        # sits 0.30 above the earlier's (0.30) — a short-tier gap (10 days,
        # threshold 15%) — so the bot buys YES on TS-EARLY at pA=0.30 and NO on
        # TS-LATE at nB=0.40. Same event title, distinct event tickers.
        events.extend([
            _ev("TS Event", _mk_market(
                _TICKER_TS_EARLY, "EVT-TS-EARLY", _TITLE_TS_EARLY, "Outcome",
                "0.30", "0.70", price_level_structure="linear_cent",
            )),
            _ev("TS Event", _mk_market(
                _TICKER_TS_LATE, "EVT-TS-LATE", _TITLE_TS_LATE, "Outcome",
                "0.60", "0.40", price_level_structure="linear_cent",
                close_time=_CLOSE_LATE,
            )),
        ])
    return events


def _raw_events_page(events: list, cursor: str | None = None) -> SimpleNamespace:
    return _raw_json_response({"events": events, "cursor": cursor})


# Orderbook depth for the one tradeable same-title pair: market A's YES bids
# complement to a NO ask of 0.45 (matching nA above); market B's NO bids
# complement to a YES ask of 0.20 (matching pB above) — see
# scanner._bids_to_ask_levels. Buying side S on a market consumes that
# market's OPPOSITE-side bids, so the opt-in time-series pair is the mirror
# image: TS-EARLY (a YES buy) serves NO bids at 0.70 => YES asks 0.30 with an
# empty yes side; TS-LATE (a NO buy) serves YES bids at 0.60 => NO asks 0.40,
# and a NO bid at 0.40 => the reference YES ask of 0.60 enrichment needs (at its
# own YES bid: uncrossed). 100 contracts each — the depth cap the replay pins.
_ORDERBOOK_PAYLOADS = {
    _TICKER_SAME_EXP: {"orderbook_fp": {"yes_dollars": [["0.55", "100"]], "no_dollars": []}},
    _TICKER_SAME_CHEAP: {"orderbook_fp": {"yes_dollars": [], "no_dollars": [["0.80", "100"]]}},
    _TICKER_TS_EARLY: {"orderbook_fp": {"yes_dollars": [], "no_dollars": [["0.70", "100"]]}},
    _TICKER_TS_LATE: {"orderbook_fp": {"yes_dollars": [["0.60", "100"]],
                                       "no_dollars": [["0.40", "100"]]}},
}


def _orderbook_side_effect(ticker: str) -> SimpleNamespace:
    payload = _ORDERBOOK_PAYLOADS.get(
        ticker, {"orderbook_fp": {"yes_dollars": [], "no_dollars": []}}
    )
    return _raw_json_response(payload)


def _positions_side_effect(held_payload: dict, lookup_map: dict):
    """Dispatches get_positions_without_preload_content calls: a "ticker"
    kwarg means trader._position_count's per-ticker lookup; otherwise it's
    scanner.get_held_tickers' paginated listing.

    A lookup_map value may be a single payload (every read of that ticker sees
    it) or a LIST of payloads consumed in order, which is what models a
    position CHANGING across reads. trader attributes an ambiguous fill by
    DELTA, never by the absolute holding, so a leg whose fill must read as
    "filled" needs a baseline read (taken before either order is submitted)
    followed by a post-exception read that differs by exactly the leg's count.
    The last entry of a list repeats once exhausted."""
    sequences = {t: list(v) for t, v in lookup_map.items() if isinstance(v, list)}

    def _effect(**kwargs):
        if "ticker" in kwargs:
            ticker = kwargs["ticker"]
            if ticker in sequences:
                seq = sequences[ticker]
                payload = seq.pop(0) if len(seq) > 1 else seq[0]
            else:
                payload = lookup_map.get(ticker, {"market_positions": []})
            # A callable entry is resolved at read time, so a payload can be
            # derived from what was actually submitted rather than hardcoding
            # a count the Kelly sizing might change
            if callable(payload):
                payload = payload()
        else:
            payload = held_payload
        return _raw_json_response(payload)

    return _effect


def _balance_side_effect(first: dict, later: dict | None):
    """Programs get_balance_without_preload_content. The FIRST read (the one
    _run_prod sizes on) returns `first`; every later read — trader's transfer
    settlement poll, then the post-trade balance — returns `later`, which is
    how a replay models funds actually landing on a shard."""
    state = {"n": 0}

    def _effect(*args, **kwargs):
        state["n"] += 1
        if later is None or state["n"] == 1:
            return _raw_json_response(first)
        return _raw_json_response(later)

    return _effect


def _order_side_effect(fill_pattern: list):
    """Programs client.rest_client.request for a sequence of V2 order POSTs.

    Each tag in fill_pattern is "full" (fills the exact requested count,
    read back from the submitted body so the response is always self-
    consistent regardless of the sized contract count), "kill" (a 2xx with
    fill_count 0), "fok_kill" (the HTTP 409 fill_or_kill_insufficient_resting_volume
    error the exchange really sends when it kills a fill-or-kill), or "error"
    (HTTP 500, exercising the ambiguous-response path)."""
    state = {"i": 0}

    def _effect(verb, url, headers=None, body=None):
        idx = state["i"]
        state["i"] += 1
        tag = fill_pattern[idx]
        if tag == "error":
            return _raw_json_response(
                {"error": "internal"}, status=500, reason="Internal Server Error"
            )
        if tag == "fok_kill":
            return _raw_json_response(
                {
                    "error": {
                        "code": "fill_or_kill_insufficient_resting_volume",
                        "message": "fill or kill insufficient resting volume",
                    }
                },
                status=409, reason="Conflict",
            )
        requested = int(Decimal(body["count"]))
        fill = requested if tag == "full" else 0
        payload = {"order": {"order_id": f"ord-{idx}", "fill_count": fill, "remaining_count": 0}}
        return _raw_json_response(payload)

    return _effect


def _transfer_and_order_side_effect(fill_pattern: list):
    """client.rest_client.request carries BOTH signed POSTs the live path
    makes — the collateral transfer and the V2 orders — so dispatch on the
    URL. Transfers are accepted with a transfer_id; orders fall through to
    _order_side_effect, whose fill_pattern therefore only counts orders."""
    orders = _order_side_effect(fill_pattern)

    def _effect(verb, url, headers=None, body=None):
        if TRANSFER_PATH in url:
            return _raw_json_response({"transfer_id": "xfer-1"})
        return orders(verb, url, headers=headers, body=body)

    return _effect


def _live_shape_client(
    monkeypatch,
    *,
    balance_payload: dict,
    balance_payload_after: dict | None = None,
    exchange_status_payload: dict | None = _EXCHANGE_STATUS_PAYLOAD,
    same_cheap_shard: int = 0,
    include_held_position: bool = True,
    position_lookup_responses: dict | None = None,
    order_side_effect=None,
    mve_bailout: bool = False,
    include_time_series: bool = False,
    extra_events: tuple = (),
    extra_held: tuple = (),
):
    """Build a MagicMock KalshiClient wired end-to-end with current-generation
    payload shapes over the fixed 4-group market set described above.

    Args:
        monkeypatch: pytest's monkeypatch fixture, used to configure
            scanner.INCLUDE_MVE_MARKETS / MVE_MAX_EMPTY_PAGES.
        balance_payload (dict): Body for the FIRST get_balance_without_preload_content
            read — the one prod sizing is based on.
        balance_payload_after (dict | None): Body for every LATER balance read
            (trader's transfer settlement poll, then the post-trade balance).
            None (default) keeps returning balance_payload forever.
        exchange_status_payload (dict | None): Body for
            get_exchange_status_without_preload_content, defaulting to shards
            0 and 1 both trading-active with transfers active. Pass None to
            leave the attribute unwired, which is what every OTHER test module's
            MagicMock client looks like: fetch_json_page then chokes on the
            auto-created MagicMock attribute, scanner.fetch_shard_statuses'
            broad except swallows it, and the run degrades to single-shard
            semantics (see test_unwired_exchange_status_degrades_to_single_shard).
        same_cheap_shard (int): exchange_index for the tradeable pair's cheap
            leg (SAME-CHEAP / market B, the YES leg). 0 (default) keeps the pair on one shard;
            1 makes it the cross-shard pair the routing and collateral replays
            need.
        include_held_position (bool): Whether the account holds HELD-A —
            True (default) matches every test's expectation that the held
            pair never trades in prod.
        position_lookup_responses (dict | None): ticker -> positions body,
            consulted only by trader._position_count's ambiguous-leg lookup.
        order_side_effect: Optional callable for client.rest_client.request
            (see _order_side_effect). Only the V2 live-execution tests need
            this — dry-run and dev paths never submit orders.
        mve_bailout (bool): When True, leaves INCLUDE_MVE_MARKETS at its
            default (True) and shrinks MVE_MAX_EMPTY_PAGES to 1 so the MVE
            pull's bail-out fires almost immediately instead of the loop
            terminating on a None cursor — this is what actually exercises
            the bail-out path rather than the ordinary end-of-listing exit.
            When False (default), INCLUDE_MVE_MARKETS is turned off so the
            MVE loop is skipped entirely — the cleaner setup for every test
            that isn't specifically about the MVE bail-out.
        include_time_series (bool): When True, the events page also carries
            the opt-in later-pricier time-series pair (TS-EARLY / TS-LATE)
            and the orderbook mock serves its depth. MUST default to False:
            the ingest-census pins in the replays count the fixed set.
        extra_events (tuple): More events for the events page, after the rest.
        extra_held (tuple): More tickers the account holds, beside HELD-A.
    """
    if mve_bailout:
        monkeypatch.setattr(scanner_mod, "INCLUDE_MVE_MARKETS", True)
        monkeypatch.setattr(scanner_mod, "MVE_MAX_EMPTY_PAGES", 1)
    else:
        monkeypatch.setattr(scanner_mod, "INCLUDE_MVE_MARKETS", False)

    client = MagicMock()

    client.get_events_without_preload_content = MagicMock(
        return_value=_raw_events_page(_build_events(
            same_cheap_shard=same_cheap_shard, include_time_series=include_time_series,
        ) + list(extra_events))
    )
    if mve_bailout:
        # A non-None cursor on every page means only the consecutive-empty-
        # page bail-out (not "no cursor left") can end this loop.
        client.get_multivariate_events_without_preload_content = MagicMock(
            return_value=_raw_events_page([], cursor="MVE-CURSOR-1")
        )

    # Per-shard exchange status: read raw, so the run derives real
    # trading_active/transfers_active facts rather than falling back to
    # single-shard semantics.
    if exchange_status_payload is not None:
        client.get_exchange_status_without_preload_content = MagicMock(
            return_value=_raw_json_response(exchange_status_payload)
        )

    client.get_balance_without_preload_content = MagicMock(
        side_effect=_balance_side_effect(balance_payload, balance_payload_after)
    )

    held_payload = {
        "market_positions": (
            [{"ticker": _TICKER_HELD_A, "position_fp": "3.00"}] if include_held_position else []
        ) + [{"ticker": ticker, "position_fp": "2.00"} for ticker in extra_held],
        "cursor": None,
    }
    client.get_positions_without_preload_content = MagicMock(
        side_effect=_positions_side_effect(held_payload, position_lookup_responses or {})
    )

    client.get_market_orderbook_without_preload_content = MagicMock(
        side_effect=_orderbook_side_effect
    )

    # Never called: every order goes through signed_request_json
    client.create_order_without_preload_content = MagicMock()

    # V2 submission plumbing — signed_request_json reads these directly.
    client.configuration.host = "https://api.elections.kalshi.com/trade-api/v2"
    client.kalshi_auth.create_auth_headers = MagicMock(return_value={})
    client.rest_client.request = MagicMock(
        side_effect=order_side_effect if order_side_effect is not None else []
    )

    return client


def _capture_dev_simulation(monkeypatch) -> dict:
    """Patch out the Excel writer and capture what _run_dev hands it."""
    captured: dict = {}

    def fake_write_dev_simulation(results, all_candidates, balance_cents):
        captured["results"] = results
        captured["all_candidates"] = all_candidates
        captured["balance_cents"] = balance_cents
        return pathlib.Path("/fake/dev_sim.xlsx")

    monkeypatch.setattr(main, "write_dev_simulation", fake_write_dev_simulation)
    return captured


def _candidate_tickers(candidates) -> set:
    tickers: set = set()
    for pair in candidates:
        tickers.add(pair.market_a.ticker)
        tickers.add(pair.market_b.ticker)
    return tickers


class TestRunDevLiveShapeReplay:
    def test_run_dev_dry_run_end_to_end_current_payload_shapes(self, monkeypatch, caplog):
        client = _live_shape_client(
            monkeypatch,
            balance_payload=_LIVE_BALANCE_PAYLOAD,  # unused in dev, harmless
            mve_bailout=True,
        )
        captured = _capture_dev_simulation(monkeypatch)

        args = SimpleNamespace(sandbox_balance=1000.0, max_horizon_days=None)

        with caplog.at_level(logging.INFO):
            main._run_dev(client, args)

        assert "results" in captured
        simulated = [r for r in captured["results"] if r.status == "simulated"]
        assert simulated, "expected at least one simulated TradeResult"

        # The simulated trade's prices came straight from the *_dollars payloads.
        main_spec = next(
            r.spec for r in simulated if r.spec.pair.market_a.ticker == _TICKER_SAME_EXP
        )
        assert main_spec.pair.nA == pytest.approx(0.45)
        assert main_spec.pair.pB == pytest.approx(0.20)

        # Market data is cross-shard: the exchange_index=1 market is INGESTED
        # and tagged, not dropped. (It was dropped before the multi-shard flip.)
        all_candidates = captured["all_candidates"]
        assert _TICKER_SHARD_A in _candidate_tickers(all_candidates)
        shard_pair = next(
            p for p in all_candidates if p.market_a.ticker == _TICKER_SHARD_A
        )
        assert shard_pair.market_a.exchange_index == 1
        assert shard_pair.market_b.exchange_index == 0

        # Sub-cent tick fields flowed through ingest without breaking the run.
        tick_pair = next(
            p for p in all_candidates if p.market_a.ticker == _TICKER_TICK_A
        )
        assert tick_pair.market_a.price_level_structure == "center_deci_edge_centi_cent"
        assert tick_pair.market_a.price_ranges is not None

        # The old drop-at-ingest warning is gone, and the per-shard ingest
        # census (the only signal of which shards a run actually saw) is on.
        assert "non-routable" not in caplog.text
        assert "Ingested markets by shard: {0: 7, 1: 1}" in caplog.text
        # Both advertised shards produced markets — nothing to report.
        assert "Full shard coverage: shards [0, 1] scanned" in caplog.text
        assert "SHARD COVERAGE FAILURE" not in caplog.text

        assert "MVE fetch:" in caplog.text  # proves the bail-out path actually fired

        client.create_order_without_preload_content.assert_not_called()
        assert client.rest_client.request.call_count == 0

    def test_run_dev_drops_markets_on_trading_inactive_shard(self, monkeypatch, caplog):
        # The ONE ingest-time shard exclusion: nothing on a halted shard can be
        # traded, nor should it linger as a stale candidate.
        client = _live_shape_client(
            monkeypatch,
            balance_payload=_LIVE_BALANCE_PAYLOAD,
            exchange_status_payload=_status_payload(
                _status_entry(0, description="Main"),
                _status_entry(1, trading_active=False, description="Combos"),
            ),
        )
        captured = _capture_dev_simulation(monkeypatch)

        args = SimpleNamespace(sandbox_balance=1000.0, max_horizon_days=None)

        with caplog.at_level(logging.INFO):
            main._run_dev(client, args)

        assert _TICKER_SHARD_A not in _candidate_tickers(captured["all_candidates"])
        assert "Skipped 1 markets on trading-inactive exchange shards [1]" in caplog.text
        assert "Ingested markets by shard: {0: 7}" in caplog.text
        # A deliberately-halted shard is never re-reported as a coverage gap —
        # the ingest drop above already warned about it.
        assert "SHARD COVERAGE FAILURE" not in caplog.text
        assert "Shard coverage:" not in caplog.text
        # ...but it is no longer folded into the full-coverage claim either:
        # the line names the shards actually scanned (TS-01).
        assert "Full shard coverage: shards [0] scanned" in caplog.text

    def test_run_dev_all_shards_inactive_returns_blind_code(self, monkeypatch, caplog):
        # Dev's exit code is not consumed by the scheduler, but the two modes
        # must not disagree about what a blind run is (TS-01).
        client = _live_shape_client(
            monkeypatch,
            balance_payload=_LIVE_BALANCE_PAYLOAD,
            exchange_status_payload=_status_payload(
                _status_entry(0, trading_active=False, description="Main"),
                _status_entry(1, trading_active=False, description="Combos"),
            ),
        )
        wrote: list = []
        monkeypatch.setattr(main, "write_dev_simulation", lambda *a, **k: wrote.append(a))

        args = SimpleNamespace(sandbox_balance=1000.0, max_horizon_days=None)

        with caplog.at_level(logging.INFO):
            code = main._run_dev(client, args)

        assert code == EXIT_NO_TRADEABLE_SHARDS
        assert "Every advertised exchange shard is trading-inactive ([0, 1])" in caplog.text
        assert "Shard coverage NOT claimable" in caplog.text
        assert "Full shard coverage" not in caplog.text
        assert not wrote, "a blind run must short-circuit before the simulation write"

    def test_unwired_exchange_status_degrades_to_single_shard(self, monkeypatch, caplog):
        # Every other test module's MagicMock client leaves
        # get_exchange_status_without_preload_content unwired. That must remain
        # harmless: the auto-created MagicMock attribute makes fetch_json_page
        # raise, scanner.fetch_shard_statuses swallows it, and the run keeps
        # every market and simply cannot assess coverage.
        client = _live_shape_client(
            monkeypatch,
            balance_payload=_LIVE_BALANCE_PAYLOAD,
            exchange_status_payload=None,
        )
        captured = _capture_dev_simulation(monkeypatch)

        args = SimpleNamespace(sandbox_balance=1000.0, max_horizon_days=None)

        with caplog.at_level(logging.INFO):
            main._run_dev(client, args)

        assert _TICKER_SHARD_A in _candidate_tickers(captured["all_candidates"])
        assert "Per-shard exchange status unavailable — coverage not assessable." in caplog.text
        assert "trading-inactive" not in caplog.text
        assert "SHARD COVERAGE FAILURE" not in caplog.text

    def test_run_dev_time_series_pair_flows_through_end_to_end(self, monkeypatch, caplog):
        # The 2026-09 strategy: a LATER-closing contract priced well above the
        # earlier one is the anomaly; the bot buys YES on the earlier (A) at
        # pA and NO on the later (B) at nB. This replay drives the inverted
        # finder, the mirrored orderbook enrichment (a YES buy consumes NO
        # bids, a NO buy consumes YES bids), the discounted-gap Kelly model
        # and the NO-first dry-run order log end-to-end on the flow-through
        # fixture: pA 0.30, pB 0.60, nB 0.40, 10-day gap.
        client = _live_shape_client(
            monkeypatch,
            balance_payload=_LIVE_BALANCE_PAYLOAD,  # unused in dev, harmless
            include_time_series=True,
        )
        captured = _capture_dev_simulation(monkeypatch)

        args = SimpleNamespace(sandbox_balance=1000.0, max_horizon_days=None)

        with caplog.at_level(logging.INFO):
            main._run_dev(client, args)

        # The candidate: time_series, market_a is the EARLIER contract, prices
        # are the depth-weighted leg prices written back by enrichment.
        ts_pair = next(
            p for p in captured["all_candidates"]
            if p.market_a.ticker == _TICKER_TS_EARLY
        )
        assert ts_pair.pair_type == "time_series"
        assert ts_pair.market_b.ticker == _TICKER_TS_LATE
        assert ts_pair.pA == pytest.approx(0.30)
        assert ts_pair.nB == pytest.approx(0.40)
        assert ts_pair.tradeable

        # The fixed same-title set still flows through alongside it.
        assert any(
            r.spec.pair.market_a.ticker == _TICKER_SAME_EXP
            for r in captured["results"] if r.status == "simulated"
        )

        # Kelly on $1,000 (f* ~ 0.106 at config.py's k 0.80) outsizes the
        # 100-contract book: x == y == 100. total_cost = 70.00 and profit
        # if won = 100 x (1 - 0.70) - exact fees (1.47 + 1.68) = 26.85.
        ts_result = next(
            r for r in captured["results"]
            if r.spec.pair.market_a.ticker == _TICKER_TS_EARLY
        )
        assert ts_result.status == "simulated"
        assert ts_result.spec.x == 100
        assert ts_result.spec.y == 100
        assert ts_result.spec.total_cost == pytest.approx(70.00)
        assert ts_result.spec.min_payoff == pytest.approx(26.85)

        # The dry-run order line lists legs in SUBMISSION order: the NO leg
        # (the later market) first, then the YES leg (the earlier market).
        dry_run_lines = [
            rec.getMessage() for rec in caplog.records
            if "[DRY RUN]" in rec.getMessage() and _TITLE_TS_EARLY in rec.getMessage()
        ]
        assert len(dry_run_lines) == 1, dry_run_lines
        line = dry_run_lines[0]
        assert "NO on" in line and "YES on" in line
        assert line.index("NO on") < line.index("YES on")
        assert line.index(_TITLE_TS_LATE) < line.index(_TITLE_TS_EARLY)

        # The operator views render legs in MARKET order with the side next
        # to each count: YES on A, NO on B.
        assert "100× YES(A) + 100× NO(B)" in caplog.text

        client.create_order_without_preload_content.assert_not_called()
        assert client.rest_client.request.call_count == 0


    def test_pairs_table_still_shows_the_trade_for_a_selected_pair(
        self, monkeypatch, caplog,
    ):
        # Regression guard for an identity coupling. print_pairs_table looks a
        # spec up by id(pair) against the candidate list, and compute_trade now
        # returns a RE-PRICED COPY of the pair (the marginal fill price for the
        # size it settled on). Keyed off id(spec.pair), as it was, every
        # selected row would silently render "—" in the trade columns while the
        # trade itself executed perfectly normally — a reporting-only break that
        # no other assertion in this file would catch.
        client = _live_shape_client(
            monkeypatch,
            balance_payload=_LIVE_BALANCE_PAYLOAD,
            include_time_series=True,
        )
        _capture_dev_simulation(monkeypatch)
        args = SimpleNamespace(sandbox_balance=1000.0, max_horizon_days=None)

        with caplog.at_level(logging.INFO):
            main._run_dev(client, args)

        # The pairs table is logged one line per row; find the time-series row.
        row = next(
            (line for line in caplog.text.splitlines()
             if "time_series" in line and "│" in line),
            None,
        )
        assert row is not None, "time-series row missing from the pairs table"
        assert "YES(A)" in row and "NO(B)" in row, row
        assert "—" not in row.split("YES(A)")[0].split("│")[-2], row


class TestRunProdDryRunLiveShapeReplay:
    def test_run_prod_dry_run_end_to_end_current_payload_shapes(self, monkeypatch, caplog):
        client = _live_shape_client(monkeypatch, balance_payload=_LIVE_BALANCE_PAYLOAD)
        captured: dict = {}

        def fake_append_to_prod_log(results, balance_before, balance_after, *, run_note=""):
            captured["results"] = results
            captured["balance_before"] = balance_before
            captured["balance_after"] = balance_after
            captured["run_note"] = run_note
            return pathlib.Path("/fake/trade_log.xlsx")

        monkeypatch.setattr(main, "append_to_prod_log", fake_append_to_prod_log)

        args = SimpleNamespace(dry_run=True, max_horizon_days=None)

        with caplog.at_level(logging.INFO):
            main._run_prod(client, args)

        # Sum of the breakdown ($250.00 + $9,999.00 = $10,249.00) — never the
        # $10,250.00 top-level balance_dollars aggregate. Sizing is
        # portfolio-wide across shards, but must read the breakdown, not the
        # aggregate field, so this pins the distinction.
        assert "$10249.00" in caplog.text

        assert "results" in captured
        results = captured["results"]
        assert results, "expected at least one result"
        assert all(r.status == "simulated" for r in results)

        # The separator row names config.py's toggles (the run was handed none)
        assert captured["run_note"].startswith("settings: ")
        assert captured["run_note"] == f"settings: {describe_live_settings(live_settings())}"
        assert "(config:" not in captured["run_note"]

        # HELD-A is held — the pair it would have formed with HELD-B must never surface.
        tickers: set = set()
        for r in results:
            tickers.add(r.spec.pair.market_a.ticker)
            tickers.add(r.spec.pair.market_b.ticker)
        assert _TICKER_HELD_A not in tickers
        assert _TICKER_HELD_B not in tickers

        held_lookup_calls = [
            c for c in client.get_positions_without_preload_content.call_args_list
            if "ticker" not in c.kwargs
        ]
        assert held_lookup_calls, "expected get_held_tickers to have queried positions"

        client.create_order_without_preload_content.assert_not_called()
        assert client.rest_client.request.call_count == 0

    def test_run_prod_flags_funded_advertised_shard_with_no_markets(self, monkeypatch, caplog):
        # Shard 2 is advertised, trading-active, holds $100 — and produced zero
        # ingested markets. That is a real blind spot (a pair could exist there
        # and go undetected), so it is CRITICAL. The run still continues.
        client = _live_shape_client(
            monkeypatch,
            balance_payload=_SHARD2_FUNDED_BALANCE,
            exchange_status_payload=_status_payload(
                _status_entry(0, description="Main"),
                _status_entry(1, description="Combos"),
                _status_entry(2, description="Crypto"),
            ),
        )
        captured: dict = {}

        def fake_append_to_prod_log(results, balance_before, balance_after, *, run_note=""):
            captured["results"] = results
            return pathlib.Path("/fake/trade_log.xlsx")

        monkeypatch.setattr(main, "append_to_prod_log", fake_append_to_prod_log)
        args = SimpleNamespace(dry_run=True, max_horizon_days=None)

        with caplog.at_level(logging.INFO):
            main._run_prod(client, args)

        critical = [r for r in caplog.records if r.levelno == logging.CRITICAL]
        assert critical, "expected a CRITICAL coverage record"
        assert any("SHARD COVERAGE FAILURE" in r.getMessage() for r in critical)
        assert any(
            "shard 2" in r.getMessage() and "holds account funds" in r.getMessage()
            for r in critical
        )
        # Never an abort — the run trades on whatever shards WERE covered.
        assert captured.get("results"), "coverage failure must not stop the run"

    def test_run_prod_warns_when_empty_advertised_shard_holds_no_funds(self, monkeypatch, caplog):
        # Same topology minus the money on shard 2: an advertised-but-empty
        # shard is expected during the rollout, so it must not cry wolf.
        client = _live_shape_client(
            monkeypatch,
            balance_payload=_LIVE_BALANCE_PAYLOAD,
            exchange_status_payload=_status_payload(
                _status_entry(0, description="Main"),
                _status_entry(1, description="Combos"),
                _status_entry(2, description="Crypto"),
            ),
        )
        monkeypatch.setattr(
            main, "append_to_prod_log", lambda *a, **k: pathlib.Path("/fake/trade_log.xlsx"),
        )
        args = SimpleNamespace(dry_run=True, max_horizon_days=None)

        with caplog.at_level(logging.INFO):
            main._run_prod(client, args)

        assert "SHARD COVERAGE FAILURE" not in caplog.text
        assert "Shard coverage: advertised active shard 2" in caplog.text
        assert "may be legitimately empty" in caplog.text

    def test_run_prod_all_shards_inactive_returns_blind_code(self, monkeypatch, caplog):
        # TS-01: an exchange-wide halt (observed live 2026-09-03) makes ingest
        # drop every market. The run used to find no pairs, log the same "Full
        # shard coverage" line a healthy run logs, and exit 0 — so the
        # scheduler counted the weekly slot as satisfied and the bot did not
        # trade again for a week. It must now say so and exit 30.
        client = _live_shape_client(
            monkeypatch,
            balance_payload=_LIVE_BALANCE_PAYLOAD,
            exchange_status_payload=_status_payload(
                _status_entry(0, trading_active=False, description="Main"),
                _status_entry(1, trading_active=False, description="Combos"),
            ),
        )
        monkeypatch.setattr(
            main, "append_to_prod_log", lambda *a, **k: pathlib.Path("/fake/trade_log.xlsx"),
        )
        enriched: list = []
        monkeypatch.setattr(
            main, "enrich_with_orderbook_prices",
            lambda *a, **k: enriched.append(a) or [],
        )
        args = SimpleNamespace(dry_run=True, max_horizon_days=None)

        with caplog.at_level(logging.INFO):
            code = main._run_prod(client, args)

        assert code == EXIT_NO_TRADEABLE_SHARDS
        assert "Every advertised exchange shard is trading-inactive ([0, 1])" in caplog.text
        # The false all-clear this fixes: ([], []) from check_shard_coverage
        # must never read as success.
        assert "Full shard coverage" not in caplog.text
        assert "Shard coverage NOT claimable" in caplog.text
        assert not enriched, "a blind run must short-circuit before pair enrichment"

    def test_run_prod_full_coverage_lists_only_scannable_shards(self, monkeypatch, caplog):
        # A halted shard is no longer folded into the "full coverage" claim:
        # the line names the shards actually scanned, so an operator reading it
        # can tell a full ingest from a partial one (TS-01).
        client = _live_shape_client(
            monkeypatch,
            balance_payload=_LIVE_BALANCE_PAYLOAD,
            exchange_status_payload=_status_payload(
                _status_entry(0, description="Main"),
                _status_entry(1, trading_active=False, description="Combos"),
            ),
        )
        monkeypatch.setattr(
            main, "append_to_prod_log", lambda *a, **k: pathlib.Path("/fake/trade_log.xlsx"),
        )
        args = SimpleNamespace(dry_run=True, max_horizon_days=None)

        with caplog.at_level(logging.INFO):
            code = main._run_prod(client, args)

        assert code == EXIT_OK, "one halted shard is not a blind run"
        assert "Full shard coverage: shards [0] scanned" in caplog.text
        assert "Full shard coverage: shards [0, 1] scanned" not in caplog.text

    def test_run_prod_unknown_trading_flag_is_not_a_blind_run(self, monkeypatch, caplog):
        # TS-04 x TS-01: a dropped trading_active field parses to None, which
        # keeps every shard scannable — so this must NOT trip the all-inactive
        # short-circuit (that would turn a drift event into a skipped week).
        entries = [
            {"exchange_index": 0, "exchange_active": True,
             "intra_exchange_transfers_active": True, "description": "Main"},
            {"exchange_index": 1, "exchange_active": True,
             "intra_exchange_transfers_active": True, "description": "Combos"},
        ]
        client = _live_shape_client(
            monkeypatch,
            balance_payload=_LIVE_BALANCE_PAYLOAD,
            exchange_status_payload=_status_payload(*entries),
        )
        monkeypatch.setattr(
            main, "append_to_prod_log", lambda *a, **k: pathlib.Path("/fake/trade_log.xlsx"),
        )
        args = SimpleNamespace(dry_run=True, max_horizon_days=None)

        with caplog.at_level(logging.INFO):
            code = main._run_prod(client, args)

        assert code == EXIT_OK
        assert "Every advertised exchange shard is trading-inactive" not in caplog.text
        assert "Full shard coverage: shards [0, 1] scanned" in caplog.text

    def test_run_prod_aborts_below_min_balance(self, monkeypatch, caplog):
        client = _live_shape_client(monkeypatch, balance_payload=_LOW_BALANCE_PAYLOAD)
        args = SimpleNamespace(dry_run=True, max_horizon_days=None)

        with caplog.at_level(logging.WARNING):
            main._run_prod(client, args)

        assert "below minimum" in caplog.text
        client.get_events_without_preload_content.assert_not_called()
        client.get_positions_without_preload_content.assert_not_called()
        # The shard-status read happens after the balance gate — an account
        # that can't trade shouldn't spend an API call on exchange status.
        client.get_exchange_status_without_preload_content.assert_not_called()
        client.create_order_without_preload_content.assert_not_called()
        assert client.rest_client.request.call_count == 0


class TestRunProdLiveV2Replay:
    """dry_run=False replays of the shipped V2 order path end to end.
    Every test tunes the market set down to exactly one tradeable,
    selectable pair (SAME-EXP / SAME-CHEAP) so order counts are fully
    deterministic."""

    @pytest.fixture(autouse=True)
    def _fresh_v2_mapping_latch(self, monkeypatch):
        """Every replay starts as a fresh process does: the V2 NO-leg mapping
        unconfirmed, so trader._confirm_v2_no_mapping's first-fill check is
        genuinely exercised end-to-end. The latch is a process global that real
        execution flips, so monkeypatch also keeps it from leaking between
        tests."""
        monkeypatch.setattr(trader_mod, "_V2_NO_MAPPING_CONFIRMED", False)

    @staticmethod
    def _run(
        monkeypatch,
        fill_pattern,
        position_lookup_responses=None,
        *,
        same_cheap_shard: int = 0,
        balance_payload: dict = _LIVE_BALANCE_PAYLOAD,
        balance_payload_after: dict | None = None,
        order_side_effect=None,
        dry_run: bool = False,
    ):
        # trader's first-fill backstop judges the V2 NO-leg mapping by how the
        # NO-leg position MOVED across the fill, not by its absolute sign, so
        # the replay account must model a CHANGE: flat on the baseline read
        # (taken before either order is submitted), then short by exactly the
        # contracts the NO leg bought. A single static payload would give a delta of
        # 0 and stop every pair at manual_review. The moved count is read back
        # from the submitted body rather than hardcoded, so it can't drift out
        # of step with Kelly sizing.
        submitted: dict = {}
        base_orders = (
            order_side_effect if order_side_effect is not None
            else _order_side_effect(fill_pattern)
        )

        def recording_orders(verb, url, headers=None, body=None):
            # First body per ticker is that leg's opening order; a later
            # rollback on the same ticker must not overwrite it.
            if TRANSFER_PATH not in url and isinstance(body, dict):
                submitted.setdefault(body["ticker"], body["count"])
            return base_orders(verb, url, headers=headers, body=body)

        lookups = {
            _TICKER_SAME_EXP: [
                {"market_positions": []},
                lambda: {
                    "market_positions": [
                        {
                            "ticker": _TICKER_SAME_EXP,
                            "position_fp": f"-{submitted.get(_TICKER_SAME_EXP, '0')}",
                        }
                    ]
                },
            ],
        }
        lookups.update(position_lookup_responses or {})
        client = _live_shape_client(
            monkeypatch,
            balance_payload=balance_payload,
            balance_payload_after=balance_payload_after,
            same_cheap_shard=same_cheap_shard,
            order_side_effect=recording_orders,
            position_lookup_responses=lookups,
        )
        captured: dict = {}

        def fake_append_to_prod_log(results, balance_before, balance_after, *, run_note=""):
            captured["results"] = results
            return pathlib.Path("/fake/trade_log.xlsx")

        monkeypatch.setattr(main, "append_to_prod_log", fake_append_to_prod_log)
        args = SimpleNamespace(dry_run=dry_run, max_horizon_days=None)
        main._run_prod(client, args)
        return client, captured

    def test_run_prod_live_v2_all_filled_end_to_end(self, monkeypatch):
        # The replays run under the shipped setting
        assert ORDER_API_VERSION == "v2"

        client, captured = self._run(monkeypatch, ["full", "full"])

        calls = client.rest_client.request.call_args_list
        assert len(calls) == 2

        price_re = re.compile(r"^\d\.\d{4}$")
        count_re = re.compile(r"^\d+\.00$")
        for call in calls:
            url = call.args[1]
            assert V2_ORDER_PATH in url
            body = call.kwargs["body"]
            assert price_re.match(body["price"])
            assert count_re.match(body["count"])
            assert body["time_in_force"] == "fill_or_kill"
            # Required by the V2 endpoint, which rejects a body without it
            assert body["self_trade_prevention_type"] == "taker_at_cross"
            assert body["exchange_index"] == 0

        assert len(captured["results"]) == 1
        assert captured["results"][0].status == "executed"
        client.create_order_without_preload_content.assert_not_called()

    def test_run_prod_live_v2_leg_b_killed_rolls_back_end_to_end(self, monkeypatch):
        client, captured = self._run(monkeypatch, ["full", "kill", "full"])

        calls = client.rest_client.request.call_args_list
        assert len(calls) == 3

        # Every V2 body carries the endpoint's required self-trade-prevention
        # field, the unwind included
        for call in calls:
            assert call.kwargs["body"]["self_trade_prevention_type"] == "taker_at_cross"
        rollback_body = calls[2].kwargs["body"]
        assert rollback_body["side"] == "bid"
        assert rollback_body["reduce_only"] is True
        # The endpoint accepts reduce_only only with immediate_or_cancel
        assert rollback_body["time_in_force"] == "immediate_or_cancel"
        # The unwind is LOSS-FLOORED, not a flat top-of-grid bid: the NO leg's
        # scanned NO entry is 0.45, so the floor is 45 - 12 = 33c and the bid
        # cap is its YES-book mirror, 1 - 0.33 = 0.67, already on the replay
        # markets' default $0.01 grid and well under the 0.99 top-of-grid clamp
        assert rollback_body["price"] == "0.6700"

        assert captured["results"][0].status == "rolled_back"
        client.create_order_without_preload_content.assert_not_called()

    def test_run_prod_live_v2_leg_a_killed_is_failed_end_to_end(self, monkeypatch):
        client, captured = self._run(monkeypatch, ["kill"])

        assert client.rest_client.request.call_count == 1
        assert captured["results"][0].status == "failed"
        client.create_order_without_preload_content.assert_not_called()

    def test_run_prod_live_v2_leg_a_kill_response_is_failed_end_to_end(
        self, monkeypatch, caplog,
    ):
        # The exchange's real kill: an HTTP 409 through the signed transport,
        # raised by the SDK as its ConflictException. The pair ends "failed" on
        # one order request, with no position read after it and no pause.
        slept: list = []
        monkeypatch.setattr(trader_mod.time, "sleep", lambda s: slept.append(s))
        with caplog.at_level(logging.INFO):
            client, captured = self._run(monkeypatch, ["fok_kill"])

        assert client.rest_client.request.call_count == 1
        result = captured["results"][0]
        assert result.status == "failed"
        assert result.error == "NO leg FoK not filled: status=canceled"
        assert slept == []
        assert "killed by the exchange (HTTP 409" in caplog.text
        # The only per-ticker position reads are the two baselines taken
        # before the NO leg was submitted: none follows the kill
        ticker_reads = [
            c for c in client.get_positions_without_preload_content.call_args_list
            if "ticker" in c.kwargs
        ]
        assert len(ticker_reads) == 2
        client.create_order_without_preload_content.assert_not_called()

    def test_run_prod_live_v2_leg_b_kill_response_rolls_back_end_to_end(self, monkeypatch):
        client, captured = self._run(monkeypatch, ["full", "fok_kill", "full"])

        calls = client.rest_client.request.call_args_list
        assert len(calls) == 3
        rollback_body = calls[2].kwargs["body"]
        assert rollback_body["reduce_only"] is True
        assert rollback_body["time_in_force"] == "immediate_or_cancel"
        result = captured["results"][0]
        assert result.status == "rolled_back"
        assert result.error == "YES leg FoK not filled: status=canceled"
        client.create_order_without_preload_content.assert_not_called()

    def test_run_prod_live_v2_error_response_routes_to_position_lookup(self, monkeypatch):
        # Record what each order actually asked for, so the modelled position
        # move can be the YES leg's OWN count rather than a hardcoded number that
        # would silently drift with Kelly sizing.
        submitted: list = []
        base_orders = _order_side_effect(["full", "error"])

        def recording_orders(verb, url, headers=None, body=None):
            if TRANSFER_PATH not in url:
                submitted.append(body["count"])
            return base_orders(verb, url, headers=headers, body=body)

        client, captured = self._run(
            monkeypatch,
            ["full", "error"],
            order_side_effect=recording_orders,
            position_lookup_responses={
                # Delta semantics: the baseline read (taken before the NO leg is
                # submitted) must show FLAT, and the post-exception read must
                # show exactly the contracts the YES leg bought. A single static
                # payload would give a delta of 0 — a confirmed non-fill —
                # and roll the NO leg back, the opposite of what this pins.
                _TICKER_SAME_CHEAP: [
                    {"market_positions": []},
                    lambda: {
                        "market_positions": [
                            {"ticker": _TICKER_SAME_CHEAP, "position_fp": submitted[1]}
                        ]
                    },
                ],
            },
        )

        # NO leg filled, YES leg raised — exactly two order POSTs, no rollback.
        assert client.rest_client.request.call_count == 2

        lookup_calls = [
            c for c in client.get_positions_without_preload_content.call_args_list
            if c.kwargs.get("ticker") == _TICKER_SAME_CHEAP
        ]
        assert lookup_calls, "expected a position lookup for the YES-leg ticker"

        result = captured["results"][0]
        assert result.status == "executed"
        assert "ambiguous" in (result.error or "").lower()
        client.create_order_without_preload_content.assert_not_called()

    def test_run_prod_live_v2_routes_each_leg_to_its_own_shard(self, monkeypatch):
        # The behavioral point of multi-shard support: a pair whose legs live
        # on different shards is now tradeable, and each V2 body must carry
        # ITS OWN market's exchange_index — never one shared value, never the
        # -1 auto-route sentinel. Shard 1 is already funded here ($9,999), so
        # no collateral transfer is involved.
        client, captured = self._run(monkeypatch, ["full", "full"], same_cheap_shard=1)

        calls = client.rest_client.request.call_args_list
        assert len(calls) == 2
        assert all(V2_ORDER_PATH in c.args[1] for c in calls)
        assert all(TRANSFER_PATH not in c.args[1] for c in calls)

        by_ticker = {c.kwargs["body"]["ticker"]: c.kwargs["body"] for c in calls}
        assert by_ticker[_TICKER_SAME_EXP]["exchange_index"] == 0
        assert by_ticker[_TICKER_SAME_CHEAP]["exchange_index"] == 1

        assert len(captured["results"]) == 1
        assert captured["results"][0].status == "executed"
        client.create_order_without_preload_content.assert_not_called()

    def test_run_prod_dry_run_plans_but_never_posts_a_collateral_transfer(
        self, monkeypatch, caplog,
    ):
        # The YES leg (market B) sits on shard 1, which holds $0 — a genuine deficit. In dry-run
        # the plan must be logged and NOTHING posted: not the transfer, not the
        # orders.
        with caplog.at_level(logging.INFO):
            client, _ = self._run(
                monkeypatch,
                [],
                same_cheap_shard=1,
                balance_payload=_SHARD1_EMPTY_BALANCE,
                dry_run=True,
            )

        assert "DRY RUN: would transfer" in caplog.text
        assert "shard 0→1" in caplog.text
        assert client.rest_client.request.call_count == 0
        client.create_order_without_preload_content.assert_not_called()

    def test_run_prod_live_transfers_collateral_before_ordering(self, monkeypatch, caplog):
        # Same deficit, live: the transfer POST must land BEFORE any order
        # POST, and the settlement poll must see the funds actually arrive
        # (the second balance read) before execution proceeds.
        with caplog.at_level(logging.INFO):
            client, captured = self._run(
                monkeypatch,
                ["full", "full"],
                same_cheap_shard=1,
                balance_payload=_SHARD1_EMPTY_BALANCE,
                balance_payload_after=_SHARD1_SETTLED_BALANCE,
                order_side_effect=_transfer_and_order_side_effect(["full", "full"]),
            )

        urls = [c.args[1] for c in client.rest_client.request.call_args_list]
        assert len(urls) == 3
        assert TRANSFER_PATH in urls[0], "collateral must move before any order"
        assert all(V2_ORDER_PATH in u for u in urls[1:])

        transfer_body = client.rest_client.request.call_args_list[0].kwargs["body"]
        assert transfer_body["source_exchange_shard"] == 0
        assert transfer_body["destination_exchange_shard"] == 1
        assert transfer_body["amount"] > 0

        assert "Collateral transfer accepted" in caplog.text
        assert "All shard collateral requirements confirmed funded." in caplog.text

        order_bodies = {
            c.kwargs["body"]["ticker"]: c.kwargs["body"]
            for c in client.rest_client.request.call_args_list[1:]
        }
        assert order_bodies[_TICKER_SAME_EXP]["exchange_index"] == 0
        assert order_bodies[_TICKER_SAME_CHEAP]["exchange_index"] == 1

        assert captured["results"][0].status == "executed"
        client.create_order_without_preload_content.assert_not_called()

    def test_run_prod_drops_the_trade_when_its_shard_cannot_be_funded(
        self, monkeypatch, caplog,
    ):
        # Transfers reported inactive on shard 1 => the transfer is never
        # attempted, the only selected trade is dropped, and NO order is
        # submitted underfunded.
        client = _live_shape_client(
            monkeypatch,
            balance_payload=_SHARD1_EMPTY_BALANCE,
            same_cheap_shard=1,
            exchange_status_payload=_status_payload(
                _status_entry(0, description="Main"),
                _status_entry(1, transfers_active=False, description="Combos"),
            ),
            order_side_effect=_transfer_and_order_side_effect(["full", "full"]),
        )
        monkeypatch.setattr(
            main, "append_to_prod_log", lambda *a, **k: pathlib.Path("/fake/trade_log.xlsx"),
        )
        args = SimpleNamespace(dry_run=False, max_horizon_days=None)

        with caplog.at_level(logging.INFO):
            main._run_prod(client, args)

        assert "Intra-exchange transfers are not active" in caplog.text
        assert "No selected pair could be funded on its exchange shard" in caplog.text
        assert client.rest_client.request.call_count == 0
        client.create_order_without_preload_content.assert_not_called()


@pytest.mark.usefixtures("pinned_config_toggles")
class TestLiveSettingsReachEverySite:
    """The runtime tripwire behind test_strategy.py's
    test_ast_live_path_reads_toggles_only_through_live_settings: a prod dry run and a
    dev run handed _SETTINGS (every field departing from the pinned config) and an
    explicit reference while live_settings raises in config, scanner, strategy, trader
    and main, so a site reading config.py or the reference raises or hands a spy the
    wrong value. _SETTINGS keeps both pairs trading, the time-series f* between the
    0.25 same-title and 0.35 per-trade caps."""

    _SETTINGS = LiveSettings(tier_floors=False, spread_band=(0.05, 0.9),
                             interval_discount=0.6, size_cap=0.35, same_title_size_cap=0.25,
                             categories=("economics", "POLITICS"),
                             tags=("Inflation", "elections"))

    # The cached /series listing (SHARDAEVT and HELDAEVT unlisted: filed as Other)
    _LISTING = {
        "EVT": ["Economics", ["Inflation", "Fed"]],
        "EXPEVT": ["Politics", ["Elections"]],
        "TICKAEVT": ["Sports", ["Soccer"]],
    }

    # (module, spied name) — the CONSUMING modules' own bindings, and
    # config's, for the calls config's helpers make among themselves
    _SPIED = (
        (scanner_mod, ("live_time_series_floor", "max_affordable_pairs",
                       "max_kelly_fraction", "time_series_spread_refusal")),
        (strategy_mod, ("max_affordable_pairs", "time_series_profit_prob", "pair_size_cap")),
        (config, ("live_time_series_floor", "pair_size_cap")),
    )

    def _run_under_the_tripwire(self, monkeypatch, caplog, mode: str):
        """
        Run one mode on _SETTINGS under the tripwire, recording every _SPIED call.

        Returns:
            tuple: (settings, reference, calls, captured) — calls maps (module
                short name, function name) to each call's (args, kwargs);
                captured holds "results", "filter", "enriched" and, in prod,
                "run_note".
        """
        settings = self._SETTINGS
        # config.py's toggles, read before the tripwire; every field must differ
        reference = live_settings()
        for field in dataclasses.fields(LiveSettings):
            assert getattr(settings, field.name) != getattr(reference, field.name), field.name

        client = _live_shape_client(
            monkeypatch, balance_payload=_LIVE_BALANCE_PAYLOAD, include_time_series=True,
        )
        captured: dict = {}
        # A fresh cached listing, so a prod run makes no request for it
        _seed_series_listing(self._LISTING)
        real_filter = main._filter_by_category

        def filter_spy(pairs, settings_, listing_client):
            kept = real_filter(pairs, settings_, listing_client)
            captured.setdefault("filter", []).append(
                (list(pairs), settings_, listing_client, list(kept)))
            return kept

        monkeypatch.setattr(main, "_filter_by_category", filter_spy)
        # Neither mode may request it (recorded: a raise there only WARNs)
        series_requests: list = []

        def no_series_request(client_, path, **params):
            series_requests.append(path)
            raise AssertionError("the fresh /series listing was requested")

        monkeypatch.setattr(historical, "_historical_get", no_series_request)
        # The filter runs first: enrichment must be handed exactly its kept pairs
        real_enrich = main.enrich_with_orderbook_prices

        def enrich_spy(client_, pairs, *args, **kwargs):
            captured.setdefault("enriched", []).append(list(pairs))
            return real_enrich(client_, pairs, *args, **kwargs)

        monkeypatch.setattr(main, "enrich_with_orderbook_prices", enrich_spy)

        def fake_append_to_prod_log(results, balance_before, balance_after, *, run_note=""):
            captured["run_note"] = run_note
            return pathlib.Path("/fake/trade_log.xlsx")

        monkeypatch.setattr(main, "append_to_prod_log", fake_append_to_prod_log)
        monkeypatch.setattr(main, "write_dev_simulation",
                            lambda *a, **k: pathlib.Path("/fake/dev_sim.xlsx"))
        real_execute = main.execute_trades

        def execute_spy(client_, specs, dry_run=False):
            captured["results"] = real_execute(client_, specs, dry_run=dry_run)
            return captured["results"]

        monkeypatch.setattr(main, "execute_trades", execute_spy)

        # The tripwire: no module may resolve config.py's settings this run
        def tripwire(*args, **kwargs):
            raise AssertionError("live_settings() read during a run handed its settings")

        for module in (config, scanner_mod, strategy_mod, trader_mod, main):
            monkeypatch.setattr(module, "live_settings", tripwire)

        calls: dict = {}

        def spy(module, name):
            real = getattr(module, name)
            key = (module.__name__.rsplit(".", 1)[-1], name)

            def wrapper(*args, **kwargs):
                calls.setdefault(key, []).append((args, kwargs))
                return real(*args, **kwargs)

            monkeypatch.setattr(module, name, wrapper)

        for module, names in self._SPIED:
            for name in names:
                spy(module, name)

        with caplog.at_level(logging.INFO):
            if mode == "prod":
                code = main._run_prod(
                    client, SimpleNamespace(dry_run=True, max_horizon_days=None),
                    settings, reference,
                )
            else:
                code = main._run_dev(
                    client, SimpleNamespace(sandbox_balance=1000.0, max_horizon_days=None),
                    settings, reference,
                )
        assert code == EXIT_OK
        # The filter ran once, on the run's settings, keeping both pairs and only them
        ((pairs, filter_settings, listing_client, kept),) = captured["filter"]
        assert filter_settings is settings
        assert listing_client is (client if mode == "prod" else None)
        assert {p.market_a.ticker for p in kept} == {_TICKER_TS_EARLY, _TICKER_SAME_EXP}
        assert len(pairs) > len(kept)
        assert (f"Category/tag filter ({config.describe_trade_filter(settings)}): kept 2 of "
                f"{len(pairs)} candidate pairs") in caplog.text
        # ... before enrichment, which was handed exactly the pairs it kept
        (enriched,) = captured["enriched"]
        assert [id(p) for p in enriched] == [id(p) for p in kept]
        # ... from the fresh cached listing, with no request and no WARNING
        assert series_requests == []
        assert "Series category" not in caplog.text
        return settings, reference, calls, captured

    @staticmethod
    def _arg(call, index, keyword):
        args, kwargs = call
        return kwargs[keyword] if keyword in kwargs else args[index]

    def _assert_every_site_read_the_runs_settings(self, settings, calls):
        """Every spy got the run's settings (or values); both pair types reached sizing."""
        arg = self._arg
        for module, names in self._SPIED:
            for name in names:
                key = (module.__name__.rsplit(".", 1)[-1], name)
                assert calls.get(key), f"{key} was never called"
        for key, index in ((("scanner", "live_time_series_floor"), 1),
                           (("config", "live_time_series_floor"), 1),
                           (("scanner", "time_series_spread_refusal"), 2),
                           (("scanner", "max_kelly_fraction"), 1)):
            for call in calls[key]:
                assert arg(call, index, "settings") is settings, key
        for call in calls[("strategy", "time_series_profit_prob")]:
            assert arg(call, 2, "k") == settings.interval_discount
        for key in (("strategy", "pair_size_cap"), ("config", "pair_size_cap")):
            for call in calls[key]:
                assert arg(call, 1, "size_cap") == settings.size_cap, key
                assert arg(call, 2, "same_title_size_cap") == settings.same_title_size_cap, key
        # Enrichment's affordability bound is the run's, per pair type
        bounds = {arg(c, 2, "fraction") for c in calls[("scanner", "max_affordable_pairs")]}
        assert bounds == {0.35, 0.25}
        assert {arg(c, 0, "pair_type") for c in calls[("scanner", "max_kelly_fraction")]} >= {
            "time_series", "same_title"}
        # The sizer's fractions sit under the run's caps, above config.py's
        fractions = {arg(c, 2, "fraction") for c in calls[("strategy", "max_affordable_pairs")]}
        assert all(0 < f <= settings.size_cap for f in fractions), fractions
        assert settings.same_title_size_cap in fractions
        assert any(settings.same_title_size_cap < f < settings.size_cap for f in fractions)
        sized = {arg(c, 0, "pair_type") for c in calls[("strategy", "pair_size_cap")]}
        assert sized == {"time_series", "same_title"}

    def test_a_run_handed_its_settings_reads_them_at_every_site(self, monkeypatch, caplog):
        settings, reference, calls, captured = self._run_under_the_tripwire(
            monkeypatch, caplog, "prod")
        self._assert_every_site_read_the_runs_settings(settings, calls)

        results = captured["results"]
        assert {r.status for r in results} == {"simulated"}
        by_type = {r.spec.pair.pair_type: r.spec for r in results}
        assert set(by_type) == {"time_series", "same_title"}
        assert by_type["time_series"].pair.market_a.ticker == _TICKER_TS_EARLY
        assert by_type["same_title"].pair.market_a.ticker == _TICKER_SAME_EXP
        assert by_type["time_series"].kelly_p == pytest.approx(1 - 0.6 * 0.30)
        assert settings.same_title_size_cap < by_type["time_series"].kelly_fraction < 0.35
        assert by_type["same_title"].kelly_fraction == settings.same_title_size_cap

        # pre_execution_check swallows exceptions — so none may have happened
        assert "Pre-execution check raised" not in caplog.text
        # The run named its rule, its settings (all marked) and both lifted exposures
        assert ("Time-series entry rule: "
                + config.describe_time_series_rule(False, (0.05, 0.9))) in caplog.text
        echo = f"Live settings: {describe_live_settings(settings, reference)}"
        assert echo in caplog.text and echo.count("(config:") == 7
        assert "This PRODUCTION run overrides" not in caplog.text
        assert "one time-series pair may stake up to 35%" in caplog.text
        assert "one same-title pair may stake up to 25%" in caplog.text
        # The workbook's separator row carries the same marked line
        assert captured["run_note"] == f"settings: {describe_live_settings(settings, reference)}"
        assert captured["run_note"].count("(config:") == 7

    def test_a_dev_run_handed_its_settings_reads_them_at_every_site(self, monkeypatch, caplog):
        settings, reference, calls, captured = self._run_under_the_tripwire(
            monkeypatch, caplog, "dev")
        self._assert_every_site_read_the_runs_settings(settings, calls)

        results = captured["results"]
        assert {r.status for r in results} == {"simulated"}
        by_ticker = {r.spec.pair.market_a.ticker: r.spec for r in results}
        ts, st = by_ticker[_TICKER_TS_EARLY], by_ticker[_TICKER_SAME_EXP]
        assert ts.pair.pair_type == "time_series" and st.pair.pair_type == "same_title"
        assert ts.kelly_p == pytest.approx(1 - 0.6 * 0.30)
        assert settings.same_title_size_cap < ts.kelly_fraction < 0.35
        assert st.kelly_fraction == settings.same_title_size_cap
        assert all(r.spec.kelly_fraction <= settings.same_title_size_cap
                   for r in results if r.spec.pair.pair_type == "same_title")
        assert ("Time-series entry rule: "
                + config.describe_time_series_rule(False, (0.05, 0.9))) in caplog.text
        echo = f"Live settings: {describe_live_settings(settings, reference)}"
        assert echo in caplog.text and echo.count("(config:") == 7
        # Dev never submits an order, so never the production WARNING
        assert "This PRODUCTION run overrides" not in caplog.text

    @pytest.mark.parametrize("mode", ["prod", "dev"])
    def test_a_run_with_no_pairs_names_its_own_rule(self, monkeypatch, caplog, mode):
        settings = dataclasses.replace(live_settings(), tier_floors=False,
                                       spread_band=(0.0, 0.5))
        reference = live_settings()
        client = _live_shape_client(monkeypatch, balance_payload=_LIVE_BALANCE_PAYLOAD)
        monkeypatch.setattr(main, "enrich_with_orderbook_prices", lambda *a, **k: [])
        monkeypatch.setattr(main, "write_dev_simulation",
                            lambda *a, **k: pathlib.Path("/fake/dev_sim.xlsx"))
        with caplog.at_level(logging.INFO):
            if mode == "prod":
                code = main._run_prod(client, SimpleNamespace(dry_run=True, max_horizon_days=None),
                                      settings, reference)
            else:
                code = main._run_dev(
                    client, SimpleNamespace(sandbox_balance=1000.0, max_horizon_days=None),
                    settings, reference)
        assert code == EXIT_OK
        (line,) = [r.getMessage() for r in caplog.records
                   if r.getMessage().startswith("No qualifying pairs found")]
        assert config.describe_time_series_rule(False, (0.0, 0.5)) in line, line
        assert config.describe_time_series_rule(True, (0.0, 1.0)) not in line, line
        assert ("in sandbox" in line) is (mode == "dev")


# A third deadline of the TS-EARLY / TS-LATE question, in an event of its own
_TICKER_TS_MID = "TS-MID"
_TS_MID_MARKET = _mk_market(
    _TICKER_TS_MID, "EVT-TS-MID", "Will Z happen by December 6, 2026?", "Outcome",
    "0.45", "0.55", price_level_structure="linear_cent", close_time="2026-12-06T00:00:00Z",
)


@pytest.mark.usefixtures("pinned_config_toggles")
class TestRunProdHeldLadders:
    """A production run makes no new time-series trade on a ladder it already
    holds, whether or not the held market is in this run's market list. If a
    held market cannot be looked up, the run makes no time-series trade at
    all."""

    @staticmethod
    def _dry_run(client, monkeypatch, caplog, expected_code=EXIT_OK) -> list:
        """Run a production dry run, check its exit code, and return the
        simulated trades' results."""
        captured: dict = {}

        def fake_append_to_prod_log(results, balance_before, balance_after, *, run_note=""):
            captured["results"] = results
            return pathlib.Path("/fake/trade_log.xlsx")

        monkeypatch.setattr(main, "append_to_prod_log", fake_append_to_prod_log)
        with caplog.at_level(logging.INFO):
            code = main._run_prod(client, _args(dry_run=True))
        assert code == expected_code
        return captured.get("results", [])

    @staticmethod
    def _traded(results) -> set:
        return {(r.spec.pair.pair_type, r.spec.pair.market_a.ticker) for r in results}

    def test_the_time_series_pair_trades_when_no_ladder_is_held(self, monkeypatch, caplog):
        # control for the tests below: the same books trade both kinds of pair
        client = _live_shape_client(monkeypatch, balance_payload=_LIVE_BALANCE_PAYLOAD,
                                    include_time_series=True)
        results = self._dry_run(client, monkeypatch, caplog)
        assert self._traded(results) == {("time_series", _TICKER_TS_EARLY),
                                         ("same_title", _TICKER_SAME_EXP)}
        # HELD-A is in the market list, so nothing was looked up
        assert ("Open ladder exposure: 1 held market(s) in 1 event(s), asking 1 question(s) "
                "(0 looked up") in caplog.text
        client.get_market_without_preload_content.assert_not_called()

    def test_a_run_that_holds_nothing_still_trades_time_series(self, monkeypatch, caplog):
        # An empty held set is a lookup that worked, not one that failed
        client = _live_shape_client(monkeypatch, balance_payload=_LIVE_BALANCE_PAYLOAD,
                                    include_time_series=True, include_held_position=False)
        results = self._dry_run(client, monkeypatch, caplog)
        assert ("time_series", _TICKER_TS_EARLY) in self._traded(results)
        assert ("Open ladder exposure: 0 held market(s) in 0 event(s), asking 0 question(s) "
                "(0 looked up") in caplog.text
        assert "on a ladder we already hold" not in caplog.text

    def test_a_held_rung_in_the_market_list_blocks_its_ladder(self, monkeypatch, caplog):
        client = _live_shape_client(
            monkeypatch, balance_payload=_LIVE_BALANCE_PAYLOAD, include_time_series=True,
            extra_events=(_ev("TS Event", _TS_MID_MARKET),), extra_held=(_TICKER_TS_MID,),
        )
        results = self._dry_run(client, monkeypatch, caplog)
        # The same-title trade goes ahead; the time-series pair shares the held question
        assert self._traded(results) == {("same_title", _TICKER_SAME_EXP)}
        assert ("Open ladder exposure: 2 held market(s) in 2 event(s), asking 2 question(s) "
                "(0 looked up") in caplog.text
        assert ("Time-series candidates refused because one of their markets is on a "
                "ladder we already hold (counted before every other check): 1") in caplog.text
        client.get_market_without_preload_content.assert_not_called()

    def test_a_held_rung_missing_from_the_market_list_is_looked_up(self, monkeypatch, caplog):
        client = _live_shape_client(
            monkeypatch, balance_payload=_LIVE_BALANCE_PAYLOAD, include_time_series=True,
            extra_held=(_TICKER_TS_MID,),
        )
        # The market is closed but not yet paid out, so the run's list lacks it
        client.get_market_without_preload_content = MagicMock(
            return_value=_raw_json_response({"market": {**_TS_MID_MARKET, "status": "closed"}}))
        client.get_event_without_preload_content = MagicMock(
            return_value=_raw_json_response({"event": {"event_ticker": "EVT-TS-MID",
                                                       "title": "TS Event"}}))
        results = self._dry_run(client, monkeypatch, caplog)
        assert self._traded(results) == {("same_title", _TICKER_SAME_EXP)}
        client.get_market_without_preload_content.assert_called_once_with(ticker=_TICKER_TS_MID)
        client.get_event_without_preload_content.assert_called_once_with(
            event_ticker="EVT-TS-MID")
        assert "(1 looked up because this run's market list did not have them)" in caplog.text
        assert ("ladder we already hold (counted before every other check): 1") in caplog.text

    def test_a_failed_lookup_stops_every_time_series_trade(self, monkeypatch, caplog):
        client = _live_shape_client(
            monkeypatch, balance_payload=_LIVE_BALANCE_PAYLOAD, include_time_series=True,
            extra_held=("TS-GONE",),
        )
        client.get_market_without_preload_content = MagicMock(
            return_value=_raw_json_response({"error": "not found"}, status=404,
                                            reason="Not Found"))
        # The run exits with its own code, so the scheduler's log says so too
        results = self._dry_run(client, monkeypatch, caplog,
                                expected_code=EXIT_TIME_SERIES_SKIPPED)
        # Same-title trades still go through; no time-series pair is even searched for
        assert self._traded(results) == {("same_title", _TICKER_SAME_EXP)}
        errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert any("Could not look up held market 'TS-GONE'" in e
                   and "no time-series trade will be made this run" in e for e in errors), errors
        assert "Time-series entry rule:" not in caplog.text
        assert "Open ladder exposure" not in caplog.text

    def test_a_failed_lookup_with_no_pair_says_time_series_was_not_searched(
            self, monkeypatch, caplog):
        client = _live_shape_client(
            monkeypatch, balance_payload=_LIVE_BALANCE_PAYLOAD, include_time_series=True,
            extra_held=("TS-GONE",),
        )
        client.get_market_without_preload_content = MagicMock(
            return_value=_raw_json_response({"error": "not found"}, status=404,
                                            reason="Not Found"))
        # No same-title pair either, so the run ends on its "no pairs" line
        monkeypatch.setattr(main, "find_same_title_pairs", lambda markets, held: [])
        assert self._dry_run(client, monkeypatch, caplog,
                             expected_code=EXIT_TIME_SERIES_SKIPPED) == []
        [line] = [r.getMessage() for r in caplog.records
                  if r.getMessage().startswith("No qualifying pairs found")]
        # The line does not describe a time-series rule the run never applied
        assert ("time-series: not searched this run, because a held market could not "
                "be identified") in line
        assert "worded as cumulative deadlines" not in line


def _args(dry_run: bool = False, max_horizon_days=None) -> SimpleNamespace:
    """Minimal stand-in for the argparse.Namespace _run_prod/_run_dev read."""
    return SimpleNamespace(dry_run=dry_run, max_horizon_days=max_horizon_days)


def make_spec() -> SimpleNamespace:
    """Minimal TradeSpec-like stub with concrete (non-Mock) scalar fields.

    print_pairs_table / _print_portfolio format several fields with a format
    spec (e.g. f"{pair.pA:.2%}") — a bare MagicMock's default __format__
    support is unreliable, so pair/spec fields are plain SimpleNamespace
    values instead of auto-attributing MagicMocks.
    """
    # A coherent time-series pair under the 2026-09 direction: the LATER
    # contract (B) is priced above the earlier (pA=0.30 -> pB=0.60); the legs
    # bought are YES on A at pA and NO on B at nB=0.40; nA=0.70 is A's
    # reporting-only NO ask. nB is a REAL float: print_pairs_table renders it
    # and scanner.leg_prices reads it directly (no getattr default).
    pair = SimpleNamespace(
        pair_type="time_series",
        market_a=SimpleNamespace(
            ticker="TICK-A", title="Market A", subtitle="", close_time=None,
            exchange_index=DEFAULT_EXCHANGE_INDEX,
        ),
        market_b=SimpleNamespace(
            ticker="TICK-B", title="Market B", subtitle="", close_time=None,
            exchange_index=DEFAULT_EXCHANGE_INDEX,
        ),
        pA=0.30,
        pB=0.60,
        nA=0.70,
        nB=0.40,
        tradeable=True,
        canonical_title="Test pair",
    )
    return SimpleNamespace(
        pair=pair,
        x=5,
        y=5,
        total_cost=2.0,
        # Fee-INCLUSIVE, and deliberately different from total_cost: every
        # human-facing cost line reports this one (TS-12), so a fixture where
        # the two matched would let a regression back to total_cost pass.
        # Equals cost_with_fees_a + cost_with_fees_b, the documented invariant.
        total_cost_with_fees=2.30,
        # Per-leg fee-inclusive costs — trader._required_cents_by_shard reads
        # these to total the collateral each shard must hold before execution
        cost_with_fees_a=1.15,
        cost_with_fees_b=1.15,
        min_payoff=0.50,
        profit_ratio=0.10,
        monthly_profit_ratio=0.20,
        kelly_fraction=0.10,
        kelly_p=0.60,
    )


class TestPairsTableOutcomeColumns:
    """DR-17: the outcome label gets its own pair of table cells.

    Appending it to the title cell alone is not enough — _truncate cuts the
    two title cells at 40 characters (the outcome cells it adds get their own
    24), and a real display_title ("<event title>: <market title> —
    <subtitle>") is well past that, so two strikes of one daily family render
    as the same truncated string. The reviewer of the 2026-09-15 dry run could
    not tell the four cross-strike trades apart for exactly that reason.
    """

    @staticmethod
    def _cells(caplog, sub_a: str, sub_b: str) -> dict:
        """Render one candidate row and return {header: cell text}."""
        pair = make_spec().pair
        pair.market_a.subtitle = sub_a
        pair.market_b.subtitle = sub_b
        caplog.clear()
        with caplog.at_level(logging.INFO):
            main.print_pairs_table([pair], {})
        # tabulate's "rounded_outline" draws the header row and the single data
        # row as the only lines carrying cell separators.
        lines = [line for line in caplog.text.splitlines() if "\u2502" in line]
        assert len(lines) == 2, lines
        headers = [c.strip() for c in lines[0].split("\u2502")[1:-1]]
        values = [c.strip() for c in lines[1].split("\u2502")[1:-1]]
        assert len(headers) == len(values)
        return dict(zip(headers, values, strict=True))

    def test_outcome_cells_carry_the_subtitles(self, caplog):
        cells = self._cells(caplog, "$82,750 or above", "$78,500 or above")
        assert cells["Outcome A"] == "$82,750 or above"
        assert cells["Outcome B"] == "$78,500 or above"

    def test_two_strikes_of_one_family_are_distinguishable(self, caplog):
        # The titles are identical by construction here (make_spec's stub pair
        # uses "Market A"/"Market B"); the outcome cells are the only thing
        # that separates one strike from another.
        cells = self._cells(caplog, "$82,750 or above", "$82,750 or above")
        same = self._cells(caplog, "$82,750 or above", "$78,500 or above")
        assert cells["Outcome A"] == cells["Outcome B"]
        assert same["Outcome A"] != same["Outcome B"]

    def test_missing_subtitle_renders_a_placeholder(self, caplog):
        cells = self._cells(caplog, "", "")
        assert cells["Outcome A"] == "\u2014"
        assert cells["Outcome B"] == "\u2014"

    def test_headers_and_row_stay_aligned(self, caplog):
        # The row cells and the header list are built in two separate places;
        # a column added to one and not the other silently shifts every price
        # column. tabulate pads the SHORTER list (with a blank header at the
        # front), so the two stay the same length and neither the length
        # assertion nor the strict zip in _cells fires — the shift surfaces as
        # a known header carrying its neighbour's value, which is what the
        # per-value assertions below catch.
        cells = self._cells(caplog, "Yes", "Yes")
        assert cells["Market A"].startswith("Market A")
        assert cells["pA (YES)"] == "30.00%"
        assert cells["pB (YES)"] == "60.00%"
        assert cells["nB (NO)"] == "40.00%"


class TestRunProdExitCodes:
    @patch("kalshi_betting.main.verify_auth")
    def test_low_balance_returns_skip_code(self, mock_verify_auth):
        # Balance below MIN_BALANCE_CENTS must short-circuit before any scan —
        # the bare `return` this used to be silently exited 0.
        mock_verify_auth.return_value = {DEFAULT_EXCHANGE_INDEX: MIN_BALANCE_CENTS - 1}
        client = MagicMock()

        code = main._run_prod(client, _args())

        assert code == EXIT_SKIPPED_LOW_BALANCE
        assert code == 10

    @patch("kalshi_betting.main.append_to_prod_log")
    @patch("kalshi_betting.main.execute_trades")
    @patch("kalshi_betting.main.pre_execution_check")
    @patch("kalshi_betting.main.select_portfolio")
    @patch("kalshi_betting.main.compute_trade")
    @patch("kalshi_betting.main.enrich_with_orderbook_prices")
    @patch("kalshi_betting.main.find_same_title_pairs")
    @patch("kalshi_betting.main.find_time_series_pairs")
    @patch("kalshi_betting.main.filter_markets_within_horizon")
    @patch("kalshi_betting.main.fetch_shard_statuses", return_value=None)
    @patch("kalshi_betting.main.fetch_open_events_with_markets")
    @patch("kalshi_betting.main.get_held_tickers")
    @patch("kalshi_betting.main.verify_auth")
    def test_manual_review_result_returns_attention_code(
        self,
        mock_verify_auth,
        mock_held,
        mock_fetch,
        mock_shard_statuses,
        mock_filter_horizon,
        mock_find_ts,
        mock_find_st,
        mock_enrich,
        mock_compute,
        mock_select,
        mock_pre_exec,
        mock_execute,
        mock_append_log,
    ):
        # Two verify_auth calls: pre-trade balance, then post-trade balance
        # for the log's separator row.
        mock_verify_auth.side_effect = [
            {DEFAULT_EXCHANGE_INDEX: 100_000},
            {DEFAULT_EXCHANGE_INDEX: 100_000},
        ]
        mock_held.return_value = set()
        mock_fetch.return_value = _stub_ingest()
        mock_filter_horizon.side_effect = lambda markets, days: markets
        mock_find_ts.return_value = []
        spec = make_spec()
        mock_find_st.return_value = [spec.pair]
        mock_enrich.return_value = [spec.pair]
        mock_compute.return_value = spec
        mock_select.return_value = [spec]
        mock_pre_exec.side_effect = lambda client, portfolio, *, settings: portfolio
        mock_execute.return_value = [
            TradeResult(spec=spec, status="manual_review", error="position lookup failed"),
        ]
        mock_append_log.return_value = "trade_log.xlsx"

        client = MagicMock()
        code = main._run_prod(client, _args(dry_run=False))

        assert code == EXIT_TRADES_NEED_ATTENTION
        assert code == 20

    @patch("kalshi_betting.main.append_to_prod_log")
    @patch("kalshi_betting.main.execute_trades")
    @patch("kalshi_betting.main.pre_execution_check")
    @patch("kalshi_betting.main.select_portfolio")
    @patch("kalshi_betting.main.compute_trade")
    @patch("kalshi_betting.main.enrich_with_orderbook_prices")
    @patch("kalshi_betting.main.find_same_title_pairs")
    @patch("kalshi_betting.main.find_time_series_pairs")
    @patch("kalshi_betting.main.filter_markets_within_horizon")
    @patch("kalshi_betting.main.fetch_shard_statuses", return_value=None)
    @patch("kalshi_betting.main.fetch_open_events_with_markets")
    @patch("kalshi_betting.main.get_held_tickers")
    @patch("kalshi_betting.main.verify_auth")
    def test_a_disproof_with_the_rest_of_the_run_stopped_returns_attention_code(
        self,
        mock_verify_auth,
        mock_held,
        mock_fetch,
        mock_shard_statuses,
        mock_filter_horizon,
        mock_find_ts,
        mock_find_st,
        mock_enrich,
        mock_compute,
        mock_select,
        mock_pre_exec,
        mock_execute,
        mock_append_log,
        caplog,
    ):
        # After a V2 NO-leg mapping disproof, trader stops every later pair
        # with status "failed" (nothing sent). The run still exits 20 through
        # the disproving pair's manual_review, and the undetermined-fill count
        # names only that one pair, not the pairs that sent nothing.
        mock_verify_auth.side_effect = [
            {DEFAULT_EXCHANGE_INDEX: 100_000},
            {DEFAULT_EXCHANGE_INDEX: 100_000},
        ]
        mock_held.return_value = set()
        mock_fetch.return_value = _stub_ingest()
        mock_filter_horizon.side_effect = lambda markets, days: markets
        mock_find_ts.return_value = []
        spec = make_spec()
        mock_find_st.return_value = [spec.pair]
        mock_enrich.return_value = [spec.pair]
        mock_compute.return_value = spec
        mock_select.return_value = [spec]
        mock_pre_exec.side_effect = lambda client, portfolio, *, settings: portfolio
        stopped = (
            "NO leg not sent: V2 NO-leg mapping disproven earlier in this run;"
            " nothing submitted"
        )
        mock_execute.return_value = [
            TradeResult(spec=spec, status="manual_review", error="V2 NO-leg mapping disproven"),
            TradeResult(spec=spec, status="failed", error=stopped),
            TradeResult(spec=spec, status="failed", error=stopped),
        ]
        mock_append_log.return_value = "trade_log.xlsx"

        with caplog.at_level(logging.INFO, logger="root"):
            code = main._run_prod(MagicMock(), _args(dry_run=False))

        assert code == EXIT_TRADES_NEED_ATTENTION
        assert any(
            "0 pair(s) may have ORPHANED positions and 1 pair(s) have an UNDETERMINED"
            in r.getMessage()
            for r in caplog.records
        )

    @patch("kalshi_betting.main.append_to_prod_log")
    @patch("kalshi_betting.main.execute_trades")
    @patch("kalshi_betting.main.pre_execution_check")
    @patch("kalshi_betting.main.select_portfolio")
    @patch("kalshi_betting.main.compute_trade")
    @patch("kalshi_betting.main.enrich_with_orderbook_prices")
    @patch("kalshi_betting.main.find_same_title_pairs")
    @patch("kalshi_betting.main.find_time_series_pairs")
    @patch("kalshi_betting.main.filter_markets_within_horizon")
    @patch("kalshi_betting.main.fetch_shard_statuses", return_value=None)
    @patch("kalshi_betting.main.fetch_open_events_with_markets")
    @patch("kalshi_betting.main.get_held_tickers")
    @patch("kalshi_betting.main.verify_auth")
    def test_clean_dry_run_returns_ok_code(
        self,
        mock_verify_auth,
        mock_held,
        mock_fetch,
        mock_shard_statuses,
        mock_filter_horizon,
        mock_find_ts,
        mock_find_st,
        mock_enrich,
        mock_compute,
        mock_select,
        mock_pre_exec,
        mock_execute,
        mock_append_log,
    ):
        mock_verify_auth.side_effect = [
            {DEFAULT_EXCHANGE_INDEX: 100_000},
            {DEFAULT_EXCHANGE_INDEX: 100_000},
        ]
        mock_held.return_value = set()
        mock_fetch.return_value = _stub_ingest()
        mock_filter_horizon.side_effect = lambda markets, days: markets
        mock_find_ts.return_value = []
        spec = make_spec()
        mock_find_st.return_value = [spec.pair]
        mock_enrich.return_value = [spec.pair]
        mock_compute.return_value = spec
        mock_select.return_value = [spec]
        mock_pre_exec.side_effect = lambda client, portfolio, *, settings: portfolio
        mock_execute.return_value = [TradeResult(spec=spec, status="simulated")]
        mock_append_log.return_value = "trade_log.xlsx"

        client = MagicMock()
        code = main._run_prod(client, _args(dry_run=True))

        assert code == EXIT_OK
        assert code == 0

    @patch("kalshi_betting.main.verify_auth")
    def test_no_qualifying_pairs_returns_ok_code(self, mock_verify_auth):
        # No-pairs / no-executable-trades paths must also resolve to EXIT_OK,
        # not just the low-balance and post-execution paths.
        mock_verify_auth.return_value = {DEFAULT_EXCHANGE_INDEX: 100_000}
        with (
            patch("kalshi_betting.main.get_held_tickers", return_value=set()),
            patch("kalshi_betting.main.fetch_shard_statuses", return_value=None),
            patch(
                "kalshi_betting.main.fetch_open_events_with_markets",
                return_value=_stub_ingest(),
            ),
            patch("kalshi_betting.main.filter_markets_within_horizon", side_effect=lambda m, d: m),
            patch("kalshi_betting.main.find_time_series_pairs", return_value=[]),
            patch("kalshi_betting.main.find_same_title_pairs", return_value=[]),
        ):
            code = main._run_prod(MagicMock(), _args())

        assert code == EXIT_OK


class TestRunProdTimeSeriesSkippedCode:
    """A production run that could not identify a held market makes no
    time-series trade and exits EXIT_TIME_SERIES_SKIPPED, not EXIT_OK, so the
    scheduler's own log says so. A trade that needs a human still wins."""

    @staticmethod
    def _run(monkeypatch, *, dry_run, status, held_ladders=None, same_title=True):
        """Run _run_prod with every request stubbed and one same-title trade
        (none when same_title is False) that ends with the given status;
        return (exit code, the time-series finder's calls)."""
        spec = make_spec()
        ts_calls = []
        monkeypatch.setattr(main, "verify_auth",
                            lambda client: {DEFAULT_EXCHANGE_INDEX: 100_000})
        monkeypatch.setattr(main, "get_held_tickers", lambda client: {"HELD-X"})
        monkeypatch.setattr(main, "fetch_shard_statuses", lambda client: None)
        monkeypatch.setattr(main, "fetch_open_events_with_markets",
                            lambda client, inactive_shards: _stub_ingest())
        monkeypatch.setattr(main, "resolve_held_ladders",
                            lambda client, markets, held: held_ladders)
        monkeypatch.setattr(main, "filter_markets_within_horizon", lambda m, d: m)
        monkeypatch.setattr(main, "find_time_series_pairs",
                            lambda *a, **k: ts_calls.append(k) or [])
        monkeypatch.setattr(main, "find_same_title_pairs",
                            lambda markets, held: [spec.pair] if same_title else [])
        monkeypatch.setattr(main, "enrich_with_orderbook_prices",
                            lambda client, pairs, balance, *, settings: pairs)
        monkeypatch.setattr(main, "compute_trade", lambda pair, balance, *, settings: spec)
        monkeypatch.setattr(main, "select_portfolio",
                            lambda specs, balance, *, held_ladders: specs)
        monkeypatch.setattr(main, "pre_execution_check",
                            lambda client, portfolio, *, settings: portfolio)
        monkeypatch.setattr(main, "execute_trades", lambda client, portfolio, *, dry_run: [
            TradeResult(spec=s, status=status) for s in portfolio])
        monkeypatch.setattr(main, "append_to_prod_log",
                            lambda *a, **k: pathlib.Path("/fake/trade_log.xlsx"))
        return main._run_prod(MagicMock(), _args(dry_run=dry_run)), ts_calls

    @pytest.mark.parametrize("dry_run, status", [(True, "simulated"), (False, "executed")])
    def test_a_failed_lookup_exits_with_its_own_code(self, monkeypatch, dry_run, status):
        code, ts_calls = self._run(monkeypatch, dry_run=dry_run, status=status)
        assert code == EXIT_TIME_SERIES_SKIPPED == 40
        assert ts_calls == []

    @pytest.mark.parametrize("dry_run, status", [(True, "simulated"), (False, "executed")])
    def test_a_lookup_that_worked_still_exits_ok(self, monkeypatch, dry_run, status):
        code, ts_calls = self._run(monkeypatch, dry_run=dry_run, status=status,
                                   held_ladders=frozenset())
        assert code == EXIT_OK
        assert len(ts_calls) == 1

    def test_a_trade_needing_a_human_wins(self, monkeypatch):
        code, _ = self._run(monkeypatch, dry_run=False, status="manual_review")
        assert code == EXIT_TRADES_NEED_ATTENTION

    def test_no_pair_at_all_still_exits_with_its_own_code(self, monkeypatch):
        code, _ = self._run(monkeypatch, dry_run=True, status="simulated", same_title=False)
        assert code == EXIT_TIME_SERIES_SKIPPED

    def test_every_clean_exit_after_the_lookup_uses_the_run_code(self):
        # Every return in _run_prod after clean_exit is set returns it or the
        # manual-review code, so no later path can report EXIT_OK instead.
        tree = ast.parse(inspect.getsource(main._run_prod).lstrip())
        fn = tree.body[0]
        assigned = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "clean_exit"
                            for t in n.targets)]
        assert len(assigned) == 1
        after = assigned[0].lineno
        returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return) and n.lineno > after]
        assert len(returns) >= 7
        for ret in returns:
            assert isinstance(ret.value, ast.Name), ast.dump(ret)
            assert ret.value.id in {"clean_exit", "EXIT_TRADES_NEED_ATTENTION"}, ret.value.id


class TestBlindRunReason:
    """VI-02: a run whose ingest produced nothing scanned nothing, and must
    report EXIT_NO_TRADEABLE_SHARDS rather than EXIT_OK. EXIT_OK claims
    "scanned everything, found no edge", which lets scheduler.run_job record
    the weekly slot as satisfied by a run that never looked at a single order
    book — TS-01's bug surviving through a second door, because
    scanner.fetch_shard_statuses is fail-soft and returns None on ANY internal
    failure, which makes the all-shards-halted test unevaluable.
    """

    _HALTED = {0: {"trading_active": False}, 1: {"trading_active": False}}
    _ACTIVE = {0: {"trading_active": True}, 1: {"trading_active": True}}

    def test_halt_fires_even_when_a_stray_market_survived_ingest(self):
        # The disjunct that must NEVER be collapsed into a bare census test.
        # /exchange/status can advertise every shard halted while a market
        # tagged with an UNADVERTISED shard still survives ingest:
        # scanner._shard_index only defaults a MISSING/unparseable index to
        # DEFAULT_EXCHANGE_INDEX, so an explicit shard 9 stays 9, and 9 is not
        # in inactive_shards. Reading that run as healthy would carry it into
        # pair discovery, Kelly sizing and — on a live Monday run — order
        # submission with an explicit exchange_index during an exchange-wide
        # halt.
        stray = [SimpleNamespace(ticker="STRAY", exchange_index=9)]
        reason = main._blind_run_reason(stray, self._HALTED, {0, 1})
        assert reason is not None
        assert "Every advertised exchange shard is trading-inactive ([0, 1])" in reason

    def test_zero_census_fires_when_status_was_unavailable(self):
        # The VI-02 hole itself: status is None, so the halt disjunct cannot
        # fire, and an ingest that came back empty used to exit EXIT_OK.
        assert main._blind_run_reason([], None, set()) == (
            "Ingest produced zero markets — nothing can be scanned this run"
        )

    def test_zero_census_fires_on_a_fully_active_exchange(self):
        # Nothing halted, nothing ingested — API drift (a renamed events or
        # markets key empties the ingest silently) rather than a halt, but
        # equally blind.
        assert main._blind_run_reason([], self._ACTIVE, set()) is not None

    def test_halt_is_reported_in_preference_to_the_census(self):
        # Both causes hold during a real halt; the halt is the more specific
        # diagnosis, so that is the sentence an operator reads.
        reason = main._blind_run_reason([], self._HALTED, {0, 1})
        assert "trading-inactive" in reason
        assert "Ingest produced zero markets" not in reason

    def test_a_run_that_ingested_something_is_never_blind(self):
        markets = _stub_ingest()
        assert main._blind_run_reason(markets, self._ACTIVE, set()) is None
        assert main._blind_run_reason(markets, None, set()) is None
        # One halted shard out of two is a PARTIAL ingest — reported by the
        # coverage check, never by this gate.
        partial = {0: {"trading_active": True}, 1: {"trading_active": False}}
        assert main._blind_run_reason(markets, partial, {1}) is None

    def test_empty_status_dict_cannot_fire_the_halt_disjunct(self):
        # fetch_shard_statuses never returns {} (it returns None for that
        # shape), but the `shard_statuses and` guard is what stops
        # set() == set({}) from reading as "every advertised shard halted".
        assert main._blind_run_reason(_stub_ingest(), {}, set()) is None


class TestBlindRunCensusEndToEnd:
    def test_run_prod_zero_market_ingest_returns_blind_code(self, caplog):
        # An empty prod ingest with the status breakdown unavailable: the
        # all-halted disjunct cannot fire, so only the census catches it.
        with (
            patch(
                "kalshi_betting.main.verify_auth",
                return_value={DEFAULT_EXCHANGE_INDEX: MIN_BALANCE_CENTS * 10},
            ),
            patch("kalshi_betting.main.get_held_tickers", return_value=set()),
            patch("kalshi_betting.main.fetch_shard_statuses", return_value=None),
            patch("kalshi_betting.main.fetch_open_events_with_markets", return_value=[]),
            patch("kalshi_betting.main.enrich_with_orderbook_prices") as mock_enrich,
        ):
            with caplog.at_level(logging.INFO):
                code = main._run_prod(MagicMock(), _args())

        assert code == EXIT_NO_TRADEABLE_SHARDS
        assert "Ingest produced zero markets" in caplog.text
        assert mock_enrich.call_count == 0, "a blind run short-circuits before enrichment"

    def test_run_dev_zero_market_ingest_returns_blind_code(self, caplog):
        wrote: list = []
        with (
            patch("kalshi_betting.main.fetch_shard_statuses", return_value=None),
            patch("kalshi_betting.main.fetch_open_events_with_markets", return_value=[]),
            patch(
                "kalshi_betting.main.write_dev_simulation",
                side_effect=lambda *a, **k: wrote.append(a),
            ),
        ):
            with caplog.at_level(logging.INFO):
                code = main._run_dev(
                    MagicMock(),
                    SimpleNamespace(sandbox_balance=1000.0, max_horizon_days=None),
                )

        assert code == EXIT_NO_TRADEABLE_SHARDS
        assert "Ingest produced zero markets" in caplog.text
        # Same rule test_run_dev_all_shards_inactive_returns_blind_code already
        # pins for the halt path: the empty simulation file records a run that
        # SCANNED and found no pairs, never one that never looked.
        assert not wrote, "a blind run must short-circuit before the simulation write"


class TestRunDevExitCode:
    def test_run_dev_returns_ok_code(self):
        client = MagicMock()
        with (
            patch("kalshi_betting.main.fetch_shard_statuses", return_value=None),
            patch(
                "kalshi_betting.main.fetch_open_events_with_markets",
                return_value=_stub_ingest(),
            ),
            patch("kalshi_betting.main.filter_markets_within_horizon", side_effect=lambda m, d: m),
            patch("kalshi_betting.main.find_time_series_pairs", return_value=[]),
            patch("kalshi_betting.main.find_same_title_pairs", return_value=[]),
            patch("kalshi_betting.main.enrich_with_orderbook_prices", return_value=[]),
            patch("kalshi_betting.main.write_dev_simulation", return_value="dev_sim.xlsx"),
        ):
            code = main._run_dev(client, SimpleNamespace(sandbox_balance=1000.0, max_horizon_days=None))

        assert code == EXIT_OK


class TestMainEntryPoint:
    @patch("kalshi_betting.main.write_dev_simulation")
    @patch("kalshi_betting.main.enrich_with_orderbook_prices")
    @patch("kalshi_betting.main.find_same_title_pairs")
    @patch("kalshi_betting.main.find_time_series_pairs")
    @patch("kalshi_betting.main.filter_markets_within_horizon")
    @patch("kalshi_betting.main.fetch_shard_statuses", return_value=None)
    @patch("kalshi_betting.main.fetch_open_events_with_markets")
    @patch("kalshi_betting.main.build_client")
    def test_main_dev_mode_exits_ok(
        self,
        mock_build_client,
        mock_fetch,
        mock_shard_statuses,
        mock_filter_horizon,
        mock_find_ts,
        mock_find_st,
        mock_enrich,
        mock_write_sim,
        tmp_path,
        monkeypatch,
    ):
        mock_build_client.return_value = MagicMock()
        mock_fetch.return_value = _stub_ingest()
        mock_filter_horizon.side_effect = lambda m, d: m
        mock_find_ts.return_value = []
        mock_find_st.return_value = []
        mock_enrich.return_value = []
        mock_write_sim.return_value = "dev_sim.xlsx"

        # main() configures logging with a FileHandler under PROJECT_ROOT —
        # point that at tmp_path so this test never touches the real
        # repo-root kalshi_arb.log.
        monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(sys, "argv", ["kalshi_betting.main", "--mode", "dev"])

        with pytest.raises(SystemExit) as exc_info:
            main.main()

        assert exc_info.value.code == EXIT_OK

    @patch("kalshi_betting.main.verify_auth")
    @patch("kalshi_betting.main.build_client")
    def test_main_prod_mode_low_balance_exits_skip_code(
        self, mock_build_client, mock_verify_auth, tmp_path, monkeypatch,
    ):
        mock_build_client.return_value = MagicMock()
        mock_verify_auth.return_value = {DEFAULT_EXCHANGE_INDEX: MIN_BALANCE_CENTS - 1}

        monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(sys, "argv", ["kalshi_betting.main", "--mode", "prod"])

        with pytest.raises(SystemExit) as exc_info:
            main.main()

        assert exc_info.value.code == EXIT_SKIPPED_LOW_BALANCE
        assert exc_info.value.code == 10

    @patch("kalshi_betting.main.fetch_open_events_with_markets", return_value=[])
    @patch("kalshi_betting.main.fetch_shard_statuses")
    @patch("kalshi_betting.main.get_held_tickers", return_value=set())
    @patch("kalshi_betting.main.verify_auth")
    @patch("kalshi_betting.main.build_client")
    def test_main_prod_mode_blind_run_exits_no_tradeable_shards_code(
        self, mock_build_client, mock_verify_auth, mock_held, mock_shard_statuses,
        mock_fetch, tmp_path, monkeypatch,
    ):
        # _run_prod's exit-30 return is covered directly elsewhere, but nothing
        # asserted that main() actually propagates it to sys.exit — and that
        # process code is the ONLY signal scheduler.run_job has that the weekly
        # slot went unscanned rather than merely finding no edge (TS-01).
        mock_build_client.return_value = MagicMock()
        mock_verify_auth.return_value = {DEFAULT_EXCHANGE_INDEX: MIN_BALANCE_CENTS * 10}
        # Every advertised shard halted: ingest drops every market, so nothing
        # could be scanned this run.
        mock_shard_statuses.return_value = {
            0: {"exchange_index": 0, "trading_active": False},
            1: {"exchange_index": 1, "trading_active": False},
        }

        monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(
            sys, "argv", ["kalshi_betting.main", "--mode", "prod", "--dry-run"],
        )

        with pytest.raises(SystemExit) as exc_info:
            main.main()

        assert exc_info.value.code == EXIT_NO_TRADEABLE_SHARDS
        assert exc_info.value.code == 30


def _fake_write_dev_simulation(results, candidate_pairs, balance_cents):
    """Stand-in for reporter.write_dev_simulation() that reproduces its one
    real log line (reporter.py:544, "Dev simulation written: %s") so the
    BS-26 tests below can assert main._run_dev's caller side does not log a
    duplicate of it on any exit path.
    """
    logging.info("Dev simulation written: %s", "dev_sim.xlsx")
    return "dev_sim.xlsx"


class TestDevSimulationLoggedOnce:
    """BS-26: write_dev_simulation() already logs "Dev simulation written: %s"
    itself — main._run_dev must not repeat that line (with or without an
    "(empty)"/"(candidates only)" qualifier) on any of its three exit paths.
    """

    def test_empty_candidate_pairs_logs_written_once(self, caplog):
        caplog.set_level(logging.INFO)
        client = MagicMock()
        with (
            patch("kalshi_betting.main.fetch_shard_statuses", return_value=None),
            patch(
                "kalshi_betting.main.fetch_open_events_with_markets",
                return_value=_stub_ingest(),
            ),
            patch("kalshi_betting.main.filter_markets_within_horizon", side_effect=lambda m, d: m),
            patch("kalshi_betting.main.find_time_series_pairs", return_value=[]),
            patch("kalshi_betting.main.find_same_title_pairs", return_value=[]),
            patch("kalshi_betting.main.enrich_with_orderbook_prices", return_value=[]),
            patch("kalshi_betting.main.write_dev_simulation", side_effect=_fake_write_dev_simulation),
        ):
            code = main._run_dev(client, SimpleNamespace(sandbox_balance=1000.0, max_horizon_days=None))

        assert code == EXIT_OK
        assert caplog.text.count("Dev simulation written:") == 1

    def test_empty_portfolio_logs_written_once(self, caplog):
        caplog.set_level(logging.INFO)
        client = MagicMock()
        pair = make_spec().pair
        with (
            patch("kalshi_betting.main.fetch_shard_statuses", return_value=None),
            patch(
                "kalshi_betting.main.fetch_open_events_with_markets",
                return_value=_stub_ingest(),
            ),
            patch("kalshi_betting.main.filter_markets_within_horizon", side_effect=lambda m, d: m),
            patch("kalshi_betting.main.find_time_series_pairs", return_value=[]),
            patch("kalshi_betting.main.find_same_title_pairs", return_value=[pair]),
            patch("kalshi_betting.main.enrich_with_orderbook_prices", return_value=[pair]),
            patch("kalshi_betting.main.compute_trade", return_value=None),
            patch("kalshi_betting.main.select_portfolio", return_value=[]),
            patch("kalshi_betting.main.write_dev_simulation", side_effect=_fake_write_dev_simulation),
        ):
            code = main._run_dev(client, SimpleNamespace(sandbox_balance=1000.0, max_horizon_days=None))

        assert code == EXIT_OK
        assert caplog.text.count("Dev simulation written:") == 1

    def test_full_run_logs_written_once(self, caplog):
        caplog.set_level(logging.INFO)
        client = MagicMock()
        spec = make_spec()
        with (
            patch("kalshi_betting.main.fetch_shard_statuses", return_value=None),
            patch(
                "kalshi_betting.main.fetch_open_events_with_markets",
                return_value=_stub_ingest(),
            ),
            patch("kalshi_betting.main.filter_markets_within_horizon", side_effect=lambda m, d: m),
            patch("kalshi_betting.main.find_time_series_pairs", return_value=[]),
            patch("kalshi_betting.main.find_same_title_pairs", return_value=[spec.pair]),
            patch("kalshi_betting.main.enrich_with_orderbook_prices", return_value=[spec.pair]),
            patch("kalshi_betting.main.compute_trade", return_value=spec),
            patch("kalshi_betting.main.select_portfolio", return_value=[spec]),
            patch(
                "kalshi_betting.main.execute_trades",
                return_value=[TradeResult(spec=spec, status="simulated")],
            ),
            patch("kalshi_betting.main.write_dev_simulation", side_effect=_fake_write_dev_simulation),
        ):
            code = main._run_dev(client, SimpleNamespace(sandbox_balance=1000.0, max_horizon_days=None))

        assert code == EXIT_OK
        assert caplog.text.count("Dev simulation written:") == 1


class TestLoggingRotation:
    """BS-25: kalshi_arb.log must rotate (5MB x 3 backups) instead of growing
    unbounded — a scheduler daemon re-runs this process weekly forever.

    Targets _setup_logging() directly rather than main(): logging.basicConfig()
    is a no-op once the root logger already has handlers (pytest installs its
    own), so the root logger is cleared first to actually exercise the handler
    configuration.
    """

    def test_setup_logging_file_handler_rotates(self, tmp_path):
        root = logging.getLogger()
        saved_handlers = root.handlers[:]
        saved_level = root.level
        root.handlers = []
        try:
            main._setup_logging(tmp_path / "kalshi_arb.log")

            file_handlers = [
                h for h in root.handlers
                if isinstance(h, logging.handlers.RotatingFileHandler)
            ]
            assert len(file_handlers) == 1
            handler = file_handlers[0]
            assert handler.maxBytes == 5 * 1024 * 1024
            assert handler.backupCount == 3
            # A plain FileHandler would satisfy isinstance(h, FileHandler) too,
            # so pin the concrete type — that is the whole point of BS-25
            assert type(handler) is logging.handlers.RotatingFileHandler
        finally:
            for h in root.handlers:
                h.close()
            root.handlers = saved_handlers
            root.level = saved_level

    def test_main_installs_the_rotating_handler(self, tmp_path, monkeypatch):
        # End-to-end: main() must route through _setup_logging, so the
        # rotating handler is what a real run actually gets.
        root = logging.getLogger()
        saved_handlers = root.handlers[:]
        saved_level = root.level
        for h in saved_handlers:
            root.removeHandler(h)

        try:
            monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
            monkeypatch.setattr(sys, "argv", ["kalshi_betting.main", "--mode", "dev"])
            with (
                patch("kalshi_betting.main.build_client", return_value=MagicMock()),
                patch("kalshi_betting.main.fetch_shard_statuses", return_value=None),
                patch(
                    "kalshi_betting.main.fetch_open_events_with_markets",
                    return_value=_stub_ingest(),
                ),
                patch("kalshi_betting.main.filter_markets_within_horizon", side_effect=lambda m, d: m),
                patch("kalshi_betting.main.find_time_series_pairs", return_value=[]),
                patch("kalshi_betting.main.find_same_title_pairs", return_value=[]),
                patch("kalshi_betting.main.enrich_with_orderbook_prices", return_value=[]),
                patch("kalshi_betting.main.write_dev_simulation", return_value="dev_sim.xlsx"),
            ):
                with pytest.raises(SystemExit):
                    main.main()

            file_handlers = [
                h for h in root.handlers
                if isinstance(h, logging.handlers.RotatingFileHandler)
            ]
            assert len(file_handlers) == 1
        finally:
            # Close whatever main() attached so tmp_path teardown isn't blocked
            # by an open file handle on any platform, then restore the root
            # logger exactly as pytest had it configured.
            for h in root.handlers[:]:
                h.close()
                root.removeHandler(h)
            for h in saved_handlers:
                root.addHandler(h)
            root.setLevel(saved_level)


class TestDryRunInertInDev:
    """BS-32: --dry-run has no effect in dev mode (dev always simulates), so
    main() logs a warning naming that rather than leaving it silently
    ignored.
    """

    def test_dev_dry_run_logs_inert_warning(self, tmp_path, monkeypatch, caplog):
        caplog.set_level(logging.WARNING)
        monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(sys, "argv", ["kalshi_betting.main", "--mode", "dev", "--dry-run"])

        with (
            patch("kalshi_betting.main.build_client", return_value=MagicMock()),
            patch("kalshi_betting.main.fetch_shard_statuses", return_value=None),
            patch(
                "kalshi_betting.main.fetch_open_events_with_markets",
                return_value=_stub_ingest(),
            ),
            patch("kalshi_betting.main.filter_markets_within_horizon", side_effect=lambda m, d: m),
            patch("kalshi_betting.main.find_time_series_pairs", return_value=[]),
            patch("kalshi_betting.main.find_same_title_pairs", return_value=[]),
            patch("kalshi_betting.main.enrich_with_orderbook_prices", return_value=[]),
            patch("kalshi_betting.main.write_dev_simulation", return_value="dev_sim.xlsx"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                main.main()

        assert exc_info.value.code == EXIT_OK
        assert "--dry-run is inert in dev mode" in caplog.text

    def test_prod_dry_run_does_not_log_inert_warning(self, tmp_path, monkeypatch, caplog):
        caplog.set_level(logging.WARNING)
        monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(sys, "argv", ["kalshi_betting.main", "--mode", "prod", "--dry-run"])

        with (
            patch("kalshi_betting.main.build_client", return_value=MagicMock()),
            patch(
                "kalshi_betting.main.verify_auth",
                return_value={DEFAULT_EXCHANGE_INDEX: MIN_BALANCE_CENTS - 1},
            ),
        ):
            with pytest.raises(SystemExit):
                main.main()

        assert "--dry-run is inert in dev mode" not in caplog.text


def test_exit_code_constants_distinct():
    # Guard against a future accidental collision between the codes — the
    # scheduler's log-level mapping depends on them being distinguishable,
    # and EXIT_NO_TRADEABLE_SHARDS additionally drives its retry (TS-01).
    assert len({
        EXIT_OK,
        EXIT_SKIPPED_LOW_BALANCE,
        EXIT_TRADES_NEED_ATTENTION,
        EXIT_NO_TRADEABLE_SHARDS,
    }) == 4


class TestSandboxBalanceInertInProd:
    """
    TS-19: --sandbox-balance is read only by _run_dev. Passing it in prod
    silently did nothing, so an operator who meant to cap their exposure got
    full-size live orders. Mirrors the --dry-run-in-dev twin beside it.
    """

    @staticmethod
    def _run(argv, caplog):
        with patch.object(sys, "argv", argv), \
             patch("kalshi_betting.main.build_client", return_value=MagicMock()), \
             patch("kalshi_betting.main._run_prod", return_value=0), \
             patch("kalshi_betting.main._run_dev", return_value=0), \
             patch("kalshi_betting.main._setup_logging"), \
             caplog.at_level(logging.WARNING), \
             pytest.raises(SystemExit):
            main.main()
        return caplog.text

    def test_warns_when_passed_in_prod(self, caplog):
        text = self._run(
            ["main", "--mode", "prod", "--sandbox-balance", "50"], caplog)
        assert "--sandbox-balance is inert in prod mode" in text

    def test_silent_when_not_passed_in_prod(self, caplog):
        text = self._run(["main", "--mode", "prod"], caplog)
        assert "--sandbox-balance" not in text

    def test_silent_in_dev(self, caplog):
        text = self._run(
            ["main", "--mode", "dev", "--sandbox-balance", "50"], caplog)
        assert "--sandbox-balance is inert" not in text
