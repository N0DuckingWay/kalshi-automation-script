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

    On the live toggles: a live run starts only from the saved live defaults
    (config.LIVE_DEFAULTS_FILE, each test's own tmp_path under conftest's
    _isolate_live_defaults). TestLiveDefaultsRequired pins that it exits 2
    with none saved or a refused file; TestSavedLiveDefaults that it reads
    them once and names them. TestLiveSettingsFlags, TestLogLiveSettings and
    TestLiveSettingsReachEverySite (the runtime half of test_strategy.py's
    live-toggle AST pin) run under pinned_config_toggles, which pins config's
    toggles and saves them as the test's live defaults; every other test that
    runs main.main() saves config.py's toggles first (conftest's
    saved_live_defaults). TestCategoryFilter covers main._filter_by_category.

    On the live-run lock: TestRunLock pins that a production run that sends
    orders holds run_lock's machine-wide lock from before it builds a client
    until main() ends, and stops with EXIT_RUN_IN_PROGRESS when another run
    holds it; dry runs and dev runs neither take it nor wait for it.
    tests/conftest.py's _isolate_live_runs points the lock at each test's own
    tmp_path, so every production run here takes a lock of its own.

    On the run result: TestResultFile runs main() with --result-file PATH
    (in each test's tmp_path) down every way a production run ends and reads
    the JSON back; TestLiveSettingsArgv pins config.live_settings_argv's round
    trip through main._build_parser and _resolve_live_settings. Every stand-in
    for _run_prod takes the keyword report=None, since main() passes one to
    a production run.

Dependencies:
    Imports _run_dev/_run_prod and the pure helpers from kalshi_betting.main,
    plus config constants asserted against and conftest's
    apply_pre_toggle_defaults and save_config_live_defaults; for the
    category/tag filter, historical, dashboard and backtester.BacktestTrade.
    The live-shape replays mock all
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

    _run_prod reads the balance through main.read_account_balance, which
    returns an auth.AccountBalance: each shard's cash (a dict, never a
    scalar) and Kalshi's value of the open positions. Every stand-in for it
    here returns one (_account builds it); _run_prod spends the cash summed
    over the shards and sizes on that sum plus the positions' value.

    That value counts only when the contracts held can back it: a contract
    pays at most $1, so main._checked_positions_value refuses a larger value
    and the run sizes on the cash alone (TestPositionsValueCheck). A test
    whose balance carries a positions value above 0 therefore holds enough
    contracts (_held_positions' count, or the positions rows of the
    live-shape client).
"""
import ast
import dataclasses
import inspect
import json
import logging
import logging.handlers
import math
import os
import pathlib
import re
import sys
from datetime import UTC, date, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from kalshi_betting import config, dashboard, historical, main, reporter, run_lock
from kalshi_betting import scanner as scanner_mod
from kalshi_betting import strategy as strategy_mod
from kalshi_betting import trader as trader_mod
from kalshi_betting.auth import AccountBalance
from kalshi_betting.backtester import BacktestTrade
from kalshi_betting.config import (
    DEFAULT_EXCHANGE_INDEX,
    EXIT_NO_TRADEABLE_SHARDS,
    EXIT_OK,
    EXIT_RUN_IN_PROGRESS,
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

from .conftest import apply_pre_toggle_defaults, save_config_live_defaults


def _held_positions(*tickers: str, count: float = 1.0):
    """
    Build a stand-in for scanner.get_held_positions that holds these tickers.

    Each is held with `count` YES contracts (one by default) and no readable
    cost, and the listing reads as complete (complete_out["complete"] is set
    True), as a real walk that reached its last page does. A contract pays at
    most $1, so a production run counts Kalshi's value of the open positions
    only when the contracts held can back it (main._checked_positions_value):
    a test whose balance carries a positions value holds enough of them.

    Args:
        *tickers (str): The tickers held.
        count (float): Keyword-only. The YES contracts held on each.

    Returns:
        callable: (client, *, complete_out=None) -> {ticker: HeldPosition}.
    """
    def fake(client, *, complete_out=None):
        if complete_out is not None:
            complete_out["complete"] = True
        return {t: scanner_mod.HeldPosition(t, count, None, None) for t in tickers}
    return fake


def make_pair(ticker_a: str, ticker_b: str, pair_type: str = "time_series"):
    """Minimal stand-in for a CandidatePair — only the attributes
    _dedup_pairs actually reads (market_a.ticker, market_b.ticker)."""
    return SimpleNamespace(
        market_a=SimpleNamespace(ticker=ticker_a),
        market_b=SimpleNamespace(ticker=ticker_b),
        pair_type=pair_type,
    )


def _account(cash_cents: int, positions_value_cents: int | None = 0) -> AccountBalance:
    """
    One balance read, as main.read_account_balance returns it.

    The cash sits on the default shard. The positions value defaults to 0 (an
    account that holds nothing), so the run's portfolio value is its cash and
    no "no readable portfolio_value" WARNING is logged; pass None to model a
    reply that carried no readable value.

    Args:
        cash_cents (int): The cash, in whole cents.
        positions_value_cents (int | None): Kalshi's value of the open
            positions, in whole cents; None when the reply had none.

    Returns:
        AccountBalance: The read.
    """
    return AccountBalance({DEFAULT_EXCHANGE_INDEX: cash_cents}, positions_value_cents)


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

        def fake_compute_trade(pair, portfolio_value_cents, *, settings, cash_cents):
            # Only pair_ok produces a spec — pair_none has no edge (returns None)
            seen.append((settings, portfolio_value_cents, cash_cents))
            if pair is pair_ok:
                return SimpleNamespace(pair=pair)
            return None

        monkeypatch.setattr(main, "compute_trade", fake_compute_trade)

        specs = main._compute_trade_specs([pair_ok, pair_none], 100_000, settings,
                                          cash_cents=40_000)

        assert list(specs.keys()) == [id(pair_ok)]
        assert specs[id(pair_ok)].pair is pair_ok
        # Every pair is sized under the ONE settings object the run handed in,
        # on the portfolio value, with the cash as the budget's ceiling
        assert len(seen) == 2
        assert all(s is settings and value == 100_000 and cash == 40_000
                   for s, value, cash in seen)

    def test_the_cash_is_required(self):
        # No default: a caller that forgot it would size every trade as if
        # the whole portfolio value were cash
        with pytest.raises(TypeError):
            main._compute_trade_specs([], 100_000, live_settings())


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
def pinned_config_toggles(monkeypatch, _isolate_live_defaults):
    """
    Pin the eight live toggles and save them as this test's live defaults.

    Through conftest's apply_pre_toggle_defaults, the one definition of their
    values, then save_config_live_defaults into the test's own path (requested
    by name, so the save lands after conftest's per-test redirect): a run
    through main.main() starts from those values, and every "(default: X)"
    mark reads the same whatever config.py ships. config.live_settings()
    returns the same toggles, so a run mode handed none reads them too.

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest's per-test patcher.
        _isolate_live_defaults (None): conftest's per-test redirect, set up first.
    """
    apply_pre_toggle_defaults(monkeypatch)
    save_config_live_defaults()


def _save_live_defaults(**changes) -> LiveSettings:
    """
    Save config.py's toggles with these fields replaced as this test's live defaults.

    Args:
        **changes: LiveSettings fields to replace in config.live_settings().

    Returns:
        LiveSettings: The saved defaults as read back (their origin names the file).
    """
    return config.save_live_defaults(dataclasses.replace(live_settings(), **changes),
                                     source="")


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
    "client_built" and, when a run-mode recorder ran, "settings", "reference",
    "mode" and "report" (the run report main() handed a production run, None
    for dev or without --result-file).
    """
    seen: dict = {"logging_set_up": False, "client_built": False}

    def record_run(client, args, settings, reference, report=None):
        seen.update(settings=settings, reference=reference, mode=args.mode, report=report)
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

    def test_the_gate_comes_before_the_result_file_check(self, monkeypatch, capsys, tmp_path):
        # --result-file in dev is itself refused, but the order-path check
        # runs first, and nothing touches the file before it
        path = tmp_path / "result.json"
        path.write_text("an older result", encoding="utf-8")
        monkeypatch.setattr(config, "ORDER_API_VERSION", "legacy")
        seen = _main_with(
            monkeypatch, ["--mode", "dev", "--result-file", str(path)],
            _resolve_live_settings=lambda *a: pytest.fail("resolved settings first"),
        )
        assert seen["code"] == 2
        err = capsys.readouterr().err
        assert config.order_api_version_error() in err
        assert "--result-file is for --mode prod only" not in err
        assert path.read_text(encoding="utf-8") == "an older result"

    @pytest.mark.usefixtures("saved_live_defaults")
    def test_the_shipped_value_reaches_the_run_mode(self, monkeypatch):
        # Saved live defaults too: with none saved, the run exits 2 before
        # its run mode (TestLiveDefaultsRequired)
        assert config.ORDER_API_VERSION == "v2"
        seen = _main_with(monkeypatch, ["--mode", "dev"])
        assert seen["code"] == EXIT_OK and seen["mode"] == "dev"


@pytest.mark.usefixtures("pinned_config_toggles")
class TestLiveSettingsFlags:
    """Each live-toggle flag overrides ONE saved live default for one run, over
    the saved file read at call time; a value LiveSettings refuses is a usage
    error (exit 2) before logging is configured (TS-20) or a client built."""

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
        (["--add-to-held-pairs"], "add_to_held_pairs", True),
    ])
    def test_each_flag_overrides_only_its_own_field(self, monkeypatch, mode, argv, field, value):
        seen = _main_with(monkeypatch, ["--mode", mode, *argv])
        assert seen["code"] == EXIT_OK and seen["mode"] == mode
        settings, reference = seen["settings"], seen["reference"]
        assert reference == live_settings()
        assert getattr(settings, field) == value
        assert getattr(settings, field) != getattr(reference, field)
        for other in config.LIVE_TOGGLE_FIELDS:
            if other != field:
                assert getattr(settings, other) == getattr(reference, other), other

    def test_no_add_to_held_pairs_turns_a_saved_on_off_for_one_run(self, monkeypatch):
        # The --no- form departs only from defaults saved with it on
        _save_live_defaults(add_to_held_pairs=True)
        seen = _main_with(monkeypatch, ["--mode", "prod", "--no-add-to-held-pairs"])
        assert seen["reference"].add_to_held_pairs is True
        assert seen["settings"].add_to_held_pairs is False
        assert seen["settings"] == dataclasses.replace(seen["reference"],
                                                       add_to_held_pairs=False)
        line = describe_live_settings(seen["settings"], seen["reference"])
        assert "add to held pairs off (default: on)" in line and line.count("(default:") == 1
        # No flag keeps the saved value
        assert _main_with(monkeypatch, ["--mode", "prod"])["settings"].add_to_held_pairs is True

    def test_no_flag_hands_the_run_the_saved_defaults_themselves(self, monkeypatch):
        # The scheduler's exact argv (tests/test_scheduler.py pins it)
        seen = _main_with(monkeypatch, ["--mode", "prod"])
        assert seen["settings"] == seen["reference"] == config.read_saved_live_defaults()
        # ... which the fixture saved from the pinned constants
        assert seen["reference"] == live_settings()
        assert seen["settings"].origin == seen["reference"].origin
        assert seen["reference"].origin.startswith("live_defaults.json, saved ")

    def test_a_flag_equal_to_the_saved_default_departs_nothing(self, monkeypatch):
        saved = config.read_saved_live_defaults()
        seen = _main_with(monkeypatch, [
            "--mode", "prod", "--tier-floors" if saved.tier_floors else "--no-tier-floors",
            "--interval-discount", repr(saved.interval_discount),
            "--size-cap", str(round(saved.size_cap * 100)),
        ])
        assert seen["settings"] == seen["reference"]

    def test_one_band_flag_keeps_the_saved_other_bound(self, monkeypatch):
        # Read at call time from the saved file, never bound at import
        _save_live_defaults(spread_band=(0.1, 0.8))
        assert _main_with(monkeypatch, ["--spread-max", "0.5"])["settings"].spread_band == (0.1, 0.5)
        assert _main_with(monkeypatch, ["--spread-min", "0.2"])["settings"].spread_band == (0.2, 0.8)
        # 0 is a value, not "not given": it lowers the saved floor to 0
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
        # 0 is a value, not "not given": refused, never silently the saved one
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

    @pytest.mark.parametrize("field, flag", [
        ("categories", "--any-category"),
        ("tags", "--any-tag"),
    ])
    def test_an_any_flag_clears_the_saved_filter_for_one_run(self, monkeypatch, field, flag):
        _save_live_defaults(**{field: ("Sports",)})
        seen = _main_with(monkeypatch, ["--mode", "prod", flag])
        assert getattr(seen["reference"], field) == ("Sports",)
        assert getattr(seen["settings"], field) is None
        line = describe_live_settings(seen["settings"], seen["reference"])
        assert f"{field} any (default: Sports)" in line and line.count("(default:") == 1
        # No flag keeps the saved filter
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
                     "--add-to-held-pairs", "--no-add-to-held-pairs",
                     "--category", "--any-category", "--tag", "--any-tag"):
            assert flag in out, flag
        # Every value flag defaults to the saved live defaults, never a config.py constant
        assert out.count("default: the saved live defaults") == 9
        assert out.count("whatever the saved live defaults say") == 2
        assert "config.TIME_SERIES" not in out and "config.TRADE" not in out
        assert "config.BUDGET_FRACTION" not in out and "config.SAME_TITLE" not in out
        assert ("Override one live default for THIS run only, in either mode "
                "(adding to held pairs: production runs only). The live "
                "defaults are the ones saved through python3 -m "
                "kalshi_betting.defaults_server (live_defaults.json); a run refuses to "
                "start without them. The weekly scheduler passes none of these flags, so "
                "a scheduled run trades exactly the saved defaults.") in out
        assert out.count(f"in {config.SIZE_CAP_STEP * 100:g}% steps") == 2
        assert "100 = no cap" in out
        assert "100 = no extra cap beyond --size-cap" in out
        assert "Production runs only: a dev run holds nothing" in out
        assert "config.ADD_TO_HELD_PAIRS" not in out
        assert "live trading toggles" in out

    def test_the_echo_marks_exactly_the_departing_fields(self, monkeypatch):
        seen = _main_with(monkeypatch, ["--interval-discount", "0.751"])
        line = describe_live_settings(seen["settings"], seen["reference"])
        # k renders exactly, so 0.751 never prints as the saved 0.75
        assert "k 0.751 (default: 0.75)" in line
        assert line.count("(default:") == 1 and "(config:" not in line

        seen = _main_with(monkeypatch, [
            "--no-tier-floors", "--spread-max", "0.5", "--size-cap", "35"])
        line = describe_live_settings(seen["settings"], seen["reference"])
        assert "tier floors off (default: on)" in line
        assert "spread band 0-0.5 (default: none)" in line
        assert "per-trade cap 35% (default: 20%)" in line
        assert line.count("(default:") == 3


def _prod_until_the_balance_gate(monkeypatch, argv: list, caplog) -> int:
    """Run the real _run_prod up to the MIN_BALANCE_CENTS gate (it logs its
    settings first) and return the exit code."""
    with caplog.at_level(logging.INFO):
        seen = _main_with(
            monkeypatch, ["--mode", "prod", *argv],
            _run_prod=main._run_prod,
            read_account_balance=lambda client: _account(MIN_BALANCE_CENTS - 1),
        )
    return seen["code"]


@pytest.mark.usefixtures("pinned_config_toggles")
class TestLogLiveSettings:
    """Each live run logs where its defaults came from and its toggles on INFO
    lines, marking each departure from the saved live defaults; it WARNS on
    every live_rule_warnings sentence and, when a prod run that submits orders
    departs, on that."""

    _DEPARTURE = "This PRODUCTION run overrides the saved live defaults"
    # The same WARNING for a reference built from config.py's constants (one a
    # test hands a run mode explicitly, or builds by hand)
    _CONFIG_DEPARTURE = "This PRODUCTION run overrides config.py's live settings"

    def test_a_scheduled_run_logs_the_saved_defaults_with_no_mark_and_no_warning(
        self, monkeypatch, caplog,
    ):
        code = _prod_until_the_balance_gate(monkeypatch, [], caplog)
        assert code == EXIT_SKIPPED_LOW_BALANCE
        lines = [r.getMessage() for r in caplog.records
                 if r.getMessage().startswith("Live settings:")]
        assert lines == [f"Live settings: {describe_live_settings(live_settings())}"]
        assert "(default:" not in lines[0] and "(config:" not in lines[0]
        # The line before it names the saved file the run started from
        saved = config.read_saved_live_defaults()
        assert f"Live defaults: {saved.origin}" in caplog.text
        assert self._DEPARTURE not in caplog.text
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING
                    and r.getMessage().startswith("Live settings:")]

    # One departing flag per live toggle (config.LIVE_TOGGLE_FIELDS), with the
    # mark its line must carry
    _ONE_FLAG_PER_FIELD = {
        "tier_floors": (["--no-tier-floors"], "tier floors off (default: on)"),
        "spread_band": (["--spread-max", "0.5"], "spread band 0-0.5 (default: none)"),
        "interval_discount": (["--interval-discount", "0.6"], "k 0.6 (default: 0.75)"),
        "size_cap": (["--size-cap", "35"], "per-trade cap 35% (default: 20%)"),
        "same_title_size_cap": (["--same-title-size-cap", "15"],
                                "same-title cap 15% (default: 100% (no extra cap))"),
        "categories": (["--category", "Economics"], "categories Economics (default: any)"),
        "tags": (["--tag", "Oil & Gas"], "tags Oil & Gas (default: any)"),
        "add_to_held_pairs": (["--add-to-held-pairs"], "add to held pairs on (default: off)"),
    }

    def test_every_field_has_a_departing_flag(self):
        # A new toggle needs a row, so its departure WARNING is tested
        assert set(self._ONE_FLAG_PER_FIELD) == set(config.LIVE_TOGGLE_FIELDS)

    @pytest.mark.parametrize("field", sorted(_ONE_FLAG_PER_FIELD))
    def test_a_departing_production_run_warns(self, monkeypatch, caplog, field):
        argv, mark = self._ONE_FLAG_PER_FIELD[field]
        code = _prod_until_the_balance_gate(monkeypatch, argv, caplog)
        assert code == EXIT_SKIPPED_LOW_BALANCE
        (line,) = [r.getMessage() for r in caplog.records
                   if r.getMessage().startswith("Live settings: tier floors")]
        assert mark in line and line.count("(default:") == 1, line
        warnings = [r for r in caplog.records
                    if r.levelno == logging.WARNING and self._DEPARTURE in r.getMessage()]
        assert len(warnings) == 1
        assert self._CONFIG_DEPARTURE not in caplog.text

    def test_a_departing_dry_run_marks_but_does_not_warn(self, monkeypatch, caplog):
        _prod_until_the_balance_gate(
            monkeypatch, ["--dry-run", "--interval-discount", "0.6"], caplog)
        assert "k 0.6 (default: 0.75)" in caplog.text
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
        # Neither wording: this reference is config.py's, so a WARNING would name it
        assert "This PRODUCTION run overrides" not in caplog.text

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

    def test_it_never_resolves_config_py_or_the_saved_defaults_itself(
        self, monkeypatch, caplog,
    ):
        def tripwire():
            raise AssertionError("_log_live_settings read settings of its own")
        s = LiveSettings(False, (0.0, 0.5), 0.8, 1.0, 0.2)
        r = LiveSettings(True, (0.0, 1.0), 0.75, 0.2, 1.0, origin=self._ORIGIN)
        for name in ("live_settings", "live_defaults"):
            monkeypatch.setattr(main, name, tripwire)
        for name in ("live_settings", "live_defaults", "read_saved_live_defaults"):
            monkeypatch.setattr(config, name, tripwire)
        with caplog.at_level(logging.INFO):
            main._log_live_settings(s, r, real_money=True)
        assert f"Live defaults: {self._ORIGIN}" in caplog.text
        assert f"Live settings: {describe_live_settings(s, r)}" in caplog.text
        assert "(default: on)" in caplog.text
        assert self._DEPARTURE in caplog.text

    # An origin as config.read_saved_live_defaults writes it
    _ORIGIN = "live_defaults.json, saved 2026-09-27T21:05:13Z from a note"

    def test_a_config_reference_keeps_the_config_mark_and_warning(self, caplog):
        # A reference built from config.py's constants (origin "config.py"):
        # marked "(config: X)", and the WARNING names config.py
        s = LiveSettings(False, (0.0, 0.5), 0.8, 1.0, 0.2)
        r = LiveSettings(True, (0.0, 1.0), 0.75, 0.2, 1.0)
        with caplog.at_level(logging.INFO):
            main._log_live_settings(s, r, real_money=True)
        assert f"Live defaults: {config.LIVE_DEFAULTS_FROM_CONFIG}" in caplog.text
        assert "tier floors off (config: on)" in caplog.text
        assert "(default:" not in caplog.text
        warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
                  and "PRODUCTION run overrides" in r.getMessage()]
        assert warned == [
            "This PRODUCTION run overrides config.py's live settings (see the \"(config: …)\" "
            "marks on the line above): its trades follow the flags, not the committed "
            "configuration"]

    def test_the_saved_warning_is_worded_for_the_saved_defaults(self, caplog):
        s = LiveSettings(False, (0.0, 0.5), 0.8, 1.0, 0.2)
        r = LiveSettings(True, (0.0, 1.0), 0.75, 0.2, 1.0, origin=self._ORIGIN)
        with caplog.at_level(logging.INFO):
            main._log_live_settings(s, r, real_money=True)
        warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
                  and "PRODUCTION run overrides" in r.getMessage()]
        assert warned == [
            "This PRODUCTION run overrides the saved live defaults (see the \"(default: …)\" "
            "marks on the line above): its trades follow the flags, not the saved defaults"]


def _saved_record(**toggles) -> str:
    """
    A saved live defaults file's text: config.py's toggles with these replaced.

    Built as the file's JSON record by hand, so a test can hold a value
    save_live_defaults itself would refuse to write.

    Args:
        **toggles: Raw JSON values for the "settings" block's fields.

    Returns:
        str: The file's text.
    """
    cfg = live_settings()
    settings = {name: getattr(cfg, name) for name in config.LIVE_TOGGLE_FIELDS}
    settings["spread_band"] = list(settings["spread_band"])
    settings.update(toggles)
    return json.dumps({"format": config.LIVE_DEFAULTS_FORMAT,
                       "saved_at": "2026-09-27T21:05:13Z", "source": "",
                       "settings": settings})


class TestLiveDefaultsRequired:
    """A live run starts only from the saved live defaults: with none saved, or
    a saved file refused, main() exits 2 (parser.error) in either mode, with or
    without a toggle flag, before logging is configured, any client is built or
    either run mode runs — and never falls back to config.py's toggles."""

    @pytest.mark.parametrize("mode", ["dev", "prod"])
    @pytest.mark.parametrize("flags", [[], ["--interval-discount", "0.6"], ["--any-tag"]])
    def test_no_saved_file_exits_2_before_anything_runs(self, monkeypatch, capsys, mode,
                                                        flags):
        assert not config.LIVE_DEFAULTS_FILE.exists()
        seen = _main_with(monkeypatch, ["--mode", mode, *flags])
        assert seen["code"] == 2
        assert not seen["logging_set_up"] and not seen["client_built"]
        assert "settings" not in seen
        err = " ".join(capsys.readouterr().err.split())
        assert f"no live defaults are saved at {config.LIVE_DEFAULTS_FILE}" in err
        assert "\"Save as live defaults…\" button" in err
        assert "python3 -m kalshi_betting.defaults_server --seed" in err
        assert "./start_dashboard.sh --seed" in err
        assert "live runs never fall back to config.py's toggles" in err

    @staticmethod
    def _recursion_bomb(path) -> None:
        """
        Write a saved-defaults file nested too deeply to parse.

        Args:
            path (pathlib.Path): Where to write it (config.LIVE_DEFAULTS_FILE).
        """
        path.write_text("[" * 200_000, encoding="utf-8")

    @pytest.mark.parametrize("mode", ["dev", "prod"])
    @pytest.mark.parametrize("write", [
        pytest.param(lambda path: path.write_text("not json", encoding="utf-8"), id="not-json"),
        pytest.param(lambda path: path.write_text(_saved_record(size_cap=0.33),
                                                  encoding="utf-8"), id="cap-0.33"),
        pytest.param(lambda path: path.mkdir(), id="directory"),
        pytest.param(_recursion_bomb, id="recursion-bomb"),
    ])
    def test_a_refused_file_exits_2_naming_it(self, monkeypatch, capsys, mode, write):
        write(config.LIVE_DEFAULTS_FILE)
        seen = _main_with(monkeypatch, ["--mode", mode])
        assert seen["code"] == 2
        assert not seen["logging_set_up"] and not seen["client_built"]
        assert "settings" not in seen
        err = " ".join(capsys.readouterr().err.split())
        assert "the saved live defaults are refused" in err
        assert str(config.LIVE_DEFAULTS_FILE) in err
        assert "python3 -m kalshi_betting.defaults_server" in err
        assert "./start_dashboard.sh" in err
        # The server will not save over a refused file, so the stderr says to
        # fix it, or delete it first
        assert "fix the file, or delete it and then save new ones" in err
        assert "the server will not save over a file it refuses" in err

    def test_an_invalid_config_constant_does_not_stop_a_run_with_a_saved_file(
        self, monkeypatch, saved_live_defaults,
    ):
        # config.py's toggles are no live run's defaults: an invalid one only
        # breaks a caller that hands no settings, never main()
        monkeypatch.setattr(config, "BUDGET_FRACTION", 0.37)
        with pytest.raises(ValueError):
            live_settings()
        seen = _main_with(monkeypatch, ["--mode", "prod"])
        assert seen["code"] == EXIT_OK
        assert seen["settings"] == seen["reference"] == config.read_saved_live_defaults()


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
# $250.00 and shard 1 holds $9,999.00 of cash. The run's cash is the
# BREAKDOWN's sum (1024900 cents = $10,249.00), deliberately != the
# $10,250.00 top-level balance_dollars aggregate, so the replay proves the
# breakdown sum is the cash the run spends, not the top-level field. Kelly
# sizes on that cash plus portfolio_value, Kalshi's value of the open
# positions — 0 here, as a live reply carries it, so the portfolio value is
# the cash and the run takes the normal path; the reply without the field,
# which sizes on the cash with a WARNING, has a test of its own.
_LIVE_BALANCE_PAYLOAD = {
    "balance": 114,
    "balance_dollars": "10250.0000",
    "balance_breakdown": [
        {"exchange_index": 0, "balance": "250.0000"},
        {"exchange_index": 1, "balance": "9999.0000"},
    ],
    "portfolio_value": 0,
}

# Same shape, but the portfolio value is below MIN_BALANCE_CENTS ($50 = 5000
# cents): cash shard0 $5.00 + shard1 $3.00 = 800 cents, and no open
# positions (portfolio_value 0). Cash on shard 1 is still summed in — the
# gate reads the cash on every shard together plus the positions' value, never
# one shard — so this fixture aborts only because that TOTAL is under the
# minimum; the same cash beside enough positions scans (see
# TestSizesOnPortfolioValue).
_LOW_BALANCE_PAYLOAD = {
    "balance": 1,
    "balance_dollars": "8.0000",
    "balance_breakdown": [
        {"exchange_index": 0, "balance": "5.0000"},
        {"exchange_index": 1, "balance": "3.0000"},
    ],
    "portfolio_value": 0,
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
# poll re-reads the balance through auth.read_shard_balances, so this is how the
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
    scanner.get_held_positions' paginated listing.

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
    held_rows: tuple = (),
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
        held_rows (tuple): More positions-listing rows, each a dict as the
            listing sends it (ticker, position_fp, market_exposure_dollars,
            fees_paid_dollars), after the rest.
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
        ) + [{"ticker": ticker, "position_fp": "2.00"} for ticker in extra_held]
        + list(held_rows),
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
        assert held_lookup_calls, "expected get_held_positions to have queried positions"

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
    dev run handed _SETTINGS (every toggle departing from the pinned config) and an
    explicit reference while live_settings, live_defaults and read_saved_live_defaults
    raise in config, scanner, strategy, trader and main, so a site reading config.py,
    the saved file or the reference raises or hands a spy the wrong value. It runs
    twice: with config.py's toggles as the reference, and with _SETTINGS saved as the
    live defaults and resolved as main() resolves them. _SETTINGS keeps both pairs
    trading, the time-series f* between the 0.25 same-title and 0.35 per-trade caps."""

    _SETTINGS = LiveSettings(tier_floors=False, spread_band=(0.05, 0.9),
                             interval_discount=0.6, size_cap=0.35, same_title_size_cap=0.25,
                             categories=("economics", "POLITICS"),
                             tags=("Inflation", "elections"), add_to_held_pairs=True)

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

    def _run_under_the_tripwire(self, monkeypatch, caplog, mode: str, *,
                                settings: LiveSettings | None = None,
                                reference: LiveSettings | None = None):
        """
        Run one mode under the tripwire, recording every _SPIED call.

        Args:
            monkeypatch (pytest.MonkeyPatch): pytest's per-test patcher.
            caplog (pytest.LogCaptureFixture): Captures the run's log.
            mode (str): "prod" (a dry run) or "dev".
            settings (LiveSettings | None): Keyword-only. The run's toggles;
                None hands _SETTINGS.
            reference (LiveSettings | None): Keyword-only. The defaults the
                run's toggles depart from; None reads config.py's toggles
                (live_settings()) before the tripwire, and every toggle must
                then differ from settings'.

        Returns:
            tuple: (settings, reference, calls, captured) — calls maps (module
                short name, function name) to each call's (args, kwargs);
                captured holds "results", "filter", "enriched" and, in prod,
                "run_note".
        """
        settings = self._SETTINGS if settings is None else settings
        if reference is None:
            # config.py's toggles, read before the tripwire; every toggle must differ
            reference = live_settings()
            for name in config.LIVE_TOGGLE_FIELDS:
                assert getattr(settings, name) != getattr(reference, name), name

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

        # The tripwire: no module may resolve config.py's settings, or read the
        # saved defaults, this run
        def tripwire(*args, **kwargs):
            raise AssertionError("settings read during a run handed its settings")

        for module in (config, scanner_mod, strategy_mod, trader_mod, main):
            monkeypatch.setattr(module, "live_settings", tripwire)
        for module, name in ((config, "live_defaults"), (config, "read_saved_live_defaults"),
                             (main, "live_defaults")):
            monkeypatch.setattr(module, name, tripwire)

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
        assert echo in caplog.text and echo.count("(config:") == 8
        assert "This PRODUCTION run overrides" not in caplog.text
        assert "one time-series pair may stake up to 35%" in caplog.text
        assert "one same-title pair may stake up to 25%" in caplog.text
        # The workbook's separator row carries the same marked line
        assert captured["run_note"] == f"settings: {describe_live_settings(settings, reference)}"
        assert captured["run_note"].count("(config:") == 8

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
        assert echo in caplog.text and echo.count("(config:") == 8
        # Dev never submits an order, so never the production WARNING
        assert "This PRODUCTION run overrides" not in caplog.text

    def _saved_and_resolved(self) -> tuple[LiveSettings, LiveSettings]:
        """
        Save _SETTINGS as the live defaults and resolve them as main() does.

        Through main._resolve_live_settings, with the scheduler's argv plus
        --dry-run (no toggle flag), before any tripwire is in place.

        Returns:
            tuple[LiveSettings, LiveSettings]: (the run's settings, the saved
                defaults they were built from).
        """
        cfg = live_settings()
        # Every toggle of the saved defaults differs from config.py's
        for name in config.LIVE_TOGGLE_FIELDS:
            assert getattr(self._SETTINGS, name) != getattr(cfg, name), name
        config.save_live_defaults(self._SETTINGS, source="a note")
        args = SimpleNamespace(mode="prod", dry_run=True)
        settings, reference = main._resolve_live_settings(args, MagicMock())
        assert settings == reference == self._SETTINGS
        assert reference.origin.startswith("live_defaults.json, saved ")
        assert reference.origin.endswith(" from a note")
        return settings, reference

    @pytest.mark.parametrize("mode", ["prod", "dev"])
    def test_a_run_from_saved_defaults_reads_them_at_every_site(
        self, monkeypatch, caplog, mode,
    ):
        settings, reference = self._saved_and_resolved()
        settings, reference, calls, captured = self._run_under_the_tripwire(
            monkeypatch, caplog, mode, settings=settings, reference=reference)
        self._assert_every_site_read_the_runs_settings(settings, calls)
        results = captured["results"]
        assert {r.status for r in results} == {"simulated"}
        assert {r.spec.pair.pair_type for r in results} == {"time_series", "same_title"}
        # The run names the saved file, and its settings carry no mark: no flag moved one
        assert f"Live defaults: {reference.origin}" in caplog.text
        echo = f"Live settings: {describe_live_settings(settings, reference)}"
        assert echo in caplog.text
        assert "(default:" not in echo and "(config:" not in echo
        assert "This PRODUCTION run overrides" not in caplog.text
        if mode == "prod":
            # The workbook's separator row names the saved file too
            assert captured["run_note"] == (
                f"settings: {describe_live_settings(settings, reference)} | defaults: "
                f"{reference.origin}")

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


@pytest.mark.usefixtures("pinned_config_toggles")
class TestSavedLiveDefaults:
    """A live run through main() starts from the saved live defaults: one read of
    the file per run (a counting spy), the file named on a "Live defaults:" line,
    each field a flag moves marked "(default: X)" (plus the production WARNING
    when orders go out), and the trade log's separator note naming the file. The
    saved values differ from the pinned config.py toggles in every field, so a
    site that read config.py instead would show."""

    # Every toggle differs from the pinned constants; both pairs of the
    # live-shape replay trade under it
    _SAVED = TestLiveSettingsReachEverySite._SETTINGS

    def _save(self) -> LiveSettings:
        """
        Save _SAVED as this test's live defaults.

        Returns:
            LiveSettings: The saved defaults as read back (their origin names the
                file and the "a note" source).
        """
        cfg = live_settings()
        for name in config.LIVE_TOGGLE_FIELDS:
            assert getattr(self._SAVED, name) != getattr(cfg, name), name
        return config.save_live_defaults(self._SAVED, source="a note")

    def test_the_schedulers_argv_runs_on_the_saved_defaults_read_once(
        self, monkeypatch, caplog,
    ):
        saved = self._save()
        reads: list = []
        real_read = config.read_saved_live_defaults

        def counting_read():
            """
            Count one read of the saved file, then read it.

            Returns:
                LiveSettings | None: What config.read_saved_live_defaults returns.
            """
            reads.append(1)
            return real_read()

        monkeypatch.setattr(config, "read_saved_live_defaults", counting_read)
        # Every parse of the file's bytes too, so a second read that bypasses the
        # reader would still be counted
        parses: list = []
        real_parse = config._settings_from_bytes

        def counting_parse(data: bytes) -> LiveSettings:
            """
            Count one parse of saved-defaults bytes, then parse them.

            Args:
                data (bytes): The file's bytes.

            Returns:
                LiveSettings: What config._settings_from_bytes returns.
            """
            parses.append(1)
            return real_parse(data)

        monkeypatch.setattr(config, "_settings_from_bytes", counting_parse)
        seen: dict = {}
        real_prod = main._run_prod

        def prod_spy(client, args, settings, reference, report=None):
            """
            Record what main() hands the production run mode, then run it.

            Args:
                client: The client main() built.
                args (argparse.Namespace): The parsed flags.
                settings (LiveSettings): The run's settings.
                reference (LiveSettings): The saved defaults they were built from.
                report (reporter.RunReport | None): The run report, passed on.

            Returns:
                int: What main._run_prod returns.
            """
            seen.update(settings=settings, reference=reference)
            return real_prod(client, args, settings, reference, report=report)

        with caplog.at_level(logging.INFO):
            out = _main_with(
                monkeypatch, ["--mode", "prod"], _run_prod=prod_spy,
                read_account_balance=lambda client: _account(MIN_BALANCE_CENTS - 1),
            )
        assert out["code"] == EXIT_SKIPPED_LOW_BALANCE
        # One read for the whole run, before logging, the client and the run mode
        assert len(reads) == 1 and len(parses) == 1
        assert seen["settings"] == seen["reference"] == self._SAVED
        assert seen["settings"].origin == seen["reference"].origin == saved.origin
        assert saved.origin.startswith("live_defaults.json, saved ")
        assert saved.origin.endswith(" from a note")
        # The file is named, and the settings line carries no mark
        info = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        assert f"Live defaults: {saved.origin}" in info
        (line,) = [m for m in info if m.startswith("Live settings:")]
        assert line == f"Live settings: {describe_live_settings(self._SAVED)}"
        assert "(default:" not in line and "(config:" not in line
        assert "PRODUCTION run overrides" not in caplog.text

    def test_a_flag_marks_the_saved_default_and_warns(self, monkeypatch, caplog):
        self._save()
        code = _prod_until_the_balance_gate(
            monkeypatch, ["--interval-discount", "0.7"], caplog)
        assert code == EXIT_SKIPPED_LOW_BALANCE
        (line,) = [r.getMessage() for r in caplog.records
                   if r.levelno == logging.INFO and r.getMessage().startswith("Live settings:")]
        assert "k 0.7 (default: 0.6)" in line
        assert line.count("(default:") == 1 and "(config:" not in line
        warned = [r for r in caplog.records if r.levelno == logging.WARNING
                  and "This PRODUCTION run overrides the saved live defaults" in r.getMessage()]
        assert len(warned) == 1

    def test_the_separator_note_names_the_saved_defaults(self, monkeypatch, caplog):
        saved = self._save()
        client = _live_shape_client(
            monkeypatch, balance_payload=_LIVE_BALANCE_PAYLOAD, include_time_series=True)
        # A fresh cached listing for the saved category/tag filter: no request
        _seed_series_listing(TestLiveSettingsReachEverySite._LISTING)
        captured: dict = {}

        def fake_append_to_prod_log(results, balance_before, balance_after, *, run_note=""):
            """
            Record the run's results and separator note instead of writing a workbook.

            Args:
                results (list[TradeResult]): The run's trade results.
                balance_before (float): The balance before the run, in dollars.
                balance_after (float | None): The balance after it, in dollars.
                run_note (str): Keyword-only. The separator row's note.

            Returns:
                pathlib.Path: A path standing in for the workbook.
            """
            captured["results"] = results
            captured["run_note"] = run_note
            return pathlib.Path("/fake/trade_log.xlsx")

        monkeypatch.setattr(main, "append_to_prod_log", fake_append_to_prod_log)
        with caplog.at_level(logging.INFO):
            out = _main_with(monkeypatch, ["--mode", "prod", "--dry-run",
                                           "--same-title-size-cap", "20"],
                             _run_prod=main._run_prod, build_client=lambda mode: client)
        assert out["code"] == EXIT_OK
        assert {r.status for r in captured["results"]} == {"simulated"}
        note = captured["run_note"]
        assert note.startswith("settings: ")
        assert "same-title cap 20% (default: 25%)" in note
        assert note.endswith(f" | defaults: {saved.origin}")


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
        monkeypatch.setattr(main, "find_same_title_pairs",
                            lambda markets, held, *, add_on_pairs=None: [])
        assert self._dry_run(client, monkeypatch, caplog,
                             expected_code=EXIT_TIME_SERIES_SKIPPED) == []
        [line] = [r.getMessage() for r in caplog.records
                  if r.getMessage().startswith("No qualifying pairs found")]
        # The line does not describe a time-series rule the run never applied
        assert ("time-series: not searched this run, because a held market could not "
                "be identified") in line
        assert "worded as cumulative deadlines" not in line


# The account holds the flow-through time-series pair itself: YES on TS-EARLY
# and NO on TS-LATE, 30 contracts each, with their costs (DR-77)
_HELD_TS_PAIR = (
    {"ticker": _TICKER_TS_EARLY, "position_fp": "30.00",
     "market_exposure_dollars": "9.00", "fees_paid_dollars": "0.45"},
    {"ticker": _TICKER_TS_LATE, "position_fp": "-30.00",
     "market_exposure_dollars": "12.00", "fees_paid_dollars": "0.50"},
)

# Two more markets of the Held Event / Held Question group, on two more
# series, closing with the rest: a same-title pair no one holds (HELD-C is
# market A, the pricier YES; its YES bids give a NO ask of 0.40, HELD-D's NO
# bids a YES ask of 0.25)
_TICKER_HELD_C, _TICKER_HELD_D = "HELD-C", "HELD-D"
_HELD_RUNNER_UP_EVENTS = (
    _ev("Held Event", _mk_market(_TICKER_HELD_C, "HELDCEVT-1", "Held Question", "Outcome",
                                 "0.55", "0.40", price_level_structure="linear_cent")),
    _ev("Held Event", _mk_market(_TICKER_HELD_D, "HELDDEVT-1", "Held Question", "Outcome",
                                 "0.25", "0.70", price_level_structure="linear_cent")),
)
# Books for the whole Held Question group. HELD-A's YES bids give its NO ask
# of 0.45 and HELD-B's NO bids its YES ask of 0.20, so an ordinary pair on
# either held market would trade if the run let one through; HELD-C's and
# HELD-D's give the asks above
_HELD_GROUP_BOOKS = {
    _TICKER_HELD_A: {"orderbook_fp": {"yes_dollars": [["0.55", "100"]], "no_dollars": []}},
    _TICKER_HELD_B: {"orderbook_fp": {"yes_dollars": [], "no_dollars": [["0.80", "100"]]}},
    _TICKER_HELD_C: {"orderbook_fp": {"yes_dollars": [["0.60", "100"]], "no_dollars": []}},
    _TICKER_HELD_D: {"orderbook_fp": {"yes_dollars": [], "no_dollars": [["0.75", "100"]]}},
}


@pytest.mark.usefixtures("pinned_config_toggles")
class TestRunProdAddsToHeldPairs:
    """A production run with add_to_held_pairs on may add to an exact pair the
    account holds (the same two markets, the same side on each), sized on the
    whole position; off, a held market is refused exactly as before. Any
    listing or lookup it cannot trust means no add-on and one WARNING naming
    why; a held pair with no room even at the per-trade cap is left out, so
    its group's other pair can still trade. The whole position is the held
    pair's stake (its worth at today's prices plus the fees paid for it) plus
    the new stake, weighed against the portfolio value (the cash plus
    Kalshi's value of the open positions), so a run that could not count that
    value adds to nothing either."""

    @staticmethod
    def _dry_run(client, monkeypatch, caplog, *, add_on: bool,
                 expected_code=EXIT_OK, **changes) -> tuple[list, dict]:
        """
        Run a production dry run with adding to held pairs on or off.

        Args:
            client: The live-shape client.
            monkeypatch (pytest.MonkeyPatch): pytest's per-test patcher.
            caplog (pytest.LogCaptureFixture): Captures the run's log.
            add_on (bool): Keyword-only. The run's add_to_held_pairs.
            expected_code (int): Keyword-only. The exit code the run must return.
            **changes: Other LiveSettings fields to replace for this run.

        Returns:
            tuple[list, dict]: The simulated results, and what the run handed
                on: "held_pairs" (each call's positions), "markets" (the
                tickers of each call's market map), "ts" and "st" (the
                add_on_pairs each finder was handed, per call).
        """
        seen: dict = {"held_pairs": [], "markets": [], "ts": [], "st": []}
        real_held_pairs = main.held_pairs
        real_ts, real_st = main.find_time_series_pairs, main.find_same_title_pairs

        def held_pairs_spy(positions, labels, markets_by_ticker):
            seen["held_pairs"].append(dict(positions))
            seen["markets"].append(set(markets_by_ticker))
            return real_held_pairs(positions, labels, markets_by_ticker)

        def ts_spy(*args, add_on_pairs=None, **kwargs):
            seen["ts"].append(add_on_pairs)
            return real_ts(*args, add_on_pairs=add_on_pairs, **kwargs)

        def st_spy(*args, add_on_pairs=None, **kwargs):
            seen["st"].append(add_on_pairs)
            return real_st(*args, add_on_pairs=add_on_pairs, **kwargs)

        captured: dict = {}

        def fake_append_to_prod_log(results, balance_before, balance_after, *, run_note=""):
            captured["results"] = results
            return pathlib.Path("/fake/trade_log.xlsx")

        monkeypatch.setattr(main, "held_pairs", held_pairs_spy)
        monkeypatch.setattr(main, "find_time_series_pairs", ts_spy)
        monkeypatch.setattr(main, "find_same_title_pairs", st_spy)
        monkeypatch.setattr(main, "append_to_prod_log", fake_append_to_prod_log)
        reference = live_settings()
        settings = dataclasses.replace(reference, add_to_held_pairs=add_on, **changes)
        with caplog.at_level(logging.INFO):
            code = main._run_prod(client, _args(dry_run=True), settings, reference)
        assert code == expected_code
        return captured.get("results", []), seen

    # The one same-title pair of the live-shape client, which trades in every run
    _SAME_TITLE = ("same_title", _TICKER_SAME_EXP, _TICKER_SAME_CHEAP)

    @staticmethod
    def _traded(results) -> set:
        """Each trade as (pair type, market A, market B): both legs, so a held
        market traded as either leg shows."""
        return {(r.spec.pair.pair_type, r.spec.pair.market_a.ticker,
                 r.spec.pair.market_b.ticker) for r in results}

    @staticmethod
    def _warned(caplog) -> list:
        return [r.getMessage() for r in caplog.records
                if r.levelno == logging.WARNING and "adding to held pairs" in r.getMessage()]

    def _client(self, monkeypatch, *, positions_value=2_250, **kwargs):
        """
        The live-shape client holding HELD-A and the TS pair (unless replaced).

        Kalshi's value of the open positions defaults to $22.50, what they are
        worth at the fixture's asks: 3 YES on HELD-A at 0.50, 30 YES on
        TS-EARLY at 0.30 and 30 NO on TS-LATE at 0.40. The 63 contracts held
        back it (a contract pays at most $1), so the run counts it and sizes
        on the cash plus it.

        Args:
            monkeypatch (pytest.MonkeyPatch): pytest's per-test patcher.
            positions_value (int | None): Keyword-only. The balance reply's
                portfolio_value, in cents; None leaves the field out.
            **kwargs: More _live_shape_client arguments.

        Returns:
            MagicMock: The client.
        """
        kwargs.setdefault("held_rows", _HELD_TS_PAIR)
        payload = {**_LIVE_BALANCE_PAYLOAD, "portfolio_value": positions_value}
        if positions_value is None:
            del payload["portfolio_value"]
        return _live_shape_client(monkeypatch, balance_payload=payload,
                                  include_time_series=True, **kwargs)

    def test_with_it_on_the_held_pair_is_added_to_on_its_whole_position(
            self, monkeypatch, caplog):
        results, seen = self._dry_run(self._client(monkeypatch), monkeypatch, caplog,
                                      add_on=True)
        assert self._traded(results) == {
            ("time_series", _TICKER_TS_EARLY, _TICKER_TS_LATE), self._SAME_TITLE}
        [add_on] = [r for r in results if r.spec.pair.pair_type == "time_series"]
        held = scanner_mod.pair_held(add_on.spec.pair)
        assert held.sides == ((_TICKER_TS_EARLY, "yes"), (_TICKER_TS_LATE, "no"))
        assert held.count == 30.0
        assert held.cost_dollars == pytest.approx(21.95)
        # Worth at today's prices: 30 YES at TS-EARLY's 0.30 ask, 30 NO at TS-LATE's 0.40
        assert held.value_dollars == 30 * 0.30 + 30 * 0.40
        # The stake counts the fees paid for the pair too ($0.45 + $0.50)
        assert held.fees_dollars == pytest.approx(0.95)
        assert held.stake_dollars == pytest.approx(21.95)
        assert add_on.spec.pair.market_b.ticker == _TICKER_TS_LATE
        # The portfolio value: the cash plus Kalshi's value of the open positions
        value, cash = 10_271.50, 10_249.0
        assert ("Sizing on portfolio value $10271.50 = cash $10249.00 + open positions "
                "$22.50") in caplog.text
        # Sized on the whole position. The pair's own Kelly fraction at its
        # prices (pA 0.30, pB 0.60, NO ask 0.40, k 0.75): p = 1 - 0.75 x 0.30 =
        # 0.775 and b = 0.2685 / 0.7315 after the $0.0315 fee, so f* = 0.775 -
        # 0.225 / b = 0.16201, under the 20% cap. The add-on takes f* less the
        # stake's share of the portfolio value ($21.95 of $10,271.50), and its
        # cost stays within that share (and the cash); the stake and the new
        # cost together stay within 20% of the portfolio value
        f_star = 0.775 - 0.225 * 0.7315 / 0.2685
        assert add_on.spec.kelly_fraction == pytest.approx(f_star - 21.95 / value)
        assert add_on.spec.kelly_fraction <= 0.20
        assert add_on.spec.total_cost_with_fees <= config.kelly_budget(
            value, add_on.spec.kelly_fraction, cash)
        assert held.stake_dollars + add_on.spec.total_cost_with_fees <= 0.20 * value + 1e-9
        # The book's 100 contracts a level bind, not the $1,642.15 budget: 100
        # pairs at 0.30 + 0.40 plus $1.47 + $1.68 of fees
        assert (add_on.spec.x, add_on.spec.y) == (100, 100)
        assert add_on.spec.total_cost_with_fees == pytest.approx(73.15)
        # One held_pairs call over every held market, and both finders handed its pair
        assert [sorted(positions) for positions in seen["held_pairs"]] == [
            [_TICKER_HELD_A, _TICKER_TS_EARLY, _TICKER_TS_LATE]]
        # ... valued from the whole market list, read before held markets are dropped
        [markets] = seen["markets"]
        assert {_TICKER_HELD_A, _TICKER_TS_EARLY, _TICKER_TS_LATE} <= markets
        key = frozenset((_TICKER_TS_EARLY, _TICKER_TS_LATE))
        assert [set(pairs) for pairs in seen["ts"]] == [{key}]
        assert seen["st"] == seen["ts"]
        # The log names the held pair, its cost and fees, its worth at today's
        # prices and the add-on
        assert ("Held pair to add to: YES TS-EARLY / NO TS-LATE, 30 contracts each, "
                "cost $21.95 (fees $0.95), worth $21.00 at today's prices") in caplog.text
        assert "Held pairs to add to: 1 (other held markets, never added to: 1)" in caplog.text
        assert "Time-series pairs that add to a held pair: 1" in caplog.text
        assert "adds to 30 held" in caplog.text
        assert self._warned(caplog) == []

    def test_with_it_off_a_held_pair_is_refused_as_before(self, monkeypatch, caplog):
        results, seen = self._dry_run(self._client(monkeypatch), monkeypatch, caplog,
                                      add_on=False)
        assert self._traded(results) == {self._SAME_TITLE}
        # No held_pairs call, nothing handed on, no line of the add-on's own
        assert seen["held_pairs"] == []
        assert seen["ts"] == seen["st"] == [{}]
        for text in ("Account value for sizing held pairs", "Held pair",
                     "Time-series pairs that add to a held pair", "adds to",
                     "Not adding to held pairs", "at today's prices"):
            assert text not in caplog.text, text
        # Both held markets were dropped before either finder saw them, as before
        assert "Open ladder exposure: 3 held market(s)" in caplog.text

    def test_the_no_flag_turns_saved_defaults_off_for_one_run(self, monkeypatch, caplog):
        # Saved on, --no-add-to-held-pairs on the command line: no add-on
        _save_live_defaults(add_to_held_pairs=True)
        client = self._client(monkeypatch)
        captured: dict = {}

        def fake_append_to_prod_log(results, balance_before, balance_after, *, run_note=""):
            captured["results"], captured["run_note"] = results, run_note
            return pathlib.Path("/fake/trade_log.xlsx")

        held_pairs_calls = []
        monkeypatch.setattr(main, "held_pairs", lambda *a: held_pairs_calls.append(a) or {})
        with caplog.at_level(logging.INFO):
            seen = _main_with(monkeypatch, ["--mode", "prod", "--dry-run",
                                            "--no-add-to-held-pairs"],
                              _run_prod=main._run_prod, build_client=lambda mode: client,
                              append_to_prod_log=fake_append_to_prod_log)
        assert seen["code"] == EXIT_OK
        assert held_pairs_calls == []
        assert self._traded(captured["results"]) == {self._SAME_TITLE}
        assert "add to held pairs off (default: on)" in captured["run_note"]

    @pytest.mark.parametrize("held_rows, others", [
        # Unequal counts, as a partial unwind would leave them
        ((_HELD_TS_PAIR[0], {**_HELD_TS_PAIR[1], "position_fp": "-29.00"}), 3),
        # One side held twice
        ((_HELD_TS_PAIR[0], {**_HELD_TS_PAIR[1], "position_fp": "30.00"}), 3),
        # A cost the listing does not report
        ((_HELD_TS_PAIR[0], {k: v for k, v in _HELD_TS_PAIR[1].items()
                             if k != "market_exposure_dollars"}), 3),
    ], ids=["unequal counts", "one side twice", "no cost"])
    def test_a_pair_that_is_not_exact_is_never_added_to(self, monkeypatch, caplog,
                                                          held_rows, others):
        results, seen = self._dry_run(self._client(monkeypatch, held_rows=held_rows),
                                      monkeypatch, caplog, add_on=True)
        assert self._traded(results) == {self._SAME_TITLE}
        assert seen["ts"] == seen["st"] == [{}]
        assert f"Held pairs to add to: 0 (other held markets, never added to: {others})" in (
            caplog.text)
        assert self._warned(caplog) == []

    def test_a_third_held_market_on_the_ladder_means_no_add_on(self, monkeypatch, caplog):
        # A third held market on the pair's ladder: the pair is no longer alone
        # there, so the held position is not one exact pair and is not added to
        client = self._client(
            monkeypatch, extra_events=(_ev("TS Event", _TS_MID_MARKET),),
            held_rows=(*_HELD_TS_PAIR, {"ticker": _TICKER_TS_MID, "position_fp": "2.00",
                                        "market_exposure_dollars": "0.90",
                                        "fees_paid_dollars": "0.05"}))
        results, seen = self._dry_run(client, monkeypatch, caplog, add_on=True)
        assert self._traded(results) == {self._SAME_TITLE}
        assert seen["ts"] == seen["st"] == [{}]
        assert "Held pairs to add to: 0 (other held markets, never added to: 4)" in caplog.text

    def test_a_failed_held_market_lookup_means_no_add_on(self, monkeypatch, caplog):
        # A held market that cannot be looked up could share the pair's
        # ladder, so the run adds to nothing (and makes no time-series pair)
        client = self._client(monkeypatch, extra_held=("TS-GONE",))
        client.get_market_without_preload_content = MagicMock(
            return_value=_raw_json_response({"error": "not found"}, status=404,
                                            reason="Not Found"))
        results, seen = self._dry_run(client, monkeypatch, caplog, add_on=True,
                                      expected_code=EXIT_TIME_SERIES_SKIPPED)
        assert self._traded(results) == {self._SAME_TITLE}
        assert seen["held_pairs"] == [] and seen["ts"] == [] and seen["st"] == [{}]
        assert self._warned(caplog) == [
            "Not adding to held pairs this run: a market the account holds could not "
            "be looked up"]

    def test_a_listing_that_stopped_early_means_no_add_on(self, monkeypatch, caplog):
        real = main.get_held_positions

        def cut_short(client, *, complete_out=None):
            # The whole listing, but reported as a walk a cursor guard stopped
            held = real(client, complete_out=complete_out)
            complete_out["complete"] = False
            return held

        monkeypatch.setattr(main, "get_held_positions", cut_short)
        results, seen = self._dry_run(self._client(monkeypatch), monkeypatch, caplog,
                                      add_on=True)
        assert self._traded(results) == {self._SAME_TITLE}
        assert seen["held_pairs"] == [] and seen["ts"] == seen["st"] == [{}]
        assert self._warned(caplog) == [
            "Not adding to held pairs this run: the list of the account's positions was "
            "cut short, so a held market may be missing from it"]

    def test_a_pair_at_its_cap_is_left_out_so_its_group_can_trade(self, monkeypatch, caplog):
        # A held same-title pair (NO on HELD-A, the pricier YES; YES on
        # HELD-B) whose stake is already over 5% of the portfolio value, at a
        # 5% cap: its markets stay blocked, so the group's unheld pair trades.
        # It is worth $1,950.00 at today's asks (3000 NO at 0.45, 3000 YES at
        # 0.20), Kalshi's value of the positions too, so the portfolio value
        # is $12,199.00; with the $30.00 of fees paid for it, its stake is
        # $1,980.00, over the $609.95 that 5% allows and under the $2,439.80
        # of 20%. Every market of the group has a book, so an ordinary pair
        # on a held market (HELD-C against HELD-B is the group's widest) would
        # trade if the run let one through
        held_rows = (
            {"ticker": _TICKER_HELD_A, "position_fp": "-3000.00",
             "market_exposure_dollars": "1350.00", "fees_paid_dollars": "15.00"},
            {"ticker": _TICKER_HELD_B, "position_fp": "3000.00",
             "market_exposure_dollars": "600.00", "fees_paid_dollars": "15.00"},
        )
        client = self._client(monkeypatch, include_held_position=False, held_rows=held_rows,
                              extra_events=_HELD_RUNNER_UP_EVENTS, positions_value=195_000)
        books = {**_ORDERBOOK_PAYLOADS, **_HELD_GROUP_BOOKS}
        client.get_market_orderbook_without_preload_content = MagicMock(
            side_effect=lambda ticker: _raw_json_response(books.get(
                ticker, {"orderbook_fp": {"yes_dollars": [], "no_dollars": []}})))
        results, seen = self._dry_run(client, monkeypatch, caplog, add_on=True,
                                      size_cap=0.05)
        assert ("Sizing on portfolio value $12199.00 = cash $10249.00 + open positions "
                "$1950.00") in caplog.text
        # held_pairs found it (its stake's two parts: $1,950.00 of worth and
        # $30.00 of fees), then the run left it out as full
        assert ("Held pair to add to: NO HELD-A / YES HELD-B, 3000 contracts each, "
                "cost $1980.00 (fees $30.00), worth $1950.00 at today's prices") in caplog.text
        assert "Held pairs to add to: 1 (other held markets, never added to: 0)" in caplog.text
        assert "Held pairs already at their size cap, not added to this run: 1" in caplog.text
        assert seen["ts"] == seen["st"] == [{}]
        traded = self._traded(results)
        assert ("same_title", _TICKER_HELD_C, _TICKER_HELD_D) in traded
        # Neither held market is traded as either leg of any pair
        legs = {ticker for _, a, b in traded for ticker in (a, b)}
        assert not legs & {_TICKER_HELD_A, _TICKER_HELD_B}, traded
        assert "adds to" not in caplog.text
        # Control: at the run's own 20% cap the stake leaves $459.80 of room,
        # so the same pair is handed on
        results, seen = self._dry_run(client, monkeypatch, caplog, add_on=True)
        assert [set(pairs) for pairs in seen["st"]][-1] == {
            frozenset((_TICKER_HELD_A, _TICKER_HELD_B))}

    def test_the_cap_check_counts_the_fees_paid_for_the_pair(self, monkeypatch, caplog):
        """
        Pins that the at-cap check reads the held pair's stake, its worth at
        today's prices plus the fees paid for it, and not its worth alone.

        The pair (NO on HELD-A, YES on HELD-B, 3900 contracts each) is worth
        $2,535.00 at today's asks (0.45 and 0.20), which is Kalshi's value of
        the positions too, so the portfolio value is $12,784.00 and the run's
        20% cap allows $2,556.80. Its worth alone would leave $21.80 of room,
        but with the $30.00 of fees paid for it the stake is $2,565.00, over
        the cap, so the pair is left out as full and its group's unheld pair
        trades. A check that left the fees out would hand the pair on.
        """
        held_rows = (
            {"ticker": _TICKER_HELD_A, "position_fp": "-3900.00",
             "market_exposure_dollars": "1755.00", "fees_paid_dollars": "15.00"},
            {"ticker": _TICKER_HELD_B, "position_fp": "3900.00",
             "market_exposure_dollars": "780.00", "fees_paid_dollars": "15.00"},
        )
        client = self._client(monkeypatch, include_held_position=False, held_rows=held_rows,
                              extra_events=_HELD_RUNNER_UP_EVENTS, positions_value=253_500)
        books = {**_ORDERBOOK_PAYLOADS, **_HELD_GROUP_BOOKS}
        client.get_market_orderbook_without_preload_content = MagicMock(
            side_effect=lambda ticker: _raw_json_response(books.get(
                ticker, {"orderbook_fp": {"yes_dollars": [], "no_dollars": []}})))
        results, seen = self._dry_run(client, monkeypatch, caplog, add_on=True)
        value = 12_784.0
        assert ("Sizing on portfolio value $12784.00 = cash $10249.00 + open positions "
                "$2535.00") in caplog.text
        assert ("Held pair to add to: NO HELD-A / YES HELD-B, 3900 contracts each, "
                "cost $2565.00 (fees $30.00), worth $2535.00 at today's prices") in caplog.text
        # Its worth alone leaves room at the 20% cap; its worth plus its fees does not
        assert config.held_pair_fraction(0.20, 2_535.0, value) > 0
        assert config.held_pair_fraction(0.20, 2_535.0 + 30.0, value) <= 0
        # So the run leaves it out as full: nothing handed on, the unheld pair
        # trades, and neither held market is traded as either leg
        assert "Held pairs already at their size cap, not added to this run: 1" in caplog.text
        assert seen["ts"] == seen["st"] == [{}]
        traded = self._traded(results)
        assert ("same_title", _TICKER_HELD_C, _TICKER_HELD_D) in traded
        legs = {ticker for _, a, b in traded for ticker in (a, b)}
        assert not legs & {_TICKER_HELD_A, _TICKER_HELD_B}, traded
        assert "adds to" not in caplog.text

    def test_the_cap_check_reads_the_portfolio_value_and_the_worth_at_todays_prices(
            self, monkeypatch, caplog):
        """
        Pins that the at-cap check weighs the held pair's stake (its worth at
        today's prices plus the fees paid for it) against the portfolio value
        (cash plus Kalshi's value of the open positions), not its cost and not
        the cash alone.

        The pair (NO on HELD-A, YES on HELD-B, 1700 contracts each, the only
        two markets of their group) cost $1,200.00, $20.00 of it fees, but is
        worth $1,105.00 at today's asks (0.45 and 0.20), which is Kalshi's
        value of the positions too, so the portfolio value is $11,354.00 and
        the pair's stake $1,125.00. At a 10% cap it has $10.40 of room
        ($1,135.40 less its stake), so it is handed on and added to within
        that room. Weighed at its cost ($1,200.00) against the same value, or
        at its stake against the cash alone ($1,024.90), it would have been
        left out as full.
        """
        held_rows = (
            {"ticker": _TICKER_HELD_A, "position_fp": "-1700.00",
             "market_exposure_dollars": "800.00", "fees_paid_dollars": "10.00"},
            {"ticker": _TICKER_HELD_B, "position_fp": "1700.00",
             "market_exposure_dollars": "380.00", "fees_paid_dollars": "10.00"},
        )
        client = self._client(monkeypatch, include_held_position=False, held_rows=held_rows,
                              positions_value=110_500)
        books = {**_ORDERBOOK_PAYLOADS, **_HELD_GROUP_BOOKS}
        client.get_market_orderbook_without_preload_content = MagicMock(
            side_effect=lambda ticker: _raw_json_response(books.get(
                ticker, {"orderbook_fp": {"yes_dollars": [], "no_dollars": []}})))
        results, seen = self._dry_run(client, monkeypatch, caplog, add_on=True,
                                      size_cap=0.10)
        value = 11_354.0
        assert ("Sizing on portfolio value $11354.00 = cash $10249.00 + open positions "
                "$1105.00") in caplog.text
        assert ("Held pair to add to: NO HELD-A / YES HELD-B, 1700 contracts each, "
                "cost $1200.00 (fees $20.00), worth $1105.00 at today's prices") in caplog.text
        assert "Held pairs already at their size cap" not in caplog.text
        key = frozenset((_TICKER_HELD_A, _TICKER_HELD_B))
        assert [set(pairs) for pairs in seen["st"]] == [{key}]
        # By cost, or its stake on the cash alone, the same pair has no room at 10%
        assert config.held_pair_fraction(0.10, 1_200.0, value) <= 0
        assert config.held_pair_fraction(0.10, 1_125.0, 10_249.0) <= 0
        # The add-on takes at most the room left: the stake and the new cost
        # within 10% of the portfolio value
        [add_on] = [r for r in results
                    if {r.spec.pair.market_a.ticker, r.spec.pair.market_b.ticker}
                    == {_TICKER_HELD_A, _TICKER_HELD_B}]
        held = scanner_mod.pair_held(add_on.spec.pair)
        assert held.value_dollars == 1700 * 0.45 + 1700 * 0.20
        assert held.fees_dollars == pytest.approx(20.0)
        assert held.stake_dollars == pytest.approx(1_125.0)
        assert add_on.spec.kelly_fraction == config.held_pair_fraction(
            0.10, held.stake_dollars, value)
        # That leaves $10.40 to spend, where the worth alone, without the
        # fees, would have left $30.40
        budget = config.kelly_budget(value, add_on.spec.kelly_fraction, 10_249.0)
        assert budget == pytest.approx(10.40)
        assert config.kelly_budget(value, config.held_pair_fraction(
            0.10, held.value_dollars, value), 10_249.0) == pytest.approx(30.40)
        assert add_on.spec.total_cost_with_fees <= budget
        assert held.stake_dollars + add_on.spec.total_cost_with_fees <= 0.10 * value + 1e-9
        # The $10.40 of room buys 15 pairs at 0.45 + 0.20 with $0.26 + $0.17
        # of fees ($10.18 in all); 16 would cost $10.86
        assert (add_on.spec.x, add_on.spec.y) == (15, 15)
        assert add_on.spec.total_cost_with_fees == pytest.approx(10.18)

    @pytest.mark.parametrize("positions_value, why, cause", [
        (None, "Kalshi's balance reply carried no readable portfolio_value — sizing on "
               "cash alone ($10249.00) this run, as if no position were held",
         "Kalshi's value of the open positions could not be read, so the portfolio "
         "value counts only the cash"),
        (1_000_000, "Kalshi's value of the open positions ($10000.00) is not used: that "
                    "is more than the 63 contract(s) held can be worth at $1.00 each — "
                    "sizing on cash alone ($10249.00) this run, as if no position were held",
         "Kalshi's value of the open positions was refused, so the portfolio value "
         "counts only the cash"),
    ], ids=["not read", "refused"])
    def test_an_unread_or_refused_positions_value_means_no_add_on(
            self, monkeypatch, caplog, positions_value, why, cause):
        """
        Pins that a run which holds positions but could not count Kalshi's
        value of them (not read, or refused as more than the 63 contracts held
        can be worth) adds to no held pair: an add-on is sized on the
        portfolio value, which then leaves out what the account holds. One
        WARNING names the cause, held_pairs is never called, neither finder
        is handed a pair, and the same-title pair still trades on the cash.
        """
        results, seen = self._dry_run(
            self._client(monkeypatch, positions_value=positions_value), monkeypatch, caplog,
            add_on=True)
        assert self._traded(results) == {self._SAME_TITLE}
        assert seen["held_pairs"] == [] and seen["ts"] == seen["st"] == [{}]
        assert self._warned(caplog) == [f"Not adding to held pairs this run: {cause}"]
        # The one WARNING that says why the run sizes on the cash alone
        assert why in [r.getMessage() for r in caplog.records
                       if r.levelno == logging.WARNING]
        assert ("Kalshi Pair Scan — Portfolio value: $10249.00 (cash $10249.00) | Mode: PROD"
                in caplog.text)
        assert "adds to" not in caplog.text

    @pytest.mark.parametrize("positions_value", [None, 1_000_000], ids=["not read", "refused"])
    def test_holding_nothing_the_add_on_check_says_nothing(self, monkeypatch, caplog,
                                                           positions_value):
        """
        Pins that a run holding nothing logs no "Not adding to held pairs"
        WARNING when Kalshi's value of the positions was not read or refused:
        there is no held pair to leave out, so held_pairs runs over nothing.
        """
        client = self._client(monkeypatch, positions_value=positions_value,
                              include_held_position=False, held_rows=())
        results, seen = self._dry_run(client, monkeypatch, caplog, add_on=True)
        assert self._warned(caplog) == []
        assert seen["held_pairs"] == [{}]
        assert "Held pairs to add to: 0 (other held markets, never added to: 0)" in caplog.text
        assert self._SAME_TITLE in self._traded(results)

    def test_a_dev_run_reads_no_positions_even_with_it_on(self, monkeypatch, caplog):
        # A dev run trades the sandbox account, which holds nothing of
        # production's: with the setting on it still reads no positions, looks
        # up no held market and hands neither finder a held pair
        calls: list = []
        for name in ("get_held_positions", "resolve_held_ladders", "held_pairs"):
            monkeypatch.setattr(main, name,
                                lambda *a, _name=name, **k: calls.append(_name))
        seen: dict = {"ts": [], "st": []}
        real_ts, real_st = main.find_time_series_pairs, main.find_same_title_pairs

        def ts_spy(*args, add_on_pairs=None, **kwargs):
            seen["ts"].append(add_on_pairs)
            return real_ts(*args, add_on_pairs=add_on_pairs, **kwargs)

        def st_spy(*args, add_on_pairs=None, **kwargs):
            seen["st"].append(add_on_pairs)
            return real_st(*args, add_on_pairs=add_on_pairs, **kwargs)

        monkeypatch.setattr(main, "find_time_series_pairs", ts_spy)
        monkeypatch.setattr(main, "find_same_title_pairs", st_spy)
        captured = _capture_dev_simulation(monkeypatch)
        reference = live_settings()
        settings = dataclasses.replace(reference, add_to_held_pairs=True)
        with caplog.at_level(logging.INFO):
            code = main._run_dev(self._client(monkeypatch),
                                 SimpleNamespace(sandbox_balance=1000.0, max_horizon_days=None),
                                 settings, reference)
        assert code == EXIT_OK
        assert calls == []
        assert len(seen["ts"]) == len(seen["st"]) == 1
        assert not seen["ts"][0] and not seen["st"][0]
        # The run still trades as a dev run does, and says nothing of held pairs
        assert captured["results"]
        for text in ("Not adding to held pairs", "Held pair", "adds to"):
            assert text not in caplog.text, text


# One question at three deadlines, as three rungs of one event (a same-event
# ladder): will L happen by Dec 13, by Dec 20, by Dec 27. Their YES asks are
# 0.20, 0.40 and 0.70, so with nothing held the group's widest pair is
# Dec 13 / Dec 27
_TICKER_LAD_13, _TICKER_LAD_20, _TICKER_LAD_27 = "LAD-13", "LAD-20", "LAD-27"
_LADDER_TICKERS = (_TICKER_LAD_13, _TICKER_LAD_20, _TICKER_LAD_27)
_LADDER_EVENTS = (
    {"title": "Ladder Question", "markets": [
        _mk_market(_TICKER_LAD_13, "KXLAD-26", "Will L happen by Dec 13, 2026?", "Outcome",
                   "0.20", "0.80", price_level_structure="linear_cent",
                   close_time="2026-12-13T00:00:00Z"),
        _mk_market(_TICKER_LAD_20, "KXLAD-26", "Will L happen by Dec 20, 2026?", "Outcome",
                   "0.40", "0.60", price_level_structure="linear_cent",
                   close_time="2026-12-20T00:00:00Z"),
        _mk_market(_TICKER_LAD_27, "KXLAD-26", "Will L happen by Dec 27, 2026?", "Outcome",
                   "0.70", "0.30", price_level_structure="linear_cent",
                   close_time="2026-12-27T00:00:00Z"),
    ]},
)
# Each rung's book, 100 contracts a level: its NO bids give the YES asks
# above, and its YES bids the NO asks
_LADDER_BOOKS = {
    _TICKER_LAD_13: {"orderbook_fp": {"yes_dollars": [], "no_dollars": [["0.80", "100"]]}},
    _TICKER_LAD_20: {"orderbook_fp": {"yes_dollars": [["0.40", "100"]],
                                      "no_dollars": [["0.60", "100"]]}},
    _TICKER_LAD_27: {"orderbook_fp": {"yes_dollars": [["0.70", "100"]],
                                      "no_dollars": [["0.30", "100"]]}},
}


@pytest.mark.usefixtures("pinned_config_toggles")
class TestRunProdAddsToHeldPairsLive:
    """Live (not dry-run) production runs that add to a held pair, with every
    order body the run sends. The dry runs above send nothing, so they cannot
    show which orders go out, the check just before the NO leg, the V2
    NO-leg mapping check on an add-on's fill, or what an unwind touches.

    The account holds the ladder above: YES on the Dec 13 rung and NO on the
    Dec 20 rung, 30 contracts each (the user's own example); the Dec 27 rung
    is not held. Kalshi's value of those positions is $24.00, what they are
    worth at the rungs' asks (30 YES at 0.20 and 30 NO at 0.60), so the
    portfolio value is $10,273.00: the cash plus that."""

    @staticmethod
    def _run(monkeypatch, caplog, *, add_on: bool, held, killed=frozenset(),
             mapping_confirmed: bool = True, same_title: bool = True,
             lone_no: str | None = None):
        """
        Run one live production run against a stand-in exchange that keeps a ledger.

        Every order fills in full, except one whose (ticker, side) is in
        killed, which gets the exchange's HTTP 409 fill-or-kill kill. A fill
        moves its market's position from what the account held at the start
        (a bid by +count, an ask by -count), and every position read returns
        that running position, so each read after a fill sees the fill.

        Args:
            monkeypatch (pytest.MonkeyPatch): pytest's per-test patcher.
            caplog (pytest.LogCaptureFixture): Captures the run's log.
            add_on (bool): Keyword-only. The run's add_to_held_pairs.
            held (tuple[str, str] | None): Keyword-only. The ticker held YES
                and the ticker held NO, 30 contracts each, or None for an
                account that holds nothing. The balance reply values the
                positions at $24.00 when a pair is held, else at $0.
            killed (frozenset): Keyword-only. (ticker, side) of each opening
                order the exchange kills.
            mapping_confirmed (bool): Keyword-only. Whether this process has
                already confirmed the V2 NO-leg mapping (False: the run's first
                NO fill is checked).
            same_title (bool): Keyword-only. False leaves out the run's one
                same-title pair, so the ladder pair is the only one sent.
            lone_no (str | None): Keyword-only. With held None, a ticker held
                NO alone (30 contracts; its partner has paid out), valued by
                the balance reply at $18.00.

        Returns:
            tuple[int, list, list, dict]: The exit code, the run's results,
                every order body sent (in order) and each market's position at
                the end.
        """
        positions = {t: Decimal(0) for t in (*_LADDER_TICKERS, _TICKER_SAME_EXP,
                                             _TICKER_SAME_CHEAP)}
        held_rows = ()
        if held is not None:
            yes_ticker, no_ticker = held
            positions[yes_ticker], positions[no_ticker] = Decimal(30), Decimal(-30)
            held_rows = (
                {"ticker": yes_ticker, "position_fp": "30.00",
                 "market_exposure_dollars": "6.00", "fees_paid_dollars": "0.30"},
                {"ticker": no_ticker, "position_fp": "-30.00",
                 "market_exposure_dollars": "18.00", "fees_paid_dollars": "0.50"},
            )
        elif lone_no is not None:
            positions[lone_no] = Decimal(-30)
            held_rows = ({"ticker": lone_no, "position_fp": "-30.00",
                          "market_exposure_dollars": "18.00", "fees_paid_dollars": "0.50"},)
        sent: list = []

        def orders(verb, url, headers=None, body=None):
            sent.append(dict(body))
            if (body["ticker"], body["side"]) in killed and not body.get("reduce_only"):
                return _raw_json_response(
                    {"error": {"code": "fill_or_kill_insufficient_resting_volume",
                               "message": "fill or kill insufficient resting volume"}},
                    status=409, reason="Conflict")
            count = Decimal(body["count"])
            positions[body["ticker"]] += count if body["side"] == "bid" else -count
            return _raw_json_response({"order": {"order_id": f"ord-{len(sent)}",
                                                 "fill_count": int(count),
                                                 "remaining_count": 0}})

        def reader(ticker):
            return lambda: {"market_positions": [
                {"ticker": ticker, "position_fp": f"{positions[ticker]:.2f}"}]}

        client = _live_shape_client(
            monkeypatch,
            balance_payload={**_LIVE_BALANCE_PAYLOAD,
                             "portfolio_value": (2_400 if held is not None
                                                 else 1_800 if lone_no is not None else 0)},
            include_held_position=False, extra_events=_LADDER_EVENTS,
            held_rows=held_rows, order_side_effect=orders,
            position_lookup_responses={t: reader(t) for t in positions})
        books = {**_ORDERBOOK_PAYLOADS, **_LADDER_BOOKS}
        client.get_market_orderbook_without_preload_content = MagicMock(
            side_effect=lambda ticker: _raw_json_response(books.get(
                ticker, {"orderbook_fp": {"yes_dollars": [], "no_dollars": []}})))
        monkeypatch.setattr(trader_mod, "_V2_NO_MAPPING_CONFIRMED", mapping_confirmed)
        if not same_title:
            monkeypatch.setattr(main, "find_same_title_pairs",
                                lambda markets, held, *, add_on_pairs=None: [])
        captured: dict = {}

        def fake_append_to_prod_log(results, balance_before, balance_after, *, run_note=""):
            captured["results"] = results
            return pathlib.Path("/fake/trade_log.xlsx")

        monkeypatch.setattr(main, "append_to_prod_log", fake_append_to_prod_log)
        reference = live_settings()
        settings = dataclasses.replace(reference, add_to_held_pairs=add_on)
        caplog.clear()
        with caplog.at_level(logging.INFO):
            code = main._run_prod(client, _args(dry_run=False), settings, reference)
        return code, captured.get("results", []), sent, positions

    @staticmethod
    def _ladder(results) -> list:
        """The run's results on the ladder, either leg on one of its rungs."""
        return [r for r in results
                if {r.spec.pair.market_a.ticker, r.spec.pair.market_b.ticker}
                & set(_LADDER_TICKERS)]

    @staticmethod
    def _ladder_orders(sent) -> list:
        """(ticker, side, reduce_only) of every order sent on the ladder, in order."""
        return [(b["ticker"], b["side"], b["reduce_only"]) for b in sent
                if b["ticker"] in _LADDER_TICKERS]

    def test_a_lone_leg_adds_only_beside_its_held_side(self, monkeypatch, caplog):
        """
        NO is held on Dec 20 alone (the Dec 13 YES it was bought with has
        paid out). Adding on, the run trades YES on Dec 13 beside NO on Dec
        20, for the new contracts only; Dec 20 / Dec 27 would buy YES on Dec
        20 and Dec 13 / Dec 27 holds neither, so neither is sent. Adding off,
        no ladder order is sent.
        """
        l13, l20, _l27 = _LADDER_TICKERS
        _code, results, sent, positions = self._run(
            monkeypatch, caplog, add_on=True, held=None, lone_no=l20, same_title=False)
        assert self._ladder_orders(sent) == [(l20, "ask", False), (l13, "bid", False)]
        [result] = self._ladder(results)
        assert result.status == "executed"
        held = result.spec.pair.held
        assert held is not None and held.lone and held.sides == ((l20, "no"),)
        # The new contracts only: Dec 20 holds its 30 plus them, Dec 13 only them
        assert positions[l20] == -30 - result.spec.x
        assert positions[l13] == result.spec.x
        assert "Held markets to add to whose partner has paid out: 1" in caplog.text
        # Adding off: Dec 20 stays blocked and nothing is sent on the ladder
        _code, results, sent, _positions = self._run(
            monkeypatch, caplog, add_on=False, held=None, lone_no=l20, same_title=False)
        assert self._ladder_orders(sent) == []

    def test_on_trades_only_yes_13_and_no_20(self, monkeypatch, caplog):
        """
        The user's own example, live: with YES held on Dec 13 and NO on Dec
        20, adding on trades only that exact pair, buying YES on Dec 13 and NO
        on Dec 20, each order for the new contracts only; it never touches
        Dec 27, though with nothing held Dec 13 / Dec 27 is the pair the run
        would trade. With adding off no ladder order is sent, and with the
        holding the other way round (NO on Dec 13, YES on Dec 20) none is
        either, since buying YES 13 / NO 20 would reverse the held pair.
        """
        l13, l20, l27 = _LADDER_TICKERS
        # Control: holding nothing, the run trades the group's widest pair, so
        # Dec 27 is a rung the run would trade if it were let
        code, results, sent, _ = self._run(monkeypatch, caplog, add_on=True, held=None)
        assert code == EXIT_OK
        assert [(r.spec.pair.market_a.ticker, r.spec.pair.market_b.ticker)
                for r in self._ladder(results)] == [(l13, l27)]

        # Held YES 13 / NO 20, adding on: that pair alone, NO leg first
        code, results, sent, positions = self._run(monkeypatch, caplog, add_on=True,
                                                   held=(l13, l20))
        assert code == EXIT_OK, caplog.text[-3000:]
        [add_on] = self._ladder(results)
        assert add_on.status == "executed", add_on.error
        assert (add_on.spec.pair.market_a.ticker, add_on.spec.pair.market_b.ticker) == (l13, l20)
        # An ask on Dec 20 buys NO, a bid on Dec 13 buys YES; nothing reduce-only
        assert self._ladder_orders(sent) == [(l20, "ask", False), (l13, "bid", False)]
        new = add_on.spec.x
        assert [Decimal(b["count"]) for b in sent if b["ticker"] in _LADDER_TICKERS] == [
            Decimal(new), Decimal(new)]
        assert positions[l13] == 30 + new and positions[l20] == -30 - new
        assert positions[l27] == 0
        # Kelly sizes the whole position. The held pair's stake is its worth
        # at today's prices plus the fees paid for it ($0.30 + $0.50)
        held = scanner_mod.pair_held(add_on.spec.pair)
        assert held.count == 30.0
        assert held.value_dollars == 30 * 0.20 + 30 * 0.60
        assert held.fees_dollars == pytest.approx(0.80)
        assert held.stake_dollars == pytest.approx(24.80)
        value, cash = 10_273.0, 10_249.0
        assert ("Sizing on portfolio value $10273.00 = cash $10249.00 + open positions "
                "$24.00") in caplog.text
        # The pair's own Kelly fraction at its prices (pA 0.20, pB 0.40, NO ask
        # 0.60, k 0.75): p = 1 - 0.75 x 0.20 = 0.85 and b = 0.172 / 0.828
        # after the $0.028 fee, so f* = 0.85 - 0.15 / b = 0.12791, under the
        # 20% cap. The add-on takes f* less the stake's share of the portfolio
        # value ($24.80 of $10,273.00), its cost stays within that share (at
        # most the cash), and the stake and the new cost together stay within
        # 20% of the portfolio value
        f_star = 0.85 - 0.15 * 0.828 / 0.172
        assert add_on.spec.kelly_fraction == pytest.approx(f_star - 24.80 / value)
        assert add_on.spec.total_cost_with_fees <= config.kelly_budget(
            value, add_on.spec.kelly_fraction, cash)
        assert held.stake_dollars + add_on.spec.total_cost_with_fees <= 0.20 * value + 1e-9
        # The rungs' 100 contracts a level bind, not the $1,289.19 budget: 100
        # pairs at 0.20 + 0.60 plus $1.12 + $1.68 of fees
        assert new == 100
        assert add_on.spec.total_cost_with_fees == pytest.approx(82.80)
        assert ("Held pair to add to: YES LAD-13 / NO LAD-20, 30 contracts each, "
                "cost $24.80 (fees $0.80), worth $24.00 at today's prices") in caplog.text
        assert "adds to 30 held" in caplog.text
        # Dec 13 / Dec 27 and Dec 20 / Dec 27 touch the held pair's markets
        assert ("Time-series candidates refused because one of their markets is on a "
                "ladder we already hold") in caplog.text

        # Adding off: every held market is blocked, so nothing on the ladder
        code, results, sent, positions = self._run(monkeypatch, caplog, add_on=False,
                                                   held=(l13, l20))
        assert code == EXIT_OK
        assert self._ladder(results) == [] and self._ladder_orders(sent) == []
        assert (positions[l13], positions[l20]) == (30, -30)

        # Held the other way round: the pair is found, but buying YES 13 /
        # NO 20 would reverse it, so nothing on the ladder either
        code, results, sent, positions = self._run(monkeypatch, caplog, add_on=True,
                                                   held=(l20, l13))
        assert code == EXIT_OK
        assert "Held pairs to add to: 1 (other held markets, never added to: 0)" in (
            caplog.text)
        assert self._ladder(results) == [] and self._ladder_orders(sent) == []
        assert (positions[l20], positions[l13]) == (30, -30)

    def test_the_first_no_fill_of_the_process_is_an_add_on(self, monkeypatch, caplog):
        """
        When an add-on's NO leg is the first NO fill of the process, the V2
        NO-leg mapping check judges the change across that fill (-30 to
        -30 - n), not the position itself, which already read -30 before the
        pair: it confirms the mapping, the YES leg is sent and no alert fires.
        A check that read the older holding as this order's fill would stop a
        sound pair and tell the operator the mapping is wrong.
        """
        l13, l20, _ = _LADDER_TICKERS
        code, results, sent, positions = self._run(
            monkeypatch, caplog, add_on=True, held=(l13, l20),
            mapping_confirmed=False, same_title=False)
        assert code == EXIT_OK, caplog.text[-3000:]
        [add_on] = results
        assert add_on.status == "executed", add_on.error
        assert self._ladder_orders(sent) == [(l20, "ask", False), (l13, "bid", False)]
        new = add_on.spec.x
        assert positions[l20] == -30 - new and positions[l13] == 30 + new
        assert trader_mod._V2_NO_MAPPING_CONFIRMED is True
        assert f"moved the position on {l20} by -{new}" in caplog.text
        assert not [r for r in caplog.records if r.levelno >= logging.CRITICAL]

    def test_a_killed_yes_leg_unwinds_only_the_new_contracts(self, monkeypatch, caplog):
        """
        When an add-on's YES leg is killed after its NO leg filled, the unwind
        is a reduce-only bid for this pair's NO contracts only (the NO leg's
        count, never a position read), so Dec 20 goes from -30 to -30 - n and
        back to -30: the 30 contracts held before the pair stay open, and the
        Dec 13 holding is untouched.
        """
        l13, l20, _ = _LADDER_TICKERS
        code, results, sent, positions = self._run(
            monkeypatch, caplog, add_on=True, held=(l13, l20), killed={(l13, "bid")})
        assert code == EXIT_OK, caplog.text[-3000:]
        [add_on] = self._ladder(results)
        assert add_on.status == "rolled_back", add_on.error
        assert self._ladder_orders(sent) == [
            (l20, "ask", False), (l13, "bid", False), (l20, "bid", True)]
        new = add_on.spec.x
        ladder_counts = [Decimal(b["count"]) for b in sent if b["ticker"] in _LADDER_TICKERS]
        assert ladder_counts == [Decimal(new)] * 3
        assert positions[l20] == -30 and positions[l13] == 30


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


class TestAddOnMarkers:
    """A trade that adds to a pair the account already holds says so, in
    plain words, in the pairs table's Recommended Trade cell and on its
    portfolio line; an ordinary trade's cell and line are unchanged."""

    @staticmethod
    def _add_on(spec) -> SimpleNamespace:
        """
        Make make_spec()'s pair an add-on to a held pair of 30 a side.

        Args:
            spec (SimpleNamespace): A make_spec() stub.

        Returns:
            SimpleNamespace: The same stub, its pair carrying a HeldPair.
        """
        spec.pair.held = scanner_mod.HeldPair(
            sides=(("TICK-A", "yes"), ("TICK-B", "no")), count=30.0,
            cost_dollars=18.9, value_dollars=18.0, fees_dollars=0.9)
        return spec

    @staticmethod
    def _trade_cell(caplog, spec) -> str:
        """
        Render one selected row of the pairs table and return its Recommended Trade cell.

        Args:
            caplog: pytest's log capture.
            spec (SimpleNamespace): The row's spec.

        Returns:
            str: The cell's text.
        """
        caplog.clear()
        with caplog.at_level(logging.INFO):
            main.print_pairs_table([spec.pair], {id(spec.pair): spec})
        lines = [line for line in caplog.text.splitlines() if "\u2502" in line]
        headers = [c.strip() for c in lines[0].split("\u2502")[1:-1]]
        values = [c.strip() for c in lines[1].split("\u2502")[1:-1]]
        return dict(zip(headers, values, strict=True))["Recommended Trade"]

    def test_the_pairs_table_names_the_held_count(self, caplog):
        """Pins the Recommended Trade cell's "(adds to 30 held)" for an add-on
        and an ordinary trade's cell left exactly as it was."""
        assert self._trade_cell(caplog, make_spec()) == "5× YES(A) + 5× NO(B)"
        assert self._trade_cell(caplog, self._add_on(make_spec())) == (
            "5× YES(A) + 5× NO(B) (adds to 30 held)")

    def test_the_portfolio_line_names_the_held_count(self, caplog):
        """Pins that an add-on's portfolio line is an ordinary trade's line
        followed by " — adds to 30 held", and nothing else changes."""
        with caplog.at_level(logging.INFO):
            main._print_portfolio([make_spec(), self._add_on(make_spec())], "Executing")
        ordinary, add_on = [r.getMessage() for r in caplog.records
                            if r.getMessage().startswith("  [")]
        assert ordinary.endswith("(10.0% return)")
        assert add_on == ordinary + " — adds to 30 held"

    def test_a_shrunk_add_on_shows_its_own_count_and_the_held_count(self, caplog):
        """Pins that an add-on select_portfolio shrank to 2 contract pairs to fit
        the cash shows in the pairs table at that count and still says
        "(adds to 30 held)": the shrunk spec is a new copy on a copy of the
        pair, which keeps the held pair, and _display_specs finds it by its
        tickers."""
        candidate = self._add_on(make_spec()).pair
        # compute_trade returns a re-priced copy of the candidate, never the candidate
        spec = SimpleNamespace(**{**vars(make_spec()), "pair": SimpleNamespace(**vars(candidate))})
        # select_portfolio's shrink is a new, smaller spec on another copy of the pair
        shrunk = SimpleNamespace(**{**vars(spec), "x": 2, "y": 2,
                                    "pair": SimpleNamespace(**vars(spec.pair))})
        shown = main._display_specs({id(candidate): spec}, [shrunk])
        assert shown == {id(candidate): shrunk}
        caplog.clear()
        with caplog.at_level(logging.INFO):
            main.print_pairs_table([candidate], shown)
        lines = [line for line in caplog.text.splitlines() if "\u2502" in line]
        headers = [c.strip() for c in lines[0].split("\u2502")[1:-1]]
        values = [c.strip() for c in lines[1].split("\u2502")[1:-1]]
        cell = dict(zip(headers, values, strict=True))["Recommended Trade"]
        assert cell == "2× YES(A) + 2× NO(B) (adds to 30 held)"


class TestRunProdExitCodes:
    @patch("kalshi_betting.main.read_account_balance")
    def test_low_balance_returns_skip_code(self, mock_read_balance):
        # Balance below MIN_BALANCE_CENTS must short-circuit before any scan —
        # the bare `return` this used to be silently exited 0.
        mock_read_balance.return_value = _account(MIN_BALANCE_CENTS - 1)
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
    @patch("kalshi_betting.main.get_held_positions")
    @patch("kalshi_betting.main.read_account_balance")
    def test_manual_review_result_returns_attention_code(
        self,
        mock_read_balance,
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
        # Two balance reads: pre-trade balance, then post-trade balance
        # for the log's separator row.
        mock_read_balance.side_effect = [_account(100_000), _account(100_000)]
        mock_held.side_effect = _held_positions()
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
    @patch("kalshi_betting.main.get_held_positions")
    @patch("kalshi_betting.main.read_account_balance")
    def test_a_disproof_with_the_rest_of_the_run_stopped_returns_attention_code(
        self,
        mock_read_balance,
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
        mock_read_balance.side_effect = [_account(100_000), _account(100_000)]
        mock_held.side_effect = _held_positions()
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
    @patch("kalshi_betting.main.get_held_positions")
    @patch("kalshi_betting.main.read_account_balance")
    def test_clean_dry_run_returns_ok_code(
        self,
        mock_read_balance,
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
        mock_read_balance.side_effect = [_account(100_000), _account(100_000)]
        mock_held.side_effect = _held_positions()
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

    @patch("kalshi_betting.main.read_account_balance")
    def test_no_qualifying_pairs_returns_ok_code(self, mock_read_balance):
        # No-pairs / no-executable-trades paths must also resolve to EXIT_OK,
        # not just the low-balance and post-execution paths.
        mock_read_balance.return_value = _account(100_000)
        with (
            patch("kalshi_betting.main.get_held_positions", side_effect=_held_positions()),
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
        monkeypatch.setattr(main, "read_account_balance",
                            lambda client: _account(100_000))
        monkeypatch.setattr(main, "get_held_positions", _held_positions("HELD-X"))
        monkeypatch.setattr(main, "fetch_shard_statuses", lambda client: None)
        monkeypatch.setattr(main, "fetch_open_events_with_markets",
                            lambda client, inactive_shards: _stub_ingest())
        monkeypatch.setattr(main, "resolve_held_ladders",
                            lambda client, markets, held, *, labels_out=None: held_ladders)
        monkeypatch.setattr(main, "filter_markets_within_horizon", lambda m, d: m)
        monkeypatch.setattr(main, "find_time_series_pairs",
                            lambda *a, **k: ts_calls.append(k) or [])
        monkeypatch.setattr(main, "find_same_title_pairs",
                            lambda markets, held, *, add_on_pairs=None:
                            [spec.pair] if same_title else [])
        monkeypatch.setattr(main, "enrich_with_orderbook_prices",
                            lambda client, pairs, value, *, settings, cash_cents: pairs)
        monkeypatch.setattr(main, "compute_trade",
                            lambda pair, value, *, settings, cash_cents: spec)
        monkeypatch.setattr(main, "select_portfolio",
                            lambda specs, cash, *, held_ladders: specs)
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


class TestSizesOnPortfolioValue:
    """
    A production run sizes every Kelly fraction on the portfolio value — the
    cash on every shard plus Kalshi's value of the open positions — and
    spends only the cash: enrichment and compute_trade get both numbers,
    select_portfolio gets the cash, the MIN_BALANCE_CENTS gate reads the
    portfolio value, and the pairs table shows the size that trades, a spec
    select_portfolio shrank included. A dev run spends its virtual balance as
    both.
    """

    @staticmethod
    def _run(monkeypatch, caplog, account, *, dry_run=True, portfolio=None, report=None,
             positions=None):
        """
        Run _run_prod with every request stubbed and one same-title-found pair.

        Args:
            monkeypatch (pytest.MonkeyPatch): For the stubs.
            caplog (pytest.LogCaptureFixture): Captures the run's log.
            account (AccountBalance): What the one balance read before trading
                returns. The read after trading returns the same cash beside a
                different positions value, so a figure that wrongly added the
                positions to the cash after trading would show.
            dry_run (bool): Keyword-only. Run as --dry-run.
            portfolio (Callable | None): Keyword-only. Maps select_portfolio's
                specs to the portfolio it returns; None keeps them all.
            report (RunReport | None): Keyword-only. The report to fill.
            positions (Callable | None): Keyword-only. The stand-in for
                get_held_positions; None holds just enough contracts on one
                market to back the account's positions value (a contract
                pays at most $1), so the run counts that value.

        Returns:
            tuple[int, dict]: The exit code, and what the stand-ins saw, by
                name: "enrich" and "compute_trade" (the portfolio value and
                cash each was handed), "select_portfolio" (the cash),
                "trade_log" (append_to_prod_log's positional arguments),
                "held" (whether the positions were read) and "held_at" (how
                many log records the run had written when it read them).
        """
        spec = make_spec()
        seen: dict = {"held": False}
        if positions is None:
            contracts = math.ceil((account.positions_value_cents or 0) / 100)
            positions = _held_positions(*(["HELD-X"] if contracts else []),
                                        count=float(contracts))

        def held(client, *, complete_out=None):
            seen["held"] = True
            seen["held_at"] = len(caplog.records)
            return positions(client, complete_out=complete_out)

        def enrich(client, pairs, value, *, settings, cash_cents):
            seen["enrich"] = (value, cash_cents)
            return pairs

        def sized(pair, value, *, settings, cash_cents):
            seen["compute_trade"] = (value, cash_cents)
            return spec

        def select(specs, cash, *, held_ladders):
            seen["select_portfolio"] = cash
            return specs if portfolio is None else portfolio(specs)

        def trade_log(*args, **kwargs):
            seen["trade_log"] = args
            return pathlib.Path("/fake/trade_log.xlsx")

        reads = iter([account, _account(sum(account.shard_cash_cents.values()), 7_000)])
        monkeypatch.setattr(main, "read_account_balance", lambda client: next(reads))
        monkeypatch.setattr(main, "get_held_positions", held)
        monkeypatch.setattr(main, "fetch_shard_statuses", lambda client: None)
        monkeypatch.setattr(main, "fetch_open_events_with_markets",
                            lambda client, inactive_shards: _stub_ingest())
        monkeypatch.setattr(main, "resolve_held_ladders",
                            lambda client, markets, held, *, labels_out=None: frozenset())
        monkeypatch.setattr(main, "filter_markets_within_horizon", lambda m, d: m)
        monkeypatch.setattr(main, "find_time_series_pairs", lambda *a, **k: [])
        monkeypatch.setattr(main, "find_same_title_pairs",
                            lambda markets, held, *, add_on_pairs=None: [spec.pair])
        monkeypatch.setattr(main, "enrich_with_orderbook_prices", enrich)
        monkeypatch.setattr(main, "compute_trade", sized)
        monkeypatch.setattr(main, "select_portfolio", select)
        monkeypatch.setattr(main, "pre_execution_check",
                            lambda client, portfolio, *, settings: portfolio)
        monkeypatch.setattr(main, "ensure_shard_collateral",
                            lambda client, portfolio, balances, statuses, *, dry_run: portfolio)
        monkeypatch.setattr(main, "execute_trades", lambda client, portfolio, *, dry_run: [
            TradeResult(spec=s, status="simulated" if dry_run else "executed")
            for s in portfolio])
        monkeypatch.setattr(main, "append_to_prod_log", trade_log)
        with caplog.at_level(logging.INFO):
            code = main._run_prod(MagicMock(), _args(dry_run=dry_run), report=report)
        return code, seen

    @staticmethod
    def _warnings(caplog) -> list:
        """The run's WARNING lines, as logged."""
        return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]

    @staticmethod
    def _table_row(caplog) -> str:
        """The pairs table's one data row: the table's lines carry cell separators."""
        (row,) = [line for line in caplog.text.splitlines()
                  if "\u2502" in line and "Market A" in line and "Type" not in line]
        return row

    def test_the_portfolio_value_is_the_cash_plus_the_open_positions(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert main._bankroll_cents(1_000, 10_000) == 11_000
            assert main._bankroll_cents(1_000, 0) == 1_000
        # A value that was read, zero included, draws no WARNING
        assert not caplog.records

    def test_an_unreadable_positions_value_sizes_on_the_cash_alone(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert main._bankroll_cents(1_000, None) == 1_000
        assert [r.getMessage() for r in caplog.records] == [
            "Kalshi's balance reply carried no readable portfolio_value — sizing on "
            "cash alone ($10.00) this run, as if no position were held"]

    def test_the_positions_carry_a_small_cash_balance_over_the_minimum(
        self, monkeypatch, caplog,
    ):
        # Cash $10 is under the $50 minimum; with $100 of positions the
        # portfolio value is $110, so the run scans, WARNS that only the cash
        # is spent, and sizes on $110 while spending at most $10
        report = main.RunReport(dry_run=True, started_at=datetime.now(UTC))
        code, seen = self._run(monkeypatch, caplog, _account(1_000, 10_000), report=report)
        assert code == EXIT_OK
        assert seen["held"] is True
        assert seen["enrich"] == (11_000, 1_000)
        assert seen["compute_trade"] == (11_000, 1_000)
        assert seen["select_portfolio"] == 1_000
        assert ("Cash $10.00 is below the $50.00 minimum but the portfolio value is not: "
                "the run goes on, and no trade spends more than the cash left"
                ) in self._warnings(caplog)
        assert ("Sizing on portfolio value $110.00 = cash $10.00 + open positions $100.00"
                in caplog.text)
        assert ("Kalshi Pair Scan — Portfolio value: $110.00 (cash $10.00) | Mode: PROD"
                in caplog.text)
        # The run result names both: the cash before, and what was sized on
        assert report.balance_before == 10.0
        assert report.portfolio_value_before == 110.0

    def test_the_gate_reads_the_portfolio_value(self, monkeypatch, caplog):
        # $10 of cash and $39.99 of positions: $49.99 in all, one cent short
        report = main.RunReport(dry_run=True, started_at=datetime.now(UTC))
        code, seen = self._run(monkeypatch, caplog, _account(1_000, 3_999), report=report)
        assert code == EXIT_SKIPPED_LOW_BALANCE
        # Stopped before any scan
        assert seen["held"] is False and "enrich" not in seen
        message = ("Portfolio value $49.99 (cash $10.00) is below minimum $50.00 "
                   "— skipping run.")
        assert message in self._warnings(caplog)
        assert report.message == message
        assert report.balance_before == 10.0
        assert report.portfolio_value_before == 49.99
        # The cash WARNING belongs to a run that goes on, and this one did not
        assert not any(w.startswith("Cash $") for w in self._warnings(caplog))
        # The sizing line is logged on every production run, a skipped one too
        assert ("Sizing on portfolio value $49.99 = cash $10.00 + open positions $39.99"
                in caplog.text)

    def test_with_no_readable_positions_value_the_gate_reads_the_cash(
        self, monkeypatch, caplog,
    ):
        # Kalshi's value of the positions could not be read: $10 of cash alone
        # is under the minimum, so the run stops, saying why it sized on cash
        code, seen = self._run(monkeypatch, caplog, _account(1_000, None))
        assert code == EXIT_SKIPPED_LOW_BALANCE
        assert seen["held"] is False
        warnings = self._warnings(caplog)
        assert any("no readable portfolio_value" in w for w in warnings)
        assert ("Portfolio value $10.00 (cash $10.00) is below minimum $50.00 "
                "— skipping run.") in warnings

    def test_a_portfolio_value_of_exactly_the_minimum_is_traded(self, monkeypatch, caplog):
        code, seen = self._run(monkeypatch, caplog, _account(MIN_BALANCE_CENTS - 1, 1))
        assert code == EXIT_OK
        assert seen["enrich"] == (MIN_BALANCE_CENTS, MIN_BALANCE_CENTS - 1)

    def test_the_trade_log_and_the_run_result_balances_stay_on_cash(
        self, monkeypatch, caplog,
    ):
        # The workbook's separator row and the run result's before and after
        # figures are cash, as they always were; only sizing reads the
        # portfolio value. Both reads carry open positions, so a figure that
        # added them in, before or after trading, would not equal the cash
        report = main.RunReport(dry_run=False, started_at=datetime.now(UTC))
        code, seen = self._run(monkeypatch, caplog, _account(100_000, 50_000),
                               dry_run=False, report=report)
        assert code == EXIT_OK
        assert seen["trade_log"][1:] == (1_000.0, 1_000.0)
        assert report.balance_before == 1_000.0
        assert report.balance_after == 1_000.0
        assert report.portfolio_value_before == 1_500.0

    def test_a_shrunk_spec_shows_in_the_pairs_table(self, monkeypatch, caplog):
        # select_portfolio hands back a spec it shrank to fit the cash as a new,
        # smaller spec; the table shows that size, not "—" and not the old one
        def shrink(specs):
            [spec] = specs
            return [SimpleNamespace(**{**vars(spec), "x": 2, "y": 2})]

        code, _ = self._run(monkeypatch, caplog, _account(100_000), portfolio=shrink)
        assert code == EXIT_OK
        row = self._table_row(caplog)
        assert "2× YES(A) + 2× NO(B)" in row and "5×" not in row, row

    def test_display_specs_matches_the_selected_spec_to_its_candidate_by_tickers(self):
        candidate_1, candidate_2 = make_spec().pair, make_spec().pair
        candidate_2.market_a.ticker, candidate_2.market_b.ticker = "TICK-C", "TICK-D"
        # compute_trade returns a re-priced copy of each candidate, never the candidate
        spec_1 = SimpleNamespace(pair=SimpleNamespace(**vars(candidate_1)), x=5)
        spec_2 = SimpleNamespace(pair=SimpleNamespace(**vars(candidate_2)), x=7)
        trade_specs = {id(candidate_1): spec_1, id(candidate_2): spec_2}
        # Unshrunk: the very spec compute_trade returned, and only it
        shown = main._display_specs(trade_specs, [spec_2])
        assert list(shown) == [id(candidate_2)] and shown[id(candidate_2)] is spec_2
        # Shrunk into a new object with the same two tickers: the new one shows
        shrunk = SimpleNamespace(pair=SimpleNamespace(**vars(spec_1.pair)), x=2)
        shown = main._display_specs(trade_specs, [shrunk, spec_2])
        assert set(shown) == {id(candidate_1), id(candidate_2)}
        assert shown[id(candidate_1)] is shrunk and shown[id(candidate_2)] is spec_2
        # Nothing selected, nothing shown
        assert main._display_specs(trade_specs, []) == {}

    def test_a_live_shape_balance_with_a_portfolio_value_sizes_on_both(
        self, monkeypatch, caplog,
    ):
        # The balance reply carries portfolio_value as Kalshi sends it: integer
        # cents, the positions alone. $50.00 of positions beside $10,249.00 of
        # cash (the breakdown's sum) is a $10,299.00 portfolio value. The
        # account holds 100 YES on HELD-A, worth $50.00 at its 0.50 ask: a
        # contract pays at most $1, so the run counts the value only when that
        # many contracts back it
        client = _live_shape_client(
            monkeypatch, balance_payload={**_LIVE_BALANCE_PAYLOAD, "portfolio_value": 5_000},
            include_held_position=False,
            held_rows=({"ticker": _TICKER_HELD_A, "position_fp": "100.00"},))
        monkeypatch.setattr(main, "append_to_prod_log",
                            lambda *a, **k: pathlib.Path("/fake/trade_log.xlsx"))
        seen: dict = {"enrich": [], "compute_trade": []}
        real_enrich, real_compute = main.enrich_with_orderbook_prices, main.compute_trade

        def enrich(client_, pairs, value, *, settings, cash_cents):
            seen["enrich"].append((value, cash_cents))
            return real_enrich(client_, pairs, value, settings=settings, cash_cents=cash_cents)

        def sized(pair, value, *, settings, cash_cents):
            seen["compute_trade"].append((value, cash_cents))
            return real_compute(pair, value, settings=settings, cash_cents=cash_cents)

        monkeypatch.setattr(main, "enrich_with_orderbook_prices", enrich)
        monkeypatch.setattr(main, "compute_trade", sized)
        report = main.RunReport(dry_run=True, started_at=datetime.now(UTC))
        with caplog.at_level(logging.INFO):
            code = main._run_prod(client, SimpleNamespace(dry_run=True, max_horizon_days=None),
                                  report=report)
        assert code == EXIT_OK
        assert ("Sizing on portfolio value $10299.00 = cash $10249.00 + open positions $50.00"
                in caplog.text)
        assert "no readable portfolio_value" not in caplog.text
        assert seen["enrich"] == [(1_029_900, 1_024_900)]
        assert seen["compute_trade"]
        assert set(seen["compute_trade"]) == {(1_029_900, 1_024_900)}
        assert report.balance_before == 10_249.0
        assert report.portfolio_value_before == 10_299.0
        # The read after trading reports the cash, never the positions
        assert report.balance_after == 10_249.0
        assert report.trades and {t.status for t in report.trades} == {"simulated"}

    def test_a_live_shape_balance_without_a_portfolio_value_sizes_on_cash(
        self, monkeypatch, caplog,
    ):
        # The same reply with no portfolio_value: the run sizes on its cash,
        # says so, and trades as it did before the positions were counted
        payload = {key: value for key, value in _LIVE_BALANCE_PAYLOAD.items()
                   if key != "portfolio_value"}
        client = _live_shape_client(monkeypatch, balance_payload=payload)
        monkeypatch.setattr(main, "append_to_prod_log",
                            lambda *a, **k: pathlib.Path("/fake/trade_log.xlsx"))
        report = main.RunReport(dry_run=True, started_at=datetime.now(UTC))
        with caplog.at_level(logging.INFO):
            code = main._run_prod(client, SimpleNamespace(dry_run=True, max_horizon_days=None),
                                  report=report)
        assert code == EXIT_OK
        assert any("no readable portfolio_value" in r.getMessage()
                   for r in caplog.records if r.levelno == logging.WARNING)
        # The unread value is said to be unread, never printed as a measured $0.00
        assert ("Sizing on portfolio value $10249.00 = cash $10249.00 + open positions "
                "not read (counted as $0.00)" in caplog.text)
        assert report.portfolio_value_before == report.balance_before == 10_249.0

    def test_a_live_shape_account_short_of_cash_but_not_of_value_scans(
        self, monkeypatch, caplog,
    ):
        # $8.00 of cash, under the minimum, beside $100.00 of positions: 200
        # YES on HELD-A, worth $100.00 at its 0.50 ask, which the contracts
        # held can back (a contract pays at most $1)
        client = _live_shape_client(
            monkeypatch, balance_payload={**_LOW_BALANCE_PAYLOAD, "portfolio_value": 10_000},
            include_held_position=False,
            held_rows=({"ticker": _TICKER_HELD_A, "position_fp": "200.00"},))
        monkeypatch.setattr(main, "append_to_prod_log",
                            lambda *a, **k: pathlib.Path("/fake/trade_log.xlsx"))
        with caplog.at_level(logging.INFO):
            code = main._run_prod(client, SimpleNamespace(dry_run=True, max_horizon_days=None))
        assert code == EXIT_OK
        assert "below minimum" not in caplog.text
        assert ("Cash $8.00 is below the $50.00 minimum but the portfolio value is not"
                in caplog.text)
        client.get_events_without_preload_content.assert_called()
        # No trade spends more than the cash: at most $8.00 in all, at the
        # orders' limit prices
        (portfolio_line,) = [r.getMessage() for r in caplog.records
                             if r.getMessage().startswith("Portfolio: ")]
        spent = float(portfolio_line.rsplit("up to $", 1)[1].split(" ", 1)[0])
        assert spent <= 8.00, portfolio_line

    def test_a_dev_run_spends_its_virtual_balance_as_cash(self, monkeypatch, caplog):
        # Dev holds nothing: the virtual balance is both the portfolio value
        # and the cash, for enrichment, the sizer and the portfolio walk
        spec = make_spec()
        seen: dict = {}

        def enrich(client, pairs, value, *, settings, cash_cents):
            seen["enrich"] = (value, cash_cents)
            return pairs

        def sized(pair, value, *, settings, cash_cents):
            seen["compute_trade"] = (value, cash_cents)
            return spec

        def select(specs, cash, **kwargs):
            seen["select_portfolio"] = (cash, kwargs)
            return [SimpleNamespace(**{**vars(specs[0]), "x": 3, "y": 3})]

        monkeypatch.setattr(main, "fetch_shard_statuses", lambda client: None)
        monkeypatch.setattr(main, "fetch_open_events_with_markets",
                            lambda client, inactive_shards: _stub_ingest())
        monkeypatch.setattr(main, "filter_markets_within_horizon", lambda m, d: m)
        monkeypatch.setattr(main, "find_time_series_pairs", lambda *a, **k: [])
        monkeypatch.setattr(main, "find_same_title_pairs",
                            lambda markets, held_tickers: [spec.pair])
        monkeypatch.setattr(main, "enrich_with_orderbook_prices", enrich)
        monkeypatch.setattr(main, "compute_trade", sized)
        monkeypatch.setattr(main, "select_portfolio", select)
        monkeypatch.setattr(main, "execute_trades", lambda client, portfolio, *, dry_run: [
            TradeResult(spec=s, status="simulated") for s in portfolio])
        monkeypatch.setattr(main, "write_dev_simulation", lambda *a, **k: pathlib.Path("/f"))
        with caplog.at_level(logging.INFO):
            code = main._run_dev(
                MagicMock(), SimpleNamespace(sandbox_balance=1234.5, max_horizon_days=None))
        assert code == EXIT_OK
        assert seen["enrich"] == (123_450, 123_450)
        assert seen["compute_trade"] == (123_450, 123_450)
        # Dev holds nothing, so no held ladders are passed
        assert seen["select_portfolio"] == (123_450, {})
        # The shrunk spec is what the pairs table shows
        row = self._table_row(caplog)
        assert "3× YES(A) + 3× NO(B)" in row, row


class TestPositionsValueCheck:
    """
    Kalshi's value of the open positions counts in the portfolio value only
    when the contracts held can back it: a contract pays at most
    config.CONTRACT_PAYOUT_DOLLARS ($1), so the value may be at most that
    much per contract held, counted over a positions listing read to its end
    whose every count can be read (main._checked_positions_value). A refused
    value draws one WARNING, and the run sizes on the cash alone, records the
    cash as its portfolio value and applies the $50 minimum to it again. The
    check runs once the positions are read, after the first minimum, and
    #86's low-cash WARNING comes after it.
    """

    @staticmethod
    def _positions(*counts, complete=True):
        """
        Build a stand-in for get_held_positions holding one market per count.

        Args:
            *counts (float | None): Each market's signed contract count (None: unreadable).
            complete (bool): Keyword-only. Whether the listing reads as read to its end.

        Returns:
            callable: (client, *, complete_out=None) -> {ticker: HeldPosition}.
        """
        def fake(client, *, complete_out=None):
            if complete_out is not None:
                complete_out["complete"] = complete
            return {f"HELD-{i}": scanner_mod.HeldPosition(f"HELD-{i}", count, None, None)
                    for i, count in enumerate(counts)}
        return fake

    @staticmethod
    def _check(value, *counts, complete=True, cash=10_000):
        """
        Call main._checked_positions_value on markets held with these counts.

        Args:
            value (int | None): Kalshi's value of the open positions, in cents.
            *counts (float | None): Each held market's signed contract count.
            complete (bool): Keyword-only. Whether the listing was read to its end.
            cash (int): Keyword-only. The cash, in cents, named in the WARNING.

        Returns:
            int | None: What the check returns.
        """
        held = {f"HELD-{i}": scanner_mod.HeldPosition(f"HELD-{i}", count, None, None)
                for i, count in enumerate(counts)}
        return main._checked_positions_value(cash, value, held, complete)

    @staticmethod
    def _warnings(caplog) -> list:
        """The WARNING lines logged, as written."""
        return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]

    def test_a_value_the_contracts_can_back_is_kept(self, caplog):
        # 30 YES and 30 NO held: 60 contracts can be worth at most $60.00
        with caplog.at_level(logging.WARNING):
            assert self._check(2_250, 30.0, -30.0) == 2_250
            assert self._check(6_000, 30.0, -30.0) == 6_000
        assert not caplog.records

    def test_a_value_above_what_the_contracts_can_pay_is_refused(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert self._check(6_001, 30.0, -30.0) is None
        assert self._warnings(caplog) == [
            "Kalshi's value of the open positions ($60.01) is not used: that is more than "
            "the 60 contract(s) held can be worth at $1.00 each — sizing on cash alone "
            "($100.00) this run, as if no position were held"]

    def test_fractional_counts_are_summed_without_float_noise(self, caplog):
        # 0.7 + 0.1 contracts sums to 0.7999999999999999 in floating point;
        # exactly 80 cents is still within what they can pay, and 81 is not
        with caplog.at_level(logging.WARNING):
            assert self._check(80, 0.7, 0.1) == 80
            assert not caplog.records
            assert self._check(81, 0.7, 0.1) is None
        assert self._warnings(caplog) == [
            "Kalshi's value of the open positions ($0.81) is not used: that is more than "
            "the 0.8 contract(s) held can be worth at $1.00 each — sizing on cash alone "
            "($100.00) this run, as if no position were held"]

    def test_holding_nothing_keeps_a_zero_value_and_refuses_any_other(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert self._check(0) == 0
            assert not caplog.records
            assert self._check(1) is None
        assert self._warnings(caplog) == [
            "Kalshi's value of the open positions ($0.01) is not used: that is more than "
            "the 0 contract(s) held can be worth at $1.00 each — sizing on cash alone "
            "($100.00) this run, as if no position were held"]

    def test_an_unread_value_is_left_to_the_balance_read_s_own_warning(self, caplog):
        # _bankroll_cents already counted it as $0 and said so; nothing to check
        with caplog.at_level(logging.WARNING):
            assert self._check(None, 30.0, None, complete=False) is None
        assert not caplog.records

    def test_a_cut_short_listing_refuses_any_value_but_zero(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert self._check(0, 30.0, complete=False) == 0
            assert not caplog.records
            assert self._check(100, 300.0, complete=False) is None
        assert self._warnings(caplog) == [
            "Kalshi's value of the open positions ($1.00) is not used: the list of the "
            "account's positions was cut short, so not every contract held is known — "
            "sizing on cash alone ($100.00) this run, as if no position were held"]

    def test_an_unreadable_count_refuses_the_value(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert self._check(100, 300.0, None) is None
        assert self._warnings(caplog) == [
            "Kalshi's value of the open positions ($1.00) is not used: the contract count "
            "of 1 held market(s) could not be read — sizing on cash alone ($100.00) this "
            "run, as if no position were held"]

    def test_a_refused_value_sizes_on_the_cash_alone(self, monkeypatch, caplog):
        # $500.00 of positions beside $1,000.00 of cash, but only 400 contracts held
        report = main.RunReport(dry_run=True, started_at=datetime.now(UTC))
        code, seen = TestSizesOnPortfolioValue._run(
            monkeypatch, caplog, _account(100_000, 50_000), report=report,
            positions=self._positions(400.0))
        assert code == EXIT_OK
        assert seen["enrich"] == seen["compute_trade"] == (100_000, 100_000)
        assert seen["select_portfolio"] == 100_000
        # The one WARNING that says why, after the line that read the value
        assert ("Sizing on portfolio value $1500.00 = cash $1000.00 + open positions $500.00"
                in caplog.text)
        assert [w for w in self._warnings(caplog) if "is not used" in w] == [
            "Kalshi's value of the open positions ($500.00) is not used: that is more than "
            "the 400 contract(s) held can be worth at $1.00 each — sizing on cash alone "
            "($1000.00) this run, as if no position were held"]
        # ... and it comes after that line and after the positions were read,
        # since the check needs the contracts the account holds
        messages = [r.getMessage() for r in caplog.records]
        sizing = messages.index("Sizing on portfolio value $1500.00 = cash $1000.00 + "
                                "open positions $500.00")
        [refused] = [i for i, m in enumerate(messages) if "is not used" in m]
        assert sizing < seen["held_at"] <= refused
        assert ("Kalshi Pair Scan — Portfolio value: $1000.00 (cash $1000.00) | Mode: PROD"
                in caplog.text)
        # The run result records the cash as what the run sized on
        assert report.balance_before == 1_000.0
        assert report.portfolio_value_before == 1_000.0

    def test_a_value_the_contracts_back_is_counted(self, monkeypatch, caplog):
        # The same account holding 500 contracts: the value is counted
        report = main.RunReport(dry_run=True, started_at=datetime.now(UTC))
        code, seen = TestSizesOnPortfolioValue._run(
            monkeypatch, caplog, _account(100_000, 50_000), report=report,
            positions=self._positions(250.0, -250.0))
        assert code == EXIT_OK
        assert seen["enrich"] == seen["compute_trade"] == (150_000, 100_000)
        assert not [w for w in self._warnings(caplog) if "is not used" in w]
        assert report.portfolio_value_before == 1_500.0

    @pytest.mark.parametrize("positions", [
        _positions(1_000.0, complete=False),
        _positions(1_000.0, None),
    ], ids=["cut short", "unreadable count"])
    def test_a_cut_short_listing_or_an_unreadable_count_refuses_the_value(
            self, monkeypatch, caplog, positions):
        # 1,000 contracts would back the $500.00, but the listing stopped
        # early or a count could not be read, so the check cannot count them
        report = main.RunReport(dry_run=True, started_at=datetime.now(UTC))
        code, seen = TestSizesOnPortfolioValue._run(
            monkeypatch, caplog, _account(100_000, 50_000), report=report,
            positions=positions)
        assert code == EXIT_OK
        assert seen["enrich"] == seen["compute_trade"] == (100_000, 100_000)
        assert len([w for w in self._warnings(caplog) if "is not used" in w]) == 1
        assert report.portfolio_value_before == report.balance_before == 1_000.0

    def test_holding_nothing_a_value_above_zero_is_refused(self, monkeypatch, caplog):
        code, seen = TestSizesOnPortfolioValue._run(
            monkeypatch, caplog, _account(100_000, 1), positions=self._positions())
        assert code == EXIT_OK
        assert seen["enrich"] == (100_000, 100_000)
        assert len([w for w in self._warnings(caplog) if "is not used" in w]) == 1
        # ... while a value of 0 holding nothing is counted as it is, silently
        caplog.clear()
        code, seen = TestSizesOnPortfolioValue._run(
            monkeypatch, caplog, _account(100_000, 0), positions=self._positions())
        assert code == EXIT_OK
        assert seen["enrich"] == (100_000, 100_000)
        assert self._warnings(caplog) == ["Running in PRODUCTION mode — real money will be used!"]

    def test_a_refused_value_with_the_cash_under_the_minimum_skips_the_run(
            self, monkeypatch, caplog):
        # $10.00 of cash and $100.00 of positions pass the first minimum, but
        # only 50 contracts are held: on the cash alone the run is under $50
        report = main.RunReport(dry_run=True, started_at=datetime.now(UTC))
        code, seen = TestSizesOnPortfolioValue._run(
            monkeypatch, caplog, _account(1_000, 10_000), report=report,
            positions=self._positions(50.0))
        assert code == EXIT_SKIPPED_LOW_BALANCE
        # The positions were read, then the run stopped before any scan
        assert seen["held"] is True and "enrich" not in seen
        message = ("Portfolio value $10.00 (cash $10.00) is below minimum $50.00 "
                   "— skipping run.")
        assert self._warnings(caplog)[-1] == message
        assert report.message == message
        assert report.balance_before == report.portfolio_value_before == 10.0
        # The low-cash WARNING comes after the check, so a run the check
        # stopped never logs it
        assert not any(w.startswith("Cash $") for w in self._warnings(caplog))

    def test_a_run_stopped_by_the_first_minimum_reads_no_positions(self, monkeypatch, caplog):
        # $49.99 in all, holding nothing that could back the $39.99: the run
        # stops at the first minimum, before the positions or the check
        code, seen = TestSizesOnPortfolioValue._run(
            monkeypatch, caplog, _account(1_000, 3_999), positions=self._positions())
        assert code == EXIT_SKIPPED_LOW_BALANCE
        assert seen["held"] is False
        assert not [w for w in self._warnings(caplog) if "is not used" in w]

    def test_the_low_cash_warning_comes_after_the_check(self, monkeypatch, caplog):
        # $10.00 of cash under the minimum beside $100.00 of positions the 100
        # contracts held back: the run goes on, and says so once the check has run
        code, seen = TestSizesOnPortfolioValue._run(
            monkeypatch, caplog, _account(1_000, 10_000), positions=self._positions(100.0))
        assert code == EXIT_OK
        assert seen["enrich"] == (11_000, 1_000)
        [index] = [i for i, r in enumerate(caplog.records)
                   if r.getMessage().startswith("Cash $10.00 is below the $50.00 minimum")]
        assert index >= seen["held_at"]


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
                "kalshi_betting.main.read_account_balance",
                return_value=_account(MIN_BALANCE_CENTS * 10),
            ),
            patch("kalshi_betting.main.get_held_positions", side_effect=_held_positions()),
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


# main.main() starts only from saved live defaults: config.py's, saved first
@pytest.mark.usefixtures("saved_live_defaults")
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

    @patch("kalshi_betting.main.read_account_balance")
    @patch("kalshi_betting.main.build_client")
    def test_main_prod_mode_low_balance_exits_skip_code(
        self, mock_build_client, mock_read_balance, tmp_path, monkeypatch,
    ):
        mock_build_client.return_value = MagicMock()
        mock_read_balance.return_value = _account(MIN_BALANCE_CENTS - 1)

        monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(sys, "argv", ["kalshi_betting.main", "--mode", "prod"])

        with pytest.raises(SystemExit) as exc_info:
            main.main()

        assert exc_info.value.code == EXIT_SKIPPED_LOW_BALANCE
        assert exc_info.value.code == 10

    @patch("kalshi_betting.main.fetch_open_events_with_markets", return_value=[])
    @patch("kalshi_betting.main.fetch_shard_statuses")
    @patch("kalshi_betting.main.get_held_positions", side_effect=_held_positions())
    @patch("kalshi_betting.main.read_account_balance")
    @patch("kalshi_betting.main.build_client")
    def test_main_prod_mode_blind_run_exits_no_tradeable_shards_code(
        self, mock_build_client, mock_read_balance, mock_held, mock_shard_statuses,
        mock_fetch, tmp_path, monkeypatch,
    ):
        # _run_prod's exit-30 return is covered directly elsewhere, but nothing
        # asserted that main() actually propagates it to sys.exit — and that
        # process code is the ONLY signal scheduler.run_job has that the weekly
        # slot went unscanned rather than merely finding no edge (TS-01).
        mock_build_client.return_value = MagicMock()
        mock_read_balance.return_value = _account(MIN_BALANCE_CENTS * 10)
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


@pytest.mark.usefixtures("saved_live_defaults")
class TestRunLock:
    """
    main() holds the machine-wide live-run lock (run_lock) through a
    production run that sends orders, from before it builds a client until the
    run mode returns or raises, and stops with EXIT_RUN_IN_PROGRESS, before
    building a client, when another run still holds it. A prod dry run and a
    dev run neither take the lock nor wait for it. tests/conftest.py's
    _isolate_live_runs points the lock at this test's tmp_path and shortens
    the wait, so holding it here (run_lock.acquire on a second open of the
    file, which an flock refuses even within one process) stands in for
    another run on the machine.
    """

    def test_a_held_lock_stops_a_production_run_before_any_client(self, monkeypatch, caplog):
        holder = run_lock.acquire()
        try:
            with caplog.at_level(logging.INFO):
                seen = _main_with(monkeypatch, ["--mode", "prod"])
        finally:
            os.close(holder)
        assert seen["code"] == EXIT_RUN_IN_PROGRESS == 50
        # Logged (the WARNING lands in kalshi_arb.log), then stopped: no
        # client, no run mode, no request
        assert seen["logging_set_up"] is True
        assert seen["client_built"] is False
        assert "mode" not in seen
        warned = [r for r in caplog.records if r.levelno == logging.WARNING
                  and "Another live trading run is in progress" in r.getMessage()]
        assert len(warned) == 1
        message = warned[0].getMessage()
        assert f"process {os.getpid()} in {config.PROJECT_ROOT}, since " in message
        assert "without contacting Kalshi" in message
        assert f"(exit {EXIT_RUN_IN_PROGRESS})" in message

    def test_two_production_runs_in_one_process_each_take_the_lock(self, monkeypatch):
        seen_during_run = []

        def prod_run(client, args, settings, reference, report=None):
            """
            Record whether the lock is held while the run mode runs, and by whom.

            Args:
                client: The client main() built.
                args (argparse.Namespace): The parsed flags.
                settings (LiveSettings): The run's settings.
                reference (LiveSettings): The saved defaults they were built from.
                report (reporter.RunReport | None): The run report (unused).

            Returns:
                int: EXIT_OK.
            """
            seen_during_run.append((run_lock.held(), run_lock.holder().pid))
            return EXIT_OK

        first = _main_with(monkeypatch, ["--mode", "prod"], _run_prod=prod_run)
        second = _main_with(monkeypatch, ["--mode", "prod"], _run_prod=prod_run)
        assert first["code"] == second["code"] == EXIT_OK
        # Held through each run, by this process, and released after each
        assert seen_during_run == [(True, os.getpid()), (True, os.getpid())]
        assert run_lock.held() is False

    def test_a_run_that_raises_still_releases_the_lock(self, monkeypatch):
        def failing_run(client, args, settings, reference, report=None):
            """
            Fail the way an unhandled error in the run mode would.

            Args:
                client: The client main() built.
                args (argparse.Namespace): The parsed flags.
                settings (LiveSettings): The run's settings.
                reference (LiveSettings): The saved defaults they were built from.
                report (reporter.RunReport | None): The run report (unused).

            Raises:
                RuntimeError: Always.
            """
            raise RuntimeError("run failed")

        with pytest.raises(RuntimeError, match="run failed"):
            _main_with(monkeypatch, ["--mode", "prod"], _run_prod=failing_run)
        assert run_lock.held() is False

    @pytest.mark.parametrize("argv", [["--mode", "prod", "--dry-run"], ["--mode", "dev"],
                                      ["--mode", "dev", "--dry-run"]],
                             ids=["prod-dry-run", "dev", "dev-dry-run"])
    def test_a_run_that_sends_no_orders_neither_takes_nor_waits_for_the_lock(
        self, monkeypatch, argv,
    ):
        # Another run holds the lock, and taking it would fail this test
        holder = run_lock.acquire()

        def no_acquire() -> int | None:
            """
            Stand in for run_lock.acquire, which a run that sends no orders never calls.

            Returns:
                int | None: Never returns.

            Raises:
                AssertionError: Always, so the test fails if the run takes the lock.
            """
            raise AssertionError("a run that sends no orders took the live-run lock")

        monkeypatch.setattr(run_lock, "acquire", no_acquire)
        try:
            seen = _main_with(monkeypatch, argv)
        finally:
            os.close(holder)
        assert seen["code"] == EXIT_OK
        assert seen["client_built"] is True
        assert seen["mode"] == argv[1]

    def test_an_unwritable_lock_folder_stops_the_run_with_an_error(self, monkeypatch, tmp_path):
        # Not EXIT_RUN_IN_PROGRESS, which the scheduler counts as a done slot:
        # the error propagates and the interpreter exits 1
        locked = tmp_path / "read-only"
        locked.mkdir()
        locked.chmod(0o500)
        monkeypatch.setattr(config, "LIVE_RUN_LOCK_FILE", locked / "sub" / "live_run.lock")
        built = []
        try:
            with pytest.raises(PermissionError):
                _main_with(monkeypatch, ["--mode", "prod"],
                           build_client=lambda mode: built.append(mode))
        finally:
            locked.chmod(0o700)
        assert built == []


def _read_result(path: pathlib.Path) -> dict:
    """
    Parse a run result file as strict JSON.

    Args:
        path (pathlib.Path): The result file.

    Returns:
        dict: The parsed record.

    Raises:
        AssertionError: If the file holds NaN or an infinity, which strict JSON has not.
    """
    def refuse(token):
        """
        Fail on a constant strict JSON has not.

        Args:
            token (str): "NaN", "Infinity" or "-Infinity".

        Raises:
            AssertionError: Always.
        """
        raise AssertionError(f"the run result is not strict JSON: {token}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=refuse)


def _no_report_handler() -> bool:
    """
    Whether the root logger carries no run-report handler.

    Returns:
        bool: True when no reporter.RunReportHandler is attached to the root logger.
    """
    return not any(isinstance(h, main.RunReportHandler) for h in logging.getLogger().handlers)


# The record make_spec()'s pair gives in a run result: a time-series pair
# buying YES on A at pA and NO on B at nB, five contracts a side
_MAKE_SPEC_TRADE = {
    "pair_type": "time_series", "title": "Test pair",
    "a": {"ticker": "TICK-A", "market": "Market A", "side": "yes", "count": 5, "price": 0.3},
    "b": {"ticker": "TICK-B", "market": "Market B", "side": "no", "count": 5, "price": 0.4},
    "cost_with_fees": 2.3, "profit_if_won": 0.5, "adds_to_held": None,
}


@pytest.mark.usefixtures("saved_live_defaults")
class TestResultFile:
    """
    main.py --result-file writes what a production run did as JSON, however
    it ends once logging is set up: one test per way _run_prod ends, each
    checking the exit code, the message (the line the run logged, word for
    word), the trades and the balances; then an exception, a run stopped
    while sending orders, an undescribable pair, a lock refusal, a usage
    error (exit 2, no file, an older one removed), the flag in dev, no flag,
    and the handler's removal.
    """

    # The balance before trading and after it, in cents
    _BEFORE = 100_000
    _AFTER = 99_000

    def _run(self, monkeypatch, tmp_path, caplog, *, dry_run=False, status="executed",
             **stubs) -> tuple[dict, dict, list]:
        """
        Run main.main() in production with --result-file and every request stubbed.

        By default one same-title-found pair (make_spec's) is sized, passes
        every check and ends with the given status. Each keyword in stubs
        replaces one of main's names for this run.

        Args:
            monkeypatch (pytest.MonkeyPatch): For the stubs.
            tmp_path (pathlib.Path): Where the result file goes.
            caplog (pytest.LogCaptureFixture): Captures the run's log lines.
            dry_run (bool): Pass --dry-run.
            status (str): The status execute_trades gives each pair.
            **stubs: main's names to replace, e.g. read_account_balance=...

        Returns:
            tuple[dict, dict, list]: What _main_with saw, the parsed result
                file, and the run's log records.
        """
        spec = make_spec()
        path = tmp_path / "result.json"
        balances = iter([_account(self._BEFORE), _account(self._AFTER)])
        logged = []
        patches = {
            "_run_prod": main._run_prod,
            "read_account_balance": lambda client: next(balances),
            "get_held_positions": _held_positions(),
            "fetch_shard_statuses": lambda client: None,
            "fetch_open_events_with_markets": lambda client, inactive_shards: _stub_ingest(),
            "resolve_held_ladders":
                lambda client, markets, held, *, labels_out=None: frozenset(),
            "filter_markets_within_horizon": lambda markets, days: markets,
            "find_time_series_pairs": lambda *a, **k: [],
            "find_same_title_pairs":
                lambda markets, held, *, add_on_pairs=None: [spec.pair],
            "enrich_with_orderbook_prices":
                lambda client, pairs, value, *, settings, cash_cents: pairs,
            "compute_trade": lambda pair, value, *, settings, cash_cents: spec,
            "select_portfolio": lambda specs, cash, *, held_ladders: specs,
            "pre_execution_check": lambda client, portfolio, *, settings: portfolio,
            "ensure_shard_collateral":
                lambda client, portfolio, balances, statuses, *, dry_run: portfolio,
            "execute_trades": lambda client, portfolio, *, dry_run: [
                TradeResult(spec=s, status=status) for s in portfolio],
            "append_to_prod_log": lambda *a, **k: logged.append(a) or pathlib.Path("/fake"),
        }
        patches.update(stubs)
        argv = ["--mode", "prod", "--result-file", str(path)] + (["--dry-run"] if dry_run else [])
        with caplog.at_level(logging.INFO):
            seen = _main_with(monkeypatch, argv, **patches)
        assert _no_report_handler()
        seen["trade_log_calls"] = logged
        return seen, _read_result(path), list(caplog.records)

    @staticmethod
    def _logged_once(records, message: str, level: int) -> None:
        """
        Assert the result's message is exactly one line the run logged, at that level.

        Args:
            records (list[logging.LogRecord]): The run's log records.
            message (str): The result's message.
            level (int): The level it must have been logged at.
        """
        matches = [r for r in records if r.getMessage() == message]
        assert len(matches) == 1, message
        assert matches[0].levelno == level

    @staticmethod
    def _trade(status: str, error=None) -> dict:
        """
        The run result's record of make_spec()'s pair with this outcome.

        Args:
            status (str): The trader's status.
            error (str | None): The trader's error.

        Returns:
            dict: The trade as the result file holds it.
        """
        return {"status": status, "error": error, **_MAKE_SPEC_TRADE}

    def test_the_record_names_the_run(self, monkeypatch, tmp_path, caplog):
        seen, result, _ = self._run(monkeypatch, tmp_path, caplog)
        saved = config.read_saved_live_defaults()
        assert set(result) == {
            "format", "mode", "dry_run", "started_at", "finished_at", "exit_code",
            "settings", "defaults", "message", "balance_before", "balance_after",
            "portfolio_value_before", "submission_started", "trades", "warnings",
            "warnings_dropped", "error"}
        assert result["format"] == config.LIVE_RUN_RESULT_FORMAT
        assert result["mode"] == "prod" and result["dry_run"] is False
        assert result["settings"] == describe_live_settings(saved, saved)
        assert result["defaults"] == saved.origin
        for key in ("started_at", "finished_at"):
            assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", result[key]), result[key]
        assert result["started_at"] <= result["finished_at"]
        assert result["error"] is None
        # The portfolio value the run sized on: its cash, since it holds nothing
        assert result["portfolio_value_before"] == self._BEFORE / 100
        # The result's exit code is the one main() exited with
        assert seen["code"] == result["exit_code"] == EXIT_OK

    def test_low_balance(self, monkeypatch, tmp_path, caplog):
        seen, result, records = self._run(
            monkeypatch, tmp_path, caplog,
            read_account_balance=lambda client: _account(MIN_BALANCE_CENTS - 1))
        assert seen["code"] == result["exit_code"] == EXIT_SKIPPED_LOW_BALANCE
        assert result["message"] == (
            f"Portfolio value ${(MIN_BALANCE_CENTS - 1) / 100:.2f} (cash "
            f"${(MIN_BALANCE_CENTS - 1) / 100:.2f}) is below minimum "
            f"${MIN_BALANCE_CENTS / 100:.2f} — skipping run.")
        self._logged_once(records, result["message"], logging.WARNING)
        assert result["trades"] == []
        assert result["balance_before"] == (MIN_BALANCE_CENTS - 1) / 100
        assert result["portfolio_value_before"] == (MIN_BALANCE_CENTS - 1) / 100
        assert result["balance_after"] is None
        assert result["submission_started"] is False
        assert f"WARNING: {result['message']}" in result["warnings"]

    def test_the_portfolio_value_names_the_open_positions(self, monkeypatch, tmp_path, caplog):
        # With positions held, the result names the cash and, beside it, the
        # cash plus Kalshi's value of the positions: what Kelly was sized on.
        # The 250 contracts held back the $250.00 (a contract pays at most $1)
        reads = iter([_account(self._BEFORE, 25_000), _account(self._AFTER, 26_000)])
        seen, result, _ = self._run(monkeypatch, tmp_path, caplog,
                                    read_account_balance=lambda client: next(reads),
                                    get_held_positions=_held_positions("HELD-X", count=250.0))
        assert seen["code"] == result["exit_code"] == EXIT_OK
        assert result["balance_before"] == self._BEFORE / 100
        assert result["portfolio_value_before"] == (self._BEFORE + 25_000) / 100
        # The balance after trading is the cash alone
        assert result["balance_after"] == self._AFTER / 100

    def test_blind_run(self, monkeypatch, tmp_path, caplog):
        seen, result, records = self._run(
            monkeypatch, tmp_path, caplog,
            fetch_open_events_with_markets=lambda client, inactive_shards: [])
        assert seen["code"] == result["exit_code"] == EXIT_NO_TRADEABLE_SHARDS
        assert result["message"] == main._blind_run_reason([], None, set())
        self._logged_once(records, result["message"], logging.WARNING)
        assert result["trades"] == []
        assert result["balance_before"] == self._BEFORE / 100
        assert result["balance_after"] is None

    def test_no_pairs(self, monkeypatch, tmp_path, caplog):
        seen, result, records = self._run(
            monkeypatch, tmp_path, caplog,
            find_same_title_pairs=lambda markets, held, *, add_on_pairs=None: [])
        assert seen["code"] == result["exit_code"] == EXIT_OK
        assert result["message"].startswith("No qualifying pairs found (")
        self._logged_once(records, result["message"], logging.INFO)
        assert result["trades"] == []
        assert result["balance_before"] == self._BEFORE / 100
        assert result["balance_after"] is None

    def test_empty_portfolio(self, monkeypatch, tmp_path, caplog):
        seen, result, records = self._run(
            monkeypatch, tmp_path, caplog,
            select_portfolio=lambda specs, cash, *, held_ladders: [])
        assert seen["code"] == result["exit_code"] == EXIT_OK
        assert result["message"] == "No executable trades found."
        self._logged_once(records, result["message"], logging.INFO)
        assert result["trades"] == [] and result["balance_after"] is None

    def test_failed_pre_execution_check(self, monkeypatch, tmp_path, caplog):
        seen, result, records = self._run(
            monkeypatch, tmp_path, caplog,
            pre_execution_check=lambda client, portfolio, *, settings: [])
        assert seen["code"] == result["exit_code"] == EXIT_OK
        assert result["message"] == (
            "All selected pairs failed pre-execution price check — no trades submitted.")
        self._logged_once(records, result["message"], logging.INFO)
        assert result["trades"] == [] and result["balance_after"] is None

    def test_collateral(self, monkeypatch, tmp_path, caplog):
        seen, result, records = self._run(
            monkeypatch, tmp_path, caplog,
            ensure_shard_collateral=lambda client, portfolio, balances, statuses, *, dry_run: [])
        assert seen["code"] == result["exit_code"] == EXIT_OK
        assert result["message"] == (
            "No selected pair could be funded on its exchange shard — no trades submitted.")
        self._logged_once(records, result["message"], logging.INFO)
        assert result["trades"] == [] and result["balance_after"] is None
        # Stopped before any order was sent
        assert result["submission_started"] is False

    def test_dry_run(self, monkeypatch, tmp_path, caplog):
        seen, result, records = self._run(
            monkeypatch, tmp_path, caplog, dry_run=True, status="simulated")
        assert seen["code"] == result["exit_code"] == EXIT_OK
        assert result["dry_run"] is True
        # A dry run sends no orders, so it never records starting to
        assert result["submission_started"] is False
        assert result["message"] == "[DRY RUN] No orders were actually submitted."
        self._logged_once(records, result["message"], logging.INFO)
        assert result["trades"] == [self._trade("simulated")]
        assert result["balance_before"] == self._BEFORE / 100
        assert result["balance_after"] == self._AFTER / 100
        # The trade log is still written
        assert len(seen["trade_log_calls"]) == 1

    def test_executed(self, monkeypatch, tmp_path, caplog):
        seen, result, records = self._run(monkeypatch, tmp_path, caplog)
        assert seen["code"] == result["exit_code"] == EXIT_OK
        assert result["message"] == (
            "Submitted 1 of 1 order pair(s) successfully. 0 rolled back, "
            "0 rollback failure(s), 0 unknown fill state(s).")
        self._logged_once(records, result["message"], logging.INFO)
        assert result["trades"] == [self._trade("executed")]
        assert result["balance_before"] == self._BEFORE / 100
        assert result["balance_after"] == self._AFTER / 100
        assert result["submission_started"] is True
        # A real-money run's first WARNING, copied in as it was logged
        assert result["warnings"][0] == (
            "WARNING: Running in PRODUCTION mode — real money will be used!")
        assert result["warnings_dropped"] == 0

    def test_needs_attention(self, monkeypatch, tmp_path, caplog):
        seen, result, records = self._run(monkeypatch, tmp_path, caplog, status="manual_review")
        assert seen["code"] == result["exit_code"] == EXIT_TRADES_NEED_ATTENTION
        assert result["message"] == (
            "Submitted 0 of 1 order pair(s) successfully. 0 rolled back, "
            "0 rollback failure(s), 1 unknown fill state(s).")
        self._logged_once(records, result["message"], logging.INFO)
        assert result["trades"] == [self._trade("manual_review")]
        assert result["balance_after"] == self._AFTER / 100
        assert ("CRITICAL: 0 pair(s) may have ORPHANED positions and 1 pair(s) have an "
                "UNDETERMINED fill state — manual review required (see trade log)."
                ) in result["warnings"]

    def test_time_series_skipped(self, monkeypatch, tmp_path, caplog):
        seen, result, records = self._run(
            monkeypatch, tmp_path, caplog,
            resolve_held_ladders=lambda client, markets, held, *, labels_out=None: None)
        assert seen["code"] == result["exit_code"] == EXIT_TIME_SERIES_SKIPPED
        assert result["message"] == (
            "Submitted 1 of 1 order pair(s) successfully. 0 rolled back, "
            "0 rollback failure(s), 0 unknown fill state(s).")
        self._logged_once(records, result["message"], logging.INFO)
        assert result["trades"] == [self._trade("executed")]
        assert result["balance_after"] == self._AFTER / 100

    def test_a_failed_balance_read_after_trading_is_not_known(self, monkeypatch, tmp_path,
                                                              caplog):
        reads = []

        def read_account_balance(client):
            """
            Read the balance once, then fail as a lost connection would.

            Args:
                client: The run's client.

            Returns:
                AccountBalance: The balance before trading, on the first read.

            Raises:
                ConnectionError: On every later read.
            """
            reads.append(1)
            if len(reads) > 1:
                raise ConnectionError("connection reset")
            # Open positions beside the cash, so a fallback to the portfolio
            # value rather than the cash would show in the trade log
            return _account(self._BEFORE, 25_000)

        # The 250 contracts held back the $250.00 of positions, so the run
        # sizes on the cash plus them, and the two figures differ
        seen, result, _ = self._run(monkeypatch, tmp_path, caplog,
                                    read_account_balance=read_account_balance,
                                    get_held_positions=_held_positions("HELD-X", count=250.0))
        assert seen["code"] == result["exit_code"] == EXIT_OK
        assert result["portfolio_value_before"] == (self._BEFORE + 25_000) / 100
        # The trade log falls back to the cash before trading (never the
        # portfolio value); the result does not claim a balance it never read
        assert seen["trade_log_calls"][0][1:] == (self._BEFORE / 100, self._BEFORE / 100)
        assert result["balance_before"] == self._BEFORE / 100
        assert result["balance_after"] is None
        assert any(w.startswith("ERROR: Post-trade balance fetch failed")
                   for w in result["warnings"])

    @pytest.mark.parametrize("error, text", [
        (RuntimeError("run failed\nsecond line"), "RuntimeError: run failed"),
        (KeyboardInterrupt(), "KeyboardInterrupt"),
    ], ids=["error", "ctrl-c"])
    def test_a_run_that_raises_writes_no_exit_code_and_one_line(self, monkeypatch, tmp_path,
                                                                error, text):
        path = tmp_path / "result.json"

        def failing_run(client, args, settings, reference, report=None):
            """
            Stop the way an unhandled error or a Ctrl-C in the run mode would.

            Args:
                client: The client main() built.
                args (argparse.Namespace): The parsed flags.
                settings (LiveSettings): The run's settings.
                reference (LiveSettings): The saved defaults they were built from.
                report (reporter.RunReport | None): The run report main() handed over.

            Raises:
                BaseException: Always, the test's error.
            """
            assert report is not None
            report.balance_before = 12.5
            raise error

        with pytest.raises(type(error)):
            _main_with(monkeypatch, ["--mode", "prod", "--result-file", str(path)],
                       _run_prod=failing_run)
        result = _read_result(path)
        assert result["exit_code"] is None
        assert result["error"] == text
        assert result["balance_before"] == 12.5
        assert _no_report_handler()
        assert run_lock.held() is False

    @pytest.mark.parametrize("error, text", [
        (KeyboardInterrupt(), "KeyboardInterrupt"),
        (RuntimeError("worker crashed"), "RuntimeError: worker crashed"),
    ], ids=["ctrl-c", "error"])
    def test_a_run_stopped_while_sending_says_orders_may_have_been_placed(
        self, monkeypatch, tmp_path, caplog, error, text,
    ):
        def stopped_while_sending(client, portfolio, *, dry_run):
            """
            Stop the way a Ctrl-C or a crash inside execute_trades would.

            Args:
                client: The run's client.
                portfolio (list): The selected specs.
                dry_run (bool): False here.

            Raises:
                BaseException: Always, the test's error.
            """
            raise error

        path = tmp_path / "result.json"
        with pytest.raises(type(error)):
            self._run(monkeypatch, tmp_path, caplog, execute_trades=stopped_while_sending)
        result = _read_result(path)
        assert result["exit_code"] is None and result["error"] == text
        # No trade is recorded, yet the record says sending had begun
        assert result["trades"] == [] and result["message"] == ""
        assert result["submission_started"] is True
        assert run_lock.held() is False

    def test_a_dry_run_stopped_while_simulating_records_no_sending(
        self, monkeypatch, tmp_path, caplog,
    ):
        def stopped(client, portfolio, *, dry_run):
            """
            Stop inside execute_trades on a dry run.

            Args:
                client: The run's client.
                portfolio (list): The selected specs.
                dry_run (bool): True here.

            Raises:
                KeyboardInterrupt: Always.
            """
            raise KeyboardInterrupt

        path = tmp_path / "result.json"
        with pytest.raises(KeyboardInterrupt):
            self._run(monkeypatch, tmp_path, caplog, dry_run=True, execute_trades=stopped)
        assert _read_result(path)["submission_started"] is False

    @pytest.mark.parametrize("argv, remove_saved, reason", [
        (["--size-cap", "7"], False, "invalid live setting for this run (--size-cap)"),
        ([], True, "live_defaults.json"),
        (["--max-horizon-days", "0"], False, "--max-horizon-days must be a positive integer"),
    ], ids=["bad-flag", "no-saved-defaults", "bad-horizon"])
    def test_a_usage_error_leaves_no_file_not_an_older_one(
        self, monkeypatch, tmp_path, capsys, argv, remove_saved, reason,
    ):
        # Exit 2 before logging is set up writes no result; an older result
        # already at the path is removed first, so it cannot pass for this run's
        path = tmp_path / "result.json"
        path.write_text('{"exit_code": 0, "trades": []}', encoding="utf-8")
        if remove_saved:
            config.LIVE_DEFAULTS_FILE.unlink()
        seen = _main_with(monkeypatch, ["--mode", "prod", "--result-file", str(path), *argv])
        assert seen["code"] == 2
        assert seen["logging_set_up"] is False and "mode" not in seen
        assert reason in capsys.readouterr().err
        assert not path.exists()
        assert _no_report_handler()

    def test_an_older_result_is_replaced_by_this_runs(self, monkeypatch, tmp_path):
        path = tmp_path / "result.json"
        path.write_text('{"exit_code": 20, "trades": ["old"]}', encoding="utf-8")
        seen = _main_with(monkeypatch, ["--mode", "prod", "--dry-run", "--result-file", str(path)])
        result = _read_result(path)
        assert seen["code"] == result["exit_code"] == EXIT_OK
        assert result["trades"] == [] and result["dry_run"] is True

    def test_a_folder_at_the_path_is_not_removed_and_the_run_goes_on(
        self, monkeypatch, tmp_path, caplog,
    ):
        # What cannot be removed is left to the final write, which logs why
        path = tmp_path / "result.json"
        path.mkdir()
        with caplog.at_level(logging.INFO):
            seen = _main_with(
                monkeypatch, ["--mode", "prod", "--dry-run", "--result-file", str(path)])
        assert seen["code"] == EXIT_OK
        assert path.is_dir()
        assert any(r.levelno == logging.ERROR
                   and r.getMessage().startswith(f"Could not write the run result to {path}")
                   for r in caplog.records)

    def test_the_result_is_written_before_the_lock_is_released(self, monkeypatch, tmp_path):
        path = tmp_path / "result.json"
        at_write = []

        def spy(where, report, exit_code):
            """
            Record the lock and the handler as the result is written, then write it.

            Args:
                where (pathlib.Path): The result file.
                report (reporter.RunReport): The report.
                exit_code (int | None): The run's exit code.
            """
            at_write.append((run_lock.held(), _no_report_handler(), exit_code))
            reporter.write_run_report(where, report, exit_code)

        seen = _main_with(monkeypatch, ["--mode", "prod", "--result-file", str(path)],
                          write_run_report=spy)
        assert seen["code"] == EXIT_OK
        # Written once, while this run still held the lock and the handler was attached
        assert at_write == [(True, False, EXIT_OK)]
        assert run_lock.held() is False and _no_report_handler()
        assert _read_result(path)["exit_code"] == EXIT_OK

    def test_a_trade_log_that_cannot_be_written_still_leaves_a_result(
        self, monkeypatch, tmp_path, caplog,
    ):
        def failing_log(*args, **kwargs):
            """
            Fail the way a trade log open in another program would.

            Args:
                *args: append_to_prod_log's positional arguments.
                **kwargs: Its keyword arguments.

            Raises:
                PermissionError: Always.
            """
            raise PermissionError("trade_log.xlsx is open elsewhere")

        path = tmp_path / "result.json"
        with pytest.raises(PermissionError):
            self._run(monkeypatch, tmp_path, caplog, append_to_prod_log=failing_log)
        result = _read_result(path)
        # The trades were recorded before the trade log, and the rescue dump ran
        assert result["exit_code"] is None
        assert result["error"] == "PermissionError: trade_log.xlsx is open elsewhere"
        assert result["trades"] == [self._trade("executed")]
        assert result["balance_after"] == self._AFTER / 100
        assert result["message"] == ""
        assert any(w.startswith("CRITICAL:   RESCUE | executed | Test pair")
                   for w in result["warnings"])
        assert _no_report_handler()

    def test_the_rescue_dump_names_an_add_ons_held_count(self, monkeypatch, tmp_path, caplog):
        """Pins that the trade log's rescue dump marks a trade that added to a
        held pair "(adds to 30 held)" right after its counts, so x=5 y=5 is
        never read as the whole position, and leaves an ordinary line as it was."""
        add_on = make_spec()
        add_on.pair.held = scanner_mod.HeldPair(
            sides=(("TICK-A", "yes"), ("TICK-B", "no")), count=30.0,
            cost_dollars=18.9, value_dollars=18.0, fees_dollars=0.9)

        def executed(client, portfolio, *, dry_run):
            """
            Stand in for trader.execute_trades: the run's pair, then an add-on, both executed.

            Args:
                client: The run's client (unused).
                portfolio (list): The run's selected specs.
                dry_run (bool): Whether the run is a dry run (unused).

            Returns:
                list[TradeResult]: One executed result per spec.
            """
            return [TradeResult(spec=s, status="executed") for s in (*portfolio, add_on)]

        def failing_log(*args, **kwargs):
            """
            Fail the way a trade log open in another program would.

            Args:
                *args: append_to_prod_log's positional arguments.
                **kwargs: Its keyword arguments.

            Raises:
                PermissionError: Always.
            """
            raise PermissionError("trade_log.xlsx is open elsewhere")

        with pytest.raises(PermissionError):
            self._run(monkeypatch, tmp_path, caplog, execute_trades=executed,
                      append_to_prod_log=failing_log)
        rescue = [r.getMessage() for r in caplog.records if "RESCUE |" in r.getMessage()]
        assert rescue == [
            "  RESCUE | executed | Test pair | A=TICK-A B=TICK-B | x=5 y=5 cost=$2.30 incl."
            " fees | ",
            "  RESCUE | executed | Test pair | A=TICK-A B=TICK-B | x=5 y=5 (adds to 30 held)"
            " cost=$2.30 incl. fees | ",
        ]

    def test_a_pair_that_cannot_be_described_still_reaches_the_trade_log(
        self, monkeypatch, tmp_path, caplog,
    ):
        def fail(result):
            """
            Stand in for reporter.trade_record, failing on every pair.

            Args:
                result (TradeResult): A pair's outcome.

            Raises:
                KeyError: Always.
            """
            raise KeyError("no such field")

        first, second = make_spec(), make_spec()
        second.pair.market_a.ticker, second.pair.market_b.ticker = "TICK-C", "TICK-D"
        monkeypatch.setattr(reporter, "trade_record", fail)
        seen, result, records = self._run(
            monkeypatch, tmp_path, caplog,
            execute_trades=lambda client, portfolio, *, dry_run: [
                TradeResult(spec=first, status="manual_review", error="lookup failed"),
                TradeResult(spec=second, status="executed")])
        # The run went on: the trade log was written, and the exit code is the run's
        assert len(seen["trade_log_calls"]) == 1
        assert [r.status for r in seen["trade_log_calls"][0][0]] == ["manual_review",
                                                                     "executed"]
        assert seen["code"] == result["exit_code"] == EXIT_TRADES_NEED_ATTENTION
        # Each pair is still listed, with what could be read of it
        assert [(t["status"], t["error"], t["a"]["ticker"], t["b"]["ticker"],
                 t["a"]["count"], t["b"]["count"]) for t in result["trades"]] == [
            ("manual_review", "lookup failed", "TICK-A", "TICK-B", 5, 5),
            ("executed", None, "TICK-C", "TICK-D", 5, 5)]
        assert all(t["cost_with_fees"] is None and t["a"]["price"] is None
                   for t in result["trades"])
        errors = [r for r in records if r.levelno == logging.ERROR
                  and r.getMessage().startswith("Could not describe a pair for the run result")]
        assert len(errors) == 2

    def test_a_lock_refusal_writes_50_and_its_message(self, monkeypatch, tmp_path, caplog):
        path = tmp_path / "result.json"
        holder = run_lock.acquire()
        try:
            with caplog.at_level(logging.INFO):
                seen = _main_with(monkeypatch, ["--mode", "prod", "--result-file", str(path)])
        finally:
            os.close(holder)
        result = _read_result(path)
        assert seen["code"] == result["exit_code"] == EXIT_RUN_IN_PROGRESS
        assert "mode" not in seen and seen["client_built"] is False
        assert result["message"].startswith("Another live trading run is in progress (process ")
        assert result["message"].endswith(
            f"— this run stops without contacting Kalshi, so nothing is sent "
            f"(exit {EXIT_RUN_IN_PROGRESS}).")
        self._logged_once(caplog.records, result["message"], logging.WARNING)
        assert result["warnings"] == [f"WARNING: {result['message']}"]
        assert result["trades"] == [] and result["balance_before"] is None
        assert _no_report_handler()

    def test_the_flag_is_refused_in_dev(self, monkeypatch, tmp_path, capsys):
        path = tmp_path / "result.json"
        seen = _main_with(monkeypatch, ["--mode", "dev", "--result-file", str(path)])
        assert seen["code"] == 2
        assert "--result-file is for --mode prod only" in capsys.readouterr().err
        # Refused before logging, the client or the run mode
        assert seen["logging_set_up"] is False and "mode" not in seen
        assert not path.exists()

    def test_the_dev_refusal_leaves_a_file_at_the_path_alone(self, monkeypatch, tmp_path):
        path = tmp_path / "result.json"
        path.write_text("not this run's", encoding="utf-8")
        seen = _main_with(monkeypatch, ["--mode", "dev", "--result-file", str(path)])
        assert seen["code"] == 2
        assert path.read_text(encoding="utf-8") == "not this run's"

    def test_without_the_flag_nothing_is_written(self, monkeypatch):
        def no_write(path, report, exit_code):
            """
            Stand in for write_run_report, which a run without --result-file never calls.

            Args:
                path (pathlib.Path): The result file.
                report (reporter.RunReport): The report.
                exit_code (int | None): The exit code.

            Raises:
                AssertionError: Always.
            """
            raise AssertionError("a run without --result-file wrote a result")

        for argv in (["--mode", "prod"], ["--mode", "prod", "--dry-run"]):
            seen = _main_with(monkeypatch, argv, write_run_report=no_write)
            assert seen["code"] == EXIT_OK
            assert seen["mode"] == "prod" and seen["report"] is None
            assert _no_report_handler()

    def test_the_report_reaches_the_run_mode_and_its_warnings_stop_at_the_end(
        self, monkeypatch, tmp_path,
    ):
        path = tmp_path / "result.json"
        seen = _main_with(monkeypatch, ["--mode", "prod", "--dry-run", "--result-file", str(path)])
        report = seen["report"]
        assert isinstance(report, main.RunReport) and report.dry_run is True
        assert _no_report_handler()
        # A line logged after main() returns is not the run's
        logging.getLogger().warning("after the run")
        assert "WARNING: after the run" not in report.warnings
        assert _read_result(path)["exit_code"] == EXIT_OK


def _other_toggles(target: LiveSettings) -> LiveSettings:
    """
    Build live defaults that differ from target in every one of the eight toggles.

    Args:
        target (LiveSettings): The settings a run should trade.

    Returns:
        LiveSettings: Saved defaults under which every one of target's flags
            has to take effect for the run to trade target.
    """
    band = (0.2, 0.3) if target.spread_band != (0.2, 0.3) else (0.25, 0.4)
    other = LiveSettings(
        tier_floors=not target.tier_floors,
        spread_band=band,
        interval_discount=0.55 if target.interval_discount != 0.55 else 0.65,
        size_cap=0.5 if target.size_cap != 0.5 else 0.6,
        same_title_size_cap=0.25 if target.same_title_size_cap != 0.25 else 0.3,
        categories=None if target.categories is not None else ("Saved",),
        tags=None if target.tags is not None else ("Saved",),
        add_to_held_pairs=not target.add_to_held_pairs,
    )
    for name in config.LIVE_TOGGLE_FIELDS:
        assert getattr(other, name) != getattr(target, name), name
    assert other.spread_band[0] != target.spread_band[0]
    assert other.spread_band[1] != target.spread_band[1]
    return other


# Settings config.live_settings_argv must spell so a run trades them exactly
_ARGV_TARGETS = {
    "seed": config.LIVE_DEFAULTS_SEED,
    "k-with-many-decimals-tier-floors-on": LiveSettings(
        tier_floors=True, spread_band=(0.15, 0.85), interval_discount=0.7123456789012345,
        size_cap=0.05, same_title_size_cap=1.0, categories=("Economics",), tags=None,
        add_to_held_pairs=True),
    "tier-floors-off-caps-100-and-5": LiveSettings(
        tier_floors=False, spread_band=(0.0, 1.0), interval_discount=1.0,
        size_cap=1.0, same_title_size_cap=0.05,
        categories=("Economics", "Sports", "Climate and Weather"), tags=("Soccer",)),
    "names-starting-with-a-dash": LiveSettings(
        tier_floors=True, spread_band=(0.1234567, 0.6543210987654321),
        interval_discount=1 / 3, size_cap=0.35, same_title_size_cap=0.45,
        categories=("-Minus", "Politics"), tags=("-Dash", "Basketball", "Pro Football")),
    "several-tags-any-category": LiveSettings(
        tier_floors=False, spread_band=(0.05, 0.5), interval_discount=0.8,
        size_cap=0.2, same_title_size_cap=0.2, categories=None,
        tags=("Soccer", "Basketball", "--double dash")),
}


class TestLiveSettingsArgv:
    """
    config.live_settings_argv spells settings as main.py's toggle flags so a
    run trades exactly them: parsed by main._build_parser and laid over saved
    defaults that differ in every toggle by main._resolve_live_settings, they
    give back the same settings.
    """

    @pytest.mark.parametrize("target", _ARGV_TARGETS.values(), ids=_ARGV_TARGETS.keys())
    def test_the_flags_round_trip_over_different_saved_defaults(self, target):
        saved = config.save_live_defaults(_other_toggles(target), source="")
        parser = main._build_parser()
        args = parser.parse_args(["--mode", "prod", *config.live_settings_argv(target)])
        settings, reference = main._resolve_live_settings(args, parser)
        assert settings == target
        for name in config.LIVE_TOGGLE_FIELDS:
            assert getattr(settings, name) == getattr(target, name), name
        # The reference is still the saved file, which the run names
        assert reference == saved and settings.origin == saved.origin
        # Every toggle was given as a flag
        assert args.tier_floors is target.tier_floors
        assert args.spread_min is not None and args.spread_max is not None
        assert args.interval_discount is not None
        assert args.size_cap is not None and args.same_title_size_cap is not None
        assert (args.category is not None) or args.any_category
        assert (args.tag is not None) or args.any_tag
        assert args.add_to_held_pairs is target.add_to_held_pairs

    def test_the_seed_is_spelled_flag_by_flag(self):
        assert config.live_settings_argv(config.LIVE_DEFAULTS_SEED) == [
            "--no-tier-floors", "--spread-min=0.0", "--spread-max=0.5",
            "--interval-discount=0.8", "--size-cap=10", "--same-title-size-cap=20",
            "--add-to-held-pairs", "--any-category", "--any-tag"]

    def test_a_name_that_begins_with_a_dash_is_read_as_a_name(self):
        target = _ARGV_TARGETS["names-starting-with-a-dash"]
        argv = config.live_settings_argv(target)
        assert "--category=-Minus" in argv and "--tag=-Dash" in argv
        args = main._build_parser().parse_args(["--mode", "prod", *argv])
        assert args.category == ["-Minus", "Politics"]
        assert args.tag == ["-Dash", "Basketball", "Pro Football"]

    def test_main_trades_the_spelled_settings(self, monkeypatch):
        target = _ARGV_TARGETS["k-with-many-decimals-tier-floors-on"]
        saved = config.save_live_defaults(_other_toggles(target), source="")
        seen = _main_with(monkeypatch, ["--mode", "prod", "--dry-run",
                                        *config.live_settings_argv(target)])
        assert seen["code"] == EXIT_OK
        assert seen["settings"] == target and seen["reference"] == saved


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

    @pytest.mark.usefixtures("saved_live_defaults")
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


@pytest.mark.usefixtures("saved_live_defaults")
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
                "kalshi_betting.main.read_account_balance",
                return_value=_account(MIN_BALANCE_CENTS - 1),
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


@pytest.mark.usefixtures("saved_live_defaults")
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
        # It says what prod sizes on instead: the account's own portfolio value
        assert ("prod sizes on the account's portfolio value (its cash plus its open "
                "positions' value) and spends only its cash") in text

    def test_silent_when_not_passed_in_prod(self, caplog):
        text = self._run(["main", "--mode", "prod"], caplog)
        assert "--sandbox-balance" not in text

    def test_silent_in_dev(self, caplog):
        text = self._run(
            ["main", "--mode", "dev", "--sandbox-balance", "50"], caplog)
        assert "--sandbox-balance is inert" not in text
