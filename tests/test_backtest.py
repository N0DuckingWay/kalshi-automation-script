"""Tests for backtest.py — the backtester's CLI argument surface.

Covers the two interval-discount flags added with the calibration sweep:
argparse accepts a valid --interval-discount, main() rejects one at or below
0 or above 1 through parser.error(), and both --interval-discount and --no-sweep are
threaded into backtester.run_backtest_sweep() (the same threading-assertion
idiom test_backtester.py uses for --max-horizon-days).

Also covers the PB4 spread-band flags: --spread-min/--spread-max resolve to
None (both omitted) or a validated (floor, ceiling) tuple whose omitted side
comes from config.BACKTEST_DEFAULT_SPREAD_BAND at call time; an out-of-range
value or a floor-not-less-than-ceiling band (either flag alone included) is
rejected through parser.error() before logging is configured (TS-20 — both
the order and the absent kalshi_backtest.log are pinned); the pre-fetch echo
line names the resolved band and band-sweep setting; a ceiling at or below a
deadline-gap tier gets a WARNING naming the tier and the primary scenario;
and --no-band-sweep threads band_sweep=False — and with it
tier_off_sweep=False, since the tier-floors-off runs have no flag of their
own and ride the band sweep — into run_backtest_sweep().

And --no-cap-sweep: the per-trade size-cap sweep is on by default
(cap_sweep=True), the flag threads cap_sweep=False, and the echo line names
the setting.

And --no-sell-sweep: the dashboard's Sell family is on by default
(sell_sweep=True), the flag threads sell_sweep=False, and the echo line names
the setting right after the cap sweep's. --no-rebalance-sweep does the same
for the Rebalance family (rebalance_sweep), named after that, and the old
--no-add-on-sweep, --no-trim-sweep and --no-trim-rise-sweep are refused.
--no-move-sweep leaves that family on without its minimum price movements
(move_sweep=False), and changes only the echo's rebalance clause.

And the echo's "live rule=" clause: the saved live defaults' rule (with their
origin), "none saved" with no file, "not recorded" with a refused one — a read
of its own, never a scenario of this run and never config.py's toggles. And the
run's last line, after the one pointing at the dashboard: how the filter bar's
scenario becomes the live defaults (the defaults server, then the page's save
button).

And the depth model: main() fits it (depth_model.load_depth_model) before the
fetch, hands it to run_backtest_sweep, and the config echo's last clause says
whether trades fill over a walked book (with its snapshot and ladder counts) or
at the top of the book.

And the starting balance: with no --balance the run starts from the account's
value (its cash on every shard plus Kalshi's value of its open positions, or
the cash alone with a WARNING when that value is unreadable), read once through
the production live client before the fetch; --balance skips the read; a
balance that is not a positive number is refused before logging is
configured; a read that fails, or comes to nothing, stops the run before the
fetch; and a balance below config.MIN_BALANCE_CENTS draws a WARNING. The
amount reaches the sweep, the log, the summary block and the dashboard, and
where it came from reaches the log and the dashboard.

And the saved sweep: every full run saves what it hands generate_dashboard
(config.BACKTEST_SWEEP_FILENAME, a JSON header line with the simulator
fingerprint, then the inputs pickled) right after the sweep and before the
page is built, a save that fails costs only the file; --dashboard-only builds
the page from it with the run's own arguments and no Kalshi request, refuses
the flags that shape a run, a missing file (exit 2), a damaged one (exit 1)
and a file from other simulator code (exit 2, unless
--accept-changed-simulator); its --no-*-sweep flags drop their family as a run
given them builds it; the fingerprint counts code, never comments,
docstrings or config the simulator does not read; and, on the backtester's
selling golden fixture with every lazy family on, the rebuilt page and its
sidecar files are byte for byte the run's.

And the Data page: after the backtest page, a run and a --dashboard-only
rebuild each build it (eda.build_eda_page) from the run's sweep and listing,
with at most config.EDA_MAX_WORKERS workers; --no-eda skips it on both paths;
and one that raises, or writes nothing, is a WARNING that changes neither the
exit code nor the backtest page.

Fully offline: run_backtest_sweep, generate_dashboard, build_eda_page, both client builders,
read_account_balance, load_risk_free_rates and load_depth_model are monkeypatched, so no network
call, no credential read and no real backtest happen. PROJECT_ROOT is redirected at tmp_path and
logging.basicConfig is stubbed, so the run's RotatingFileHandler can neither
write into the repo root nor leak a handler onto the root logger for the rest
of the session.
"""
import ast
import dataclasses
import gzip
import json
import logging
import pickle
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from kalshi_python_sync.exceptions import ApiException

from kalshi_betting import backtest, backtester, config, dashboard
from kalshi_betting.auth import AccountBalance
from kalshi_betting.backtester import (
    BacktestSweep,
    CapSweep,
    CorpusProvenance,
    RebalanceSweep,
    SellSweep,
    SweepPoint,
)
from kalshi_betting.config import (
    MAX_DEADLINE_GAP_DAYS,
    MIN_PRICE_DIFF_LONG_GAP,
    MIN_PRICE_DIFF_SHORT_GAP,
    PRICE_EPSILON,
    SHORT_DEADLINE_GAP_DAYS,
    min_price_diff_for_gap,
    time_series_spread_too_wide,
)
from kalshi_betting.depth_model import DepthModel
from kalshi_betting.treasury import SOURCE_API, RiskFreeRates

from . import test_backtester as _tb
from .conftest import apply_pre_toggle_defaults, save_config_live_defaults


def _equity(final_value: float = 10_691.38) -> pd.DataFrame:
    """A two-row equity curve in _build_equity_curve's shape."""
    df = pd.DataFrame({
        "date": [date(2026, 1, 5), date(2026, 1, 12)],
        "portfolio_value": [10_000.0, final_value],
    })
    df["daily_return"] = df["portfolio_value"].pct_change().fillna(0.0)
    return df


def _sweep(k: float | None = None, n_trades: int = 0) -> BacktestSweep:
    """A BacktestSweep whose primary point carries n_trades profitable trades.

    main() reads only `.profit` off each trade (the win-rate count), so a plain
    SimpleNamespace stands in for BacktestTrade without duplicating its fixture.
    k None reads backtest's binding of TIME_SERIES_INTERVAL_PROB_DISCOUNT at call
    time, as a run given no --interval-discount does, so a patched value holds.
    """
    point = SweepPoint(
        k=backtest.TIME_SERIES_INTERVAL_PROB_DISCOUNT if k is None else k,
        trades=[SimpleNamespace(profit=5.0) for _ in range(n_trades)],
        equity_df=_equity(),
    )
    return BacktestSweep(primary=point, points=[point], calibration=None)


@pytest.fixture
def cli(monkeypatch, tmp_path):
    """Run main() offline and capture what it handed to its collaborators.

    Returns a dict that fills in with "sweep_kwargs" (run_backtest_sweep's
    keyword arguments) and "dashboard" ((args, kwargs)) as main() proceeds, plus
    "result" — the BacktestSweep the stub returned, so a test can assert
    identity rather than equality. "account" is what the account read returns
    (an AccountBalance, or an exception to raise), "balance_reads" the client
    each read was handed, and "live_client" the one production live client
    main() builds.
    """
    calls: dict = {"result": _sweep()}

    def _fake_sweep(**kwargs):
        calls["sweep_kwargs"] = kwargs
        calls["order"].append("run_backtest_sweep")
        return calls["result"]

    def _fake_dashboard(*args, **kwargs):
        calls["dashboard"] = (args, kwargs)
        calls["pages"].append("backtest")
        return tmp_path / "dashboard.html"

    # The Data page, never the real one (it would read the real backtest_cache):
    # each call's keyword arguments are recorded, and "eda_result" is what it
    # returns (an exception to raise, or a path)
    calls["pages"] = []
    calls["eda"] = []
    calls["eda_result"] = tmp_path / config.EDA_DASHBOARD_FILENAME

    def _fake_eda(**kwargs):
        calls["eda"].append(kwargs)
        calls["pages"].append("data")
        if isinstance(calls["eda_result"], BaseException):
            raise calls["eda_result"]
        return calls["eda_result"]

    def _no_basic_config(**kwargs):
        # Never let a test install a handler on the ROOT logger: it would
        # outlive this test and keep writing into a deleted tmp dir. The
        # handler itself is still constructed by main(), so close it here.
        for handler in kwargs.get("handlers", []):
            handler.close()

    # The log file is built as PROJECT_ROOT / "kalshi_backtest.log" and the
    # handler opens it eagerly, so redirect the root before main() runs.
    monkeypatch.setattr(backtest, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(backtest.logging, "basicConfig", _no_basic_config)
    monkeypatch.setattr(backtest, "build_historical_client", lambda: MagicMock())
    calls["live_client"] = MagicMock()
    monkeypatch.setattr(backtest, "build_prod_live_client", lambda: calls["live_client"])
    # The account read a run given no --balance makes, never a real one: $116.15
    # of cash and $95.27 of open positions, so such a run starts from $211.42.
    # Every client it is handed is recorded
    calls["account"] = AccountBalance({0: 11_615, 1: 0}, 9_527)
    calls["balance_reads"] = []

    def _fake_balance_read(client):
        calls["balance_reads"].append(client)
        if isinstance(calls["account"], Exception):
            raise calls["account"]
        return calls["account"]

    monkeypatch.setattr(backtest, "read_account_balance", _fake_balance_read)
    monkeypatch.setattr(backtest, "run_backtest_sweep", _fake_sweep)
    monkeypatch.setattr(backtest, "generate_dashboard", _fake_dashboard)
    monkeypatch.setattr(backtest, "build_eda_page", _fake_eda)
    # Never read the real backtest_cache or the network for series categories
    calls["series_categories"] = {"KXTEST": ("Sports", ("Basketball",))}
    monkeypatch.setattr(backtest, "load_series_categories",
                        lambda client: calls["series_categories"])
    # ... and for the series fee multipliers (carried by the sweep, unused)
    calls["fee_multipliers"] = {"KXTEST": 0.5}
    monkeypatch.setattr(backtest, "load_series_fee_multipliers",
                        lambda client, **kwargs: calls["fee_multipliers"])
    # Never read the real backtest_cache or the network for the risk-free rate
    calls["risk_free"] = RiskFreeRates(
        ((date(2026, 1, 5), 0.04),), SOURCE_API, datetime(2026, 9, 27, tzinfo=UTC),
    )
    monkeypatch.setattr(backtest, "load_risk_free_rates", lambda: calls["risk_free"])
    # Never read the real saved order-book snapshots: None (no model) unless a
    # test sets calls["depth_model"]. Each call is recorded, in order, in
    # calls["order"] with the sweep's, so a test can say what came first
    calls["depth_model"] = None
    calls["order"] = []

    def _fake_load_depth_model():
        calls["order"].append("load_depth_model")
        return calls["depth_model"]

    monkeypatch.setattr(backtest, "load_depth_model", _fake_load_depth_model)
    return calls


def _run(monkeypatch, *argv: str) -> None:
    """Invoke main() with the given CLI arguments."""
    monkeypatch.setattr(sys, "argv", ["backtest", *argv])
    backtest.main()


def _option_help(monkeypatch, capsys, option: str, next_option: str) -> str:
    """
    Return one option's --help text, its whitespace collapsed.

    The terminal is made wide first, so argparse never splits a flag's name
    at a hyphen when it wraps.

    Args:
        option (str): The option whose help is read.
        next_option (str): The option listed right after it, where its help ends.

    Returns:
        str: The text between the two option names in the options list.
    """
    monkeypatch.setenv("COLUMNS", "1000")
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, "--help")
    assert exc.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    # Skip the usage line, which names every option too
    listed = text.split("options:", 1)[1]
    return listed.split(f" {option} ", 1)[1].split(f" {next_option} ", 1)[0]


class TestIntervalDiscountArgument:
    """--interval-discount is optional, range-checked, and passed through."""

    def test_default_is_no_override(self, cli, monkeypatch):
        _run(monkeypatch)
        # None is the "no override" sentinel config.time_series_profit_prob
        # resolves to TIME_SERIES_INTERVAL_PROB_DISCOUNT at call time — the CLI
        # must not pre-resolve it, or a monkeypatched constant would be ignored.
        assert cli["sweep_kwargs"]["interval_discount"] is None

    def test_valid_value_threads_through(self, cli, monkeypatch):
        _run(monkeypatch, "--interval-discount", "0.62")
        assert cli["sweep_kwargs"]["interval_discount"] == pytest.approx(0.62)

    @pytest.mark.parametrize("value", ["0.01", "1.0", "0.5"])
    def test_boundary_values_are_accepted(self, cli, monkeypatch, value):
        _run(monkeypatch, "--interval-discount", value)
        assert cli["sweep_kwargs"]["interval_discount"] == pytest.approx(float(value))

    # 0 included: the backtest sizes through the live sizer, which takes a k above 0
    @pytest.mark.parametrize("value", ["1.5", "-0.1", "42", "0", "0.0", "nan"])
    def test_out_of_range_value_errors(self, cli, monkeypatch, capsys, value):
        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, "--interval-discount", value)
        assert exc.value.code == 2
        assert "--interval-discount must be above 0 and at most 1" in capsys.readouterr().err
        # parser.error() aborts before any client is built or any run starts
        assert "sweep_kwargs" not in cli

    def test_non_numeric_value_errors(self, cli, monkeypatch):
        with pytest.raises(SystemExit):
            _run(monkeypatch, "--interval-discount", "three-quarters")
        assert "sweep_kwargs" not in cli


class TestNoSweepArgument:
    """--no-sweep is the escape hatch for a full-history run."""

    def test_sweep_is_on_by_default(self, cli, monkeypatch):
        _run(monkeypatch)
        assert cli["sweep_kwargs"]["sweep"] is True

    def test_no_sweep_turns_it_off(self, cli, monkeypatch):
        _run(monkeypatch, "--no-sweep")
        assert cli["sweep_kwargs"]["sweep"] is False

    def test_both_new_flags_thread_together(self, cli, monkeypatch):
        _run(monkeypatch, "--interval-discount", "0.4", "--no-sweep",
             "--max-horizon-days", "14", "--balance", "500")
        kwargs = cli["sweep_kwargs"]
        assert kwargs["interval_discount"] == pytest.approx(0.4)
        assert kwargs["sweep"] is False
        # The pre-existing arguments still arrive unchanged alongside them
        assert kwargs["max_horizon_days"] == 14
        assert kwargs["initial_balance"] == pytest.approx(500.0)
        assert kwargs["use_cache"] is True


class TestScheduledCloseArgument:
    """--scheduled-close / --no-scheduled-close (C5): the backtest reads each
    market's scheduled close, or its realized close, for this run."""

    def test_default_is_no_override(self, cli, monkeypatch):
        # None is resolved by _prepare_candidates at call time
        _run(monkeypatch)
        assert cli["sweep_kwargs"]["scheduled_close"] is None

    @pytest.mark.parametrize("flag, value", [("--scheduled-close", True),
                                             ("--no-scheduled-close", False)])
    def test_the_flags_reach_the_sweep(self, cli, monkeypatch, flag, value):
        _run(monkeypatch, flag)
        assert cli["sweep_kwargs"]["scheduled_close"] is value

    @pytest.mark.parametrize("flag, configured, echo", [
        (None, True, "scheduled close=on | k="),
        (None, False, "scheduled close=off | k="),
        ("--no-scheduled-close", True, "scheduled close=off (config: on) | k="),
        ("--scheduled-close", False, "scheduled close=on (config: off) | k="),
    ])
    def test_the_config_echo_names_it(self, cli, monkeypatch, caplog, flag, configured, echo):
        monkeypatch.setattr(backtest, "BACKTEST_SCHEDULED_CLOSE", configured)
        with caplog.at_level(logging.INFO):
            _run(monkeypatch, *([flag] if flag else []))
        assert f"| cache=on | {echo}" in caplog.text

    @pytest.mark.parametrize("configured, word", [(True, "on"), (False, "off")])
    def test_the_help_text_names_the_configured_value(self, cli, monkeypatch, capsys,
                                                       configured, word):
        monkeypatch.setattr(backtest, "BACKTEST_SCHEDULED_CLOSE", configured)
        with pytest.raises(SystemExit):
            _run(monkeypatch, "--help")
        help_text = " ".join(capsys.readouterr().out.split())
        assert f"config.BACKTEST_SCHEDULED_CLOSE, currently {word})" in help_text


class TestStaleQuoteArgument:
    """--stale-quote-hours N / --no-stale-quote-check (C6): how recently each
    market of an entry, a sale or a Kelly trim must have moved, for this run."""

    def test_default_is_no_override(self, cli, monkeypatch):
        # None is resolved by backtester at call time
        _run(monkeypatch)
        assert cli["sweep_kwargs"]["stale_quote_max_age"] is None

    @pytest.mark.parametrize("flags, value", [(["--stale-quote-hours", "48"], 48 * 3600.0),
                                              (["--stale-quote-hours", "1.5"], 5400.0),
                                              (["--no-stale-quote-check"], False)])
    def test_the_flags_reach_the_sweep(self, cli, monkeypatch, flags, value):
        _run(monkeypatch, *flags)
        assert cli["sweep_kwargs"]["stale_quote_max_age"] == value
        assert type(cli["sweep_kwargs"]["stale_quote_max_age"]) is type(value)

    @pytest.mark.parametrize("hours", ["0", "-1", "nan", "inf"])
    def test_a_bad_number_of_hours_is_a_usage_error(self, cli, monkeypatch, capsys, hours):
        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, "--stale-quote-hours", hours)
        assert exc.value.code == 2
        assert "--stale-quote-hours must be a positive number of hours" in capsys.readouterr().err

    def test_the_two_flags_are_exclusive(self, cli, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, "--stale-quote-hours", "48", "--no-stale-quote-check")
        assert exc.value.code == 2

    @pytest.mark.parametrize("flags, configured, echo", [
        ([], 86_400, "| stale quotes=24 h | volume cap="),
        ([], None, "| stale quotes=off | volume cap="),
        (["--no-stale-quote-check"], 86_400, "| stale quotes=off (config: 24 h) | volume cap="),
        (["--stale-quote-hours", "48"], 86_400, "| stale quotes=48 h (config: 24 h) | volume cap="),
        (["--stale-quote-hours", "24"], 86_400, "| stale quotes=24 h | volume cap="),
        (["--stale-quote-hours", "6"], None, "| stale quotes=6 h (config: off) | volume cap="),
    ])
    def test_the_config_echo_names_it(self, cli, monkeypatch, caplog, flags, configured, echo):
        monkeypatch.setattr(backtest, "BACKTEST_STALE_QUOTE_MAX_AGE_SECONDS", configured)
        with caplog.at_level(logging.INFO):
            _run(monkeypatch, *flags)
        assert echo in caplog.text
        # Between k and the volume cap, so the scheduled-close echo still meets k
        assert "| scheduled close=on | k=" in caplog.text

    @pytest.mark.parametrize("configured, word", [(86_400, "24 h"), (None, "off")])
    def test_the_help_text_names_the_configured_value(self, cli, monkeypatch, capsys,
                                                       configured, word):
        monkeypatch.setattr(backtest, "BACKTEST_STALE_QUOTE_MAX_AGE_SECONDS", configured)
        with pytest.raises(SystemExit):
            _run(monkeypatch, "--help")
        help_text = " ".join(capsys.readouterr().out.split())
        assert f"config.BACKTEST_STALE_QUOTE_MAX_AGE_SECONDS, currently {word})" in help_text

    def test_the_shipped_binding_is_the_config_s(self):
        from kalshi_betting import config
        assert backtest.BACKTEST_STALE_QUOTE_MAX_AGE_SECONDS == \
            config.BACKTEST_STALE_QUOTE_MAX_AGE_SECONDS == 86_400


class TestMaxVolumeShareArgument:
    """--max-volume-share PCT / --no-volume-cap (C7): the most of a market's
    next-day traded contracts a buy or a sale may take, for this run."""

    def test_default_is_no_override(self, cli, monkeypatch):
        _run(monkeypatch)
        assert cli["sweep_kwargs"]["max_volume_share"] is None

    @pytest.mark.parametrize("flags, value", [(["--max-volume-share", "10"], 0.10),
                                              (["--max-volume-share", "2.5"], 0.025),
                                              (["--max-volume-share", "100"], 1.0),
                                              (["--no-volume-cap"], False)])
    def test_the_flags_reach_the_sweep(self, cli, monkeypatch, flags, value):
        _run(monkeypatch, *flags)
        assert cli["sweep_kwargs"]["max_volume_share"] == value
        assert type(cli["sweep_kwargs"]["max_volume_share"]) is type(value)

    @pytest.mark.parametrize("pct", ["0", "-1", "100.5", "nan", "inf"])
    def test_a_bad_percent_is_a_usage_error(self, cli, monkeypatch, capsys, pct):
        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, "--max-volume-share", pct)
        assert exc.value.code == 2
        assert "--max-volume-share must be a percent above 0 and at most 100" in \
            capsys.readouterr().err

    def test_the_two_flags_are_exclusive(self, cli, monkeypatch):
        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, "--max-volume-share", "5", "--no-volume-cap")
        assert exc.value.code == 2

    @pytest.mark.parametrize("flags, configured, echo", [
        ([], 0.10, "| volume cap=10% | ladders="),
        ([], None, "| volume cap=off | ladders="),
        (["--no-volume-cap"], 0.10, "| volume cap=off (config: 10%) | ladders="),
        (["--max-volume-share", "5"], 0.10, "| volume cap=5% (config: 10%) | ladders="),
        (["--max-volume-share", "10"], 0.10, "| volume cap=10% | ladders="),
    ])
    def test_the_config_echo_names_it(self, cli, monkeypatch, caplog, flags, configured, echo):
        monkeypatch.setattr(backtest, "BACKTEST_MAX_VOLUME_SHARE", configured)
        with caplog.at_level(logging.INFO):
            _run(monkeypatch, *flags)
        assert echo in caplog.text

    @pytest.mark.parametrize("configured, word", [(0.10, "10%"), (None, "off")])
    def test_the_help_text_names_the_configured_value(self, cli, monkeypatch, capsys,
                                                       configured, word):
        monkeypatch.setattr(backtest, "BACKTEST_MAX_VOLUME_SHARE", configured)
        with pytest.raises(SystemExit):
            _run(monkeypatch, "--help")
        help_text = " ".join(capsys.readouterr().out.split())
        assert f"config.BACKTEST_MAX_VOLUME_SHARE, currently {word})" in help_text

    def test_the_shipped_binding_is_the_config_s(self):
        from kalshi_betting import config
        assert backtest.BACKTEST_MAX_VOLUME_SHARE == config.BACKTEST_MAX_VOLUME_SHARE == 0.10
        assert config.BACKTEST_VOLUME_SHARE_WINDOW_SECONDS == 86_400


class TestSameEventLaddersArgument:
    """--same-event-ladders / --no-same-event-ladders (DR-73c)."""

    def test_default_is_no_override(self, cli, monkeypatch):
        # None is the "no override" sentinel _extract_pairs and _find_entry
        # resolve against their own module's TIME_SERIES_SAME_EVENT_LADDERS at
        # call time — the CLI must not pre-resolve it, exactly as it must not
        # pre-resolve k.
        _run(monkeypatch)
        assert cli["sweep_kwargs"]["same_event_ladders"] is None

    def test_the_flag_turns_ladders_on(self, cli, monkeypatch):
        _run(monkeypatch, "--same-event-ladders")
        assert cli["sweep_kwargs"]["same_event_ladders"] is True

    def test_the_negated_flag_turns_ladders_off(self, cli, monkeypatch):
        # BooleanOptionalAction gives the --no- form from one declaration, and
        # it must reach the sweep as an explicit False rather than as None —
        # False overrides a switched-ON config, None defers to it.
        _run(monkeypatch, "--no-same-event-ladders")
        assert cli["sweep_kwargs"]["same_event_ladders"] is False

    def test_it_threads_alongside_the_other_flags(self, cli, monkeypatch):
        _run(monkeypatch, "--same-event-ladders", "--interval-discount", "0.4",
             "--no-sweep")
        kwargs = cli["sweep_kwargs"]
        assert kwargs["same_event_ladders"] is True
        assert kwargs["interval_discount"] == pytest.approx(0.4)
        assert kwargs["sweep"] is False

    @pytest.mark.parametrize("flag, configured, word", [
        ("--same-event-ladders", False, "on"),
        ("--no-same-event-ladders", True, "off"),
    ])
    def test_the_config_echo_names_the_effective_setting(self, cli, monkeypatch, caplog,
                                                         flag, configured, word):
        # Echoed BEFORE the fetch, beside k, so an operator can abort a
        # multi-hour run configured the wrong way round. Each flag is run
        # against the OPPOSITE configured value, so the echo can only print
        # `word` by honouring the flag: a flag that agreed with the constant
        # would pass an echo that ignores the flag entirely.
        monkeypatch.setattr(backtest, "TIME_SERIES_SAME_EVENT_LADDERS", configured)
        with caplog.at_level(logging.INFO):
            _run(monkeypatch, flag)
        assert f"ladders={word}" in caplog.text

    @pytest.mark.parametrize("configured, word", [(True, "on"), (False, "off")])
    def test_the_help_text_names_the_configured_value(self, cli, monkeypatch, capsys,
                                                       configured, word):
        # The flag's help is built when main() builds its parser, from
        # backtest's own binding, so it names the value a flagless run uses and
        # cannot go stale when the shipped value changes. argparse wraps help
        # lines, so whitespace is collapsed before matching.
        monkeypatch.setattr(backtest, "TIME_SERIES_SAME_EVENT_LADDERS", configured)
        with pytest.raises(SystemExit):
            _run(monkeypatch, "--help")
        assert f"currently {word})" in " ".join(capsys.readouterr().out.split())

    @pytest.mark.parametrize("configured, word", [(True, "on"), (False, "off")])
    def test_the_config_echo_defaults_to_the_config_constant(self, cli, monkeypatch,
                                                             caplog, configured, word):
        # Patched to BOTH values: one row at either value passes an echo
        # hard-coded to that value (`bool(args.same_event_ladders)` prints
        # "off" on a genuinely ON run, a literal "on" prints "on" on an OFF
        # one) — the misreport this pre-fetch echo exists to catch before a
        # multi-hour fetch. The module attribute is the seam, not config's:
        # backtest.py binds the constant by value at import.
        monkeypatch.setattr(backtest, "TIME_SERIES_SAME_EVENT_LADDERS", configured)
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        assert f"ladders={word}" in caplog.text

    @pytest.mark.parametrize("configured, flag, echo", [
        (True, "--no-same-event-ladders", "ladders=off (config: on) | spread band="),
        (False, "--same-event-ladders", "ladders=on (config: off) | spread band="),
    ])
    def test_the_config_echo_names_a_departure_from_the_configured_switch(
        self, cli, monkeypatch, caplog, configured, flag, echo,
    ):
        # A run overridden the OTHER way measures a strategy this checkout's
        # live finder does not trade — the echo names the CONFIGURED value
        # (not just the resolved one) so an operator reading it before a
        # multi-hour fetch can see the run departs from what the bot trades.
        monkeypatch.setattr(backtest, "TIME_SERIES_SAME_EVENT_LADDERS", configured)
        with caplog.at_level(logging.INFO):
            _run(monkeypatch, flag)
        assert echo in caplog.text

    @pytest.mark.parametrize("configured, flags", [
        (True, ["--same-event-ladders"]),
        (False, ["--no-same-event-ladders"]),
        (True, []),
        (False, []),
    ])
    def test_the_config_echo_names_no_departure_when_the_run_matches(
        self, cli, monkeypatch, caplog, configured, flags,
    ):
        # Whether by an override that agrees with the config or by leaving the
        # flag to it, a run that replays the configured switch gets the bare
        # on/off reading, with no "(config: ...)" clause to depart from.
        monkeypatch.setattr(backtest, "TIME_SERIES_SAME_EVENT_LADDERS", configured)
        with caplog.at_level(logging.INFO):
            _run(monkeypatch, *flags)
        assert f"ladders={'on' if configured else 'off'} | spread band=" in caplog.text
        assert "(config:" not in caplog.text


class TestSpreadBandArguments:
    """--spread-min/--spread-max/--no-band-sweep (PB4)."""

    def test_default_is_no_band(self, cli, monkeypatch):
        _run(monkeypatch)
        # None is the "no override" sentinel config.time_series_spread_band
        # resolves to BACKTEST_DEFAULT_SPREAD_BAND at call time — the CLI
        # must not pre-resolve it when both flags are omitted.
        assert cli["sweep_kwargs"]["spread_band"] is None

    def test_spread_min_alone_resolves_against_the_default_ceiling(self, cli, monkeypatch):
        _run(monkeypatch, "--spread-min", "0.3")
        assert cli["sweep_kwargs"]["spread_band"] == pytest.approx((0.3, 1.0))

    def test_spread_max_alone_resolves_against_the_default_floor(self, cli, monkeypatch):
        _run(monkeypatch, "--spread-max", "0.6")
        assert cli["sweep_kwargs"]["spread_band"] == pytest.approx((0.0, 0.6))

    def test_both_flags_resolve_to_their_own_values(self, cli, monkeypatch):
        _run(monkeypatch, "--spread-min", "0.3", "--spread-max", "0.6")
        assert cli["sweep_kwargs"]["spread_band"] == pytest.approx((0.3, 0.6))

    @pytest.mark.parametrize("flag", ["--spread-min", "--spread-max"])
    @pytest.mark.parametrize("value", ["-0.1", "1.5"])
    def test_out_of_range_value_errors(self, cli, monkeypatch, capsys, flag, value):
        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, flag, value)
        assert exc.value.code == 2
        assert f"{flag} must be between 0 and 1" in capsys.readouterr().err
        # parser.error() aborts before any client is built or any run starts
        assert "sweep_kwargs" not in cli

    def test_boundary_values_are_accepted(self, cli, monkeypatch):
        _run(monkeypatch, "--spread-min", "0.0", "--spread-max", "1.0")
        assert cli["sweep_kwargs"]["spread_band"] == pytest.approx((0.0, 1.0))

    def test_floor_equal_to_ceiling_errors(self, cli, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, "--spread-min", "0.5", "--spread-max", "0.5")
        assert exc.value.code == 2
        assert "must be strictly less than" in capsys.readouterr().err
        assert "sweep_kwargs" not in cli

    def test_floor_above_ceiling_errors(self, cli, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, "--spread-min", "0.6", "--spread-max", "0.3")
        assert exc.value.code == 2
        assert "must be strictly less than" in capsys.readouterr().err
        assert "sweep_kwargs" not in cli

    def test_spread_min_alone_at_1_errors(self, cli, monkeypatch, capsys):
        # One flag alone is checked against the OTHER side's configured
        # default (ceiling 1.0 here), not skipped: a floor of 1.0 leaves no
        # band, and must exit through parser.error rather than reach
        # config.time_series_spread_band's ValueError after logging.
        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, "--spread-min", "1.0")
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "must be strictly less than" in err
        assert "(config default)" in err
        assert "sweep_kwargs" not in cli

    def test_spread_max_alone_at_0_errors(self, cli, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, "--spread-max", "0")
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "must be strictly less than" in err
        assert "(config default)" in err
        assert "sweep_kwargs" not in cli

    def test_rejection_prints_the_bounds_exactly(self, cli, monkeypatch, capsys):
        # Two bounds that :g would both print as "0.3" must stay tellable apart
        with pytest.raises(SystemExit):
            _run(monkeypatch, "--spread-min", "0.3000001", "--spread-max", "0.3")
        err = capsys.readouterr().err
        assert "0.3000001" in err
        assert "floor 0.3 must" not in err

    def test_band_sweep_is_on_by_default(self, cli, monkeypatch):
        _run(monkeypatch)
        assert cli["sweep_kwargs"]["band_sweep"] is True

    def test_no_band_sweep_turns_it_off(self, cli, monkeypatch):
        _run(monkeypatch, "--no-band-sweep")
        assert cli["sweep_kwargs"]["band_sweep"] is False

    def test_no_sweep_does_not_turn_off_the_band_sweep(self, cli, monkeypatch):
        # --no-sweep skips only the k grid; with the band sweep still on,
        # each band is simulated at the primary k only (see the --no-sweep
        # help text) — the two flags are independent.
        _run(monkeypatch, "--no-sweep")
        kwargs = cli["sweep_kwargs"]
        assert kwargs["sweep"] is False
        assert kwargs["band_sweep"] is True

    def test_no_sweep_and_no_band_sweep_combine(self, cli, monkeypatch):
        _run(monkeypatch, "--no-sweep", "--no-band-sweep")
        kwargs = cli["sweep_kwargs"]
        assert kwargs["sweep"] is False
        assert kwargs["band_sweep"] is False

    @pytest.mark.parametrize("argv,expected", [
        ((), True), (("--no-sweep",), True), (("--no-band-sweep",), False),
    ])
    def test_the_tier_floors_off_runs_ride_the_band_sweep(self, cli, monkeypatch, argv,
                                                        expected):
        # No flag of their own: the dashboard's tier-floors-off data is on
        # exactly when the band grid is, and --no-band-sweep skips both
        _run(monkeypatch, *argv)
        kwargs = cli["sweep_kwargs"]
        assert kwargs["tier_off_sweep"] is expected
        assert kwargs["tier_off_sweep"] is kwargs["band_sweep"]


class TestSpreadBandEcho:
    """The pre-fetch echo names the resolved band and the band-sweep setting."""

    def test_default_band_echo(self, cli, monkeypatch, caplog):
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        assert "spread band=0-1" in caplog.text
        assert "band sweep=on" in caplog.text

    def test_custom_band_echo(self, cli, monkeypatch, caplog):
        with caplog.at_level(logging.INFO):
            _run(monkeypatch, "--spread-min", "0.3", "--spread-max", "0.6")
        assert "spread band=0.3-0.6" in caplog.text

    def test_no_band_sweep_echo(self, cli, monkeypatch, caplog):
        with caplog.at_level(logging.INFO):
            _run(monkeypatch, "--no-band-sweep")
        assert "band sweep=off" in caplog.text

    def test_echo_keeps_the_k_and_ladders_substrings(self, cli, monkeypatch, caplog):
        # The existing "k=..."/"ladders=..." substrings other tests and
        # tooling grep for must survive the appended band fields verbatim.
        # The switch is pinned off, the setting this row was written under,
        # so its "ladders=on" comes from the flag and not from the constant.
        monkeypatch.setattr(backtest, "TIME_SERIES_SAME_EVENT_LADDERS", False)
        with caplog.at_level(logging.INFO):
            _run(monkeypatch, "--interval-discount", "0.62", "--same-event-ladders")
        text = caplog.text
        assert "k=0.620" in text
        assert "ladders=on" in text
        assert "spread band=0-1" in text


class TestCapSweepArgument:
    """--no-cap-sweep: the per-trade size-cap sweep is ON by default, like the
    band sweep, and the flag threads cap_sweep=False into run_backtest_sweep."""

    def test_cap_sweep_is_on_by_default(self, cli, monkeypatch):
        _run(monkeypatch)
        assert cli["sweep_kwargs"]["cap_sweep"] is True

    def test_no_cap_sweep_turns_it_off(self, cli, monkeypatch):
        _run(monkeypatch, "--no-cap-sweep")
        assert cli["sweep_kwargs"]["cap_sweep"] is False

    def test_it_is_independent_of_the_band_and_k_sweeps(self, cli, monkeypatch):
        _run(monkeypatch, "--no-band-sweep", "--no-sweep")
        kwargs = cli["sweep_kwargs"]
        assert (kwargs["cap_sweep"], kwargs["band_sweep"], kwargs["sweep"]) == (
            True, False, False)
        _run(monkeypatch, "--no-cap-sweep")
        kwargs = cli["sweep_kwargs"]
        assert (kwargs["cap_sweep"], kwargs["band_sweep"], kwargs["sweep"]) == (
            False, True, True)

    def test_the_echo_names_the_setting(self, cli, monkeypatch, caplog):
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        assert "| band sweep=on | cap sweep=on" in caplog.text
        caplog.clear()
        with caplog.at_level(logging.INFO):
            _run(monkeypatch, "--no-cap-sweep", "--interval-discount", "0.62")
        text = caplog.text
        assert "cap sweep=off" in text
        # The earlier fields other tests and tooling grep for are unchanged
        assert "k=0.620" in text and "spread band=0-1" in text and "band sweep=on" in text


class TestSellSweepArgument:
    """--no-sell-sweep: the dashboard's Sell family is ON by default, like the
    cap sweep, and the flag threads sell_sweep=False into run_backtest_sweep;
    the echo names the setting right after the cap sweep's."""

    def test_sell_sweep_is_on_by_default(self, cli, monkeypatch, caplog):
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        assert cli["sweep_kwargs"]["sell_sweep"] is True
        # The band and cap sweeps' substrings other tests read stay a prefix
        assert "| band sweep=on | cap sweep=on | sell sweep=on (" in caplog.text

    def test_no_sell_sweep_turns_only_it_off(self, cli, monkeypatch, caplog):
        with caplog.at_level(logging.INFO):
            _run(monkeypatch, "--no-sell-sweep")
        kwargs = cli["sweep_kwargs"]
        assert (kwargs["sell_sweep"], kwargs["rebalance_sweep"], kwargs["band_sweep"],
                kwargs["cap_sweep"]) == (False, True, True, True)
        assert "| cap sweep=on | sell sweep=off | rebalance sweep=on | live rule=" in caplog.text


class TestRebalanceSweepArgument:
    """--no-rebalance-sweep: the dashboard's Rebalance family is ON by default,
    like the cap and Sell families, and the flag threads rebalance_sweep=False
    into run_backtest_sweep; the echo names the setting after the sell
    sweep's. There is no flag for adding to or trimming held pairs alone."""

    def test_rebalance_sweep_is_on_by_default(self, cli, monkeypatch, caplog):
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        assert cli["sweep_kwargs"]["rebalance_sweep"] is True
        assert ") | rebalance sweep=on | live rule=" in caplog.text
        # The family the old flags named is gone from the call
        assert "add_on_sweep" not in cli["sweep_kwargs"]
        assert "trim_sweep" not in cli["sweep_kwargs"]

    def test_no_rebalance_sweep_turns_only_it_off(self, cli, monkeypatch, caplog):
        with caplog.at_level(logging.INFO):
            _run(monkeypatch, "--no-rebalance-sweep", "--no-cap-sweep", "--sell-workers", "1")
        kwargs = cli["sweep_kwargs"]
        assert (kwargs["rebalance_sweep"], kwargs["sell_sweep"], kwargs["band_sweep"],
                kwargs["cap_sweep"]) == (False, True, True, False)
        assert ("| cap sweep=off | sell sweep=on (1 worker process) | rebalance sweep=off "
                "| live rule=") in caplog.text

    @pytest.mark.parametrize("flag", ["--no-add-on-sweep", "--no-trim-sweep",
                                      "--no-trim-rise-sweep"])
    def test_the_old_flags_are_refused(self, cli, monkeypatch, capsys, flag):
        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, flag)
        assert exc.value.code == 2
        assert "unrecognized arguments" in capsys.readouterr().err
        assert "sweep_kwargs" not in cli

    def test_the_help_names_both_selects_it_skips(self, cli, monkeypatch, capsys):
        text = _option_help(monkeypatch, capsys, "--no-rebalance-sweep", "--no-move-sweep")
        assert text.startswith(
            "Skip the dashboard's Rebalance select and its Min. price movement select")
        assert "a live run adds to and trims held pairs by its own saved settings" in text


class TestMoveSweepArgument:
    """--no-move-sweep: the Rebalance family's minimum price movements (the
    dashboard's Min. price movement select) are ON with the family by
    default; the flag threads move_sweep=False into run_backtest_sweep and
    leaves the family itself on, and --no-rebalance-sweep turns both off.
    The echo's rebalance clause says which, and a default run's echo reads
    "rebalance sweep=on"."""

    @staticmethod
    def _echo(caplog) -> str:
        (line,) = [r.getMessage() for r in caplog.records
                   if r.getMessage().startswith("Backtest config:")]
        return line

    def test_the_minimum_price_movements_are_on_with_the_family_by_default(
            self, cli, monkeypatch, caplog):
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        kwargs = cli["sweep_kwargs"]
        assert (kwargs["rebalance_sweep"], kwargs["move_sweep"]) == (True, True)
        assert ") | rebalance sweep=on | live rule=" in self._echo(caplog)

    def test_no_move_sweep_turns_only_them_off(self, cli, monkeypatch, caplog):
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        default_kwargs, default_echo = cli["sweep_kwargs"], self._echo(caplog)
        caplog.clear()
        with caplog.at_level(logging.INFO):
            _run(monkeypatch, "--no-move-sweep")
        kwargs = cli["sweep_kwargs"]
        assert (kwargs["move_sweep"], kwargs["rebalance_sweep"], kwargs["sell_sweep"],
                kwargs["band_sweep"], kwargs["cap_sweep"]) == (False, True, True, True, True)
        # No other keyword moves (each run builds a client of its own)
        skipped = {"move_sweep", "hist_client"}
        assert set(kwargs) == set(default_kwargs)
        assert ({key: value for key, value in kwargs.items() if key not in skipped}
                == {key: value for key, value in default_kwargs.items()
                    if key not in skipped})
        # ... and only the echo's rebalance clause does
        assert "| rebalance sweep=on | live rule=" in default_echo
        assert self._echo(caplog) == default_echo.replace(
            "| rebalance sweep=on | live rule=",
            "| rebalance sweep=on (no minimum price movements) | live rule=")

    @pytest.mark.parametrize("flags", [("--no-rebalance-sweep",),
                                       ("--no-rebalance-sweep", "--no-move-sweep")])
    def test_no_rebalance_sweep_turns_both_off(self, cli, monkeypatch, caplog, flags):
        with caplog.at_level(logging.INFO):
            _run(monkeypatch, *flags)
        kwargs = cli["sweep_kwargs"]
        assert (kwargs["rebalance_sweep"], kwargs["move_sweep"]) == (False, False)
        assert ") | rebalance sweep=off | live rule=" in self._echo(caplog)

    def test_the_help_names_the_select_it_skips(self, cli, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, "--help")
        assert exc.value.code == 0
        text = " ".join(capsys.readouterr().out.split())
        assert "--no-move-sweep" in text
        assert "Skip the dashboard's Min. price movement select" in text
        assert "Rebalance itself is still built" in text
        assert "sweep_kwargs" not in cli

    def test_the_help_names_the_flag_that_sets_one_live_runs_minimum(self, cli, monkeypatch,
                                                                       capsys):
        # The help names main.py --min-price-move, which sets one live run's minimum,
        # and says the minimum holds back a live run's add-ons and trims alike
        text = _option_help(monkeypatch, capsys, "--no-move-sweep", "--sell-workers")
        assert text.startswith("Skip the dashboard's Min. price movement select")
        assert text.endswith("a live run's rebalancing (add-ons and trims) waits for a price "
                             "move by its own saved min_price_move (main.py --min-price-move "
                             "for one run)")


class TestSellWorkersArgument:
    """--sell-workers: how many worker processes the dashboard simulates its
    Sell select in — by default one less than the CPU count, within
    config.DASHBOARD_SELL_MAX_WORKERS, and at least 1 — handed to
    generate_dashboard and named in the echo."""

    @pytest.mark.parametrize(("cpus", "expected"), [
        (1, 1), (2, 1), (4, 3), (64, config.DASHBOARD_SELL_MAX_WORKERS), (None, 1)])
    def test_the_default_leaves_one_cpu_and_stays_in_the_bound(self, cli, monkeypatch,
                                                                 cpus, expected):
        monkeypatch.setattr(backtest.os, "cpu_count", lambda: cpus)
        _run(monkeypatch)
        assert cli["dashboard"][1]["sell_workers"] == expected

    def test_a_count_is_handed_through_and_echoed(self, cli, monkeypatch, caplog):
        with caplog.at_level(logging.INFO):
            _run(monkeypatch, "--sell-workers", "3")
        assert cli["dashboard"][1]["sell_workers"] == 3
        assert ("| sell sweep=on (3 worker processes) | rebalance sweep=on | live rule="
                in caplog.text)

    @pytest.mark.parametrize("value", ["0", "-2"])
    def test_a_count_below_one_is_refused(self, cli, monkeypatch, capsys, value):
        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, "--sell-workers", value)
        assert exc.value.code == 2
        assert "--sell-workers must be a positive integer" in capsys.readouterr().err
        assert "sweep_kwargs" not in cli


def _depth_model(snapshots: int = 3, ladders: int = 4_120) -> DepthModel:
    """A fitted-looking DepthModel; main() reads only its snapshot and ladder counts."""
    return DepthModel(cells={}, volume_rows={}, overall=(0.0,) * 7, snapshots=snapshots,
                      ladders=ladders, first_taken="2026-10-02T21:30:00Z",
                      last_taken="2026-10-09T21:30:00Z", digest="test")


class TestDepthModel:
    """main() fits the depth model before the fetch, hands it to the sweep, and
    the config echo says how trades will fill."""

    def test_the_model_reaches_the_sweep(self, cli, monkeypatch):
        cli["depth_model"] = _depth_model()
        _run(monkeypatch)
        # The very object load_depth_model returned, not a copy or a rebuild
        assert cli["sweep_kwargs"]["depth_model"] is cli["depth_model"]

    def test_with_no_model_the_sweep_gets_none(self, cli, monkeypatch):
        _run(monkeypatch)
        assert "depth_model" in cli["sweep_kwargs"]
        assert cli["sweep_kwargs"]["depth_model"] is None

    def test_it_is_loaded_before_the_fetch(self, cli, monkeypatch):
        _run(monkeypatch)
        assert cli["order"] == ["load_depth_model", "run_backtest_sweep"]

    def test_the_echo_names_a_walked_book(self, cli, monkeypatch, caplog):
        cli["depth_model"] = _depth_model(snapshots=3, ladders=4_120)
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        assert "fills=walked book (3 snapshots, 4120 ladders)" in caplog.text

    def test_the_echo_counts_one_snapshot_in_the_singular(self, cli, monkeypatch, caplog):
        cli["depth_model"] = _depth_model(snapshots=1, ladders=1)
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        assert "fills=walked book (1 snapshot, 1 ladder)" in caplog.text

    def test_the_echo_names_the_top_of_the_book_with_no_model(self, cli, monkeypatch, caplog):
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        assert "fills=top of book (no usable depth snapshot)" in caplog.text

    def test_the_clause_closes_the_echo_and_leaves_the_rest_alone(self, cli, monkeypatch,
                                                                    caplog):
        cli["depth_model"] = _depth_model()
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        [echo] = [r.getMessage() for r in caplog.records
                  if r.getMessage().startswith("Backtest config:")]
        # After the live rule, so every clause other tests read keeps its place
        assert echo.index("| live rule=") < echo.index("| fills=")
        assert echo.endswith("| fills=walked book (3 snapshots, 4120 ladders)")
        k = backtest.TIME_SERIES_INTERVAL_PROB_DISCOUNT
        assert "| sell sweep=on (" in echo and f"k={k:.3f}" in echo

    def test_the_echo_comes_before_the_fetch(self, cli, monkeypatch, caplog):
        # The line is logged before run_backtest_sweep is called, so an operator
        # can stop a run that would fill at the top of the book
        seen = {}
        original = cli["result"]

        def _sweep_after_echo(**kwargs):
            seen["echoed"] = any(r.getMessage().startswith("Backtest config:")
                                 for r in caplog.records)
            return original

        monkeypatch.setattr(backtest, "run_backtest_sweep", _sweep_after_echo)
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        assert seen == {"echoed": True}


class TestLiveRuleEcho:
    """The pre-fetch echo's "| live rule=..." clause names the saved live
    defaults' time-series rule and where they were saved — never a backtest
    scenario, never config.py's toggles, and never main.py's per-run overrides
    (a separate CLI this module cannot see). With none saved it says so, and
    with a refused file it fails soft."""

    @staticmethod
    def _save(monkeypatch, **constants) -> config.LiveSettings:
        """
        Patch config's toggle constants and save them as the live defaults.

        Args:
            monkeypatch (pytest.MonkeyPatch): pytest's per-test patcher.
            **constants: config toggle constants to patch, by name.

        Returns:
            config.LiveSettings: The saved defaults as read back.
        """
        for name, value in constants.items():
            monkeypatch.setattr(config, name, value)
        save_config_live_defaults()
        return config.read_saved_live_defaults()

    def test_the_echo_names_the_saved_rule(self, cli, monkeypatch, caplog):
        saved = self._save(monkeypatch, TIME_SERIES_TIER_FLOORS=False,
                           TIME_SERIES_SPREAD_BAND=(0.1, 0.6))
        # config.py's own toggles move after the save: the echo does not follow them
        monkeypatch.setattr(config, "TIME_SERIES_SPREAD_BAND", (0.0, 1.0))
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        rule = config.describe_time_series_rule(False, (0.1, 0.6))
        assert f"| live rule={rule} ({saved.origin})" in caplog.text
        # With no category/tag filter set, the clause is the rule alone
        assert "category/tag filter" not in caplog.text

    def test_the_echo_names_a_set_filter(self, cli, monkeypatch, caplog):
        saved = self._save(monkeypatch, TIME_SERIES_TIER_FLOORS=True,
                           TIME_SERIES_SPREAD_BAND=(0.0, 1.0),
                           TRADE_EXCLUDE_CATEGORIES=("Economics", "Sports"), TRADE_TAGS=("Fed",))
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        rule = config.describe_time_series_rule(True, (0.0, 1.0))
        assert (f"| live rule={rule}; category/tag filter "
                f"({config.describe_trade_filter(saved)}) ({saved.origin})") in caplog.text
        assert "exclude categories Economics, Sports; tags Fed" in caplog.text

    def test_a_cli_argument_never_moves_it(self, cli, monkeypatch, caplog):
        # --spread-min/--spread-max/--interval-discount name a BACKTEST
        # scenario; the live rule clause reads the saved defaults alone
        saved = self._save(monkeypatch, TIME_SERIES_TIER_FLOORS=True,
                           TIME_SERIES_SPREAD_BAND=(0.0, 1.0))
        with caplog.at_level(logging.INFO):
            _run(monkeypatch, "--spread-min", "0.3", "--spread-max", "0.6",
                 "--interval-discount", "0.62")
        rule = config.describe_time_series_rule(True, (0.0, 1.0))
        text = caplog.text
        assert f"| live rule={rule} ({saved.origin})" in text
        # The scenario clauses moved as usual, right beside the unmoved rule
        assert "k=0.620" in text and "spread band=0.3-0.6" in text

    def test_with_none_saved_it_says_so(self, cli, monkeypatch, caplog):
        assert not config.LIVE_DEFAULTS_FILE.exists()
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        text = caplog.text
        assert "| live rule=none saved (live runs refuse to start)" in text
        # The run still went ahead
        assert "sweep_kwargs" in cli

    def test_it_fails_soft_on_a_refused_file(self, cli, monkeypatch, caplog):
        # A refused file makes live_defaults() raise: a reporting clause fails
        # soft, while an invalid --spread-min/--spread-max pair exits
        config.LIVE_DEFAULTS_FILE.write_text("not json", encoding="utf-8")
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        text = caplog.text
        assert "live rule=not recorded — the saved live defaults are refused (" in text
        assert str(config.LIVE_DEFAULTS_FILE) in text
        # The run still went ahead
        assert "sweep_kwargs" in cli

    def test_a_file_an_older_version_saved_is_refused_too(self, cli, monkeypatch, caplog):
        # A file an older version saved is a refused file here: the echo never
        # names its values in their new form, and the run still goes ahead
        from .test_config import _OLD_VERSION_BYTES
        config.LIVE_DEFAULTS_FILE.write_bytes(_OLD_VERSION_BYTES)
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        text = caplog.text
        assert "live rule=not recorded — the saved live defaults are refused (" in text
        assert str(config.LIVE_DEFAULTS_FILE) in text
        assert "sweep_kwargs" in cli


# Tier labels and messages derived from config, exactly as backtest.main
# derives them — never a literal "16-30".
_SHORT_TIER_DAYS = f"0-{SHORT_DEADLINE_GAP_DAYS}-day"
_LONG_TIER_DAYS = f"{SHORT_DEADLINE_GAP_DAYS + 1}-{MAX_DEADLINE_GAP_DAYS}-day"
_GRID_NOTE = "the band sweep's grid bands are unaffected"


def _emptied(tier_days: str) -> str:
    """The WARNING's note for a tier the ceiling refuses outright."""
    return f"no {tier_days}-gap time-series candidate can enter"


def _on_tier(tier_days: str) -> str:
    """The WARNING's note for a ceiling within PRICE_EPSILON of the tier."""
    return f"only {tier_days}-gap spreads sitting exactly on the"


def _find_entry_keeps(gap_days: int, pA: float, pB: float, ceiling: float) -> bool:
    """
    Whether backtester._find_entry's two band tests keep this spread at no floor.

    The floor test is `gap < min_price_diff_for_gap(gap_days, spread_min=floor)
    - PRICE_EPSILON` and the ceiling test is config.time_series_spread_too_wide,
    both exactly as _find_entry applies them — the independent oracle for what
    the CLI's WARNING claims about an on-tier spread.
    """
    gap = pB - pA
    threshold = min_price_diff_for_gap(gap_days, spread_min=0.0)
    return not (gap < threshold - PRICE_EPSILON) and not time_series_spread_too_wide(gap, ceiling)


class TestSpreadBandTierWarning:
    """A ceiling at or below a deadline-gap tier gets a WARNING naming the tier.

    Only the PRIMARY scenario is affected: the band sweep's grid ceilings all
    sit above both tiers, so the message must not claim the whole backtest.
    """

    def test_ceiling_below_short_tier_empties_both_tiers(self, cli, monkeypatch, caplog):
        with caplog.at_level(logging.WARNING):
            _run(monkeypatch, "--spread-max", "0.1")
        text = caplog.text
        assert _emptied(_SHORT_TIER_DAYS) in text
        assert _emptied(_LONG_TIER_DAYS) in text
        assert "PRIMARY scenario" in text
        # The band sweep is on by default and its grid is untouched
        assert _GRID_NOTE in text
        assert "this backtest" not in text
        # The tiers are MINIMUM spreads; never print them as upper bounds
        assert "<=" not in text
        assert f"deadline-gap tier {MIN_PRICE_DIFF_SHORT_GAP:.2f}" in text

    def test_no_band_sweep_drops_the_grid_note(self, cli, monkeypatch, caplog):
        with caplog.at_level(logging.WARNING):
            _run(monkeypatch, "--spread-max", "0.1", "--no-band-sweep")
        assert _emptied(_LONG_TIER_DAYS) in caplog.text
        assert _GRID_NOTE not in caplog.text

    def test_ceiling_between_tiers_empties_the_long_tier_only(self, cli, monkeypatch, caplog):
        midpoint = (MIN_PRICE_DIFF_SHORT_GAP + MIN_PRICE_DIFF_LONG_GAP) / 2
        with caplog.at_level(logging.WARNING):
            _run(monkeypatch, "--spread-max", repr(midpoint))
        text = caplog.text
        assert _emptied(_LONG_TIER_DAYS) in text
        assert _SHORT_TIER_DAYS not in text

    def test_ceiling_on_the_long_tier_keeps_only_on_tier_spreads(self, cli, monkeypatch, caplog):
        # config.time_series_spread_band's docstring: a ceiling exactly ON a
        # tier keeps only spreads sitting on that tier, and a caller should
        # warn when the ceiling sits at or below a tier.
        with caplog.at_level(logging.WARNING):
            _run(monkeypatch, "--spread-max", repr(MIN_PRICE_DIFF_LONG_GAP))
        text = caplog.text
        assert _on_tier(_LONG_TIER_DAYS) in text
        assert _emptied(_LONG_TIER_DAYS) not in text
        assert _SHORT_TIER_DAYS not in text

    def test_ceiling_on_the_short_tier_empties_the_long_one(self, cli, monkeypatch, caplog):
        with caplog.at_level(logging.WARNING):
            _run(monkeypatch, "--spread-max", repr(MIN_PRICE_DIFF_SHORT_GAP))
        text = caplog.text
        assert _on_tier(_SHORT_TIER_DAYS) in text
        assert _emptied(_LONG_TIER_DAYS) in text

    # (gap_days, pA, pB): a real price pair whose spread sits on the tier,
    # carrying the float noise real candle prices carry.
    _ON_TIER = {
        "short": (5, 0.20, 0.35),    # 0.14999999999999997
        "long": (20, 0.15, 0.45),    # 0.30000000000000004
    }

    @pytest.mark.parametrize("which", ["short", "long"])
    @pytest.mark.parametrize("offset", [-2e-6, -5e-7, 0.0, 5e-7, 2e-6])
    def test_emptied_note_matches_what_find_entry_does(self, cli, monkeypatch, caplog,
                                                       which, offset):
        # The note must fire exactly when _find_entry's own tests refuse an
        # on-tier spread — including a hair below the tier, where
        # PRICE_EPSILON's keep side still admits it (TS-09).
        gap_days, pA, pB = self._ON_TIER[which]
        tier = MIN_PRICE_DIFF_SHORT_GAP if which == "short" else MIN_PRICE_DIFF_LONG_GAP
        tier_days = _SHORT_TIER_DAYS if which == "short" else _LONG_TIER_DAYS
        assert abs((pB - pA) - tier) < 1e-12  # the fixture really is on the tier
        ceiling = tier + offset
        with caplog.at_level(logging.WARNING):
            _run(monkeypatch, "--spread-max", repr(ceiling))
        kept = _find_entry_keeps(gap_days, pA, pB, ceiling)
        assert (_emptied(tier_days) in caplog.text) is (not kept)
        # Within PRICE_EPSILON of the tier and still admitted: the softer note
        within = abs(offset) <= PRICE_EPSILON
        assert (_on_tier(tier_days) in caplog.text) is (kept and within)

    def test_default_band_warns_nothing(self, cli, monkeypatch, caplog):
        with caplog.at_level(logging.WARNING):
            _run(monkeypatch)
        assert "deadline-gap tier" not in caplog.text

    def test_wide_custom_ceiling_warns_nothing(self, cli, monkeypatch, caplog):
        with caplog.at_level(logging.WARNING):
            _run(monkeypatch, "--spread-min", "0.3", "--spread-max", "0.9")
        assert "deadline-gap tier" not in caplog.text


class TestSpreadBandConfigDefault:
    """The omitted side, and the no-flag echo, come from config at call time.

    config itself is patched — time_series_spread_band reads its own module
    global, and backtest never binds BACKTEST_DEFAULT_SPREAD_BAND by value —
    so a hardcoded (0.0, 1.0) anywhere in the CLI fails these.
    """

    @pytest.fixture
    def narrow_default(self, monkeypatch):
        monkeypatch.setattr(config, "BACKTEST_DEFAULT_SPREAD_BAND", (0.2, 0.7))

    def test_no_flags_passes_none_and_echoes_the_config_band(
            self, narrow_default, cli, monkeypatch, caplog):
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        assert cli["sweep_kwargs"]["spread_band"] is None
        assert "spread band=0.2-0.7" in caplog.text

    def test_spread_min_alone_takes_the_config_ceiling(self, narrow_default, cli, monkeypatch):
        _run(monkeypatch, "--spread-min", "0.3")
        assert cli["sweep_kwargs"]["spread_band"] == pytest.approx((0.3, 0.7))

    def test_spread_max_alone_takes_the_config_floor(self, narrow_default, cli, monkeypatch):
        _run(monkeypatch, "--spread-max", "0.6")
        assert cli["sweep_kwargs"]["spread_band"] == pytest.approx((0.2, 0.6))

    def test_spread_max_below_the_config_floor_errors(
            self, narrow_default, cli, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, "--spread-max", "0.1")
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "must be strictly less than" in err
        assert "0.2 (config default)" in err
        assert "sweep_kwargs" not in cli

    def test_invalid_config_default_raises_before_logging(self, cli, monkeypatch):
        # A bad default is a config bug, not operator input, so it raises
        # config's own ValueError rather than a parser error — but on every
        # run, flags or not, BEFORE logging is configured (TS-20).
        monkeypatch.setattr(config, "BACKTEST_DEFAULT_SPREAD_BAND", (0.7, 0.2))
        configured = []
        monkeypatch.setattr(backtest.logging, "basicConfig",
                            lambda **kwargs: configured.append(kwargs))
        with pytest.raises(ValueError, match="spread band"):
            _run(monkeypatch)
        assert configured == []
        assert "sweep_kwargs" not in cli


class TestSpreadValidationPrecedesLogging:
    """TS-20's ORDER, pinned directly rather than through its symptom.

    delay=True on the file handler means "no log file" alone would still pass
    with the spread checks moved after basicConfig, so these record whether
    basicConfig was called at all.
    """

    @pytest.fixture
    def basic_config_calls(self, cli, monkeypatch):
        calls: list = []

        def _record(**kwargs):
            calls.append(kwargs)
            for handler in kwargs.get("handlers", []):
                handler.close()

        # Overrides the cli fixture's stub, which it depends on and so follows
        monkeypatch.setattr(backtest.logging, "basicConfig", _record)
        return calls

    @pytest.mark.parametrize("argv", [
        ("--spread-min", "1.5"),
        ("--spread-max", "-0.1"),
        ("--spread-min", "0.6", "--spread-max", "0.3"),
        ("--spread-min", "1.0"),
        ("--spread-max", "0"),
    ])
    def test_rejection_happens_before_logging_is_configured(
            self, basic_config_calls, cli, monkeypatch, argv):
        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, *argv)
        assert exc.value.code == 2
        assert basic_config_calls == []
        assert "sweep_kwargs" not in cli

    def test_a_valid_band_does_configure_logging(self, basic_config_calls, cli, monkeypatch):
        # Control: the recorder does see basicConfig when nothing is rejected
        _run(monkeypatch, "--spread-min", "0.3", "--spread-max", "0.6")
        assert len(basic_config_calls) == 1


class TestDashboardHandoff:
    """generate_dashboard gets the whole sweep plus the RESOLVED primary k, and
    the run closes by saying how the page's scenario becomes the live defaults."""

    def test_sweep_and_primary_k_are_passed(self, cli, monkeypatch):
        _run(monkeypatch, "--interval-discount", "0.62")
        args, kwargs = cli["dashboard"]
        result = cli["result"]
        # The four positional arguments are unchanged from before the sweep;
        # the starting balance is the account's value (see TestStartingBalance)
        assert args[0] is result.primary.trades
        assert args[1] is result.primary.equity_df
        assert args[2] == date.fromisoformat("2024-01-01")
        assert args[3] == pytest.approx(211.42)
        # ...and the sweep travels whole rather than unpacked
        assert kwargs["sweep"] is result
        # The k the plotted trades were SIZED at — read back off the point, not
        # re-derived from the CLI flag, so the two can never disagree
        assert kwargs["interval_discount"] == result.primary.k

    def test_the_series_categories_are_passed_through(self, cli, monkeypatch):
        # One read of the listing, before the sweep: the sweep's same-title
        # sport check files each series by it (DR-79), and the dashboard files
        # each trade by the same object
        reads: list = []

        def load(client):
            reads.append(client)
            cli["order"].append("load_series_categories")
            return cli["series_categories"]

        monkeypatch.setattr(backtest, "load_series_categories", load)
        _run(monkeypatch)
        _, kwargs = cli["dashboard"]
        assert kwargs["series_categories"] is cli["series_categories"]
        assert cli["sweep_kwargs"]["series_categories"] is cli["series_categories"]
        assert reads == [cli["live_client"]]
        assert cli["order"].index("load_series_categories") < cli["order"].index(
            "run_backtest_sweep")

    def test_the_fee_multipliers_are_read_once_and_passed_to_the_sweep(self, cli, monkeypatch):
        # Read with the production client beside the listing, handed to the
        # sweep (which carries them; no fee reads them yet)
        reads: list = []

        def load(client, **kwargs):
            reads.append((client, kwargs))
            return cli["fee_multipliers"]

        monkeypatch.setattr(backtest, "load_series_fee_multipliers", load)
        _run(monkeypatch)
        # The backtest alone may upgrade a current file that has no fees
        assert reads == [(cli["live_client"], {"upgrade": True})]
        assert cli["sweep_kwargs"]["fee_multipliers"] is cli["fee_multipliers"]

    def test_the_risk_free_rates_are_passed_through(self, cli, monkeypatch):
        _run(monkeypatch)
        _, kwargs = cli["dashboard"]
        assert kwargs["risk_free"] is cli["risk_free"]

    def test_the_run_closes_with_how_to_save_the_live_defaults(self, cli, monkeypatch, caplog):
        # After pointing at the page, the run says how its filter bar's
        # scenario becomes the live defaults, or is traded: the launcher (which
        # shows this page as the live dashboard's Backtest tab), then the page's
        # buttons (the page itself cannot write a file)
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        opened = messages.index("Open the HTML file in a browser to view the interactive "
                                "charts.")
        assert messages[opened + 1] == (
            "To save the filter bar's scenario as the live defaults, or to trade: run "
            "./start_dashboard.sh (it starts the defaults server and the live dashboard, whose "
            "Backtest tab is this page), then use the filter bar's Save as live defaults… or "
            "Trade using defaults… button.")
        assert messages[opened + 1:] == [messages[opened + 1]]


class TestDataPage:
    """After the backtest page, a run (and a --dashboard-only rebuild) builds the
    Data page from the run's sweep and listing, with at most config.EDA_MAX_WORKERS
    workers; --no-eda skips it, and a Data page that fails is one WARNING that
    changes neither the exit code nor the backtest page."""

    def test_it_is_built_after_the_backtest_page_with_the_runs_sweep_and_listing(
            self, cli, monkeypatch, caplog):
        with caplog.at_level(logging.INFO):
            _run(monkeypatch, "--sell-workers", "2")
        assert cli["pages"] == ["backtest", "data"]
        [kwargs] = cli["eda"]
        assert kwargs["sweep"] is cli["result"]
        assert kwargs["series_categories"] is cli["series_categories"]
        assert kwargs["workers"] == 2
        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        written = messages.index(
            f"Data page written: {cli['eda_result']} (the live dashboard's Data tab)")
        # Said before the build, so a long first build is not taken for a hang
        building = messages.index(
            "Building the Data page from backtest_cache/: the first build (or one after many "
            "new day files) reads each cached day file once and can take several minutes "
            "(about 10 on a full cache); later builds take seconds. --no-eda skips it")
        assert building < written

    def test_its_workers_never_pass_the_eda_bound(self, cli, monkeypatch):
        _run(monkeypatch, "--sell-workers", str(config.EDA_MAX_WORKERS + 3))
        assert cli["eda"][0]["workers"] == config.EDA_MAX_WORKERS
        assert cli["dashboard"][1]["sell_workers"] == config.EDA_MAX_WORKERS + 3

    def test_no_eda_skips_it(self, cli, monkeypatch, caplog):
        with caplog.at_level(logging.INFO):
            _run(monkeypatch, "--no-eda")
        assert cli["pages"] == ["backtest"]
        assert cli["eda"] == []
        assert not any("Data page" in r.getMessage() for r in caplog.records)

    def test_a_data_page_that_raises_changes_nothing_else(self, cli, monkeypatch, caplog,
                                                          tmp_path):
        # A real backtest page, so its bytes can be compared with and without
        # the Data page failing
        def write_page(*args, **kwargs):
            cli["pages"].append("backtest")
            page = tmp_path / "dashboard.html"
            page.write_text("<html>the backtest</html>", encoding="utf-8")
            return page

        monkeypatch.setattr(backtest, "generate_dashboard", write_page)
        cli["eda_result"] = RuntimeError("the cache could not be read")
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)    # returns: exit 0
        assert (tmp_path / "dashboard.html").read_text(encoding="utf-8") == (
            "<html>the backtest</html>")
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert ("Data page not built (RuntimeError: the cache could not be read); the "
                "backtest page above is unchanged") in warnings
        # The run still closes as it always does
        messages = [r.getMessage() for r in caplog.records]
        assert messages[-2] == "Open the HTML file in a browser to view the interactive charts."

    def test_a_data_page_that_writes_nothing_is_a_warning(self, cli, monkeypatch, caplog):
        cli["eda_result"] = None
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        assert any(r.levelno == logging.WARNING
                   and r.getMessage().startswith("Data page: no page could be written")
                   for r in caplog.records)

    def test_the_rebuild_builds_it_too(self, rebuild, monkeypatch):
        rebuild["eda"].clear()
        rebuild["pages"].clear()
        _run(monkeypatch, "--dashboard-only", "--sell-workers", "1")
        assert rebuild["pages"] == ["backtest", "data"]
        [kwargs] = rebuild["eda"]
        # The saved run's sweep and listing, read back from the file
        assert kwargs["sweep"] is rebuild["dashboard"][1]["sweep"]
        assert kwargs["sweep"].primary.k == rebuild["full_dashboard"][1]["sweep"].primary.k
        assert kwargs["series_categories"] == rebuild["series_categories"]
        assert kwargs["workers"] == 1

    def test_the_rebuild_honours_no_eda(self, rebuild, monkeypatch):
        rebuild["eda"].clear()
        rebuild["pages"].clear()
        _run(monkeypatch, "--dashboard-only", "--no-eda")
        assert rebuild["pages"] == ["backtest"]
        assert rebuild["eda"] == []

    def test_a_rebuild_whose_data_page_raises_still_builds_its_backtest_page(
            self, rebuild, monkeypatch, caplog):
        rebuild["eda_result"] = OSError("disk full")
        rebuild["pages"].clear()
        with caplog.at_level(logging.INFO):
            _run(monkeypatch, "--dashboard-only")
        assert rebuild["pages"] == ["backtest", "data"]
        assert any(r.levelno == logging.WARNING
                   and r.getMessage().startswith("Data page not built (OSError: disk full)")
                   for r in caplog.records)


class TestSummaryBlock:
    """A default run's summary block is unchanged: it reports the PRIMARY
    point, exactly as the plain run_backtest() path used to."""

    def test_summary_reports_the_primary_point(self, cli, monkeypatch, caplog):
        cli["result"] = _sweep(n_trades=2)
        with caplog.at_level(logging.INFO):
            # The curve opens at $10,000, so the return is measured against that
            _run(monkeypatch, "--balance", "10000")
        text = caplog.text
        assert "Backtest Summary" in text
        assert "Total trades:  2" in text
        assert "Win rate:      100.0%" in text
        assert "Total return:  +6.9%" in text
        assert "Final balance: $10,691.38" in text

    def test_zero_trade_run_logs_the_empty_notice(self, cli, monkeypatch, caplog):
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        assert "No backtest trades found." in caplog.text
        assert "Backtest Summary" not in caplog.text

    def test_config_echo_reports_the_effective_k(self, cli, monkeypatch, caplog):
        with caplog.at_level(logging.INFO):
            _run(monkeypatch, "--interval-discount", "0.62")
        assert "k=0.620" in caplog.text

    def test_config_echo_defaults_to_the_config_constant(self, cli, monkeypatch, caplog):
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        assert f"k={config.TIME_SERIES_INTERVAL_PROB_DISCOUNT:.3f}" in caplog.text


class _FrozenNow(datetime):
    """A datetime whose now() is 2026-10-02 21:30:05 UTC, so a read time is fixed."""

    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 10, 2, 21, 30, 5, tzinfo=UTC)


class TestStartingBalance:
    """With no --balance the run starts from what the Kalshi account is worth
    now — its cash on every shard plus Kalshi's value of its open positions,
    the figure a live run sizes its trades on — read once, through the
    production client, before the fetch; --balance gives an amount and skips
    the read. The amount reaches the sweep, the log, the summary block and the
    dashboard, and where it came from reaches the log and the dashboard. A
    balance that cannot be read, or a read that comes to nothing, stops the
    run: an amount nobody chose would size every simulated trade for some
    other account."""

    SOURCE = ("the account's value at 2026-10-02 21:30 UTC: cash $116.15 + open "
              "positions $95.27")

    @pytest.fixture(autouse=True)
    def _frozen_now(self, monkeypatch):
        monkeypatch.setattr(backtest, "datetime", _FrozenNow)

    def test_by_default_the_run_starts_from_the_account_value(self, cli, monkeypatch, caplog):
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        # One read, through the production client the fetch uses
        assert cli["balance_reads"] == [cli["live_client"]]
        assert cli["sweep_kwargs"]["live_client"] is cli["live_client"]
        # The cash on every shard plus the open positions: $116.15 + $95.27
        assert cli["sweep_kwargs"]["initial_balance"] == pytest.approx(211.42)
        args, kwargs = cli["dashboard"]
        assert args[3] == pytest.approx(211.42)
        assert kwargs["balance_source"] == self.SOURCE
        # The log names the amount and its source, then the config echo carries it
        messages = [r.getMessage() for r in caplog.records]
        at = messages.index(f"Starting balance: $211.42 ({self.SOURCE})")
        assert messages[at + 1].startswith(
            "Backtest config: start=2024-01-01 | balance=$211.42 | ")

    def test_cash_on_every_shard_counts(self, cli, monkeypatch):
        cli["account"] = AccountBalance({0: 10_000, 1: 5_050, 2: 1}, 0)
        _run(monkeypatch)
        assert cli["sweep_kwargs"]["initial_balance"] == pytest.approx(150.51)
        assert cli["dashboard"][1]["balance_source"] == (
            "the account's value at 2026-10-02 21:30 UTC: cash $150.51 + open "
            "positions $0.00")

    def test_with_no_readable_positions_value_it_starts_from_the_cash(
            self, cli, monkeypatch, caplog):
        # As a live run does: the cash alone, and a WARNING saying so
        cli["account"] = AccountBalance({0: 11_615}, None)
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        assert cli["sweep_kwargs"]["initial_balance"] == pytest.approx(116.15)
        source = ("the account's cash at 2026-10-02 21:30 UTC; its open positions' "
                  "value could not be read")
        assert cli["dashboard"][1]["balance_source"] == source
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert ("Kalshi's balance reply carried no readable portfolio_value — the "
                "backtest starts from the account's cash alone ($116.15), as if no "
                "position were held") in warnings
        assert f"Starting balance: $116.15 ({source})" in caplog.text

    def test_the_balance_flag_skips_the_read(self, cli, monkeypatch, caplog):
        with caplog.at_level(logging.INFO):
            _run(monkeypatch, "--balance", "500")
        assert cli["balance_reads"] == []
        assert cli["sweep_kwargs"]["initial_balance"] == pytest.approx(500.0)
        args, kwargs = cli["dashboard"]
        assert args[3] == pytest.approx(500.0)
        assert kwargs["balance_source"] == "set by --balance"
        assert "Starting balance: $500.00 (set by --balance)" in caplog.text
        assert "| balance=$500.00 |" in caplog.text

    def test_the_summary_is_measured_against_the_account_value(
            self, cli, monkeypatch, caplog):
        # A $5,000 account whose primary point ends at $10,691.38
        cli["account"] = AccountBalance({0: 400_000}, 100_000)
        cli["result"] = _sweep(n_trades=2)
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        assert "Total return:  +113.8%" in caplog.text
        assert "Final balance: $10,691.38" in caplog.text

    @pytest.mark.parametrize("error, reason", [
        (RuntimeError("connection refused"), "RuntimeError: connection refused"),
        (ApiException(status=401, reason="Unauthorized",
                      body='{"error": {"code": "authentication_error", '
                           '"message": "invalid signature"}}'),
         "HTTP 401 Unauthorized — authentication_error: invalid signature"),
        (ValueError("Unparseable balance payload: keys=[]"),
         "ValueError: Unparseable balance payload: keys=[]"),
    ])
    def test_a_failed_read_stops_the_run_before_the_fetch(
            self, cli, monkeypatch, caplog, error, reason):
        cli["account"] = error
        with caplog.at_level(logging.INFO), pytest.raises(SystemExit) as stopped:
            _run(monkeypatch)
        message = (f"could not read the account balance ({reason}); pass --balance "
                   "DOLLARS to choose a starting balance")
        # A SystemExit carrying text prints it to stderr and exits 1
        assert stopped.value.code == f"backtest: {message}"
        errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert errors == [f"Backtest not run: {message}"]
        # Nothing was fetched, simulated or written
        assert "sweep_kwargs" not in cli and "dashboard" not in cli
        assert "Backtest config:" not in caplog.text

    def test_an_account_worth_nothing_stops_the_run(self, cli, monkeypatch, caplog):
        cli["account"] = AccountBalance({0: 0, 1: 0}, 0)
        with caplog.at_level(logging.INFO), pytest.raises(SystemExit) as stopped:
            _run(monkeypatch)
        assert stopped.value.code == (
            "backtest: the account is worth $0.00, so there is nothing to start "
            "from; pass --balance DOLLARS to choose a starting balance")
        assert "sweep_kwargs" not in cli and "dashboard" not in cli

    def test_no_cash_beside_an_unreadable_positions_value_stops_the_run(
            self, cli, monkeypatch, caplog):
        # The account may hold positions of unknown worth, so it is never
        # called "worth $0", and the run never says it starts from the cash
        cli["account"] = AccountBalance({0: 0, 1: 0}, None)
        with caplog.at_level(logging.INFO), pytest.raises(SystemExit) as stopped:
            _run(monkeypatch)
        assert stopped.value.code == (
            "backtest: the account's cash is $0.00 and its open positions' value "
            "could not be read, so there is nothing to start from; pass --balance "
            "DOLLARS to choose a starting balance")
        assert "worth $0.00" not in caplog.text
        assert "starts from the account's cash alone" not in caplog.text
        assert "sweep_kwargs" not in cli and "dashboard" not in cli

    @pytest.mark.parametrize("argv, account, amount", [
        ((), AccountBalance({0: 900}, 300), "12.00"),
        (("--balance", "20"), None, "20.00"),
    ])
    def test_a_balance_below_the_live_minimum_draws_a_warning(
            self, cli, monkeypatch, caplog, argv, account, amount):
        # A live run does not trade below config.MIN_BALANCE_CENTS; the
        # backtest still runs, and says so
        if account is not None:
            cli["account"] = account
        with caplog.at_level(logging.INFO):
            _run(monkeypatch, *argv)
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert (f"Starting balance ${amount} is below the "
                f"${config.MIN_BALANCE_CENTS / 100:,.2f} minimum "
                "(config.MIN_BALANCE_CENTS) below which a live run does not trade; "
                "the backtest trades from it anyway") in warnings
        assert "sweep_kwargs" in cli

    def test_a_balance_at_or_above_the_live_minimum_draws_no_warning(
            self, cli, monkeypatch, caplog):
        cli["account"] = AccountBalance({0: config.MIN_BALANCE_CENTS}, 0)
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        assert "minimum (config.MIN_BALANCE_CENTS)" not in caplog.text

    @pytest.mark.parametrize("dollars", [0.0, -1.0, float("nan"), float("inf")])
    def test_a_starting_balance_is_always_a_finite_amount_above_zero(self, dollars):
        with pytest.raises(ValueError, match="a starting balance must be a finite amount"):
            backtest.StartingBalance(dollars, "set by --balance")

    @pytest.mark.parametrize("value", ["0", "-5", "nan", "inf", "-inf"])
    def test_a_balance_that_is_not_a_positive_amount_is_refused(
            self, cli, monkeypatch, capsys, value):
        with pytest.raises(SystemExit) as stopped:
            _run(monkeypatch, f"--balance={value}")
        assert stopped.value.code == 2
        assert "--balance must be a positive number of dollars" in capsys.readouterr().err
        assert cli["balance_reads"] == []
        assert "sweep_kwargs" not in cli


class TestCorpusProvenanceLine:
    """DR-13 (P2): the run's report closes, on every run, with what
    settled-market corpus it read — the Period line runs to today, the corpus
    only to its assembly, which every run brings up to date (a cache from an
    earlier UTC day is extended). The archive cutoff is information only: a
    window starting after it is priced from the live candlestick endpoint, so
    the old "structurally 0-trade" WARNING is gone."""

    PROV = CorpusProvenance(
        from_cache=True, assembled_at=datetime(2026, 9, 24, 12, 37, 49, tzinfo=UTC),
        archive_cutoff=datetime(2026, 7, 25, tzinfo=UTC))

    @pytest.mark.parametrize("n_trades", [0, 2])
    def test_the_corpus_line_closes_every_run(self, cli, monkeypatch, caplog, n_trades):
        cli["result"] = _sweep(n_trades=n_trades)
        cli["result"].corpus_provenance = self.PROV
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        assert ("Settled-market corpus: assembled 2026-09-24 12:37 UTC, holding no "
                "market settled after that (served from an earlier run's cache, "
                "assembled earlier today; --no-cache re-assembles it in full); "
                "archive cutoff at assembly: 2026-07-25") in caplog.text
        assert "structural" not in caplog.text

    def test_a_window_after_the_cutoff_draws_no_warning(self, cli, monkeypatch, caplog):
        # The run's start date (the CLI default here) is irrelevant now: a
        # cutoff after it, or before it, is reported and judged on nothing.
        cli["result"].corpus_provenance = dataclasses.replace(
            self.PROV, archive_cutoff=datetime(2030, 1, 1, tzinfo=UTC))
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        assert "archive cutoff at assembly: 2030-01-01" in caplog.text
        assert not [r for r in caplog.records if r.levelname == "WARNING"
                    and "archive cutoff" in r.getMessage()]
        assert "structural" not in caplog.text and "verdict" not in caplog.text

    @pytest.mark.parametrize("from_cache, source", [
        (False, "extended through today by this run"),
        (True, "served from an earlier run's cache, assembled earlier today; "
               "--no-cache re-assembles it in full"),
    ])
    def test_an_extended_corpus_names_its_full_assembly(
            self, cli, monkeypatch, caplog, from_cache, source):
        cli["result"].corpus_provenance = dataclasses.replace(
            self.PROV, from_cache=from_cache,
            full_assembly_at=datetime(2026, 9, 20, 8, 15, tzinfo=UTC))
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        assert ("Settled-market corpus: assembled 2026-09-24 12:37 UTC, holding no "
                "market settled after that (extended day by day since a full "
                f"assembly of 2026-09-20 08:15 UTC) ({source}); archive cutoff at "
                "assembly: 2026-07-25") in caplog.text

    def test_a_cache_that_could_not_be_brought_up_to_date_says_so(
            self, cli, monkeypatch, caplog):
        # An earlier day's cache served as it was after its extension failed:
        # never called today's
        cli["result"].corpus_provenance = dataclasses.replace(self.PROV, stale=True)
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        assert ("Settled-market corpus: assembled 2026-09-24 12:37 UTC, holding no "
                "market settled after that (served as an earlier day's cache: it could "
                "not be brought up to date this run (see the WARNING above); --no-cache "
                "re-assembles it in full); archive cutoff at assembly: 2026-07-25"
                ) in caplog.text
        assert "assembled earlier today" not in caplog.text

    def test_a_freshly_assembled_corpus_says_so(self, cli, monkeypatch, caplog):
        cli["result"].corpus_provenance = dataclasses.replace(
            self.PROV, from_cache=False, archive_cutoff=None)
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        assert ("holding no market settled after that (assembled by this run); "
                "archive cutoff at assembly: not recorded") in caplog.text

    def test_no_provenance_says_not_recorded(self, cli, monkeypatch, caplog):
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        assert ("Settled-market corpus: assembly time and archive cutoff not "
                "recorded") in caplog.text


class TestRejectedArgumentLeavesNoLogFile:
    """
    TS-20: --start-date was parsed AFTER logging.basicConfig, so a rejected
    value still created kalshi_backtest.log — or, on an existing 20 MB file,
    rotated it, evicting real history to record a run that never happened.
    """

    @staticmethod
    def _run(argv, tmp_path, monkeypatch):
        monkeypatch.setattr(backtest, "PROJECT_ROOT", tmp_path)
        # basicConfig is a no-op once the root logger has handlers, which it
        # does under pytest — force it to actually build ours.
        root = logging.getLogger()
        saved = root.handlers[:]
        root.handlers = []
        try:
            with patch.object(sys, "argv", argv), pytest.raises(SystemExit):
                backtest.main()
        finally:
            for h in root.handlers:
                h.close()
            root.handlers = saved

    def test_bad_start_date_writes_no_log_file(self, tmp_path, monkeypatch):
        self._run(["backtest", "--start-date", "not-a-date"], tmp_path, monkeypatch)
        assert not (tmp_path / "kalshi_backtest.log").exists()

    def test_bad_start_date_leaves_an_existing_log_untouched(self, tmp_path, monkeypatch):
        # GUARD, not proof: this passes pre-fix too, because a tiny file never
        # trips the 20 MB rotation threshold. The damaging case — a rejected
        # argument rotating a full log and evicting real history — needs a
        # 20 MB fixture to reproduce and is not worth one. What this pins is
        # that the existing content survives either way.
        existing = tmp_path / "kalshi_backtest.log"
        existing.write_text("real history\n")
        self._run(["backtest", "--start-date", "2026-13-99"], tmp_path, monkeypatch)
        assert existing.read_text() == "real history\n"
        assert not (tmp_path / "kalshi_backtest.log.1").exists()

    def test_bad_horizon_also_writes_no_log_file(self, tmp_path, monkeypatch):
        # GUARD: the other two argument checks already ran before basicConfig
        # and must keep doing so.
        self._run(["backtest", "--max-horizon-days", "0"], tmp_path, monkeypatch)
        assert not (tmp_path / "kalshi_backtest.log").exists()

    def test_out_of_range_spread_min_writes_no_log_file(self, tmp_path, monkeypatch):
        # PB4: the spread-band checks must run before basicConfig too.
        self._run(["backtest", "--spread-min", "1.5"], tmp_path, monkeypatch)
        assert not (tmp_path / "kalshi_backtest.log").exists()

    def test_out_of_range_spread_max_writes_no_log_file(self, tmp_path, monkeypatch):
        self._run(["backtest", "--spread-max", "-0.1"], tmp_path, monkeypatch)
        assert not (tmp_path / "kalshi_backtest.log").exists()

    def test_floor_not_less_than_ceiling_writes_no_log_file(self, tmp_path, monkeypatch):
        self._run(["backtest", "--spread-min", "0.6", "--spread-max", "0.3"],
                   tmp_path, monkeypatch)
        assert not (tmp_path / "kalshi_backtest.log").exists()

    @pytest.mark.parametrize("value", ["0", "-1", "nan"])
    def test_a_balance_that_is_not_a_positive_amount_writes_no_log_file(
            self, tmp_path, monkeypatch, value):
        self._run(["backtest", f"--balance={value}"], tmp_path, monkeypatch)
        assert not (tmp_path / "kalshi_backtest.log").exists()

    def test_one_sided_floor_at_the_default_ceiling_writes_no_log_file(
            self, tmp_path, monkeypatch):
        self._run(["backtest", "--spread-min", "1.0"], tmp_path, monkeypatch)
        assert not (tmp_path / "kalshi_backtest.log").exists()


def _saved_header_and_inputs(path) -> tuple[dict, dict]:
    """Read a saved sweep file as a rebuild does: its header line, then its pickled inputs."""
    with gzip.open(path, "rb") as handle:
        header = json.loads(handle.readline())
        inputs = pickle.load(handle)
    return header, inputs


def _rewrite_header(path, **changes) -> None:
    """Change fields of a saved sweep file's header line, keeping its pickled inputs."""
    with gzip.open(path, "rb") as handle:
        header = json.loads(handle.readline())
        rest = handle.read()
    header.update(changes)
    with gzip.open(path, "wb") as handle:
        handle.write(json.dumps(header).encode("utf-8") + b"\n" + rest)


def _changed_part(path, module: str) -> None:
    """Make a saved file's fingerprint say one simulator module's code differed."""
    with gzip.open(path, "rb") as handle:
        parts = json.loads(handle.readline())["fingerprint_parts"]
    parts[module] = "0" * 64
    _rewrite_header(path, fingerprint_parts=parts,
                    fingerprint=backtest._fingerprint_of(parts))


@pytest.fixture
def rebuild(cli, monkeypatch, tmp_path):
    """A full run's saved file, then nothing a rebuild may not do: every collaborator
    that reads the account, the market data or the corpus fails the test if called.

    Returns the cli dict, the full run's generate_dashboard call moved to
    "full_dashboard" and the saved file's path at "path".
    """
    _run(monkeypatch)
    cli["full_dashboard"] = cli.pop("dashboard")
    cli["path"] = tmp_path / config.BACKTEST_SWEEP_FILENAME

    def forbidden(name):
        def call(*args, **kwargs):
            raise AssertionError(f"--dashboard-only called {name}")
        return call

    for name in ("build_historical_client", "build_prod_live_client", "read_account_balance",
                 "run_backtest_sweep", "load_series_categories",
                 "load_series_fee_multipliers", "load_depth_model"):
        monkeypatch.setattr(backtest, name, forbidden(name))
    return cli


class TestTheFullRunSavesItsDashboardInputs:
    """Every full run saves what it hands generate_dashboard, right after the
    sweep and before the page is built, in config.BACKTEST_SWEEP_FILENAME:
    one JSON header line (format, simulator fingerprint, save time, Python,
    start date), then the inputs pickled. A save that fails never stops the
    run, and leaves no file behind — not even an earlier run's."""

    def test_the_file_holds_what_the_dashboard_is_handed(self, cli, monkeypatch, tmp_path):
        _run(monkeypatch)
        path = tmp_path / config.BACKTEST_SWEEP_FILENAME
        header, inputs = _saved_header_and_inputs(path)
        args, kwargs = cli["dashboard"]
        assert header["format"] == config.BACKTEST_SWEEP_FORMAT
        parts = dict(backtest._loaded_simulator_fingerprint())
        assert header["fingerprint_parts"] == parts
        assert header["fingerprint"] == backtest._fingerprint_of(parts)
        assert header["start_date"] == "2024-01-01"
        assert set(inputs) == backtest._DASHBOARD_INPUT_KEYS
        assert (inputs["start_date"], inputs["initial_balance"]) == (args[2], args[3])
        for key in ("interval_discount", "series_categories", "balance_source", "period_end"):
            assert inputs[key] == kwargs[key], key
        saved = inputs["sweep"]
        assert isinstance(saved, BacktestSweep)
        assert saved.primary.k == kwargs["sweep"].primary.k
        pd.testing.assert_frame_equal(saved.primary.equity_df, args[1])
        assert not path.with_name(path.name + ".tmp").exists()

    def test_the_window_ends_on_the_day_the_sweep_finished(self, cli, monkeypatch, tmp_path):
        # The page's Period line and the saved file name one day, read when
        # the sweep returns, so a later rebuild still names the run's day
        monkeypatch.setattr(backtest, "datetime", _FrozenNow)
        _run(monkeypatch)
        _, kwargs = cli["dashboard"]
        assert kwargs["period_end"] == date(2026, 10, 2)
        _, inputs = _saved_header_and_inputs(tmp_path / config.BACKTEST_SWEEP_FILENAME)
        assert inputs["period_end"] == date(2026, 10, 2)

    def test_it_is_saved_before_the_page_is_built(self, cli, monkeypatch, tmp_path):
        # So a page build that stops part way can be finished by a rebuild
        path = tmp_path / config.BACKTEST_SWEEP_FILENAME
        seen: list = []

        def dashboard(*args, **kwargs):
            seen.append(_saved_header_and_inputs(path)[1]["start_date"])
            return tmp_path / "dashboard.html"

        monkeypatch.setattr(backtest, "generate_dashboard", dashboard)
        _run(monkeypatch, "--start-date", "2025-09-24")
        assert seen == [date(2025, 9, 24)]

    def test_it_replaces_an_earlier_runs_file(self, cli, monkeypatch, tmp_path):
        path = tmp_path / config.BACKTEST_SWEEP_FILENAME
        path.write_bytes(b"an earlier run")
        _run(monkeypatch)
        assert _saved_header_and_inputs(path)[0]["format"] == config.BACKTEST_SWEEP_FORMAT

    def test_a_run_that_cannot_be_saved_still_builds_its_page(
            self, cli, monkeypatch, tmp_path, caplog):
        # A function cannot be pickled: the save fails, the run goes on, and an
        # earlier run's file goes too, since it no longer matches the page
        path = tmp_path / config.BACKTEST_SWEEP_FILENAME
        path.write_bytes(b"an earlier run")
        cli["series_categories"] = {"KXTEST": lambda: None}
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        assert "dashboard" in cli
        assert not path.exists()
        assert not path.with_name(path.name + ".tmp").exists()
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any(m.startswith("Could not save this run's dashboard inputs (")
                   and "any earlier run's saved file was removed" in m for m in warnings)

    def test_code_that_cannot_be_fingerprinted_saves_nothing(
            self, cli, monkeypatch, tmp_path, caplog):
        def unreadable():
            raise OSError("no such file")

        monkeypatch.setattr(backtest, "_loaded_simulator_fingerprint", unreadable)
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        assert "dashboard" in cli
        assert not (tmp_path / config.BACKTEST_SWEEP_FILENAME).exists()
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any(m.startswith("The simulator's code could not be fingerprinted")
                   for m in warnings)
        assert any("could not be fingerprinted when the run started" in m for m in warnings)


class TestDashboardOnly:
    """--dashboard-only rebuilds the page from the saved inputs: no fetch and
    no Kalshi request, the T-bill yields read again, generate_dashboard called
    with the run's own arguments and this command's --sell-workers."""

    def test_it_hands_generate_dashboard_what_the_run_did(self, rebuild, monkeypatch):
        _run(monkeypatch, "--dashboard-only", "--sell-workers", "3")
        args, kwargs = rebuild["dashboard"]
        full_args, full_kwargs = rebuild["full_dashboard"]
        assert args[2:] == full_args[2:]
        pd.testing.assert_frame_equal(args[1], full_args[1])
        assert args[0] is kwargs["sweep"].primary.trades
        assert set(kwargs) == set(full_kwargs)
        for key in set(kwargs) - {"sweep", "risk_free", "sell_workers"}:
            assert kwargs[key] == full_kwargs[key], key
        assert kwargs["sweep"].primary.k == full_kwargs["sweep"].primary.k
        assert kwargs["risk_free"] is rebuild["risk_free"]
        assert kwargs["sell_workers"] == 3

    def test_the_default_workers_are_the_full_runs(self, rebuild, monkeypatch):
        _run(monkeypatch, "--dashboard-only")
        assert (rebuild["dashboard"][1]["sell_workers"]
                == rebuild["full_dashboard"][1]["sell_workers"])

    def test_the_rebuild_says_what_it_read(self, rebuild, monkeypatch, caplog):
        with caplog.at_level(logging.INFO):
            _run(monkeypatch, "--dashboard-only")
        messages = [r.getMessage() for r in caplog.records]
        assert any(m.startswith(f"Dashboard rebuild from {rebuild['path']}: saved ")
                   and "by a run that started 2024-01-01" in m for m in messages)
        assert messages[-2] == "Open the HTML file in a browser to view the interactive charts."

    def test_a_missing_file_exits_2(self, cli, monkeypatch, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, "--dashboard-only")
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert f"no saved backtest at {tmp_path / config.BACKTEST_SWEEP_FILENAME}" in err
        assert "python3 -m kalshi_betting.backtest" in err
        assert "dashboard" not in cli and "sweep_kwargs" not in cli

    @pytest.mark.parametrize("damage", ["not gzip", "no header line", "header not json",
                                        "other format", "no fingerprint"])
    def test_a_file_whose_header_cannot_be_read_exits_1(self, rebuild, monkeypatch, damage):
        path = rebuild["path"]
        if damage == "not gzip":
            path.write_bytes(b"not a gzip stream")
        elif damage == "no header line":
            with gzip.open(path, "wb") as handle:
                handle.write(b"{" * 10)
        elif damage == "header not json":
            with gzip.open(path, "wb") as handle:
                handle.write(b"not json\n")
        elif damage == "other format":
            # The format a run saved before the one Rebalance family
            _rewrite_header(path, format="backtest-sweep-v1")
        else:
            _rewrite_header(path, fingerprint_parts=None)
        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, "--dashboard-only")
        # A message, so exit 1, with the reason and what to do
        assert isinstance(exc.value.code, str)
        assert exc.value.code.startswith(f"backtest: {path} cannot be used: ")
        assert exc.value.code.endswith("run the backtest again to save a new one")
        if damage == "other format":
            assert "'backtest-sweep-v1'" in exc.value.code
            assert repr(config.BACKTEST_SWEEP_FORMAT) in exc.value.code
            assert config.BACKTEST_SWEEP_FORMAT == "backtest-sweep-v3"
        assert "dashboard" not in rebuild

    @pytest.mark.parametrize("damage", ["cut short", "not the inputs"])
    def test_a_run_that_cannot_be_loaded_exits_1(self, rebuild, monkeypatch, caplog, damage):
        path = rebuild["path"]
        with gzip.open(path, "rb") as handle:
            line, rest = handle.readline(), handle.read()
        payload = (rest[: len(rest) // 2] if damage == "cut short"
                   else pickle.dumps({"sweep": 1}))
        with gzip.open(path, "wb") as handle:
            handle.write(line + payload)
        with caplog.at_level(logging.INFO), pytest.raises(SystemExit) as exc:
            _run(monkeypatch, "--dashboard-only")
        assert isinstance(exc.value.code, str) and "cannot be used" in exc.value.code
        assert any(r.levelno == logging.ERROR
                   and r.getMessage().startswith("Dashboard not rebuilt: ")
                   for r in caplog.records)
        assert "dashboard" not in rebuild

    def test_a_changed_simulator_is_refused(self, rebuild, monkeypatch, capsys):
        _changed_part(rebuild["path"], "backtester")
        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, "--dashboard-only")
        assert exc.value.code == 2
        err = " ".join(capsys.readouterr().err.split())
        assert "was simulated by other code: backtester.py changed since it was saved." in err
        assert "two different versions of the code" in err
        assert "--accept-changed-simulator" in err
        assert "Python" not in err
        assert "dashboard" not in rebuild

    def test_the_refusal_names_a_different_python(self, rebuild, monkeypatch, capsys):
        _changed_part(rebuild["path"], "config")
        _rewrite_header(rebuild["path"], python="3.0.0")
        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, "--dashboard-only")
        assert exc.value.code == 2
        err = " ".join(capsys.readouterr().err.split())
        assert "config.py changed since it was saved (under Python 3.0.0; this is Python" in err

    def test_accept_changed_simulator_builds_the_page_with_a_warning(
            self, rebuild, monkeypatch, caplog):
        _changed_part(rebuild["path"], "scanner")
        with caplog.at_level(logging.INFO):
            _run(monkeypatch, "--dashboard-only", "--accept-changed-simulator")
        assert "dashboard" in rebuild
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any(m.startswith("The simulator's code changed since this run was saved "
                                "(scanner.py)") for m in warnings)

    def test_accept_changed_simulator_alone_is_refused(self, cli, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, "--accept-changed-simulator")
        assert exc.value.code == 2
        assert "--accept-changed-simulator applies only with --dashboard-only" in (
            capsys.readouterr().err)
        assert "sweep_kwargs" not in cli

    @pytest.mark.parametrize("argv", [
        ["--start-date", "2025-01-01"], ["--balance", "100"], ["--no-cache"],
        ["--max-horizon-days", "5"], ["--interval-discount", "0.5"], ["--no-sweep"],
        ["--same-event-ladders"], ["--no-same-event-ladders"], ["--spread-min", "0.1"],
        ["--spread-max", "0.9"], ["--no-band-sweep"],
    ])
    def test_a_flag_that_shapes_a_run_is_refused(self, rebuild, monkeypatch, capsys, argv):
        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, "--dashboard-only", *argv)
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "cannot be used with --dashboard-only" in err
        assert argv[0] in err
        assert "dashboard" not in rebuild

    def test_every_option_is_a_run_flag_or_one_a_rebuild_reads(self):
        # A new option must be put on one side or the other, so --dashboard-only
        # never quietly ignores it
        dests = {a.dest for a in backtest._build_parser()._actions} - {"help"}
        run_only = {dest for dest, _ in backtest._RUN_ONLY_FLAGS}
        rebuild = {"dashboard_only", "accept_changed_simulator", "sell_workers",
                   "no_cap_sweep", "no_sell_sweep", "no_rebalance_sweep", "no_move_sweep",
                   "no_eda"}
        assert run_only | rebuild == dests
        assert not run_only & rebuild

    def test_a_bad_worker_count_is_refused(self, rebuild, monkeypatch):
        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, "--dashboard-only", "--sell-workers", "0")
        assert exc.value.code == 2

    def test_the_skip_flags_drop_their_families(self, cli, monkeypatch, tmp_path, caplog):
        # The saved sweep carries a stand-in for every family each flag drops
        cli["result"] = dataclasses.replace(
            _sweep(), sell_sweep=SimpleNamespace(), rebalance_sweep=SimpleNamespace())
        _run(monkeypatch)
        with caplog.at_level(logging.INFO):
            _run(monkeypatch, "--dashboard-only", "--no-sell-sweep", "--no-rebalance-sweep")
        sweep = cli["dashboard"][1]["sweep"]
        assert (sweep.sell_sweep, sweep.rebalance_sweep) == (None,) * 2
        assert any(r.getMessage() == (
            "Dashboard rebuilt without: Sell (--no-sell-sweep); "
            "Rebalance (--no-rebalance-sweep)")
            for r in caplog.records)


def _families() -> dict:
    """One small real family of each lazy kind, at two caps (the run's own 20%, and 10%)."""
    common = {"primary_cap": 0.2, "ks": (0.5,), "primary_k": 0.5,
              "start_date": date(2026, 1, 5), "initial_balance": 1000.0}
    band = ((0.0, 1.0),)
    lazy = {"split_date": None, "checks": False, "entries_by_band": {}, "st_entries": [],
            "eager": {}}
    lists = {"off_bands": (), "entries_by_band": {}, "off_entries_by_band": {},
             "end_dates": {}, "off_end_dates": {}}
    return {
        "cap_sweep": CapSweep(caps=(0.1, 0.2), bands=band, **lazy, **common),
        "sell_sweep": SellSweep(levels=(0.5,), min_days=(1,), caps=(0.1, 0.2), bands=band,
                                **lists, **common),
        "rebalance_sweep": RebalanceSweep(caps=(0.1, 0.2), bands=band, min_moves=(0.05, 0.1),
                                          **lists, **common),
    }


class TestWithoutFamilies:
    """_without_families drops each family its flag names and, without the
    size-cap sweep, keeps the others at the run's own cap, as a run given the
    same flags builds them; the loaded sweep itself is left as it was."""

    @staticmethod
    def _sweep():
        return dataclasses.replace(_sweep(), **_families())

    @staticmethod
    def _call(sweep, **off):
        flags = {"cap_sweep": True, "sell_sweep": True, "rebalance_sweep": True,
                 "move_sweep": True}
        flags.update(dict.fromkeys(off, False))
        return backtest._without_families(sweep, **flags)

    def test_no_flag_keeps_the_sweep(self):
        sweep = self._sweep()
        assert self._call(sweep) == (sweep, [])
        assert self._call(sweep)[0] is sweep

    def test_without_the_cap_sweep_the_others_keep_the_runs_own_cap(self):
        sweep = self._sweep()
        out, notes = self._call(sweep, cap_sweep=False)
        assert out.cap_sweep is None and out.tier_off_cap_sweep is None
        for name in ("sell_sweep", "rebalance_sweep"):
            assert getattr(out, name) == dataclasses.replace(getattr(sweep, name), caps=(0.2,))
        assert notes == ["the size caps (--no-cap-sweep)"]
        # The loaded sweep is left as it was
        assert sweep.cap_sweep.caps == (0.1, 0.2) and sweep.sell_sweep.caps == (0.1, 0.2)

    @pytest.mark.parametrize("flag, names, note", [
        ("sell_sweep", ("sell_sweep",), "Sell (--no-sell-sweep)"),
        ("rebalance_sweep", ("rebalance_sweep",), "Rebalance (--no-rebalance-sweep)"),
    ])
    def test_each_flag_drops_its_family(self, flag, names, note):
        out, notes = self._call(self._sweep(), **{flag: False})
        assert all(getattr(out, name) is None for name in names)
        assert notes == [note]

    def test_without_the_minimum_price_movements_rebalancing_stays(self):
        sweep = self._sweep()
        out, notes = self._call(sweep, move_sweep=False)
        assert out.rebalance_sweep == dataclasses.replace(sweep.rebalance_sweep, min_moves=())
        assert notes == ["Min. price movement (--no-move-sweep)"]
        # Dropping the family drops its minimums with it, named once
        out, notes = self._call(sweep, rebalance_sweep=False, move_sweep=False)
        assert out.rebalance_sweep is None and notes == ["Rebalance (--no-rebalance-sweep)"]

    def test_a_family_the_run_did_not_build_is_not_named(self):
        out, notes = self._call(_sweep(), cap_sweep=False, sell_sweep=False,
                                rebalance_sweep=False)
        assert notes == []


class TestSimulatorFingerprint:
    """The saved file's simulator fingerprint: one digest per module of
    _SIMULATOR_MODULES over its parsed code, docstrings left out, and one over
    the config.py statements those modules read, with everything they use."""

    @staticmethod
    def _sources() -> dict[str, str]:
        here = Path(backtest.__file__).parent
        return {name: (here / f"{name}.py").read_text(encoding="utf-8")
                for name in (*backtest._SIMULATOR_MODULES, "config")}

    @staticmethod
    def _changed(before: dict, after: dict) -> set[str]:
        return {name for name in before if before[name] != after[name]}

    def test_the_loaded_fingerprint_is_the_files_own(self):
        assert (dict(backtest._loaded_simulator_fingerprint())
                == backtest._simulator_fingerprint_parts(self._sources()))

    def test_it_covers_the_simulator_and_not_the_dashboard(self):
        assert backtest._SIMULATOR_MODULES == (
            "backtester", "strategy", "scanner", "depth_model", "historical")
        assert set(backtest._simulator_fingerprint_parts(self._sources())) == {
            *backtest._SIMULATOR_MODULES, "config"}

    def test_comments_docstrings_and_layout_do_not_count(self):
        sources = self._sources()
        base = backtest._simulator_fingerprint_parts(sources)
        edited = dict(sources)
        backtester_doc = "Simulate (or share) one population of one cell at every cap."
        assert backtester_doc in sources["backtester"]
        edited["backtester"] = (sources["backtester"].replace(
            backtester_doc, "Reworded, as a docstring edit would.")
            .replace("File: backtester.py", "File: backtester.py (reworded)", 1)
            + "\n\n# A comment added at the end\n")
        edited["config"] = sources["config"].replace(
            "REBALANCE_MIN_DAYS_SINCE_TRADE = 7",
            "REBALANCE_MIN_DAYS_SINCE_TRADE = (\n    7  # spread over two lines\n)")
        assert edited["config"] != sources["config"]
        assert backtest._simulator_fingerprint_parts(edited) == base

    @pytest.mark.parametrize("module", ["backtester", "strategy", "scanner", "depth_model",
                                        "historical"])
    def test_a_code_edit_changes_its_modules_part_alone(self, module):
        sources = self._sources()
        base = backtest._simulator_fingerprint_parts(sources)
        edited = {**sources, module: sources[module] + "\n_FINGERPRINT_PROBE = 1\n"}
        assert self._changed(base, backtest._simulator_fingerprint_parts(edited)) == {module}

    def test_a_config_value_the_simulator_reads_changes_the_config_part(self):
        sources = self._sources()
        base = backtest._simulator_fingerprint_parts(sources)
        edited = {**sources, "config": sources["config"].replace(
            "REBALANCE_MIN_DAYS_SINCE_TRADE = 7", "REBALANCE_MIN_DAYS_SINCE_TRADE = 8")}
        assert self._changed(base, backtest._simulator_fingerprint_parts(edited)) == {"config"}

    def test_a_config_value_reached_through_another_changes_it_too(self):
        # Read by config.rebalance_min_days, which the backtester calls; no
        # simulator module names it itself
        sources = self._sources()
        read = set().union(*(backtest._config_names_read(ast.parse(sources[name]))
                             for name in backtest._SIMULATOR_MODULES))
        assert "REBALANCE_MIN_DAYS_SINCE_TRADE_MAX" not in read
        base = backtest._simulator_fingerprint_parts(sources)
        edited = {**sources, "config": sources["config"].replace(
            "REBALANCE_MIN_DAYS_SINCE_TRADE_MAX = 30",
            "REBALANCE_MIN_DAYS_SINCE_TRADE_MAX = 31")}
        assert self._changed(base, backtest._simulator_fingerprint_parts(edited)) == {"config"}

    def test_config_the_simulator_never_reads_does_not_count(self):
        # A dashboard constant, a changed one and a new one: rebuilding the
        # page after such an edit is what the saved file is for
        sources = self._sources()
        base = backtest._simulator_fingerprint_parts(sources)
        assert "KHAT_CONFIDENCE_LEVEL: float = 0.95" in sources["config"]
        edited = {**sources, "config": sources["config"].replace(
            "KHAT_CONFIDENCE_LEVEL: float = 0.95", "KHAT_CONFIDENCE_LEVEL: float = 0.9")
            + "\nDASHBOARD_PROBE = 1\n"}
        assert backtest._simulator_fingerprint_parts(edited) == base

    def test_the_simulator_reaches_config_only_as_the_fingerprint_reads(self):
        # _config_names_read sees `from .config import NAME` and `config.NAME`
        # after `from . import config`; any other way would leave names out
        for name in backtest._SIMULATOR_MODULES:
            tree = ast.parse(self._sources()[name])
            aliases = {alias.asname or alias.name for node in ast.walk(tree)
                       if isinstance(node, ast.ImportFrom) and node.level == 1
                       and node.module is None
                       for alias in node.names if alias.name == "config"}
            parents = {child: node for node in ast.walk(tree)
                       for child in ast.iter_child_nodes(node)}
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    assert not any("config" in alias.name for alias in node.names), name
                if isinstance(node, ast.ImportFrom) and node.level == 0:
                    assert "kalshi_betting" not in (node.module or ""), name
                if isinstance(node, ast.Name) and node.id in aliases:
                    # Only ever read as config.NAME
                    parent = parents[node]
                    assert isinstance(parent, ast.Attribute) and parent.value is node, (
                        name, node.lineno)


@pytest.fixture(scope="class")
def saved_golden_run(tmp_path_factory):
    """The backtester's selling golden fixture through run_backtest_sweep with every
    lazy family on, on a narrowed grid (band 0-1, k 0.5, four sell levels, three
    minimum-days options, three minimum price movements), saved by the full run's own
    save and loaded back by the rebuild's own load, then paged twice: once from the
    run's inputs as a full run pages them, once from the loaded copy.

    The patches stay on until the class is done, so a test can run the fixture again
    with other flags. Each page is written in a folder of its own, with its Sell
    build folder renamed to one fixed name, so two pages can be compared byte for
    byte. Yields the run, the saved file, the loaded inputs, a page builder and the
    two pages (each a (page path, folder) pair).
    """
    toggles = pytest.MonkeyPatch()
    mp = pytest.MonkeyPatch()
    try:
        apply_pre_toggle_defaults(toggles)
        golden = _tb._SellingGolden()
        golden._patch(mp)
        mp.setattr(backtester, "SPREAD_BAND_SWEEP_FLOORS", (0.0,))
        mp.setattr(backtester, "SPREAD_BAND_SWEEP_CEILINGS", (1.0,))
        mp.setattr(backtester, "INTERVAL_DISCOUNT_SWEEP", (0.5,))
        mp.setattr(backtester, "TAKE_PROFIT_LEVELS", (0.05, 0.25, 0.5, 1.0))
        mp.setattr(backtester, "TAKE_PROFIT_MIN_DAYS", (1, 14, 30))
        mp.setattr(backtester, "MIN_PRICE_MOVES", (0.05, 0.1, 0.3))
        # One clock for every run of the class, so their curves end on one day
        mp.setattr(backtester, "datetime", type(
            "Clock", (_tb.TestCapSweepEndDate._Clock,),
            {"moment": datetime(2026, 9, 26, 12, 0, tzinfo=UTC)}))
        mp.setattr(dashboard.yf, "download",
                   lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
        new_build_folder = dashboard._new_build_folder

        def fixed_build_folder(page):
            folder = new_build_folder(page)
            return folder.rename(folder.with_name("build"))

        mp.setattr(dashboard, "_new_build_folder", fixed_build_folder)

        def run(**flags):
            options = {"band_sweep": True, "tier_off_sweep": True, "cap_sweep": True,
                       "sell_sweep": True, "rebalance_sweep": True, "move_sweep": True}
            options.update(flags)
            # As backtest.py sets it: the minimum price movements go with their family
            options["move_sweep"] = options["move_sweep"] and options["rebalance_sweep"]
            return backtester.run_backtest_sweep(
                MagicMock(), MagicMock(), golden._START, 10_000.0, same_event_ladders=True,
                interval_discount=0.5, **options)

        def inputs_of(sweep):
            return backtest._dashboard_inputs(
                sweep, golden._START, backtest.StartingBalance(10_000.0, "set by --balance"),
                {}, date(2026, 9, 26))

        def page(inputs, label):
            folder = tmp_path_factory.mktemp(label)
            mp.setattr(dashboard, "PROJECT_ROOT", folder)
            return backtest._build_dashboard(inputs, risk_free=None, sell_workers=1), folder

        swept = run()
        inputs = inputs_of(swept)
        path = tmp_path_factory.mktemp("saved") / config.BACKTEST_SWEEP_FILENAME
        backtest._save_dashboard_inputs(path, inputs,
                                        dict(backtest._loaded_simulator_fingerprint()))
        loaded = backtest._load_saved_inputs(path)
        yield SimpleNamespace(run=run, swept=swept, path=path, loaded=loaded,
                              inputs_of=inputs_of, page=page,
                              live_page=page(inputs, "live"), loaded_page=page(loaded, "loaded"))
    finally:
        mp.undo()
        toggles.undo()


def _page_and_sidecars(page) -> tuple[bytes, dict]:
    """A page's bytes, and each sidecar chunk file beside it by its path under the page's folder."""
    html, folder = page
    files = {str(f.relative_to(folder)): f.read_bytes()
             for f in (folder / config.DASHBOARD_FILES_DIRNAME).rglob("*.js")}
    return html.read_bytes(), files


@pytest.mark.usefixtures("saved_golden_run")
class TestRebuildFromTheSavedSweep:
    """A page rebuilt from the saved file is the page the run built: every lazy
    family survives the round trip, and each --no-*-sweep flag gives the page a
    run given that flag builds."""

    def test_every_lazy_family_is_saved(self, saved_golden_run):
        swept, loaded = saved_golden_run.swept, saved_golden_run.loaded["sweep"]
        for name in ("cap_sweep", "tier_off_cap_sweep", "sell_sweep", "rebalance_sweep"):
            assert getattr(swept, name) is not None, name
            assert type(getattr(loaded, name)) is type(getattr(swept, name)), name
        assert (loaded.rebalance_sweep.min_moves == swept.rebalance_sweep.min_moves
                == (0.05, 0.1, 0.3))
        assert loaded.sell_sweep.levels == swept.sell_sweep.levels

    def test_the_rebuilt_page_is_byte_identical(self, saved_golden_run):
        live = _page_and_sidecars(saved_golden_run.live_page)
        rebuilt = _page_and_sidecars(saved_golden_run.loaded_page)
        # Not vacuous: the Sell select wrote its chunk files
        assert live[1]
        assert rebuilt == live

    @pytest.mark.parametrize("off", [
        ("cap_sweep", "move_sweep"),
        ("sell_sweep", "rebalance_sweep"),
    ])
    def test_the_skip_flags_build_the_page_a_run_with_them_builds(self, saved_golden_run, off):
        flags = dict.fromkeys(off, False)
        ran = saved_golden_run.run(**flags)
        expected = _page_and_sidecars(saved_golden_run.page(saved_golden_run.inputs_of(ran),
                                                            "run-with-flags"))
        kept = {"cap_sweep": True, "sell_sweep": True, "rebalance_sweep": True,
                "move_sweep": True}
        kept.update(flags)
        loaded = saved_golden_run.loaded
        sweep, dropped = backtest._without_families(loaded["sweep"], **kept)
        assert dropped
        rebuilt = _page_and_sidecars(saved_golden_run.page({**loaded, "sweep": sweep},
                                                           "rebuilt-with-flags"))
        assert rebuilt == expected

    def test_the_file_holds_only_the_backtests_data(self, saved_golden_run):
        # No client, credential or other live object is pickled: every class
        # the file names is the backtest's own, or plain data
        named: set[str] = set()

        class Recording(pickle.Unpickler):
            def find_class(self, module, name):
                named.add(module)
                return super().find_class(module, name)

        with gzip.open(saved_golden_run.path, "rb") as handle:
            handle.readline()
            Recording(handle).load()
        allowed = ("kalshi_betting.backtester", "kalshi_betting.depth_model",
                   "kalshi_betting.historical", "builtins", "datetime", "numpy", "pandas",
                   "copyreg", "collections", "_codecs")
        assert "kalshi_betting.backtester" in named
        assert all(any(module == prefix or module.startswith(prefix + ".") for prefix in allowed)
                   for module in named), sorted(named)
