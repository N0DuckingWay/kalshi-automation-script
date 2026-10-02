"""Tests for backtest.py — the backtester's CLI argument surface.

Covers the two interval-discount flags added with the calibration sweep:
argparse accepts a valid --interval-discount, main() rejects an out-of-range
one through parser.error(), and both --interval-discount and --no-sweep are
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

And --no-add-on-sweep: the dashboard's Add to held pairs family is on by
default (add_on_sweep=True), the flag threads add_on_sweep=False, and the echo
line names the setting right after the cap sweep's. --no-sell-sweep does the
same for the Sell family (sell_sweep), named right after the add-on sweep.

And the echo's "live rule=" clause: the saved live defaults' rule (with their
origin), "none saved" with no file, "not recorded" with a refused one — a read
of its own, never a scenario of this run and never config.py's toggles. And the
run's last line, after the one pointing at the dashboard: how the filter bar's
scenario becomes the live defaults (the defaults server, then the page's save
button).

And the starting balance: with no --balance the run starts from the account's
value (its cash on every shard plus Kalshi's value of its open positions, or
the cash alone with a WARNING when that value is unreadable), read once through
the production live client before the fetch; --balance skips the read; a
balance that is not a positive number is refused before logging is
configured; a read that fails, or comes to nothing, stops the run before the
fetch; and a balance below config.MIN_BALANCE_CENTS draws a WARNING. The
amount reaches the sweep, the log, the summary block and the dashboard, and
where it came from reaches the log and the dashboard.

Fully offline: run_backtest_sweep, generate_dashboard, both client builders,
read_account_balance and load_risk_free_rates are monkeypatched, so no network
call, no credential read and no real backtest happen. PROJECT_ROOT is redirected at tmp_path and
logging.basicConfig is stubbed, so the run's RotatingFileHandler can neither
write into the repo root nor leak a handler onto the root logger for the rest
of the session.
"""
import dataclasses
import logging
import sys
from datetime import UTC, date, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from kalshi_python_sync.exceptions import ApiException

from kalshi_betting import backtest, config
from kalshi_betting.auth import AccountBalance
from kalshi_betting.backtester import BacktestSweep, CorpusProvenance, SweepPoint
from kalshi_betting.config import (
    MAX_DEADLINE_GAP_DAYS,
    MIN_PRICE_DIFF_LONG_GAP,
    MIN_PRICE_DIFF_SHORT_GAP,
    PRICE_EPSILON,
    SHORT_DEADLINE_GAP_DAYS,
    min_price_diff_for_gap,
    time_series_spread_too_wide,
)
from kalshi_betting.treasury import SOURCE_API, RiskFreeRates

from .conftest import save_config_live_defaults


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
        return calls["result"]

    def _fake_dashboard(*args, **kwargs):
        calls["dashboard"] = (args, kwargs)
        return tmp_path / "dashboard.html"

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
    # Never read the real backtest_cache or the network for series categories
    calls["series_categories"] = {"KXTEST": ("Sports", ("Basketball",))}
    monkeypatch.setattr(backtest, "load_series_categories",
                        lambda client: calls["series_categories"])
    # Never read the real backtest_cache or the network for the risk-free rate
    calls["risk_free"] = RiskFreeRates(
        ((date(2026, 1, 5), 0.04),), SOURCE_API, datetime(2026, 9, 27, tzinfo=UTC),
    )
    monkeypatch.setattr(backtest, "load_risk_free_rates", lambda: calls["risk_free"])
    return calls


def _run(monkeypatch, *argv: str) -> None:
    """Invoke main() with the given CLI arguments."""
    monkeypatch.setattr(sys, "argv", ["backtest", *argv])
    backtest.main()


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

    @pytest.mark.parametrize("value", ["0.0", "1.0", "0.5"])
    def test_boundary_values_are_accepted(self, cli, monkeypatch, value):
        _run(monkeypatch, "--interval-discount", value)
        assert cli["sweep_kwargs"]["interval_discount"] == pytest.approx(float(value))

    @pytest.mark.parametrize("value", ["1.5", "-0.1", "42"])
    def test_out_of_range_value_errors(self, cli, monkeypatch, capsys, value):
        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, "--interval-discount", value)
        assert exc.value.code == 2
        assert "--interval-discount must be between 0 and 1" in capsys.readouterr().err
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


class TestAddOnSweepArgument:
    """--no-add-on-sweep: the "Add to held pairs" family is ON by default, like
    the cap sweep, and the flag threads add_on_sweep=False into
    run_backtest_sweep."""

    def test_add_on_sweep_is_on_by_default(self, cli, monkeypatch):
        _run(monkeypatch)
        assert cli["sweep_kwargs"]["add_on_sweep"] is True

    def test_no_add_on_sweep_turns_it_off(self, cli, monkeypatch):
        _run(monkeypatch, "--no-add-on-sweep")
        assert cli["sweep_kwargs"]["add_on_sweep"] is False

    def test_it_is_independent_of_the_other_sweeps(self, cli, monkeypatch):
        _run(monkeypatch, "--no-band-sweep", "--no-cap-sweep", "--no-sweep")
        kwargs = cli["sweep_kwargs"]
        assert (kwargs["add_on_sweep"], kwargs["band_sweep"], kwargs["cap_sweep"],
                kwargs["sweep"]) == (True, False, False, False)
        _run(monkeypatch, "--no-add-on-sweep")
        kwargs = cli["sweep_kwargs"]
        assert (kwargs["add_on_sweep"], kwargs["band_sweep"], kwargs["cap_sweep"],
                kwargs["sweep"]) == (False, True, True, True)

    def test_the_echo_names_the_setting_after_the_cap_sweep(self, cli, monkeypatch, caplog):
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        # The band and cap sweeps' substrings other tests read stay a prefix
        assert "| band sweep=on | cap sweep=on | add-on sweep=on | sell sweep=on (" \
            in caplog.text
        caplog.clear()
        with caplog.at_level(logging.INFO):
            _run(monkeypatch, "--no-add-on-sweep", "--no-cap-sweep", "--sell-workers", "1")
        assert ("| cap sweep=off | add-on sweep=off | sell sweep=on (1 worker process) "
                "| live rule=") in caplog.text


class TestSellSweepArgument:
    """--no-sell-sweep: the dashboard's Sell family is ON by default, like the
    add-on family, and the flag threads sell_sweep=False into
    run_backtest_sweep; the echo names the setting after the add-on sweep's."""

    def test_sell_sweep_is_on_by_default(self, cli, monkeypatch):
        _run(monkeypatch)
        assert cli["sweep_kwargs"]["sell_sweep"] is True

    def test_no_sell_sweep_turns_only_it_off(self, cli, monkeypatch, caplog):
        with caplog.at_level(logging.INFO):
            _run(monkeypatch, "--no-sell-sweep")
        kwargs = cli["sweep_kwargs"]
        assert (kwargs["sell_sweep"], kwargs["add_on_sweep"], kwargs["band_sweep"],
                kwargs["cap_sweep"]) == (False, True, True, True)
        assert "| add-on sweep=on | sell sweep=off | live rule=" in caplog.text


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
        assert "| sell sweep=on (3 worker processes) | live rule=" in caplog.text

    @pytest.mark.parametrize("value", ["0", "-2"])
    def test_a_count_below_one_is_refused(self, cli, monkeypatch, capsys, value):
        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, "--sell-workers", value)
        assert exc.value.code == 2
        assert "--sell-workers must be a positive integer" in capsys.readouterr().err
        assert "sweep_kwargs" not in cli


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
                           TRADE_CATEGORIES=("Economics", "Sports"), TRADE_TAGS=("Fed",))
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        rule = config.describe_time_series_rule(True, (0.0, 1.0))
        assert (f"| live rule={rule}; category/tag filter "
                f"({config.describe_trade_filter(saved)}) ({saved.origin})") in caplog.text
        assert "categories Economics, Sports; tags Fed" in caplog.text

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
        _run(monkeypatch)
        _, kwargs = cli["dashboard"]
        assert kwargs["series_categories"] is cli["series_categories"]

    def test_the_risk_free_rates_are_passed_through(self, cli, monkeypatch):
        _run(monkeypatch)
        _, kwargs = cli["dashboard"]
        assert kwargs["risk_free"] is cli["risk_free"]

    def test_the_run_closes_with_how_to_save_the_live_defaults(self, cli, monkeypatch, caplog):
        # After pointing at the page, the run says how its filter bar's
        # scenario becomes the live defaults, or is traded: the launcher,
        # then the page's buttons (the page itself cannot write a file)
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        opened = messages.index("Open the HTML file in a browser to view the interactive "
                                "charts.")
        assert messages[opened + 1] == (
            "To save the filter bar's scenario as the live defaults, or to trade: run "
            "./start_dashboard.sh (it starts the defaults server and opens this page), then "
            "use the filter bar's Save as live defaults… or Trade using defaults… button.")
        assert messages[opened + 1:] == [messages[opened + 1]]


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
