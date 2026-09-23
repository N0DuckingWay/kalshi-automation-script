"""Tests for backtest.py — the backtester's CLI argument surface.

Covers the two interval-discount flags added with the calibration sweep:
argparse accepts a valid --interval-discount, main() rejects an out-of-range
one through parser.error(), and both --interval-discount and --no-sweep are
threaded into backtester.run_backtest_sweep() (the same threading-assertion
idiom test_backtester.py uses for --max-horizon-days).

Fully offline: run_backtest_sweep, generate_dashboard and both client builders
are monkeypatched, so no network call, no credential read and no real backtest
happen. PROJECT_ROOT is redirected at tmp_path and logging.basicConfig is
stubbed, so the run's RotatingFileHandler can neither write into the repo root
nor leak a handler onto the root logger for the rest of the session.
"""
import logging
import sys
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from kalshi_betting import backtest
from kalshi_betting.backtester import BacktestSweep, SweepPoint
from kalshi_betting.config import TIME_SERIES_INTERVAL_PROB_DISCOUNT


def _equity(final_value: float = 10_691.38) -> pd.DataFrame:
    """A two-row equity curve in _build_equity_curve's shape."""
    df = pd.DataFrame({
        "date": [date(2026, 1, 5), date(2026, 1, 12)],
        "portfolio_value": [10_000.0, final_value],
    })
    df["daily_return"] = df["portfolio_value"].pct_change().fillna(0.0)
    return df


def _sweep(k: float = TIME_SERIES_INTERVAL_PROB_DISCOUNT, n_trades: int = 0) -> BacktestSweep:
    """A BacktestSweep whose primary point carries n_trades profitable trades.

    main() reads only `.profit` off each trade (the win-rate count), so a plain
    SimpleNamespace stands in for BacktestTrade without duplicating its fixture.
    """
    point = SweepPoint(
        k=k,
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
    identity rather than equality.
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
    monkeypatch.setattr(backtest, "build_prod_live_client", lambda: MagicMock())
    monkeypatch.setattr(backtest, "run_backtest_sweep", _fake_sweep)
    monkeypatch.setattr(backtest, "generate_dashboard", _fake_dashboard)
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

    def test_the_config_echo_names_the_effective_setting(self, cli, monkeypatch, caplog):
        # Echoed BEFORE the fetch, beside k, so an operator can abort a
        # multi-hour run configured the wrong way round.
        with caplog.at_level(logging.INFO):
            _run(monkeypatch, "--same-event-ladders")
        assert "ladders=on" in caplog.text

    def test_the_config_echo_defaults_to_the_config_constant(self, cli, monkeypatch,
                                                             caplog):
        # Patched to the NON-shipped value on purpose: with the constant set
        # to its own default (False) this row passes for any implementation
        # that ignores it entirely, including `bool(args.same_event_ladders)`
        # — and would then print "off" on a genuinely ON run the moment the
        # switch is flipped, which is exactly what this echo exists to catch.
        # The module attribute is the seam, not config's: backtest.py binds
        # the constant by value at import.
        monkeypatch.setattr(backtest, "TIME_SERIES_SAME_EVENT_LADDERS", True)
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        assert "ladders=on" in caplog.text


class TestDashboardHandoff:
    """generate_dashboard gets the whole sweep plus the RESOLVED primary k."""

    def test_sweep_and_primary_k_are_passed(self, cli, monkeypatch):
        _run(monkeypatch, "--interval-discount", "0.62")
        args, kwargs = cli["dashboard"]
        result = cli["result"]
        # The four positional arguments are unchanged from before the sweep
        assert args[0] is result.primary.trades
        assert args[1] is result.primary.equity_df
        assert args[2] == date.fromisoformat("2024-01-01")
        assert args[3] == pytest.approx(10_000.0)
        # ...and the sweep travels whole rather than unpacked
        assert kwargs["sweep"] is result
        # The k the plotted trades were SIZED at — read back off the point, not
        # re-derived from the CLI flag, so the two can never disagree
        assert kwargs["interval_discount"] == result.primary.k


class TestSummaryBlock:
    """A default run's summary block is unchanged: it reports the PRIMARY
    point, exactly as the plain run_backtest() path used to."""

    def test_summary_reports_the_primary_point(self, cli, monkeypatch, caplog):
        cli["result"] = _sweep(n_trades=2)
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
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
        assert f"k={TIME_SERIES_INTERVAL_PROB_DISCOUNT:.3f}" in caplog.text


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
