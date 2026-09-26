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
and --no-band-sweep threads band_sweep=False into run_backtest_sweep().

Fully offline: run_backtest_sweep, generate_dashboard and both client builders
are monkeypatched, so no network call, no credential read and no real backtest
happen. PROJECT_ROOT is redirected at tmp_path and logging.basicConfig is
stubbed, so the run's RotatingFileHandler can neither write into the repo root
nor leak a handler onto the root logger for the rest of the session.
"""
import dataclasses
import logging
import sys
from datetime import UTC, date, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from kalshi_betting import backtest, config
from kalshi_betting.backtester import BacktestSweep, CorpusProvenance, SweepPoint
from kalshi_betting.config import (
    MAX_DEADLINE_GAP_DAYS,
    MIN_PRICE_DIFF_LONG_GAP,
    MIN_PRICE_DIFF_SHORT_GAP,
    PRICE_EPSILON,
    SHORT_DEADLINE_GAP_DAYS,
    TIME_SERIES_INTERVAL_PROB_DISCOUNT,
    min_price_diff_for_gap,
    time_series_spread_too_wide,
)


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
    # Never read the real backtest_cache or the network for series categories
    calls["series_categories"] = {"KXTEST": ("Sports", ("Basketball",))}
    monkeypatch.setattr(backtest, "load_series_categories",
                        lambda client: calls["series_categories"])
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
        with caplog.at_level(logging.INFO):
            _run(monkeypatch, "--interval-discount", "0.62", "--same-event-ladders")
        text = caplog.text
        assert "k=0.620" in text
        assert "ladders=on" in text
        assert "spread band=0-1" in text


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

    def test_the_series_categories_are_passed_through(self, cli, monkeypatch):
        _run(monkeypatch)
        _, kwargs = cli["dashboard"]
        assert kwargs["series_categories"] is cli["series_categories"]


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


class TestCorpusProvenanceLine:
    """DR-13 / M2 (P2): the run's report closes, on every run, with what
    settled-market corpus it read — the Period line runs to today, the corpus
    only to its assembly — and a post-cutoff window's zero is called
    structural beside the result, not only at the top of a long log, unless
    the run's own trades prove that stamped verdict stale."""

    PROV = CorpusProvenance(
        from_cache=True, assembled_at=datetime(2026, 9, 24, 12, 37, 49, tzinfo=UTC),
        archive_cutoff=datetime(2026, 7, 25, tzinfo=UTC), post_cutoff=False)

    @pytest.mark.parametrize("n_trades", [0, 2])
    def test_the_corpus_line_closes_every_run(self, cli, monkeypatch, caplog, n_trades):
        cli["result"] = _sweep(n_trades=n_trades)
        cli["result"].corpus_provenance = self.PROV
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        assert ("Settled-market corpus: assembled 2026-09-24 12:37 UTC, holding no "
                "market settled after that (served from an earlier run's cache; "
                "--no-cache extends it); archive cutoff at assembly: 2026-07-25"
                ) in caplog.text
        assert "structural" not in caplog.text

    def test_a_post_cutoff_window_is_called_structural(self, cli, monkeypatch, caplog):
        cli["result"].corpus_provenance = dataclasses.replace(self.PROV, post_cutoff=True)
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        warned = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert any("starts at or after the archive cutoff as of its corpus's assembly"
                   in m and "no trade could be entered whatever pairs formed" in m
                   and "--no-cache re-checks it" in m for m in warned)
        assert not any("that verdict is stale" in m for m in warned)

    def test_a_post_cutoff_verdict_contradicted_by_trades_is_reported_as_stale(
            self, cli, monkeypatch, caplog):
        # P2 review (R3/C3/ADV-3): after a trade summary, "no trade could be
        # entered" would be false on its face. The same helper the dashboard
        # header reads (backtester.max_trades_simulated) turns it into a
        # stale-verdict WARNING, so the page and the log agree.
        cli["result"] = _sweep(n_trades=2)
        cli["result"].corpus_provenance = dataclasses.replace(self.PROV, post_cutoff=True)
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        warned = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert not any("no trade could be entered whatever pairs formed" in m
                       for m in warned)
        assert any("The archive cutoff recorded at this corpus's assembly is at or "
                   "after this window's start date" in m
                   and "entered trades (up to 2 in one simulated scenario), so that "
                   "verdict is stale" in m and "--no-cache re-reads the cutoff" in m
                   for m in warned)

    def test_a_legacy_cache_names_its_file_time(self, cli, monkeypatch, caplog):
        # P2 review (C1/ADV-1): a legacy .json hit carries its file time here
        # too, named as such, and claims no cutoff.
        cli["result"].corpus_provenance = CorpusProvenance(
            from_cache=True, assembled_at=datetime(2026, 8, 3, 19, 5, tzinfo=UTC),
            archive_cutoff=None, post_cutoff=None, legacy=True)
        with caplog.at_level(logging.INFO):
            _run(monkeypatch)
        assert ("Settled-market corpus: last written 2026-08-03 19:05 UTC (a legacy "
                "cache's file time), holding no market settled after that (served "
                "from an earlier run's cache; --no-cache extends it); archive cutoff "
                "at assembly: not recorded (the legacy format records none)"
                ) in caplog.text
        assert not [r for r in caplog.records if r.levelname == "WARNING"
                    and "archive cutoff" in r.getMessage()]

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

    def test_one_sided_floor_at_the_default_ceiling_writes_no_log_file(
            self, tmp_path, monkeypatch):
        self._run(["backtest", "--spread-min", "1.0"], tmp_path, monkeypatch)
        assert not (tmp_path / "kalshi_backtest.log").exists()
