"""Tests for main.py's process exit-code contract (BS-14).

_run_prod / _run_dev return an int outcome code, and main() propagates it to
the OS via sys.exit() so the scheduler (a separate subprocess, see
scheduler.run_job) can distinguish a clean run from a low-balance skip or a
run whose trades need manual review. All heavy collaborators (auth, scanner,
strategy, trader, reporter) are mocked at their main-module import sites per
project policy — tests run offline and never touch the real Kalshi API or
the real kalshi_arb.log / trade_log.xlsx files.
"""
import logging
import logging.handlers
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from kalshi_betting import main
from kalshi_betting.config import (
    EXIT_OK,
    EXIT_SKIPPED_LOW_BALANCE,
    EXIT_TRADES_NEED_ATTENTION,
    MIN_BALANCE_CENTS,
)
from kalshi_betting.reporter import TradeResult


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
    pair = SimpleNamespace(
        pair_type="time_series",
        market_a=SimpleNamespace(ticker="TICK-A", title="Market A", subtitle="", close_time=None),
        market_b=SimpleNamespace(ticker="TICK-B", title="Market B", subtitle="", close_time=None),
        pA=0.60,
        pB=0.30,
        nA=0.40,
        tradeable=True,
        canonical_title="Test pair",
    )
    return SimpleNamespace(
        pair=pair,
        x=5,
        y=5,
        total_cost=2.0,
        min_payoff=0.50,
        profit_ratio=0.10,
        monthly_profit_ratio=0.20,
        kelly_fraction=0.10,
        kelly_p=0.60,
    )


class TestRunProdExitCodes:
    @patch("kalshi_betting.main.verify_auth")
    def test_low_balance_returns_skip_code(self, mock_verify_auth):
        # Balance below MIN_BALANCE_CENTS must short-circuit before any scan —
        # the bare `return` this used to be silently exited 0.
        mock_verify_auth.return_value = MIN_BALANCE_CENTS - 1
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
    @patch("kalshi_betting.main.fetch_open_events_with_markets")
    @patch("kalshi_betting.main.get_held_tickers")
    @patch("kalshi_betting.main.verify_auth")
    def test_manual_review_result_returns_attention_code(
        self,
        mock_verify_auth,
        mock_held,
        mock_fetch,
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
        mock_verify_auth.side_effect = [100_000, 100_000]
        mock_held.return_value = set()
        mock_fetch.return_value = []
        mock_filter_horizon.side_effect = lambda markets, days: markets
        mock_find_ts.return_value = []
        spec = make_spec()
        mock_find_st.return_value = [spec.pair]
        mock_enrich.return_value = [spec.pair]
        mock_compute.return_value = spec
        mock_select.return_value = [spec]
        mock_pre_exec.side_effect = lambda client, portfolio: portfolio
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
    @patch("kalshi_betting.main.fetch_open_events_with_markets")
    @patch("kalshi_betting.main.get_held_tickers")
    @patch("kalshi_betting.main.verify_auth")
    def test_clean_dry_run_returns_ok_code(
        self,
        mock_verify_auth,
        mock_held,
        mock_fetch,
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
        mock_verify_auth.side_effect = [100_000, 100_000]
        mock_held.return_value = set()
        mock_fetch.return_value = []
        mock_filter_horizon.side_effect = lambda markets, days: markets
        mock_find_ts.return_value = []
        spec = make_spec()
        mock_find_st.return_value = [spec.pair]
        mock_enrich.return_value = [spec.pair]
        mock_compute.return_value = spec
        mock_select.return_value = [spec]
        mock_pre_exec.side_effect = lambda client, portfolio: portfolio
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
        mock_verify_auth.return_value = 100_000
        with (
            patch("kalshi_betting.main.get_held_tickers", return_value=set()),
            patch("kalshi_betting.main.fetch_open_events_with_markets", return_value=[]),
            patch("kalshi_betting.main.filter_markets_within_horizon", side_effect=lambda m, d: m),
            patch("kalshi_betting.main.find_time_series_pairs", return_value=[]),
            patch("kalshi_betting.main.find_same_title_pairs", return_value=[]),
        ):
            code = main._run_prod(MagicMock(), _args())

        assert code == EXIT_OK


class TestRunDevExitCode:
    def test_run_dev_returns_ok_code(self):
        client = MagicMock()
        with (
            patch("kalshi_betting.main.fetch_open_events_with_markets", return_value=[]),
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
    @patch("kalshi_betting.main.fetch_open_events_with_markets")
    @patch("kalshi_betting.main.build_client")
    def test_main_dev_mode_exits_ok(
        self,
        mock_build_client,
        mock_fetch,
        mock_filter_horizon,
        mock_find_ts,
        mock_find_st,
        mock_enrich,
        mock_write_sim,
        tmp_path,
        monkeypatch,
    ):
        mock_build_client.return_value = MagicMock()
        mock_fetch.return_value = []
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
        mock_verify_auth.return_value = MIN_BALANCE_CENTS - 1

        monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(sys, "argv", ["kalshi_betting.main", "--mode", "prod"])

        with pytest.raises(SystemExit) as exc_info:
            main.main()

        assert exc_info.value.code == EXIT_SKIPPED_LOW_BALANCE
        assert exc_info.value.code == 10


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
            patch("kalshi_betting.main.fetch_open_events_with_markets", return_value=[]),
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
            patch("kalshi_betting.main.fetch_open_events_with_markets", return_value=[]),
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
            patch("kalshi_betting.main.fetch_open_events_with_markets", return_value=[]),
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
    unbounded. logging.basicConfig() is a no-op once the root logger already
    has handlers (pytest installs its own), so this test clears the root
    logger first to actually exercise main()'s handler configuration.
    """

    def test_main_configures_rotating_file_handler(self, tmp_path, monkeypatch):
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
                patch("kalshi_betting.main.fetch_open_events_with_markets", return_value=[]),
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
            handler = file_handlers[0]
            assert handler.maxBytes == 5 * 1024 * 1024
            assert handler.backupCount == 3
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


def test_exit_code_constants_distinct():
    # Guard against a future accidental collision between the three codes —
    # the scheduler's log-level mapping depends on them being distinguishable.
    assert len({EXIT_OK, EXIT_SKIPPED_LOW_BALANCE, EXIT_TRADES_NEED_ATTENTION}) == 3
