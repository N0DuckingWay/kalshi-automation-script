"""Tests for scheduler.py's run_job() exit-code mapping (BS-14).

run_job spawns kalshi_betting.main as a subprocess and used to log only
"completed successfully" / a generic failure based on returncode == 0. It now
maps the shared EXIT_* contract from config.py to a distinct log level and
message per outcome, so a low-balance skip or a manual-review run is visible
in the scheduler's own log stream, not just buried inside kalshi_arb.log
(which this process never reads). subprocess.run is mocked — tests run
offline and never spawn the real bot subprocess.
"""
import logging
from types import SimpleNamespace
from unittest.mock import patch

from kalshi_betting import scheduler
from kalshi_betting.config import (
    EXIT_OK,
    EXIT_SKIPPED_LOW_BALANCE,
    EXIT_TRADES_NEED_ATTENTION,
)


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> SimpleNamespace:
    """Stand-in for subprocess.CompletedProcess, the shape run_job reads."""
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


class TestRunJobExitCodeMapping:
    @patch("kalshi_betting.scheduler.subprocess.run")
    def test_ok_code_logs_info(self, mock_run, caplog):
        mock_run.return_value = _completed(EXIT_OK)

        with caplog.at_level(logging.INFO):
            scheduler.run_job()

        matches = [r for r in caplog.records if "Job completed successfully." in r.getMessage()]
        assert len(matches) == 1
        assert matches[0].levelno == logging.INFO

    @patch("kalshi_betting.scheduler.subprocess.run")
    def test_skipped_low_balance_logs_warning(self, mock_run, caplog):
        mock_run.return_value = _completed(EXIT_SKIPPED_LOW_BALANCE)

        with caplog.at_level(logging.INFO):
            scheduler.run_job()

        matches = [r for r in caplog.records if "balance below minimum" in r.getMessage()]
        assert len(matches) == 1
        assert matches[0].levelno == logging.WARNING

    @patch("kalshi_betting.scheduler.subprocess.run")
    def test_trades_need_attention_logs_error(self, mock_run, caplog):
        mock_run.return_value = _completed(EXIT_TRADES_NEED_ATTENTION)

        with caplog.at_level(logging.INFO):
            scheduler.run_job()

        matches = [r for r in caplog.records if "MANUAL REVIEW" in r.getMessage()]
        assert len(matches) == 1
        assert matches[0].levelno == logging.ERROR
        # Points the operator at where the detail actually lives.
        assert "kalshi_arb.log" in matches[0].getMessage()
        assert "trade_log.xlsx" in matches[0].getMessage()

    @patch("kalshi_betting.scheduler.subprocess.run")
    def test_other_nonzero_code_logs_existing_failure_path(self, mock_run, caplog):
        mock_run.return_value = _completed(1, stderr="Traceback: boom")

        with caplog.at_level(logging.INFO):
            scheduler.run_job()

        matches = [r for r in caplog.records if "Job failed (exit 1)" in r.getMessage()]
        assert len(matches) == 1
        assert matches[0].levelno == logging.ERROR
        assert "Traceback: boom" in matches[0].getMessage()

    @patch("kalshi_betting.scheduler.subprocess.run")
    def test_stdout_still_logged_regardless_of_code(self, mock_run, caplog):
        mock_run.return_value = _completed(EXIT_OK, stdout="scan output here")

        with caplog.at_level(logging.INFO):
            scheduler.run_job()

        assert any("scan output here" in r.getMessage() for r in caplog.records)
