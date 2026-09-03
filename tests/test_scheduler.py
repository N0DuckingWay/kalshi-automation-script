"""Tests for scheduler.py's exit-code mapping (BS-14), TimeoutExpired/OSError
handling (BS-16, BS-31), and missed-run catch-up (BS-17).

run_job spawns kalshi_betting.main as a subprocess and used to log only
"completed successfully" / a generic failure based on returncode == 0. It now
maps the shared EXIT_* contract from config.py to a distinct log level and
message per outcome, so a low-balance skip or a manual-review run is visible
in the scheduler's own log stream, not just buried inside kalshi_arb.log
(which this process never reads). subprocess.run is mocked — tests run
offline and never spawn the real bot subprocess.

BS-17 gave run_job() a side effect: it now claims and finalizes its Monday
09:00 slot in scheduler_state.json (PROJECT_ROOT / "scheduler_state.json").
The `_tmp_project_root` fixture below is applied to every test in this file
(autouse) so those writes land under pytest's tmp_path instead of the real
repo root.
"""
import json
import logging
import subprocess
import sys
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from kalshi_betting import scheduler
from kalshi_betting.config import (
    EXIT_OK,
    EXIT_SKIPPED_LOW_BALANCE,
    EXIT_TRADES_NEED_ATTENTION,
    SCHEDULER_JOB_TIMEOUT_SECONDS,
)


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> SimpleNamespace:
    """Stand-in for subprocess.CompletedProcess, the shape run_job reads."""
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


@pytest.fixture(autouse=True)
def _tmp_project_root(tmp_path, monkeypatch):
    """
    Redirect scheduler.PROJECT_ROOT to a pytest tmp_path for every test in
    this module.

    run_job() now writes scheduler_state.json under PROJECT_ROOT on every
    invocation (BS-17); without this, the pre-existing exit-code-mapping
    tests would write into the real repo root. _state_file_path() reads
    PROJECT_ROOT from the module namespace at call time (not a frozen
    module-level constant), so this patch is picked up by every read/write.
    """
    monkeypatch.setattr(scheduler, "PROJECT_ROOT", tmp_path)
    return tmp_path


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


class TestDecode:
    """_decode() normalizes TimeoutExpired's bytes-despite-text=True streams."""

    def test_bytes_are_decoded(self):
        assert scheduler._decode(b"hello\nworld") == "hello\nworld"

    def test_str_passes_through(self):
        assert scheduler._decode("already text") == "already text"

    def test_none_becomes_empty_string(self):
        assert scheduler._decode(None) == ""

    def test_invalid_bytes_do_not_raise(self):
        # errors="replace" — must not raise on undecodable bytes.
        assert scheduler._decode(b"\xff\xfe") != ""


class TestTimeoutExpiredHandling:
    """BS-16: TimeoutExpired.stdout/.stderr are bytes; both must be logged."""

    @patch("kalshi_betting.scheduler.subprocess.run")
    def test_bytes_streams_logged_as_decoded_text(self, mock_run, caplog):
        # Reproduces the documented CPython quirk: bytes despite text=True.
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd=["python", "-m", "kalshi_betting.main"],
            timeout=3600,
            output=b"partial stdout line\nsecond line",
            stderr=b"traceback goes here\nsecond traceback line",
        )

        with caplog.at_level(logging.INFO):
            scheduler.run_job()

        full_log = "\n".join(r.getMessage() for r in caplog.records)
        assert "partial stdout line" in full_log
        assert "second line" in full_log
        assert "traceback goes here" in full_log
        assert "second traceback line" in full_log
        # The bug this fixes: a bytes repr renders as b'...' with literal \n.
        assert "b'" not in full_log
        assert "\\n" not in full_log

    @patch("kalshi_betting.scheduler.subprocess.run")
    def test_none_streams_do_not_crash(self, mock_run, caplog):
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd=["python"], timeout=3600, output=None, stderr=None,
        )

        with caplog.at_level(logging.INFO):
            scheduler.run_job()  # must not raise

        assert any("timeout" in r.getMessage() for r in caplog.records)

    @patch("kalshi_betting.scheduler.subprocess.run")
    def test_timeout_finalizes_state_with_no_exit_code(self, mock_run, tmp_path):
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd=["python"], timeout=3600, output=b"", stderr=b"",
        )

        scheduler.run_job()

        state = json.loads((tmp_path / "scheduler_state.json").read_text())
        # Sentinel choice (documented in scheduler.py): no subprocess exit
        # code exists on a timeout, so exit_code stays None while
        # finished_at is set — distinguishes "attempted and ended" from
        # "claimed, still running".
        assert state["exit_code"] is None
        assert state["finished_at"] is not None
        assert state["started_at"] is not None


class TestOSErrorHandling:
    """BS-31: a bare OSError from subprocess.run() gets a specific log and
    does not escape run_job() to be swallowed by the generic tick handler."""

    @patch("kalshi_betting.scheduler.subprocess.run")
    def test_oserror_logs_specific_message_and_does_not_raise(self, mock_run, caplog):
        mock_run.side_effect = OSError("[Errno 2] No such file or directory")

        with caplog.at_level(logging.INFO):
            scheduler.run_job()  # must not raise

        matches = [r for r in caplog.records if "Failed to spawn bot subprocess" in r.getMessage()]
        assert len(matches) == 1
        assert matches[0].levelno == logging.ERROR
        assert "No such file or directory" in matches[0].getMessage()

    @patch("kalshi_betting.scheduler.subprocess.run")
    def test_oserror_finalizes_state_with_no_exit_code(self, mock_run, tmp_path):
        mock_run.side_effect = OSError("fork failed")

        scheduler.run_job()

        state = json.loads((tmp_path / "scheduler_state.json").read_text())
        assert state["exit_code"] is None
        assert state["finished_at"] is not None


class TestMostRecentSlot:
    """_most_recent_slot: latest Monday 09:00 LOCAL time <= now."""

    def test_monday_before_0900_uses_previous_week(self):
        # Monday 2026-08-31 is a real Monday.
        now = datetime(2026, 8, 31, 8, 59)
        slot = scheduler._most_recent_slot(now)
        assert slot == datetime(2026, 8, 24, 9, 0)

    def test_monday_after_0900_uses_today(self):
        now = datetime(2026, 8, 31, 9, 1)
        slot = scheduler._most_recent_slot(now)
        assert slot == datetime(2026, 8, 31, 9, 0)

    def test_monday_exactly_0900_uses_today(self):
        now = datetime(2026, 8, 31, 9, 0)
        slot = scheduler._most_recent_slot(now)
        assert slot == datetime(2026, 8, 31, 9, 0)

    def test_midweek_uses_that_weeks_monday(self):
        # Wednesday 2026-09-02.
        now = datetime(2026, 9, 2, 14, 30)
        slot = scheduler._most_recent_slot(now)
        assert slot == datetime(2026, 8, 31, 9, 0)

    def test_sunday_uses_previous_monday(self):
        # Sunday 2026-09-06.
        now = datetime(2026, 9, 6, 12, 0)
        slot = scheduler._most_recent_slot(now)
        assert slot == datetime(2026, 8, 31, 9, 0)


def _write_state(tmp_path, **fields):
    path = tmp_path / "scheduler_state.json"
    payload = {
        "schema": 1,
        "last_slot": None,
        "started_at": None,
        "finished_at": None,
        "exit_code": None,
    }
    payload.update(fields)
    path.write_text(json.dumps(payload))


class TestCatchUp:
    """BS-17: _maybe_catch_up() decides whether to run an immediate catch-up
    job on startup, extracted from main() so tests don't need to enter the
    infinite poll loop."""

    def test_missing_state_file_triggers_catch_up(self, tmp_path, caplog):
        now = datetime(2026, 9, 2, 10, 0)  # no scheduler_state.json written

        with patch("kalshi_betting.scheduler.run_job") as mock_run_job, \
             caplog.at_level(logging.WARNING):
            scheduler._maybe_catch_up(now=now)

        mock_run_job.assert_called_once()
        assert any("no recorded run" in r.getMessage() for r in caplog.records)

    def test_stale_slot_triggers_catch_up(self, tmp_path):
        now = datetime(2026, 9, 2, 10, 0)  # midweek -> this week's Monday slot
        current_slot = scheduler._most_recent_slot(now)
        stale_slot = current_slot - timedelta(days=7)
        _write_state(tmp_path, last_slot=stale_slot.isoformat())

        with patch("kalshi_betting.scheduler.run_job") as mock_run_job:
            scheduler._maybe_catch_up(now=now)

        mock_run_job.assert_called_once()

    def test_current_slot_recorded_skips_catch_up(self, tmp_path):
        now = datetime(2026, 9, 2, 10, 0)
        current_slot = scheduler._most_recent_slot(now)
        _write_state(tmp_path, last_slot=current_slot.isoformat())

        with patch("kalshi_betting.scheduler.run_job") as mock_run_job:
            scheduler._maybe_catch_up(now=now)

        mock_run_job.assert_not_called()

    def test_corrupt_state_file_warns_and_catches_up(self, tmp_path, caplog):
        now = datetime(2026, 9, 2, 10, 0)
        (tmp_path / "scheduler_state.json").write_text("{not valid json")

        with patch("kalshi_betting.scheduler.run_job") as mock_run_job, \
             caplog.at_level(logging.WARNING):
            scheduler._maybe_catch_up(now=now)

        mock_run_job.assert_called_once()
        assert any("Corrupt scheduler state file" in r.getMessage() for r in caplog.records)
        assert any("no recorded run" in r.getMessage() for r in caplog.records)

    def test_state_missing_last_slot_field_warns_and_catches_up(self, tmp_path, caplog):
        now = datetime(2026, 9, 2, 10, 0)
        (tmp_path / "scheduler_state.json").write_text(json.dumps({"schema": 1}))

        with patch("kalshi_betting.scheduler.run_job") as mock_run_job, \
             caplog.at_level(logging.WARNING):
            scheduler._maybe_catch_up(now=now)

        mock_run_job.assert_called_once()
        assert any("missing expected fields" in r.getMessage() for r in caplog.records)


class TestSlotClaimAndFinalize:
    """run_job() claims the slot before spawning and finalizes it at the end
    of every exit path."""

    @patch("kalshi_betting.scheduler.subprocess.run")
    def test_success_writes_state_with_exit_code(self, mock_run, tmp_path):
        mock_run.return_value = _completed(EXIT_OK)

        scheduler.run_job()

        state_path = tmp_path / "scheduler_state.json"
        assert state_path.exists()
        state = json.loads(state_path.read_text())
        assert state["schema"] == 1
        assert state["exit_code"] == EXIT_OK
        assert state["started_at"] is not None
        assert state["finished_at"] is not None
        # last_slot must be a real Monday 09:00, not a placeholder.
        parsed_slot = datetime.fromisoformat(state["last_slot"])
        assert parsed_slot.weekday() == 0
        assert (parsed_slot.hour, parsed_slot.minute) == (9, 0)

    @patch("kalshi_betting.scheduler.subprocess.run")
    def test_nonzero_exit_still_finalizes_state(self, mock_run, tmp_path):
        mock_run.return_value = _completed(1, stderr="boom")

        scheduler.run_job()

        state = json.loads((tmp_path / "scheduler_state.json").read_text())
        assert state["exit_code"] == 1
        assert state["finished_at"] is not None


class TestAtomicStateWrite:
    def test_no_tmp_residue_after_write(self, tmp_path):
        scheduler._save_state(
            last_slot=datetime(2026, 8, 31, 9, 0),
            started_at=datetime(2026, 8, 31, 9, 0),
            finished_at=datetime(2026, 8, 31, 9, 5),
            exit_code=0,
        )

        assert (tmp_path / "scheduler_state.json").exists()
        assert not (tmp_path / "scheduler_state.json.tmp").exists()

    def test_second_write_leaves_no_residue_and_overwrites_cleanly(self, tmp_path):
        for code in (None, 0):
            scheduler._save_state(
                last_slot=datetime(2026, 8, 31, 9, 0),
                started_at=datetime(2026, 8, 31, 9, 0),
                finished_at=None if code is None else datetime(2026, 8, 31, 9, 5),
                exit_code=code,
            )

        assert not (tmp_path / "scheduler_state.json.tmp").exists()
        state = json.loads((tmp_path / "scheduler_state.json").read_text())
        assert state["exit_code"] == 0


class TestRunJob:
    """Ported from main's test_scheduler.py: the argv/cwd/timeout contract of the
    subprocess run_job() spawns. Its timeout and nonzero-exit cases are subsumed
    by TestTimeoutExpiredHandling and TestRunJobExitCodeMapping above."""

    @patch("kalshi_betting.scheduler.subprocess.run")
    def test_run_job_uses_project_root_cwd_and_timeout(self, mock_run, tmp_path):
        captured = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            return _completed(EXIT_OK, stdout="ok")

        mock_run.side_effect = fake_run
        scheduler.run_job()

        assert captured["argv"] == [
            sys.executable, "-m", "kalshi_betting.main", "--mode", "prod",
        ]
        # PROJECT_ROOT is redirected to tmp_path by the autouse fixture, so this
        # asserts run_job reads it live rather than freezing a module constant
        assert captured["kwargs"]["cwd"] == str(tmp_path)
        assert captured["kwargs"]["timeout"] == SCHEDULER_JOB_TIMEOUT_SECONDS

